from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


Array = np.ndarray


def rgb_to_srgb(x: Array) -> Array:
    return np.where(
        x > 0.0031308,
        np.maximum(x, 0.0031308) ** (1.0 / 2.4) * 1.055 - 0.055,
        12.92 * x,
    ).clip(0.0, 1.0)


def _load_image_uncached(path: Path, force_rgb: bool = True) -> Array:
    arr = np.asarray(Image.open(path)).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if force_rgb and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


@lru_cache(maxsize=4096)
def _load_image_cached(path: str, force_rgb: bool = True) -> Array:
    return _load_image_uncached(Path(path), force_rgb=force_rgb)


def load_image(path: Path, force_rgb: bool = True) -> Array:
    return _load_image_cached(str(path), force_rgb)


def find_ours_dir(path: Path) -> Path:
    if path.name.startswith("ours_"):
        return path

    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("ours_")]
    if not candidates:
        return path

    def iteration_key(p: Path) -> int:
        try:
            return int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=iteration_key)


def list_png_names(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("*.png"))


def safe_alpha(root: Path, subdir: str, name: str) -> Optional[Array]:
    path = root / subdir / name
    if not path.exists():
        return None
    return load_image(path, force_rgb=False)[..., :1]


def unpremultiply(value: Array, alpha: Optional[Array], eps: float = 1e-6) -> Array:
    if alpha is None:
        return value
    return value / np.clip(alpha, eps, 1.0)


def premul_linear_to_srgb(value: Array, alpha: Optional[Array]) -> Array:
    if alpha is None:
        return rgb_to_srgb(value)
    return rgb_to_srgb(unpremultiply(value, alpha)) * alpha


def broadcast_mask(mask: Array, target: Array) -> Array:
    if mask.ndim == 2:
        mask = mask[..., None]
    return np.broadcast_to(mask.astype(bool), target.shape)


def metric_values(left: Array, right: Array, mask: Optional[Array]) -> Dict[str, Optional[float]]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")

    diff = left - right
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    if mask is not None:
        metric_mask = broadcast_mask(mask, diff)
        if not metric_mask.any():
            return {
                "pixels": 0,
                "mae": None,
                "rmse": None,
                "psnr": None,
                "max_abs": None,
                "left_mean": None,
                "right_mean": None,
            }
        abs_vals = abs_diff[metric_mask]
        sq_vals = sq_diff[metric_mask]
        left_vals = left[metric_mask]
        right_vals = right[metric_mask]
        pixels = int(metric_mask.sum())
    else:
        abs_vals = abs_diff.reshape(-1)
        sq_vals = sq_diff.reshape(-1)
        left_vals = left.reshape(-1)
        right_vals = right.reshape(-1)
        pixels = int(abs_vals.size)

    mse = float(sq_vals.mean())
    psnr = 120.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
    return {
        "pixels": pixels,
        "mae": float(abs_vals.mean()),
        "rmse": float(math.sqrt(mse)),
        "psnr": float(psnr),
        "max_abs": float(abs_vals.max()),
        "left_mean": float(left_vals.mean()),
        "right_mean": float(right_vals.mean()),
    }


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "p90": None, "max": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
    }


@dataclass(frozen=True)
class Comparison:
    name: str
    sop_dir: str
    irgs_dir: str
    note: str
    transform_sop: Optional[str] = None
    transform_irgs: Optional[str] = None
    unpremultiply_sop: bool = False
    unpremultiply_irgs: bool = False
    foreground_only: bool = False


COMPARISONS: Tuple[Comparison, ...] = (
    Comparison("render", "render", "render", "final rendered RGB"),
    Comparison("gt", "gt", "gt", "saved ground truth images"),
    Comparison("diffuse", "diffuse", "diffuse", "diffuse component"),
    Comparison("specular", "specular", "specular", "specular component"),
    Comparison("roughness_premul", "roughness", "roughness", "saved roughness, alpha-premultiplied"),
    Comparison("metallic_premul", "metallic", "metallic", "saved metallic, alpha-premultiplied"),
    Comparison("normal", "normal", "rend_normal", "saved normals in [0, 1] visualization space"),
    Comparison("alpha", "weight", "rend_alpha", "SOP weight vs IRGS rend_alpha"),
    Comparison("albedo_linear_premul", "albedo", "base_color_linear", "SOP albedo corresponds to IRGS base_color_linear"),
    Comparison(
        "albedo_srgb_premul",
        "albedo",
        "base_color",
        "SOP albedo converted from linear to sRGB before comparing to IRGS base_color",
        transform_sop="linear_premul_to_srgb_premul",
    ),
    Comparison(
        "albedo_linear_unpremul_fg",
        "albedo",
        "base_color_linear",
        "foreground true linear albedo after dividing by alpha/weight",
        unpremultiply_sop=True,
        unpremultiply_irgs=True,
        foreground_only=True,
    ),
    Comparison(
        "roughness_unpremul_fg",
        "roughness",
        "roughness",
        "foreground true roughness after dividing by alpha/weight",
        unpremultiply_sop=True,
        unpremultiply_irgs=True,
        foreground_only=True,
    ),
    Comparison(
        "metallic_unpremul_fg",
        "metallic",
        "metallic",
        "foreground true metallic after dividing by alpha/weight",
        unpremultiply_sop=True,
        unpremultiply_irgs=True,
        foreground_only=True,
    ),
)


def apply_transform(value: Array, transform: Optional[str], alpha: Optional[Array]) -> Array:
    if transform is None:
        return value
    if transform == "linear_premul_to_srgb_premul":
        return premul_linear_to_srgb(value, alpha)
    raise ValueError(f"unknown transform: {transform}")


def common_foreground_mask(
    sop_root: Path,
    irgs_root: Path,
    name: str,
    alpha_threshold: float,
) -> Optional[Array]:
    sop_alpha = safe_alpha(sop_root, "weight", name)
    irgs_alpha = safe_alpha(irgs_root, "rend_alpha", name)
    if sop_alpha is None or irgs_alpha is None:
        return None
    return (sop_alpha[..., 0] > alpha_threshold) & (irgs_alpha[..., 0] > alpha_threshold)


def compare_one(
    cmp: Comparison,
    sop_root: Path,
    irgs_root: Path,
    name: str,
    alpha_threshold: float,
) -> Dict[str, object]:
    sop_path = sop_root / cmp.sop_dir / name
    irgs_path = irgs_root / cmp.irgs_dir / name
    left = load_image(sop_path)
    right = load_image(irgs_path)

    sop_alpha = safe_alpha(sop_root, "weight", name)
    irgs_alpha = safe_alpha(irgs_root, "rend_alpha", name)

    left = apply_transform(left, cmp.transform_sop, sop_alpha)
    right = apply_transform(right, cmp.transform_irgs, irgs_alpha)

    if cmp.unpremultiply_sop:
        left = unpremultiply(left, sop_alpha)
    if cmp.unpremultiply_irgs:
        right = unpremultiply(right, irgs_alpha)

    fg_mask = common_foreground_mask(sop_root, irgs_root, name, alpha_threshold)
    all_metrics = metric_values(left, right, None) if not cmp.foreground_only else {}
    fg_metrics = metric_values(left, right, fg_mask) if fg_mask is not None else {}

    row: Dict[str, object] = {
        "comparison": cmp.name,
        "image": name,
        "note": cmp.note,
    }
    for prefix, metrics in (("all", all_metrics), ("fg", fg_metrics)):
        for key in ("pixels", "mae", "rmse", "psnr", "max_abs", "left_mean", "right_mean"):
            row[f"{prefix}_{key}"] = metrics.get(key) if metrics else None
    return row


def run_analysis(
    irgs_test: Path,
    sop_test: Path,
    out_dir: Path,
    alpha_threshold: float,
    limit: int,
) -> Dict[str, object]:
    irgs_root = find_ours_dir(irgs_test)
    sop_root = find_ours_dir(sop_test)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    skipped: Dict[str, str] = {}

    for cmp in COMPARISONS:
        sop_names = list_png_names(sop_root / cmp.sop_dir)
        irgs_names = list_png_names(irgs_root / cmp.irgs_dir)
        common = sorted(set(sop_names) & set(irgs_names))
        if limit > 0:
            common = common[:limit]
        if not common:
            skipped[cmp.name] = f"missing or empty dirs: SOP/{cmp.sop_dir}, IRGS/{cmp.irgs_dir}"
            continue

        for name in common:
            rows.append(compare_one(cmp, sop_root, irgs_root, name, alpha_threshold))

    csv_path = out_dir / "per_image_metrics.csv"
    fieldnames = [
        "comparison",
        "image",
        "note",
        "all_pixels",
        "all_mae",
        "all_rmse",
        "all_psnr",
        "all_max_abs",
        "all_left_mean",
        "all_right_mean",
        "fg_pixels",
        "fg_mae",
        "fg_rmse",
        "fg_psnr",
        "fg_max_abs",
        "fg_left_mean",
        "fg_right_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, object] = {
        "irgs_test": str(irgs_test),
        "sop_test": str(sop_test),
        "irgs_ours": str(irgs_root),
        "sop_ours": str(sop_root),
        "alpha_threshold": alpha_threshold,
        "num_rows": len(rows),
        "skipped": skipped,
        "comparisons": {},
    }
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["comparison"]), []).append(row)

    for name, group in grouped.items():
        summary["comparisons"][name] = {
            "num_images": len(group),
            "all_mae": summarize(row.get("all_mae") for row in group),
            "all_rmse": summarize(row.get("all_rmse") for row in group),
            "all_psnr": summarize(row.get("all_psnr") for row in group),
            "fg_mae": summarize(row.get("fg_mae") for row in group),
            "fg_rmse": summarize(row.get("fg_rmse") for row in group),
            "fg_psnr": summarize(row.get("fg_psnr") for row in group),
            "fg_left_mean": summarize(row.get("fg_left_mean") for row in group),
            "fg_right_mean": summarize(row.get("fg_right_mean") for row in group),
        }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
    }


def print_summary(result: Dict[str, object]) -> None:
    summary = result["summary"]
    print(f"IRGS: {summary['irgs_ours']}")
    print(f"SOP : {summary['sop_ours']}")
    print(f"CSV : {result['csv_path']}")
    print(f"JSON: {result['summary_path']}")
    print("")

    comparisons = summary["comparisons"]
    for name in sorted(comparisons):
        item = comparisons[name]
        all_mae = item["all_mae"]["mean"]
        fg_mae = item["fg_mae"]["mean"]
        fg_psnr = item["fg_psnr"]["mean"]
        all_mae_s = "n/a" if all_mae is None else f"{all_mae:.6f}"
        fg_mae_s = "n/a" if fg_mae is None else f"{fg_mae:.6f}"
        fg_psnr_s = "n/a" if fg_psnr is None else f"{fg_psnr:.3f}"
        print(f"{name:28s} images={item['num_images']:4d} all_mae={all_mae_s:>10s} fg_mae={fg_mae_s:>10s} fg_psnr={fg_psnr_s:>9s}")

    skipped = summary.get("skipped") or {}
    if skipped:
        print("\nSkipped:")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SOP test_sop outputs against IRGS test outputs."
    )
    parser.add_argument("irgs_test", type=Path, help="IRGS test folder containing an ours_* subfolder, e.g. .../irgs_octa.../test")
    parser.add_argument("sop_test", type=Path, help="SOP test_sop folder containing an ours_* subfolder, e.g. .../irgs_sop.../test_sop")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output folder for summary.json and per_image_metrics.csv")
    parser.add_argument("--alpha_threshold", type=float, default=0.5, help="Foreground mask threshold using SOP weight and IRGS rend_alpha")
    parser.add_argument("--limit", type=int, default=-1, help="Only compare the first N images per comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    irgs_test = args.irgs_test.resolve()
    sop_test = args.sop_test.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = sop_test / "compare_to_irgs"
    result = run_analysis(
        irgs_test=irgs_test,
        sop_test=sop_test,
        out_dir=out_dir.resolve(),
        alpha_threshold=float(args.alpha_threshold),
        limit=int(args.limit),
    )
    print_summary(result)


if __name__ == "__main__":
    main()
