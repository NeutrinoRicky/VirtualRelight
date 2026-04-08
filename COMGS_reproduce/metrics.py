#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import re
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "test", "val")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _composite_rgba(image: Image.Image, background: str) -> Image.Image:
    rgba = image.convert("RGBA")
    if background == "white":
        bg_color = (255, 255, 255, 255)
    elif background == "black":
        bg_color = (0, 0, 0, 255)
    else:
        raise ValueError(f"Unsupported background mode: {background}")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, bg_color), rgba).convert("RGB")


def _load_image_tensor(image_path: Path, background: str = "none") -> torch.Tensor:
    image = Image.open(image_path)
    if background in {"white", "black"} and "A" in image.getbands():
        image = _composite_rgba(image, background)
    else:
        image = image.convert("RGB")
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(DEVICE)


def _default_gt_background(layout_kind: str, requested_background: str) -> str:
    if requested_background != "auto":
        return requested_background
    if layout_kind == "stage2_split":
        return "white"
    return "none"


def _compute_metrics(image_pairs, gt_background: str):
    ssims = []
    psnrs = []
    lpipss = []
    image_names = []

    for render_path, gt_path, image_name in tqdm(image_pairs, desc="Metric evaluation progress"):
        render = _load_image_tensor(render_path)
        gt = _load_image_tensor(gt_path, background=gt_background)

        ssims.append(float(ssim(render, gt).item()))
        psnrs.append(float(psnr(render, gt).item()))
        lpipss.append(float(lpips(render, gt, net_type="vgg").item()))
        image_names.append(image_name)

    ssim_values = torch.tensor(ssims)
    psnr_values = torch.tensor(psnrs)
    lpips_values = torch.tensor(lpipss)

    print(f"  SSIM : {ssim_values.mean():>12.7f}")
    print(f"  PSNR : {psnr_values.mean():>12.7f}")
    print(f"  LPIPS: {lpips_values.mean():>12.7f}")
    print("")

    summary = {
        "SSIM": ssim_values.mean().item(),
        "PSNR": psnr_values.mean().item(),
        "LPIPS": lpips_values.mean().item(),
    }
    per_view = {
        "SSIM": {name: value for name, value in zip(image_names, ssim_values.tolist())},
        "PSNR": {name: value for name, value in zip(image_names, psnr_values.tolist())},
        "LPIPS": {name: value for name, value in zip(image_names, lpips_values.tolist())},
    }
    return summary, per_view


def _collect_legacy_targets(scene_dir: Path):
    targets = []
    for split_name in SPLIT_NAMES:
        split_dir = scene_dir / split_name
        if not split_dir.is_dir():
            continue

        for method_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            if not gt_dir.is_dir() or not renders_dir.is_dir():
                continue

            pairs = []
            missing_gt = []
            for render_path in sorted(path for path in renders_dir.iterdir() if _is_image_file(path)):
                gt_path = gt_dir / render_path.name
                if gt_path.is_file():
                    pairs.append((render_path, gt_path, render_path.name))
                else:
                    missing_gt.append(render_path.name)

            if missing_gt:
                print(f"  Warning: {len(missing_gt)} render(s) missing GT in {method_dir.name}")
            if pairs:
                targets.append({"split": split_name, "method": method_dir.name, "pairs": pairs})
    return targets


def _find_cfg_root(scene_dir: Path):
    candidates = [scene_dir, scene_dir.parent, scene_dir.parent.parent]
    for candidate in candidates:
        if candidate is not None and (candidate / "cfg_args").is_file():
            return candidate
    return None


def _infer_source_path(scene_dir: Path):
    cfg_root = _find_cfg_root(scene_dir)
    if cfg_root is None:
        return None

    cfg_text = (cfg_root / "cfg_args").read_text()
    match = re.search(r"source_path=(['\"])(.+?)\1", cfg_text)
    if match is None:
        return None

    source_path = Path(match.group(2))
    if source_path.is_absolute():
        return source_path
    return (cfg_root / source_path).resolve()


def _looks_like_stage2_method_dir(method_dir: Path, render_suffix: str) -> bool:
    if not method_dir.is_dir():
        return False
    return any(
        child.is_file() and child.name.endswith(render_suffix)
        for child in method_dir.iterdir()
    )


def _resolve_stage2_split_dirs(scene_dir: Path, render_suffix: str):
    direct_method_dirs = [
        path for path in sorted(scene_dir.iterdir())
        if _looks_like_stage2_method_dir(path, render_suffix)
    ]
    if direct_method_dirs:
        return [(scene_dir.name, direct_method_dirs)]

    split_dirs = []
    for split_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
        method_dirs = [
            path for path in sorted(split_dir.iterdir())
            if _looks_like_stage2_method_dir(path, render_suffix)
        ]
        if method_dirs and split_dir.name in SPLIT_NAMES:
            split_dirs.append((split_dir.name, method_dirs))
    return split_dirs


def _load_split_gt_lookup(source_path: Path, split_name: str):
    transforms_path = source_path / f"transforms_{split_name}.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing transforms file: {transforms_path}")

    with open(transforms_path, "r") as f:
        contents = json.load(f)

    gt_lookup = {}
    for idx, frame in enumerate(contents["frames"]):
        file_path = Path(frame["file_path"])
        if file_path.suffix:
            gt_rel_path = file_path
        else:
            gt_rel_path = Path(f"{frame['file_path']}.png")
        gt_path = (source_path / gt_rel_path).resolve()
        gt_lookup[idx] = gt_path
    return gt_lookup


def _parse_render_uid(render_name: str, render_suffix: str):
    if not render_name.endswith(render_suffix):
        return None
    prefix = render_name[: -len(render_suffix)]
    match = re.match(r"^(\d+)(?:_|$)", prefix)
    if match is None:
        return None
    return int(match.group(1))


def _collect_stage2_pairs(method_dir: Path, gt_lookup, render_suffix: str):
    pairs = []
    skipped = []
    missing_gt = []

    render_paths = [
        path for path in sorted(method_dir.iterdir())
        if path.is_file() and path.name.endswith(render_suffix)
    ]

    for render_path in render_paths:
        uid = _parse_render_uid(render_path.name, render_suffix)
        if uid is None:
            skipped.append(render_path.name)
            continue

        gt_path = gt_lookup.get(uid)
        if gt_path is None or not gt_path.is_file():
            missing_gt.append(render_path.name)
            continue

        pairs.append((render_path, gt_path, render_path.name))

    return pairs, skipped, missing_gt


def _collect_stage2_targets(scene_dir: Path, render_suffix: str, source_path=None):
    split_dirs = _resolve_stage2_split_dirs(scene_dir, render_suffix)
    if not split_dirs:
        return []

    resolved_source_path = Path(source_path) if source_path else _infer_source_path(scene_dir)
    if resolved_source_path is None:
        raise ValueError(
            "Could not infer source_path from cfg_args. Please pass --source_path explicitly."
        )

    targets = []
    for split_name, method_dirs in split_dirs:
        gt_lookup = _load_split_gt_lookup(resolved_source_path, split_name)
        for method_dir in method_dirs:
            pairs, skipped, missing_gt = _collect_stage2_pairs(method_dir, gt_lookup, render_suffix)
            if skipped:
                print(f"  Warning: skipped {len(skipped)} file(s) without numeric view id in {method_dir.name}")
            if missing_gt:
                print(f"  Warning: {len(missing_gt)} render(s) missing GT in {method_dir.name}")
            if pairs:
                targets.append({"split": split_name, "method": method_dir.name, "pairs": pairs})
    return targets


def _write_results(scene_dir: Path, full_results, per_view_results):
    with open(scene_dir / "results.json", "w") as fp:
        json.dump(full_results, fp, indent=True)
    with open(scene_dir / "per_view.json", "w") as fp:
        json.dump(per_view_results, fp, indent=True)


def evaluate(model_paths, render_suffix: str, source_path=None, gt_background: str = "auto"):
    for scene_dir_str in model_paths:
        scene_dir = Path(scene_dir_str)
        try:
            print("Scene:", scene_dir)

            legacy_targets = _collect_legacy_targets(scene_dir)
            if legacy_targets:
                full_results = {}
                per_view_results = {}
                effective_background = _default_gt_background("legacy", gt_background)
                legacy_splits = {target["split"] for target in legacy_targets}
                use_split_layout = legacy_splits != {"test"}

                for target in legacy_targets:
                    split_name = target["split"]
                    method = target["method"]
                    if use_split_layout:
                        print("Split:", split_name)
                    print("Method:", method)
                    summary, per_view = _compute_metrics(target["pairs"], gt_background=effective_background)
                    if use_split_layout:
                        full_results.setdefault(split_name, {})[method] = summary
                        per_view_results.setdefault(split_name, {})[method] = per_view
                    else:
                        full_results[method] = summary
                        per_view_results[method] = per_view

                _write_results(scene_dir, full_results, per_view_results)
                continue

            stage2_targets = _collect_stage2_targets(
                scene_dir,
                render_suffix=render_suffix,
                source_path=source_path,
            )
            if not stage2_targets:
                raise RuntimeError(
                    "No supported evaluation layout found. Expected either "
                    "test/<method>/{gt,renders} or <split>/<method>/*{render_suffix}."
                )

            full_results = {}
            per_view_results = {}
            effective_background = _default_gt_background("stage2_split", gt_background)

            for target in stage2_targets:
                split_name = target["split"]
                method = target["method"]
                print("Split:", split_name)
                print("Method:", method)
                summary, per_view = _compute_metrics(target["pairs"], gt_background=effective_background)
                full_results.setdefault(split_name, {})[method] = summary
                per_view_results.setdefault(split_name, {})[method] = per_view

            _write_results(scene_dir, full_results, per_view_results)
        except Exception as exc:
            print(f"Unable to compute metrics for model {scene_dir}: {exc}")


if __name__ == "__main__":
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)
    print(f"Using device: {DEVICE}")

    parser = ArgumentParser(description="Metric evaluation")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument(
        "--render_suffix",
        type=str,
        default="_rgb.png",
        help="Used by split-based render directories, for example '_rgb.png' or '_rgb_raw.png'.",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        help="Optional source dataset path. If omitted, metrics.py tries to infer it from cfg_args.",
    )
    parser.add_argument(
        "--gt_background",
        type=str,
        default="auto",
        choices=["auto", "none", "white", "black"],
        help="How to composite GT images with alpha before comparison.",
    )
    args = parser.parse_args()

    evaluate(
        args.model_paths,
        render_suffix=args.render_suffix,
        source_path=args.source_path,
        gt_background=args.gt_background,
    )
