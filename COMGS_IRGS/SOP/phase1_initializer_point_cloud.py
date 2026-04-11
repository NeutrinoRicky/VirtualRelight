from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_sop_gbuffer
from scene import GaussianModel, Scene
from utils.general_utils import safe_state

try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class ViewSurface:
    index: int
    camera: object
    depth: torch.Tensor
    normal: torch.Tensor
    valid: torch.Tensor
    points: torch.Tensor
    normals: torch.Tensor
    num_valid_pixels: int


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _bbox_extent(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    side_lengths = maxs - mins
    return side_lengths, float(np.linalg.norm(side_lengths)), float(side_lengths.max())


def _compute_object_extent(points: np.ndarray, mode: str) -> float:
    side_lengths, diagonal, max_side = _bbox_extent(points)
    if mode == "bbox_diagonal":
        return diagonal
    if mode == "max_side":
        return max_side
    raise ValueError(f"Unknown extent mode: {mode}")


def _summarize_array(values: np.ndarray) -> Dict[str, float]:
    if values.shape[0] == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _save_point_cloud(
    path: Path,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
) -> None:
    if points.shape[0] == 0:
        raise ValueError(f"Cannot write empty point cloud to {path}")

    if normals is None:
        normals = np.zeros_like(points, dtype=np.float32)
    if colors is None:
        colors = np.full_like(points, 0.75, dtype=np.float32)

    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(points.shape[0], dtype=dtype)
    attributes = np.concatenate([points.astype(np.float32), normals.astype(np.float32), colors_u8], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def _save_point_cloud_limited(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    num_points = int(points.shape[0])
    if num_points == 0:
        return {"path": str(path), "num_points": 0, "written": False}

    if max_points > 0 and num_points > max_points:
        indices = np.sort(rng.choice(num_points, size=max_points, replace=False))
        write_points = points[indices]
        write_normals = normals[indices]
        write_colors = colors[indices]
    else:
        indices = None
        write_points = points
        write_normals = normals
        write_colors = colors

    _save_point_cloud(path, write_points, write_normals, write_colors)
    return {
        "path": str(path),
        "num_points": num_points,
        "num_written": int(write_points.shape[0]),
        "subsampled": indices is not None,
        "written": True,
    }


def _save_lineset(path: Path, starts: np.ndarray, ends: np.ndarray, color: Sequence[float]) -> bool:
    if o3d is None or starts.shape[0] == 0:
        return False

    line_points = np.concatenate([starts, ends], axis=0)
    line_indices = np.stack(
        [
            np.arange(starts.shape[0], dtype=np.int32),
            np.arange(starts.shape[0], dtype=np.int32) + starts.shape[0],
        ],
        axis=1,
    )
    colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (starts.shape[0], 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    try:
        return bool(o3d.io.write_line_set(str(path), line_set))
    except Exception:
        return False


def _voxel_downsample_numpy(points: np.ndarray, normals: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points, normals

    mins = points.min(axis=0, keepdims=True)
    voxel_coords = np.floor((points - mins) / voxel_size).astype(np.int64)
    unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    point_sums = np.zeros((unique_coords.shape[0], 3), dtype=np.float64)
    normal_sums = np.zeros((unique_coords.shape[0], 3), dtype=np.float64)
    counts = np.zeros((unique_coords.shape[0], 1), dtype=np.float64)
    np.add.at(point_sums, inverse, points.astype(np.float64))
    np.add.at(normal_sums, inverse, normals.astype(np.float64))
    np.add.at(counts, inverse, 1.0)
    down_points = point_sums / np.clip(counts, 1.0, None)
    down_normals = _normalize_np(normal_sums.astype(np.float32))
    return down_points.astype(np.float32), down_normals.astype(np.float32)


def _radius_denoise_open3d(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    stats = {
        "enabled": False,
        "mode": "radius",
        "radius": float(radius),
        "min_neighbors": int(min_neighbors),
        "input_points": int(points.shape[0]),
        "output_points": int(points.shape[0]),
    }
    if radius <= 0.0 or min_neighbors <= 0:
        stats["reason"] = "non_positive_radius_or_neighbors"
        return points, normals, stats
    if o3d is None:
        stats["reason"] = "open3d_unavailable"
        return points, normals, stats

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    _, keep_indices = pcd.remove_radius_outlier(nb_points=int(min_neighbors), radius=float(radius))
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.shape[0] == 0:
        raise RuntimeError(
            "Radius denoising removed all points. Reduce --radius_denoise_radius_factor "
            "or --radius_denoise_min_neighbors."
        )

    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "output_points": int(keep_indices.shape[0]),
            "removed_points": int(points.shape[0] - keep_indices.shape[0]),
        }
    )
    return points[keep_indices].astype(np.float32), normals[keep_indices].astype(np.float32), stats


def _binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = mask.bool()
    if radius <= 0:
        return mask

    inv_mask = (~mask).float().unsqueeze(0).unsqueeze(0)
    kernel = radius * 2 + 1
    eroded = F.max_pool2d(inv_mask, kernel_size=kernel, stride=1, padding=radius)[0, 0] < 0.5
    return eroded


def _build_object_valid_mask(
    depth: torch.Tensor,
    weight: torch.Tensor,
    gt_alpha_mask: Optional[torch.Tensor],
    weight_thresh: float,
    mask_thresh: float,
    object_filter_mode: str,
    mask_erosion_radius: int,
) -> torch.Tensor:
    depth_valid = torch.isfinite(depth) & (depth > 0.0)
    weight_valid = torch.isfinite(weight) & (weight > weight_thresh)

    mask_available = gt_alpha_mask is not None
    if mask_available:
        mask_valid = gt_alpha_mask[0].to(device=depth.device) > mask_thresh
        core_mask = _binary_erode(mask_valid, mask_erosion_radius)
    else:
        mask_valid = torch.zeros_like(depth_valid, dtype=torch.bool)
        core_mask = torch.ones_like(depth_valid, dtype=torch.bool)

    if object_filter_mode == "weight_only":
        object_valid = weight_valid
    elif object_filter_mode == "mask_only":
        object_valid = mask_valid if mask_available else weight_valid
    elif object_filter_mode == "weight_or_mask":
        object_valid = (weight_valid | mask_valid) if mask_available else weight_valid
    elif object_filter_mode == "weight_and_mask":
        object_valid = (weight_valid & mask_valid) if mask_available else weight_valid
    else:
        raise ValueError(f"Unknown object_filter_mode: {object_filter_mode}")
    return depth_valid & object_valid & core_mask


def _choose_cameras(scene: Scene, camera_set: str) -> List:
    if camera_set == "train":
        return list(scene.getTrainCameras())
    if camera_set == "test":
        return list(scene.getTestCameras())
    if camera_set == "all":
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    raise ValueError(f"Unknown camera_set: {camera_set}")


def _view_object_mask(viewpoint_cam) -> Optional[torch.Tensor]:
    mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
    if mask is None:
        mask = getattr(viewpoint_cam, "mask", None)
    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask


def _farthest_point_sampling(points: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    if points.shape[0] <= k:
        return np.arange(points.shape[0], dtype=np.int64)

    pts = torch.from_numpy(points.astype(np.float32)).to(device)
    center = pts.mean(dim=0, keepdim=True)
    distances = torch.sum((pts - center) ** 2, dim=1)
    farthest = int(torch.argmax(distances).item())

    selected = torch.empty((k,), dtype=torch.long, device=device)
    min_dist = torch.full((pts.shape[0],), float("inf"), device=device)
    for i in range(k):
        selected[i] = farthest
        centroid = pts[farthest].unsqueeze(0)
        dist = torch.sum((pts - centroid) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = int(torch.argmax(min_dist).item())
    return selected.cpu().numpy().astype(np.int64)


def _load_checkpoint_into_gaussians(gaussians: GaussianModel, opt, checkpoint_path: str) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=torch.device("cuda"), weights_only=False)
    if isinstance(payload, dict) and "gaussians" in payload:
        gaussians.restore(payload["gaussians"], opt)
        return {
            "source": "checkpoint_dict",
            "path": checkpoint_path,
            "iteration": int(payload.get("iteration", 0)),
            "format": str(payload.get("format", "checkpoint_dict")),
        }

    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    model_params = payload[0]
    iteration = int(payload[1]) if len(payload) >= 2 else 0
    if isinstance(model_params, (tuple, list)) and len(model_params) in {16}:
        gaussians.restore(model_params, opt)
        label = "irgs_checkpoint"
    elif isinstance(model_params, (tuple, list)) and len(model_params) in {19, 26}:
        gaussians.restore_from_refgs(model_params, opt)
        label = "irgs_refgs"
    else:
        raise RuntimeError(f"Unsupported checkpoint model args len={len(model_params) if isinstance(model_params, (tuple, list)) else 'n/a'}")

    return {
        "source": "checkpoint_tuple",
        "path": checkpoint_path,
        "iteration": iteration,
        "format": label,
    }


def _set_gaussian_material_defaults(gaussians: GaussianModel, opt) -> None:
    gaussians.init_base_color_value = getattr(opt, "init_base_color_value", gaussians.init_base_color_value)
    gaussians.init_metallic_value = getattr(opt, "init_metallic_value", gaussians.init_metallic_value)
    gaussians.init_roughness_value = getattr(opt, "init_roughness_value", gaussians.init_roughness_value)


def _orient_normals_by_center(points: np.ndarray, normals: np.ndarray, center: np.ndarray) -> np.ndarray:
    outward_hint = points - center.reshape(1, 3)
    flip = np.sum(normals * outward_hint, axis=1) < 0.0
    oriented = normals.copy()
    oriented[flip] *= -1.0
    return _normalize_np(oriented.astype(np.float32))


@torch.no_grad()
def _render_view_surface(
    view_index: int,
    viewpoint_cam,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    args,
) -> ViewSurface:
    render_pkg = render_sop_gbuffer(viewpoint_cam, gaussians, pipe, background)
    depth = torch.nan_to_num(render_pkg["depth_unbiased"], nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.nan_to_num(torch.clamp(render_pkg["weight"], 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    normal_chw = torch.nan_to_num(F.normalize(render_pkg["normal"], dim=0, eps=1e-6), nan=0.0, posinf=0.0, neginf=0.0)

    valid = _build_object_valid_mask(
        depth=depth,
        weight=weight,
        gt_alpha_mask=_view_object_mask(viewpoint_cam),
        weight_thresh=args.weight_thresh,
        mask_thresh=args.mask_thresh,
        object_filter_mode=args.object_filter_mode,
        mask_erosion_radius=args.mask_erosion_radius,
    )
    valid = valid & (torch.linalg.norm(normal_chw, dim=0) > float(args.normal_min_norm))

    points_world = depth[..., None] * viewpoint_cam.rays_d_hw_unnormalized + viewpoint_cam.camera_center
    normals_world = F.normalize(normal_chw.permute(1, 2, 0), dim=-1, eps=1e-6)

    to_camera = F.normalize(viewpoint_cam.camera_center.view(1, 1, 3) - points_world, dim=-1, eps=1e-6)
    flip = (normals_world * to_camera).sum(dim=-1, keepdim=True) < 0.0
    normals_world = torch.where(flip, -normals_world, normals_world)

    valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    num_valid_pixels = int(valid_idx.numel())
    if valid_idx.numel() > 0 and args.max_points_per_view > 0 and valid_idx.numel() > args.max_points_per_view:
        perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[: args.max_points_per_view]
        valid_idx = valid_idx[perm]

    if valid_idx.numel() == 0:
        points = torch.zeros((0, 3), dtype=torch.float32)
        normals = torch.zeros((0, 3), dtype=torch.float32)
    else:
        points = points_world.reshape(-1, 3)[valid_idx].detach().cpu().float()
        normals = normals_world.reshape(-1, 3)[valid_idx].detach().cpu().float()

    return ViewSurface(
        index=view_index,
        camera=viewpoint_cam,
        depth=depth.detach().cpu().float().unsqueeze(0),
        normal=normals_world.detach().cpu().float().permute(2, 0, 1),
        valid=valid.detach().cpu().bool().unsqueeze(0),
        points=points,
        normals=normals,
        num_valid_pixels=num_valid_pixels,
    )


def _build_target_view_indices(views: List[ViewSurface], args) -> Dict[int, List[int]]:
    centers = np.stack([_to_numpy(v.camera.camera_center).astype(np.float32) for v in views], axis=0)
    target_indices: Dict[int, List[int]] = {}
    stride = max(1, int(args.consistency_view_stride))
    max_views = int(args.consistency_max_views)

    for src_idx in range(len(views)):
        dists = np.linalg.norm(centers - centers[src_idx][None], axis=1)
        order = [int(i) for i in np.argsort(dists) if int(i) != src_idx]
        order = order[::stride]
        if max_views > 0:
            order = order[:max_views]
        target_indices[src_idx] = order
    return target_indices


def _project_points_to_view(points: torch.Tensor, viewpoint_cam, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    projected = points_h @ viewpoint_cam.full_proj_transform.to(device=points.device, dtype=points.dtype)
    w = projected[:, 3]
    valid_w = torch.isfinite(w) & (torch.abs(w) > eps) & (w > 0.0)
    safe_w = torch.where(valid_w, w, torch.ones_like(w))
    ndc_xy = projected[:, :2] / safe_w[:, None]
    in_image = (
        valid_w
        & torch.isfinite(ndc_xy).all(dim=-1)
        & (ndc_xy[:, 0] > -1.0)
        & (ndc_xy[:, 0] < 1.0)
        & (ndc_xy[:, 1] > -1.0)
        & (ndc_xy[:, 1] < 1.0)
    )
    return ndc_xy, w, in_image


def _check_points_against_target(
    points: torch.Tensor,
    normals: torch.Tensor,
    target_camera,
    target_depth: torch.Tensor,
    target_normal: torch.Tensor,
    target_valid: torch.Tensor,
    depth_abs_tol: float,
    depth_rel_tol: float,
    normal_cos_thresh: float,
    use_abs_normal_dot: bool,
) -> torch.Tensor:
    ndc_xy, target_point_depth, in_image = _project_points_to_view(points, target_camera)
    grid = ndc_xy.reshape(1, 1, -1, 2)

    sampled_valid = F.grid_sample(
        target_valid,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1) > 0.5
    sampled_depth = F.grid_sample(
        target_depth,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    sampled_normal = F.grid_sample(
        target_normal,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(3, -1).T
    sampled_normal = F.normalize(sampled_normal, dim=-1, eps=1e-6)

    depth_tol = float(depth_abs_tol) + float(depth_rel_tol) * torch.abs(target_point_depth).clamp_min(1e-6)
    depth_ok = torch.abs(sampled_depth - target_point_depth) <= depth_tol
    normal_dot = torch.sum(normals * sampled_normal, dim=-1).clamp(-1.0, 1.0)
    if use_abs_normal_dot:
        normal_dot = torch.abs(normal_dot)
    normal_ok = normal_dot >= float(normal_cos_thresh)
    return in_image & sampled_valid & depth_ok & normal_ok


@torch.no_grad()
def _cross_view_consistency_filter(
    views: List[ViewSurface],
    object_extent: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(views) < 2 and int(args.consistency_min_views) > 0:
        raise RuntimeError("Cross-view consistency needs at least two selected cameras.")

    device = torch.device("cuda")
    target_indices = _build_target_view_indices(views, args)
    depth_abs_tol = float(args.depth_tolerance) if args.depth_tolerance > 0.0 else float(args.depth_tolerance_factor) * object_extent
    normal_cos_thresh = math.cos(math.radians(float(args.normal_angle_thresh)))
    chunk_size = max(1, int(args.consistency_chunk_size))

    all_points: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_source_indices: List[np.ndarray] = []
    all_support_counts: List[np.ndarray] = []
    per_view_stats: List[Dict[str, object]] = []

    for src_pos, src in enumerate(views):
        num_candidates = int(src.points.shape[0])
        targets = target_indices[src_pos]
        effective_min_views = min(max(0, int(args.consistency_min_views)), len(targets))
        support_counts = np.zeros((num_candidates,), dtype=np.int16)

        if effective_min_views > 0 and num_candidates > 0:
            for target_pos in targets:
                target = views[target_pos]
                target_depth = target.depth.unsqueeze(0).to(device=device, dtype=torch.float32)
                target_normal = target.normal.unsqueeze(0).to(device=device, dtype=torch.float32)
                target_valid = target.valid.float().unsqueeze(0).to(device=device, dtype=torch.float32)

                target_success = np.zeros((num_candidates,), dtype=bool)
                for start in range(0, num_candidates, chunk_size):
                    end = min(start + chunk_size, num_candidates)
                    points_chunk = src.points[start:end].to(device=device, dtype=torch.float32)
                    normals_chunk = src.normals[start:end].to(device=device, dtype=torch.float32)
                    success = _check_points_against_target(
                        points=points_chunk,
                        normals=normals_chunk,
                        target_camera=target.camera,
                        target_depth=target_depth,
                        target_normal=target_normal,
                        target_valid=target_valid,
                        depth_abs_tol=depth_abs_tol,
                        depth_rel_tol=float(args.depth_relative_tolerance),
                        normal_cos_thresh=normal_cos_thresh,
                        use_abs_normal_dot=bool(args.use_abs_normal_dot),
                    )
                    target_success[start:end] = _to_numpy(success).astype(bool)

                support_counts += target_success.astype(np.int16)
                del target_depth, target_normal, target_valid

        keep = support_counts >= effective_min_views
        if effective_min_views == 0:
            keep[:] = True

        survived = int(np.sum(keep))
        if survived > 0:
            all_points.append(src.points.numpy()[keep].astype(np.float32))
            all_normals.append(src.normals.numpy()[keep].astype(np.float32))
            all_source_indices.append(np.full((survived,), src.index, dtype=np.int32))
            all_support_counts.append(support_counts[keep].astype(np.int16))

        per_view_stats.append(
            {
                "view_index": int(src.index),
                "valid_pixels_before_sampling": int(src.num_valid_pixels),
                "candidates_after_sampling": num_candidates,
                "target_views": [int(views[t].index) for t in targets],
                "effective_min_consistent_views": int(effective_min_views),
                "survived": survived,
                "support_count": _summarize_array(support_counts.astype(np.float32)),
            }
        )
        print(
            "[SOP-Phase1-PC] Consistency view "
            f"{src_pos + 1}/{len(views)}: kept {survived}/{num_candidates} "
            f"(min_views={effective_min_views}, targets={len(targets)})"
        )

    if not all_points:
        raise RuntimeError(
            "Cross-view consistency removed all candidates. Try lowering --consistency_min_views, "
            "--weight_thresh, or --normal_angle_thresh, or increasing --depth_tolerance_factor."
        )

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    normals = _normalize_np(np.concatenate(all_normals, axis=0).astype(np.float32))
    source_indices = np.concatenate(all_source_indices, axis=0).astype(np.int32)
    support_counts = np.concatenate(all_support_counts, axis=0).astype(np.int16)

    stats = {
        "depth_abs_tolerance": float(depth_abs_tol),
        "depth_relative_tolerance": float(args.depth_relative_tolerance),
        "normal_angle_thresh": float(args.normal_angle_thresh),
        "normal_cos_thresh": float(normal_cos_thresh),
        "use_abs_normal_dot": bool(args.use_abs_normal_dot),
        "min_views_requested": int(args.consistency_min_views),
        "max_target_views": int(args.consistency_max_views),
        "target_view_stride": int(args.consistency_view_stride),
        "num_candidates": int(sum(int(v.points.shape[0]) for v in views)),
        "num_survivors": int(points.shape[0]),
        "support_count": _summarize_array(support_counts.astype(np.float32)),
        "per_view": per_view_stats,
    }
    return points, normals, source_indices, support_counts, stats


def _clean_surface_cloud(
    points: np.ndarray,
    normals: np.ndarray,
    object_extent: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    voxel_size = float(args.fusion_voxel_size) if args.fusion_voxel_size > 0.0 else float(args.fusion_voxel_factor) * object_extent
    before_voxel = int(points.shape[0])
    points, normals = _voxel_downsample_numpy(points, normals, voxel_size)
    if points.shape[0] == 0:
        raise RuntimeError("Voxel downsampling removed all points.")

    denoise_stats: Dict[str, object] = {"enabled": False, "mode": args.denoise_mode}
    if args.denoise_mode == "radius":
        radius = (
            float(args.radius_denoise_radius)
            if args.radius_denoise_radius > 0.0
            else float(args.radius_denoise_radius_factor) * object_extent
        )
        points, normals, denoise_stats = _radius_denoise_open3d(
            points=points,
            normals=normals,
            radius=radius,
            min_neighbors=int(args.radius_denoise_min_neighbors),
        )
    elif args.denoise_mode != "none":
        raise ValueError(f"Unknown denoise_mode: {args.denoise_mode}")

    stats = {
        "input_points": before_voxel,
        "voxel_size": float(voxel_size),
        "after_voxel_points": int(points.shape[0] if not denoise_stats.get("enabled", False) else denoise_stats.get("input_points", points.shape[0])),
        "after_clean_points": int(points.shape[0]),
        "denoise": denoise_stats,
    }
    return points.astype(np.float32), _normalize_np(normals.astype(np.float32)), stats


def _prepare_scene_and_gaussians(args, dataset, pipe, opt) -> Tuple[Scene, GaussianModel, Dict[str, object]]:
    gaussians = GaussianModel(dataset.sh_degree)
    _set_gaussian_material_defaults(gaussians, opt)

    load_iteration = None if args.checkpoint else args.iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)

    if args.checkpoint:
        loaded_from = _load_checkpoint_into_gaussians(gaussians, opt, args.checkpoint)
    else:
        loaded_from = {
            "source": "scene_iteration",
            "iteration": int(scene.loaded_iter) if scene.loaded_iter is not None else 0,
        }

    if scene.light_rotate:
        transform = torch.tensor(
            [
                [0, -1, 0],
                [0, 0, 1],
                [-1, 0, 0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        gaussians.env_map.set_transform(transform)
    return scene, gaussians, loaded_from


@torch.no_grad()
def initialize_sop_phase1_point_cloud(args):
    if not torch.cuda.is_available():
        raise RuntimeError("SOP point-cloud phase1 initializer currently requires CUDA.")

    rng = np.random.default_rng(args.seed)
    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)
    _ensure_dir(Path(dataset.model_path))

    scene, gaussians, loaded_from = _prepare_scene_and_gaussians(args, dataset, pipe, opt)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    cameras = _choose_cameras(scene, args.camera_set)
    cameras = cameras[:: max(1, int(args.view_stride))]
    if args.max_views > 0:
        cameras = cameras[: int(args.max_views)]
    if len(cameras) == 0:
        raise RuntimeError("No cameras selected for SOP point-cloud phase1 initialization.")

    output_root = _ensure_dir(Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "SOP_phase1")

    print(f"[SOP-Phase1-PC] Rendering {len(cameras)} views for depth/normal/weight.")
    view_surfaces: List[ViewSurface] = []
    for view_pos, viewpoint_cam in enumerate(cameras):
        view_surface = _render_view_surface(
            view_index=view_pos,
            viewpoint_cam=viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            args=args,
        )
        view_surfaces.append(view_surface)
        print(
            "[SOP-Phase1-PC] View "
            f"{view_pos + 1}/{len(cameras)}: valid_pixels={view_surface.num_valid_pixels}, "
            f"candidates={view_surface.points.shape[0]}"
        )
        torch.cuda.empty_cache()

    if sum(int(v.points.shape[0]) for v in view_surfaces) == 0:
        raise RuntimeError("No high-confidence object pixels were backprojected. Lower --weight_thresh or check masks.")

    candidate_points = np.concatenate([v.points.numpy() for v in view_surfaces if v.points.shape[0] > 0], axis=0).astype(np.float32)
    candidate_normals = _normalize_np(
        np.concatenate([v.normals.numpy() for v in view_surfaces if v.normals.shape[0] > 0], axis=0).astype(np.float32)
    )
    object_extent = _compute_object_extent(candidate_points, args.extent_mode)
    object_center = candidate_points.mean(axis=0).astype(np.float32)
    if object_extent <= 0.0:
        raise RuntimeError("Rendered candidate point cloud has zero extent.")

    consistent_points, consistent_normals, source_indices, support_counts, consistency_stats = _cross_view_consistency_filter(
        views=view_surfaces,
        object_extent=object_extent,
        args=args,
    )

    if not args.skip_save_debug_clouds:
        candidate_colors = np.tile(np.array([[0.35, 0.58, 0.86]], dtype=np.float32), (candidate_points.shape[0], 1))
        consistent_colors = np.tile(np.array([[0.38, 0.70, 0.42]], dtype=np.float32), (consistent_points.shape[0], 1))
        candidate_debug = _save_point_cloud_limited(
            output_root / "surface_view_candidates.ply",
            candidate_points,
            candidate_normals,
            candidate_colors,
            max_points=int(args.max_debug_points),
            rng=rng,
        )
        consistent_debug = _save_point_cloud_limited(
            output_root / "surface_consistent_raw.ply",
            consistent_points,
            consistent_normals,
            consistent_colors,
            max_points=int(args.max_debug_points),
            rng=rng,
        )
    else:
        candidate_debug = {"written": False, "reason": "skip_save_debug_clouds"}
        consistent_debug = {"written": False, "reason": "skip_save_debug_clouds"}

    clean_points, clean_normals, clean_stats = _clean_surface_cloud(
        points=consistent_points,
        normals=consistent_normals,
        object_extent=object_extent,
        args=args,
    )
    if not args.disable_outward_normal_hint:
        clean_normals = _orient_normals_by_center(clean_points, clean_normals, object_center)

    target_num_probes = min(int(args.target_num_probes), int(clean_points.shape[0]))
    if target_num_probes <= 0:
        raise RuntimeError("target_num_probes must be positive.")
    probe_surface_indices = _farthest_point_sampling(clean_points, target_num_probes, device=torch.device("cuda"))
    probe_surface_points = clean_points[probe_surface_indices]
    probe_normals = _normalize_np(clean_normals[probe_surface_indices])

    offset_distance = float(args.offset_distance) if args.offset_distance > 0.0 else float(args.offset_scale) * object_extent
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0. Check extent_mode/offset_scale.")
    probe_points = probe_surface_points + probe_normals * offset_distance

    surface_colors = np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float32), (clean_points.shape[0], 1))
    probe_surface_colors = np.tile(np.array([[0.95, 0.62, 0.22]], dtype=np.float32), (probe_surface_points.shape[0], 1))
    probe_colors = np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1))

    _save_point_cloud(output_root / "surface_fused_clean.ply", clean_points, clean_normals, surface_colors)
    _save_point_cloud(output_root / "probe_surface_samples.ply", probe_surface_points, probe_normals, probe_surface_colors)
    _save_point_cloud(output_root / "probe_offset_points.ply", probe_points, probe_normals, probe_colors)

    normal_vis_idx = np.arange(probe_points.shape[0], dtype=np.int64)
    if args.max_probe_normals_vis > 0 and normal_vis_idx.shape[0] > args.max_probe_normals_vis:
        normal_vis_idx = np.sort(rng.choice(normal_vis_idx, size=args.max_probe_normals_vis, replace=False))
    _save_lineset(
        output_root / "probe_normals_lineset.ply",
        starts=probe_points[normal_vis_idx],
        ends=probe_points[normal_vis_idx] + probe_normals[normal_vis_idx] * offset_distance,
        color=(0.12, 0.47, 0.71),
    )

    np.savez(
        output_root / "probe_init_data.npz",
        surface_points=clean_points.astype(np.float32),
        surface_normals=clean_normals.astype(np.float32),
        probe_reference_points=clean_points.astype(np.float32),
        probe_reference_normals=clean_normals.astype(np.float32),
        probe_surface_points=probe_surface_points.astype(np.float32),
        probe_points=probe_points.astype(np.float32),
        probe_normals=probe_normals.astype(np.float32),
        probe_surface_indices=probe_surface_indices.astype(np.int64),
        consistent_points=consistent_points.astype(np.float32),
        consistent_normals=consistent_normals.astype(np.float32),
        consistent_source_view_indices=source_indices.astype(np.int32),
        consistent_support_counts=support_counts.astype(np.int16),
        offset_distance=np.array([offset_distance], dtype=np.float32),
    )

    summary = {
        "format": "irgs_sop_phase1_point_cloud_v1",
        "loaded_from": loaded_from,
        "output_root": str(output_root),
        "camera_set": args.camera_set,
        "probe_source": "multi_view_point_cloud",
        "num_selected_cameras": int(len(cameras)),
        "object_extent": float(object_extent),
        "extent_mode": args.extent_mode,
        "object_center": object_center.astype(float).tolist(),
        "candidate_points": int(candidate_points.shape[0]),
        "consistent_points": int(consistent_points.shape[0]),
        "num_surface_points": int(clean_points.shape[0]),
        "num_probes": int(probe_points.shape[0]),
        "offset_scale": float(args.offset_scale),
        "offset_distance": float(offset_distance),
        "filtering": {
            "weight_thresh": float(args.weight_thresh),
            "mask_thresh": float(args.mask_thresh),
            "object_filter_mode": args.object_filter_mode,
            "mask_erosion_radius": int(args.mask_erosion_radius),
            "normal_min_norm": float(args.normal_min_norm),
        },
        "cross_view_consistency": consistency_stats,
        "cleaning": clean_stats,
        "debug_clouds": {
            "candidate": candidate_debug,
            "consistent": consistent_debug,
        },
        "args": dict(vars(args)),
    }
    with open(output_root / "probe_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SOP-Phase1-PC] Candidate points: {candidate_points.shape[0]}")
    print(f"[SOP-Phase1-PC] Consistent points: {consistent_points.shape[0]}")
    print(f"[SOP-Phase1-PC] Clean surface points: {clean_points.shape[0]}")
    print(f"[SOP-Phase1-PC] Probes: {probe_points.shape[0]}")
    print(f"[SOP-Phase1-PC] Offset distance: {offset_distance:.6f}")
    print(f"[SOP-Phase1-PC] Output root: {output_root}")
    return summary


def _build_parser():
    parser = ArgumentParser(description="SOP phase1 initializer from multi-view point-cloud consistency for COMGS_IRGS")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--camera_set", choices=["train", "test", "all"], default="train")
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--max_points_per_view", default=50000, type=int)

    parser.add_argument("--weight_thresh", default=0.5, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument(
        "--object_filter_mode",
        choices=["weight_only", "mask_only", "weight_or_mask", "weight_and_mask"],
        default="weight_and_mask",
    )
    parser.add_argument("--mask_erosion_radius", default=4, type=int)
    parser.add_argument("--normal_min_norm", default=0.5, type=float)

    parser.add_argument("--consistency_min_views", default=2, type=int)
    parser.add_argument("--consistency_max_views", default=8, type=int)
    parser.add_argument("--consistency_view_stride", default=1, type=int)
    parser.add_argument("--consistency_chunk_size", default=65536, type=int)
    parser.add_argument("--depth_tolerance", default=0.0, type=float)
    parser.add_argument("--depth_tolerance_factor", default=0.005, type=float)
    parser.add_argument("--depth_relative_tolerance", default=0.01, type=float)
    parser.add_argument("--normal_angle_thresh", default=35.0, type=float)
    parser.add_argument("--use_abs_normal_dot", action="store_true")

    parser.add_argument("--fusion_voxel_factor", default=0.002, type=float)
    parser.add_argument("--fusion_voxel_size", default=0.0, type=float)
    parser.add_argument("--denoise_mode", choices=["none", "radius"], default="radius")
    parser.add_argument("--radius_denoise_radius", default=0.0, type=float)
    parser.add_argument("--radius_denoise_radius_factor", default=0.006, type=float)
    parser.add_argument("--radius_denoise_min_neighbors", default=4, type=int)

    parser.add_argument("--extent_mode", choices=["bbox_diagonal", "max_side"], default="bbox_diagonal")
    parser.add_argument("--target_num_probes", default=5000, type=int)
    parser.add_argument("--offset_scale", "--offset_factor", dest="offset_scale", default=0.005, type=float)
    parser.add_argument("--offset_distance", default=0.0, type=float)
    parser.add_argument("--disable_outward_normal_hint", action="store_true")

    parser.add_argument("--max_probe_normals_vis", default=400, type=int)
    parser.add_argument("--max_debug_points", default=500000, type=int)
    parser.add_argument("--skip_save_debug_clouds", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = get_combined_args(parser)
    if not getattr(args, "model_path", ""):
        raise RuntimeError("SOP point-cloud phase1 initializer requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("SOP point-cloud phase1 initializer requires --source_path/-s, or a cfg_args under --model_path.")
    _ensure_dir(Path(args.model_path))
    with open(Path(args.model_path) / "cfg_args", "w") as cfg_log_f:
        cfg_log_f.write(str(args))
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialize_sop_phase1_point_cloud(args)


if __name__ == "__main__":
    main()
