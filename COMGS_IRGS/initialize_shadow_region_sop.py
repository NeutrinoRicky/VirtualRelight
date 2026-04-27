from __future__ import annotations

import json
import math
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov

from render_lego_into_ignatius import (
    _ensure_repo_cwd,
    _as_parameter,
    apply_gaussian_transform,
    apply_post_transform_scale,
)
from SOP.phase1_initializer_point_cloud import (
    ViewSurface,
    _check_points_against_target,
    _clean_surface_cloud,
    _cross_view_consistency_filter,
    _farthest_point_sampling_with_min_distance,
    _normalize_np,
    _orient_normals_by_view_votes,
    _render_view_surface,
    _save_lineset,
    _save_point_cloud,
    _save_point_cloud_limited,
    _summarize_array,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_legacy_comgs_checkpoint(path: Path, gaussians: GaussianModel) -> int:
    payload = torch.load(path, map_location="cuda", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload at {path}")

    model_args = payload[0]
    loaded_iter = int(payload[1]) if len(payload) > 1 else 0
    if len(model_args) != 15:
        raise RuntimeError(
            f"{path} does not look like a legacy COMGS checkpoint; expected 15 model fields, got {len(model_args)}"
        )

    (
        gaussians.active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        albedo,
        roughness,
        metallic,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        _opt_dict,
        gaussians.spatial_lr_scale,
    ) = model_args

    gaussians._xyz = _as_parameter(xyz)
    gaussians._features_dc = _as_parameter(features_dc)
    gaussians._features_rest = _as_parameter(features_rest)
    gaussians._scaling = _as_parameter(scaling)
    gaussians._rotation = _as_parameter(rotation)
    gaussians._opacity = _as_parameter(opacity)
    gaussians._base_color = _as_parameter(albedo)
    gaussians._roughness = _as_parameter(roughness)
    gaussians._metallic = _as_parameter(metallic)
    gaussians.max_radii2D = max_radii2D.detach().cuda()
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().cuda()
    gaussians.denom = denom.detach().cuda()
    return loaded_iter


def load_gaussians_auto(checkpoint_path: Path, sh_degree: int, label: str) -> Tuple[GaussianModel, int, str]:
    gaussians = GaussianModel(sh_degree)
    payload = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    if isinstance(payload, dict) and "gaussians" in payload:
        gaussians.restore(payload["gaussians"], None)
        loaded_iter = int(payload.get("iteration", 0))
        fmt = str(payload.get("format", "checkpoint_dict"))
    elif isinstance(payload, (tuple, list)) and len(payload) >= 1:
        model_args = payload[0]
        loaded_iter = int(payload[1]) if len(payload) > 1 else 0
        if isinstance(model_args, (tuple, list)) and len(model_args) == 15:
            loaded_iter = _load_legacy_comgs_checkpoint(checkpoint_path, gaussians)
            fmt = "legacy_comgs"
        elif isinstance(model_args, (tuple, list)) and len(model_args) in {19, 26}:
            gaussians.restore_from_refgs(model_args, None)
            fmt = "refgs"
        elif isinstance(model_args, (tuple, list)) and len(model_args) == 16:
            gaussians.restore(model_args, None)
            fmt = "irgs"
        else:
            raise RuntimeError(
                f"Unsupported tuple checkpoint payload at {checkpoint_path}; model_args len="
                f"{len(model_args) if isinstance(model_args, (tuple, list)) else 'n/a'}"
            )
    else:
        raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload).__name__} at {checkpoint_path}")

    print(f"[load] {label}: {checkpoint_path} ({gaussians.get_xyz.shape[0]} gaussians, iter={loaded_iter}, format={fmt})")
    return gaussians, loaded_iter, fmt


def _bbox_info(points: np.ndarray) -> Dict[str, object]:
    mins = points.min(axis=0).astype(np.float32)
    maxs = points.max(axis=0).astype(np.float32)
    center = 0.5 * (mins + maxs)
    side_lengths = maxs - mins
    diagonal = float(np.linalg.norm(side_lengths))
    radius = 0.5 * diagonal
    max_side = float(np.max(side_lengths))
    mean = points.mean(axis=0).astype(np.float32)
    return {
        "min": mins,
        "max": maxs,
        "center": center,
        "mean": mean,
        "side_lengths": side_lengths.astype(np.float32),
        "diagonal": diagonal,
        "radius": float(radius),
        "max_side": max_side,
    }


def _compute_extent(points: np.ndarray, mode: str) -> float:
    bbox = _bbox_info(points)
    if mode == "bbox_diagonal":
        return float(bbox["diagonal"])
    if mode == "max_side":
        return float(bbox["max_side"])
    raise ValueError(f"Unsupported extent mode: {mode}")


def _axis_to_vector(axis_name: str) -> np.ndarray:
    axis_name = axis_name.lower()
    table = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    if axis_name not in table:
        raise ValueError(f"Unsupported axis specifier: {axis_name}")
    return table[axis_name]


def _make_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = axis.astype(np.float32)
    axis = axis / np.linalg.norm(axis)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, fallback))) > 0.95:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent = np.cross(axis, fallback)
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)
    return tangent.astype(np.float32), bitangent.astype(np.float32), axis.astype(np.float32)


def sample_hemisphere_points(
    center: np.ndarray,
    radius: float,
    up_axis: np.ndarray,
    num_points: int,
    min_elevation_deg: float,
) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    tangent, bitangent, up = _make_basis_from_axis(up_axis)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    min_up_dot = math.sin(math.radians(float(min_elevation_deg)))
    min_up_dot = float(np.clip(min_up_dot, 0.0, 1.0))

    samples = np.zeros((num_points, 3), dtype=np.float32)
    for idx in range(num_points):
        frac = (idx + 0.5) / float(num_points)
        up_dot = min_up_dot + (1.0 - min_up_dot) * frac
        radial = math.sqrt(max(1.0 - up_dot * up_dot, 0.0))
        azimuth = golden_angle * idx
        local = (
            math.cos(azimuth) * radial * tangent
            + math.sin(azimuth) * radial * bitangent
            + up_dot * up
        )
        samples[idx] = center + float(radius) * local.astype(np.float32)
    return samples.astype(np.float32)


def sample_sphere_points(center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    samples = np.zeros((num_points, 3), dtype=np.float32)
    for idx in range(num_points):
        frac = (idx + 0.5) / float(num_points)
        z = 1.0 - 2.0 * frac
        radial = math.sqrt(max(1.0 - z * z, 0.0))
        azimuth = golden_angle * idx
        direction = np.array(
            [
                math.cos(azimuth) * radial,
                z,
                math.sin(azimuth) * radial,
            ],
            dtype=np.float32,
        )
        samples[idx] = center + float(radius) * direction
    return samples.astype(np.float32)


def _lookat_c2w(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = target.astype(np.float32) - position.astype(np.float32)
    forward = forward / np.clip(np.linalg.norm(forward), 1e-8, None)

    right = np.cross(up_hint.astype(np.float32), forward)
    if np.linalg.norm(right) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, forward))) > 0.95:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(fallback, forward)
    right = right / np.clip(np.linalg.norm(right), 1e-8, None)
    up = np.cross(forward, right)
    up = up / np.clip(np.linalg.norm(up), 1e-8, None)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position.astype(np.float32)
    return c2w


def build_lookat_camera(
    position: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
    image_width: int,
    image_height: int,
    fov_deg: float,
    uid: int,
) -> Tuple[Camera, np.ndarray]:
    c2w = _lookat_c2w(position=position, target=target, up_hint=up_hint)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]

    fovx = math.radians(float(fov_deg))
    focal = float(image_width) / (2.0 * math.tan(fovx * 0.5))
    fovy = focal2fov(focal, int(image_height))
    K = np.array(
        [
            [focal, 0.0, image_width / 2.0],
            [0.0, focal, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image = torch.zeros(3, int(image_height), int(image_width))
    camera = Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        gt_alpha_mask=None,
        image_name=f"hemi_{uid:04d}",
        uid=uid,
        data_device="cuda",
        HWK=(int(image_height), int(image_width), K),
    )
    return camera, c2w


def _compute_camera_fov_deg(
    target_radius: float,
    camera_distance: float,
    fov_margin: float,
    min_fov_deg: float,
    max_fov_deg: float,
) -> float:
    if target_radius <= 0.0 or camera_distance <= 0.0:
        raise ValueError("target_radius and camera_distance must be positive.")
    raw = 2.0 * math.degrees(math.atan(float(fov_margin) * float(target_radius) / float(camera_distance)))
    return float(np.clip(raw, min_fov_deg, max_fov_deg))


def _points_inside_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.all(points >= bbox_min[None, :], axis=1) & np.all(points <= bbox_max[None, :], axis=1)


def _filter_points_to_shadow_region(
    points: np.ndarray,
    shadow_center: np.ndarray,
    shadow_radius: float,
    object_bbox_min: np.ndarray,
    object_bbox_max: np.ndarray,
    object_bbox_margin: float,
) -> np.ndarray:
    rel = points - shadow_center.reshape(1, 3)
    inside_sphere = np.linalg.norm(rel, axis=1) <= float(shadow_radius)
    bbox_min = object_bbox_min - float(object_bbox_margin)
    bbox_max = object_bbox_max + float(object_bbox_margin)
    outside_object_bbox = ~_points_inside_bbox(points, bbox_min, bbox_max)
    return inside_sphere & outside_object_bbox


def _filter_view_surface(
    view_surface: ViewSurface,
    shadow_center: np.ndarray,
    shadow_radius: float,
    object_bbox_min: np.ndarray,
    object_bbox_max: np.ndarray,
    object_bbox_margin: float,
    max_points_after_region_filter: int,
) -> Tuple[ViewSurface, Dict[str, int]]:
    if view_surface.points.shape[0] == 0:
        stats = {
            "raw_candidates": 0,
            "after_region_filter": 0,
        }
        return view_surface, stats

    points = view_surface.points.numpy().astype(np.float32)
    normals = view_surface.normals.numpy().astype(np.float32)
    keep = _filter_points_to_shadow_region(
        points=points,
        shadow_center=shadow_center,
        shadow_radius=shadow_radius,
        object_bbox_min=object_bbox_min,
        object_bbox_max=object_bbox_max,
        object_bbox_margin=object_bbox_margin,
    )
    stats = {
        "raw_candidates": int(points.shape[0]),
        "after_region_filter": int(np.sum(keep)),
    }
    points = points[keep]
    normals = normals[keep]

    if max_points_after_region_filter > 0 and points.shape[0] > max_points_after_region_filter:
        choice = np.linspace(0, points.shape[0] - 1, num=max_points_after_region_filter, dtype=np.float32)
        choice = np.unique(np.round(choice).astype(np.int64))
        points = points[choice]
        normals = normals[choice]
        stats["after_cap"] = int(points.shape[0])
    else:
        stats["after_cap"] = int(points.shape[0])

    filtered = ViewSurface(
        index=view_surface.index,
        camera=view_surface.camera,
        depth=view_surface.depth,
        normal=view_surface.normal,
        valid=view_surface.valid,
        points=torch.from_numpy(points.astype(np.float32)),
        normals=torch.from_numpy(normals.astype(np.float32)),
        num_valid_pixels=view_surface.num_valid_pixels,
    )
    return filtered, stats


def _voxel_downsample_points_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points.astype(np.float32), colors.astype(np.float32)

    mins = points.min(axis=0, keepdims=True)
    voxel_coords = np.floor((points - mins) / voxel_size).astype(np.int64)
    unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    point_sums = np.zeros((unique_coords.shape[0], 3), dtype=np.float64)
    color_sums = np.zeros((unique_coords.shape[0], 3), dtype=np.float64)
    counts = np.zeros((unique_coords.shape[0], 1), dtype=np.float64)
    np.add.at(point_sums, inverse, points.astype(np.float64))
    np.add.at(color_sums, inverse, colors.astype(np.float64))
    np.add.at(counts, inverse, 1.0)
    down_points = (point_sums / np.clip(counts, 1.0, None)).astype(np.float32)
    down_colors = (color_sums / np.clip(counts, 1.0, None)).astype(np.float32)
    return down_points, np.clip(down_colors, 0.0, 1.0)


def _save_line_segments_obj(
    path: Path,
    starts: np.ndarray,
    ends: np.ndarray,
    header_comments: Sequence[str] | None = None,
) -> None:
    if starts.shape[0] != ends.shape[0]:
        raise ValueError("starts and ends must have the same number of rows.")

    lines: List[str] = []
    if header_comments:
        for comment in header_comments:
            lines.append(f"# {comment}")
    for point in np.concatenate([starts, ends], axis=0):
        lines.append(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}")
    offset = starts.shape[0]
    for idx in range(starts.shape[0]):
        lines.append(f"l {idx + 1} {offset + idx + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_bbox_wireframe_obj(path: Path, bbox_min: np.ndarray, bbox_max: np.ndarray, title: str) -> None:
    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=np.float32,
    )
    edges = (
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 5),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
    )
    lines = [f"# {title}"]
    for point in corners:
        lines.append(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}")
    for start, end in edges:
        lines.append(f"l {start} {end}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _camera_frustum_segments(
    c2w: np.ndarray,
    fovx_deg: float,
    image_width: int,
    image_height: int,
    frustum_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    center = c2w[:3, 3].astype(np.float32)
    fovx = math.radians(float(fovx_deg))
    focal = float(image_width) / (2.0 * math.tan(fovx * 0.5))
    fovy = focal2fov(focal, int(image_height))
    half_w = math.tan(fovx * 0.5) * frustum_length
    half_h = math.tan(fovy * 0.5) * frustum_length
    corners_cam = np.array(
        [
            [-half_w, -half_h, frustum_length],
            [half_w, -half_h, frustum_length],
            [half_w, half_h, frustum_length],
            [-half_w, half_h, frustum_length],
        ],
        dtype=np.float32,
    )
    rot = c2w[:3, :3].astype(np.float32)
    corners_world = corners_cam @ rot.T + center[None, :]

    starts = []
    ends = []
    for corner in corners_world:
        starts.append(center)
        ends.append(corner)
    rect_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    for src, dst in rect_edges:
        starts.append(corners_world[src])
        ends.append(corners_world[dst])
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


def _save_camera_frustums_obj(
    path: Path,
    camera_records: List[Dict[str, object]],
    frustum_length: float,
) -> None:
    all_starts = []
    all_ends = []
    for record in camera_records:
        starts, ends = _camera_frustum_segments(
            c2w=np.asarray(record["c2w"], dtype=np.float32),
            fovx_deg=float(record["fovx_deg"]),
            image_width=int(record["width"]),
            image_height=int(record["height"]),
            frustum_length=float(frustum_length),
        )
        all_starts.append(starts)
        all_ends.append(ends)
    if not all_starts:
        return
    _save_line_segments_obj(
        path=path,
        starts=np.concatenate(all_starts, axis=0),
        ends=np.concatenate(all_ends, axis=0),
        header_comments=["sampled hemisphere camera frustums"],
    )


def _serialize_camera_record(camera: Camera, c2w: np.ndarray) -> Dict[str, object]:
    focal_x = float(camera.HWK[2][0, 0])
    focal_y = float(camera.HWK[2][1, 1])
    return {
        "name": camera.image_name,
        "uid": int(camera.uid),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "position": c2w[:3, 3].astype(float).tolist(),
        "rotation": c2w[:3, :3].astype(float).tolist(),
        "c2w": c2w.astype(float).tolist(),
        "fx": focal_x,
        "fy": focal_y,
        "fovx_deg": math.degrees(float(camera.FoVx)),
        "fovy_deg": math.degrees(float(camera.FoVy)),
    }


def _resolve_probe_extent_reference(
    args,
    object_bbox: Dict[str, object],
    shadow_radius: float,
    local_clean_points: np.ndarray,
) -> float:
    if args.probe_extent_reference == "object_bbox_diagonal":
        return float(object_bbox["diagonal"])
    if args.probe_extent_reference == "object_max_side":
        return float(object_bbox["max_side"])
    if args.probe_extent_reference == "object_radius":
        return float(object_bbox["radius"])
    if args.probe_extent_reference == "shadow_radius":
        return float(shadow_radius)
    if args.probe_extent_reference == "local_surface_bbox_diagonal":
        return _compute_extent(local_clean_points, "bbox_diagonal")
    if args.probe_extent_reference == "local_surface_max_side":
        return _compute_extent(local_clean_points, "max_side")
    raise ValueError(f"Unsupported probe_extent_reference: {args.probe_extent_reference}")


def _sample_rows(points: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or points.shape[0] <= max_count:
        return points
    indices = np.sort(rng.choice(points.shape[0], size=max_count, replace=False))
    return points[indices]


def _project_points_to_plane(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    signed_distance = np.sum((points - plane_point[None, :]) * plane_normal[None, :], axis=1)
    return (points - signed_distance[:, None] * plane_normal[None, :]).astype(np.float32)


def _make_plane_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = normal.astype(np.float32)
    normal = normal / np.clip(np.linalg.norm(normal), 1e-8, None)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(fallback, normal))) > 0.95:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent = np.cross(normal, fallback)
    tangent = tangent / np.clip(np.linalg.norm(tangent), 1e-8, None)
    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / np.clip(np.linalg.norm(bitangent), 1e-8, None)
    return tangent.astype(np.float32), bitangent.astype(np.float32), normal.astype(np.float32)


def _plane_uv(points: np.ndarray, origin: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray) -> np.ndarray:
    rel = points - origin[None, :]
    u = rel @ tangent
    v = rel @ bitangent
    return np.stack([u, v], axis=1).astype(np.float32)


def _nearest_distance_2d(query_uv: np.ndarray, ref_uv: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    if query_uv.shape[0] == 0 or ref_uv.shape[0] == 0:
        return np.full((query_uv.shape[0],), np.inf, dtype=np.float32)
    out = np.full((query_uv.shape[0],), np.inf, dtype=np.float32)
    for start in range(0, query_uv.shape[0], chunk_size):
        end = min(start + chunk_size, query_uv.shape[0])
        diff = query_uv[start:end, None, :] - ref_uv[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        out[start:end] = np.min(dist, axis=1).astype(np.float32)
    return out


def _resample_support_surface_to_plane(
    points: np.ndarray,
    normals: np.ndarray,
    target_num_points: int,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": True,
        "input_points": int(points.shape[0]),
        "target_num_points": int(target_num_points),
    }
    if points.shape[0] == 0:
        stats["reason"] = "empty_input"
        return points, normals, stats
    if points.shape[0] >= target_num_points:
        stats["reason"] = "not_needed"
        return points, normals, stats

    plane_normal = normals.mean(axis=0).astype(np.float32)
    if np.linalg.norm(plane_normal) < 1e-8:
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent, bitangent, plane_normal = _make_plane_tangent_basis(plane_normal)
    plane_point = points.mean(axis=0).astype(np.float32)
    support_uv = _plane_uv(points, plane_point, tangent, bitangent)

    uv_min = support_uv.min(axis=0)
    uv_max = support_uv.max(axis=0)
    uv_extent = np.maximum(uv_max - uv_min, 1e-6)
    bbox_area = float(uv_extent[0] * uv_extent[1])

    pairwise_diff = support_uv[:, None, :] - support_uv[None, :, :]
    pairwise_dist = np.sqrt(np.sum(pairwise_diff * pairwise_diff, axis=-1))
    np.fill_diagonal(pairwise_dist, np.inf)
    valid_nn = np.min(pairwise_dist, axis=1)
    valid_nn = valid_nn[np.isfinite(valid_nn)]
    if valid_nn.shape[0] == 0:
        valid_nn = np.array([math.sqrt(max(bbox_area, 1e-6))], dtype=np.float32)

    oversample = max(1.0, float(args.support_resample_oversample_factor))
    base_spacing = math.sqrt(max(bbox_area, 1e-6) / max(float(target_num_points) * oversample, 1.0))
    base_spacing = max(base_spacing, 1e-4)
    spacing = base_spacing
    occupancy_radius = max(
        float(np.percentile(valid_nn, 75)) * float(args.support_resample_occupancy_radius_factor),
        spacing * 0.75,
    )

    selected_grid_uv = np.zeros((0, 2), dtype=np.float32)
    rounds = max(1, int(args.support_resample_max_rounds))
    for _ in range(rounds):
        pad = spacing * float(args.support_resample_bbox_pad)
        u_vals = np.arange(uv_min[0] - pad, uv_max[0] + pad + spacing * 0.5, spacing, dtype=np.float32)
        v_vals = np.arange(uv_min[1] - pad, uv_max[1] + pad + spacing * 0.5, spacing, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(u_vals, v_vals, indexing="xy")
        grid_uv = np.stack([grid_u.reshape(-1), grid_v.reshape(-1)], axis=1).astype(np.float32)
        grid_min_dist = _nearest_distance_2d(grid_uv, support_uv)
        keep = grid_min_dist <= occupancy_radius
        selected_grid_uv = grid_uv[keep]
        if selected_grid_uv.shape[0] >= target_num_points:
            break
        spacing *= float(args.support_resample_spacing_shrink)
        occupancy_radius *= float(args.support_resample_radius_expand)

    if selected_grid_uv.shape[0] == 0:
        stats.update(
            {
                "reason": "empty_grid_after_filter",
                "bbox_area": float(bbox_area),
                "spacing": float(spacing),
                "occupancy_radius": float(occupancy_radius),
            }
        )
        return points, normals, stats

    dense_points = (
        plane_point[None, :]
        + selected_grid_uv[:, 0:1] * tangent[None, :]
        + selected_grid_uv[:, 1:2] * bitangent[None, :]
    ).astype(np.float32)
    dense_normals = np.tile(plane_normal[None, :], (dense_points.shape[0], 1)).astype(np.float32)
    stats.update(
        {
            "reason": "ok",
            "bbox_area": float(bbox_area),
            "base_spacing": float(base_spacing),
            "final_spacing": float(spacing),
            "final_occupancy_radius": float(occupancy_radius),
            "generated_points": int(dense_points.shape[0]),
            "plane_point": plane_point.astype(float).tolist(),
            "plane_normal": plane_normal.astype(float).tolist(),
            "uv_extent": uv_extent.astype(float).tolist(),
        }
    )
    return dense_points.astype(np.float32), _normalize_np(dense_normals.astype(np.float32)), stats


def _extract_support_surface(
    points: np.ndarray,
    normals: np.ndarray,
    up_axis: np.ndarray,
    object_points: np.ndarray,
    object_radius: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": True,
        "input_points": int(points.shape[0]),
    }
    if points.shape[0] == 0:
        stats["reason"] = "empty_input"
        return points, normals, stats

    stats.update(
        {
            "reason": "ok",
            "mode": "passthrough_all_local_surface",
            "candidate_points": int(points.shape[0]),
            "kept_points": int(points.shape[0]),
            "snap_to_plane": False,
            "resample_to_plane": False,
            "normal_source": "rendered_normals_passthrough",
        }
    )
    return points.astype(np.float32), _normalize_np(normals.astype(np.float32)), stats


@torch.no_grad()
def _count_surface_visibility_support(
    points: np.ndarray,
    normals: np.ndarray,
    views: List[ViewSurface],
    object_extent: float,
    args,
) -> np.ndarray:
    if points.shape[0] == 0 or len(views) == 0:
        return np.zeros((points.shape[0],), dtype=np.int16)

    device = torch.device("cuda")
    pts = torch.from_numpy(points.astype(np.float32)).to(device=device)
    nrms = torch.from_numpy(_normalize_np(normals.astype(np.float32))).to(device=device)
    depth_abs_tol = (
        float(args.depth_tolerance)
        if args.depth_tolerance > 0.0
        else float(args.depth_tolerance_factor) * float(object_extent)
    )
    normal_cos_thresh = math.cos(math.radians(float(args.normal_angle_thresh)))
    chunk_size = max(1, int(args.consistency_chunk_size))
    support_counts = np.zeros((points.shape[0],), dtype=np.int16)

    for target in views:
        target_depth = target.depth.unsqueeze(0).to(device=device, dtype=torch.float32)
        target_normal = target.normal.unsqueeze(0).to(device=device, dtype=torch.float32)
        target_valid = target.valid.float().unsqueeze(0).to(device=device, dtype=torch.float32)

        for start in range(0, pts.shape[0], chunk_size):
            end = min(start + chunk_size, pts.shape[0])
            success = _check_points_against_target(
                points=pts[start:end],
                normals=nrms[start:end],
                target_camera=target.camera,
                target_depth=target_depth,
                target_normal=target_normal,
                target_valid=target_valid,
                depth_abs_tol=depth_abs_tol,
                depth_rel_tol=float(args.depth_relative_tolerance),
                normal_cos_thresh=normal_cos_thresh,
                use_abs_normal_dot=bool(args.use_abs_normal_dot),
            )
            support_counts[start:end] += success.detach().cpu().numpy().astype(np.int16)

        del target_depth, target_normal, target_valid

    return support_counts.astype(np.int16)


@torch.no_grad()
def _resolve_probe_surface_side_from_red_points(
    probe_points: np.ndarray,
    probe_normals: np.ndarray,
    views: List[ViewSurface],
    object_extent: float,
    args,
    offset_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": True,
        "num_probes": int(probe_points.shape[0]),
        "offset_distance": float(offset_distance),
    }
    if probe_points.shape[0] == 0:
        stats["reason"] = "empty_input"
        return probe_points, probe_points, probe_normals, stats

    normals = _normalize_np(probe_normals.astype(np.float32))
    surface_a = (probe_points - normals * float(offset_distance)).astype(np.float32)
    normal_a = normals
    surface_b = (probe_points + normals * float(offset_distance)).astype(np.float32)
    normal_b = (-normals).astype(np.float32)

    support_a = _count_surface_visibility_support(
        points=surface_a,
        normals=normal_a,
        views=views,
        object_extent=object_extent,
        args=args,
    )
    support_b = _count_surface_visibility_support(
        points=surface_b,
        normals=normal_b,
        views=views,
        object_extent=object_extent,
        args=args,
    )

    choose_b = support_b > support_a
    corrected_surface = np.where(choose_b[:, None], surface_b, surface_a).astype(np.float32)
    corrected_normals = np.where(choose_b[:, None], normal_b, normal_a).astype(np.float32)
    corrected_probe_points = (corrected_surface + corrected_normals * float(offset_distance)).astype(np.float32)

    stats.update(
        {
            "reason": "ok",
            "side_a_selected": int(np.sum(~choose_b)),
            "side_b_selected": int(np.sum(choose_b)),
            "tie_kept_side_a": int(np.sum(support_a == support_b)),
            "both_zero_support": int(np.sum((support_a == 0) & (support_b == 0))),
            "support_a": _summarize_array(support_a.astype(np.float32)),
            "support_b": _summarize_array(support_b.astype(np.float32)),
            "support_delta_b_minus_a": _summarize_array((support_b.astype(np.float32) - support_a.astype(np.float32))),
        }
    )
    return corrected_surface, corrected_probe_points, _normalize_np(corrected_normals), stats


def _save_scene_and_probe_combo(
    path: Path,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    probe_surface_points: np.ndarray,
    probe_points: np.ndarray,
) -> None:
    chunks = []
    colors = []
    normals = []
    if scene_points.shape[0] > 0:
        chunks.append(scene_points.astype(np.float32))
        colors.append(np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float32), (scene_points.shape[0], 1)))
        normals.append(np.zeros_like(scene_points, dtype=np.float32))
    if object_points.shape[0] > 0:
        chunks.append(object_points.astype(np.float32))
        colors.append(np.tile(np.array([[0.20, 0.72, 0.40]], dtype=np.float32), (object_points.shape[0], 1)))
        normals.append(np.zeros_like(object_points, dtype=np.float32))
    if probe_surface_points.shape[0] > 0:
        chunks.append(probe_surface_points.astype(np.float32))
        colors.append(np.tile(np.array([[0.95, 0.62, 0.22]], dtype=np.float32), (probe_surface_points.shape[0], 1)))
        normals.append(np.zeros_like(probe_surface_points, dtype=np.float32))
    if probe_points.shape[0] > 0:
        chunks.append(probe_points.astype(np.float32))
        colors.append(np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1)))
        normals.append(np.zeros_like(probe_points, dtype=np.float32))
    if not chunks:
        return
    _save_point_cloud(
        path,
        np.concatenate(chunks, axis=0),
        np.concatenate(normals, axis=0),
        np.concatenate(colors, axis=0),
    )


@torch.no_grad()
def initialize_shadow_region_sop(args) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Shadow-region SOP initialization currently requires CUDA.")

    _ensure_repo_cwd()
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_root = _ensure_dir(Path(args.output_dir))
    vis_root = _ensure_dir(output_root / "visualization")

    scene_gaussians, scene_iter, scene_format = load_gaussians_auto(Path(args.scene_checkpoint), sh_degree=3, label="scene")
    object_gaussians, object_iter, object_format = load_gaussians_auto(Path(args.object_checkpoint), sh_degree=3, label="object")

    transform = apply_gaussian_transform(
        object_gaussians,
        args.translation,
        args.rotation_deg,
        args.scale,
        args.rotation_order,
    )
    post_scale_center = apply_post_transform_scale(object_gaussians, args.post_scale, args.post_scale_center)

    object_points = object_gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    object_colors = object_gaussians.get_base_color.detach().cpu().numpy().astype(np.float32)
    object_bbox = _bbox_info(object_points)

    obj_radius = float(object_bbox["radius"])
    if obj_radius <= 0.0:
        raise RuntimeError("Object bbox radius is zero after placement. Check transform/scale settings.")
    shadow_center = np.asarray(object_bbox["center"], dtype=np.float32) + np.asarray(args.shadow_center_offset, dtype=np.float32)
    shadow_radius = float(args.shadow_radius_factor) * obj_radius
    object_bbox_margin = float(args.object_bbox_margin) if args.object_bbox_margin > 0.0 else float(args.object_bbox_margin_factor) * obj_radius
    if shadow_radius <= 0.0:
        raise RuntimeError("Computed shadow_radius <= 0. Check --shadow_radius_factor.")

    up_axis = _axis_to_vector(args.up_axis)
    hemisphere_distance = (
        float(args.hemisphere_distance)
        if args.hemisphere_distance > 0.0
        else float(args.hemisphere_distance_factor) * shadow_radius
    )
    if hemisphere_distance <= shadow_radius:
        print(
            "[warn] hemisphere_distance is not larger than shadow_radius. "
            "The sampled cameras may be too close to cover the full shadow region."
        )

    camera_fov_deg = (
        float(args.camera_fov_deg)
        if args.camera_fov_deg > 0.0
        else _compute_camera_fov_deg(
            target_radius=shadow_radius,
            camera_distance=hemisphere_distance,
            fov_margin=float(args.camera_fov_margin),
            min_fov_deg=float(args.camera_fov_min_deg),
            max_fov_deg=float(args.camera_fov_max_deg),
        )
    )

    dense_hemisphere_points = sample_hemisphere_points(
        center=shadow_center,
        radius=hemisphere_distance,
        up_axis=up_axis,
        num_points=int(args.hemisphere_vis_points),
        min_elevation_deg=float(args.hemisphere_min_elevation_deg),
    )
    camera_centers = sample_hemisphere_points(
        center=shadow_center,
        radius=hemisphere_distance,
        up_axis=up_axis,
        num_points=int(args.num_hemisphere_cameras),
        min_elevation_deg=float(args.hemisphere_min_elevation_deg),
    )

    cameras: List[Camera] = []
    camera_records: List[Dict[str, object]] = []
    for idx, cam_center in enumerate(camera_centers):
        camera, c2w = build_lookat_camera(
            position=cam_center,
            target=shadow_center,
            up_hint=up_axis,
            image_width=int(args.camera_width),
            image_height=int(args.camera_height),
            fov_deg=float(camera_fov_deg),
            uid=idx,
        )
        cameras.append(camera)
        camera_record = _serialize_camera_record(camera=camera, c2w=c2w)
        camera_record["look_at"] = shadow_center.astype(float).tolist()
        camera_records.append(camera_record)

    _save_point_cloud(
        vis_root / "object_transformed_gaussians.ply",
        object_points,
        np.zeros_like(object_points, dtype=np.float32),
        object_colors,
    )
    _save_point_cloud(
        vis_root / "hemisphere_surface_points.ply",
        dense_hemisphere_points,
        np.zeros_like(dense_hemisphere_points, dtype=np.float32),
        np.tile(np.array([[0.35, 0.66, 0.93]], dtype=np.float32), (dense_hemisphere_points.shape[0], 1)),
    )
    _save_point_cloud(
        vis_root / "hemisphere_camera_centers.ply",
        camera_centers,
        np.zeros_like(camera_centers, dtype=np.float32),
        np.tile(np.array([[0.17, 0.48, 0.73]], dtype=np.float32), (camera_centers.shape[0], 1)),
    )
    shadow_shell_points = sample_sphere_points(
        center=shadow_center,
        radius=shadow_radius,
        num_points=max(2048, int(args.hemisphere_vis_points)),
    )
    _save_point_cloud(
        vis_root / "shadow_region_sphere_points.ply",
        shadow_shell_points,
        np.zeros_like(shadow_shell_points, dtype=np.float32),
        np.tile(np.array([[0.60, 0.75, 0.98]], dtype=np.float32), (shadow_shell_points.shape[0], 1)),
    )
    _save_bbox_wireframe_obj(
        vis_root / "object_bbox_wireframe.obj",
        np.asarray(object_bbox["min"], dtype=np.float32),
        np.asarray(object_bbox["max"], dtype=np.float32),
        title="transformed object bbox",
    )
    _save_camera_frustums_obj(
        vis_root / "hemisphere_camera_frustums.obj",
        camera_records=camera_records,
        frustum_length=float(args.camera_frustum_length_factor) * shadow_radius,
    )
    _save_line_segments_obj(
        vis_root / "hemisphere_camera_lookat_lines.obj",
        starts=camera_centers.astype(np.float32),
        ends=np.repeat(shadow_center.reshape(1, 3), camera_centers.shape[0], axis=0).astype(np.float32),
        header_comments=["camera centers to shadow-region center"],
    )
    _write_json(
        vis_root / "hemisphere_cameras.json",
        {
            "shadow_center": shadow_center.astype(float).tolist(),
            "shadow_radius": float(shadow_radius),
            "hemisphere_distance": float(hemisphere_distance),
            "up_axis": up_axis.astype(float).tolist(),
            "camera_fov_deg": float(camera_fov_deg),
            "cameras": camera_records,
        },
    )

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=float(args.depth_ratio),
        debug=bool(args.debug),
    )
    background_value = 1.0 if args.white_background else 0.0
    background = torch.tensor([background_value, background_value, background_value], dtype=torch.float32, device="cuda")

    print(f"[shadow-sop] Rendering {len(cameras)} hemisphere cameras.")
    filtered_views: List[ViewSurface] = []
    per_view_stats: List[Dict[str, object]] = []
    for view_pos, camera in enumerate(cameras):
        view_surface = _render_view_surface(
            view_index=view_pos,
            viewpoint_cam=camera,
            gaussians=scene_gaussians,
            pipe=pipe,
            background=background,
            args=args,
        )
        filtered_surface, view_stats = _filter_view_surface(
            view_surface=view_surface,
            shadow_center=shadow_center,
            shadow_radius=shadow_radius,
            object_bbox_min=np.asarray(object_bbox["min"], dtype=np.float32),
            object_bbox_max=np.asarray(object_bbox["max"], dtype=np.float32),
            object_bbox_margin=object_bbox_margin,
            max_points_after_region_filter=int(args.max_points_after_region_filter),
        )
        filtered_views.append(filtered_surface)
        per_view_stats.append(
            {
                "view_index": int(view_pos),
                "camera_name": camera.image_name,
                "valid_pixels": int(view_surface.num_valid_pixels),
                **view_stats,
            }
        )
        print(
            "[shadow-sop] View "
            f"{view_pos + 1}/{len(cameras)}: raw={view_stats['raw_candidates']}, "
            f"region={view_stats['after_region_filter']}, final={view_stats['after_cap']}"
        )
        torch.cuda.empty_cache()

    if sum(int(v.points.shape[0]) for v in filtered_views) == 0:
        raise RuntimeError(
            "No local scene points survived the shadow-region filter. "
            "Increase hemisphere coverage, lower weight thresholds, or move the shadow center."
        )

    candidate_points = np.concatenate(
        [v.points.numpy() for v in filtered_views if v.points.shape[0] > 0],
        axis=0,
    ).astype(np.float32)
    candidate_normals = _normalize_np(
        np.concatenate(
            [v.normals.numpy() for v in filtered_views if v.normals.shape[0] > 0],
            axis=0,
        ).astype(np.float32)
    )
    local_extent = _compute_extent(candidate_points, args.extent_mode)
    if local_extent <= 0.0:
        raise RuntimeError("Local candidate point cloud has zero extent.")

    consistent_points, consistent_normals, source_indices, support_counts, consistency_stats = _cross_view_consistency_filter(
        views=filtered_views,
        object_extent=local_extent,
        args=args,
    )
    if consistent_points.shape[0] == 0:
        raise RuntimeError(
            "Cross-view consistency removed all local candidates. "
            "Try lowering --consistency_min_views or --normal_angle_thresh."
        )

    candidate_debug = _save_point_cloud_limited(
        output_root / "surface_view_candidates.ply",
        candidate_points,
        candidate_normals,
        np.tile(np.array([[0.35, 0.58, 0.86]], dtype=np.float32), (candidate_points.shape[0], 1)),
        max_points=int(args.max_debug_points),
        rng=rng,
    )
    consistent_debug = _save_point_cloud_limited(
        output_root / "surface_consistent_raw.ply",
        consistent_points,
        consistent_normals,
        np.tile(np.array([[0.38, 0.70, 0.42]], dtype=np.float32), (consistent_points.shape[0], 1)),
        max_points=int(args.max_debug_points),
        rng=rng,
    )

    clean_points, clean_normals, clean_stats = _clean_surface_cloud(
        points=consistent_points,
        normals=consistent_normals,
        object_extent=local_extent,
        args=args,
    )
    if clean_points.shape[0] == 0:
        raise RuntimeError("Local surface cleaning removed all points.")

    clean_normals, normal_orientation_stats = _orient_normals_by_view_votes(
        points=clean_points,
        normals=clean_normals,
        views=filtered_views,
        object_extent=local_extent,
        args=args,
    )
    _save_point_cloud(
        output_root / "surface_fused_clean_all.ply",
        clean_points,
        clean_normals,
        np.tile(np.array([[0.72, 0.72, 0.72]], dtype=np.float32), (clean_points.shape[0], 1)),
    )

    support_points, support_normals, support_surface_stats = _extract_support_surface(
        points=clean_points,
        normals=clean_normals,
        up_axis=up_axis,
        object_points=object_points,
        object_radius=obj_radius,
        args=args,
    )
    if support_points.shape[0] == 0:
        raise RuntimeError("Support-surface extraction removed all points.")
    support_resample_stats: Dict[str, object] = {
        "enabled": False,
        "reason": "disabled_conservative_top_facing_only",
        "input_points": int(support_points.shape[0]),
    }
    clean_points = support_points
    clean_normals = support_normals
    _save_point_cloud(
        output_root / "surface_passthrough_candidates.ply",
        clean_points,
        clean_normals,
        np.tile(np.array([[0.78, 0.78, 0.78]], dtype=np.float32), (clean_points.shape[0], 1)),
    )

    target_num_probes = int(args.target_num_probes)
    if clean_points.shape[0] < target_num_probes:
        raise RuntimeError(
            f"Need exactly {target_num_probes} probes, but only {clean_points.shape[0]} clean local surface points are available."
        )

    probe_extent_reference = _resolve_probe_extent_reference(
        args=args,
        object_bbox=object_bbox,
        shadow_radius=shadow_radius,
        local_clean_points=clean_points,
    )
    offset_distance = (
        float(args.offset_distance)
        if args.offset_distance > 0.0
        else float(args.offset_scale) * probe_extent_reference
    )
    probe_min_distance = (
        float(args.probe_min_distance)
        if args.probe_min_distance > 0.0
        else float(args.probe_min_distance_factor) * probe_extent_reference
    )
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0. Check offset settings.")

    clean_probe_normals = _normalize_np(clean_normals.astype(np.float32))
    candidate_probe_points = clean_points + clean_probe_normals * offset_distance
    probe_surface_indices, probe_selection_stats = _farthest_point_sampling_with_min_distance(
        candidate_probe_points,
        target_num_probes,
        device=torch.device("cuda"),
        min_distance=probe_min_distance,
        distance_space="final_probe_points",
    )
    probe_surface_points = clean_points[probe_surface_indices]
    probe_normals = clean_probe_normals[probe_surface_indices]
    probe_points = candidate_probe_points[probe_surface_indices]
    probe_surface_points, probe_points, probe_normals, probe_visibility_side_stats = _resolve_probe_surface_side_from_red_points(
        probe_points=probe_points,
        probe_normals=probe_normals,
        views=filtered_views,
        object_extent=local_extent,
        args=args,
        offset_distance=offset_distance,
    )

    _save_point_cloud(
        output_root / "surface_fused_clean.ply",
        clean_points,
        clean_normals,
        np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float32), (clean_points.shape[0], 1)),
    )
    _save_point_cloud(
        output_root / "probe_surface_samples.ply",
        probe_surface_points,
        probe_normals,
        np.tile(np.array([[0.95, 0.62, 0.22]], dtype=np.float32), (probe_surface_points.shape[0], 1)),
    )
    _save_point_cloud(
        output_root / "probe_offset_points.ply",
        probe_points,
        probe_normals,
        np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1)),
    )

    normal_vis_idx = np.arange(probe_points.shape[0], dtype=np.int64)
    if args.max_probe_normals_vis > 0 and normal_vis_idx.shape[0] > args.max_probe_normals_vis:
        normal_vis_idx = np.sort(rng.choice(normal_vis_idx, size=args.max_probe_normals_vis, replace=False))
    normal_line_starts = probe_points[normal_vis_idx]
    normal_line_ends = probe_points[normal_vis_idx] + probe_normals[normal_vis_idx] * offset_distance
    _save_lineset(
        output_root / "probe_normals_lineset.ply",
        starts=normal_line_starts,
        ends=normal_line_ends,
        color=(0.12, 0.47, 0.71),
    )
    _save_line_segments_obj(
        output_root / "probe_normals_lineset.obj",
        starts=normal_line_starts.astype(np.float32),
        ends=normal_line_ends.astype(np.float32),
        header_comments=["probe normal debug lines"],
    )

    scene_points = scene_gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    scene_colors = scene_gaussians.get_base_color.detach().cpu().numpy().astype(np.float32)
    scene_opacity = scene_gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float32)
    local_scene_radius = float(args.local_scene_radius_factor) * shadow_radius
    local_scene_keep = _filter_points_to_shadow_region(
        points=scene_points,
        shadow_center=shadow_center,
        shadow_radius=local_scene_radius,
        object_bbox_min=np.asarray(object_bbox["min"], dtype=np.float32),
        object_bbox_max=np.asarray(object_bbox["max"], dtype=np.float32),
        object_bbox_margin=object_bbox_margin,
    )
    if args.scene_opacity_thresh > 0.0:
        local_scene_keep &= scene_opacity >= float(args.scene_opacity_thresh)
    local_scene_points = scene_points[local_scene_keep]
    local_scene_colors = scene_colors[local_scene_keep]
    if args.local_scene_voxel_size > 0.0 and local_scene_points.shape[0] > 0:
        local_scene_points, local_scene_colors = _voxel_downsample_points_colors(
            local_scene_points,
            local_scene_colors,
            float(args.local_scene_voxel_size),
        )
    local_scene_normals = np.zeros_like(local_scene_points, dtype=np.float32)
    local_scene_debug = _save_point_cloud_limited(
        output_root / "scene_local_gaussians.ply",
        local_scene_points,
        local_scene_normals,
        local_scene_colors,
        max_points=int(args.max_scene_points_save),
        rng=rng,
    )

    _save_scene_and_probe_combo(
        output_root / "scene_local_plus_object_plus_sops.ply",
        scene_points=_sample_rows(local_scene_points, int(args.max_scene_points_in_combo), rng),
        object_points=_sample_rows(object_points, int(args.max_object_points_in_combo), rng),
        probe_surface_points=probe_surface_points,
        probe_points=probe_points,
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
        object_bbox_min=np.asarray(object_bbox["min"], dtype=np.float32),
        object_bbox_max=np.asarray(object_bbox["max"], dtype=np.float32),
        object_bbox_center=np.asarray(object_bbox["center"], dtype=np.float32),
        shadow_center=shadow_center.astype(np.float32),
        object_radius=np.array([obj_radius], dtype=np.float32),
        shadow_radius=np.array([shadow_radius], dtype=np.float32),
        offset_distance=np.array([offset_distance], dtype=np.float32),
    )

    summary = {
        "format": "irgs_shadow_region_sop_v1",
        "scene_checkpoint": str(Path(args.scene_checkpoint).resolve()),
        "scene_iteration": int(scene_iter),
        "scene_format": scene_format,
        "object_checkpoint": str(Path(args.object_checkpoint).resolve()),
        "object_iteration": int(object_iter),
        "object_format": object_format,
        "output_root": str(output_root.resolve()),
        "transform": {
            "translation": list(map(float, args.translation)),
            "rotation_deg": list(map(float, args.rotation_deg)),
            "rotation_order": args.rotation_order,
            "scale": float(args.scale),
            "post_scale": float(args.post_scale),
            "post_scale_center": args.post_scale_center,
            "post_scale_center_xyz": post_scale_center.detach().cpu().tolist(),
            "matrix": transform.detach().cpu().numpy().astype(float).tolist(),
        },
        "object_bbox": {
            "min": np.asarray(object_bbox["min"], dtype=np.float32).astype(float).tolist(),
            "max": np.asarray(object_bbox["max"], dtype=np.float32).astype(float).tolist(),
            "center": np.asarray(object_bbox["center"], dtype=np.float32).astype(float).tolist(),
            "mean": np.asarray(object_bbox["mean"], dtype=np.float32).astype(float).tolist(),
            "side_lengths": np.asarray(object_bbox["side_lengths"], dtype=np.float32).astype(float).tolist(),
            "diagonal": float(object_bbox["diagonal"]),
            "radius": float(object_bbox["radius"]),
            "max_side": float(object_bbox["max_side"]),
        },
        "shadow_region": {
            "center": shadow_center.astype(float).tolist(),
            "object_radius": float(obj_radius),
            "shadow_radius_factor": float(args.shadow_radius_factor),
            "shadow_radius": float(shadow_radius),
            "object_bbox_margin": float(object_bbox_margin),
        },
        "hemisphere": {
            "up_axis": up_axis.astype(float).tolist(),
            "distance": float(hemisphere_distance),
            "num_cameras": int(len(cameras)),
            "camera_fov_deg": float(camera_fov_deg),
            "camera_resolution": [int(args.camera_width), int(args.camera_height)],
            "min_elevation_deg": float(args.hemisphere_min_elevation_deg),
        },
        "local_scene": {
            "radius": float(local_scene_radius),
            "scene_opacity_thresh": float(args.scene_opacity_thresh),
            "voxel_size": float(args.local_scene_voxel_size),
            "num_raw_selected": int(np.sum(local_scene_keep)),
            "num_saved_points": int(local_scene_points.shape[0]),
            "export": local_scene_debug,
        },
        "candidate_points": int(candidate_points.shape[0]),
        "consistent_points": int(consistent_points.shape[0]),
        "num_surface_points": int(clean_points.shape[0]),
        "num_probes": int(probe_points.shape[0]),
        "local_extent": float(local_extent),
        "extent_mode": args.extent_mode,
        "probe_extent_reference": args.probe_extent_reference,
        "probe_selection": probe_selection_stats,
        "offset_scale": float(args.offset_scale),
        "offset_distance": float(offset_distance),
        "probe_min_distance": float(probe_min_distance),
        "view_filter_stats": {
            "raw_candidates": _summarize_array(np.asarray([row["raw_candidates"] for row in per_view_stats], dtype=np.float32)),
            "after_region_filter": _summarize_array(np.asarray([row["after_region_filter"] for row in per_view_stats], dtype=np.float32)),
            "after_cap": _summarize_array(np.asarray([row["after_cap"] for row in per_view_stats], dtype=np.float32)),
            "per_view": per_view_stats,
        },
        "cross_view_consistency": consistency_stats,
        "cleaning": clean_stats,
        "normal_orientation": normal_orientation_stats,
        "support_surface": support_surface_stats,
        "support_plane_resample": support_resample_stats,
        "probe_visibility_side_correction": probe_visibility_side_stats,
        "debug_clouds": {
            "candidate": candidate_debug,
            "consistent": consistent_debug,
        },
        "args": dict(vars(args)),
    }
    _write_json(output_root / "shadow_region_summary.json", summary)

    print(f"[shadow-sop] Object radius: {obj_radius:.6f}")
    print(f"[shadow-sop] Shadow radius: {shadow_radius:.6f}")
    print(f"[shadow-sop] Hemisphere cameras: {len(cameras)}")
    print(f"[shadow-sop] Candidate points: {candidate_points.shape[0]}")
    print(f"[shadow-sop] Consistent points: {consistent_points.shape[0]}")
    print(f"[shadow-sop] Support-surface points: {clean_points.shape[0]}")
    print(f"[shadow-sop] SOP probes: {probe_points.shape[0]}")
    print(f"[shadow-sop] Offset distance: {offset_distance:.6f}")
    print(f"[shadow-sop] Probe min distance: {probe_min_distance:.6f}")
    print(f"[shadow-sop] Output root: {output_root}")
    return summary


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Build a local shadow-region SOP initialization around an inserted object by "
            "reusing the lego->Ignatius transform, sampling hemisphere cameras, and "
            "exporting local scene/probe visualizations."
        )
    )
    parser.add_argument(
        "--scene_checkpoint",
        default=str(REPO_DIR / "outputs/tnt/Ignatius/chkpnt_best.pth"),
    )
    parser.add_argument(
        "--object_checkpoint",
        default=str(REPO_DIR / "outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_DIR / "outputs/composite/lego_shadow_region_sop"),
    )

    parser.add_argument("--translation", nargs=3, type=float, default=[-2.37, 1.451, -0.45])
    parser.add_argument("--rotation_deg", nargs=3, type=float, default=[83.57, 190.7, 0.254])
    parser.add_argument("--rotation_order", default="XYZ")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--post_scale", type=float, default=0.632)
    parser.add_argument("--post_scale_center", choices=["bbox", "mean", "origin"], default="bbox")

    parser.add_argument("--shadow_radius_factor", type=float, default=6.0)
    parser.add_argument("--shadow_center_offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--object_bbox_margin", type=float, default=0.0)
    parser.add_argument("--object_bbox_margin_factor", type=float, default=0.05)

    parser.add_argument("--up_axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="y")
    parser.add_argument("--num_hemisphere_cameras", type=int, default=96)
    parser.add_argument("--hemisphere_distance", type=float, default=0.0)
    parser.add_argument("--hemisphere_distance_factor", type=float, default=1.75)
    parser.add_argument("--hemisphere_min_elevation_deg", type=float, default=10.0)
    parser.add_argument("--hemisphere_vis_points", type=int, default=4096)
    parser.add_argument("--camera_width", type=int, default=768)
    parser.add_argument("--camera_height", type=int, default=768)
    parser.add_argument("--camera_fov_deg", type=float, default=0.0)
    parser.add_argument("--camera_fov_margin", type=float, default=1.10)
    parser.add_argument("--camera_fov_min_deg", type=float, default=35.0)
    parser.add_argument("--camera_fov_max_deg", type=float, default=95.0)
    parser.add_argument("--camera_frustum_length_factor", type=float, default=0.18)

    parser.add_argument("--depth_ratio", type=float, default=0.0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--weight_thresh", default=0.35, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument(
        "--object_filter_mode",
        choices=["weight_only", "mask_only", "weight_or_mask", "weight_and_mask"],
        default="weight_only",
    )
    parser.add_argument("--mask_erosion_radius", default=0, type=int)
    parser.add_argument("--normal_min_norm", default=0.5, type=float)
    parser.add_argument("--max_points_per_view", default=60000, type=int)
    parser.add_argument("--max_points_after_region_filter", default=30000, type=int)

    parser.add_argument("--consistency_min_views", default=2, type=int)
    parser.add_argument("--consistency_max_views", default=12, type=int)
    parser.add_argument("--consistency_view_stride", default=1, type=int)
    parser.add_argument("--consistency_chunk_size", default=65536, type=int)
    parser.add_argument("--depth_tolerance", default=0.0, type=float)
    parser.add_argument("--depth_tolerance_factor", default=0.005, type=float)
    parser.add_argument("--depth_relative_tolerance", default=0.01, type=float)
    parser.add_argument("--normal_angle_thresh", default=35.0, type=float)
    parser.add_argument("--use_abs_normal_dot", action="store_true")

    parser.add_argument("--fusion_voxel_factor", default=0.01, type=float)
    parser.add_argument("--fusion_voxel_size", default=0.0, type=float)
    parser.add_argument("--denoise_mode", choices=["none", "radius"], default="radius")
    parser.add_argument("--radius_denoise_radius", default=0.0, type=float)
    parser.add_argument("--radius_denoise_radius_factor", default=0.015, type=float)
    parser.add_argument("--radius_denoise_min_neighbors", default=6, type=int)

    parser.add_argument("--extent_mode", choices=["bbox_diagonal", "max_side"], default="bbox_diagonal")
    parser.add_argument("--target_num_probes", default=10000, type=int)
    parser.add_argument("--probe_extent_reference", choices=[
        "object_bbox_diagonal",
        "object_max_side",
        "object_radius",
        "shadow_radius",
        "local_surface_bbox_diagonal",
        "local_surface_max_side",
    ], default="object_bbox_diagonal")
    parser.add_argument("--probe_min_distance_factor", default=0.0, type=float)
    parser.add_argument("--probe_min_distance", default=0.0, type=float)
    parser.add_argument("--offset_scale", default=0.005, type=float)
    parser.add_argument("--offset_distance", default=0.0, type=float)
    parser.add_argument("--normal_orientation_min_views", default=1, type=int)
    parser.add_argument("--normal_orientation_chunk_size", default=65536, type=int)
    parser.add_argument("--support_min_normal_up_dot", default=0.35, type=float)
    parser.add_argument("--support_plane_height_percentile", default=95.0, type=float)
    parser.add_argument("--support_above_object_bottom_margin", default=0.0, type=float)
    parser.add_argument("--support_above_object_bottom_margin_factor", default=0.08, type=float)
    parser.add_argument("--support_below_object_bottom_margin", default=0.0, type=float)
    parser.add_argument("--support_below_object_bottom_margin_factor", default=0.45, type=float)
    parser.add_argument("--support_plane_band", default=0.0, type=float)
    parser.add_argument("--support_plane_band_factor", default=0.04, type=float)
    parser.set_defaults(snap_support_points_to_plane=True)
    parser.add_argument("--disable_support_plane_snap", action="store_false", dest="snap_support_points_to_plane")
    parser.set_defaults(enable_support_plane_resample=True)
    parser.add_argument("--disable_support_plane_resample", action="store_false", dest="enable_support_plane_resample")
    parser.add_argument("--support_resample_oversample_factor", default=1.6, type=float)
    parser.add_argument("--support_resample_occupancy_radius_factor", default=1.35, type=float)
    parser.add_argument("--support_resample_bbox_pad", default=1.0, type=float)
    parser.add_argument("--support_resample_max_rounds", default=5, type=int)
    parser.add_argument("--support_resample_spacing_shrink", default=0.72, type=float)
    parser.add_argument("--support_resample_radius_expand", default=1.25, type=float)

    parser.add_argument("--local_scene_radius_factor", default=1.05, type=float)
    parser.add_argument("--scene_opacity_thresh", default=0.02, type=float)
    parser.add_argument("--local_scene_voxel_size", default=0.0, type=float)
    parser.add_argument("--max_scene_points_save", default=250000, type=int)
    parser.add_argument("--max_scene_points_in_combo", default=120000, type=int)
    parser.add_argument("--max_object_points_in_combo", default=40000, type=int)

    parser.add_argument("--max_probe_normals_vis", default=600, type=int)
    parser.add_argument("--max_debug_points", default=500000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    initialize_shadow_region_sop(args)


if __name__ == "__main__":
    main()
