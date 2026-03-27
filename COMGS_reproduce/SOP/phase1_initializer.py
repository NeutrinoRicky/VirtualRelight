from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

try:
    import open3d as o3d
except ImportError:
    o3d = None

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_multitarget
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.point_utils import depths_to_points


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _bbox_extent(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    side_lengths = maxs - mins
    return side_lengths, float(np.linalg.norm(side_lengths)), float(side_lengths.max())


def _compute_object_extent(points: np.ndarray, mode: str) -> float:
    _, diagonal, max_side = _bbox_extent(points)
    if mode == "bbox_diagonal":
        return diagonal
    if mode == "max_side":
        return max_side
    raise ValueError(f"Unknown extent mode: {mode}")


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _pick_indices(num_items: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if max_items <= 0 or num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


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
    attributes = np.concatenate(
        [
            points.astype(np.float32),
            normals.astype(np.float32),
            colors_u8,
        ],
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


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


def _merge_triangle_meshes(meshes) -> "o3d.geometry.TriangleMesh":
    if o3d is None:
        raise RuntimeError("open3d is required for mesh export")

    vertices = []
    triangles = []
    colors = []
    vertex_offset = 0

    for mesh in meshes:
        if mesh is None or len(mesh.vertices) == 0:
            continue
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh_triangles = np.asarray(mesh.triangles, dtype=np.int32)
        if mesh_triangles.size == 0:
            continue
        if mesh.has_vertex_colors():
            mesh_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        else:
            mesh_colors = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (mesh_vertices.shape[0], 1))

        vertices.append(mesh_vertices)
        triangles.append(mesh_triangles + vertex_offset)
        colors.append(mesh_colors)
        vertex_offset += mesh_vertices.shape[0]

    merged = o3d.geometry.TriangleMesh()
    if not vertices:
        return merged

    merged.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices, axis=0))
    merged.triangles = o3d.utility.Vector3iVector(np.concatenate(triangles, axis=0))
    merged.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
    merged.compute_vertex_normals()
    return merged


@torch.no_grad()
def _extract_mesh_with_probes(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    dataset,
    output_root: Path,
    probe_points: np.ndarray,
    offset_distance: float,
    args,
) -> Dict[str, object]:
    if o3d is None:
        return {"enabled": False, "reason": "open3d_unavailable"}

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    extractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    prev_sh_degree = gaussians.active_sh_degree

    try:
        extractor.gaussians.active_sh_degree = 0
        extractor.reconstruction(scene.getTrainCameras())
        depth_trunc = (extractor.radius * 2.0) if args.mesh_depth_trunc < 0 else args.mesh_depth_trunc
        voxel_size = (depth_trunc / args.mesh_res) if args.mesh_voxel_size < 0 else args.mesh_voxel_size
        sdf_trunc = (5.0 * voxel_size) if args.mesh_sdf_trunc < 0 else args.mesh_sdf_trunc
        mesh = extractor.extract_mesh_bounded(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
            mask_backgrond=True,
        )
    finally:
        extractor.gaussians.active_sh_degree = prev_sh_degree

    raw_mesh_path = output_root / "mesh_fuse_colored.ply"
    largest_mesh_path = output_root / "mesh_fuse_colored_largest.ply"
    mesh_with_probes_path = output_root / "mesh_largest_with_probes.ply"
    o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh)

    mesh_largest = post_process_mesh(mesh, cluster_to_keep=max(1, int(args.mesh_num_cluster)))
    o3d.io.write_triangle_mesh(str(largest_mesh_path), mesh_largest)

    if args.mesh_max_probe_spheres > 0 and probe_points.shape[0] > args.mesh_max_probe_spheres:
        probe_indices = np.linspace(0, probe_points.shape[0] - 1, args.mesh_max_probe_spheres, dtype=np.int64)
        probe_points_mesh = probe_points[probe_indices]
    else:
        probe_points_mesh = probe_points

    probe_radius = args.mesh_probe_radius if args.mesh_probe_radius > 0 else max(offset_distance * 0.12, voxel_size * 1.5)
    probe_meshes = [mesh_largest]
    for point in probe_points_mesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=probe_radius, resolution=max(4, int(args.mesh_probe_resolution)))
        sphere.translate(point.astype(np.float64))
        sphere.paint_uniform_color([0.85, 0.28, 0.25])
        probe_meshes.append(sphere)

    mesh_with_probes = _merge_triangle_meshes(probe_meshes)
    o3d.io.write_triangle_mesh(str(mesh_with_probes_path), mesh_with_probes)

    return {
        "enabled": True,
        "raw_mesh_path": str(raw_mesh_path),
        "largest_mesh_path": str(largest_mesh_path),
        "mesh_with_probes_path": str(mesh_with_probes_path),
        "depth_trunc": float(depth_trunc),
        "voxel_size": float(voxel_size),
        "sdf_trunc": float(sdf_trunc),
        "mesh_num_cluster": int(args.mesh_num_cluster),
        "probe_radius": float(probe_radius),
        "probe_spheres": int(probe_points_mesh.shape[0]),
        "raw_mesh_vertices": int(len(mesh.vertices)),
        "largest_mesh_vertices": int(len(mesh_largest.vertices)),
    }


def _voxel_downsample_numpy(points: np.ndarray, normals: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] == 0:
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


def _radius_filter_open3d(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or radius <= 0.0 or min_neighbors <= 1 or o3d is None:
        return points, normals

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    filtered, indices = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    if len(indices) == 0:
        return points, normals
    filtered_points = np.asarray(filtered.points, dtype=np.float32)
    filtered_normals = _normalize_np(np.asarray(filtered.normals, dtype=np.float32))
    return filtered_points, filtered_normals


def _merge_batches(
    fused_points: Optional[np.ndarray],
    fused_normals: Optional[np.ndarray],
    pending_points: List[np.ndarray],
    pending_normals: List[np.ndarray],
    voxel_size: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if fused_points is None and len(pending_points) == 0:
        return None, None

    merged_points: List[np.ndarray] = []
    merged_normals: List[np.ndarray] = []
    if fused_points is not None:
        merged_points.append(fused_points)
        merged_normals.append(fused_normals)
    if pending_points:
        merged_points.extend(pending_points)
        merged_normals.extend(pending_normals)

    points = np.concatenate(merged_points, axis=0).astype(np.float32)
    normals = _normalize_np(np.concatenate(merged_normals, axis=0).astype(np.float32))
    points, normals = _voxel_downsample_numpy(points, normals, voxel_size)
    return points, normals


def _choose_cameras(scene: Scene, camera_set: str) -> List:
    if camera_set == "train":
        return list(scene.getTrainCameras())
    if camera_set == "test":
        return list(scene.getTestCameras())
    if camera_set == "all":
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    raise ValueError(f"Unknown camera set: {camera_set}")


def _erode_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
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
    normals_world: torch.Tensor,
    gt_alpha_mask: Optional[torch.Tensor],
    weight_thresh: float,
    mask_thresh: float,
    object_filter_mode: str,
    mask_erosion_radius: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    depth_valid = torch.isfinite(depth[0]) & (depth[0] > 0.0)
    weight_valid = torch.isfinite(weight[0]) & (weight[0] > weight_thresh)
    normal_valid = torch.isfinite(normals_world).all(dim=-1)

    mask_available = gt_alpha_mask is not None
    if mask_available:
        mask_valid = gt_alpha_mask[0] > mask_thresh
        core_mask = _erode_binary_mask(mask_valid, mask_erosion_radius)
    else:
        mask_valid = torch.zeros_like(depth_valid, dtype=torch.bool)
        core_mask = torch.ones_like(depth_valid, dtype=torch.bool)

    effective_filter_mode = object_filter_mode
    if object_filter_mode == "weight_only":
        object_valid = weight_valid
    elif object_filter_mode == "mask_only":
        if mask_available:
            object_valid = mask_valid
        else:
            object_valid = weight_valid
            effective_filter_mode = "weight_only_fallback_no_mask"
    elif object_filter_mode == "weight_or_mask":
        object_valid = (weight_valid | mask_valid) if mask_available else weight_valid
        if not mask_available:
            effective_filter_mode = "weight_only_fallback_no_mask"
    elif object_filter_mode == "weight_and_mask":
        object_valid = (weight_valid & mask_valid) if mask_available else weight_valid
        if not mask_available:
            effective_filter_mode = "weight_only_fallback_no_mask"
    else:
        raise ValueError(f"Unknown object_filter_mode: {object_filter_mode}")

    valid_before_core = depth_valid & normal_valid & object_valid
    valid = valid_before_core & core_mask
    stats = {
        "depth_pixels": float(depth_valid.sum().item()),
        "weight_pixels": float(weight_valid.sum().item()),
        "mask_pixels": float(mask_valid.sum().item()) if mask_available else 0.0,
        "core_mask_pixels": float(core_mask.sum().item()) if mask_available else float(depth_valid.numel()),
        "mask_available": bool(mask_available),
        "mask_erosion_radius": int(mask_erosion_radius),
        "effective_filter_mode": effective_filter_mode,
        "valid_pixels_before_core": float(valid_before_core.sum().item()),
    }
    return valid, stats


@torch.no_grad()
def _extract_view_surface_points(
    viewpoint_cam,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    weight_thresh: float,
    mask_thresh: float,
    max_points_per_view: int,
    object_filter_mode: str,
    mask_erosion_radius: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
    depth = render_pkg["depth_unbiased"]
    weight = render_pkg["weight"]
    normal = render_pkg["normal"]

    points_world = depths_to_points(viewpoint_cam, depth).reshape(depth.shape[1], depth.shape[2], 3)
    normals_world = F.normalize(normal.permute(1, 2, 0), dim=-1, eps=1e-6)
    valid, filter_stats = _build_object_valid_mask(
        depth=depth,
        weight=weight,
        normals_world=normals_world,
        gt_alpha_mask=getattr(viewpoint_cam, "gt_alpha_mask", None),
        weight_thresh=weight_thresh,
        mask_thresh=mask_thresh,
        object_filter_mode=object_filter_mode,
        mask_erosion_radius=mask_erosion_radius,
    )

    cam_center = viewpoint_cam.camera_center.view(1, 1, 3)
    to_camera = F.normalize(cam_center - points_world, dim=-1, eps=1e-6)
    flip = (normals_world * to_camera).sum(dim=-1, keepdim=True) < 0.0
    normals_world = torch.where(flip, -normals_world, normals_world)

    valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    stats = {
        **filter_stats,
        "valid_pixels": float(valid_indices.numel()),
        "total_pixels": float(valid.numel()),
        "selected_points": 0.0,
    }
    if valid_indices.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), stats

    if max_points_per_view > 0 and valid_indices.numel() > max_points_per_view:
        perm = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:max_points_per_view]
        valid_indices = valid_indices[perm]

    points_flat = points_world.reshape(-1, 3)[valid_indices]
    normals_flat = normals_world.reshape(-1, 3)[valid_indices]

    stats["selected_points"] = float(valid_indices.numel())
    return _to_numpy(points_flat).astype(np.float32), _to_numpy(normals_flat).astype(np.float32), stats


@torch.no_grad()
def _fuse_surface_cloud(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    all_cameras = _choose_cameras(scene, args.camera_set)
    if len(all_cameras) == 0:
        raise RuntimeError(f"No cameras available for camera_set={args.camera_set}")

    selected_cameras = all_cameras[:: max(1, args.view_stride)]
    if args.max_views > 0:
        selected_cameras = selected_cameras[: args.max_views]
    if len(selected_cameras) == 0:
        raise RuntimeError("Camera selection is empty after applying view_stride/max_views.")

    gaussian_points = _to_numpy(gaussians.get_xyz).astype(np.float32)
    gaussian_colors = _to_numpy(gaussians.get_albedo).astype(np.float32)
    gaussian_extent = _compute_object_extent(gaussian_points, args.extent_mode)
    fusion_voxel_size = args.fusion_voxel_size if args.fusion_voxel_size > 0.0 else args.fusion_voxel_factor * gaussian_extent
    outlier_radius = args.outlier_radius if args.outlier_radius > 0.0 else args.outlier_radius_factor * fusion_voxel_size

    fused_points: Optional[np.ndarray] = None
    fused_normals: Optional[np.ndarray] = None
    pending_points: List[np.ndarray] = []
    pending_normals: List[np.ndarray] = []
    pending_count = 0
    view_stats: List[Dict[str, object]] = []

    print(f"[SOP-Phase1] Using {len(selected_cameras)} cameras from set={args.camera_set}.")
    print(f"[SOP-Phase1] Fusion voxel size = {fusion_voxel_size:.6f}, outlier radius = {outlier_radius:.6f}.")

    for camera_idx, viewpoint_cam in enumerate(selected_cameras):
        points_np, normals_np, stats = _extract_view_surface_points(
            viewpoint_cam=viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            weight_thresh=args.weight_thresh,
            mask_thresh=args.mask_thresh,
            max_points_per_view=args.max_points_per_view,
            object_filter_mode=args.object_filter_mode,
            mask_erosion_radius=args.mask_erosion_radius,
        )
        stats["image_name"] = viewpoint_cam.image_name
        stats["camera_index"] = camera_idx
        view_stats.append(stats)

        if points_np.shape[0] == 0:
            continue

        pending_points.append(points_np)
        pending_normals.append(normals_np)
        pending_count += points_np.shape[0]

        if pending_count >= args.fusion_buffer_points:
            fused_points, fused_normals = _merge_batches(
                fused_points=fused_points,
                fused_normals=fused_normals,
                pending_points=pending_points,
                pending_normals=pending_normals,
                voxel_size=fusion_voxel_size,
            )
            pending_points.clear()
            pending_normals.clear()
            pending_count = 0

    fused_points, fused_normals = _merge_batches(
        fused_points=fused_points,
        fused_normals=fused_normals,
        pending_points=pending_points,
        pending_normals=pending_normals,
        voxel_size=fusion_voxel_size,
    )

    if fused_points is None or fused_points.shape[0] == 0:
        raise RuntimeError("Surface fusion produced an empty point cloud. Lower thresholds or check stage1 geometry.")

    preclean_points = fused_points.copy()
    preclean_normals = fused_normals.copy()
    fused_points, fused_normals = _radius_filter_open3d(
        fused_points,
        fused_normals,
        radius=outlier_radius,
        min_neighbors=args.outlier_min_neighbors,
    )
    fused_normals = _normalize_np(fused_normals.astype(np.float32))

    fusion_stats = {
        "camera_names": [cam.image_name for cam in selected_cameras],
        "num_selected_cameras": len(selected_cameras),
        "gaussian_extent": gaussian_extent,
        "fusion_voxel_size": fusion_voxel_size,
        "outlier_radius": outlier_radius,
        "preclean_points": int(preclean_points.shape[0]),
        "clean_points": int(fused_points.shape[0]),
        "object_filter_mode": args.object_filter_mode,
        "cameras_with_mask": int(sum(1 for s in view_stats if s.get("mask_available", False))),
        "view_stats": view_stats,
    }
    return preclean_points, preclean_normals, fused_points, fused_normals, fusion_stats


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


def _chunked_nearest_neighbors(
    queries: np.ndarray,
    references: np.ndarray,
    device: torch.device,
    query_chunk_size: int = 512,
    ref_chunk_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    if references.shape[0] == 0:
        raise ValueError("Reference set is empty")

    queries_t = torch.from_numpy(queries.astype(np.float32)).to(device)
    refs_t = torch.from_numpy(references.astype(np.float32)).to(device)
    distances_all: List[np.ndarray] = []
    indices_all: List[np.ndarray] = []

    for q_start in range(0, queries_t.shape[0], query_chunk_size):
        q_end = min(q_start + query_chunk_size, queries_t.shape[0])
        q = queries_t[q_start:q_end]
        best_dist = torch.full((q.shape[0],), float("inf"), device=device)
        best_idx = torch.zeros((q.shape[0],), dtype=torch.long, device=device)

        for r_start in range(0, refs_t.shape[0], ref_chunk_size):
            r_end = min(r_start + ref_chunk_size, refs_t.shape[0])
            refs = refs_t[r_start:r_end]
            dists = torch.cdist(q, refs)
            local_dist, local_idx = torch.min(dists, dim=1)
            update = local_dist < best_dist
            best_dist = torch.where(update, local_dist, best_dist)
            best_idx = torch.where(update, local_idx + r_start, best_idx)

        distances_all.append(best_dist.detach().cpu().numpy())
        indices_all.append(best_idx.detach().cpu().numpy())

    return np.concatenate(distances_all, axis=0), np.concatenate(indices_all, axis=0)


def _summarize_array(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _build_probe_status(
    signed_surface_offset: np.ndarray,
    normal_alignment: np.ndarray,
    surface_distances: np.ndarray,
    offset_distance: float,
    too_close_factor: float,
    too_far_factor: float,
    backface_alignment_thresh: float,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    too_close_threshold = too_close_factor * offset_distance
    too_far_threshold = too_far_factor * offset_distance

    inside = signed_surface_offset < 0.0
    too_close = surface_distances < too_close_threshold
    too_far = surface_distances > too_far_threshold
    backface = normal_alignment < backface_alignment_thresh

    status = np.full(surface_distances.shape[0], "ok", dtype="<U16")
    status[too_close] = "too_close"
    status[too_far] = "too_far"
    status[backface] = "backface"
    status[inside] = "inside"

    counts = {
        "ok": int(np.sum(status == "ok")),
        "too_close": int(np.sum(status == "too_close")),
        "too_far": int(np.sum(status == "too_far")),
        "backface": int(np.sum(status == "backface")),
        "inside": int(np.sum(status == "inside")),
    }
    thresholds = {
        "too_close_threshold": float(too_close_threshold),
        "too_far_threshold": float(too_far_threshold),
        "backface_alignment_thresh": float(backface_alignment_thresh),
    }
    return status, counts, thresholds


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _plot_object_and_probes(
    gaussian_points: np.ndarray,
    probe_points: np.ndarray,
    save_path: Path,
    rng: np.random.Generator,
    max_gaussians_vis: int,
) -> None:
    gaussian_idx = _pick_indices(gaussian_points.shape[0], max_gaussians_vis, rng)
    vis_gaussians = gaussian_points[gaussian_idx]
    stacked = np.concatenate([vis_gaussians, probe_points], axis=0)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vis_gaussians[:, 0], vis_gaussians[:, 1], vis_gaussians[:, 2], s=1.5, c="#999999", alpha=0.25)
    ax.scatter(probe_points[:, 0], probe_points[:, 1], probe_points[:, 2], s=8.0, c="#d94841", alpha=0.85)
    _set_axes_equal(ax, stacked)
    ax.set_title("Object Gaussians + SOP probes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_context_and_probes(
    context_points: np.ndarray,
    probe_points: np.ndarray,
    save_path: Path,
    rng: np.random.Generator,
    max_context_vis: int,
    title: str,
    context_color: str = "#999999",
    context_alpha: float = 0.25,
) -> None:
    context_idx = _pick_indices(context_points.shape[0], max_context_vis, rng)
    vis_context = context_points[context_idx]
    stacked = np.concatenate([vis_context, probe_points], axis=0)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vis_context[:, 0], vis_context[:, 1], vis_context[:, 2], s=1.5, c=context_color, alpha=context_alpha)
    ax.scatter(probe_points[:, 0], probe_points[:, 1], probe_points[:, 2], s=8.0, c="#d94841", alpha=0.85)
    _set_axes_equal(ax, stacked)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_probe_normals(
    probe_points: np.ndarray,
    probe_normals: np.ndarray,
    offset_distance: float,
    save_path: Path,
    rng: np.random.Generator,
    max_probe_normals_vis: int,
) -> None:
    probe_idx = _pick_indices(probe_points.shape[0], max_probe_normals_vis, rng)
    vis_points = probe_points[probe_idx]
    vis_normals = probe_normals[probe_idx]
    endpoints = vis_points + vis_normals * offset_distance
    stacked = np.concatenate([vis_points, endpoints], axis=0)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], s=8.0, c="#d94841", alpha=0.8)
    ax.quiver(
        vis_points[:, 0],
        vis_points[:, 1],
        vis_points[:, 2],
        vis_normals[:, 0],
        vis_normals[:, 1],
        vis_normals[:, 2],
        length=offset_distance,
        normalize=True,
        color="#1f78b4",
        linewidth=0.75,
    )
    _set_axes_equal(ax, stacked)
    ax.set_title("Probe normal directions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_distance_histogram(
    surface_distances: np.ndarray,
    offset_distance: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(surface_distances, bins=40, color="#4c78a8", alpha=0.9, edgecolor="white")
    ax.axvline(offset_distance, color="#d94841", linestyle="--", linewidth=2.0, label="target offset")
    ax.axvline(float(np.mean(surface_distances)), color="#3c8d2f", linestyle="-.", linewidth=2.0, label="mean distance")
    ax.set_title("Probe to nearest surface distance")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_probe_quality(
    probe_points: np.ndarray,
    statuses: np.ndarray,
    counts: Dict[str, int],
    save_path: Path,
) -> None:
    status_colors = {
        "ok": "#3c8d2f",
        "too_close": "#e17c05",
        "too_far": "#d94841",
        "backface": "#1f78b4",
        "inside": "#7f3b08",
    }

    fig = plt.figure(figsize=(12, 6.5))
    ax_scatter = fig.add_subplot(121, projection="3d")
    for key in ["ok", "too_close", "too_far", "backface", "inside"]:
        mask = statuses == key
        if not np.any(mask):
            continue
        pts = probe_points[mask]
        ax_scatter.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8.0, c=status_colors[key], alpha=0.85, label=f"{key} ({counts[key]})")
    _set_axes_equal(ax_scatter, probe_points)
    ax_scatter.set_title("Probe quality status")
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.set_zlabel("Z")
    ax_scatter.legend(loc="upper right", fontsize=8)

    ax_bar = fig.add_subplot(122)
    keys = ["ok", "too_close", "too_far", "backface", "inside"]
    values = [counts[key] for key in keys]
    ax_bar.bar(keys, values, color=[status_colors[key] for key in keys])
    ax_bar.set_title("Probe status counts")
    ax_bar.set_ylabel("Count")
    ax_bar.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _write_probe_csv(
    csv_path: Path,
    probe_points: np.ndarray,
    probe_normals: np.ndarray,
    nearest_surface_distance: np.ndarray,
    nearest_gaussian_distance: np.ndarray,
    signed_surface_offset: np.ndarray,
    normal_alignment: np.ndarray,
    statuses: np.ndarray,
) -> None:
    fieldnames = [
        "probe_id",
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "nearest_surface_distance",
        "nearest_gaussian_distance",
        "signed_surface_offset",
        "normal_alignment",
        "status",
    ]
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(probe_points.shape[0]):
            writer.writerow(
                {
                    "probe_id": idx,
                    "x": float(probe_points[idx, 0]),
                    "y": float(probe_points[idx, 1]),
                    "z": float(probe_points[idx, 2]),
                    "nx": float(probe_normals[idx, 0]),
                    "ny": float(probe_normals[idx, 1]),
                    "nz": float(probe_normals[idx, 2]),
                    "nearest_surface_distance": float(nearest_surface_distance[idx]),
                    "nearest_gaussian_distance": float(nearest_gaussian_distance[idx]),
                    "signed_surface_offset": float(signed_surface_offset[idx]),
                    "normal_alignment": float(normal_alignment[idx]),
                    "status": str(statuses[idx]),
                }
            )


def _load_gaussians_from_checkpoint(gaussians: GaussianModel, opt, checkpoint_path: str) -> Tuple[int, str]:
    payload = torch.load(checkpoint_path)
    if isinstance(payload, dict) and "gaussians" in payload:
        model_params = payload["gaussians"]
        iteration = int(payload.get("iteration", 0))
        label = payload.get("format", "checkpoint_dict")
    elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
        model_params = payload[0]
        iteration = int(payload[1])
        label = "stage1_checkpoint"
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    gaussians.training_setup(opt)
    gaussians.restore(model_params, opt)
    return iteration, label


def _to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@torch.no_grad()
def initialize_sop_phase1(args) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("SOP Phase 1 currently requires CUDA because it reuses the Gaussian renderer.")

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    loaded_from: Dict[str, object]
    checkpoint_path = getattr(args, "checkpoint", "")
    if checkpoint_path:
        scene = Scene(dataset, gaussians, shuffle=False)
        ckpt_iteration, ckpt_label = _load_gaussians_from_checkpoint(gaussians, opt, checkpoint_path)
        loaded_from = {
            "source": "checkpoint",
            "path": checkpoint_path,
            "iteration": ckpt_iteration,
            "format": ckpt_label,
        }
    else:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        loaded_from = {
            "source": "point_cloud",
            "iteration": int(scene.loaded_iter),
        }

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    output_dir = getattr(args, "output_dir", "")
    output_root = Path(output_dir) if output_dir else Path(dataset.model_path) / "SOP_phase1"
    _ensure_dir(output_root)
    debug_root = _ensure_dir(output_root / "debug")

    preclean_points, preclean_normals, surface_points, surface_normals, fusion_stats = _fuse_surface_cloud(
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        args=args,
    )

    object_extent = _compute_object_extent(surface_points, args.extent_mode)
    offset_distance = args.offset_scale * object_extent
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0. Check the fused surface cloud extent.")

    target_num_probes = min(int(args.target_num_probes), int(surface_points.shape[0]))
    if target_num_probes <= 0:
        raise RuntimeError("No surface points available for probe sampling.")

    device = torch.device("cuda")
    probe_surface_indices = _farthest_point_sampling(surface_points, target_num_probes, device)
    probe_surface_points = surface_points[probe_surface_indices]
    probe_normals = _normalize_np(surface_normals[probe_surface_indices])
    probe_points = probe_surface_points + probe_normals * offset_distance

    gaussian_points = _to_numpy(gaussians.get_xyz).astype(np.float32)
    gaussian_colors = _to_numpy(gaussians.get_albedo).astype(np.float32)
    _, surface_color_idx = _chunked_nearest_neighbors(
        queries=surface_points,
        references=gaussian_points,
        device=device,
    )
    surface_colors = gaussian_colors[surface_color_idx]
    _, preclean_color_idx = _chunked_nearest_neighbors(
        queries=preclean_points,
        references=gaussian_points,
        device=device,
    )
    preclean_colors = gaussian_colors[preclean_color_idx]
    nearest_surface_distance, nearest_surface_idx = _chunked_nearest_neighbors(
        queries=probe_points,
        references=surface_points,
        device=device,
    )
    nearest_surface_points = surface_points[nearest_surface_idx]
    nearest_surface_normals = surface_normals[nearest_surface_idx]
    signed_surface_offset = np.sum((probe_points - nearest_surface_points) * nearest_surface_normals, axis=1)
    normal_alignment = np.sum(probe_normals * nearest_surface_normals, axis=1)
    nearest_gaussian_distance, _ = _chunked_nearest_neighbors(
        queries=probe_points,
        references=gaussian_points,
        device=device,
    )
    statuses, status_counts, status_thresholds = _build_probe_status(
        signed_surface_offset=signed_surface_offset,
        normal_alignment=normal_alignment,
        surface_distances=nearest_surface_distance,
        offset_distance=offset_distance,
        too_close_factor=args.too_close_factor,
        too_far_factor=args.too_far_factor,
        backface_alignment_thresh=args.backface_alignment_thresh,
    )

    probe_surface_colors = surface_colors[probe_surface_indices]
    probe_offset_colors = np.clip(0.55 * probe_surface_colors + 0.45 * np.array([[1.0, 0.20, 0.20]], dtype=np.float32), 0.0, 1.0)

    rng = np.random.default_rng(args.seed)
    _save_point_cloud(output_root / "surface_fused_voxel.ply", preclean_points, preclean_normals, colors=preclean_colors)
    _save_point_cloud(output_root / "surface_fused_clean.ply", surface_points, surface_normals, colors=surface_colors)
    _save_point_cloud(output_root / "object_points_weight_mask_voxel.ply", preclean_points, preclean_normals, colors=preclean_colors)
    _save_point_cloud(output_root / "object_points_weight_mask_clean.ply", surface_points, surface_normals, colors=surface_colors)
    _save_point_cloud(
        output_root / "probe_surface_samples.ply",
        probe_surface_points,
        probe_normals,
        colors=probe_surface_colors,
    )
    _save_point_cloud(
        output_root / "probe_offset_points.ply",
        probe_points,
        probe_normals,
        colors=probe_offset_colors,
    )

    gaussian_vis_idx = _pick_indices(gaussian_points.shape[0], args.max_gaussians_vis, rng)
    gaussian_vis = gaussian_points[gaussian_vis_idx]
    gaussian_vis_colors = gaussian_colors[gaussian_vis_idx]
    combined_points = np.concatenate([gaussian_vis, probe_points], axis=0)
    combined_normals = np.zeros_like(combined_points, dtype=np.float32)
    combined_colors = np.concatenate(
        [
            gaussian_vis_colors,
            np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1)),
        ],
        axis=0,
    )
    _save_point_cloud(output_root / "gaussians_plus_probes.ply", combined_points, combined_normals, combined_colors)

    object_vis_idx = _pick_indices(surface_points.shape[0], args.max_gaussians_vis, rng)
    object_vis = surface_points[object_vis_idx]
    object_combined_points = np.concatenate([object_vis, probe_points], axis=0)
    object_combined_normals = np.concatenate([surface_normals[object_vis_idx], probe_normals], axis=0)
    object_combined_colors = np.concatenate(
        [
            surface_colors[object_vis_idx],
            np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1)),
        ],
        axis=0,
    )
    _save_point_cloud(output_root / "object_points_plus_probes.ply", object_combined_points, object_combined_normals, object_combined_colors)

    normal_vis_idx = _pick_indices(probe_points.shape[0], args.max_probe_normals_vis, rng)
    _save_lineset(
        output_root / "probe_normals_lineset.ply",
        starts=probe_points[normal_vis_idx],
        ends=probe_points[normal_vis_idx] + probe_normals[normal_vis_idx] * offset_distance,
        color=(0.12, 0.47, 0.71),
    )

    mesh_stats = {"enabled": False, "reason": "skip_mesh_export"}
    if not args.skip_mesh_export:
        mesh_stats = _extract_mesh_with_probes(
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            dataset=dataset,
            output_root=output_root,
            probe_points=probe_points,
            offset_distance=offset_distance,
            args=args,
        )

    _plot_object_and_probes(gaussian_points, probe_points, debug_root / "object_gaussians_probes.png", rng, args.max_gaussians_vis)
    _plot_context_and_probes(surface_points, probe_points, debug_root / "object_points_probes.png", rng, args.max_gaussians_vis, "Object-only points + SOP probes", context_color="#666666", context_alpha=0.35)
    _plot_probe_normals(probe_points, probe_normals, offset_distance, debug_root / "probe_normals.png", rng, args.max_probe_normals_vis)
    _plot_distance_histogram(nearest_surface_distance, offset_distance, debug_root / "probe_surface_distance_hist.png")
    _plot_probe_quality(probe_points, statuses, status_counts, debug_root / "probe_quality_status.png")

    _write_probe_csv(
        output_root / "probe_quality.csv",
        probe_points=probe_points,
        probe_normals=probe_normals,
        nearest_surface_distance=nearest_surface_distance,
        nearest_gaussian_distance=nearest_gaussian_distance,
        signed_surface_offset=signed_surface_offset,
        normal_alignment=normal_alignment,
        statuses=statuses,
    )

    np.savez(
        output_root / "probe_init_data.npz",
        surface_points=surface_points.astype(np.float32),
        surface_normals=surface_normals.astype(np.float32),
        probe_surface_points=probe_surface_points.astype(np.float32),
        probe_points=probe_points.astype(np.float32),
        probe_normals=probe_normals.astype(np.float32),
        object_points_voxel=preclean_points.astype(np.float32),
        object_points_clean=surface_points.astype(np.float32),
        surface_colors=surface_colors.astype(np.float32),
        preclean_colors=preclean_colors.astype(np.float32),
        probe_surface_colors=probe_surface_colors.astype(np.float32),
        probe_offset_colors=probe_offset_colors.astype(np.float32),
        nearest_surface_distance=nearest_surface_distance.astype(np.float32),
        nearest_gaussian_distance=nearest_gaussian_distance.astype(np.float32),
        signed_surface_offset=signed_surface_offset.astype(np.float32),
        normal_alignment=normal_alignment.astype(np.float32),
        status=statuses,
    )

    with open(output_root / "views_used.txt", "w") as view_file:
        for name in fusion_stats["camera_names"]:
            view_file.write(f"{name}\n")

    quality_summary = {
        "loaded_from": loaded_from,
        "output_root": str(output_root),
        "camera_set": args.camera_set,
        "object_filter_mode": args.object_filter_mode,
        "num_surface_points_voxel": int(preclean_points.shape[0]),
        "num_surface_points_clean": int(surface_points.shape[0]),
        "num_probes": int(probe_points.shape[0]),
        "extent_mode": args.extent_mode,
        "object_extent": float(object_extent),
        "offset_scale": float(args.offset_scale),
        "offset_distance": float(offset_distance),
        "surface_distance": _summarize_array(nearest_surface_distance),
        "gaussian_distance": _summarize_array(nearest_gaussian_distance),
        "signed_surface_offset": _summarize_array(signed_surface_offset),
        "normal_alignment": _summarize_array(normal_alignment),
        "outward_ratio": float(np.mean(signed_surface_offset > 0.0)),
        "status_thresholds": status_thresholds,
        "status_counts": status_counts,
        "fusion": fusion_stats,
        "mesh": mesh_stats,
        "args": dict(vars(args)),
    }
    with open(output_root / "probe_quality_summary.json", "w") as summary_file:
        json.dump(quality_summary, summary_file, indent=2, default=_to_serializable)

    print(f"[SOP-Phase1] Fused clean surface points: {surface_points.shape[0]}")
    print(f"[SOP-Phase1] Probes: {probe_points.shape[0]}, offset distance: {offset_distance:.6f}")
    print(f"[SOP-Phase1] Status counts: {status_counts}")
    print(f"[SOP-Phase1] Outputs written to {output_root}")

    return quality_summary


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Phase 1 SOP initialization without training")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--camera_set", choices=["train", "test", "all"], default="train")
    parser.add_argument("--object_filter_mode", choices=["weight_only", "mask_only", "weight_or_mask", "weight_and_mask"], default="weight_and_mask")
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--max_points_per_view", default=100000, type=int)
    parser.add_argument("--fusion_buffer_points", default=1500000, type=int)
    parser.add_argument("--fusion_voxel_size", default=0.0, type=float)
    parser.add_argument("--fusion_voxel_factor", default=0.0005, type=float)
    parser.add_argument("--outlier_radius", default=0.0, type=float)
    parser.add_argument("--outlier_radius_factor", default=2.5, type=float)
    parser.add_argument("--outlier_min_neighbors", default=4, type=int)
    parser.add_argument("--skip_mesh_export", action="store_true")
    parser.add_argument("--mesh_voxel_size", default=-1.0, type=float)
    parser.add_argument("--mesh_depth_trunc", default=-1.0, type=float)
    parser.add_argument("--mesh_sdf_trunc", default=-1.0, type=float)
    parser.add_argument("--mesh_res", default=1024, type=int)
    parser.add_argument("--mesh_num_cluster", default=1, type=int)
    parser.add_argument("--mesh_probe_radius", default=-1.0, type=float)
    parser.add_argument("--mesh_probe_resolution", default=5, type=int)
    parser.add_argument("--mesh_max_probe_spheres", default=0, type=int)
    parser.add_argument("--weight_thresh", default=0.1, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument("--mask_erosion_radius", default=4, type=int)
    parser.add_argument("--target_num_probes", default=5000, type=int)
    parser.add_argument("--offset_scale", default=0.01, type=float)
    parser.add_argument("--extent_mode", choices=["bbox_diagonal", "max_side"], default="bbox_diagonal")
    parser.add_argument("--too_close_factor", default=0.5, type=float)
    parser.add_argument("--too_far_factor", default=3.0, type=float)
    parser.add_argument("--backface_alignment_thresh", default=0.0, type=float)
    parser.add_argument("--max_gaussians_vis", default=30000, type=int)
    parser.add_argument("--max_probe_normals_vis", default=400, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = get_combined_args(parser)
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialize_sop_phase1(args)


if __name__ == "__main__":
    main()
