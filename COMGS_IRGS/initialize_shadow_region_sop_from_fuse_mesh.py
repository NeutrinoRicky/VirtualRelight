from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scene.cameras import Camera
from utils.general_utils import safe_state

from initialize_shadow_region_sop import (
    _bbox_info,
    _compute_extent,
    _ensure_dir,
    _sample_rows,
    _save_lineset,
    _save_point_cloud,
    _save_point_cloud_limited,
    _save_scene_and_probe_combo,
    _summarize_array,
    _write_json,
    load_gaussians_auto,
)
from render_lego_into_ignatius import (
    _ensure_repo_cwd,
    apply_gaussian_transform,
    apply_post_transform_scale,
    camera_from_json,
    load_camera_json,
    parse_int_list,
    select_camera_jsons,
)
from SOP.phase1_initializer import (
    _build_mesh_raycast_scene,
    _compute_mesh_first_hit_visibility_counts,
    _refine_probe_points_outside_mesh,
    _sample_points_from_mesh,
)
from SOP.phase1_initializer_point_cloud import _farthest_point_sampling_with_min_distance, _normalize_np

try:
    import open3d as o3d
except ImportError:
    o3d = None


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


def _make_output_dirs(output_root: Path) -> None:
    _ensure_dir(output_root)
    _ensure_dir(output_root / "visualization")


def _load_mesh(mesh_path: Path):
    if o3d is None:
        raise RuntimeError("open3d is required for fuse_post mesh initialization, but it is unavailable.")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh is None:
        raise RuntimeError(f"Failed to read mesh: {mesh_path}")
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh is empty: {mesh_path}")
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _crop_mesh_to_region(mesh, center: np.ndarray, radius: float):
    if o3d is None:
        raise RuntimeError("open3d is required for mesh cropping, but it is unavailable.")
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(center - radius).astype(np.float64),
        max_bound=(center + radius).astype(np.float64),
    )
    cropped = mesh.crop(bbox)
    if len(cropped.vertices) == 0 or len(cropped.triangles) == 0:
        raise RuntimeError("Cropped local mesh is empty. Increase --local_scene_radius_factor or move the object.")
    cropped.compute_triangle_normals()
    cropped.compute_vertex_normals()
    return cropped


def _filter_points_inside_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    rel = points - center.reshape(1, 3)
    return np.linalg.norm(rel, axis=1) <= float(radius)


@torch.no_grad()
def _orient_normals_by_visible_cameras(
    points: np.ndarray,
    normals: np.ndarray,
    cameras: List[Camera],
    raycast_scene,
    chunk_size: int,
    occlusion_tolerance: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    stats: Dict[str, object] = {
        "enabled": False,
        "mode": "visible_camera_vote",
        "num_points": int(points.shape[0]),
        "num_cameras": int(len(cameras)),
    }
    if points.shape[0] == 0 or len(cameras) == 0:
        stats["reason"] = "empty_points_or_cameras"
        return normals, stats

    device = torch.device("cuda")
    pts = torch.from_numpy(points.astype(np.float32)).to(device=device)
    nrms = torch.from_numpy(_normalize_np(normals.astype(np.float32))).to(device=device)
    vote_vectors = torch.zeros_like(pts)
    vote_counts = torch.zeros((pts.shape[0],), dtype=torch.int32, device=device)
    eps = 1e-6

    chunk_size = max(1, int(chunk_size))
    for start in range(0, pts.shape[0], chunk_size):
        end = min(start + chunk_size, pts.shape[0])
        points_chunk = pts[start:end]
        points_h = torch.cat([points_chunk, torch.ones_like(points_chunk[:, :1])], dim=-1)
        chunk_votes = torch.zeros_like(points_chunk)
        chunk_counts = torch.zeros((end - start,), dtype=torch.int32, device=device)

        for viewpoint_cam in cameras:
            projected = points_h @ viewpoint_cam.full_proj_transform
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
            if raycast_scene is not None:
                to_point = points_chunk - viewpoint_cam.camera_center.view(1, 3)
                target_dist = torch.linalg.norm(to_point, dim=-1)
                valid_ray = in_image & torch.isfinite(target_dist) & (target_dist > eps)
                ray_visible = torch.zeros_like(in_image)
                if torch.any(valid_ray):
                    ray_origins = viewpoint_cam.camera_center.view(1, 3).expand(int(valid_ray.sum().item()), 3)
                    ray_dirs = F.normalize(to_point[valid_ray], dim=-1, eps=eps)
                    rays_np = torch.cat([ray_origins, ray_dirs], dim=-1).detach().cpu().numpy().astype(np.float32)
                    ray_hits = raycast_scene.cast_rays(o3d.core.Tensor(rays_np))
                    t_hit = np.asarray(ray_hits["t_hit"].numpy(), dtype=np.float32).reshape(-1)
                    target_dist_np = target_dist[valid_ray].detach().cpu().numpy().astype(np.float32)
                    visible_np = np.isfinite(t_hit) & (np.abs(t_hit - target_dist_np) <= float(occlusion_tolerance))
                    ray_visible[valid_ray] = torch.from_numpy(visible_np).to(device=device, dtype=torch.bool)
                in_image = ray_visible

            if not torch.any(in_image):
                continue
            to_camera = F.normalize(viewpoint_cam.camera_center.view(1, 3) - points_chunk, dim=-1, eps=eps)
            visible_f = in_image.to(dtype=points_chunk.dtype)
            chunk_votes += to_camera * visible_f[:, None]
            chunk_counts += in_image.to(dtype=torch.int32)

        vote_vectors[start:end] = chunk_votes
        vote_counts[start:end] = chunk_counts

    vote_norm = torch.linalg.norm(vote_vectors, dim=-1)
    has_votes = (vote_counts > 0) & (vote_norm > 1e-8)
    alignment = torch.sum(nrms * vote_vectors, dim=-1)
    flip = has_votes & (alignment < 0.0)
    nrms = torch.where(flip[:, None], -nrms, nrms)

    vote_counts_np = vote_counts.detach().cpu().numpy().astype(np.float32)
    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "points_with_votes": int(np.sum(vote_counts_np > 0)),
            "points_without_votes": int(np.sum(vote_counts_np <= 0)),
            "flipped_points": int(torch.sum(flip).item()),
            "vote_count": _summarize_array(vote_counts_np),
        }
    )
    return _normalize_np(nrms.detach().cpu().numpy().astype(np.float32)), stats


def _build_camera_list(args) -> List[Camera]:
    cameras_json = select_camera_jsons(
        load_camera_json(Path(args.scene_cameras)),
        first_k=int(args.first_k),
        camera_ids=parse_int_list(args.camera_ids),
    )
    if not cameras_json:
        raise RuntimeError("No scene cameras selected for mesh visibility filtering.")
    return [
        camera_from_json(cam_json, max_width=int(args.max_width), scale=float(args.resolution_scale))
        for cam_json in cameras_json
    ]


@torch.no_grad()
def initialize_shadow_region_sop_from_fuse_mesh(args) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Fuse-mesh SOP initialization currently requires CUDA.")

    _ensure_repo_cwd()
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_root = Path(args.output_dir)
    _make_output_dirs(output_root)

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
    object_bbox = _bbox_info(object_points)
    obj_radius = float(object_bbox["radius"])
    if obj_radius <= 0.0:
        raise RuntimeError("Object bbox radius is zero after placement.")

    shadow_center = np.asarray(object_bbox["center"], dtype=np.float32) + np.asarray(args.shadow_center_offset, dtype=np.float32)
    shadow_radius = float(args.shadow_radius_factor) * obj_radius
    local_scene_radius = float(args.local_scene_radius_factor) * shadow_radius
    if local_scene_radius <= 0.0:
        raise RuntimeError("local_scene_radius must be positive.")

    mesh = _load_mesh(Path(args.mesh_path))
    cropped_mesh = _crop_mesh_to_region(mesh, center=shadow_center, radius=local_scene_radius)
    if o3d is not None:
        o3d.io.write_triangle_mesh(str(output_root / "visualization" / "mesh_local_cropped.ply"), cropped_mesh)

    requested_num_probes = int(args.target_num_probes)
    if requested_num_probes <= 0:
        raise RuntimeError("target_num_probes must be positive.")

    dense_sample_count = int(args.mesh_surface_sample_count)
    if dense_sample_count <= 0:
        dense_sample_count = max(requested_num_probes * 30, 300000)
    dense_sample_count = max(dense_sample_count, requested_num_probes)

    mesh_surface_points_all, mesh_surface_normals_all = _sample_points_from_mesh(cropped_mesh, dense_sample_count, rng)
    local_mask = _filter_points_inside_sphere(mesh_surface_points_all, shadow_center, local_scene_radius)
    mesh_surface_points_local = mesh_surface_points_all[local_mask]
    mesh_surface_normals_local = mesh_surface_normals_all[local_mask]
    if mesh_surface_points_local.shape[0] == 0:
        raise RuntimeError("No mesh surface samples remained inside the local shadow region.")

    visibility_cameras = _build_camera_list(args)
    raycast_scene = _build_mesh_raycast_scene(cropped_mesh)
    if raycast_scene is None:
        raise RuntimeError("Failed to build raycasting scene for cropped mesh.")

    mesh_vertices = np.asarray(cropped_mesh.vertices, dtype=np.float32)
    mesh_extent = _compute_extent(mesh_vertices if mesh_vertices.shape[0] > 0 else mesh_surface_points_local, args.extent_mode)
    visibility_tol = (
        float(args.mesh_visibility_occlusion_tol)
        if args.mesh_visibility_occlusion_tol > 0.0
        else float(args.mesh_visibility_occlusion_tol_factor) * mesh_extent
    )
    visibility_counts = _compute_mesh_first_hit_visibility_counts(
        points=mesh_surface_points_local,
        cameras=visibility_cameras,
        device=torch.device("cuda"),
        chunk_size=int(args.mesh_visibility_chunk_size),
        raycast_scene=raycast_scene,
        occlusion_tolerance=visibility_tol,
    )
    visible_mask = visibility_counts >= int(args.mesh_visibility_min_views)
    if not np.any(visible_mask):
        raise RuntimeError(
            "Mesh visibility filtering removed all local surface samples. "
            "Try lowering --mesh_visibility_min_views or increasing --mesh_surface_sample_count."
        )

    mesh_surface_points = mesh_surface_points_local[visible_mask]
    mesh_surface_normals = mesh_surface_normals_local[visible_mask]
    visibility_counts_kept = visibility_counts[visible_mask]

    mesh_surface_normals, normal_orientation_stats = _orient_normals_by_visible_cameras(
        points=mesh_surface_points,
        normals=mesh_surface_normals,
        cameras=visibility_cameras,
        raycast_scene=raycast_scene,
        chunk_size=int(args.mesh_visibility_chunk_size),
        occlusion_tolerance=visibility_tol,
    )

    if mesh_surface_points.shape[0] < requested_num_probes:
        raise RuntimeError(
            f"Need exactly {requested_num_probes} probes, but only {mesh_surface_points.shape[0]} visible mesh samples are available."
        )

    probe_extent_reference = (
        float(args.offset_reference_distance)
        if args.offset_reference_distance > 0.0
        else float(object_bbox["diagonal"])
    )
    offset_distance = (
        float(args.offset_distance)
        if args.offset_distance > 0.0
        else float(args.offset_scale) * probe_extent_reference
    )
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0.")
    probe_min_distance = (
        float(args.probe_min_distance)
        if args.probe_min_distance > 0.0
        else float(args.probe_min_distance_factor) * probe_extent_reference
    )

    candidate_probe_points = mesh_surface_points + mesh_surface_normals * offset_distance
    probe_surface_indices, probe_selection_stats = _farthest_point_sampling_with_min_distance(
        candidate_probe_points,
        requested_num_probes,
        device=torch.device("cuda"),
        min_distance=probe_min_distance,
        distance_space="mesh_final_probe_points",
    )
    probe_surface_points = mesh_surface_points[probe_surface_indices]
    probe_normals = mesh_surface_normals[probe_surface_indices]
    probe_points = candidate_probe_points[probe_surface_indices]

    raw_probe_points = probe_points.copy()
    raw_probe_normals = probe_normals.copy()
    (
        probe_points,
        probe_normals,
        mesh_outside_check_stats,
        mesh_signed_distance_before,
        mesh_signed_distance_after,
        mesh_correction_method,
    ) = _refine_probe_points_outside_mesh(
        probe_surface_points=probe_surface_points,
        probe_points=probe_points,
        probe_normals=probe_normals,
        mesh_largest=cropped_mesh,
        offset_distance=offset_distance,
        args=Namespace(
            mesh_outside_check_nsamples=int(args.mesh_outside_check_nsamples),
            mesh_outside_search_steps=int(args.mesh_outside_search_steps),
            mesh_outside_search_max_scale=float(args.mesh_outside_search_max_scale),
        ),
    )

    _save_point_cloud(
        output_root / "mesh_surface_samples_all.ply",
        mesh_surface_points_all,
        mesh_surface_normals_all,
        np.full_like(mesh_surface_points_all, 0.45, dtype=np.float32),
    )
    _save_point_cloud(
        output_root / "mesh_surface_samples_local_visible.ply",
        mesh_surface_points,
        mesh_surface_normals,
        np.full_like(mesh_surface_points, 0.65, dtype=np.float32),
    )
    rejected_mask = ~visible_mask
    if np.any(rejected_mask):
        _save_point_cloud(
            output_root / "mesh_surface_samples_local_visibility_rejected.ply",
            mesh_surface_points_local[rejected_mask],
            mesh_surface_normals_local[rejected_mask],
            np.tile(np.array([[0.20, 0.20, 0.20]], dtype=np.float32), (int(np.sum(rejected_mask)), 1)),
        )
    _save_point_cloud(
        output_root / "surface_fused_clean.ply",
        mesh_surface_points,
        mesh_surface_normals,
        np.full_like(mesh_surface_points, 0.65, dtype=np.float32),
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
    if mesh_outside_check_stats.get("enabled", False) and mesh_outside_check_stats.get("initial_inside", 0) > 0:
        _save_point_cloud(
            output_root / "probe_offset_points_before_mesh_outside_check.ply",
            raw_probe_points,
            raw_probe_normals,
            np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (raw_probe_points.shape[0], 1)),
        )

    normal_vis_idx = np.arange(probe_points.shape[0], dtype=np.int64)
    if args.max_probe_normals_vis > 0 and normal_vis_idx.shape[0] > args.max_probe_normals_vis:
        normal_vis_idx = np.sort(rng.choice(normal_vis_idx, size=args.max_probe_normals_vis, replace=False))
    _save_lineset(
        output_root / "probe_normals_lineset.ply",
        starts=probe_points[normal_vis_idx],
        ends=probe_points[normal_vis_idx] + probe_normals[normal_vis_idx] * offset_distance,
        color=(0.12, 0.47, 0.71),
    )

    object_colors = object_gaussians.get_base_color.detach().cpu().numpy().astype(np.float32)
    _save_point_cloud(
        output_root / "visualization" / "object_transformed_gaussians.ply",
        object_points,
        np.zeros_like(object_points, dtype=np.float32),
        object_colors,
    )
    local_scene_debug = _save_point_cloud_limited(
        output_root / "scene_local_mesh_visible.ply",
        mesh_surface_points,
        mesh_surface_normals,
        np.full_like(mesh_surface_points, 0.65, dtype=np.float32),
        max_points=int(args.max_scene_points_save),
        rng=rng,
    )
    _save_scene_and_probe_combo(
        output_root / "scene_local_plus_object_plus_sops.ply",
        scene_points=_sample_rows(mesh_surface_points, int(args.max_scene_points_in_combo), rng),
        object_points=_sample_rows(object_points, int(args.max_object_points_in_combo), rng),
        probe_surface_points=probe_surface_points,
        probe_points=probe_points,
    )

    np.savez(
        output_root / "probe_init_data.npz",
        surface_points=mesh_surface_points.astype(np.float32),
        surface_normals=mesh_surface_normals.astype(np.float32),
        probe_reference_points=mesh_surface_points.astype(np.float32),
        probe_reference_normals=mesh_surface_normals.astype(np.float32),
        probe_surface_points=probe_surface_points.astype(np.float32),
        probe_points=probe_points.astype(np.float32),
        probe_normals=probe_normals.astype(np.float32),
        probe_surface_indices=probe_surface_indices.astype(np.int64),
        mesh_surface_points_all=mesh_surface_points_all.astype(np.float32),
        mesh_surface_normals_all=mesh_surface_normals_all.astype(np.float32),
        mesh_surface_points_local=mesh_surface_points_local.astype(np.float32),
        mesh_surface_normals_local=mesh_surface_normals_local.astype(np.float32),
        mesh_surface_visibility_counts=visibility_counts_kept.astype(np.int32),
        mesh_surface_visibility_counts_local=visibility_counts.astype(np.int32),
        mesh_surface_visible_mask=visible_mask,
        probe_points_before_mesh_outside_check=raw_probe_points.astype(np.float32),
        probe_normals_before_mesh_outside_check=raw_probe_normals.astype(np.float32),
        mesh_signed_distance_before=mesh_signed_distance_before.astype(np.float32),
        mesh_signed_distance_after=mesh_signed_distance_after.astype(np.float32),
        mesh_correction_method=mesh_correction_method,
        object_bbox_min=np.asarray(object_bbox["min"], dtype=np.float32),
        object_bbox_max=np.asarray(object_bbox["max"], dtype=np.float32),
        shadow_center=shadow_center.astype(np.float32),
        shadow_radius=np.array([shadow_radius], dtype=np.float32),
        offset_distance=np.array([offset_distance], dtype=np.float32),
    )

    summary = {
        "format": "irgs_shadow_region_sop_from_fuse_mesh_v1",
        "mesh_path": str(Path(args.mesh_path).resolve()),
        "scene_cameras": str(Path(args.scene_cameras).resolve()),
        "output_root": str(output_root.resolve()),
        "object_checkpoint": str(Path(args.object_checkpoint).resolve()),
        "object_iteration": int(object_iter),
        "object_format": object_format,
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
            "side_lengths": np.asarray(object_bbox["side_lengths"], dtype=np.float32).astype(float).tolist(),
            "diagonal": float(object_bbox["diagonal"]),
            "radius": float(object_bbox["radius"]),
        },
        "shadow_region": {
            "center": shadow_center.astype(float).tolist(),
            "shadow_radius_factor": float(args.shadow_radius_factor),
            "shadow_radius": float(shadow_radius),
            "local_scene_radius_factor": float(args.local_scene_radius_factor),
            "local_scene_radius": float(local_scene_radius),
        },
        "mesh": {
            "cropped_vertices": int(len(cropped_mesh.vertices)),
            "cropped_triangles": int(len(cropped_mesh.triangles)),
            "mesh_extent": float(mesh_extent),
        },
        "mesh_sampling": {
            "mesh_surface_sample_count": int(dense_sample_count),
            "num_local_samples": int(mesh_surface_points_local.shape[0]),
            "num_visible_samples": int(mesh_surface_points.shape[0]),
            "visibility_min_views": int(args.mesh_visibility_min_views),
            "visibility_occlusion_tolerance": float(visibility_tol),
            "visibility_count_local": _summarize_array(visibility_counts.astype(np.float32)),
        },
        "normal_orientation": normal_orientation_stats,
        "mesh_outside_check": mesh_outside_check_stats,
        "probe_selection": probe_selection_stats,
        "offset_distance": float(offset_distance),
        "probe_min_distance": float(probe_min_distance),
        "num_probes": int(probe_points.shape[0]),
        "mesh_signed_distance_before": _summarize_array(mesh_signed_distance_before),
        "mesh_signed_distance_after": _summarize_array(mesh_signed_distance_after),
        "local_scene_export": local_scene_debug,
        "args": dict(vars(args)),
    }
    _write_json(output_root / "shadow_region_summary.json", summary)

    print(f"[shadow-sop-mesh] Cropped mesh vertices: {len(cropped_mesh.vertices)}, triangles: {len(cropped_mesh.triangles)}")
    print(f"[shadow-sop-mesh] Local samples: {mesh_surface_points_local.shape[0]}, visible: {mesh_surface_points.shape[0]}")
    print(f"[shadow-sop-mesh] Probes: {probe_points.shape[0]}")
    print(f"[shadow-sop-mesh] Offset distance: {offset_distance:.6f}")
    print(f"[shadow-sop-mesh] Output root: {output_root}")
    return summary


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Initialize shadow-region SOPs from a precomputed fuse_post mesh by "
            "sampling visible mesh surface points and offsetting them along mesh normals."
        )
    )
    parser.add_argument(
        "--mesh_path",
        default=str(REPO_DIR / "outputs/tnt/Ignatius/train/ours_30000/fuse_post.ply"),
    )
    parser.add_argument(
        "--scene_cameras",
        default=str(REPO_DIR / "outputs/tnt/Ignatius/cameras.json"),
    )
    parser.add_argument(
        "--object_checkpoint",
        default=str(REPO_DIR / "outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_DIR / "outputs/composite/lego_shadow_region_sop_from_fuse_mesh"),
    )

    parser.add_argument("--translation", nargs=3, type=float, default=[-2.37, 1.451, -0.45])
    parser.add_argument("--rotation_deg", nargs=3, type=float, default=[83.57, 190.7, 0.254])
    parser.add_argument("--rotation_order", default="XYZ")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--post_scale", type=float, default=0.632)
    parser.add_argument("--post_scale_center", choices=["bbox", "mean", "origin"], default="bbox")

    parser.add_argument("--shadow_radius_factor", type=float, default=6.0)
    parser.add_argument("--shadow_center_offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--local_scene_radius_factor", type=float, default=1.05)
    parser.add_argument("--up_axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="y")

    parser.add_argument("--mesh_surface_sample_count", type=int, default=0)
    parser.add_argument("--mesh_visibility_min_views", type=int, default=1)
    parser.add_argument("--mesh_visibility_chunk_size", type=int, default=65536)
    parser.add_argument("--mesh_visibility_occlusion_tol", type=float, default=0.0)
    parser.add_argument("--mesh_visibility_occlusion_tol_factor", type=float, default=0.002)
    parser.add_argument("--mesh_outside_check_nsamples", type=int, default=11)
    parser.add_argument("--mesh_outside_search_steps", type=int, default=10)
    parser.add_argument("--mesh_outside_search_max_scale", type=float, default=4.0)

    parser.add_argument("--extent_mode", choices=["bbox_diagonal", "max_side"], default="bbox_diagonal")
    parser.add_argument("--target_num_probes", type=int, default=10000)
    parser.add_argument("--probe_min_distance_factor", type=float, default=0.0)
    parser.add_argument("--probe_min_distance", type=float, default=0.0)
    parser.add_argument("--offset_scale", type=float, default=0.01)
    parser.add_argument("--offset_distance", type=float, default=0.0)
    parser.add_argument("--offset_reference_distance", type=float, default=0.0)

    parser.add_argument("--first_k", type=int, default=-1)
    parser.add_argument("--camera_ids", default="", help="Comma-separated scene camera ids to use for visibility checks.")
    parser.add_argument("--max_width", type=int, default=1600)
    parser.add_argument("--resolution_scale", type=float, default=1.0)

    parser.add_argument("--max_probe_normals_vis", type=int, default=600)
    parser.add_argument("--max_debug_points", type=int, default=500000)
    parser.add_argument("--max_scene_points_save", type=int, default=250000)
    parser.add_argument("--max_scene_points_in_combo", type=int, default=120000)
    parser.add_argument("--max_object_points_in_combo", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    initialize_shadow_region_sop_from_fuse_mesh(args)


if __name__ == "__main__":
    main()
