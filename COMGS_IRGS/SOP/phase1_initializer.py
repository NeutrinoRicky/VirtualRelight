from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
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
from utils.mesh_utils import GaussianExtractor, post_process_mesh

try:
    import open3d as o3d
except ImportError:
    o3d = None


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


def _compute_object_extent(points: np.ndarray, mode: str) -> float:
    side_lengths, diagonal, max_side = _bbox_extent(points)
    if mode == "bbox_diagonal":
        return diagonal
    if mode == "max_side":
        return max_side
    raise ValueError(f"Unknown extent mode: {mode}")


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
        mask_valid = gt_alpha_mask[0] > mask_thresh
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


@torch.no_grad()
def _render_mesh_gbuffer(viewpoint_cam, gaussians: GaussianModel, pipe, bg_color: torch.Tensor):
    render_pkg = render_sop_gbuffer(viewpoint_cam, gaussians, pipe, bg_color)
    weight = torch.clamp(render_pkg["weight"], 0.0, 1.0)
    rgb = torch.clamp(render_pkg["albedo"] + bg_color[:, None, None] * (1.0 - weight[None]), 0.0, 1.0)
    return {
        "render": rgb,
        "rend_alpha": weight[None],
        "rend_normal": render_pkg["normal_shading"] * weight[None],
        "surf_depth": render_pkg["depth_unbiased"][None],
        "surf_normal": render_pkg["normal"] * weight[None],
    }


@torch.no_grad()
def _extract_largest_mesh(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    dataset,
    output_root: Path,
    args,
) -> Tuple[Optional["o3d.geometry.TriangleMesh"], Dict[str, object]]:
    if o3d is None:
        return None, {"enabled": False, "reason": "open3d_unavailable"}

    all_cameras = _choose_cameras(scene, args.camera_set)
    if len(all_cameras) == 0:
        return None, {"enabled": False, "reason": f"no_cameras_for_{args.camera_set}"}
    selected_cameras = all_cameras[:: max(1, args.view_stride)]
    if args.max_views > 0:
        selected_cameras = selected_cameras[: args.max_views]
    if len(selected_cameras) == 0:
        return None, {"enabled": False, "reason": "empty_mesh_camera_selection"}

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    extractor = GaussianExtractor(gaussians, _render_mesh_gbuffer, pipe, bg_color=bg_color)
    prev_sh_degree = gaussians.active_sh_degree

    try:
        extractor.gaussians.active_sh_degree = 0
        extractor.reconstruction(selected_cameras)
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
    o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh)

    mesh_largest = post_process_mesh(mesh, cluster_to_keep=max(1, int(args.mesh_num_cluster)))
    if len(mesh_largest.triangles) > 0:
        mesh_largest.compute_triangle_normals()
        mesh_largest.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(largest_mesh_path), mesh_largest)

    return mesh_largest, {
        "enabled": True,
        "raw_mesh_path": str(raw_mesh_path),
        "largest_mesh_path": str(largest_mesh_path),
        "depth_trunc": float(depth_trunc),
        "voxel_size": float(voxel_size),
        "sdf_trunc": float(sdf_trunc),
        "mesh_num_cluster": int(args.mesh_num_cluster),
        "mesh_num_cameras": int(len(selected_cameras)),
        "raw_mesh_vertices": int(len(mesh.vertices)),
        "largest_mesh_vertices": int(len(mesh_largest.vertices)),
        "largest_mesh_triangles": int(len(mesh_largest.triangles)),
    }


def _export_mesh_with_probes(
    mesh_largest,
    output_root: Path,
    probe_points: np.ndarray,
    offset_distance: float,
    args,
) -> Dict[str, object]:
    if o3d is None or mesh_largest is None:
        return {}

    mesh_with_probes_path = output_root / "mesh_largest_with_probes.ply"
    if args.mesh_max_probe_spheres > 0 and probe_points.shape[0] > args.mesh_max_probe_spheres:
        probe_indices = np.linspace(0, probe_points.shape[0] - 1, args.mesh_max_probe_spheres, dtype=np.int64)
        probe_points_mesh = probe_points[probe_indices]
    else:
        probe_points_mesh = probe_points

    mesh_vertices = np.asarray(mesh_largest.vertices, dtype=np.float32)
    if mesh_vertices.shape[0] > 0:
        _, _, largest_side = _bbox_extent(mesh_vertices)
    else:
        largest_side = offset_distance

    probe_radius = args.mesh_probe_radius if args.mesh_probe_radius > 0 else max(offset_distance * 0.12, largest_side * 0.003)
    probe_meshes = [mesh_largest]
    for point in probe_points_mesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=probe_radius, resolution=max(4, int(args.mesh_probe_resolution)))
        sphere.translate(point.astype(np.float64))
        sphere.paint_uniform_color([0.85, 0.28, 0.25])
        probe_meshes.append(sphere)

    mesh_with_probes = _merge_triangle_meshes(probe_meshes)
    o3d.io.write_triangle_mesh(str(mesh_with_probes_path), mesh_with_probes)

    return {
        "mesh_with_probes_path": str(mesh_with_probes_path),
        "probe_radius": float(probe_radius),
        "probe_spheres": int(probe_points_mesh.shape[0]),
    }


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


def _sample_points_from_mesh(
    mesh,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if o3d is None or mesh is None:
        raise RuntimeError("open3d mesh sampling requested but open3d is unavailable")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        raise RuntimeError("mesh_largest is empty and cannot be sampled")

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if vertex_normals.shape[0] != vertices.shape[0]:
        raise RuntimeError("mesh_largest vertex normals are unavailable")

    tri_vertices = vertices[triangles]
    edge_01 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge_02 = tri_vertices[:, 2] - tri_vertices[:, 0]
    tri_area2 = np.linalg.norm(np.cross(edge_01, edge_02), axis=1)
    valid = tri_area2 > 1e-12
    if not np.any(valid):
        raise RuntimeError("mesh_largest has no valid triangles for sampling")

    tri_vertices = tri_vertices[valid]
    tri_indices = triangles[valid]
    tri_weights = tri_area2[valid].astype(np.float64)
    tri_weights /= np.clip(np.sum(tri_weights), 1e-12, None)

    chosen = rng.choice(tri_vertices.shape[0], size=num_samples, replace=True, p=tri_weights)
    sampled_tris = tri_vertices[chosen]
    sampled_idx = tri_indices[chosen]

    u = rng.random(num_samples).astype(np.float32)
    v = rng.random(num_samples).astype(np.float32)
    sqrt_u = np.sqrt(u)
    bary = np.stack(
        [
            1.0 - sqrt_u,
            sqrt_u * (1.0 - v),
            sqrt_u * v,
        ],
        axis=1,
    ).astype(np.float32)

    sampled_points = np.sum(sampled_tris * bary[:, :, None], axis=1)
    sampled_normals = np.sum(vertex_normals[sampled_idx] * bary[:, :, None], axis=1)
    sampled_normals = _normalize_np(sampled_normals.astype(np.float32))

    mesh_center = vertices.mean(axis=0, keepdims=True)
    outward_hint = sampled_points - mesh_center
    flip = np.sum(sampled_normals * outward_hint, axis=1) < 0.0
    sampled_normals[flip] *= -1.0
    return sampled_points.astype(np.float32), sampled_normals.astype(np.float32)


def _sample_probe_surface_from_mesh(
    mesh,
    target_num_probes: int,
    dense_sample_count: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dense_points, dense_normals = _sample_points_from_mesh(mesh, dense_sample_count, rng)
    probe_surface_indices = _farthest_point_sampling(dense_points, target_num_probes, device)
    probe_surface_points = dense_points[probe_surface_indices]
    probe_surface_normals = _normalize_np(dense_normals[probe_surface_indices])
    return dense_points, dense_normals, probe_surface_points, probe_surface_normals, probe_surface_indices


def _select_train_visibility_cameras(scene: Scene, args) -> List:
    cameras = list(scene.getTrainCameras())
    cameras = cameras[:: max(1, int(args.mesh_visibility_view_stride))]
    if args.mesh_visibility_max_views > 0:
        cameras = cameras[: int(args.mesh_visibility_max_views)]
    return cameras


@torch.no_grad()
def _compute_mesh_first_hit_visibility_counts(
    points: np.ndarray,
    cameras: List,
    device: torch.device,
    chunk_size: int,
    raycast_scene=None,
    occlusion_tolerance: float = 0.0,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(cameras) == 0:
        return np.zeros((points.shape[0],), dtype=np.int32)

    counts = np.zeros((points.shape[0],), dtype=np.int32)
    chunk_size = max(1, int(chunk_size))
    eps = 1e-6

    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        points_t = torch.from_numpy(points[start:end].astype(np.float32)).to(device)
        points_h = torch.cat([points_t, torch.ones_like(points_t[:, :1])], dim=-1)
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
                to_point = points_t - viewpoint_cam.camera_center.view(1, 3)
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

            chunk_counts += in_image.to(torch.int32)

        counts[start:end] = chunk_counts.detach().cpu().numpy()

    return counts


def _build_mesh_raycast_scene(mesh):
    if o3d is None or mesh is None or len(mesh.triangles) == 0:
        return None

    try:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        raycast_scene = o3d.t.geometry.RaycastingScene()
        raycast_scene.add_triangles(mesh_t)
        return raycast_scene
    except Exception:
        return None


def _query_mesh_signed_distance(
    raycast_scene,
    query_points: np.ndarray,
    nsamples: int,
) -> np.ndarray:
    if raycast_scene is None:
        raise RuntimeError("Raycasting scene is unavailable for mesh signed-distance queries")
    if query_points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    nsamples = max(1, int(nsamples))
    if nsamples % 2 == 0:
        nsamples += 1

    query_tensor = o3d.core.Tensor(query_points.astype(np.float32))
    signed_distance = raycast_scene.compute_signed_distance(query_tensor, nsamples=nsamples)
    return np.asarray(signed_distance.numpy(), dtype=np.float32).reshape(-1)


def _binary_search_probe_outside(
    raycast_scene,
    surface_points: np.ndarray,
    directions: np.ndarray,
    offset_distance: float,
    inside_scales: np.ndarray,
    outside_scales: np.ndarray,
    nsamples: int,
    search_steps: int,
    outside_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside_scales = inside_scales.astype(np.float32).copy()
    outside_scales = outside_scales.astype(np.float32).copy()

    for _ in range(max(1, int(search_steps))):
        mid_scales = 0.5 * (inside_scales + outside_scales)
        mid_points = surface_points + directions * (mid_scales[:, None] * offset_distance)
        mid_signed_distance = _query_mesh_signed_distance(raycast_scene, mid_points, nsamples=nsamples)
        mid_outside = mid_signed_distance > outside_tol
        outside_scales = np.where(mid_outside, mid_scales, outside_scales)
        inside_scales = np.where(mid_outside, inside_scales, mid_scales)

    refined_points = surface_points + directions * (outside_scales[:, None] * offset_distance)
    refined_signed_distance = _query_mesh_signed_distance(raycast_scene, refined_points, nsamples=nsamples)
    return outside_scales.astype(np.float32), refined_points.astype(np.float32), refined_signed_distance.astype(np.float32)


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


def _refine_probe_points_outside_mesh(
    probe_surface_points: np.ndarray,
    probe_points: np.ndarray,
    probe_normals: np.ndarray,
    mesh_largest,
    offset_distance: float,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    default_methods = np.full(probe_points.shape[0], "not_checked", dtype="<U24")
    if o3d is None:
        return (
            probe_points,
            probe_normals,
            {"enabled": False, "reason": "open3d_unavailable"},
            np.zeros((probe_points.shape[0],), dtype=np.float32),
            np.zeros((probe_points.shape[0],), dtype=np.float32),
            default_methods,
        )

    raycast_scene = _build_mesh_raycast_scene(mesh_largest)
    if raycast_scene is None:
        return (
            probe_points,
            probe_normals,
            {"enabled": False, "reason": "raycasting_scene_unavailable"},
            np.zeros((probe_points.shape[0],), dtype=np.float32),
            np.zeros((probe_points.shape[0],), dtype=np.float32),
            default_methods,
        )

    nsamples = max(1, int(args.mesh_outside_check_nsamples))
    if nsamples % 2 == 0:
        nsamples += 1
    outside_tol = max(float(offset_distance) * 1e-4, 1e-6)
    max_scale = max(float(args.mesh_outside_search_max_scale), 1.25)
    search_steps = max(1, int(args.mesh_outside_search_steps))

    corrected_points = probe_points.astype(np.float32).copy()
    corrected_normals = probe_normals.astype(np.float32).copy()
    correction_method = np.full(probe_points.shape[0], "unchanged", dtype="<U24")

    initial_signed_distance = _query_mesh_signed_distance(raycast_scene, corrected_points, nsamples=nsamples)
    initial_inside_mask = initial_signed_distance < -outside_tol

    if not np.any(initial_inside_mask):
        stats = {
            "enabled": True,
            "nsamples": int(nsamples),
            "outside_tol": float(outside_tol),
            "search_steps": int(search_steps),
            "search_max_scale": float(max_scale),
            "num_checked": int(probe_points.shape[0]),
            "initial_inside": 0,
            "corrected_by_flip": 0,
            "corrected_by_search": 0,
            "remaining_inside": 0,
            "initial_signed_distance": _summarize_array(initial_signed_distance),
            "final_signed_distance": _summarize_array(initial_signed_distance),
        }
        return corrected_points, corrected_normals, stats, initial_signed_distance, initial_signed_distance.copy(), correction_method

    inside_indices = np.flatnonzero(initial_inside_mask)
    surface_inside = probe_surface_points[inside_indices]
    normals_inside = probe_normals[inside_indices]

    flip_points = surface_inside - normals_inside * offset_distance
    flip_signed_distance = _query_mesh_signed_distance(raycast_scene, flip_points, nsamples=nsamples)
    flip_outside = flip_signed_distance > outside_tol
    if np.any(flip_outside):
        chosen_indices = inside_indices[flip_outside]
        corrected_points[chosen_indices] = flip_points[flip_outside]
        corrected_normals[chosen_indices] = -normals_inside[flip_outside]
        correction_method[chosen_indices] = "flip_normal"

    unresolved_inside_indices = inside_indices[~flip_outside]
    search_corrected = 0
    if unresolved_inside_indices.size > 0:
        unresolved_surface = probe_surface_points[unresolved_inside_indices]
        unresolved_normals = probe_normals[unresolved_inside_indices]
        scale_candidates = np.array([0.125, 0.25, 0.5, 0.75, 0.875, 1.125, 1.25, 1.5, 2.0, 3.0, 4.0], dtype=np.float32)
        scale_candidates = scale_candidates[scale_candidates <= max_scale + 1e-6]
        if scale_candidates.size == 0 or scale_candidates[-1] < max_scale:
            scale_candidates = np.unique(np.concatenate([scale_candidates, np.array([max_scale], dtype=np.float32)]))
        scale_candidates = scale_candidates[np.abs(scale_candidates - 1.0) > 1e-6]

        best_found = np.zeros(unresolved_inside_indices.shape[0], dtype=bool)
        best_scale = np.ones(unresolved_inside_indices.shape[0], dtype=np.float32)
        best_signed_distance = np.full(unresolved_inside_indices.shape[0], -np.inf, dtype=np.float32)
        best_direction_sign = np.ones(unresolved_inside_indices.shape[0], dtype=np.float32)
        best_score = np.full(unresolved_inside_indices.shape[0], np.inf, dtype=np.float32)

        for direction_sign in (1.0, -1.0):
            directions = unresolved_normals * direction_sign
            for scale in scale_candidates:
                candidate_points = unresolved_surface + directions * (scale * offset_distance)
                candidate_signed_distance = _query_mesh_signed_distance(raycast_scene, candidate_points, nsamples=nsamples)
                candidate_outside = candidate_signed_distance > outside_tol
                candidate_score = np.full(unresolved_inside_indices.shape[0], np.inf, dtype=np.float32)
                candidate_score[candidate_outside] = np.abs(scale - 1.0)
                better = candidate_outside & (
                    (candidate_score < best_score - 1e-6)
                    | (
                        (np.abs(candidate_score - best_score) <= 1e-6)
                        & (candidate_signed_distance > best_signed_distance)
                    )
                )
                if np.any(better):
                    best_found[better] = True
                    best_scale[better] = np.float32(scale)
                    best_signed_distance[better] = candidate_signed_distance[better]
                    best_direction_sign[better] = np.float32(direction_sign)
                    best_score[better] = candidate_score[better]

        if np.any(best_found):
            found_indices = unresolved_inside_indices[best_found]
            found_surface = probe_surface_points[found_indices]
            found_directions = probe_normals[found_indices] * best_direction_sign[best_found][:, None]
            found_scales = best_scale[best_found]
            _, refined_points, _ = _binary_search_probe_outside(
                raycast_scene=raycast_scene,
                surface_points=found_surface,
                directions=found_directions,
                offset_distance=offset_distance,
                inside_scales=np.ones(found_indices.shape[0], dtype=np.float32),
                outside_scales=found_scales,
                nsamples=nsamples,
                search_steps=search_steps,
                outside_tol=outside_tol,
            )
            corrected_points[found_indices] = refined_points
            corrected_normals[found_indices] = found_directions
            correction_method[found_indices] = "line_search"
            search_corrected = int(found_indices.shape[0])

    final_signed_distance = _query_mesh_signed_distance(raycast_scene, corrected_points, nsamples=nsamples)
    remaining_inside = int(np.sum(final_signed_distance < -outside_tol))
    stats = {
        "enabled": True,
        "nsamples": int(nsamples),
        "outside_tol": float(outside_tol),
        "search_steps": int(search_steps),
        "search_max_scale": float(max_scale),
        "num_checked": int(probe_points.shape[0]),
        "initial_inside": int(np.sum(initial_inside_mask)),
        "corrected_by_flip": int(np.sum(correction_method == "flip_normal")),
        "corrected_by_search": int(search_corrected),
        "remaining_inside": remaining_inside,
        "initial_signed_distance": _summarize_array(initial_signed_distance),
        "final_signed_distance": _summarize_array(final_signed_distance),
    }
    return corrected_points, corrected_normals, stats, initial_signed_distance, final_signed_distance, correction_method


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


@torch.no_grad()
def _extract_view_surface_points(viewpoint_cam, gaussians: GaussianModel, pipe, background: torch.Tensor, args):
    render_pkg = render_sop_gbuffer(viewpoint_cam, gaussians, pipe, background)
    depth = render_pkg["depth_unbiased"]
    weight = render_pkg["weight"]
    normal = render_pkg["normal"]

    points_world = depth[..., None] * viewpoint_cam.rays_d_hw_unnormalized + viewpoint_cam.camera_center
    normals_world = F.normalize(normal.permute(1, 2, 0), dim=-1, eps=1e-6)

    valid = _build_object_valid_mask(
        depth=depth,
        weight=weight,
        gt_alpha_mask=getattr(viewpoint_cam, "gt_alpha_mask", None),
        weight_thresh=args.weight_thresh,
        mask_thresh=args.mask_thresh,
        object_filter_mode=args.object_filter_mode,
        mask_erosion_radius=args.mask_erosion_radius,
    )

    to_camera = F.normalize(viewpoint_cam.camera_center.view(1, 1, 3) - points_world, dim=-1, eps=1e-6)
    flip = (normals_world * to_camera).sum(dim=-1, keepdim=True) < 0.0
    normals_world = torch.where(flip, -normals_world, normals_world)

    valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if args.max_points_per_view > 0 and valid_idx.numel() > args.max_points_per_view:
        perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[: args.max_points_per_view]
        valid_idx = valid_idx[perm]

    points_flat = points_world.reshape(-1, 3)[valid_idx]
    normals_flat = normals_world.reshape(-1, 3)[valid_idx]
    return _to_numpy(points_flat).astype(np.float32), _to_numpy(normals_flat).astype(np.float32)


@torch.no_grad()
def _initialize_sop_phase1_from_mesh(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    dataset,
    output_root: Path,
    loaded_from: Dict[str, object],
    args,
) -> Dict[str, object]:
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    mesh_largest, mesh_stats = _extract_largest_mesh(
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        dataset=dataset,
        output_root=output_root,
        args=args,
    )
    if mesh_largest is None or not mesh_stats.get("enabled", False):
        raise RuntimeError(f"mesh_largest probe initialization requires successful mesh extraction, got: {mesh_stats.get('reason', 'unknown_error')}")

    requested_num_probes = int(args.target_num_probes)
    if requested_num_probes <= 0:
        raise RuntimeError("target_num_probes must be positive.")

    dense_sample_count = int(args.mesh_surface_sample_count)
    if dense_sample_count <= 0:
        dense_sample_count = max(requested_num_probes * 5, requested_num_probes)
    dense_sample_count = max(dense_sample_count, requested_num_probes)

    mesh_vertices = np.asarray(mesh_largest.vertices, dtype=np.float32)
    mesh_surface_points_all, mesh_surface_normals_all = _sample_points_from_mesh(mesh_largest, dense_sample_count, rng)
    object_extent = _compute_object_extent(
        mesh_vertices if mesh_vertices.shape[0] > 0 else mesh_surface_points_all,
        args.extent_mode,
    )
    visibility_occlusion_tolerance = (
        float(args.mesh_visibility_occlusion_tol)
        if args.mesh_visibility_occlusion_tol > 0.0
        else float(args.mesh_visibility_occlusion_tol_factor) * object_extent
    )
    mesh_surface_visibility_counts_all = np.zeros((mesh_surface_points_all.shape[0],), dtype=np.int32)
    mesh_surface_visible_mask = np.ones((mesh_surface_points_all.shape[0],), dtype=bool)
    visibility_cameras = []
    visibility_stats = {
        "enabled": False,
        "min_visible_views": int(args.mesh_visibility_min_views),
        "num_train_cameras": 0,
        "num_candidates_before_filter": int(mesh_surface_points_all.shape[0]),
        "num_candidates_after_filter": int(mesh_surface_points_all.shape[0]),
        "mode": "disabled",
    }

    if args.mesh_visibility_min_views > 0:
        visibility_cameras = _select_train_visibility_cameras(scene, args)
        if len(visibility_cameras) == 0:
            raise RuntimeError("mesh visibility filtering requested, but no train cameras are available.")

        visibility_raycast_scene = _build_mesh_raycast_scene(mesh_largest)
        if visibility_raycast_scene is None:
            raise RuntimeError("mesh visibility filtering requested, but the mesh raycasting scene could not be built.")

        mesh_surface_visibility_counts_all = _compute_mesh_first_hit_visibility_counts(
            points=mesh_surface_points_all,
            cameras=visibility_cameras,
            device=device,
            chunk_size=args.mesh_visibility_chunk_size,
            raycast_scene=visibility_raycast_scene,
            occlusion_tolerance=visibility_occlusion_tolerance,
        )
        mesh_surface_visible_mask = mesh_surface_visibility_counts_all >= int(args.mesh_visibility_min_views)
        if not np.any(mesh_surface_visible_mask):
            raise RuntimeError(
                "Mesh visibility filtering removed all surface samples. "
                "Try lowering --mesh_visibility_min_views, increasing --mesh_visibility_occlusion_tol, "
                "or increasing --mesh_surface_sample_count."
            )

        visibility_stats = {
            "enabled": True,
            "mode": "projection_and_mesh_first_hit",
            "min_visible_views": int(args.mesh_visibility_min_views),
            "view_stride": int(args.mesh_visibility_view_stride),
            "max_views": int(args.mesh_visibility_max_views),
            "occlusion_tolerance": float(visibility_occlusion_tolerance),
            "occlusion_tolerance_factor": float(args.mesh_visibility_occlusion_tol_factor),
            "num_train_cameras": int(len(visibility_cameras)),
            "num_candidates_before_filter": int(mesh_surface_points_all.shape[0]),
            "num_candidates_after_filter": int(np.sum(mesh_surface_visible_mask)),
            "visibility_count": _summarize_array(mesh_surface_visibility_counts_all.astype(np.float32)),
        }

    mesh_surface_points = mesh_surface_points_all[mesh_surface_visible_mask]
    mesh_surface_normals = mesh_surface_normals_all[mesh_surface_visible_mask]
    mesh_surface_visibility_counts = mesh_surface_visibility_counts_all[mesh_surface_visible_mask]

    target_num_probes = min(requested_num_probes, int(mesh_surface_points.shape[0]))
    probe_surface_indices = _farthest_point_sampling(mesh_surface_points, target_num_probes, device)
    probe_surface_points = mesh_surface_points[probe_surface_indices]
    probe_normals = _normalize_np(mesh_surface_normals[probe_surface_indices])

    offset_distance = float(args.offset_distance) if args.offset_distance > 0.0 else float(args.offset_scale) * object_extent
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0. Check extent_mode/offset_scale.")

    probe_points = probe_surface_points + probe_normals * offset_distance
    raw_probe_points = probe_points.copy()
    raw_probe_normals = probe_normals.copy()

    if args.disable_mesh_outside_check:
        mesh_outside_check_stats = {"enabled": False, "reason": "disabled_by_flag"}
        mesh_signed_distance_before = np.zeros((probe_points.shape[0],), dtype=np.float32)
        mesh_signed_distance_after = np.zeros((probe_points.shape[0],), dtype=np.float32)
        mesh_correction_method = np.full(probe_points.shape[0], "not_checked", dtype="<U24")
    else:
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
            mesh_largest=mesh_largest,
            offset_distance=offset_distance,
            args=args,
        )

    surface_colors = np.full_like(mesh_surface_points, 0.65, dtype=np.float32)
    probe_surface_colors = np.tile(np.array([[0.95, 0.62, 0.22]], dtype=np.float32), (probe_surface_points.shape[0], 1))
    probe_colors = np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1))

    _save_point_cloud(
        output_root / "mesh_largest_surface_samples_all.ply",
        mesh_surface_points_all,
        mesh_surface_normals_all,
        np.full_like(mesh_surface_points_all, 0.45, dtype=np.float32),
    )
    _save_point_cloud(output_root / "mesh_largest_surface_samples.ply", mesh_surface_points, mesh_surface_normals, surface_colors)
    _save_point_cloud(output_root / "surface_fused_clean.ply", mesh_surface_points, mesh_surface_normals, surface_colors)
    rejected_mask = ~mesh_surface_visible_mask
    if np.any(rejected_mask):
        rejected_colors = np.tile(np.array([[0.20, 0.20, 0.20]], dtype=np.float32), (int(np.sum(rejected_mask)), 1))
        _save_point_cloud(
            output_root / "mesh_largest_surface_samples_visibility_rejected.ply",
            mesh_surface_points_all[rejected_mask],
            mesh_surface_normals_all[rejected_mask],
            rejected_colors,
        )
    _save_point_cloud(output_root / "probe_surface_samples.ply", probe_surface_points, probe_normals, probe_surface_colors)
    _save_point_cloud(output_root / "probe_offset_points.ply", probe_points, probe_normals, probe_colors)
    if mesh_outside_check_stats.get("enabled", False) and mesh_outside_check_stats.get("initial_inside", 0) > 0:
        _save_point_cloud(
            output_root / "probe_offset_points_before_mesh_outside_check.ply",
            raw_probe_points,
            raw_probe_normals,
            probe_colors,
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

    mesh_stats.update(
        _export_mesh_with_probes(
            mesh_largest=mesh_largest,
            output_root=output_root,
            probe_points=probe_points,
            offset_distance=offset_distance,
            args=args,
        )
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
        mesh_surface_visibility_counts=mesh_surface_visibility_counts.astype(np.int32),
        mesh_surface_visibility_counts_all=mesh_surface_visibility_counts_all.astype(np.int32),
        mesh_surface_visible_mask=mesh_surface_visible_mask,
        probe_points_before_mesh_outside_check=raw_probe_points.astype(np.float32),
        probe_normals_before_mesh_outside_check=raw_probe_normals.astype(np.float32),
        mesh_signed_distance_before=mesh_signed_distance_before.astype(np.float32),
        mesh_signed_distance_after=mesh_signed_distance_after.astype(np.float32),
        mesh_correction_method=mesh_correction_method,
        offset_distance=np.array([offset_distance], dtype=np.float32),
    )

    summary = {
        "format": "irgs_sop_phase1_mesh_v1",
        "loaded_from": loaded_from,
        "output_root": str(output_root),
        "camera_set": args.camera_set,
        "probe_source": "mesh_largest",
        "mesh": mesh_stats,
        "mesh_visibility": visibility_stats,
        "mesh_outside_check": mesh_outside_check_stats,
        "mesh_surface_sample_count": int(dense_sample_count),
        "object_extent": float(object_extent),
        "extent_mode": args.extent_mode,
        "offset_scale": float(args.offset_scale),
        "offset_distance": float(offset_distance),
        "num_surface_points": int(mesh_surface_points.shape[0]),
        "num_probes": int(probe_points.shape[0]),
        "mesh_signed_distance_before": _summarize_array(mesh_signed_distance_before),
        "mesh_signed_distance_after": _summarize_array(mesh_signed_distance_after),
        "args": dict(vars(args)),
    }
    with open(output_root / "probe_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SOP-Phase1] Mesh vertices: {mesh_stats.get('largest_mesh_vertices', 0)}, triangles: {mesh_stats.get('largest_mesh_triangles', 0)}")
    if visibility_stats.get("enabled", False):
        print(
            "[SOP-Phase1] Mesh visibility filter: "
            f"kept={visibility_stats.get('num_candidates_after_filter', 0)}/"
            f"{visibility_stats.get('num_candidates_before_filter', 0)} "
            f"with min_views={visibility_stats.get('min_visible_views', 0)} "
            f"over train_views={visibility_stats.get('num_train_cameras', 0)} "
            f"mode={visibility_stats.get('mode', 'unknown')} "
            f"tol={visibility_stats.get('occlusion_tolerance', 0.0):.6g}"
        )
    print(f"[SOP-Phase1] Probe source: mesh_largest, probes: {probe_points.shape[0]}")
    print(f"[SOP-Phase1] Offset distance: {offset_distance:.6f}")
    if mesh_outside_check_stats.get("enabled", False):
        print(
            "[SOP-Phase1] Mesh outside check: "
            f"initial_inside={mesh_outside_check_stats.get('initial_inside', 0)}, "
            f"flip={mesh_outside_check_stats.get('corrected_by_flip', 0)}, "
            f"search={mesh_outside_check_stats.get('corrected_by_search', 0)}, "
            f"remaining_inside={mesh_outside_check_stats.get('remaining_inside', 0)}"
        )
    print(f"[SOP-Phase1] Output root: {output_root}")
    return summary


@torch.no_grad()
def initialize_sop_phase1(args):
    if not torch.cuda.is_available():
        raise RuntimeError("SOP phase1 initializer currently requires CUDA.")

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)
    _ensure_dir(Path(dataset.model_path))

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

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    output_root = _ensure_dir(Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "SOP_phase1")
    if args.probe_source == "mesh_largest":
        return _initialize_sop_phase1_from_mesh(
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            dataset=dataset,
            output_root=output_root,
            loaded_from=loaded_from,
            args=args,
        )

    cameras = _choose_cameras(scene, args.camera_set)
    cameras = cameras[:: max(1, args.view_stride)]
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    if len(cameras) == 0:
        raise RuntimeError("No cameras selected for SOP phase1 initialization.")

    all_points: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    for viewpoint_cam in cameras:
        pts, nrm = _extract_view_surface_points(viewpoint_cam, gaussians, pipe, background, args)
        if pts.shape[0] == 0:
            continue
        all_points.append(pts)
        all_normals.append(nrm)

    if len(all_points) == 0:
        raise RuntimeError("Surface fusion produced an empty point cloud. Lower thresholds or check the checkpoint.")

    fused_points = np.concatenate(all_points, axis=0).astype(np.float32)
    fused_normals = _normalize_np(np.concatenate(all_normals, axis=0).astype(np.float32))

    gaussian_points = _to_numpy(gaussians.get_xyz).astype(np.float32)
    object_extent = _compute_object_extent(gaussian_points, args.extent_mode)
    fusion_voxel_size = args.fusion_voxel_size if args.fusion_voxel_size > 0.0 else args.fusion_voxel_factor * object_extent
    fused_points, fused_normals = _voxel_downsample_numpy(fused_points, fused_normals, fusion_voxel_size)

    if fused_points.shape[0] == 0:
        raise RuntimeError("Voxel downsampling removed all fused points.")

    target_num_probes = min(int(args.target_num_probes), int(fused_points.shape[0]))
    if target_num_probes <= 0:
        raise RuntimeError("target_num_probes must be positive.")

    probe_surface_indices = _farthest_point_sampling(fused_points, target_num_probes, device=torch.device("cuda"))
    probe_surface_points = fused_points[probe_surface_indices]
    probe_normals = _normalize_np(fused_normals[probe_surface_indices])

    offset_distance = float(args.offset_distance) if args.offset_distance > 0.0 else float(args.offset_scale) * object_extent
    if offset_distance <= 0.0:
        raise RuntimeError("Computed offset_distance <= 0. Check extent_mode/offset_scale.")
    probe_points = probe_surface_points + probe_normals * offset_distance

    surface_colors = np.full_like(fused_points, 0.65, dtype=np.float32)
    probe_colors = np.tile(np.array([[0.85, 0.28, 0.25]], dtype=np.float32), (probe_points.shape[0], 1))

    _save_point_cloud(output_root / "surface_fused_clean.ply", fused_points, fused_normals, surface_colors)
    _save_point_cloud(output_root / "probe_surface_samples.ply", probe_surface_points, probe_normals, probe_colors)
    _save_point_cloud(output_root / "probe_offset_points.ply", probe_points, probe_normals, probe_colors)

    np.savez(
        output_root / "probe_init_data.npz",
        surface_points=fused_points.astype(np.float32),
        surface_normals=fused_normals.astype(np.float32),
        probe_surface_points=probe_surface_points.astype(np.float32),
        probe_points=probe_points.astype(np.float32),
        probe_normals=probe_normals.astype(np.float32),
        probe_surface_indices=probe_surface_indices.astype(np.int64),
        offset_distance=np.array([offset_distance], dtype=np.float32),
    )

    summary = {
        "format": "irgs_sop_phase1_v1",
        "loaded_from": loaded_from,
        "camera_set": args.camera_set,
        "num_selected_cameras": len(cameras),
        "fusion_voxel_size": float(fusion_voxel_size),
        "object_extent": float(object_extent),
        "offset_distance": float(offset_distance),
        "num_surface_points": int(fused_points.shape[0]),
        "num_probes": int(probe_points.shape[0]),
        "args": dict(vars(args)),
    }
    with open(output_root / "probe_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SOP-Phase1] Fused surface points: {fused_points.shape[0]}")
    print(f"[SOP-Phase1] Probes: {probe_points.shape[0]}")
    print(f"[SOP-Phase1] Offset distance: {offset_distance:.6f}")
    print(f"[SOP-Phase1] Output root: {output_root}")
    return summary


def _build_parser():
    parser = ArgumentParser(description="Simplified SOP phase1 initializer for COMGS_IRGS")
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
    parser.add_argument("--probe_source", choices=["mesh_largest", "surface_fusion"], default="mesh_largest")
    parser.add_argument("--mesh_voxel_size", default=-1.0, type=float)
    parser.add_argument("--mesh_depth_trunc", default=-1.0, type=float)
    parser.add_argument("--mesh_sdf_trunc", default=-1.0, type=float)
    parser.add_argument("--mesh_res", default=1024, type=int)
    parser.add_argument("--mesh_num_cluster", default=1, type=int)
    parser.add_argument("--mesh_surface_sample_count", default=0, type=int)
    parser.add_argument("--mesh_probe_radius", default=-1.0, type=float)
    parser.add_argument("--mesh_probe_resolution", default=5, type=int)
    parser.add_argument("--mesh_max_probe_spheres", default=0, type=int)
    parser.add_argument("--mesh_visibility_min_views", default=1, type=int)
    parser.add_argument("--mesh_visibility_view_stride", default=1, type=int)
    parser.add_argument("--mesh_visibility_max_views", default=0, type=int)
    parser.add_argument("--mesh_visibility_chunk_size", default=65536, type=int)
    parser.add_argument("--mesh_visibility_occlusion_tol", default=0.0, type=float)
    parser.add_argument("--mesh_visibility_occlusion_tol_factor", default=0.002, type=float)
    parser.add_argument("--disable_mesh_outside_check", action="store_true")
    parser.add_argument("--mesh_outside_check_nsamples", default=11, type=int)
    parser.add_argument("--mesh_outside_search_steps", default=10, type=int)
    parser.add_argument("--mesh_outside_search_max_scale", default=4.0, type=float)
    parser.add_argument("--weight_thresh", default=1e-4, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument("--object_filter_mode", choices=["weight_only", "mask_only", "weight_or_mask", "weight_and_mask"], default="weight_and_mask")
    parser.add_argument("--mask_erosion_radius", default=4, type=int)
    parser.add_argument("--fusion_voxel_factor", default=0.002, type=float)
    parser.add_argument("--fusion_voxel_size", default=0.0, type=float)
    parser.add_argument("--extent_mode", choices=["bbox_diagonal", "max_side"], default="bbox_diagonal")
    parser.add_argument("--target_num_probes", default=5000, type=int)
    parser.add_argument("--offset_scale", "--offset_factor", dest="offset_scale", default=0.01, type=float)
    parser.add_argument("--offset_distance", default=0.0, type=float)
    parser.add_argument("--max_probe_normals_vis", default=400, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = get_combined_args(parser)
    if not getattr(args, "model_path", ""):
        raise RuntimeError("SOP phase1 initializer requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("SOP phase1 initializer requires --source_path/-s, or a cfg_args under --model_path.")
    _ensure_dir(Path(args.model_path))
    with open(Path(args.model_path) / "cfg_args", "w") as cfg_log_f:
        cfg_log_f.write(str(args))
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialize_sop_phase1(args)


if __name__ == "__main__":
    main()
