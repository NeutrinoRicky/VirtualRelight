from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_multitarget
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.point_utils import depths_to_points
from utils.render_utils import save_img_u8
from SOP.phase1_initializer import _build_object_valid_mask, _load_gaussians_from_checkpoint, _save_point_cloud


def _choose_cameras(scene: Scene, camera_set: str) -> List:
    if camera_set == "train":
        return list(scene.getTrainCameras())
    if camera_set == "test":
        return list(scene.getTestCameras())
    if camera_set == "all":
        return list(scene.getTrainCameras()) + list(scene.getTestCameras())
    raise ValueError(f"Unknown camera_set: {camera_set}")


def _normalize_single_channel_for_vis(x: torch.Tensor, valid_mask: torch.Tensor | None = None, eps: float = 1e-6) -> np.ndarray:
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = valid_mask.bool() if valid_mask is not None else torch.ones_like(y, dtype=torch.bool)
    if int(mask.sum().item()) > 0:
        values = y[mask]
        vmin = values.min()
        vmax = values.max()
    else:
        vmin = y.min()
        vmax = y.max()
    vis = (y - vmin) / torch.clamp(vmax - vmin, min=eps)
    vis = torch.clamp(vis, 0.0, 1.0)
    return vis.cpu().numpy()


def _erode_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = mask.bool()
    if radius <= 0:
        return mask

    inv_mask = (~mask).float().unsqueeze(0).unsqueeze(0)
    kernel = radius * 2 + 1
    eroded = F.max_pool2d(inv_mask, kernel_size=kernel, stride=1, padding=radius)[0, 0] < 0.5
    return eroded


@torch.no_grad()
def _select_viewpoint(cameras: List, frame_name: str, camera_index: int):
    if frame_name:
        for cam in cameras:
            if cam.image_name == frame_name:
                return cam
        raise ValueError(f"frame_name={frame_name} not found in selected camera set")

    if camera_index < 0:
        raise ValueError("Please provide either --frame_name or --camera_index")
    if camera_index >= len(cameras):
        raise IndexError(f"camera_index={camera_index} out of range for {len(cameras)} cameras")
    return cameras[camera_index]


@torch.no_grad()
def debug_single_view(dataset, pipe, opt, args):
    if not torch.cuda.is_available():
        raise RuntimeError("This debug script requires CUDA because it reuses the Gaussian renderer.")

    gaussians = GaussianModel(dataset.sh_degree)
    checkpoint_path = getattr(args, "checkpoint", "")
    if checkpoint_path:
        scene = Scene(dataset, gaussians, shuffle=False)
        _load_gaussians_from_checkpoint(gaussians, opt, checkpoint_path)
    else:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    cameras = _choose_cameras(scene, args.camera_set)
    viewpoint = _select_viewpoint(cameras, args.frame_name, args.camera_index)

    render_pkg = render_multitarget(viewpoint, gaussians, pipe, background)
    depth = render_pkg["depth_unbiased"]
    weight = render_pkg["weight"]
    normal = render_pkg["normal"]
    render_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
    albedo = torch.clamp(render_pkg["albedo"], 0.0, 1.0)
    gt_rgb = torch.clamp(viewpoint.original_image[0:3], 0.0, 1.0)

    points_world = depths_to_points(viewpoint, depth).reshape(depth.shape[1], depth.shape[2], 3)
    normals_world = F.normalize(normal.permute(1, 2, 0), dim=-1, eps=1e-6)
    gt_alpha_mask = getattr(viewpoint, "gt_alpha_mask", None)
    valid_mask, filter_stats = _build_object_valid_mask(
        depth=depth,
        weight=weight,
        normals_world=normals_world,
        gt_alpha_mask=gt_alpha_mask,
        weight_thresh=args.weight_thresh,
        mask_thresh=args.mask_thresh,
        object_filter_mode=args.object_filter_mode,
    )

    if gt_alpha_mask is not None:
        object_mask = gt_alpha_mask[0] > args.mask_thresh
        object_mask_source = "gt_alpha_mask"
    else:
        object_mask = valid_mask.clone()
        object_mask_source = "valid_mask_fallback"

    core_mask = _erode_binary_mask(object_mask, args.mask_erosion_radius)
    final_mask = valid_mask & core_mask

    cam_center = viewpoint.camera_center.view(1, 1, 3)
    to_camera = F.normalize(cam_center - points_world, dim=-1, eps=1e-6)
    flip = (normals_world * to_camera).sum(dim=-1, keepdim=True) < 0.0
    normals_world = torch.where(flip, -normals_world, normals_world)

    final_mask_flat = final_mask.reshape(-1).bool()
    raw_valid_count = int(valid_mask.reshape(-1).sum().item())
    core_count = int(final_mask_flat.sum().item())
    if core_count == 0:
        raise RuntimeError("No core-region pixels found for the selected view. Try a smaller erosion radius or check mask loading.")

    if args.color_source == "gt":
        color_map = gt_rgb.permute(1, 2, 0)
    elif args.color_source == "render":
        color_map = render_rgb.permute(1, 2, 0)
    elif args.color_source == "albedo":
        color_map = albedo.permute(1, 2, 0)
    else:
        raise ValueError(f"Unknown color_source: {args.color_source}")

    if color_map.device != final_mask_flat.device:
        color_map = color_map.to(final_mask_flat.device)

    points = points_world.reshape(-1, 3)[final_mask_flat].detach().cpu().numpy().astype(np.float32)
    normals = normals_world.reshape(-1, 3)[final_mask_flat].detach().cpu().numpy().astype(np.float32)
    colors = color_map.reshape(-1, 3)[final_mask_flat].detach().cpu().numpy().astype(np.float32)

    output_root = Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "SOP_single_view_debug" / viewpoint.image_name
    output_root.mkdir(parents=True, exist_ok=True)

    _save_point_cloud(output_root / "core_reprojected_points.ply", points, normals, colors)
    _save_point_cloud(output_root / "valid_reprojected_points.ply", points, normals, colors)

    valid_mask_np = valid_mask.detach().cpu().numpy().astype(np.float32)
    object_mask_np = object_mask.detach().cpu().numpy().astype(np.float32)
    core_mask_np = core_mask.detach().cpu().numpy().astype(np.float32)
    final_mask_np = final_mask.detach().cpu().numpy().astype(np.float32)
    weight_vis = _normalize_single_channel_for_vis(weight[0], valid_mask=(weight[0] > 0.0))
    depth_vis = _normalize_single_channel_for_vis(depth[0], valid_mask=final_mask)

    valid_mask_rgb = np.repeat(valid_mask_np[..., None], 3, axis=-1)
    object_mask_rgb = np.repeat(object_mask_np[..., None], 3, axis=-1)
    core_mask_rgb = np.repeat(core_mask_np[..., None], 3, axis=-1)
    final_mask_rgb = np.repeat(final_mask_np[..., None], 3, axis=-1)
    weight_rgb = np.repeat(weight_vis[..., None], 3, axis=-1)
    depth_rgb = np.repeat(depth_vis[..., None], 3, axis=-1)
    valid_overlay = gt_rgb.permute(1, 2, 0).detach().cpu().numpy().copy()
    valid_overlay = np.clip(valid_overlay * 0.35 + valid_mask_rgb * np.array([1.0, 0.15, 0.15], dtype=np.float32) * 0.65, 0.0, 1.0)
    core_overlay = gt_rgb.permute(1, 2, 0).detach().cpu().numpy().copy()
    core_overlay = np.clip(core_overlay * 0.35 + final_mask_rgb * np.array([0.15, 1.0, 0.2], dtype=np.float32) * 0.65, 0.0, 1.0)

    save_img_u8(gt_rgb.permute(1, 2, 0).detach().cpu().numpy(), str(output_root / "gt_rgb.png"))
    save_img_u8(render_rgb.permute(1, 2, 0).detach().cpu().numpy(), str(output_root / "render_rgb.png"))
    save_img_u8(albedo.permute(1, 2, 0).detach().cpu().numpy(), str(output_root / "albedo.png"))
    save_img_u8(object_mask_rgb, str(output_root / "object_mask.png"))
    save_img_u8(core_mask_rgb, str(output_root / "core_mask.png"))
    save_img_u8(valid_mask_rgb, str(output_root / "valid_mask.png"))
    save_img_u8(final_mask_rgb, str(output_root / "final_core_mask.png"))
    save_img_u8(valid_overlay, str(output_root / "valid_mask_overlay.png"))
    save_img_u8(core_overlay, str(output_root / "core_mask_overlay.png"))
    save_img_u8(weight_rgb, str(output_root / "weight_vis.png"))
    save_img_u8(depth_rgb, str(output_root / "depth_vis.png"))

    summary = {
        "image_name": viewpoint.image_name,
        "camera_set": args.camera_set,
        "color_source": args.color_source,
        "weight_thresh": float(args.weight_thresh),
        "mask_thresh": float(args.mask_thresh),
        "mask_erosion_radius": int(args.mask_erosion_radius),
        "object_mask_source": object_mask_source,
        "object_filter_mode": args.object_filter_mode,
        "valid_pixels_before_core": raw_valid_count,
        "object_mask_pixels": int(object_mask.reshape(-1).sum().item()),
        "core_mask_pixels": int(core_mask.reshape(-1).sum().item()),
        "valid_pixels": core_count,
        "total_pixels": int(valid_mask.numel()),
        "point_cloud_path": str(output_root / "core_reprojected_points.ply"),
        "filter_stats": filter_stats,
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SOP-SingleView] image={viewpoint.image_name}")
    print(f"[SOP-SingleView] valid_pixels_before_core={raw_valid_count}/{int(valid_mask.numel())}")
    print(f"[SOP-SingleView] core_pixels={core_count}/{int(valid_mask.numel())}")
    print(f"[SOP-SingleView] output_root={output_root}")



def _build_parser():
    parser = ArgumentParser(description="Debug a single view by exporting only valid-pixel backprojected points")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--camera_set", choices=["train", "test", "all"], default="train")
    parser.add_argument("--frame_name", default="", type=str)
    parser.add_argument("--camera_index", default=-1, type=int)
    parser.add_argument("--object_filter_mode", choices=["weight_only", "mask_only", "weight_or_mask", "weight_and_mask"], default="weight_and_mask")
    parser.add_argument("--weight_thresh", default=0.1, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument("--mask_erosion_radius", default=3, type=int)
    parser.add_argument("--color_source", choices=["gt", "render", "albedo"], default="gt")
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline, optimization



def main() -> None:
    parser, model, pipeline, optimization = _build_parser()
    args = get_combined_args(parser)
    safe_state(args.quiet)
    debug_single_view(model.extract(args), pipeline.extract(args), optimization.extract(args), args)


if __name__ == "__main__":
    main()
