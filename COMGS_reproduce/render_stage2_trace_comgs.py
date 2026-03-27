import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_multitarget
from scene import Scene, GaussianModel
from utils.deferred_pbr_comgs import (
    LatLongEnvMap,
    compute_view_directions,
    integrate_incident_radiance,
    recover_shading_points,
    sample_hemisphere_hammersley,
)
from utils.general_utils import safe_state
from utils.tracing_comgs import TraceBackendConfig, build_trace_backend



def _normalize_single_channel_for_vis(x, valid_mask=None, eps=1e-6):
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        mask = valid_mask.bool()
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    if int(mask.sum().item()) > 0:
        values = y[mask]
        vmin = values.min()
        vmax = values.max()
    else:
        vmin = y.min()
        vmax = y.max()

    denom = torch.clamp(vmax - vmin, min=eps)
    vis = (y - vmin) / denom
    if valid_mask is not None:
        vis = vis * mask.float()
    return torch.clamp(vis, 0.0, 1.0)



def _tonemap_for_vis(x):
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(y / (1.0 + y), 0.0, 1.0)



def _get_gt_alpha_mask(viewpoint_cam, ref_tensor: torch.Tensor):
    gt_mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
    if gt_mask is None:
        return None

    mask = gt_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] != 1:
        mask = mask[:1]

    if mask.shape[-2:] != ref_tensor.shape[-2:]:
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=ref_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return torch.clamp(mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype), 0.0, 1.0)



def _build_export_masks(viewpoint_cam, render_pkg, valid_mask: torch.Tensor, export_mask_mode: str):
    render_alpha = torch.clamp(render_pkg["weight"].detach(), 0.0, 1.0)
    render_mask = valid_mask.unsqueeze(0).float()
    gt_alpha = _get_gt_alpha_mask(viewpoint_cam, render_alpha)

    if export_mask_mode == "gt" and gt_alpha is not None:
        export_alpha = gt_alpha
    elif export_mask_mode == "intersect" and gt_alpha is not None:
        export_alpha = torch.clamp(render_alpha * gt_alpha, 0.0, 1.0)
    else:
        export_alpha = render_alpha

    export_mask = (export_alpha > 0.5).float()
    return {
        "render_alpha": render_alpha,
        "render_mask": render_mask,
        "gt_alpha": gt_alpha,
        "export_alpha": export_alpha,
        "export_mask": export_mask,
    }



def load_stage2_trace_checkpoint(gaussians: GaussianModel, opt, checkpoint_path: str):
    payload = torch.load(checkpoint_path)
    if not isinstance(payload, dict) or payload.get("format") != "comgs_stage2_trace_v1":
        raise RuntimeError(f"{checkpoint_path} is not a valid Stage2 Trace checkpoint")

    gaussians.restore(payload["gaussians"], opt)
    envmap = LatLongEnvMap.from_capture(payload["envmap"]).cuda()
    iteration = int(payload.get("iteration", -1))
    saved_args = payload.get("args", {})
    return payload, envmap, iteration, saved_args


@torch.no_grad()
def render_stage2_trace_view(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    envmap,
    trace_backend,
    num_shading_samples: int,
    secondary_num_samples: int,
    trace_bias: float,
    trace_chunk_size: int,
    export_mask_mode: str,
    randomized_samples: bool = False,
    weight_threshold: float = 1e-4,
):
    render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
    points, valid_mask = recover_shading_points(
        view=viewpoint_cam,
        depth_unbiased=render_pkg["depth_unbiased"],
        weight=render_pkg["weight"],
        weight_threshold=weight_threshold,
    )

    height, width = valid_mask.shape
    flat_valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).squeeze(1)

    pbr_flat = render_pkg["render"].permute(1, 2, 0).reshape(-1, 3).clone()
    direct_flat = pbr_flat.new_zeros((height * width, 3))
    indirect_flat = pbr_flat.new_zeros((height * width, 3))
    occlusion_flat = pbr_flat.new_zeros((height * width, 1))

    flat_points = points.reshape(-1, 3)
    flat_normals = F.normalize(render_pkg["normal"].permute(1, 2, 0).reshape(-1, 3), dim=-1, eps=1e-6)
    flat_albedo = render_pkg["albedo"].permute(1, 2, 0).reshape(-1, 3)
    flat_roughness = render_pkg["roughness"].permute(1, 2, 0).reshape(-1, 1)
    flat_metallic = render_pkg["metallic"].permute(1, 2, 0).reshape(-1, 1)

    for chunk_idx in torch.split(flat_valid_idx, trace_chunk_size):
        pts = flat_points[chunk_idx]
        nrm = flat_normals[chunk_idx]
        albedo = flat_albedo[chunk_idx]
        roughness = flat_roughness[chunk_idx]
        metallic = flat_metallic[chunk_idx]
        viewdirs = compute_view_directions(pts, viewpoint_cam.camera_center)

        lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(
            normals=nrm,
            num_samples=num_shading_samples,
            randomized=randomized_samples,
        )
        ray_origins = pts[:, None, :] + lightdirs * trace_bias
        trace_outputs = trace_backend.trace(
            ray_origins=ray_origins,
            ray_directions=lightdirs,
            envmap=envmap,
            secondary_num_samples=secondary_num_samples,
            randomized_secondary=randomized_samples,
            camera_center=viewpoint_cam.camera_center,
        )

        direct_radiance = envmap(lightdirs)
        # Keep render-time incident lighting consistent with stage2 training:
        # L_i = (1 - O) * L_dir + L_ind.
        incident_radiance = (1.0 - trace_outputs["occlusion"]) * direct_radiance + trace_outputs["incident_radiance"]
        pbr_rgb, _aux = integrate_incident_radiance(
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            normals=nrm,
            viewdirs=viewdirs,
            lightdirs=lightdirs,
            incident_radiance=incident_radiance,
            sample_solid_angle=sample_solid_angle,
        )

        pbr_flat[chunk_idx] = pbr_rgb
        direct_flat[chunk_idx] = direct_radiance.mean(dim=1)
        indirect_flat[chunk_idx] = trace_outputs["incident_radiance"].mean(dim=1)
        occlusion_flat[chunk_idx] = trace_outputs["occlusion"].mean(dim=1)

    pbr_render_raw = pbr_flat.reshape(height, width, 3).permute(2, 0, 1)
    masks = _build_export_masks(viewpoint_cam, render_pkg, valid_mask, export_mask_mode)
    pbr_render_masked = pbr_render_raw * masks["export_alpha"]

    return {
        "pbr_render": pbr_render_masked,
        "pbr_render_raw": pbr_render_raw,
        "trace_direct": direct_flat.reshape(height, width, 3).permute(2, 0, 1),
        "trace_indirect": indirect_flat.reshape(height, width, 3).permute(2, 0, 1),
        "trace_occlusion": occlusion_flat.reshape(height, width, 1).permute(2, 0, 1),
        "valid_mask": valid_mask.unsqueeze(0),
        "masks": masks,
        "render_pkg": render_pkg,
    }


@torch.no_grad()
def save_stage2_trace_outputs(output_dir: str, view_name: str, stage2_view_pkg: dict):
    os.makedirs(output_dir, exist_ok=True)
    render_pkg = stage2_view_pkg["render_pkg"]
    masks = stage2_view_pkg["masks"]
    export_alpha = masks["export_alpha"]
    export_mask = masks["export_mask"]
    valid = export_mask > 0

    depth_raw = render_pkg["depth_unbiased"].detach().cpu().numpy()
    np.save(os.path.join(output_dir, f"{view_name}_depth_unbiased.npy"), depth_raw)

    rgb_raw = torch.clamp(stage2_view_pkg["pbr_render_raw"], 0.0, 1.0)
    rgb_masked = torch.clamp(stage2_view_pkg["pbr_render"], 0.0, 1.0)
    rgb_on_white = torch.clamp(rgb_masked + (1.0 - export_alpha), 0.0, 1.0)

    albedo_raw = torch.clamp(render_pkg["albedo"], 0.0, 1.0)
    roughness_raw = torch.clamp(render_pkg["roughness"], 0.0, 1.0)
    metallic_raw = torch.clamp(render_pkg["metallic"], 0.0, 1.0)
    normal_raw = torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0)

    albedo = albedo_raw * export_alpha
    roughness = roughness_raw * export_alpha
    metallic = metallic_raw * export_alpha
    normal = normal_raw * export_alpha
    depth_vis = _normalize_single_channel_for_vis(render_pkg["depth_unbiased"], valid_mask=valid)
    trace_direct = _tonemap_for_vis(stage2_view_pkg["trace_direct"]) * export_alpha
    trace_indirect = _tonemap_for_vis(stage2_view_pkg["trace_indirect"]) * export_alpha
    trace_occlusion = torch.clamp(stage2_view_pkg["trace_occlusion"], 0.0, 1.0) * export_alpha

    torchvision.utils.save_image(rgb_masked, os.path.join(output_dir, f"{view_name}_rgb.png"))
    torchvision.utils.save_image(rgb_raw, os.path.join(output_dir, f"{view_name}_rgb_raw.png"))
    torchvision.utils.save_image(rgb_masked, os.path.join(output_dir, f"{view_name}_rgb_masked.png"))
    torchvision.utils.save_image(rgb_on_white, os.path.join(output_dir, f"{view_name}_rgb_on_white.png"))

    torchvision.utils.save_image(export_alpha, os.path.join(output_dir, f"{view_name}_alpha.png"))
    torchvision.utils.save_image(export_mask, os.path.join(output_dir, f"{view_name}_mask.png"))
    torchvision.utils.save_image(masks["render_alpha"], os.path.join(output_dir, f"{view_name}_render_alpha.png"))
    torchvision.utils.save_image(masks["render_mask"], os.path.join(output_dir, f"{view_name}_render_mask.png"))
    if masks["gt_alpha"] is not None:
        torchvision.utils.save_image(masks["gt_alpha"], os.path.join(output_dir, f"{view_name}_gt_alpha.png"))

    torchvision.utils.save_image(albedo, os.path.join(output_dir, f"{view_name}_albedo.png"))
    torchvision.utils.save_image(albedo_raw, os.path.join(output_dir, f"{view_name}_albedo_raw.png"))
    torchvision.utils.save_image(roughness, os.path.join(output_dir, f"{view_name}_roughness.png"))
    torchvision.utils.save_image(roughness_raw, os.path.join(output_dir, f"{view_name}_roughness_raw.png"))
    torchvision.utils.save_image(metallic, os.path.join(output_dir, f"{view_name}_metallic.png"))
    torchvision.utils.save_image(metallic_raw, os.path.join(output_dir, f"{view_name}_metallic_raw.png"))
    torchvision.utils.save_image(normal, os.path.join(output_dir, f"{view_name}_normal.png"))
    torchvision.utils.save_image(normal_raw, os.path.join(output_dir, f"{view_name}_normal_raw.png"))
    torchvision.utils.save_image(depth_vis, os.path.join(output_dir, f"{view_name}_depth.png"))
    torchvision.utils.save_image(trace_direct, os.path.join(output_dir, f"{view_name}_trace_direct.png"))
    torchvision.utils.save_image(trace_indirect, os.path.join(output_dir, f"{view_name}_trace_indirect.png"))
    torchvision.utils.save_image(trace_occlusion, os.path.join(output_dir, f"{view_name}_trace_occlusion.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Render images from a Stage2 Trace checkpoint")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "both"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--trace_backend", type=str, default="auto", choices=["auto", "irgs", "irgs_adapter", "irgs_native", "open3d", "open3d_mesh"])
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--secondary_num_samples", type=int, default=16)
    parser.add_argument("--trace_bias", type=float, default=1e-3)
    parser.add_argument("--trace_chunk_size", type=int, default=2048)
    parser.add_argument("--trace_voxel_size", type=float, default=0.004)
    parser.add_argument("--trace_sdf_trunc", type=float, default=0.02)
    parser.add_argument("--trace_depth_trunc", type=float, default=0.0)
    parser.add_argument("--trace_mask_background", action="store_true", default=True)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=True)
    parser.add_argument("--export_mask_mode", type=str, default="render", choices=["render", "gt", "intersect"])
    args = get_combined_args(parser)

    safe_state(args.quiet)
    checkpoint_path = args.checkpoint or os.path.join(args.model_path, "object_step2_trace.ckpt")
    print("Rendering stage2 trace views for " + args.model_path)
    print("Loading checkpoint: " + checkpoint_path)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    payload, envmap, iteration, saved_args = load_stage2_trace_checkpoint(gaussians, opt_args, checkpoint_path)
    saved_trace_backend = payload.get("trace_backend") or saved_args.get("trace_backend")
    if saved_trace_backend is not None:
        print("Checkpoint trace backend:", saved_trace_backend)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    trace_config = TraceBackendConfig(
        backend=args.trace_backend,
        trace_bias=args.trace_bias,
        secondary_num_samples=args.secondary_num_samples,
        open3d_voxel_size=args.trace_voxel_size,
        open3d_sdf_trunc=args.trace_sdf_trunc,
        open3d_depth_trunc=args.trace_depth_trunc,
        open3d_mask_background=args.trace_mask_background,
    )
    trace_backend = build_trace_backend(trace_config, scene, gaussians, pipe, background)

    split_to_cameras = []
    if args.split in ("train", "both"):
        split_to_cameras.append(("train", scene.getTrainCameras()))
    if args.split in ("test", "both"):
        split_to_cameras.append(("test", scene.getTestCameras()))

    for split_name, cameras in split_to_cameras:
        if cameras is None or len(cameras) == 0:
            print(f"No {split_name} cameras found, skip.")
            continue

        output_dir = os.path.join(args.model_path, "stage2_trace_render", split_name, f"ours_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        torchvision.utils.save_image(envmap.visualization(), os.path.join(output_dir, "envmap.png"))
        torch.save(envmap.capture(), os.path.join(output_dir, "envmap.pt"))

        for viewpoint_cam in tqdm(cameras, desc=f"Rendering {split_name} views"):
            stage2_view_pkg = render_stage2_trace_view(
                viewpoint_cam=viewpoint_cam,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                envmap=envmap,
                trace_backend=trace_backend,
                num_shading_samples=args.num_shading_samples,
                secondary_num_samples=args.secondary_num_samples,
                trace_bias=args.trace_bias,
                trace_chunk_size=args.trace_chunk_size,
                export_mask_mode=args.export_mask_mode,
                randomized_samples=not args.disable_sample_jitter,
            )
            save_stage2_trace_outputs(output_dir, viewpoint_cam.image_name, stage2_view_pkg)

        print(f"Saved {split_name} renders to {output_dir}")
