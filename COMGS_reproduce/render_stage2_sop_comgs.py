import os
import time
import json
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
    compute_view_directions,
    integrate_incident_radiance,
    load_envmap_capture_as_octahedral,
    recover_shading_points,
    sample_hemisphere_hammersley,
)
from utils.general_utils import safe_state
from utils.sop_utils import (
    build_view_sop_neighbor_cache,
    query_sops_directional,
    save_sop_texture_previews,
    slice_view_sop_neighbor_cache_rows,
)


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _composite_on_white(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * alpha + (1.0 - alpha), 0.0, 1.0)


def _cuda_checkpoint_map_location():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda")


def load_stage2_sop_checkpoint(gaussians: GaussianModel, opt, checkpoint_path: str):
    payload = torch.load(checkpoint_path, map_location=_cuda_checkpoint_map_location())
    if not isinstance(payload, dict) or payload.get("format") != "comgs_stage2_sop_v1":
        raise RuntimeError(f"{checkpoint_path} is not a valid Stage2 SOP checkpoint")
    if "sop" not in payload:
        raise RuntimeError(f"{checkpoint_path} does not contain trained SOP payload")

    gaussians.restore(payload["gaussians"], opt)
    envmap = load_envmap_capture_as_octahedral(payload["envmap"]).cuda()
    sop_payload = payload["sop"]
    required = ["probe_xyz", "probe_normal", "probe_lin_tex", "probe_occ_tex"]
    missing = [key for key in required if key not in sop_payload]
    if missing:
        raise RuntimeError(f"SOP payload is missing required fields: {missing}")

    sop_tensors = {
        "probe_xyz": sop_payload["probe_xyz"].float().cuda(),
        "probe_normal": F.normalize(sop_payload["probe_normal"].float().cuda(), dim=-1, eps=1e-6),
        "probe_lin_tex": torch.clamp(sop_payload["probe_lin_tex"].float().cuda(), min=0.0),
        "probe_occ_tex": torch.clamp(sop_payload["probe_occ_tex"].float().cuda(), 0.0, 1.0),
    }
    for key in ("probe_albedo_tex", "probe_roughness_tex", "probe_metallic_tex", "oct_dirs"):
        if key in sop_payload and torch.is_tensor(sop_payload[key]):
            sop_tensors[key] = sop_payload[key].float().cuda()
    iteration = int(payload.get("iteration", -1))
    saved_args = payload.get("args", {})
    return payload, envmap, sop_tensors, iteration, saved_args


@torch.no_grad()
def render_stage2_sop_view(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    envmap,
    sop_tensors,
    num_shading_samples: int,
    query_chunk_size: int,
    sop_query_radius: float,
    sop_query_topk: int,
    export_mask_mode: str,
    profile_efficiency: bool = False,
    randomized_samples: bool = False,
    weight_threshold: float = 1e-4,
):
    timing = {
        "gbuffer_render_sec": 0.0,
        "recover_points_sec": 0.0,
        "neighbor_cache_build_sec": 0.0,
        "light_sample_sec": 0.0,
        "env_lookup_sec": 0.0,
        "sop_query_sec": 0.0,
        "sop_query_knn_sec": 0.0,
        "sop_query_prep_sec": 0.0,
        "sop_query_sample_sec": 0.0,
        "sop_query_blend_sec": 0.0,
        "sop_query_other_sec": 0.0,
        "brdf_integrate_sec": 0.0,
        "shading_other_sec": 0.0,
        "sop_query_shading_sec": 0.0,
        "render_core_sec": 0.0,
    }

    if profile_efficiency:
        _sync_cuda()
        gbuffer_t0 = time.perf_counter()
    render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
    if profile_efficiency:
        _sync_cuda()
        gbuffer_t1 = time.perf_counter()
        timing["gbuffer_render_sec"] = gbuffer_t1 - gbuffer_t0

    if profile_efficiency:
        _sync_cuda()
        shading_t0 = time.perf_counter()
        recover_t0 = time.perf_counter()
    points, valid_mask = recover_shading_points(
        view=viewpoint_cam,
        depth_unbiased=render_pkg["depth_unbiased"],
        weight=render_pkg["weight"],
        weight_threshold=weight_threshold,
    )
    if profile_efficiency:
        _sync_cuda()
        recover_t1 = time.perf_counter()
        timing["recover_points_sec"] = recover_t1 - recover_t0
        cache_build_t0 = time.perf_counter()
    view_neighbor_cache = build_view_sop_neighbor_cache(
        points=points,
        valid_mask=valid_mask,
        probe_xyz=sop_tensors["probe_xyz"],
        probe_normal=sop_tensors["probe_normal"],
        radius=float(sop_query_radius) if sop_query_radius and sop_query_radius > 0.0 else None,
        topk=sop_query_topk,
        chunk_size=query_chunk_size,
    )
    if profile_efficiency:
        _sync_cuda()
        cache_build_t1 = time.perf_counter()
        timing["neighbor_cache_build_sec"] = cache_build_t1 - cache_build_t0

    height, width = valid_mask.shape
    flat_valid_idx = view_neighbor_cache["flat_valid_idx"]

    pbr_flat = render_pkg["render"].permute(1, 2, 0).reshape(-1, 3).clone()
    direct_flat = pbr_flat.new_zeros((height * width, 3))
    indirect_flat = pbr_flat.new_zeros((height * width, 3))
    occlusion_flat = pbr_flat.new_zeros((height * width, 1))

    flat_normals = F.normalize(render_pkg["normal"].permute(1, 2, 0).reshape(-1, 3), dim=-1, eps=1e-6)
    flat_albedo = render_pkg["albedo"].permute(1, 2, 0).reshape(-1, 3)
    flat_roughness = render_pkg["roughness"].permute(1, 2, 0).reshape(-1, 1)
    flat_metallic = render_pkg["metallic"].permute(1, 2, 0).reshape(-1, 1)

    cache_row_idx = torch.arange(flat_valid_idx.shape[0], device=flat_valid_idx.device)
    for row_chunk in torch.split(cache_row_idx, query_chunk_size):
        chunk_idx = flat_valid_idx.index_select(0, row_chunk)
        chunk_neighbor_cache = slice_view_sop_neighbor_cache_rows(view_neighbor_cache, row_chunk)
        pts = chunk_neighbor_cache["points"]
        nrm = flat_normals[chunk_idx]
        albedo = flat_albedo[chunk_idx]
        roughness = flat_roughness[chunk_idx]
        metallic = flat_metallic[chunk_idx]
        viewdirs = compute_view_directions(pts, viewpoint_cam.camera_center)

        if profile_efficiency:
            _sync_cuda()
            light_sample_t0 = time.perf_counter()
        lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(
            normals=nrm,
            num_samples=num_shading_samples,
            randomized=randomized_samples,
        )
        if profile_efficiency:
            _sync_cuda()
            light_sample_t1 = time.perf_counter()
            timing["light_sample_sec"] += light_sample_t1 - light_sample_t0

        if profile_efficiency:
            _sync_cuda()
            env_lookup_t0 = time.perf_counter()
        direct_radiance = envmap(lightdirs)
        if profile_efficiency:
            _sync_cuda()
            env_lookup_t1 = time.perf_counter()
            timing["env_lookup_sec"] += env_lookup_t1 - env_lookup_t0
        # Keep render-time incident lighting consistent with stage2 SOP training:
        # L_i = (1 - O) * L_dir + L_in.
        if profile_efficiency:
            _sync_cuda()
            sop_query_t0 = time.perf_counter()
        sop_query_profile = {} if profile_efficiency else None
        query_indirect, query_occlusion = query_sops_directional(
            x_world=pts,
            query_dirs=lightdirs,
            probe_xyz=sop_tensors["probe_xyz"],
            probe_normal=sop_tensors["probe_normal"],
            probe_lin_tex=sop_tensors["probe_lin_tex"],
            probe_occ_tex=sop_tensors["probe_occ_tex"],
            radius=float(sop_query_radius) if sop_query_radius and sop_query_radius > 0.0 else None,
            topk=sop_query_topk,
            chunk_size=query_chunk_size,
            neighbor_cache=chunk_neighbor_cache,
            profile_timing=sop_query_profile,
        )
        if profile_efficiency:
            _sync_cuda()
            sop_query_t1 = time.perf_counter()
            timing["sop_query_sec"] += sop_query_t1 - sop_query_t0
            timing["sop_query_knn_sec"] += float(sop_query_profile.get("sop_query_knn_sec", 0.0))
            timing["sop_query_prep_sec"] += float(sop_query_profile.get("sop_query_prep_sec", 0.0))
            timing["sop_query_sample_sec"] += float(sop_query_profile.get("sop_query_sample_sec", 0.0))
            timing["sop_query_blend_sec"] += float(sop_query_profile.get("sop_query_blend_sec", 0.0))
        incident_radiance = (1.0 - query_occlusion) * direct_radiance + query_indirect

        if profile_efficiency:
            _sync_cuda()
            brdf_t0 = time.perf_counter()
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
        if profile_efficiency:
            _sync_cuda()
            brdf_t1 = time.perf_counter()
            timing["brdf_integrate_sec"] += brdf_t1 - brdf_t0

        pbr_flat[chunk_idx] = pbr_rgb
        direct_flat[chunk_idx] = direct_radiance.mean(dim=1)
        indirect_flat[chunk_idx] = query_indirect.mean(dim=1)
        occlusion_flat[chunk_idx] = query_occlusion.mean(dim=1)
    if profile_efficiency:
        _sync_cuda()
        shading_t1 = time.perf_counter()
        timing["sop_query_shading_sec"] = shading_t1 - shading_t0
        profiled_sum = (
            timing["recover_points_sec"]
            + timing["neighbor_cache_build_sec"]
            + timing["light_sample_sec"]
            + timing["env_lookup_sec"]
            + timing["sop_query_sec"]
            + timing["brdf_integrate_sec"]
        )
        timing["shading_other_sec"] = max(timing["sop_query_shading_sec"] - profiled_sum, 0.0)
        sop_query_profiled_sum = (
            timing["sop_query_knn_sec"]
            + timing["sop_query_prep_sec"]
            + timing["sop_query_sample_sec"]
            + timing["sop_query_blend_sec"]
        )
        timing["sop_query_other_sec"] = max(timing["sop_query_sec"] - sop_query_profiled_sum, 0.0)

    pbr_render_raw = pbr_flat.reshape(height, width, 3).permute(2, 0, 1)
    masks = _build_export_masks(viewpoint_cam, render_pkg, valid_mask, export_mask_mode)
    pbr_render_masked = pbr_render_raw * masks["export_alpha"]
    timing["render_core_sec"] = timing["gbuffer_render_sec"] + timing["sop_query_shading_sec"]

    return {
        "pbr_render": pbr_render_masked,
        "pbr_render_raw": pbr_render_raw,
        "sop_direct": direct_flat.reshape(height, width, 3).permute(2, 0, 1),
        "sop_indirect": indirect_flat.reshape(height, width, 3).permute(2, 0, 1),
        "sop_occlusion": occlusion_flat.reshape(height, width, 1).permute(2, 0, 1),
        "valid_mask": valid_mask.unsqueeze(0),
        "masks": masks,
        "render_pkg": render_pkg,
        "timing": timing,
    }


@torch.no_grad()
def save_stage2_sop_outputs(output_dir: str, view_name: str, stage2_view_pkg: dict):
    os.makedirs(output_dir, exist_ok=True)
    render_pkg = stage2_view_pkg["render_pkg"]
    masks = stage2_view_pkg["masks"]
    export_alpha = masks["export_alpha"]
    export_mask = masks["export_mask"]
    valid = export_mask > 0

    depth_raw = render_pkg["depth_unbiased"].detach().cpu().numpy()
    np.save(os.path.join(output_dir, f"{view_name}_depth_unbiased.npy"), depth_raw)

    rgb_raw = _composite_on_white(torch.clamp(stage2_view_pkg["pbr_render_raw"], 0.0, 1.0), export_alpha)
    rgb_masked = _composite_on_white(torch.clamp(stage2_view_pkg["pbr_render"], 0.0, 1.0), export_alpha)
    rgb_on_white = rgb_masked

    albedo_raw = _composite_on_white(torch.clamp(render_pkg["albedo"], 0.0, 1.0), export_alpha)
    roughness_raw = _composite_on_white(torch.clamp(render_pkg["roughness"], 0.0, 1.0), export_alpha)
    metallic_raw = _composite_on_white(torch.clamp(render_pkg["metallic"], 0.0, 1.0), export_alpha)
    normal_raw = _composite_on_white(torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0), export_alpha)

    albedo = albedo_raw
    roughness = roughness_raw
    metallic = metallic_raw
    normal = normal_raw
    depth_vis = _composite_on_white(_normalize_single_channel_for_vis(render_pkg["depth_unbiased"], valid_mask=valid), export_alpha)
    sop_direct = _composite_on_white(_tonemap_for_vis(stage2_view_pkg["sop_direct"]), export_alpha)
    sop_indirect = _composite_on_white(_tonemap_for_vis(stage2_view_pkg["sop_indirect"]), export_alpha)
    sop_occlusion = _composite_on_white(torch.clamp(stage2_view_pkg["sop_occlusion"], 0.0, 1.0), export_alpha)

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
    torchvision.utils.save_image(sop_direct, os.path.join(output_dir, f"{view_name}_sop_direct.png"))
    torchvision.utils.save_image(sop_indirect, os.path.join(output_dir, f"{view_name}_sop_indirect.png"))
    torchvision.utils.save_image(sop_occlusion, os.path.join(output_dir, f"{view_name}_sop_occlusion.png"))


def _init_efficiency_stats():
    return {
        "num_views": 0,
        "total_wall_sec": 0.0,
        "render_core_sec": 0.0,
        "gbuffer_render_sec": 0.0,
        "recover_points_sec": 0.0,
        "neighbor_cache_build_sec": 0.0,
        "light_sample_sec": 0.0,
        "env_lookup_sec": 0.0,
        "sop_query_sec": 0.0,
        "sop_query_knn_sec": 0.0,
        "sop_query_prep_sec": 0.0,
        "sop_query_sample_sec": 0.0,
        "sop_query_blend_sec": 0.0,
        "sop_query_other_sec": 0.0,
        "brdf_integrate_sec": 0.0,
        "shading_other_sec": 0.0,
        "sop_query_shading_sec": 0.0,
        "save_sec": 0.0,
        "other_sec": 0.0,
    }


def _finalize_efficiency_stats(stats):
    num_views = max(int(stats["num_views"]), 1)
    total_wall = float(stats["total_wall_sec"])
    render_core = float(stats["render_core_sec"])
    gbuffer = float(stats["gbuffer_render_sec"])
    recover_points = float(stats["recover_points_sec"])
    neighbor_cache_build = float(stats["neighbor_cache_build_sec"])
    light_sample = float(stats["light_sample_sec"])
    env_lookup = float(stats["env_lookup_sec"])
    sop_query = float(stats["sop_query_sec"])
    sop_query_knn = float(stats["sop_query_knn_sec"])
    sop_query_prep = float(stats["sop_query_prep_sec"])
    sop_query_sample = float(stats["sop_query_sample_sec"])
    sop_query_blend = float(stats["sop_query_blend_sec"])
    sop_query_other = float(stats["sop_query_other_sec"])
    brdf_integrate = float(stats["brdf_integrate_sec"])
    shading_other = float(stats["shading_other_sec"])
    sop_shading = float(stats["sop_query_shading_sec"])
    save_sec = float(stats["save_sec"])
    other_sec = float(stats["other_sec"])
    return {
        **stats,
        "avg_total_ms_per_view": 1000.0 * total_wall / num_views,
        "avg_render_ms_per_view": 1000.0 * render_core / num_views,
        "avg_gbuffer_ms_per_view": 1000.0 * gbuffer / num_views,
        "avg_recover_points_ms_per_view": 1000.0 * recover_points / num_views,
        "avg_neighbor_cache_build_ms_per_view": 1000.0 * neighbor_cache_build / num_views,
        "avg_light_sample_ms_per_view": 1000.0 * light_sample / num_views,
        "avg_env_lookup_ms_per_view": 1000.0 * env_lookup / num_views,
        "avg_sop_query_ms_per_view": 1000.0 * sop_query / num_views,
        "avg_sop_query_knn_ms_per_view": 1000.0 * sop_query_knn / num_views,
        "avg_sop_query_prep_ms_per_view": 1000.0 * sop_query_prep / num_views,
        "avg_sop_query_sample_ms_per_view": 1000.0 * sop_query_sample / num_views,
        "avg_sop_query_blend_ms_per_view": 1000.0 * sop_query_blend / num_views,
        "avg_sop_query_other_ms_per_view": 1000.0 * sop_query_other / num_views,
        "avg_brdf_integrate_ms_per_view": 1000.0 * brdf_integrate / num_views,
        "avg_shading_other_ms_per_view": 1000.0 * shading_other / num_views,
        "avg_sop_query_shading_ms_per_view": 1000.0 * sop_shading / num_views,
        "avg_save_ms_per_view": 1000.0 * save_sec / num_views,
        "avg_other_ms_per_view": 1000.0 * other_sec / num_views,
        "render_fps": float(stats["num_views"]) / max(render_core, 1e-8),
        "end_to_end_fps": float(stats["num_views"]) / max(total_wall, 1e-8),
    }


def _print_efficiency_summary(split_name: str, summary: dict):
    print(
        f"[Stage2-SOP][Efficiency][{split_name}] "
        f"views={summary['num_views']} "
        f"render={summary['render_core_sec']:.4f}s "
        f"(gbuffer={summary['gbuffer_render_sec']:.4f}s, "
        f"sop+pbr={summary['sop_query_shading_sec']:.4f}s, "
        f"recover={summary['recover_points_sec']:.4f}s, "
        f"cache_build={summary['neighbor_cache_build_sec']:.4f}s, "
        f"light_sample={summary['light_sample_sec']:.4f}s, "
        f"env_lookup={summary['env_lookup_sec']:.4f}s, "
        f"sop_query={summary['sop_query_sec']:.4f}s, "
        f"sop_knn={summary['sop_query_knn_sec']:.4f}s, "
        f"sop_prep={summary['sop_query_prep_sec']:.4f}s, "
        f"sop_sample={summary['sop_query_sample_sec']:.4f}s, "
        f"sop_blend={summary['sop_query_blend_sec']:.4f}s, "
        f"sop_other={summary['sop_query_other_sec']:.4f}s, "
        f"brdf={summary['brdf_integrate_sec']:.4f}s, "
        f"shading_other={summary['shading_other_sec']:.4f}s) "
        f"save={summary['save_sec']:.4f}s other={summary['other_sec']:.4f}s total={summary['total_wall_sec']:.4f}s"
    )
    print(
        f"[Stage2-SOP][Efficiency][{split_name}] "
        f"avg_render={summary['avg_render_ms_per_view']:.3f}ms/view "
        f"(gbuffer={summary['avg_gbuffer_ms_per_view']:.3f}ms, "
        f"sop+pbr={summary['avg_sop_query_shading_ms_per_view']:.3f}ms, "
        f"recover={summary['avg_recover_points_ms_per_view']:.3f}ms, "
        f"cache_build={summary['avg_neighbor_cache_build_ms_per_view']:.3f}ms, "
        f"light_sample={summary['avg_light_sample_ms_per_view']:.3f}ms, "
        f"env_lookup={summary['avg_env_lookup_ms_per_view']:.3f}ms, "
        f"sop_query={summary['avg_sop_query_ms_per_view']:.3f}ms, "
        f"sop_knn={summary['avg_sop_query_knn_ms_per_view']:.3f}ms, "
        f"sop_prep={summary['avg_sop_query_prep_ms_per_view']:.3f}ms, "
        f"sop_sample={summary['avg_sop_query_sample_ms_per_view']:.3f}ms, "
        f"sop_blend={summary['avg_sop_query_blend_ms_per_view']:.3f}ms, "
        f"sop_other={summary['avg_sop_query_other_ms_per_view']:.3f}ms, "
        f"brdf={summary['avg_brdf_integrate_ms_per_view']:.3f}ms, "
        f"shading_other={summary['avg_shading_other_ms_per_view']:.3f}ms) "
        f"avg_other={summary['avg_other_ms_per_view']:.3f}ms avg_save={summary['avg_save_ms_per_view']:.3f}ms "
        f"render_fps={summary['render_fps']:.3f} end_to_end_fps={summary['end_to_end_fps']:.3f}"
    )


def _print_efficiency_per_view(split_name: str, view_name: str, render_timing: dict, total_wall_sec: float, save_sec: float, other_sec: float):
    msg = (
        f"[Stage2-SOP][Efficiency][{split_name}][{view_name}] "
        f"render={1000.0 * float(render_timing['render_core_sec']):.3f}ms "
        f"(gbuffer={1000.0 * float(render_timing['gbuffer_render_sec']):.3f}ms, "
        f"sop+pbr={1000.0 * float(render_timing['sop_query_shading_sec']):.3f}ms, "
        f"recover={1000.0 * float(render_timing['recover_points_sec']):.3f}ms, "
        f"cache_build={1000.0 * float(render_timing['neighbor_cache_build_sec']):.3f}ms, "
        f"light_sample={1000.0 * float(render_timing['light_sample_sec']):.3f}ms, "
        f"env_lookup={1000.0 * float(render_timing['env_lookup_sec']):.3f}ms, "
        f"sop_query={1000.0 * float(render_timing['sop_query_sec']):.3f}ms, "
        f"sop_knn={1000.0 * float(render_timing['sop_query_knn_sec']):.3f}ms, "
        f"sop_prep={1000.0 * float(render_timing['sop_query_prep_sec']):.3f}ms, "
        f"sop_sample={1000.0 * float(render_timing['sop_query_sample_sec']):.3f}ms, "
        f"sop_blend={1000.0 * float(render_timing['sop_query_blend_sec']):.3f}ms, "
        f"sop_other={1000.0 * float(render_timing['sop_query_other_sec']):.3f}ms, "
        f"brdf={1000.0 * float(render_timing['brdf_integrate_sec']):.3f}ms, "
        f"shading_other={1000.0 * float(render_timing['shading_other_sec']):.3f}ms) "
        f"save={1000.0 * save_sec:.3f}ms "
        f"other={1000.0 * other_sec:.3f}ms "
        f"total={1000.0 * total_wall_sec:.3f}ms"
    )
    tqdm.write(msg)


def _build_unique_view_name(viewpoint_cam):
    base_name = getattr(viewpoint_cam, "image_name", "view")
    uid = getattr(viewpoint_cam, "uid", None)
    if uid is None:
        return str(base_name)
    try:
        return f"{int(uid):05d}_{base_name}"
    except Exception:
        return f"{uid}_{base_name}"


if __name__ == "__main__":
    parser = ArgumentParser(description="Render images from a Stage2 SOP checkpoint")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "both"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--query_chunk_size", type=int, default=2048)
    parser.add_argument("--sop_query_radius", type=float, default=0.0)
    parser.add_argument("--sop_query_topk", type=int, default=8)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=True)
    parser.add_argument("--export_mask_mode", type=str, default="render", choices=["render", "gt", "intersect"])
    parser.add_argument("--profile_efficiency", action="store_true", default=False)
    parser.add_argument("--profile_efficiency_per_view", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    checkpoint_path = args.checkpoint or os.path.join(args.model_path, "object_step2_sop.ckpt")
    print("Rendering stage2 SOP views for " + args.model_path)
    print("Loading checkpoint: " + checkpoint_path)

    dataset = model.extract(args)
    opt_args = opt.extract(args)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    payload, envmap, sop_tensors, iteration, saved_args = load_stage2_sop_checkpoint(gaussians, opt_args, checkpoint_path)
    print("Checkpoint SOP probes:", int(sop_tensors["probe_xyz"].shape[0]))
    if payload.get("sop_init_path"):
        print("Checkpoint SOP source:", payload["sop_init_path"])
    if payload.get("trace_backend") is not None:
        print("Checkpoint SOP supervision backend:", payload["trace_backend"])
    if saved_args.get("sop_query_topk") is not None:
        print("Checkpoint training query topk:", saved_args["sop_query_topk"])

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    split_to_cameras = []
    if args.split in ("train", "both"):
        split_to_cameras.append(("train", scene.getTrainCameras()))
    if args.split in ("test", "both"):
        split_to_cameras.append(("test", scene.getTestCameras()))

    for split_name, cameras in split_to_cameras:
        if cameras is None or len(cameras) == 0:
            print(f"No {split_name} cameras found, skip.")
            continue

        output_dir = os.path.join(args.model_path, "stage2_sop_render", split_name, f"ours_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        torchvision.utils.save_image(envmap.visualization(), os.path.join(output_dir, "envmap.png"))
        torch.save(envmap.capture(), os.path.join(output_dir, "envmap.pt"))
        torch.save({k: v.detach().cpu() for k, v in sop_tensors.items()}, os.path.join(output_dir, "sop.pt"))
        save_sop_texture_previews(output_dir, sop_tensors, preview_count=16, prefix="sop")
        efficiency_stats = _init_efficiency_stats()

        for viewpoint_cam in tqdm(cameras, desc=f"Rendering {split_name} views"):
            if args.profile_efficiency:
                _sync_cuda()
                view_t0 = time.perf_counter()
            stage2_view_pkg = render_stage2_sop_view(
                viewpoint_cam=viewpoint_cam,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                envmap=envmap,
                sop_tensors=sop_tensors,
                num_shading_samples=args.num_shading_samples,
                query_chunk_size=args.query_chunk_size,
                sop_query_radius=args.sop_query_radius,
                sop_query_topk=args.sop_query_topk,
                export_mask_mode=args.export_mask_mode,
                profile_efficiency=args.profile_efficiency,
                randomized_samples=not args.disable_sample_jitter,
            )
            if args.profile_efficiency:
                _sync_cuda()
                save_t0 = time.perf_counter()
            unique_view_name = _build_unique_view_name(viewpoint_cam)
            save_stage2_sop_outputs(output_dir, unique_view_name, stage2_view_pkg)

            if args.profile_efficiency:
                _sync_cuda()
                view_t1 = time.perf_counter()
                render_timing = stage2_view_pkg["timing"]
                save_sec = view_t1 - save_t0
                total_wall_sec = view_t1 - view_t0
                render_core_sec = float(render_timing["render_core_sec"])
                other_sec = max(total_wall_sec - render_core_sec - save_sec, 0.0)
                efficiency_stats["num_views"] += 1
                efficiency_stats["total_wall_sec"] += total_wall_sec
                efficiency_stats["render_core_sec"] += render_core_sec
                efficiency_stats["gbuffer_render_sec"] += float(render_timing["gbuffer_render_sec"])
                efficiency_stats["recover_points_sec"] += float(render_timing["recover_points_sec"])
                efficiency_stats["neighbor_cache_build_sec"] += float(render_timing["neighbor_cache_build_sec"])
                efficiency_stats["light_sample_sec"] += float(render_timing["light_sample_sec"])
                efficiency_stats["env_lookup_sec"] += float(render_timing["env_lookup_sec"])
                efficiency_stats["sop_query_sec"] += float(render_timing["sop_query_sec"])
                efficiency_stats["sop_query_knn_sec"] += float(render_timing["sop_query_knn_sec"])
                efficiency_stats["sop_query_prep_sec"] += float(render_timing["sop_query_prep_sec"])
                efficiency_stats["sop_query_sample_sec"] += float(render_timing["sop_query_sample_sec"])
                efficiency_stats["sop_query_blend_sec"] += float(render_timing["sop_query_blend_sec"])
                efficiency_stats["sop_query_other_sec"] += float(render_timing["sop_query_other_sec"])
                efficiency_stats["brdf_integrate_sec"] += float(render_timing["brdf_integrate_sec"])
                efficiency_stats["shading_other_sec"] += float(render_timing["shading_other_sec"])
                efficiency_stats["sop_query_shading_sec"] += float(render_timing["sop_query_shading_sec"])
                efficiency_stats["save_sec"] += save_sec
                efficiency_stats["other_sec"] += other_sec
                if args.profile_efficiency_per_view:
                    _print_efficiency_per_view(
                        split_name=split_name,
                        view_name=unique_view_name,
                        render_timing=render_timing,
                        total_wall_sec=total_wall_sec,
                        save_sec=save_sec,
                        other_sec=other_sec,
                    )

        if args.profile_efficiency and efficiency_stats["num_views"] > 0:
            summary = _finalize_efficiency_stats(efficiency_stats)
            _print_efficiency_summary(split_name, summary)
            with open(os.path.join(output_dir, "efficiency_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        print(f"Saved {split_name} renders to {output_dir}")
