import json
import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
import gaussian_renderer as gaussian_renderer_module
from gaussian_renderer import render_sop_gbuffer
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from train_stage2_sop import Stage2SOPState, set_gaussian_para, _load_sop_payload, _resolve_sop_init_path
from utils.general_utils import safe_state
from utils.graphics_utils import rgb_to_srgb
from utils.image_utils import psnr, visualize_depth
from utils.loss_utils import ssim
from utils.losses_comgs_stage2_sop import compute_stage2_sop_loss
from utils import sop_utils as sop_utils_module


_ACTIVE_FRAME_TIMINGS = None
_ACTIVE_FRAME_CUDA_EVENTS = None
_PROFILE_HOOKS_INSTALLED = False
_ORIG_QUERY_KNN_PROBES = None
_PROFILE_PROBE_ATLAS_BUFFER = "_profile_probe_atlas"
_PROFILE_ENV_ATLAS_ATTR = "_profile_env_atlas"
_PROFILE_PROBE_FLAT_META = {}
_PROFILE_USE_ENV_ATLAS = False


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _begin_frame_profile():
    global _ACTIVE_FRAME_TIMINGS, _ACTIVE_FRAME_CUDA_EVENTS
    _ACTIVE_FRAME_TIMINGS = defaultdict(float)
    _ACTIVE_FRAME_CUDA_EVENTS = defaultdict(list)


def _end_frame_profile():
    global _ACTIVE_FRAME_TIMINGS, _ACTIVE_FRAME_CUDA_EVENTS
    if _ACTIVE_FRAME_TIMINGS is None:
        return {}

    timings = dict(_ACTIVE_FRAME_TIMINGS)
    if _ACTIVE_FRAME_CUDA_EVENTS is not None and len(_ACTIVE_FRAME_CUDA_EVENTS) > 0:
        _cuda_sync()
        for key, event_pairs in _ACTIVE_FRAME_CUDA_EVENTS.items():
            total_ms = 0.0
            for start_ev, end_ev in event_pairs:
                total_ms += float(start_ev.elapsed_time(end_ev))
            timings[key] = timings.get(key, 0.0) + total_ms

    _ACTIVE_FRAME_TIMINGS = None
    _ACTIVE_FRAME_CUDA_EVENTS = None
    return timings


@contextmanager
def _profile_cuda_block(name: str):
    if _ACTIVE_FRAME_TIMINGS is None:
        yield
        return

    if torch.cuda.is_available():
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        try:
            yield
        finally:
            end_ev.record()
            _ACTIVE_FRAME_CUDA_EVENTS[name].append((start_ev, end_ev))
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        _ACTIVE_FRAME_TIMINGS[name] += (time.perf_counter() - start) * 1000.0


@contextmanager
def _profile_cpu_block(name: str, sync_cuda: bool = False):
    if _ACTIVE_FRAME_TIMINGS is None:
        yield
        return

    if sync_cuda:
        _cuda_sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda:
            _cuda_sync()
        _ACTIVE_FRAME_TIMINGS[name] += (time.perf_counter() - start) * 1000.0


def _texture_to_nchw(texture: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if texture.dim() != 4:
        raise ValueError(f"{name} must be 4D, got {tuple(texture.shape)}")
    if texture.shape[1] == channels:
        return texture.contiguous()
    if texture.shape[-1] == channels:
        return texture.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"{name} cannot be interpreted as {channels}-channel texture: {tuple(texture.shape)}")


def _build_probe_atlas(probe_lin_tex: torch.Tensor, probe_occ_tex: torch.Tensor) -> torch.Tensor:
    lin = _texture_to_nchw(probe_lin_tex, 3, "probe_lin_tex").clamp_min(0.0)
    occ = _texture_to_nchw(probe_occ_tex, 1, "probe_occ_tex").clamp(0.0, 1.0)
    if lin.shape[0] != occ.shape[0] or lin.shape[-2:] != occ.shape[-2:]:
        raise ValueError(f"Probe texture shape mismatch: lin={tuple(lin.shape)}, occ={tuple(occ.shape)}")
    return torch.cat([lin, occ], dim=1).contiguous()


def _build_probe_atlas_flat(probe_lin_tex: torch.Tensor, probe_occ_tex: torch.Tensor):
    atlas = _build_probe_atlas(probe_lin_tex, probe_occ_tex)
    num_probes, _channels, height, width = atlas.shape
    atlas_flat = atlas.permute(0, 2, 3, 1).reshape(num_probes * height * width, 4).contiguous()
    return atlas_flat, (int(num_probes), int(height), int(width))


def _looks_like_probe_atlas(probe_lin_tex: torch.Tensor, probe_occ_tex: Optional[torch.Tensor]) -> bool:
    if probe_lin_tex.dim() != 4 or probe_lin_tex.shape[1] < 4:
        return False
    if probe_occ_tex is None:
        return True
    return (
        probe_occ_tex.dim() == 4
        and probe_occ_tex.shape[1] == 1
        and probe_occ_tex.shape[0] == probe_lin_tex.shape[0]
        and probe_occ_tex.shape[-2:] == probe_lin_tex.shape[-2:]
    )


def _get_probe_atlas(probe_lin_tex: torch.Tensor, probe_occ_tex: Optional[torch.Tensor]) -> torch.Tensor:
    if _looks_like_probe_atlas(probe_lin_tex, probe_occ_tex):
        return probe_lin_tex.contiguous()
    if probe_lin_tex.dim() == 4 and probe_lin_tex.shape[-1] >= 4:
        return probe_lin_tex.permute(0, 3, 1, 2).contiguous()
    if probe_occ_tex is None:
        raise ValueError("probe_occ_tex is required when probe_lin_tex is not a packed atlas")
    return _build_probe_atlas(probe_lin_tex, probe_occ_tex)


def _get_probe_atlas_flat(probe_lin_tex: torch.Tensor, probe_occ_tex: Optional[torch.Tensor]):
    if probe_lin_tex.dim() == 2 and probe_lin_tex.shape[-1] >= 4:
        meta = _PROFILE_PROBE_FLAT_META.get(int(probe_lin_tex.data_ptr()))
        if meta is None:
            raise RuntimeError("Packed flat probe atlas is missing shape metadata.")
        return probe_lin_tex.contiguous(), meta

    atlas = _get_probe_atlas(probe_lin_tex, probe_occ_tex)
    num_probes, channels, height, width = atlas.shape
    if channels < 4:
        raise ValueError(f"Probe atlas needs at least 4 channels, got {tuple(atlas.shape)}")
    atlas_flat = atlas[:, :4].permute(0, 2, 3, 1).reshape(num_probes * height * width, 4).contiguous()
    return atlas_flat, (int(num_probes), int(height), int(width))


def _set_nonpersistent_buffer(module, name: str, value: torch.Tensor) -> None:
    if hasattr(module, "_buffers"):
        if name in module._buffers:
            module._buffers[name] = value
        else:
            module.register_buffer(name, value, persistent=False)
        return
    setattr(module, name, value)


def _prepare_profile_probe_atlas(sop_state: Stage2SOPState) -> torch.Tensor:
    with torch.no_grad():
        atlas_flat, meta = _build_probe_atlas_flat(sop_state.lin_tex, sop_state.occ_tex)
    _set_nonpersistent_buffer(sop_state, _PROFILE_PROBE_ATLAS_BUFFER, atlas_flat)
    _PROFILE_PROBE_FLAT_META[int(atlas_flat.data_ptr())] = meta
    num_probes, height, width = meta
    print(
        f"[PROFILE] packed probe atlas flat: {tuple(atlas_flat.shape)} "
        f"from [num_probes={num_probes}, H={height}, W={width}, C=4]"
    )
    return atlas_flat


def _prepare_profile_env_atlas(gaussians: GaussianModel):
    envmap = getattr(gaussians, "get_envmap", None)
    if envmap is None or not hasattr(envmap, "base"):
        cache = {"texture": None, "envmap": envmap, "representation": None}
        setattr(gaussians, _PROFILE_ENV_ATLAS_ATTR, cache)
        return cache

    base = envmap.base
    representation = getattr(envmap, "representation", "octahedral")
    if base.dim() != 3 or representation not in {"octahedral", "latlong"}:
        cache = {"texture": None, "envmap": envmap, "representation": representation}
        setattr(gaussians, _PROFILE_ENV_ATLAS_ATTR, cache)
        print("[PROFILE] env atlas skipped: unsupported environment map layout")
        return cache

    with torch.no_grad():
        texture = base.detach().permute(2, 0, 1).unsqueeze(0).contiguous()
    cache = {
        "texture": texture,
        "envmap": envmap,
        "representation": representation,
    }
    setattr(gaussians, _PROFILE_ENV_ATLAS_ATTR, cache)
    print(f"[PROFILE] packed env atlas: {tuple(texture.shape)} representation={representation}")
    return cache


def _bilinear_texel_indices(
    uv: torch.Tensor,
    height: int,
    width: int,
    wrap_u: bool = False,
):
    uv = uv.to(dtype=torch.float32)
    u = uv[..., 0]
    v = uv[..., 1].clamp(0.0, 1.0)
    if wrap_u:
        u = torch.remainder(u, 1.0)
    else:
        u = u.clamp(0.0, 1.0)

    x = u * float(width) - 0.5
    y = v * float(height) - 0.5
    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0.to(dtype=x.dtype)).to(dtype=uv.dtype)
    wy = (y - y0.to(dtype=y.dtype)).to(dtype=uv.dtype)

    if wrap_u:
        x0 = torch.remainder(x0, width)
        x1 = torch.remainder(x1, width)
    else:
        x0 = x0.clamp(0, width - 1)
        x1 = x1.clamp(0, width - 1)
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)

    idx00 = y0 * width + x0
    idx10 = y0 * width + x1
    idx01 = y1 * width + x0
    idx11 = y1 * width + x1
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    return idx00, idx10, idx01, idx11, w00, w10, w01, w11


def _gather_probe_texels_flat(
    atlas_flat: torch.Tensor,
    probe_texel_base: torch.Tensor,
    pixel_ids: torch.Tensor,
) -> torch.Tensor:
    num_points, neighbor_k = probe_texel_base.shape
    num_samples = pixel_ids.shape[1]
    linear_ids = probe_texel_base[:, :, None] + pixel_ids[:, None, :]
    gathered = atlas_flat[linear_ids.reshape(-1)]
    return gathered.view(num_points, neighbor_k, num_samples, atlas_flat.shape[-1])


def _sample_probe_atlas_bilinear(
    probe_atlas: torch.Tensor,
    probe_ids: torch.Tensor,
    uv: torch.Tensor,
) -> torch.Tensor:
    _, _, height, width = probe_atlas.shape
    atlas_flat = probe_atlas.reshape(probe_atlas.shape[0], probe_atlas.shape[1], height * width)
    idx00, idx10, idx01, idx11, w00, w10, w01, w11 = _bilinear_texel_indices(
        uv, height, width, wrap_u=False
    )

    w00 = w00[:, None, :, None].to(dtype=probe_atlas.dtype)
    w10 = w10[:, None, :, None].to(dtype=probe_atlas.dtype)
    w01 = w01[:, None, :, None].to(dtype=probe_atlas.dtype)
    w11 = w11[:, None, :, None].to(dtype=probe_atlas.dtype)

    probe_texel_base = probe_ids * (height * width)
    return (
        _gather_probe_texels_flat(atlas_flat, probe_texel_base, idx00) * w00
        + _gather_probe_texels_flat(atlas_flat, probe_texel_base, idx10) * w10
        + _gather_probe_texels_flat(atlas_flat, probe_texel_base, idx01) * w01
        + _gather_probe_texels_flat(atlas_flat, probe_texel_base, idx11) * w11
    )


def _sample_probe_atlas_bilinear_flat(
    probe_atlas_nchw: torch.Tensor,
    probe_ids: torch.Tensor,
    uv: torch.Tensor,
) -> torch.Tensor:
    with _profile_cuda_block("atlas_sample_index_ms"):
        num_points, neighbor_k = probe_ids.shape
        num_samples = uv.shape[1]
        flat_probe_ids = probe_ids.reshape(-1)
        grid = uv[:, None, :, :].expand(-1, neighbor_k, -1, -1)
        grid = grid.reshape(num_points * neighbor_k, num_samples, 1, 2)
        grid = grid.to(dtype=probe_atlas_nchw.dtype).mul(2.0).sub(1.0)

    with _profile_cuda_block("atlas_sample_gather_ms"):
        sampled_tex = probe_atlas_nchw.index_select(0, flat_probe_ids)
        sampled = F.grid_sample(
            sampled_tex,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    return sampled.view(num_points, neighbor_k, num_samples, probe_atlas_nchw.shape[1])


def _sample_texture2d_bilinear(texture: torch.Tensor, uv: torch.Tensor, wrap_u: bool = False) -> torch.Tensor:
    if texture.dim() == 4:
        if texture.shape[0] != 1:
            raise ValueError(f"Expected a single texture, got {tuple(texture.shape)}")
        texture = texture[0]
    if texture.dim() != 3:
        raise ValueError(f"Expected texture shape [C,H,W], got {tuple(texture.shape)}")

    channels, height, width = texture.shape
    tex_flat = texture.reshape(channels, height * width)
    idx00, idx10, idx01, idx11, w00, w10, w01, w11 = _bilinear_texel_indices(
        uv, height, width, wrap_u=wrap_u
    )

    def gather(pixel_ids: torch.Tensor) -> torch.Tensor:
        values = tex_flat[:, pixel_ids.reshape(-1)].t()
        return values.view(*pixel_ids.shape, channels)

    dtype = texture.dtype
    return (
        gather(idx00) * w00[..., None].to(dtype=dtype)
        + gather(idx10) * w10[..., None].to(dtype=dtype)
        + gather(idx01) * w01[..., None].to(dtype=dtype)
        + gather(idx11) * w11[..., None].to(dtype=dtype)
    )


def _dir_to_latlong_uv(dirs: torch.Tensor) -> torch.Tensor:
    dirs = F.normalize(dirs, dim=-1, eps=1e-8)
    u = torch.atan2(dirs[..., 0:1], -dirs[..., 2:3]).nan_to_num() / (2.0 * torch.pi) + 0.5
    v = torch.acos(dirs[..., 1:2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi
    return torch.cat([u, v], dim=-1)


def _sample_profile_env_atlas(gaussians: GaussianModel, dirs: torch.Tensor) -> Optional[torch.Tensor]:
    cache = getattr(gaussians, _PROFILE_ENV_ATLAS_ATTR, None)
    if cache is None:
        cache = _prepare_profile_env_atlas(gaussians)
    if cache is None or cache.get("texture", None) is None:
        return None

    envmap = cache["envmap"]
    texture = cache["texture"].to(device=dirs.device, dtype=dirs.dtype)
    dirs_for_uv = dirs
    transform = getattr(envmap, "transform", None)
    if transform is not None:
        dirs_for_uv = dirs_for_uv @ transform.to(device=dirs.device, dtype=dirs.dtype).T

    representation = cache.get("representation", getattr(envmap, "representation", "octahedral"))
    if representation == "latlong":
        uv = _dir_to_latlong_uv(dirs_for_uv)
        sampled = _sample_texture2d_bilinear(texture, uv, wrap_u=True)
    elif representation == "octahedral":
        uv = sop_utils_module.dir_to_oct_uv(dirs_for_uv).clamp(0.0, 1.0)
        sampled = _sample_texture2d_bilinear(texture, uv, wrap_u=False)
    else:
        return None

    activation = getattr(envmap, "activation", lambda x: x)
    return activation(sampled).clamp_min(0.0)


def _query_sops_directional_atlas(
    x_world: torch.Tensor,
    query_dirs: torch.Tensor,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: Optional[torch.Tensor],
    radius: Optional[float] = None,
    topk: int = 8,
    eps: float = 1e-8,
    chunk_size: int = 1024,
):
    if x_world.numel() == 0:
        device = probe_xyz.device if probe_xyz.numel() > 0 else x_world.device
        dtype = probe_xyz.dtype if probe_xyz.numel() > 0 else x_world.dtype
        if query_dirs.dim() == 3:
            return (
                torch.zeros((0, query_dirs.shape[1], 3), device=device, dtype=dtype),
                torch.zeros((0, query_dirs.shape[1], 1), device=device, dtype=dtype),
            )
        return (
            torch.zeros((0, 3), device=device, dtype=dtype),
            torch.zeros((0, 1), device=device, dtype=dtype),
        )

    squeeze_sample_dim = False
    if query_dirs.dim() == 2:
        query_dirs = query_dirs.unsqueeze(1)
        squeeze_sample_dim = True
    elif query_dirs.dim() != 3 or query_dirs.shape[0] != x_world.shape[0]:
        raise ValueError(
            f"Expected query_dirs with shape [P, 3] or [P, S, 3], got {tuple(query_dirs.shape)} "
            f"for x_world={tuple(x_world.shape)}"
        )

    with _profile_cuda_block("query_sop_prep_ms"):
        device = x_world.device
        dtype = x_world.dtype
        probe_xyz = probe_xyz.to(device=device, dtype=dtype)
        probe_normal = F.normalize(probe_normal.to(device=device, dtype=dtype), dim=-1, eps=eps)
        probe_atlas_flat, atlas_meta = _get_probe_atlas_flat(probe_lin_tex, probe_occ_tex)
        probe_atlas_flat = probe_atlas_flat.to(device=device, dtype=dtype)
        _num_probe_tex, probe_tex_h, probe_tex_w = atlas_meta
        probe_atlas_nchw = (
            probe_atlas_flat.view(_num_probe_tex, probe_tex_h, probe_tex_w, 4).permute(0, 3, 1, 2).contiguous()
        )
        query_dirs = query_dirs.to(device=device, dtype=dtype)

        num_points = x_world.shape[0]
        num_probes = probe_xyz.shape[0]
        num_samples = query_dirs.shape[1]
        lin_out = torch.zeros((num_points, num_samples, 3), device=device, dtype=dtype)
        occ_out = torch.zeros((num_points, num_samples, 1), device=device, dtype=dtype)

    if num_probes == 0:
        if squeeze_sample_dim:
            return lin_out[:, 0], occ_out[:, 0]
        return lin_out, occ_out

    neighbor_k = min(max(int(topk), 1), num_probes)
    query_stride = max(1, int(chunk_size))
    for start in range(0, num_points, query_stride):
        end = min(start + query_stride, num_points)
        x_chunk = x_world[start:end]
        dirs_chunk = query_dirs[start:end]

        knn_dist, knn_idx = sop_utils_module._query_knn_probes(x_chunk, probe_xyz, neighbor_k)
        with _profile_cuda_block("query_sop_weight_ms"):
            weights = sop_utils_module._compute_neighbor_weights(
                x_chunk=x_chunk,
                probe_xyz=probe_xyz,
                probe_normal=probe_normal,
                knn_dist=knn_dist,
                knn_idx=knn_idx,
                radius=radius,
                eps=eps,
            )

        with _profile_cuda_block("query_sop_uv_ms"):
            uv_chunk = sop_utils_module.dir_to_oct_uv(dirs_chunk).clamp(0.0, 1.0)
        with _profile_cuda_block("atlas_sample_ms"):
            sampled = _sample_probe_atlas_bilinear_flat(
                probe_atlas_nchw=probe_atlas_nchw,
                probe_ids=knn_idx,
                uv=uv_chunk,
            )
        with _profile_cuda_block("query_sop_fuse_ms"):
            with _profile_cuda_block("query_sop_fuse_reduce_ms"):
                sampled_lin = sampled[..., :3]
                sampled_occ = sampled[..., 3:4]

                weight_sum = weights.sum(dim=1, keepdim=True)
                valid = weight_sum.squeeze(-1) > eps
                denom = weight_sum.unsqueeze(-1).clamp_min(eps)
                lin_vals = torch.sum(weights[:, :, None, None] * sampled_lin, dim=1) / denom
                occ_vals = torch.sum(weights[:, :, None, None] * sampled_occ, dim=1) / denom

            with _profile_cuda_block("query_sop_fuse_write_ms"):
                if bool(valid.all()):
                    lin_out[start:end] = lin_vals
                    occ_out[start:end] = occ_vals
                elif bool(valid.any()):
                    valid_mask = valid[:, None, None]
                    lin_chunk = lin_out[start:end]
                    occ_chunk = occ_out[start:end]
                    lin_out[start:end] = torch.where(valid_mask, lin_vals, lin_chunk)
                    occ_out[start:end] = torch.where(valid_mask, occ_vals, occ_chunk)

    with _profile_cuda_block("query_sop_post_ms"):
        occ_out = torch.clamp(occ_out, 0.0, 1.0)
    if squeeze_sample_dim:
        return lin_out[:, 0], occ_out[:, 0]
    return lin_out, occ_out


def _install_profile_hooks():
    global _PROFILE_HOOKS_INSTALLED, _ORIG_QUERY_KNN_PROBES
    if _PROFILE_HOOKS_INSTALLED:
        return

    _ORIG_QUERY_KNN_PROBES = sop_utils_module._query_knn_probes

    def _query_knn_probes_profiled(*args, **kwargs):
        with _profile_cuda_block("knn_ms"):
            return _ORIG_QUERY_KNN_PROBES(*args, **kwargs)

    def _rendering_equation_sop_profiled(
        base_color,
        roughness,
        metallic,
        normals,
        position,
        viewdirs,
        pc,
        pipe,
        probe_xyz,
        probe_normal,
        probe_lin_tex,
        probe_occ_tex,
        sample_training=False,
        f0=0.04,
        sop_query_radius=None,
        sop_query_topk=8,
        sop_query_chunk_size=1024,
        eps=1e-6,
        cuda_mem_debug=None,
    ):
        gaussian_renderer_module._log_cuda_mem(cuda_mem_debug, "before rendering_equation_sop_inner")
        envmap = pc.get_envmap
        with _profile_cuda_block("incident_sample_ms"):
            incident_dirs, incident_areas = gaussian_renderer_module._sample_incident_transport_sop(
                normals, pc, pipe, sample_training
            )

        with _profile_cuda_block("query_env_ms"):
            global_incident_lights = _sample_profile_env_atlas(pc, incident_dirs) if _PROFILE_USE_ENV_ATLAS else None
            if global_incident_lights is None:
                global_incident_lights = envmap(incident_dirs, mode="pure_env")

        gaussian_renderer_module._log_cuda_mem(cuda_mem_debug, "before query_sops_directional")
        with _profile_cuda_block("query_sop_ms"):
            query_indirect, query_occlusion = _query_sops_directional_atlas(
                x_world=position,
                query_dirs=incident_dirs,
                probe_xyz=probe_xyz,
                probe_normal=probe_normal,
                probe_lin_tex=probe_lin_tex,
                probe_occ_tex=probe_occ_tex,
                radius=sop_query_radius,
                topk=sop_query_topk,
                eps=eps,
                chunk_size=sop_query_chunk_size,
            )
        gaussian_renderer_module._log_cuda_mem(cuda_mem_debug, "after query_sops_directional")

        with _profile_cuda_block("light_mix_ms"):
            incident_visibility = 1 - query_occlusion
            local_incident_lights = query_indirect
            if pipe.wo_indirect:
                local_incident_lights = torch.zeros_like(local_incident_lights)
            if pipe.detach_indirect:
                incident_visibility = incident_visibility.detach()
                local_incident_lights = local_incident_lights.detach()
            incident_lights = incident_visibility * global_incident_lights + local_incident_lights

        with _profile_cuda_block("brdf_ms"):
            dielectric_f0 = gaussian_renderer_module._broadcast_to_target(f0, base_color)
            n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
            specular_f0 = dielectric_f0 * (1 - metallic) + base_color * metallic
            f_d = (1 - metallic)[:, None] * base_color[:, None] / np.pi
            f_s = gaussian_renderer_module.GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=specular_f0)

            transport = incident_lights * incident_areas * n_d_i
            diffuse = (f_d * transport).mean(dim=-2)
            specular = (f_s * transport).mean(dim=-2)

        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
        }
        gaussian_renderer_module._log_cuda_mem(cuda_mem_debug, "after rendering_equation_sop_inner")
        return results

    sop_utils_module._query_knn_probes = _query_knn_probes_profiled
    gaussian_renderer_module.rendering_equation_sop = _rendering_equation_sop_profiled
    _PROFILE_HOOKS_INSTALLED = True


def _format_profile_line(output_name: str, idx: int, view, timing: dict) -> str:
    view_name = getattr(view, "image_name", f"{idx:05d}")
    total = timing.get("frame_total_ms", 0.0)
    render_view = timing.get("view_total_ms", 0.0)
    metric_ms = timing.get("metric_ms", 0.0)
    lpips_ms = timing.get("lpips_ms", 0.0)
    save_ms = timing.get("save_ms", 0.0)

    frame_accounted = render_view + metric_ms + lpips_ms + save_ms
    frame_other = max(total - frame_accounted, 0.0)

    loss_total = timing.get("loss_total_ms", 0.0)
    incident_sample = timing.get("incident_sample_ms", 0.0)
    q_total = timing.get("query_sop_ms", 0.0)
    q_prep = timing.get("query_sop_prep_ms", 0.0)
    q_knn = timing.get("knn_ms", 0.0)
    q_weight = timing.get("query_sop_weight_ms", 0.0)
    q_uv = timing.get("query_sop_uv_ms", 0.0)
    q_sample = timing.get("atlas_sample_ms", 0.0)
    q_sample_index = timing.get("atlas_sample_index_ms", 0.0)
    q_sample_gather = timing.get("atlas_sample_gather_ms", 0.0)
    q_fuse = timing.get("query_sop_fuse_ms", 0.0)
    q_fuse_reduce = timing.get("query_sop_fuse_reduce_ms", 0.0)
    q_fuse_write = timing.get("query_sop_fuse_write_ms", 0.0)
    q_post = timing.get("query_sop_post_ms", 0.0)
    q_accounted = q_prep + q_knn + q_weight + q_uv + q_sample + q_fuse + q_post
    q_other = max(q_total - q_accounted, 0.0)

    query_env = timing.get("query_env_ms", 0.0)
    light_mix = timing.get("light_mix_ms", 0.0)
    brdf = timing.get("brdf_ms", 0.0)
    output_pack = timing.get("output_pack_ms", 0.0)
    gt_fetch = timing.get("gt_fetch_ms", 0.0)
    loss_other = max(loss_total - (incident_sample + q_total + query_env + light_mix + brdf), 0.0)
    return (
        f"[PROFILE] set={output_name} frame={idx:05d} view={view_name} "
        f"total={total:8.2f}ms "
        f"frame_other={frame_other:8.2f}ms "
        f"view_total={render_view:8.2f}ms metric={metric_ms:7.2f}ms lpips={lpips_ms:7.2f}ms "
        f"save={save_ms:8.2f}ms "
        f"raster={timing.get('raster_ms', 0.0):8.2f}ms gt={gt_fetch:6.2f}ms output={output_pack:6.2f}ms "
        f"loss={loss_total:8.2f}ms(inc={incident_sample:6.2f},env={query_env:6.2f},lm={light_mix:6.2f},brdf={brdf:6.2f},other={loss_other:6.2f}) "
        f"query_sop={q_total:8.2f}ms "
        f"[prep={q_prep:6.2f} knn={q_knn:6.2f} w={q_weight:6.2f} uv={q_uv:6.2f} "
        f"sample={q_sample:6.2f}(idx={q_sample_index:6.2f},g={q_sample_gather:6.2f}) "
        f"fuse={q_fuse:6.2f}(r={q_fuse_reduce:6.2f},w={q_fuse_write:6.2f}) "
        f"post={q_post:6.2f} other={q_other:6.2f}]"
    )


def select_views(views, first_k=-1):
    if first_k is None or first_k <= 0 or first_k >= len(views):
        return views, ""
    return views[:first_k], f"_first{first_k}"


def _repeat_to_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def _view_output_mask(view, ref: torch.Tensor) -> Optional[torch.Tensor]:
    mask = getattr(view, "mask", None)
    if mask is None:
        mask = getattr(view, "gt_alpha_mask", None)
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=ref.device, dtype=ref.dtype)
    return _repeat_to_rgb(mask)


def _resolve_stage2_checkpoint(args) -> Optional[Path]:
    if getattr(args, "start_checkpoint", ""):
        return Path(args.start_checkpoint)

    model_path = Path(args.model_path)
    if args.iteration is not None and int(args.iteration) > 0:
        iter_ckpt = model_path / f"object_step2_sop_iter_{int(args.iteration):06d}.ckpt"
        if iter_ckpt.exists():
            return iter_ckpt

    final_ckpt = model_path / "object_step2_sop.ckpt"
    if final_ckpt.exists():
        return final_ckpt
    return None


def _load_render_state(gaussians: GaussianModel, args):
    stage2_ckpt = _resolve_stage2_checkpoint(args)
    if stage2_ckpt is not None:
        if not stage2_ckpt.exists():
            raise FileNotFoundError(f"Stage2 SOP checkpoint not found: {stage2_ckpt}")
        payload = torch.load(stage2_ckpt, map_location="cuda")
        if not isinstance(payload, dict) or payload.get("format") != "irgs_stage2_sop_v1":
            raise RuntimeError(f"{stage2_ckpt} is not a valid Stage2 SOP checkpoint")

        gaussians.restore(payload["gaussians"], None)
        sop_state = Stage2SOPState(payload["sop"])
        source_info = dict(payload.get("source_info", {}))
        source_info["stage2_checkpoint"] = str(stage2_ckpt)
        loaded_iter = int(payload.get("iteration", 0))
        return loaded_iter, sop_state, source_info

    if not getattr(args, "start_checkpoint_refgs", ""):
        raise RuntimeError(
            "render_sop_profile.py needs a Stage2 SOP checkpoint under --model_path "
            "(object_step2_sop.ckpt / object_step2_sop_iter_xxxxxx.ckpt), "
            "or an explicit --start_checkpoint, or --start_checkpoint_refgs with --sop_init."
        )

    refgs_payload = torch.load(args.start_checkpoint_refgs, map_location="cuda")
    if not isinstance(refgs_payload, (tuple, list)) or len(refgs_payload) < 1:
        raise RuntimeError(f"Unsupported refgs checkpoint payload: {type(refgs_payload).__name__}")

    gaussians.restore_from_refgs(refgs_payload[0], None)
    sop_path = _resolve_sop_init_path(args.sop_init, args.model_path)
    sop_state = Stage2SOPState(_load_sop_payload(sop_path))
    source_info = {
        "refgs_checkpoint": args.start_checkpoint_refgs,
        "sop_init": str(sop_path),
    }
    fallback_iter = int(args.iteration) if int(args.iteration) > 0 else 0
    return fallback_iter, sop_state, source_info


@torch.no_grad()
def render_stage2_sop_view(viewpoint_cam, gaussians, background, pipe, opt, sop_state, args, iteration):
    with _profile_cuda_block("raster_ms"):
        render_pkg = render_sop_gbuffer(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            opt=opt,
            iteration=iteration,
            training=False,
        )
    with _profile_cuda_block("gt_fetch_ms"):
        gt_rgb = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)
    probe_atlas = getattr(sop_state, _PROFILE_PROBE_ATLAS_BUFFER, None)
    probe_lin_tex = probe_atlas if probe_atlas is not None else sop_state.lin_tex
    probe_occ_tex = probe_atlas if probe_atlas is not None else sop_state.occ_tex
    with _profile_cuda_block("loss_total_ms"):
        _, stats, aux = compute_stage2_sop_loss(
            render_pkg=render_pkg,
            gt_rgb=gt_rgb,
            viewpoint_camera=viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            opt=opt,
            training=False,
            probe_xyz=sop_state.probe_xyz,
            probe_normal=sop_state.probe_normal,
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            lambda_lam=args.lambda_lam,
            lambda_sops=args.lambda_sops,
            lambda_d2n=args.lambda_d2n,
            lambda_mask=args.lambda_mask,
            use_mask_loss=args.use_mask_loss,
            num_shading_samples=args.num_shading_samples,
            max_shading_points=0,
            sop_query_radius=args.sop_query_radius,
            sop_query_topk=args.sop_query_topk,
            sop_query_chunk_size=args.sop_query_chunk_size,
            randomized_samples=False,
            cuda_mem_debug=None,
        )

    with _profile_cuda_block("output_pack_ms"):
        outputs = {
            "render": torch.clamp(aux["pbr_render"], 0.0, 1.0),
            "albedo": torch.clamp(render_pkg["albedo"], 0.0, 1.0),
            "roughness": _repeat_to_rgb(torch.clamp(render_pkg["roughness"], 0.0, 1.0)),
            "metallic": _repeat_to_rgb(torch.clamp(render_pkg["metallic"], 0.0, 1.0)),
            "weight": _repeat_to_rgb(torch.clamp(render_pkg["weight"], 0.0, 1.0)),
            "depth": visualize_depth(render_pkg["depth_unbiased"][None]),
            "normal": torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0),
            "depth_normal": torch.clamp(aux["depth_normal"] * 0.5 + 0.5, 0.0, 1.0),
            "query_selection": _repeat_to_rgb(torch.clamp(aux["query_selection"], 0.0, 1.0)),
            "query_occlusion": _repeat_to_rgb(torch.clamp(aux["query_occlusion"], 0.0, 1.0)),
            "query_direct": torch.clamp(aux["query_direct"], 0.0, 1.0),
            "query_indirect": torch.clamp(aux["query_indirect"], 0.0, 1.0),
            "diffuse": torch.clamp(aux["pbr_diffuse"], 0.0, 1.0),
            "specular": torch.clamp(aux["pbr_specular"], 0.0, 1.0),
        }
    return outputs, gt_rgb, stats


def render_set(model_path, name, iteration, views, gaussians, sop_state, pipeline, opt, background, args, source_info, subset_suffix=""):
    output_name = f"{name}{subset_suffix}"
    if len(views) == 0:
        print(f"No views found for {output_name}, skipping.")
        return

    output_root = os.path.join(model_path, output_name)
    path_prefix = os.path.join(model_path, output_name, f"ours_{iteration}")
    gts_path = os.path.join(path_prefix, "gt")
    keys = [
        "render",
        "albedo",
        "roughness",
        "metallic",
        "weight",
        "depth",
        "normal",
        "depth_normal",
        "query_selection",
        "query_occlusion",
        "query_direct",
        "query_indirect",
        "diffuse",
        "specular",
    ]

    os.makedirs(output_root, exist_ok=True)
    if not args.no_save:
        os.makedirs(gts_path, exist_ok=True)
        for key in keys:
            os.makedirs(os.path.join(path_prefix, key), exist_ok=True)
        env_dict = gaussians.render_env_map()
        if "env1" in env_dict and "env2" in env_dict:
            env_grid = [
                rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)),
                rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)),
            ]
            env_grid = torchvision.utils.make_grid(env_grid, nrow=1, padding=10)
            torchvision.utils.save_image(env_grid, os.path.join(path_prefix, "env.png"))
        else:
            env_image = rgb_to_srgb(env_dict["env"].permute(2, 0, 1))
            torchvision.utils.save_image(env_image, os.path.join(path_prefix, "env.png"))

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0

    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress ({output_name})")):
        _begin_frame_profile()
        _cuda_sync()
        frame_start = time.perf_counter()
        try:
            with _profile_cpu_block("view_total_ms", sync_cuda=True):
                outputs, gt_image, _stats = render_stage2_sop_view(
                    viewpoint_cam=view,
                    gaussians=gaussians,
                    background=background,
                    pipe=pipeline,
                    opt=opt,
                    sop_state=sop_state,
                    args=args,
                    iteration=iteration,
                )

            pred_image = outputs["render"]
            with _profile_cpu_block("metric_ms", sync_cuda=True):
                psnr_avg += psnr(pred_image, gt_image).mean().double().item()
                ssim_avg += ssim(pred_image, gt_image).mean().double().item()
            if not args.no_lpips:
                with _profile_cpu_block("lpips_ms", sync_cuda=True):
                    lpips_avg += lpips(pred_image, gt_image, net_type="vgg").mean().double().item()

            if not args.no_save:
                with _profile_cpu_block("save_ms", sync_cuda=True):
                    save_mask = _view_output_mask(view, gt_image)
                    gt_to_save = gt_image
                    if save_mask is not None:
                        gt_to_save = gt_to_save * save_mask
                    torchvision.utils.save_image(gt_to_save, os.path.join(gts_path, f"{idx:05d}.png"))
                    for key in keys:
                        out = outputs[key]
                        if out.shape[0] == 1:
                            out = out.repeat(3, 1, 1)
                        if save_mask is not None:
                            out = out * save_mask
                        torchvision.utils.save_image(out, os.path.join(path_prefix, key, f"{idx:05d}.png"))
        finally:
            frame_timing = _end_frame_profile()

        frame_timing["frame_total_ms"] = (time.perf_counter() - frame_start) * 1000.0
        tqdm.write(_format_profile_line(output_name, idx, view, frame_timing))

    psnr_avg /= len(views)
    ssim_avg /= len(views)
    if not args.no_lpips:
        lpips_avg /= len(views)

    results_dict = {
        "num_views": len(views),
        "iteration": int(iteration),
        "psnr_avg": psnr_avg,
        "ssim_avg": ssim_avg,
        "lpips_avg": lpips_avg,
        "lpips_enabled": not args.no_lpips,
        "source_info": source_info,
    }
    print(f"\n[ITER {iteration}] Evaluating {output_name} set: PSNR {psnr_avg} SSIM {ssim_avg} LPIPS {lpips_avg}")
    with open(os.path.join(model_path, output_name, "nvs_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(model_path, output_name, "nvs_results.json"))


def render_sets(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        set_gaussian_para(gaussians, opt)

        point_cloud_root = os.path.join(dataset.model_path, "point_cloud")
        load_iteration = -1 if os.path.isdir(point_cloud_root) else None
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

        render_iteration, sop_state, source_info = _load_render_state(gaussians, args)

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

        if gaussians.env_map is not None and hasattr(gaussians.env_map, "update_pdf"):
            gaussians.env_map.update_pdf()

        _prepare_profile_probe_atlas(sop_state)
        _prepare_profile_env_atlas(gaussians)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            train_views, train_suffix = select_views(scene.getTrainCameras(), args.first_k)
            render_set(
                dataset.model_path,
                "train_sop",
                render_iteration,
                train_views,
                gaussians,
                sop_state,
                pipeline,
                opt,
                background,
                args,
                source_info,
                subset_suffix=train_suffix,
            )

        if not args.skip_test:
            test_views, test_suffix = select_views(scene.getTestCameras(), args.first_k)
            render_set(
                dataset.model_path,
                "test_sop",
                render_iteration,
                test_views,
                gaussians,
                sop_state,
                pipeline,
                opt,
                background,
                args,
                source_info,
                subset_suffix=test_suffix,
            )


def _build_parser():
    parser = ArgumentParser(description="Stage2 SOP rendering profile parameters")
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", default=False, action="store_true")
    parser.add_argument("--no_lpips", default=False, action="store_true")
    parser.add_argument("--first_k", default=-1, type=int)

    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--start_checkpoint_refgs", type=str, default="")
    parser.add_argument("--sop_init", type=str, default="")

    parser.add_argument("--lambda_lam", type=float, default=0.001)
    parser.add_argument("--lambda_sops", type=float, default=0.0)
    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--max_shading_points", type=int, default=4096)
    parser.add_argument("--sop_query_radius", type=float, default=0.0)
    parser.add_argument("--sop_query_topk", type=int, default=4)
    parser.add_argument("--sop_query_chunk_size", type=int, default=1024)
    parser.add_argument("--use_env_atlas", action="store_true", default=False)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = get_combined_args(parser)
    print("Rendering Stage2 SOP (profile mode) from " + args.model_path)

    if not getattr(args, "model_path", ""):
        raise RuntimeError("render_sop_profile.py requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("render_sop_profile.py requires --source_path/-s, or a cfg_args under --model_path.")
    if not torch.cuda.is_available():
        raise RuntimeError("render_sop_profile.py currently requires CUDA.")

    safe_state(args.quiet)
    _PROFILE_USE_ENV_ATLAS = bool(args.use_env_atlas)
    _install_profile_hooks()

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    render_sets(dataset, opt, pipe, args)
