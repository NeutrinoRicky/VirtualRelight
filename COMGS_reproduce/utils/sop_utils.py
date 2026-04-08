from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision

try:
    from pytorch3d.ops import knn_points
except ImportError as exc:
    knn_points = None
    _PYTORCH3D_IMPORT_ERROR = exc
else:
    _PYTORCH3D_IMPORT_ERROR = None


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _accumulate_timing(profile_timing: Optional[Dict[str, float]], key: str, delta: float) -> None:
    if profile_timing is None:
        return
    profile_timing[key] = float(profile_timing.get(key, 0.0)) + float(delta)


def _tonemap_for_vis(x: torch.Tensor) -> torch.Tensor:
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(y / (1.0 + y), 0.0, 1.0)


def _repeat_single_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 1:
        return x.expand(*x.shape[:-1], 3)
    return x


@torch.no_grad()
def save_sop_texture_previews(
    output_dir: str,
    sop_tensors: Dict[str, torch.Tensor],
    preview_count: int = 16,
    prefix: str = "sop",
) -> None:
    probe_lin_tex = sop_tensors.get("probe_lin_tex")
    probe_occ_tex = sop_tensors.get("probe_occ_tex")
    if probe_lin_tex is None or probe_occ_tex is None:
        return

    count = min(int(preview_count), int(probe_lin_tex.shape[0]))
    if count <= 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    def _name(stem: str) -> str:
        return f"{prefix}_{stem}.png" if prefix else f"{stem}.png"

    lin = _tonemap_for_vis(torch.clamp(probe_lin_tex[:count].detach().cpu(), min=0.0)).permute(0, 3, 1, 2)
    occ = _repeat_single_channel(torch.clamp(probe_occ_tex[:count].detach().cpu(), 0.0, 1.0)).permute(0, 3, 1, 2)

    torchvision.utils.save_image(lin, os.path.join(output_dir, _name("probe_lin_tex")), nrow=min(4, count))
    torchvision.utils.save_image(occ, os.path.join(output_dir, _name("probe_occ_tex")), nrow=min(4, count))

    if "probe_albedo_tex" in sop_tensors:
        albedo = torch.clamp(sop_tensors["probe_albedo_tex"][:count].detach().cpu(), 0.0, 1.0).permute(0, 3, 1, 2)
        torchvision.utils.save_image(albedo, os.path.join(output_dir, _name("probe_albedo_tex")), nrow=min(4, count))

    if "probe_roughness_tex" in sop_tensors:
        roughness = _repeat_single_channel(
            torch.clamp(sop_tensors["probe_roughness_tex"][:count].detach().cpu(), 0.0, 1.0)
        ).permute(0, 3, 1, 2)
        torchvision.utils.save_image(roughness, os.path.join(output_dir, _name("probe_roughness_tex")), nrow=min(4, count))

    if "probe_metallic_tex" in sop_tensors:
        metallic = _repeat_single_channel(
            torch.clamp(sop_tensors["probe_metallic_tex"][:count].detach().cpu(), 0.0, 1.0)
        ).permute(0, 3, 1, 2)
        torchvision.utils.save_image(metallic, os.path.join(output_dir, _name("probe_metallic_tex")), nrow=min(4, count))


def _query_knn_probes(
    x_chunk: torch.Tensor,
    probe_xyz: torch.Tensor,
    neighbor_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if knn_points is None:
        raise ImportError(
            "pytorch3d.ops.knn_points is required for SOP queries. "
            "Please install pytorch3d in the active environment."
        ) from _PYTORCH3D_IMPORT_ERROR

    knn_dists2, knn_idx, _ = knn_points(
        x_chunk.unsqueeze(0),
        probe_xyz.unsqueeze(0),
        K=neighbor_k,
        return_nn=False,
    )
    knn_dist = torch.sqrt(knn_dists2[0].clamp_min(0.0))
    return knn_dist, knn_idx[0]


def build_sop_neighbor_cache(
    x_world: torch.Tensor,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    radius: Optional[float] = None,
    topk: int = 8,
    eps: float = 1e-8,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    if x_world.dim() != 2 or x_world.shape[-1] != 3:
        raise ValueError(f"Expected x_world with shape [P, 3], got {tuple(x_world.shape)}")

    device = x_world.device
    dtype = x_world.dtype
    probe_xyz = probe_xyz.to(device=device, dtype=dtype)
    probe_normal = _safe_normalize(probe_normal.to(device=device, dtype=dtype), eps=eps)

    num_points = x_world.shape[0]
    num_probes = probe_xyz.shape[0]
    if num_probes == 0:
        return {
            "knn_idx": torch.zeros((num_points, 0), device=device, dtype=torch.long),
            "weights": torch.zeros((num_points, 0), device=device, dtype=dtype),
        }

    neighbor_k = min(max(int(topk), 1), num_probes)
    knn_idx_out = torch.empty((num_points, neighbor_k), device=device, dtype=torch.long)
    weights_out = torch.zeros((num_points, neighbor_k), device=device, dtype=dtype)

    for start in range(0, num_points, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), num_points)
        x_chunk = x_world[start:end]

        knn_dist, knn_idx = _query_knn_probes(x_chunk, probe_xyz, neighbor_k)
        if radius is not None and radius > 0.0:
            neighbor_mask = knn_dist <= float(radius)
            fallback_mask = ~neighbor_mask.any(dim=1, keepdim=True)
            neighbor_mask = torch.where(fallback_mask, torch.ones_like(neighbor_mask), neighbor_mask)
        else:
            neighbor_mask = torch.ones_like(knn_dist, dtype=torch.bool)

        neighbor_xyz = probe_xyz[knn_idx]
        neighbor_normal = probe_normal[knn_idx]
        d = neighbor_xyz - x_chunk[:, None, :]
        dir_k = _safe_normalize(d, eps=eps)

        w_s = 1.0 / knn_dist.clamp_min(eps)
        w_b = 0.5 * (1.0 + torch.sum(dir_k * neighbor_normal, dim=-1)) + 0.01
        weights = w_s * w_b * neighbor_mask.to(dtype=dtype)

        knn_idx_out[start:end] = knn_idx
        weights_out[start:end] = weights

    return {
        "knn_idx": knn_idx_out,
        "weights": weights_out,
    }


def build_view_sop_neighbor_cache(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    radius: Optional[float] = None,
    topk: int = 8,
    eps: float = 1e-8,
    chunk_size: int = 4096,
    storage_device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [H, W, 3], got {tuple(points.shape)}")
    if valid_mask.shape != points.shape[:2]:
        raise ValueError(
            f"Expected valid_mask with shape {tuple(points.shape[:2])}, got {tuple(valid_mask.shape)}"
        )

    store_device = storage_device if storage_device is not None else points.device
    valid_mask = valid_mask.detach().bool()
    flat_valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).squeeze(1)
    flat_points = points.detach().reshape(-1, 3)
    valid_points = flat_points.index_select(0, flat_valid_idx)

    neighbor_cache = build_sop_neighbor_cache(
        x_world=valid_points,
        probe_xyz=probe_xyz,
        probe_normal=probe_normal,
        radius=radius,
        topk=topk,
        eps=eps,
        chunk_size=chunk_size,
    )
    return {
        "valid_mask": valid_mask.to(device=store_device),
        "flat_valid_idx": flat_valid_idx.detach().to(device=store_device),
        "points": valid_points.detach().to(device=store_device),
        "knn_idx": neighbor_cache["knn_idx"].detach().to(device=store_device),
        "weights": neighbor_cache["weights"].detach().to(device=store_device),
    }


def slice_view_sop_neighbor_cache_rows(
    view_cache: Dict[str, torch.Tensor],
    row_idx: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    cache_device = view_cache["flat_valid_idx"].device
    row_idx = row_idx.to(device=cache_device, dtype=torch.long)

    out = {
        "points": view_cache["points"].index_select(0, row_idx),
        "knn_idx": view_cache["knn_idx"].index_select(0, row_idx),
        "weights": view_cache["weights"].index_select(0, row_idx),
    }
    if device is not None:
        out = {key: value.to(device=device) for key, value in out.items()}
    return out


def map_flat_indices_to_view_cache_rows(
    view_cache: Dict[str, torch.Tensor],
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    flat_valid_idx = view_cache["flat_valid_idx"]
    flat_indices = flat_indices.to(device=flat_valid_idx.device, dtype=flat_valid_idx.dtype)
    if flat_indices.numel() == 0:
        return torch.zeros((0,), device=flat_valid_idx.device, dtype=torch.long)

    row_idx = torch.searchsorted(flat_valid_idx, flat_indices)
    if torch.any(row_idx >= flat_valid_idx.shape[0]):
        raise IndexError("Requested flat pixel index is outside the cached valid-pixel range.")
    if not torch.equal(flat_valid_idx.index_select(0, row_idx), flat_indices):
        raise KeyError("Requested flat pixel index is not present in the cached valid-pixel set.")
    return row_idx


def select_view_sop_neighbor_cache(
    view_cache: Dict[str, torch.Tensor],
    flat_indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    row_idx = map_flat_indices_to_view_cache_rows(view_cache, flat_indices)
    return slice_view_sop_neighbor_cache_rows(view_cache, row_idx=row_idx, device=device)


def dir_to_oct_uv(dirs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Map unit directions [..., 3] to octahedral UVs in [0, 1].
    """
    d = _safe_normalize(dirs, eps=eps)
    denom = d.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    p = d / denom

    px = p[..., 0]
    py = p[..., 1]
    pz = p[..., 2]

    fold_x = (1.0 - py.abs()) * torch.sign(px)
    fold_y = (1.0 - px.abs()) * torch.sign(py)
    px = torch.where(pz >= 0.0, px, fold_x)
    py = torch.where(pz >= 0.0, py, fold_y)

    u = px * 0.5 + 0.5
    v = py * 0.5 + 0.5
    return torch.stack([u, v], dim=-1)


def oct_uv_to_dir(uv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Map octahedral UVs [..., 2] in [0, 1] back to unit directions [..., 3].
    """
    f = uv * 2.0 - 1.0
    x = f[..., 0]
    y = f[..., 1]
    z = 1.0 - x.abs() - y.abs()

    t = torch.clamp(-z, min=0.0)
    x = x + torch.where(x >= 0.0, -t, t)
    y = y + torch.where(y >= 0.0, -t, t)

    dirs = torch.stack([x, y, z], dim=-1)
    return _safe_normalize(dirs, eps=eps)


def build_octahedral_direction_grid(
    tex_h: int,
    tex_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return octahedral texel-center directions [tex_h, tex_w, 3].
    """
    xs = (torch.arange(tex_w, device=device, dtype=dtype) + 0.5) / float(max(tex_w, 1))
    ys = (torch.arange(tex_h, device=device, dtype=dtype) + 0.5) / float(max(tex_h, 1))
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    uv = torch.stack([grid_x, grid_y], dim=-1)
    return oct_uv_to_dir(uv)


def sample_octahedral_texture(texture: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample octahedral textures.

    Args:
        texture: [B, H, W, C] or [H, W, C]
        dirs: [B, 3] or [B, N, 3]

    Returns:
        sampled values with shape [B, C] or [B, N, C].
    """
    if texture.dim() == 3:
        texture = texture.unsqueeze(0)
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    if dirs.dim() == 2:
        dirs = dirs.unsqueeze(1)

    if texture.dim() != 4 or dirs.dim() != 3 or dirs.shape[-1] != 3:
        raise ValueError(f"Unexpected shapes: texture={tuple(texture.shape)}, dirs={tuple(dirs.shape)}")

    batch = dirs.shape[0]
    if texture.shape[0] == 1 and batch > 1:
        texture = texture.expand(batch, -1, -1, -1)
    if texture.shape[0] != batch:
        raise ValueError(f"Texture batch {texture.shape[0]} does not match direction batch {batch}")

    uv = dir_to_oct_uv(dirs)
    grid = uv * 2.0 - 1.0
    tex = texture.permute(0, 3, 1, 2).contiguous()
    sampled = F.grid_sample(
        tex,
        grid.view(batch, -1, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    sampled = sampled.view(batch, dirs.shape[1], texture.shape[-1])
    if sampled.shape[1] == 1:
        return sampled[:, 0]
    return sampled


def _sample_joint_probe_textures(
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: torch.Tensor,
    flat_idx: torch.Tensor,
    dirs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_tex = torch.cat(
        [
            probe_lin_tex.index_select(0, flat_idx),
            probe_occ_tex.index_select(0, flat_idx),
        ],
        dim=-1,
    )
    sampled = sample_octahedral_texture(joint_tex, dirs)
    return sampled[..., :3], sampled[..., 3:4]


def query_sops(
    x_world: torch.Tensor,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: torch.Tensor,
    radius: Optional[float] = None,
    topk: int = 8,
    eps: float = 1e-8,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Query SOP indirect radiance and occlusion in world space.

    Args:
        x_world: [P, 3]
        probe_xyz: [K, 3]
        probe_normal: [K, 3]
        probe_lin_tex: [K, Ht, Wt, 3]
        probe_occ_tex: [K, Ht, Wt, 1]

    Returns:
        Lin_x: [P, 3]
        Occ_x: [P, 1]
    """
    if x_world.numel() == 0:
        device = probe_xyz.device if probe_xyz.numel() > 0 else x_world.device
        dtype = probe_xyz.dtype if probe_xyz.numel() > 0 else x_world.dtype
        return (
            torch.zeros((0, 3), device=device, dtype=dtype),
            torch.zeros((0, 1), device=device, dtype=dtype),
        )

    device = x_world.device
    dtype = x_world.dtype
    probe_xyz = probe_xyz.to(device=device, dtype=dtype)
    probe_normal = _safe_normalize(probe_normal.to(device=device, dtype=dtype), eps=eps)
    probe_lin_tex = probe_lin_tex.to(device=device, dtype=dtype)
    probe_occ_tex = probe_occ_tex.to(device=device, dtype=dtype)

    num_points = x_world.shape[0]
    num_probes = probe_xyz.shape[0]
    lin_out = torch.zeros((num_points, 3), device=device, dtype=dtype)
    occ_out = torch.zeros((num_points, 1), device=device, dtype=dtype)

    if num_probes == 0:
        return lin_out, occ_out

    neighbor_k = min(max(int(topk), 1), num_probes)
    for start in range(0, num_points, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), num_points)
        x_chunk = x_world[start:end]

        knn_dist, knn_idx = _query_knn_probes(x_chunk, probe_xyz, neighbor_k)
        if radius is not None and radius > 0.0:
            neighbor_mask = knn_dist <= float(radius)
            fallback_mask = ~neighbor_mask.any(dim=1, keepdim=True)
            neighbor_mask = torch.where(fallback_mask, torch.ones_like(neighbor_mask), neighbor_mask)
        else:
            neighbor_mask = torch.ones_like(knn_dist, dtype=torch.bool)

        neighbor_xyz = probe_xyz[knn_idx]
        neighbor_normal = probe_normal[knn_idx]
        d = neighbor_xyz - x_chunk[:, None, :]
        dir_k = _safe_normalize(d, eps=eps)

        w_s = 1.0 / knn_dist.clamp_min(eps)
        w_b = 0.5 * (1.0 + torch.sum(dir_k * neighbor_normal, dim=-1)) + 0.01
        weights = w_s * w_b * neighbor_mask.to(dtype=dtype)

        flat_idx = knn_idx.reshape(-1)
        flat_dirs = dir_k.reshape(-1, 3)
        sampled_lin, sampled_occ = _sample_joint_probe_textures(
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            flat_idx=flat_idx,
            dirs=flat_dirs,
        )
        sampled_lin = sampled_lin.view(end - start, neighbor_k, 3)
        sampled_occ = sampled_occ.view(end - start, neighbor_k, 1)

        weight_sum = weights.sum(dim=1, keepdim=True)
        valid = weight_sum.squeeze(-1) > eps
        if torch.any(valid):
            lin_vals = torch.sum(weights[..., None] * sampled_lin, dim=1) / weight_sum.clamp_min(eps)
            occ_vals = torch.sum(weights[..., None] * sampled_occ, dim=1) / weight_sum.clamp_min(eps)
            chunk_lin = lin_out[start:end]
            chunk_occ = occ_out[start:end]
            chunk_lin[valid] = lin_vals[valid]
            chunk_occ[valid] = occ_vals[valid]
            lin_out[start:end] = chunk_lin
            occ_out[start:end] = chunk_occ

    return lin_out, torch.clamp(occ_out, 0.0, 1.0)


def query_sops_directional_from_cache(
    query_dirs: torch.Tensor,
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: torch.Tensor,
    neighbor_cache: Dict[str, torch.Tensor],
    eps: float = 1e-8,
    profile_timing: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_dirs.dim() == 2:
        query_dirs = query_dirs.unsqueeze(1)
        squeeze_sample_dim = True
    elif query_dirs.dim() == 3:
        squeeze_sample_dim = False
    else:
        raise ValueError(f"Expected query_dirs with shape [P, 3] or [P, S, 3], got {tuple(query_dirs.shape)}")

    device = query_dirs.device
    dtype = query_dirs.dtype
    query_dirs = _safe_normalize(query_dirs.to(device=device, dtype=dtype), eps=eps)
    probe_lin_tex = probe_lin_tex.to(device=device, dtype=dtype)
    probe_occ_tex = probe_occ_tex.to(device=device, dtype=dtype)
    knn_idx = neighbor_cache["knn_idx"].to(device=device, dtype=torch.long)
    weights = neighbor_cache["weights"].to(device=device, dtype=dtype)

    num_points = query_dirs.shape[0]
    num_samples = query_dirs.shape[1]
    if num_points == 0 or knn_idx.numel() == 0:
        lin_out = torch.zeros((num_points, num_samples, 3), device=device, dtype=dtype)
        occ_out = torch.zeros((num_points, num_samples, 1), device=device, dtype=dtype)
        if squeeze_sample_dim:
            return lin_out[:, 0], occ_out[:, 0]
        return lin_out, occ_out

    neighbor_k = knn_idx.shape[1]
    flat_idx = knn_idx.reshape(-1)
    probe_dirs = query_dirs[:, None, :, :].expand(-1, neighbor_k, -1, -1).reshape(-1, num_samples, 3)
    if profile_timing is not None:
        _sync_if_cuda(device)
        sample_t0 = time.perf_counter()
    sampled_lin, sampled_occ = _sample_joint_probe_textures(
        probe_lin_tex=probe_lin_tex,
        probe_occ_tex=probe_occ_tex,
        flat_idx=flat_idx,
        dirs=probe_dirs,
    )
    sampled_lin = sampled_lin.view(num_points, neighbor_k, num_samples, 3)
    sampled_occ = sampled_occ.view(num_points, neighbor_k, num_samples, 1)
    if profile_timing is not None:
        _sync_if_cuda(device)
        sample_t1 = time.perf_counter()
        _accumulate_timing(profile_timing, "sop_query_sample_sec", sample_t1 - sample_t0)

    if profile_timing is not None:
        _sync_if_cuda(device)
        blend_t0 = time.perf_counter()
    weight_sum = weights.sum(dim=1, keepdim=True)
    lin_out = torch.zeros((num_points, num_samples, 3), device=device, dtype=dtype)
    occ_out = torch.zeros((num_points, num_samples, 1), device=device, dtype=dtype)
    valid = weight_sum.squeeze(-1) > eps
    if torch.any(valid):
        lin_vals = torch.sum(weights[:, :, None, None] * sampled_lin, dim=1) / weight_sum.unsqueeze(-1).clamp_min(eps)
        occ_vals = torch.sum(weights[:, :, None, None] * sampled_occ, dim=1) / weight_sum.unsqueeze(-1).clamp_min(eps)
        lin_out[valid] = lin_vals[valid]
        occ_out[valid] = occ_vals[valid]
    if profile_timing is not None:
        _sync_if_cuda(device)
        blend_t1 = time.perf_counter()
        _accumulate_timing(profile_timing, "sop_query_blend_sec", blend_t1 - blend_t0)

    occ_out = torch.clamp(occ_out, 0.0, 1.0)
    if squeeze_sample_dim:
        return lin_out[:, 0], occ_out[:, 0]
    return lin_out, occ_out


def query_sops_directional(
    x_world: torch.Tensor,
    query_dirs: torch.Tensor,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: torch.Tensor,
    radius: Optional[float] = None,
    topk: int = 8,
    eps: float = 1e-8,
    chunk_size: int = 1024,
    neighbor_cache: Optional[Dict[str, torch.Tensor]] = None,
    profile_timing: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Query SOP indirect radiance and occlusion for explicit incoming directions.

    Args:
        x_world: [P, 3] shading points in world space.
        query_dirs: [P, 3] or [P, S, 3] incoming light directions in world space.
        probe_xyz: [K, 3]
        probe_normal: [K, 3]
        probe_lin_tex: [K, Ht, Wt, 3]
        probe_occ_tex: [K, Ht, Wt, 1]

    Returns:
        Lin_x: [P, 3] or [P, S, 3]
        Occ_x: [P, 1] or [P, S, 1]
    """
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

    if neighbor_cache is not None:
        if x_world.shape[0] != neighbor_cache["knn_idx"].shape[0]:
            raise ValueError(
                f"Neighbor cache size {neighbor_cache['knn_idx'].shape[0]} does not match x_world size {x_world.shape[0]}"
            )
        return query_sops_directional_from_cache(
            query_dirs=query_dirs,
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            neighbor_cache=neighbor_cache,
            eps=eps,
            profile_timing=profile_timing,
        )

    squeeze_sample_dim = False
    if query_dirs.dim() == 2:
        if query_dirs.shape != x_world.shape:
            raise ValueError(
                f"Expected query_dirs with shape [P, 3] matching x_world, got {tuple(query_dirs.shape)} for x_world={tuple(x_world.shape)}"
            )
        query_dirs = query_dirs.unsqueeze(1)
        squeeze_sample_dim = True
    elif query_dirs.dim() != 3 or query_dirs.shape[0] != x_world.shape[0] or query_dirs.shape[-1] != 3:
        raise ValueError(
            f"Expected query_dirs with shape [P, 3] or [P, S, 3], got {tuple(query_dirs.shape)} for x_world={tuple(x_world.shape)}"
        )

    device = x_world.device
    dtype = x_world.dtype
    probe_xyz = probe_xyz.to(device=device, dtype=dtype)
    probe_normal = _safe_normalize(probe_normal.to(device=device, dtype=dtype), eps=eps)
    probe_lin_tex = probe_lin_tex.to(device=device, dtype=dtype)
    probe_occ_tex = probe_occ_tex.to(device=device, dtype=dtype)
    query_dirs = _safe_normalize(query_dirs.to(device=device, dtype=dtype), eps=eps)

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
    for start in range(0, num_points, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), num_points)
        x_chunk = x_world[start:end]
        dirs_chunk = query_dirs[start:end]

        if profile_timing is not None:
            _sync_if_cuda(device)
            knn_t0 = time.perf_counter()
        knn_dist, knn_idx = _query_knn_probes(x_chunk, probe_xyz, neighbor_k)
        if profile_timing is not None:
            _sync_if_cuda(device)
            knn_t1 = time.perf_counter()
            _accumulate_timing(profile_timing, "sop_query_knn_sec", knn_t1 - knn_t0)
            _sync_if_cuda(device)
            prep_t0 = time.perf_counter()
        if radius is not None and radius > 0.0:
            neighbor_mask = knn_dist <= float(radius)
            fallback_mask = ~neighbor_mask.any(dim=1, keepdim=True)
            neighbor_mask = torch.where(fallback_mask, torch.ones_like(neighbor_mask), neighbor_mask)
        else:
            neighbor_mask = torch.ones_like(knn_dist, dtype=torch.bool)

        neighbor_xyz = probe_xyz[knn_idx]
        neighbor_normal = probe_normal[knn_idx]
        d = neighbor_xyz - x_chunk[:, None, :]
        dir_k = _safe_normalize(d, eps=eps)

        w_s = 1.0 / knn_dist.clamp_min(eps)
        w_b = 0.5 * (1.0 + torch.sum(dir_k * neighbor_normal, dim=-1)) + 0.01
        weights = w_s * w_b * neighbor_mask.to(dtype=dtype)
        flat_idx = knn_idx.reshape(-1)
        probe_dirs = dirs_chunk[:, None, :, :].expand(-1, neighbor_k, -1, -1).reshape(-1, num_samples, 3)
        if profile_timing is not None:
            _sync_if_cuda(device)
            prep_t1 = time.perf_counter()
            _accumulate_timing(profile_timing, "sop_query_prep_sec", prep_t1 - prep_t0)
            _sync_if_cuda(device)
            sample_t0 = time.perf_counter()
        sampled_lin, sampled_occ = _sample_joint_probe_textures(
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            flat_idx=flat_idx,
            dirs=probe_dirs,
        )
        sampled_lin = sampled_lin.view(end - start, neighbor_k, num_samples, 3)
        sampled_occ = sampled_occ.view(end - start, neighbor_k, num_samples, 1)
        if profile_timing is not None:
            _sync_if_cuda(device)
            sample_t1 = time.perf_counter()
            _accumulate_timing(profile_timing, "sop_query_sample_sec", sample_t1 - sample_t0)

        if profile_timing is not None:
            _sync_if_cuda(device)
            blend_t0 = time.perf_counter()
        weight_sum = weights.sum(dim=1, keepdim=True)
        valid = weight_sum.squeeze(-1) > eps
        if torch.any(valid):
            lin_vals = torch.sum(weights[:, :, None, None] * sampled_lin, dim=1) / weight_sum.unsqueeze(-1).clamp_min(eps)
            occ_vals = torch.sum(weights[:, :, None, None] * sampled_occ, dim=1) / weight_sum.unsqueeze(-1).clamp_min(eps)
            chunk_lin = lin_out[start:end]
            chunk_occ = occ_out[start:end]
            chunk_lin[valid] = lin_vals[valid]
            chunk_occ[valid] = occ_vals[valid]
            lin_out[start:end] = chunk_lin
            occ_out[start:end] = chunk_occ
        if profile_timing is not None:
            _sync_if_cuda(device)
            blend_t1 = time.perf_counter()
            _accumulate_timing(profile_timing, "sop_query_blend_sec", blend_t1 - blend_t0)

    occ_out = torch.clamp(occ_out, 0.0, 1.0)
    if squeeze_sample_dim:
        return lin_out[:, 0], occ_out[:, 0]
    return lin_out, occ_out


__all__ = [
    "build_sop_neighbor_cache",
    "build_view_sop_neighbor_cache",
    "build_octahedral_direction_grid",
    "dir_to_oct_uv",
    "map_flat_indices_to_view_cache_rows",
    "oct_uv_to_dir",
    "query_sops",
    "query_sops_directional",
    "query_sops_directional_from_cache",
    "sample_octahedral_texture",
    "save_sop_texture_previews",
    "select_view_sop_neighbor_cache",
    "slice_view_sop_neighbor_cache_rows",
]
