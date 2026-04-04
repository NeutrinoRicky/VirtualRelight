from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from pytorch3d.ops import knn_points
except ImportError as exc:
    knn_points = None
    _PYTORCH3D_IMPORT_ERROR = exc
else:
    _PYTORCH3D_IMPORT_ERROR = None


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


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
        sampled_lin = sample_octahedral_texture(probe_lin_tex.index_select(0, flat_idx), flat_dirs).view(end - start, neighbor_k, 3)
        sampled_occ = sample_octahedral_texture(probe_occ_tex.index_select(0, flat_idx), flat_dirs).view(end - start, neighbor_k, 1)

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
        probe_dirs = dirs_chunk[:, None, :, :].expand(-1, neighbor_k, -1, -1).reshape(-1, num_samples, 3)
        sampled_lin = sample_octahedral_texture(
            probe_lin_tex.index_select(0, flat_idx),
            probe_dirs,
        ).view(end - start, neighbor_k, num_samples, 3)
        sampled_occ = sample_octahedral_texture(
            probe_occ_tex.index_select(0, flat_idx),
            probe_dirs,
        ).view(end - start, neighbor_k, num_samples, 1)

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

    occ_out = torch.clamp(occ_out, 0.0, 1.0)
    if squeeze_sample_dim:
        return lin_out[:, 0], occ_out[:, 0]
    return lin_out, occ_out


__all__ = [
    "build_octahedral_direction_grid",
    "dir_to_oct_uv",
    "oct_uv_to_dir",
    "query_sops",
    "query_sops_directional",
    "sample_octahedral_texture",
]
