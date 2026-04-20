import json
import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
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
_PROFILE_PROBE_ATLAS_NCHW_CACHE = None  # cached (N, 4, H, W) probe atlas
_PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE = None  # cached (N*H*W, C) flat atlas for manual bilinear
_PROFILE_USE_ENV_ATLAS = False
_SOP_NEIGHBOR_BACKEND = "knn"
_SOP_HASH_STATE = None
_SOP_HASH_DEBUG_PRINTED_DIRS = set()
_SOP_HASH_STATIC_EXPORTED_ROOTS = set()
_SOP_HASH_SAVE_STATIC_ONCE = False
_SOP_HASH_SAVE_HITS_PER_FRAME = False
_SOP_HASH_MAX_ROW_CANDIDATES = 4096

# ── FRNN backend state ───────────────────────────────────────────────
_SOP_FRNN_GRID = None          # cached FRNN grid (reusable when probes unchanged)
_SOP_FRNN_RADIUS = 0.0         # fixed search radius
_SOP_FRNN_PROBE_XYZ_BATCHED = None  # (1, P, 3) contiguous float32 on CUDA
try:
    import frnn as _frnn_module
except ImportError:
    _frnn_module = None


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class SparseSOPHashState:
    cell_size: float
    origin: torch.Tensor
    coord_min: torch.Tensor
    coord_extent: torch.Tensor
    key_stride_x: int
    key_stride_y: int
    unique_keys: torch.Tensor
    cell_offsets: torch.Tensor
    sorted_probe_ids: torch.Tensor
    unique_cell_coords: torch.Tensor
    query_offsets: torch.Tensor
    band_offsets: torch.Tensor
    band_bbox_min: torch.Tensor
    band_bbox_max: torch.Tensor
    debug_obj_text: Optional[str] = None
    frame_hit_mask: Optional[torch.Tensor] = None


def _sanitize_filename(name: str) -> str:
    safe = [ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in str(name)]
    token = "".join(safe).strip("._")
    return token if token else "view"


def _build_neighbor_offsets(radius_cells: int, device: torch.device) -> torch.Tensor:
    r = max(0, int(radius_cells))
    axis = torch.arange(-r, r + 1, device=device, dtype=torch.long)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).contiguous()


def _estimate_probe_spacing(
    probe_xyz: torch.Tensor,
    sample_count: int,
    chunk_size: int,
    percentile: float,
) -> float:
    num_probes = int(probe_xyz.shape[0])
    if num_probes <= 1:
        return 1.0

    device = probe_xyz.device
    sample_count = min(num_probes, max(2, int(sample_count)))
    if sample_count == num_probes:
        sample_idx = torch.arange(num_probes, device=device, dtype=torch.long)
    else:
        sample_idx = torch.randperm(num_probes, device=device)[:sample_count]
    sample_pts = probe_xyz.index_select(0, sample_idx)

    min_dist = torch.full((sample_count,), float("inf"), device=device, dtype=probe_xyz.dtype)
    stride = max(1, int(chunk_size))
    for start in range(0, num_probes, stride):
        end = min(start + stride, num_probes)
        chunk_pts = probe_xyz[start:end]
        dists = torch.cdist(sample_pts, chunk_pts)

        chunk_ids = torch.arange(start, end, device=device, dtype=torch.long)
        same_mask = sample_idx[:, None] == chunk_ids[None, :]
        if bool(same_mask.any()):
            dists = dists.masked_fill(same_mask, float("inf"))
        min_dist = torch.minimum(min_dist, dists.min(dim=1).values)

    finite = min_dist[torch.isfinite(min_dist)]
    if finite.numel() == 0:
        return 1.0

    q = float(np.clip(float(percentile) / 100.0, 0.05, 0.95))
    return float(torch.quantile(finite, q).item())


def _pack_cell_coords(
    cell_coords: torch.Tensor,
    coord_min: torch.Tensor,
    coord_extent: torch.Tensor,
    key_stride_x: int,
    key_stride_y: int,
):
    rel = cell_coords - coord_min
    valid = (
        (rel[..., 0] >= 0)
        & (rel[..., 0] < coord_extent[0])
        & (rel[..., 1] >= 0)
        & (rel[..., 1] < coord_extent[1])
        & (rel[..., 2] >= 0)
        & (rel[..., 2] < coord_extent[2])
    )
    keys = rel[..., 0] * int(key_stride_x) + rel[..., 1] * int(key_stride_y) + rel[..., 2]
    keys = torch.where(valid, keys, torch.full_like(keys, -1))
    return keys.to(torch.long), valid


def _build_sparse_hash_debug_obj_text(
    hash_state: SparseSOPHashState,
    max_cells: int,
    cell_coords: Optional[torch.Tensor] = None,
    title: str = "SOP sparse hash non-empty cell wireframe",
) -> str:
    if cell_coords is None:
        cell_coords = hash_state.unique_cell_coords

    total_cells = int(cell_coords.shape[0])
    if total_cells == 0:
        return ""

    if max_cells > 0 and total_cells > max_cells:
        sample_ids = torch.linspace(
            0,
            total_cells - 1,
            steps=max_cells,
            device=cell_coords.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        cell_coords = cell_coords.index_select(0, sample_ids)

    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
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

    mins = hash_state.origin[None, :] + cell_coords.to(hash_state.origin.dtype) * hash_state.cell_size
    mins_np = mins.detach().cpu().numpy().astype(np.float64)

    lines = [
        f"# {title}",
        f"# cells_total={total_cells}",
        f"# cells_exported={mins_np.shape[0]}",
        f"# cell_size={hash_state.cell_size:.9f}",
    ]

    for base_min in mins_np:
        verts = corners * hash_state.cell_size + base_min[None, :]
        for vx, vy, vz in verts:
            lines.append(f"v {vx:.7f} {vy:.7f} {vz:.7f}")

    for cell_i in range(mins_np.shape[0]):
        vbase = cell_i * 8
        for ea, eb in edges:
            lines.append(f"l {vbase + ea} {vbase + eb}")

    return "\n".join(lines) + "\n"


def _save_sparse_hash_static_obj_once(output_root: str) -> None:
    if not _SOP_HASH_SAVE_STATIC_ONCE:
        return
    if _SOP_HASH_STATE is None or not _SOP_HASH_STATE.debug_obj_text:
        return
    if output_root in _SOP_HASH_STATIC_EXPORTED_ROOTS:
        return

    debug_dir = os.path.join(output_root, "sop_hash_cells_3d")
    os.makedirs(debug_dir, exist_ok=True)

    if debug_dir not in _SOP_HASH_DEBUG_PRINTED_DIRS:
        _SOP_HASH_DEBUG_PRINTED_DIRS.add(debug_dir)
        print(f"[SOP-HASH] static non-empty cell wireframe OBJ will be written to {debug_dir}")

    out_path = os.path.join(debug_dir, "static_non_empty_cells.obj")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_SOP_HASH_STATE.debug_obj_text)
    _SOP_HASH_STATIC_EXPORTED_ROOTS.add(output_root)


def _begin_sparse_hash_hit_frame() -> None:
    if not _SOP_HASH_SAVE_HITS_PER_FRAME:
        return
    if _SOP_HASH_STATE is None or _SOP_HASH_STATE.frame_hit_mask is None:
        return
    _SOP_HASH_STATE.frame_hit_mask.zero_()


def _save_sparse_hash_hit_cells_for_frame(output_root: str, frame_idx: int, view_name: str, args) -> None:
    if not _SOP_HASH_SAVE_HITS_PER_FRAME:
        return
    if _SOP_HASH_STATE is None or _SOP_HASH_STATE.frame_hit_mask is None:
        return

    hit_cell_ids = torch.nonzero(_SOP_HASH_STATE.frame_hit_mask, as_tuple=False).squeeze(-1)
    if hit_cell_ids.numel() == 0:
        return

    hit_cell_coords = _SOP_HASH_STATE.unique_cell_coords.index_select(0, hit_cell_ids)
    hit_obj_text = _build_sparse_hash_debug_obj_text(
        _SOP_HASH_STATE,
        max_cells=int(getattr(args, "sop_hash_hit_debug_max_cells", 3000)),
        cell_coords=hit_cell_coords,
        title="SOP sparse hash per-frame shading-hit cells",
    )
    if not hit_obj_text:
        return

    debug_dir = os.path.join(output_root, "sop_hash_hit_cells_3d")
    os.makedirs(debug_dir, exist_ok=True)
    if debug_dir not in _SOP_HASH_DEBUG_PRINTED_DIRS:
        _SOP_HASH_DEBUG_PRINTED_DIRS.add(debug_dir)
        print(f"[SOP-HASH] per-frame shading-hit cell wireframe OBJ will be written to {debug_dir}")

    safe_view = _sanitize_filename(view_name)
    out_path = os.path.join(debug_dir, f"{frame_idx:05d}_{safe_view}.obj")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(hit_obj_text)


def _build_sparse_surface_hash(probe_xyz: torch.Tensor, args) -> SparseSOPHashState:
    if probe_xyz.numel() == 0:
        raise RuntimeError("Cannot build SOP sparse hash without probes.")

    probe_xyz = probe_xyz.detach().contiguous()
    device = probe_xyz.device
    dtype = probe_xyz.dtype

    spacing = _estimate_probe_spacing(
        probe_xyz=probe_xyz,
        sample_count=getattr(args, "sop_hash_spacing_samples", 2048),
        chunk_size=getattr(args, "sop_hash_spacing_chunk", 2048),
        percentile=getattr(args, "sop_hash_spacing_percentile", 50.0),
    )
    override_cell_size = float(getattr(args, "sop_hash_cell_size", 0.0))
    if override_cell_size > 0.0:
        cell_size = override_cell_size
    else:
        cell_scale = max(1e-4, float(getattr(args, "sop_hash_cell_scale", 1.5)))
        cell_size = max(1e-6, spacing * cell_scale)

    origin = probe_xyz.min(dim=0).values - 0.5 * cell_size
    cell_coords = torch.floor((probe_xyz - origin[None, :]) / cell_size).to(torch.long)

    coord_min = cell_coords.min(dim=0).values
    coord_max = cell_coords.max(dim=0).values
    coord_extent = coord_max - coord_min + 1

    ex = int(coord_extent[0].item())
    ey = int(coord_extent[1].item())
    ez = int(coord_extent[2].item())
    key_stride_y = ez
    key_stride_x = ey * ez
    max_key = (ex - 1) * key_stride_x + (ey - 1) * key_stride_y + (ez - 1)
    if max_key >= (2**63 - 1):
        raise RuntimeError("SOP sparse hash key range overflow. Try a larger cell size.")

    packed_keys, _ = _pack_cell_coords(
        cell_coords=cell_coords,
        coord_min=coord_min,
        coord_extent=coord_extent,
        key_stride_x=key_stride_x,
        key_stride_y=key_stride_y,
    )
    sorted_keys, sort_idx = torch.sort(packed_keys)
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    cell_offsets = torch.zeros(unique_keys.shape[0] + 1, device=device, dtype=torch.long)
    cell_offsets[1:] = torch.cumsum(counts.to(torch.long), dim=0)
    sorted_probe_ids = sort_idx.to(torch.long)

    sorted_cell_coords = cell_coords.index_select(0, sorted_probe_ids)
    first_indices = cell_offsets[:-1]
    unique_cell_coords = sorted_cell_coords.index_select(0, first_indices)

    query_cells = max(0, int(getattr(args, "sop_hash_query_radius_cells", 1)))
    band_cells = max(query_cells, int(getattr(args, "sop_hash_surface_band_cells", 3)))
    query_offsets = _build_neighbor_offsets(query_cells, device=device)
    band_offsets = _build_neighbor_offsets(band_cells, device=device)

    band_world = float(band_cells) * float(cell_size)
    probe_min = probe_xyz.min(dim=0).values
    probe_max = probe_xyz.max(dim=0).values
    band_bbox_min = probe_min - torch.tensor(band_world, device=device, dtype=dtype)
    band_bbox_max = probe_max + torch.tensor(band_world, device=device, dtype=dtype)

    hash_state = SparseSOPHashState(
        cell_size=float(cell_size),
        origin=origin,
        coord_min=coord_min,
        coord_extent=coord_extent,
        key_stride_x=int(key_stride_x),
        key_stride_y=int(key_stride_y),
        unique_keys=unique_keys,
        cell_offsets=cell_offsets,
        sorted_probe_ids=sorted_probe_ids,
        unique_cell_coords=unique_cell_coords,
        query_offsets=query_offsets,
        band_offsets=band_offsets,
        band_bbox_min=band_bbox_min,
        band_bbox_max=band_bbox_max,
    )
    hash_state.debug_obj_text = _build_sparse_hash_debug_obj_text(
        hash_state,
        max_cells=int(getattr(args, "sop_hash_debug_max_cells", 5000)),
    )

    print(
        "[SOP-HASH] built sparse surface hash: "
        f"probes={int(probe_xyz.shape[0])} "
        f"non_empty_cells={int(unique_keys.shape[0])} "
        f"cell_size={hash_state.cell_size:.7f} "
        f"spacing_est={spacing:.7f} "
        f"query_cells={query_cells} band_cells={band_cells}"
    )
    return hash_state


def _lookup_nonempty_neighbor_cells(
    point_cell_coords: torch.Tensor,
    offsets: torch.Tensor,
    hash_state: SparseSOPHashState,
):
    num_points = int(point_cell_coords.shape[0])
    device = point_cell_coords.device
    empty = torch.empty((0,), device=device, dtype=torch.long)
    if num_points == 0 or offsets.numel() == 0:
        return empty, empty

    num_offsets = int(offsets.shape[0])
    neighbor_coords = point_cell_coords[:, None, :] + offsets[None, :, :]
    keys, key_valid = _pack_cell_coords(
        cell_coords=neighbor_coords,
        coord_min=hash_state.coord_min,
        coord_extent=hash_state.coord_extent,
        key_stride_x=hash_state.key_stride_x,
        key_stride_y=hash_state.key_stride_y,
    )

    flat_valid = key_valid.reshape(-1)
    if not bool(flat_valid.any()):
        return empty, empty

    flat_keys = keys.reshape(-1)[flat_valid]
    search_pos = torch.searchsorted(hash_state.unique_keys, flat_keys)
    in_range = search_pos < hash_state.unique_keys.shape[0]
    matched = torch.zeros_like(in_range, dtype=torch.bool)
    if bool(in_range.any()):
        pos_in = search_pos[in_range]
        matched[in_range] = hash_state.unique_keys.index_select(0, pos_in) == flat_keys[in_range]
    if not bool(matched.any()):
        return empty, empty

    point_ids = torch.arange(num_points, device=device, dtype=torch.long).repeat_interleave(num_offsets)
    point_ids = point_ids[flat_valid][matched]
    cell_ids = search_pos[matched]
    return point_ids, cell_ids


def _query_frnn_neighbors(
    x_chunk: torch.Tensor,
    probe_xyz: torch.Tensor,
    neighbor_k: int,
) -> tuple:
    """Fixed-radius nearest neighbor search via the FRNN CUDA library."""
    global _SOP_FRNN_GRID
    if _frnn_module is None:
        raise RuntimeError("FRNN backend requested but `frnn` is not installed. "
                           "Install from submodules/FRNN.")

    num_points = int(x_chunk.shape[0])
    device = x_chunk.device
    dtype = x_chunk.dtype

    knn_dist = torch.full((num_points, neighbor_k), float("inf"), device=device, dtype=dtype)
    knn_idx = torch.zeros((num_points, neighbor_k), device=device, dtype=torch.long)
    row_valid = torch.zeros((num_points,), device=device, dtype=torch.bool)

    if num_points == 0 or _SOP_FRNN_PROBE_XYZ_BATCHED is None:
        return knn_dist, knn_idx, row_valid

    # FRNN expects (N, P, 3) float32 on CUDA
    query_pts = x_chunk.unsqueeze(0).to(dtype=torch.float32).contiguous()
    ref_pts = _SOP_FRNN_PROBE_XYZ_BATCHED  # already (1, P2, 3) float32

    dists2, idxs, _nn, grid_out = _frnn_module.frnn_grid_points(
        query_pts, ref_pts,
        lengths1=None, lengths2=None,
        K=neighbor_k,
        r=_SOP_FRNN_RADIUS,
        grid=_SOP_FRNN_GRID,
        return_nn=False,
        return_sorted=True,
    )
    _SOP_FRNN_GRID = grid_out  # cache for subsequent chunks

    idxs_0 = idxs[0]        # (P1, K)
    dists2_0 = dists2[0]    # (P1, K)
    valid_mask = idxs_0 >= 0
    # replace invalid entries
    idxs_0 = idxs_0.clamp_min(0).to(torch.long)
    dists2_0 = dists2_0.to(dtype)
    dists2_0[~valid_mask] = float("inf")

    knn_dist = torch.sqrt(dists2_0.clamp_min(0.0))
    knn_idx = idxs_0
    row_valid = valid_mask.any(dim=1)

    return knn_dist, knn_idx, row_valid


def _query_sparse_hash_neighbors(
    x_chunk: torch.Tensor,
    probe_xyz: torch.Tensor,
    neighbor_k: int,
    hash_state: Optional[SparseSOPHashState],
):
    num_points = int(x_chunk.shape[0])
    device = x_chunk.device
    dtype = x_chunk.dtype

    knn_dist = torch.full((num_points, neighbor_k), float("inf"), device=device, dtype=dtype)
    knn_idx = torch.zeros((num_points, neighbor_k), device=device, dtype=torch.long)
    row_valid = torch.zeros((num_points,), device=device, dtype=torch.bool)

    if hash_state is None or num_points == 0 or hash_state.unique_keys.numel() == 0:
        return knn_dist, knn_idx, row_valid

    bbox_mask = ((x_chunk >= hash_state.band_bbox_min) & (x_chunk <= hash_state.band_bbox_max)).all(dim=1)
    if not bool(bbox_mask.any()):
        return knn_dist, knn_idx, row_valid

    candidate_rows = torch.nonzero(bbox_mask, as_tuple=False).squeeze(-1)
    x_candidates = x_chunk.index_select(0, candidate_rows)
    point_cell_coords = torch.floor((x_candidates - hash_state.origin[None, :]) / hash_state.cell_size).to(torch.long)

    band_point_ids, _ = _lookup_nonempty_neighbor_cells(
        point_cell_coords=point_cell_coords,
        offsets=hash_state.band_offsets,
        hash_state=hash_state,
    )
    if band_point_ids.numel() == 0:
        return knn_dist, knn_idx, row_valid

    band_mask = torch.zeros((x_candidates.shape[0],), device=device, dtype=torch.bool)
    band_mask[band_point_ids] = True
    candidate_rows = candidate_rows[band_mask]
    if candidate_rows.numel() == 0:
        return knn_dist, knn_idx, row_valid

    x_candidates = x_chunk.index_select(0, candidate_rows)
    point_cell_coords = point_cell_coords[band_mask]

    query_point_ids, query_cell_ids = _lookup_nonempty_neighbor_cells(
        point_cell_coords=point_cell_coords,
        offsets=hash_state.query_offsets,
        hash_state=hash_state,
    )
    if query_point_ids.numel() == 0:
        return knn_dist, knn_idx, row_valid

    query_rows = candidate_rows.index_select(0, query_point_ids)
    starts = hash_state.cell_offsets.index_select(0, query_cell_ids)
    ends = hash_state.cell_offsets.index_select(0, query_cell_ids + 1)
    counts = (ends - starts).to(torch.long)
    nonzero = counts > 0
    if not bool(nonzero.any()):
        return knn_dist, knn_idx, row_valid

    query_cell_ids = query_cell_ids[nonzero]
    if _SOP_HASH_SAVE_HITS_PER_FRAME and hash_state.frame_hit_mask is not None:
        hash_state.frame_hit_mask[query_cell_ids] = True

    query_rows = query_rows[nonzero]
    starts = starts[nonzero]
    counts = counts[nonzero]

    pair_ids = torch.repeat_interleave(torch.arange(counts.shape[0], device=device, dtype=torch.long), counts)
    if pair_ids.numel() == 0:
        return knn_dist, knn_idx, row_valid

    seg_starts = torch.cumsum(counts, dim=0) - counts
    local_ids = torch.arange(pair_ids.shape[0], device=device, dtype=torch.long) - seg_starts.index_select(0, pair_ids)

    gather_offsets = starts.index_select(0, pair_ids) + local_ids
    flat_probe_ids = hash_state.sorted_probe_ids.index_select(0, gather_offsets)
    flat_rows = query_rows.index_select(0, pair_ids)

    flat_delta = probe_xyz.index_select(0, flat_probe_ids) - x_chunk.index_select(0, flat_rows)
    flat_dist = torch.linalg.norm(flat_delta, dim=-1)

    order = torch.argsort(flat_rows)
    rows_sorted = flat_rows.index_select(0, order)
    probe_sorted = flat_probe_ids.index_select(0, order)
    dist_sorted = flat_dist.index_select(0, order)

    row_counts = torch.bincount(rows_sorted, minlength=num_points)
    if row_counts.numel() == 0 or int(row_counts.max().item()) <= 0:
        return knn_dist, knn_idx, row_valid

    max_row_candidates = int(row_counts.max().item())
    if _SOP_HASH_MAX_ROW_CANDIDATES > 0 and max_row_candidates > _SOP_HASH_MAX_ROW_CANDIDATES:
        fallback_fn = _ORIG_QUERY_KNN_PROBES if _ORIG_QUERY_KNN_PROBES is not None else sop_utils_module._query_knn_probes
        if candidate_rows.numel() > 0:
            fallback_dist, fallback_idx = fallback_fn(
                x_chunk.index_select(0, candidate_rows),
                probe_xyz,
                neighbor_k,
            )
            knn_dist[candidate_rows] = fallback_dist
            knn_idx[candidate_rows] = fallback_idx
            row_valid[candidate_rows] = True
        return knn_dist, knn_idx, row_valid

    row_offsets = torch.zeros(num_points + 1, device=device, dtype=torch.long)
    row_offsets[1:] = torch.cumsum(row_counts.to(torch.long), dim=0)
    start_for_each = row_offsets.index_select(0, rows_sorted)
    pos_in_row = torch.arange(rows_sorted.shape[0], device=device, dtype=torch.long) - start_for_each

    dense_dist = torch.full(
        (num_points, max_row_candidates),
        float("inf"),
        device=device,
        dtype=dtype,
    )
    dense_idx = torch.zeros((num_points, max_row_candidates), device=device, dtype=torch.long)
    dense_dist[rows_sorted, pos_in_row] = dist_sorted
    dense_idx[rows_sorted, pos_in_row] = probe_sorted

    k_eff = min(neighbor_k, max_row_candidates)
    top_dist, top_slots = torch.topk(dense_dist, k=k_eff, dim=1, largest=False, sorted=True)
    top_idx = torch.gather(dense_idx, dim=1, index=top_slots)
    knn_dist[:, :k_eff] = top_dist
    knn_idx[:, :k_eff] = top_idx
    row_valid = row_counts > 0

    return knn_dist, knn_idx, row_valid


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
        num_probes, C, H, W = probe_atlas_nchw.shape

        # Pre-compute bilinear indices & weights from uv (shared across all neighbors)
        idx00, idx10, idx01, idx11, w00, w10, w01, w11 = _bilinear_texel_indices(
            uv, H, W, wrap_u=False,
        )

        # Compute per-probe texel base offset: probe_id * (H*W)
        hw = H * W
        probe_texel_base = probe_ids * hw  # (P, K)

    with _profile_cuda_block("atlas_sample_gather_ms"):
        # Use cached flat atlas (N*H*W, C) to avoid per-call permute+reshape
        global _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE
        if (
            _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE is None
            or _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE.shape[0] != num_probes * hw
        ):
            _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE = (
                probe_atlas_nchw.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            )
        atlas_flat = _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE

        # Gather: for each (point, neighbor, sample), compute linear index = base + texel_offset
        linear_00 = probe_texel_base[:, :, None] + idx00[:, None, :]  # (P, K, S)
        linear_10 = probe_texel_base[:, :, None] + idx10[:, None, :]
        linear_01 = probe_texel_base[:, :, None] + idx01[:, None, :]
        linear_11 = probe_texel_base[:, :, None] + idx11[:, None, :]

        # Flatten and gather all 4 corners at once
        flat_size = num_points * neighbor_k * num_samples
        all_indices = torch.stack([linear_00, linear_10, linear_01, linear_11], dim=0)  # (4, P, K, S)
        all_gathered = atlas_flat[all_indices.reshape(4, -1).reshape(-1)]  # (4*flat_size, C)
        all_gathered = all_gathered.view(4, num_points, neighbor_k, num_samples, C)

        # Bilinear weights: (P, S) -> (P, 1, S, 1) for broadcasting
        dtype_out = probe_atlas_nchw.dtype
        w00_b = w00[:, None, :, None].to(dtype=dtype_out)
        w10_b = w10[:, None, :, None].to(dtype=dtype_out)
        w01_b = w01[:, None, :, None].to(dtype=dtype_out)
        w11_b = w11[:, None, :, None].to(dtype=dtype_out)

        sampled = (all_gathered[0] * w00_b
                 + all_gathered[1] * w10_b
                 + all_gathered[2] * w01_b
                 + all_gathered[3] * w11_b)

    return sampled  # (P, K, S, C)


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
        # Use cached NCHW atlas to avoid repeated reshape/permute
        global _PROFILE_PROBE_ATLAS_NCHW_CACHE, _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE
        if (
            _PROFILE_PROBE_ATLAS_NCHW_CACHE is None
            or _PROFILE_PROBE_ATLAS_NCHW_CACHE.data_ptr() != probe_atlas_flat.data_ptr()
            or _PROFILE_PROBE_ATLAS_NCHW_CACHE.shape[0] != _num_probe_tex
        ):
            _PROFILE_PROBE_ATLAS_NCHW_CACHE = (
                probe_atlas_flat.view(_num_probe_tex, probe_tex_h, probe_tex_w, 4)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            _PROFILE_PROBE_ATLAS_FLAT_HWC_CACHE = None  # invalidate derived cache
        probe_atlas_nchw = _PROFILE_PROBE_ATLAS_NCHW_CACHE
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
    use_sparse_hash = _SOP_NEIGHBOR_BACKEND == "sparse_hash" and _SOP_HASH_STATE is not None
    use_frnn = _SOP_NEIGHBOR_BACKEND == "frnn"
    for start in range(0, num_points, query_stride):
        end = min(start + query_stride, num_points)
        x_chunk = x_world[start:end]
        dirs_chunk = query_dirs[start:end]

        if use_frnn:
            with _profile_cuda_block("frnn_neighbor_ms"):
                knn_dist, knn_idx, row_valid = _query_frnn_neighbors(
                    x_chunk=x_chunk,
                    probe_xyz=probe_xyz,
                    neighbor_k=neighbor_k,
                )
        elif use_sparse_hash:
            with _profile_cuda_block("hash_neighbor_ms"):
                knn_dist, knn_idx, row_valid = _query_sparse_hash_neighbors(
                    x_chunk=x_chunk,
                    probe_xyz=probe_xyz,
                    neighbor_k=neighbor_k,
                    hash_state=_SOP_HASH_STATE,
                )
        else:
            knn_dist, knn_idx = sop_utils_module._query_knn_probes(x_chunk, probe_xyz, neighbor_k)
            row_valid = torch.ones((end - start,), device=device, dtype=torch.bool)

        if not bool(row_valid.any()):
            continue

        active_local = torch.nonzero(row_valid, as_tuple=False).squeeze(-1)
        x_active = x_chunk.index_select(0, active_local)
        dirs_active = dirs_chunk.index_select(0, active_local)
        knn_dist_active = knn_dist.index_select(0, active_local)
        knn_idx_active = knn_idx.index_select(0, active_local)

        with _profile_cuda_block("query_sop_weight_ms"):
            weights = sop_utils_module._compute_neighbor_weights(
                x_chunk=x_active,
                probe_xyz=probe_xyz,
                probe_normal=probe_normal,
                knn_dist=knn_dist_active,
                knn_idx=knn_idx_active,
                radius=radius,
                eps=eps,
            )

        with _profile_cuda_block("query_sop_uv_ms"):
            uv_chunk = sop_utils_module.dir_to_oct_uv(dirs_active).clamp(0.0, 1.0)
        with _profile_cuda_block("atlas_sample_ms"):
            sampled = _sample_probe_atlas_bilinear_flat(
                probe_atlas_nchw=probe_atlas_nchw,
                probe_ids=knn_idx_active,
                uv=uv_chunk,
            )
        with _profile_cuda_block("query_sop_fuse_ms"):
            with _profile_cuda_block("query_sop_fuse_reduce_ms"):
                sampled_lin = sampled[..., :3]   # (A, K, S, 3)
                sampled_occ = sampled[..., 3:4]  # (A, K, S, 1)

                weight_sum = weights.sum(dim=1, keepdim=True)  # (A, 1)
                valid = weight_sum.squeeze(-1) > eps  # (A,)
                denom = weight_sum.clamp_min(eps)  # (A, 1)
                # Use einsum to avoid broadcasting large intermediates
                # weights: (A, K), sampled_lin: (A, K, S, 3)
                lin_vals = torch.einsum('ak,aksc->asc', weights, sampled_lin) / denom[:, :, None]
                occ_vals = torch.einsum('ak,aksc->asc', weights, sampled_occ) / denom[:, :, None]

            with _profile_cuda_block("query_sop_fuse_write_ms"):
                if bool(valid.any()):
                    write_local = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    write_ids = (active_local + start).index_select(0, write_local)
                    lin_out[write_ids] = lin_vals.index_select(0, write_local)
                    occ_out[write_ids] = occ_vals.index_select(0, write_local)

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
    q_hash = timing.get("hash_neighbor_ms", 0.0)
    q_frnn = timing.get("frnn_neighbor_ms", 0.0)
    q_weight = timing.get("query_sop_weight_ms", 0.0)
    q_uv = timing.get("query_sop_uv_ms", 0.0)
    q_sample = timing.get("atlas_sample_ms", 0.0)
    q_sample_index = timing.get("atlas_sample_index_ms", 0.0)
    q_sample_gather = timing.get("atlas_sample_gather_ms", 0.0)
    q_fuse = timing.get("query_sop_fuse_ms", 0.0)
    q_fuse_reduce = timing.get("query_sop_fuse_reduce_ms", 0.0)
    q_fuse_write = timing.get("query_sop_fuse_write_ms", 0.0)
    q_post = timing.get("query_sop_post_ms", 0.0)
    q_accounted = q_prep + q_knn + q_hash + q_frnn + q_weight + q_uv + q_sample + q_fuse + q_post
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
        f"[prep={q_prep:6.2f} knn={q_knn:6.2f} hash={q_hash:6.2f} frnn={q_frnn:6.2f} w={q_weight:6.2f} uv={q_uv:6.2f} "
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


def _prepare_sop_neighbor_backend(sop_state: Stage2SOPState, args) -> None:
    global _SOP_NEIGHBOR_BACKEND
    global _SOP_HASH_STATE
    global _SOP_HASH_DEBUG_PRINTED_DIRS
    global _SOP_HASH_STATIC_EXPORTED_ROOTS
    global _SOP_HASH_SAVE_STATIC_ONCE
    global _SOP_HASH_SAVE_HITS_PER_FRAME
    global _SOP_HASH_MAX_ROW_CANDIDATES

    backend = str(getattr(args, "sop_neighbor_backend", "knn")).strip().lower()
    if backend not in {"knn", "sparse_hash", "frnn"}:
        raise ValueError(f"Unsupported SOP neighbor backend: {backend}")

    _SOP_HASH_SAVE_STATIC_ONCE = bool(
        getattr(args, "sop_hash_save_static_cells_once", False)
        or getattr(args, "sop_hash_save_cells_per_frame", False)
    )
    _SOP_HASH_SAVE_HITS_PER_FRAME = bool(getattr(args, "sop_hash_save_hit_cells_per_frame", False))
    _SOP_HASH_MAX_ROW_CANDIDATES = max(0, int(getattr(args, "sop_hash_max_row_candidates", 4096)))

    _SOP_NEIGHBOR_BACKEND = backend
    _SOP_HASH_STATE = None
    _SOP_HASH_DEBUG_PRINTED_DIRS.clear()
    _SOP_HASH_STATIC_EXPORTED_ROOTS.clear()

    if backend == "sparse_hash":
        _SOP_HASH_STATE = _build_sparse_surface_hash(sop_state.probe_xyz, args)

        if _SOP_HASH_SAVE_HITS_PER_FRAME:
            _SOP_HASH_STATE.frame_hit_mask = torch.zeros(
                (_SOP_HASH_STATE.unique_keys.shape[0],),
                device=_SOP_HASH_STATE.unique_keys.device,
                dtype=torch.bool,
            )
            print("[SOP-HASH] per-frame shading-hit cell export is enabled.")
        else:
            _SOP_HASH_STATE.frame_hit_mask = None

        if _SOP_HASH_SAVE_STATIC_ONCE:
            print("[SOP-HASH] static non-empty cell export is enabled (once per render set).")
        if _SOP_HASH_MAX_ROW_CANDIDATES > 0:
            print(f"[SOP-HASH] max per-row sparse candidates={_SOP_HASH_MAX_ROW_CANDIDATES} (fallback to KNN if exceeded).")
    elif backend == "frnn":
        global _SOP_FRNN_GRID, _SOP_FRNN_RADIUS, _SOP_FRNN_PROBE_XYZ_BATCHED
        if _frnn_module is None:
            raise RuntimeError(
                "FRNN backend requested but `frnn` package is not installed. "
                "Please install from submodules/FRNN."
            )
        frnn_r = float(getattr(args, "sop_frnn_radius", 0.0))
        if frnn_r <= 0.0:
            # auto-estimate: use probe spacing as default radius
            spacing = _estimate_probe_spacing(
                sop_state.probe_xyz,
                sample_count=getattr(args, "sop_hash_spacing_samples", 2048),
                chunk_size=getattr(args, "sop_hash_spacing_chunk", 2048),
                percentile=getattr(args, "sop_hash_spacing_percentile", 50.0),
            )
            frnn_scale = max(1e-4, float(getattr(args, "sop_frnn_radius_scale", 3.0)))
            frnn_r = spacing * frnn_scale
        _SOP_FRNN_RADIUS = frnn_r
        _SOP_FRNN_GRID = None  # will be built on first query and cached
        _SOP_FRNN_PROBE_XYZ_BATCHED = (
            sop_state.probe_xyz.detach().unsqueeze(0).to(dtype=torch.float32).contiguous()
        )
        print(
            f"[SOP-FRNN] using FRNN backend: "
            f"probes={int(sop_state.probe_xyz.shape[0])} "
            f"radius={_SOP_FRNN_RADIUS:.7f}"
        )
    else:
        print("[SOP-HASH] using KNN backend for SOP neighbor query.")
        if _SOP_HASH_SAVE_STATIC_ONCE or _SOP_HASH_SAVE_HITS_PER_FRAME:
            print("[SOP-HASH] debug cell export is ignored because backend is knn.")


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

    if _SOP_NEIGHBOR_BACKEND == "sparse_hash":
        _save_sparse_hash_static_obj_once(output_root)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress ({output_name})")):
        view_name = getattr(view, "image_name", f"{idx:05d}")
        if _SOP_NEIGHBOR_BACKEND == "sparse_hash":
            _begin_sparse_hash_hit_frame()

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

        if _SOP_NEIGHBOR_BACKEND == "sparse_hash":
            _save_sparse_hash_hit_cells_for_frame(output_root, idx, view_name, args)

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
        _prepare_sop_neighbor_backend(sop_state, args)

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

    parser.add_argument(
        "--sop_neighbor_backend",
        type=str,
        default="knn",
        choices=["knn", "sparse_hash", "frnn"],
        help="SOP neighbor search backend: exact KNN, surface-adaptive sparse hash, or FRNN fixed-radius.",
    )
    parser.add_argument(
        "--sop_hash_cell_size",
        type=float,
        default=0.0,
        help="Override sparse-hash cell size. <=0 uses adaptive spacing statistics.",
    )
    parser.add_argument(
        "--sop_hash_cell_scale",
        type=float,
        default=1.5,
        help="Adaptive cell size = spacing_estimate * sop_hash_cell_scale.",
    )
    parser.add_argument("--sop_hash_spacing_samples", type=int, default=2048)
    parser.add_argument("--sop_hash_spacing_chunk", type=int, default=2048)
    parser.add_argument("--sop_hash_spacing_percentile", type=float, default=50.0)
    parser.add_argument("--sop_hash_query_radius_cells", type=int, default=1)
    parser.add_argument("--sop_hash_surface_band_cells", type=int, default=3)
    parser.add_argument(
        "--sop_hash_save_static_cells_once",
        action="store_true",
        default=False,
        help="Export sparse-hash non-empty cells once per render set as 3D wireframe OBJ.",
    )
    parser.add_argument(
        "--sop_hash_save_cells_per_frame",
        action="store_true",
        default=False,
        help="Deprecated alias of --sop_hash_save_static_cells_once.",
    )
    parser.add_argument(
        "--sop_hash_save_hit_cells_per_frame",
        action="store_true",
        default=False,
        help="Export per-frame cells actually touched by shading-point sparse-hash queries.",
    )
    parser.add_argument("--sop_hash_debug_max_cells", type=int, default=5000)
    parser.add_argument("--sop_hash_hit_debug_max_cells", type=int, default=3000)
    parser.add_argument(
        "--sop_hash_max_row_candidates",
        type=int,
        default=4096,
        help="Upper bound of sparse-hash probe candidates per shading row; larger values may be slow.",
    )

    # ── FRNN backend arguments ────────────────────────────────────────
    parser.add_argument(
        "--sop_frnn_radius",
        type=float,
        default=0.0,
        help="Fixed search radius for FRNN backend. <=0 means auto-estimate from probe spacing.",
    )
    parser.add_argument(
        "--sop_frnn_radius_scale",
        type=float,
        default=3.0,
        help="When auto-estimating FRNN radius: radius = spacing_estimate * sop_frnn_radius_scale.",
    )

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
