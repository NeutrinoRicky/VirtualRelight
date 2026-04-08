from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from utils.deferred_pbr_comgs import (
    compute_view_directions,
    integrate_incident_radiance,
    recover_shading_points,
    sample_hemisphere_hammersley,
)
from utils.loss_utils import ssim
from utils.losses_comgs_stage1 import compute_d2n_loss, compute_mask_loss
from utils.sop_utils import query_sops_directional, select_view_sop_neighbor_cache


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0
    if int(mask.sum().item()) == 0:
        return value.new_tensor(0.0)
    return value[mask].mean()


def _get_supervision_mask(viewpoint_camera, ref_tensor: torch.Tensor):
    gt_mask = getattr(viewpoint_camera, "gt_alpha_mask", None)
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


def _scatter_to_image(values: torch.Tensor, flat_indices: torch.Tensor, height: int, width: int, channels: int) -> torch.Tensor:
    canvas = values.new_zeros((height * width, channels))
    if flat_indices.numel() > 0:
        canvas[flat_indices] = values
    return canvas.reshape(height, width, channels).permute(2, 0, 1)


def compute_stage2_sop_loss(
    render_pkg: Dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    viewpoint_camera,
    envmap,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    probe_lin_tex: torch.Tensor,
    probe_occ_tex: torch.Tensor,
    lambda_lam: float = 0.001,
    lambda_sops: float = 1.0,
    lambda_d2n: float = 0.05,
    lambda_mask: float = 0.05,
    use_mask_loss: bool = False,
    num_shading_samples: int = 128,
    max_shading_points: int = 4096,
    sop_query_radius: float = 0.0,
    sop_query_topk: int = 8,
    sop_query_chunk_size: int = 1024,
    use_sop_supervision: bool = False,
    tracer=None,
    max_sop_supervision_points: int = 1024,
    trace_bias: float = 1e-3,
    secondary_num_samples: int = 16,
    randomized_samples: bool = True,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
    view_neighbor_cache: Dict[str, torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if view_neighbor_cache is not None:
        valid_mask = view_neighbor_cache["valid_mask"].to(device=gt_rgb.device)
        points = None
    else:
        points, valid_mask = recover_shading_points(
            view=viewpoint_camera,
            depth_unbiased=render_pkg["depth_unbiased"],
            weight=render_pkg["weight"],
            weight_threshold=weight_threshold,
            eps=eps,
        )

    height, width = valid_mask.shape
    supervision_mask = _get_supervision_mask(viewpoint_camera, gt_rgb)
    if supervision_mask is None:
        supervision_mask = valid_mask.unsqueeze(0).float()
    supervision_mask_bool = supervision_mask > 0.5

    valid_flat = valid_mask.reshape(-1)
    supervision_flat = supervision_mask_bool.reshape(-1)
    supervised_valid_flat = valid_flat & supervision_flat
    valid_idx = torch.nonzero(supervised_valid_flat, as_tuple=False).squeeze(1)

    if max_shading_points > 0 and valid_idx.numel() > max_shading_points:
        perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_shading_points]
        shading_idx = valid_idx[perm]
    else:
        shading_idx = valid_idx

    flat_normals = F.normalize(render_pkg["normal"].permute(1, 2, 0).reshape(-1, 3), dim=-1, eps=eps)
    flat_albedo = render_pkg["albedo"].permute(1, 2, 0).reshape(-1, 3)
    flat_roughness = render_pkg["roughness"].permute(1, 2, 0).reshape(-1, 1)
    flat_metallic = render_pkg["metallic"].permute(1, 2, 0).reshape(-1, 1)

    pred_image = gt_rgb.detach().clone()
    query_selection_map = gt_rgb.new_zeros((1, height, width))
    occlusion_map = gt_rgb.new_zeros((1, height, width))
    direct_map = gt_rgb.new_zeros((3, height, width))
    indirect_map = gt_rgb.new_zeros((3, height, width))
    pbr_diffuse_map = gt_rgb.new_zeros((3, height, width))
    pbr_specular_map = gt_rgb.new_zeros((3, height, width))
    sop_supervision_selection_map = gt_rgb.new_zeros((1, height, width))

    if shading_idx.numel() > 0:
        # Gather per-pixel G-buffer values for the subset of shading points we supervise this iteration.
        if view_neighbor_cache is not None:
            shading_neighbor_cache = select_view_sop_neighbor_cache(
                view_neighbor_cache,
                shading_idx,
                device=gt_rgb.device,
            )
            pts = shading_neighbor_cache["points"]
        else:
            shading_neighbor_cache = None
            pts = points.reshape(-1, 3)[shading_idx]
        nrm = flat_normals[shading_idx]
        albedo = flat_albedo[shading_idx]
        roughness = flat_roughness[shading_idx]
        metallic = flat_metallic[shading_idx]
        viewdirs = compute_view_directions(pts, viewpoint_camera.camera_center)

        lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(
            normals=nrm,
            num_samples=num_shading_samples,
            randomized=randomized_samples,
        )
        direct_radiance = envmap(lightdirs)

        query_indirect, query_occlusion = query_sops_directional(
            x_world=pts,
            query_dirs=lightdirs,
            probe_xyz=probe_xyz,
            probe_normal=probe_normal,
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            radius=float(sop_query_radius) if sop_query_radius and sop_query_radius > 0.0 else None,
            topk=sop_query_topk,
            eps=eps,
            chunk_size=sop_query_chunk_size,
            neighbor_cache=shading_neighbor_cache,
        )

        # Eq. (8): Li = (1 - O) * Ldir + Lin
        incident_radiance = (1.0 - query_occlusion) * direct_radiance + query_indirect

        pbr_rgb, pbr_aux = integrate_incident_radiance(
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            normals=nrm,
            viewdirs=viewdirs,
            lightdirs=lightdirs,
            incident_radiance=incident_radiance,
            sample_solid_angle=sample_solid_angle,
        )

        pred_flat = pred_image.permute(1, 2, 0).reshape(-1, 3)
        pred_flat[shading_idx] = pbr_rgb
        pred_image = pred_flat.reshape(height, width, 3).permute(2, 0, 1)

        query_selection_map = _scatter_to_image(
            torch.ones((shading_idx.shape[0], 1), device=gt_rgb.device, dtype=gt_rgb.dtype),
            shading_idx,
            height,
            width,
            1,
        )
        occlusion_map = _scatter_to_image(
            query_occlusion.mean(dim=1),
            shading_idx,
            height,
            width,
            1,
        )
        direct_map = _scatter_to_image(
            direct_radiance.mean(dim=1),
            shading_idx,
            height,
            width,
            3,
        )
        indirect_map = _scatter_to_image(
            query_indirect.mean(dim=1),
            shading_idx,
            height,
            width,
            3,
        )
        pbr_diffuse_map = _scatter_to_image(
            pbr_aux["diffuse"],
            shading_idx,
            height,
            width,
            3,
        )
        pbr_specular_map = _scatter_to_image(
            pbr_aux["specular"],
            shading_idx,
            height,
            width,
            3,
        )

        supervision_rgb_mask = supervision_mask.expand_as(gt_rgb)
        denom = supervision_rgb_mask.sum().clamp_min(1.0)
        pbr_l1 = (torch.abs(pred_image - gt_rgb) * supervision_rgb_mask).sum() / denom
        pbr_ssim = 1.0 - ssim(pred_image * supervision_mask, gt_rgb * supervision_mask)
        loss_pbr = pbr_l1 + 0.2 * pbr_ssim
    else:
        pts = gt_rgb.new_zeros((0, 3))
        lightdirs = gt_rgb.new_zeros((0, num_shading_samples, 3))
        query_indirect = gt_rgb.new_zeros((0, num_shading_samples, 3))
        query_occlusion = gt_rgb.new_zeros((0, num_shading_samples, 1))
        pbr_rgb = gt_rgb.new_zeros((0, 3))
        pbr_l1 = gt_rgb.new_tensor(0.0)
        pbr_ssim = gt_rgb.new_tensor(0.0)
        loss_pbr = gt_rgb.new_tensor(0.0)

    if (
        use_sop_supervision
        and tracer is not None
        and lambda_sops > 0.0
        and shading_idx.numel() > 0
    ):
        if max_sop_supervision_points > 0 and shading_idx.numel() > max_sop_supervision_points:
            perm = torch.randperm(shading_idx.numel(), device=shading_idx.device)[:max_sop_supervision_points]
            supervision_subset = perm
        else:
            supervision_subset = torch.arange(shading_idx.numel(), device=shading_idx.device)

        sup_idx = shading_idx[supervision_subset]
        sup_pts = pts[supervision_subset]
        sup_dirs = lightdirs[supervision_subset]
        sup_query_indirect = query_indirect[supervision_subset]
        sup_query_occlusion = query_occlusion[supervision_subset]
        ray_origins = sup_pts[:, None, :] + sup_dirs * trace_bias

        trace_outputs = tracer.trace(
            ray_origins=ray_origins,
            ray_directions=sup_dirs,
            envmap=envmap,
            secondary_num_samples=secondary_num_samples,
            randomized_secondary=randomized_samples,
            camera_center=viewpoint_camera.camera_center,
        )
        trace_indirect = trace_outputs["incident_radiance"].detach()
        trace_occlusion = trace_outputs["occlusion"].detach()
        loss_sops = F.l1_loss(sup_query_indirect, trace_indirect) + F.l1_loss(sup_query_occlusion, trace_occlusion)
        sop_supervision_selection_map = _scatter_to_image(
            torch.ones((sup_idx.shape[0], 1), device=gt_rgb.device, dtype=gt_rgb.dtype),
            sup_idx,
            height,
            width,
            1,
        )
    else:
        trace_outputs = None
        trace_indirect = gt_rgb.new_zeros((0, num_shading_samples, 3))
        trace_occlusion = gt_rgb.new_zeros((0, num_shading_samples, 1))
        loss_sops = gt_rgb.new_tensor(0.0)
        sup_idx = shading_idx.new_zeros((0,), dtype=torch.long)

    valid_ch = valid_mask.unsqueeze(0)
    roughness_reg = _masked_mean(torch.abs(render_pkg["roughness"] - 1.0), valid_ch)
    metallic_reg = _masked_mean(torch.abs(render_pkg["metallic"]), valid_ch)
    loss_lam = roughness_reg + metallic_reg

    loss_d2n, depth_normal, d2n_valid_mask = compute_d2n_loss(
        normal=render_pkg["normal"],
        depth_unbiased=render_pkg["depth_unbiased"],
        viewpoint_camera=viewpoint_camera,
        weight=render_pkg["weight"],
        weight_threshold=weight_threshold,
        eps=eps,
    )

    if use_mask_loss and getattr(viewpoint_camera, "gt_alpha_mask", None) is not None:
        loss_mask = compute_mask_loss(render_pkg["weight"], viewpoint_camera.gt_alpha_mask, eps=eps)
    else:
        loss_mask = gt_rgb.new_tensor(0.0)

    total = loss_pbr + lambda_lam * loss_lam + lambda_sops * loss_sops + lambda_d2n * loss_d2n + lambda_mask * loss_mask

    stats = {
        "loss_total": total,
        "loss_pbr": loss_pbr,
        "loss_lam": loss_lam,
        "loss_sops": loss_sops,
        "loss_d2n": loss_d2n,
        "loss_mask": loss_mask,
        "pbr_l1": pbr_l1,
        "pbr_ssim": pbr_ssim,
        "shading_points": gt_rgb.new_tensor(float(shading_idx.numel())),
        "shading_valid_ratio": gt_rgb.new_tensor(float(shading_idx.numel()) / float(max(valid_idx.numel(), 1))),
        "sop_supervision_points": gt_rgb.new_tensor(float(sup_idx.numel())),
        "supervision_ratio": supervision_mask.mean(),
        "d2n_valid_ratio": d2n_valid_mask.float().mean(),
    }

    aux = {
        "pbr_render": pred_image,
        "depth_normal": depth_normal,
        "d2n_valid_mask": d2n_valid_mask,
        "query_selection": query_selection_map,
        "query_occlusion": occlusion_map,
        "query_direct": direct_map,
        "query_indirect": indirect_map,
        "pbr_diffuse": pbr_diffuse_map,
        "pbr_specular": pbr_specular_map,
        "query_outputs": {
            "indirect": query_indirect,
            "occlusion": query_occlusion,
        },
        "query_indices": shading_idx,
        "query_rgb_values": pbr_rgb,
        "sop_supervision_selection": sop_supervision_selection_map,
        "trace_outputs": trace_outputs,
        "trace_indirect": trace_indirect,
        "trace_occlusion": trace_occlusion,
        "supervision_mask": supervision_mask,
    }
    return total, stats, aux
