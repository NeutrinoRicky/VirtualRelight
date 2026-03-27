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



def _masked_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return target.new_tensor(0.0)
    return torch.abs(pred - target).mean()



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



def compute_stage2_trace_loss(
    render_pkg: Dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    viewpoint_camera,
    envmap,
    tracer,
    lambda_lam: float = 0.001,
    lambda_d2n: float = 0.05,
    lambda_mask: float = 0.05,
    use_mask_loss: bool = False,
    num_shading_samples: int = 128,
    secondary_num_samples: int = 16,
    max_trace_points: int = 0,
    trace_bias: float = 1e-3,
    randomized_samples: bool = True,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

    if max_trace_points > 0 and valid_idx.numel() > max_trace_points:
        perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_trace_points]
        trace_idx = valid_idx[perm]
    else:
        trace_idx = valid_idx

    flat_points = points.reshape(-1, 3)
    flat_normals = F.normalize(render_pkg["normal"].permute(1, 2, 0).reshape(-1, 3), dim=-1, eps=eps)
    flat_albedo = render_pkg["albedo"].permute(1, 2, 0).reshape(-1, 3)
    flat_roughness = render_pkg["roughness"].permute(1, 2, 0).reshape(-1, 1)
    flat_metallic = render_pkg["metallic"].permute(1, 2, 0).reshape(-1, 1)
    flat_gt = gt_rgb.permute(1, 2, 0).reshape(-1, 3)

    pred_image = gt_rgb.detach().clone()
    occlusion_map = gt_rgb.new_zeros((1, height, width))
    direct_map = gt_rgb.new_zeros((3, height, width))
    indirect_map = gt_rgb.new_zeros((3, height, width))
    trace_selection_map = gt_rgb.new_zeros((1, height, width))
    pbr_diffuse_map = gt_rgb.new_zeros((3, height, width))
    pbr_specular_map = gt_rgb.new_zeros((3, height, width))

    if trace_idx.numel() > 0:
        pts = flat_points[trace_idx]
        nrm = flat_normals[trace_idx]
        albedo = flat_albedo[trace_idx]
        roughness = flat_roughness[trace_idx]
        metallic = flat_metallic[trace_idx]
        viewdirs = compute_view_directions(pts, viewpoint_camera.camera_center)

        lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(
            normals=nrm,
            num_samples=num_shading_samples,
            randomized=randomized_samples,
        )
        ray_origins = pts[:, None, :] + lightdirs * trace_bias

        trace_outputs = tracer.trace(
            ray_origins=ray_origins,
            ray_directions=lightdirs,
            envmap=envmap,
            secondary_num_samples=secondary_num_samples,
            randomized_secondary=randomized_samples,
            camera_center=viewpoint_camera.camera_center,
        )

        direct_radiance = envmap(lightdirs)
        # IRGS-style incident light composition: L_i = (1 - O) * L_dir + L_ind.
        incident_radiance = (1.0 - trace_outputs["occlusion"]) * direct_radiance + trace_outputs["incident_radiance"]

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
        pred_flat[trace_idx] = pbr_rgb
        pred_image = pred_flat.reshape(height, width, 3).permute(2, 0, 1)

        occlusion_map = _scatter_to_image(
            trace_outputs["occlusion"].mean(dim=1),
            trace_idx,
            height,
            width,
            1,
        )
        direct_map = _scatter_to_image(
            direct_radiance.mean(dim=1),
            trace_idx,
            height,
            width,
            3,
        )
        indirect_map = _scatter_to_image(
            trace_outputs["incident_radiance"].mean(dim=1),
            trace_idx,
            height,
            width,
            3,
        )
        pbr_diffuse_map = _scatter_to_image(
            pbr_aux["diffuse"],
            trace_idx,
            height,
            width,
            3,
        )
        pbr_specular_map = _scatter_to_image(
            pbr_aux["specular"],
            trace_idx,
            height,
            width,
            3,
        )
        trace_selection_map = _scatter_to_image(
            torch.ones((trace_idx.shape[0], 1), device=gt_rgb.device, dtype=gt_rgb.dtype),
            trace_idx,
            height,
            width,
            1,
        )

        supervision_rgb_mask = supervision_mask.expand_as(gt_rgb)
        denom = supervision_rgb_mask.sum().clamp_min(1.0)
        pbr_l1 = (torch.abs(pred_image - gt_rgb) * supervision_rgb_mask).sum() / denom
        pbr_ssim = 1.0 - ssim(pred_image * supervision_mask, gt_rgb * supervision_mask)
        loss_pbr = pbr_l1 + 0.2 * pbr_ssim
    else:
        trace_outputs = None
        pbr_rgb = gt_rgb.new_zeros((0, 3))
        pbr_l1 = gt_rgb.new_tensor(0.0)
        pbr_ssim = gt_rgb.new_tensor(0.0)
        loss_pbr = gt_rgb.new_tensor(0.0)

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

    total = loss_pbr + lambda_lam * loss_lam + lambda_d2n * loss_d2n + lambda_mask * loss_mask

    stats = {
        "loss_total": total,
        "loss_pbr": loss_pbr,
        "loss_lam": loss_lam,
        "loss_d2n": loss_d2n,
        "loss_mask": loss_mask,
        "pbr_l1": pbr_l1,
        "pbr_ssim": pbr_ssim,
        "trace_points": gt_rgb.new_tensor(float(trace_idx.numel())),
        "trace_valid_ratio": gt_rgb.new_tensor(float(trace_idx.numel()) / float(max(valid_idx.numel(), 1))),
        "supervision_ratio": supervision_mask.mean(),
        "d2n_valid_ratio": d2n_valid_mask.float().mean(),
    }

    aux = {
        "pbr_render": pred_image,
        "depth_normal": depth_normal,
        "d2n_valid_mask": d2n_valid_mask,
        "trace_selection": trace_selection_map,
        "trace_occlusion": occlusion_map,
        "trace_direct": direct_map,
        "trace_indirect": indirect_map,
        "pbr_diffuse": pbr_diffuse_map,
        "pbr_specular": pbr_specular_map,
        "trace_outputs": trace_outputs,
        "trace_indices": trace_idx,
        "trace_rgb_values": pbr_rgb,
        "supervision_mask": supervision_mask,
    }
    return total, stats, aux
