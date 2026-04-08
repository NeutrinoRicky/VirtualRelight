from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from utils.deferred_pbr_comgs import (
    compute_view_directions,
    integrate_incident_radiance,
    recover_shading_points,
    rgb_to_srgb,
    sample_incident_rays_irgs,
)
from utils.loss_utils import ssim


def first_order_edge_aware_loss(data: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    grad_data_x = torch.abs(data[:, 1:-1, :-2] + data[:, 1:-1, 2:] - 2 * data[:, 1:-1, 1:-1])
    grad_data_y = torch.abs(data[:, :-2, 1:-1] + data[:, 2:, 1:-1] - 2 * data[:, 1:-1, 1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_data_x *= torch.exp(-grad_img_x)
    grad_data_y *= torch.exp(-grad_img_y)
    return grad_data_x.mean() + grad_data_y.mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    h_tv = torch.square(x[..., 1:, :] - x[..., :-1, :]).mean()
    w_tv = torch.square(x[..., :, 1:] - x[..., :, :-1]).mean()
    return h_tv + w_tv


def _compute_irgs_sh_loss(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, lambda_dssim: float):
    l1 = torch.abs(pred_rgb - gt_rgb).mean()
    ssim_term = 1.0 - ssim(pred_rgb, gt_rgb)
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * ssim_term
    return loss, {"rgb_l1": l1, "rgb_ssim": ssim_term}


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


def _compute_mask_entropy_loss(weight: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    o = weight.clamp(eps, 1.0 - eps)
    return -(gt_mask * torch.log(o) + (1.0 - gt_mask) * torch.log(1.0 - o)).mean()


def _get_env_only_image(viewpoint_camera, envmap, ref_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    viewdirs = getattr(viewpoint_camera, "rays_d_hw", None)
    if viewdirs is None:
        viewdirs = getattr(viewpoint_camera, "rays_d_hw_unnormalized", None)
        if viewdirs is None:
            raise AttributeError("viewpoint_camera is missing both rays_d_hw and rays_d_hw_unnormalized")
        viewdirs = F.normalize(viewdirs, dim=-1, eps=eps)
    viewdirs = viewdirs.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    return rgb_to_srgb(envmap(viewdirs).permute(2, 0, 1))


def _scatter_to_image(values: torch.Tensor, flat_indices: torch.Tensor, height: int, width: int, channels: int) -> torch.Tensor:
    canvas = values.new_zeros((height * width, channels))
    if flat_indices.numel() > 0:
        canvas[flat_indices] = values
    return canvas.reshape(height, width, channels).permute(2, 0, 1)


def compute_stage2_trace_loss(
    render_pkg: Dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    viewpoint_camera,
    background: torch.Tensor,
    envmap,
    tracer,
    iteration: int = 0,
    normal_loss_start: int = 1000,
    lambda_raster: float = 1.0,
    lambda_dssim: float = 0.2,
    lambda_lam: float = 0.0,
    lambda_d2n: float = 0.05,
    lambda_mask: float = 0.05,
    lambda_base_color_smooth: float = 0.0,
    lambda_roughness_smooth: float = 0.0,
    lambda_metallic_smooth: float = 0.0,
    lambda_normal_smooth: float = 0.0,
    lambda_light: float = 0.0,
    lambda_light_smooth: float = 0.0,
    use_mask_loss: bool = False,
    num_shading_samples: int = 128,
    secondary_num_samples: int = 16,
    max_trace_points: int = 0,
    trace_bias: float = 1e-3,
    randomized_samples: bool = True,
    collect_debug_maps: bool = False,
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
        supervision_mask = torch.ones_like(gt_rgb[:1])
    gt_rgb_supervised = gt_rgb * supervision_mask

    valid_flat = valid_mask.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)

    loss_raster, raster_terms = _compute_irgs_sh_loss(
        render_pkg["render"] * supervision_mask,
        gt_rgb_supervised,
        lambda_dssim=lambda_dssim,
    )

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
    flat_gt = gt_rgb_supervised.permute(1, 2, 0).reshape(-1, 3)
    flat_alpha = render_pkg["weight"].permute(1, 2, 0).reshape(-1, 1).clamp(0.0, 1.0)
    background_rgb = background.to(device=gt_rgb.device, dtype=gt_rgb.dtype).view(1, 3)

    pred_image = render_pkg["render"].clone()
    occlusion_map = None
    direct_map = None
    indirect_map = None
    trace_selection_map = None
    pbr_diffuse_map = None
    pbr_specular_map = None

    if trace_idx.numel() > 0:
        pts = flat_points[trace_idx]
        nrm = flat_normals[trace_idx]
        albedo = flat_albedo[trace_idx]
        roughness = flat_roughness[trace_idx]
        metallic = flat_metallic[trace_idx]
        viewdirs = compute_view_directions(pts, viewpoint_camera.camera_center)

        lightdirs, incident_areas = sample_incident_rays_irgs(
            normals=nrm,
            training=randomized_samples,
            sample_num=num_shading_samples,
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
        incident_radiance = (1.0 - trace_outputs["occlusion"]) * direct_radiance + trace_outputs["incident_radiance"]

        sample_solid_angle = incident_areas.new_tensor((2.0 * 3.141592653589793) / float(max(incident_areas.shape[1], 1)))
        pbr_linear, pbr_aux = integrate_incident_radiance(
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            normals=nrm,
            viewdirs=viewdirs,
            lightdirs=lightdirs,
            incident_radiance=incident_radiance,
            sample_solid_angle=sample_solid_angle,
        )
        pbr_rgb = rgb_to_srgb(pbr_linear)
        ray_alpha = flat_alpha[trace_idx]
        ray_rgb = pbr_rgb * ray_alpha + background_rgb * (1.0 - ray_alpha)

        pred_flat = pred_image.permute(1, 2, 0).reshape(-1, 3)
        pred_flat[trace_idx] = ray_rgb
        pred_image = pred_flat.reshape(height, width, 3).permute(2, 0, 1)

        if collect_debug_maps:
            occlusion_map = _scatter_to_image(trace_outputs["occlusion"].mean(dim=1), trace_idx, height, width, 1)
            direct_map = _scatter_to_image(rgb_to_srgb(direct_radiance.mean(dim=1)), trace_idx, height, width, 3)
            indirect_map = _scatter_to_image(
                rgb_to_srgb(trace_outputs["incident_radiance"].mean(dim=1)),
                trace_idx,
                height,
                width,
                3,
            )
            pbr_diffuse_map = _scatter_to_image(rgb_to_srgb(pbr_aux["diffuse"]), trace_idx, height, width, 3)
            pbr_specular_map = _scatter_to_image(rgb_to_srgb(pbr_aux["specular"]), trace_idx, height, width, 3)
            trace_selection_map = _scatter_to_image(
                torch.ones((trace_idx.shape[0], 1), device=gt_rgb.device, dtype=gt_rgb.dtype),
                trace_idx,
                height,
                width,
                1,
            )

        pbr_l1 = torch.abs(ray_rgb - flat_gt[trace_idx]).mean()
        pbr_ssim = gt_rgb.new_tensor(0.0)
        loss_pbr = pbr_l1
        if lambda_light > 0.0:
            light_direct = direct_radiance.mean(dim=1)
            mean_light = light_direct.mean(dim=-1, keepdim=True).expand_as(light_direct)
            loss_light = F.l1_loss(light_direct, mean_light)
        else:
            loss_light = gt_rgb.new_tensor(0.0)
    else:
        trace_outputs = None
        ray_rgb = gt_rgb.new_zeros((0, 3))
        pbr_l1 = gt_rgb.new_tensor(0.0)
        pbr_ssim = gt_rgb.new_tensor(0.0)
        loss_pbr = gt_rgb.new_tensor(0.0)
        loss_light = gt_rgb.new_tensor(0.0)

    valid_ch = valid_mask.unsqueeze(0)
    roughness_reg = _masked_mean(torch.abs(render_pkg["roughness"] - 1.0), valid_ch)
    metallic_reg = _masked_mean(torch.abs(render_pkg["metallic"]), valid_ch)
    loss_lam = roughness_reg + metallic_reg

    d2n_valid_mask = render_pkg["weight"] > weight_threshold
    depth_normal = render_pkg["surf_normal"]
    if lambda_d2n > 0.0 and iteration > normal_loss_start:
        rendered_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        loss_d2n = (1.0 - (rendered_normal * surf_normal).sum(dim=0, keepdim=True)).mean()
    else:
        loss_d2n = gt_rgb.new_tensor(0.0)

    if use_mask_loss and getattr(viewpoint_camera, "gt_alpha_mask", None) is not None:
        loss_mask = _compute_mask_entropy_loss(render_pkg["rend_alpha"], supervision_mask, eps=eps)
    else:
        loss_mask = gt_rgb.new_tensor(0.0)

    smooth_mask = supervision_mask
    weighted_albedo = render_pkg["albedo"] * render_pkg["weight"]
    weighted_roughness = render_pkg["roughness"] * render_pkg["weight"]
    weighted_metallic = render_pkg["metallic"] * render_pkg["weight"]
    if lambda_base_color_smooth > 0.0:
        loss_base_color_smooth = first_order_edge_aware_loss(weighted_albedo * smooth_mask, gt_rgb)
    else:
        loss_base_color_smooth = gt_rgb.new_tensor(0.0)

    if lambda_roughness_smooth > 0.0:
        loss_roughness_smooth = first_order_edge_aware_loss(weighted_roughness * smooth_mask, gt_rgb)
    else:
        loss_roughness_smooth = gt_rgb.new_tensor(0.0)

    if lambda_metallic_smooth > 0.0:
        loss_metallic_smooth = first_order_edge_aware_loss(weighted_metallic * smooth_mask, gt_rgb)
    else:
        loss_metallic_smooth = gt_rgb.new_tensor(0.0)

    if lambda_normal_smooth > 0.0:
        loss_normal_smooth = first_order_edge_aware_loss(render_pkg["rend_normal"] * smooth_mask, gt_rgb)
    else:
        loss_normal_smooth = gt_rgb.new_tensor(0.0)

    if lambda_light_smooth > 0.0:
        env_tensor = _get_env_only_image(viewpoint_camera, envmap, gt_rgb, eps=eps)
        loss_light_smooth = tv_loss(env_tensor)
    else:
        loss_light_smooth = gt_rgb.new_tensor(0.0)

    total = (
        lambda_raster * loss_raster
        + loss_pbr
        + lambda_lam * loss_lam
        + lambda_d2n * loss_d2n
        + lambda_mask * loss_mask
        + lambda_base_color_smooth * loss_base_color_smooth
        + lambda_roughness_smooth * loss_roughness_smooth
        + lambda_metallic_smooth * loss_metallic_smooth
        + lambda_normal_smooth * loss_normal_smooth
        + lambda_light * loss_light
        + lambda_light_smooth * loss_light_smooth
    )

    stats = {
        "loss_total": total,
        "loss_raster": loss_raster,
        "loss_pbr": loss_pbr,
        "loss_lam": loss_lam,
        "loss_d2n": loss_d2n,
        "loss_mask": loss_mask,
        "loss_base_color_smooth": loss_base_color_smooth,
        "loss_roughness_smooth": loss_roughness_smooth,
        "loss_metallic_smooth": loss_metallic_smooth,
        "loss_normal_smooth": loss_normal_smooth,
        "loss_light": loss_light,
        "loss_light_smooth": loss_light_smooth,
        "raster_l1": raster_terms["rgb_l1"],
        "raster_ssim": raster_terms["rgb_ssim"],
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
        "trace_rgb_values": ray_rgb,
        "supervision_mask": supervision_mask,
    }
    return total, stats, aux
