from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from gaussian_renderer import rendering_equation_sop_chunk
from utils.graphics_utils import rgb_to_srgb
from utils.loss_utils import first_order_edge_aware_loss, ssim, tv_loss


def _as_single_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0
    if int(mask.sum().item()) == 0:
        return value.new_tensor(0.0)
    return value[mask].mean()


def _safe_unpremultiply(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    weight = _as_single_channel(weight).to(device=value.device, dtype=value.dtype).clamp_min(eps)
    return torch.nan_to_num(value / weight, nan=0.0, posinf=0.0, neginf=0.0)


def depths_to_points(view, depthmap: torch.Tensor) -> torch.Tensor:
    depthmap = _as_single_channel(depthmap)
    rays_d_hw = getattr(view, "rays_d_hw_unnormalized", None)
    camera_center = getattr(view, "camera_center", None)
    if rays_d_hw is None or camera_center is None:
        raise RuntimeError("Stage2 SOP expects cameras to provide rays_d_hw_unnormalized and camera_center.")

    rays_d_hw = rays_d_hw.to(device=depthmap.device, dtype=depthmap.dtype)
    camera_center = camera_center.to(device=depthmap.device, dtype=depthmap.dtype)
    points = depthmap.permute(1, 2, 0) * rays_d_hw + camera_center
    return torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def recover_shading_points(
    view,
    depth_unbiased: torch.Tensor,
    weight: torch.Tensor = None,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_unbiased = _as_single_channel(depth_unbiased)
    points = depths_to_points(view, depth_unbiased)
    valid = torch.isfinite(depth_unbiased) & (depth_unbiased > eps)
    if weight is not None:
        valid = valid & (_as_single_channel(weight) > weight_threshold)
    return points, valid.squeeze(0)


def compute_view_directions(points: torch.Tensor, camera_center: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(camera_center.view(1, 3) - points, dim=-1, eps=eps)


def _depth_to_normal(view, depthmap: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    points = depths_to_points(view, depthmap)
    normals = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    n = torch.cross(dx, dy, dim=-1)
    n = F.normalize(n, dim=-1, eps=eps)
    normals[1:-1, 1:-1] = n
    normals = normals.permute(2, 0, 1)
    return torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)


def compute_d2n_loss(
    normal: torch.Tensor,
    depth_unbiased: torch.Tensor,
    viewpoint_camera,
    weight: torch.Tensor = None,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pred = F.normalize(normal, dim=0, eps=eps)
    n_depth = _depth_to_normal(viewpoint_camera, depth_unbiased, eps=eps)
    n_depth = F.normalize(n_depth, dim=0, eps=eps)

    depth_unbiased = _as_single_channel(depth_unbiased)
    valid = torch.isfinite(depth_unbiased)
    if weight is not None:
        valid = valid & (_as_single_channel(weight) > weight_threshold)
    valid = valid & torch.isfinite(n_pred.sum(dim=0, keepdim=True)) & torch.isfinite(n_depth.sum(dim=0, keepdim=True))

    dot = (n_pred * n_depth).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    d2n_map = 1.0 - dot
    if int(valid.sum().item()) == 0:
        return d2n_map.new_tensor(0.0), n_depth, valid
    return d2n_map[valid].mean(), n_depth, valid


def compute_mask_loss(weight: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if gt_mask is None:
        return weight.new_tensor(0.0)

    weight = _as_single_channel(weight)
    k = gt_mask
    if k.dim() == 2:
        k = k.unsqueeze(0)
    if k.shape[0] != 1:
        k = k[:1]
    if k.shape[-2:] != weight.shape[-2:]:
        k = F.interpolate(k.unsqueeze(0), size=weight.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

    k = torch.clamp(k.to(device=weight.device, dtype=weight.dtype), 0.0, 1.0)
    w = torch.clamp(weight, min=eps, max=1.0 - eps)
    loss = -k * torch.log(w) - (1.0 - k) * torch.log(1.0 - w)
    return loss.mean()


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
        mask = F.interpolate(mask.unsqueeze(0), size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    return torch.clamp(mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype), 0.0, 1.0)


def _get_regularization_mask(viewpoint_camera, ref_tensor: torch.Tensor):
    mask = getattr(viewpoint_camera, "mask", None)
    if mask is None:
        return None

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] != 1:
        mask = mask[:1]

    if mask.shape[-2:] != ref_tensor.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(0), size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    return torch.clamp(mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype), 0.0, 1.0)


def _scatter_to_image(values: torch.Tensor, flat_indices: torch.Tensor, height: int, width: int, channels: int) -> torch.Tensor:
    canvas = values.new_zeros((height * width, channels))
    if flat_indices.numel() > 0:
        canvas[flat_indices] = values
    return canvas.reshape(height, width, channels).permute(2, 0, 1)


def _masked_rgb_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_rgb = mask.expand_as(pred)
    denom = mask_rgb.sum().clamp_min(1.0)
    return (torch.abs(pred - gt) * mask_rgb).sum() / denom


def _cuda_mem_debug_enabled(debug_cfg) -> bool:
    return bool(debug_cfg) and bool(debug_cfg.get("enabled", False)) and torch.cuda.is_available()


def _log_cuda_mem(debug_cfg, label: str) -> None:
    if not _cuda_mem_debug_enabled(debug_cfg):
        return
    torch.cuda.synchronize()
    prefix = debug_cfg.get("prefix", "")
    if prefix:
        label = f"{prefix} | {label}"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[CUDA-MEM] {label}: alloc={allocated:.1f} MiB peak={peak:.1f} MiB reserved={reserved:.1f} MiB")


def _opt_weight(opt, name: str) -> float:
    if opt is None:
        return 0.0
    return float(getattr(opt, name, 0.0))


def _masked_edge_aware_loss(data: torch.Tensor, gt_rgb: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        data = data * mask
    return first_order_edge_aware_loss(data, gt_rgb)


def _compute_env_smooth_loss(gaussians, viewpoint_camera, gt_rgb: torch.Tensor) -> torch.Tensor:
    envmap = getattr(gaussians, "get_envmap", None)
    rays_d = getattr(viewpoint_camera, "rays_d_hw", None)
    if envmap is None or rays_d is None:
        return gt_rgb.new_tensor(0.0)

    env_only = rgb_to_srgb(envmap(rays_d, mode="pure_env").permute(2, 0, 1))
    return tv_loss(env_only)


def compute_stage2_sop_loss(
    render_pkg: Dict[str, torch.Tensor],
    gt_rgb: torch.Tensor,
    viewpoint_camera,
    gaussians,
    pipe,
    background: torch.Tensor,
    opt=None,
    training: bool = False,
    probe_xyz: torch.Tensor = None,
    probe_normal: torch.Tensor = None,
    probe_lin_tex: torch.Tensor = None,
    probe_occ_tex: torch.Tensor = None,
    lambda_lam: float = 0.0,
    lambda_sops: float = 0.0,
    lambda_d2n: float = 0.05,
    lambda_mask: float = 0.05,
    use_mask_loss: bool = False,
    num_shading_samples: int = 128,
    max_shading_points: int = 4096,
    sop_query_radius: float = 0.0,
    sop_query_topk: int = 8,
    sop_query_chunk_size: int = 1024,
    randomized_samples: bool = True,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
    cuda_mem_debug=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    del num_shading_samples
    del max_shading_points

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
    valid_idx = torch.nonzero(valid_flat & supervision_flat, as_tuple=False).squeeze(1)
    shading_idx = valid_idx

    flat_weight = _as_single_channel(render_pkg["weight"]).permute(1, 2, 0).reshape(-1, 1)
    albedo_unbiased = _safe_unpremultiply(render_pkg["albedo"], render_pkg["weight"], eps=eps)
    roughness_unbiased = _safe_unpremultiply(render_pkg["roughness"], render_pkg["weight"], eps=eps)
    metallic_unbiased = _safe_unpremultiply(render_pkg["metallic"], render_pkg["weight"], eps=eps)

    normal_key = "normal_shading" if "normal_shading" in render_pkg else "normal"
    flat_normals = F.normalize(render_pkg[normal_key].permute(1, 2, 0).reshape(-1, 3), dim=-1, eps=eps)
    flat_albedo = albedo_unbiased.permute(1, 2, 0).reshape(-1, 3)
    flat_roughness = _as_single_channel(roughness_unbiased).permute(1, 2, 0).reshape(-1, 1)
    flat_metallic = _as_single_channel(metallic_unbiased).permute(1, 2, 0).reshape(-1, 1)
    gt_flat = gt_rgb.permute(1, 2, 0).reshape(-1, 3)

    bg_color = background.to(device=gt_rgb.device, dtype=gt_rgb.dtype)
    pred_image = bg_color[:, None, None].expand_as(gt_rgb).clone()
    query_selection_map = gt_rgb.new_zeros((1, height, width))
    occlusion_map = gt_rgb.new_zeros((1, height, width))
    direct_map = gt_rgb.new_zeros((3, height, width))
    indirect_map = gt_rgb.new_zeros((3, height, width))
    pbr_diffuse_map = gt_rgb.new_zeros((3, height, width))
    pbr_specular_map = gt_rgb.new_zeros((3, height, width))
    light_direct_values = None
    loss_sops = gt_rgb.new_tensor(0.0)
    loss_sops_indirect = gt_rgb.new_tensor(0.0)
    loss_sops_occlusion = gt_rgb.new_tensor(0.0)

    if shading_idx.numel() > 0:
        pts = points.reshape(-1, 3)[shading_idx]
        nrm = flat_normals[shading_idx]
        albedo = flat_albedo[shading_idx]
        roughness = flat_roughness[shading_idx]
        metallic = flat_metallic[shading_idx]
        alpha = flat_weight[shading_idx].clamp(0.0, 1.0)
        viewdirs = compute_view_directions(pts, viewpoint_camera.camera_center)

        _log_cuda_mem(cuda_mem_debug, "before rendering_equation_sop")
        render_results = rendering_equation_sop_chunk(
            base_color=albedo,
            roughness=roughness,
            metallic=metallic,
            normal=nrm,
            position=pts,
            w_o=viewdirs,
            pc=gaussians,
            pipe=pipe,
            probe_xyz=probe_xyz,
            probe_normal=probe_normal,
            probe_lin_tex=probe_lin_tex,
            probe_occ_tex=probe_occ_tex,
            sample_training=training and randomized_samples,
            sop_query_radius=float(sop_query_radius) if sop_query_radius and sop_query_radius > 0.0 else None,
            sop_query_topk=sop_query_topk,
            sop_query_chunk_size=sop_query_chunk_size,
            eps=eps,
            cuda_mem_debug=cuda_mem_debug,
        )
        _log_cuda_mem(cuda_mem_debug, "after rendering_equation_sop")

        diffuse_linear = render_results["diffuse"]
        specular_linear = render_results["specular"]
        light_direct_values = render_results["light_direct"]

        pbr_linear = diffuse_linear + specular_linear
        pbr_rgb = rgb_to_srgb(pbr_linear) * alpha + bg_color.view(1, 3) * (1.0 - alpha)

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
            render_results["visibility"].new_ones(render_results["visibility"].shape[0], 1) - render_results["visibility"],
            shading_idx,
            height,
            width,
            1,
        )
        direct_map = torch.clamp(
            _scatter_to_image(
                rgb_to_srgb(render_results["light_direct"] * alpha),
                shading_idx,
                height,
                width,
                3,
            ),
            0.0,
            1.0,
        )
        indirect_map = torch.clamp(
            _scatter_to_image(
                rgb_to_srgb(render_results["light_indirect"] * alpha),
                shading_idx,
                height,
                width,
                3,
            ),
            0.0,
            1.0,
        )
        pbr_diffuse_map = torch.clamp(
            _scatter_to_image(
                rgb_to_srgb(diffuse_linear),
                shading_idx,
                height,
                width,
                3,
            ),
            0.0,
            1.0,
        )
        pbr_specular_map = torch.clamp(
            _scatter_to_image(
                rgb_to_srgb(specular_linear),
                shading_idx,
                height,
                width,
                3,
            ),
            0.0,
            1.0,
        )

        if training and opt is not None and getattr(opt, "train_ray", False):
            pbr_l1 = F.l1_loss(pbr_rgb, gt_flat[shading_idx])
            pbr_ssim = gt_rgb.new_tensor(0.0)
            loss_pbr = pbr_l1
        elif shading_idx.numel() == valid_idx.numel():
            pbr_l1 = _masked_rgb_l1(pred_image, gt_rgb, supervision_mask)
            pbr_ssim = 1.0 - ssim(pred_image * supervision_mask, gt_rgb * supervision_mask)
            loss_pbr = pbr_l1 + 0.2 * pbr_ssim
        else:
            pbr_l1 = F.l1_loss(pbr_rgb, gt_flat[shading_idx])
            pbr_ssim = gt_rgb.new_tensor(0.0)
            loss_pbr = pbr_l1
    else:
        pbr_rgb = gt_rgb.new_zeros((0, 3))
        pbr_l1 = gt_rgb.new_tensor(0.0)
        pbr_ssim = gt_rgb.new_tensor(0.0)
        loss_pbr = gt_rgb.new_tensor(0.0)

    valid_ch = valid_mask.unsqueeze(0)
    loss_lam = _masked_mean(torch.abs(_as_single_channel(roughness_unbiased) - 1.0), valid_ch)
    loss_lam = loss_lam + _masked_mean(torch.abs(_as_single_channel(metallic_unbiased)), valid_ch)

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

    lambda_base_color_smooth = _opt_weight(opt, "lambda_base_color_smooth") if training else 0.0
    lambda_roughness_smooth = _opt_weight(opt, "lambda_roughness_smooth") if training else 0.0
    lambda_normal_smooth = _opt_weight(opt, "lambda_normal_smooth") if training else 0.0
    lambda_light = _opt_weight(opt, "lambda_light") if training else 0.0
    lambda_light_smooth = _opt_weight(opt, "lambda_light_smooth") if training else 0.0

    reg_mask = _get_regularization_mask(viewpoint_camera, gt_rgb)
    render_pkg_ir = render_pkg.get("render_pkg_ir", {})

    if lambda_light > 0 and light_direct_values is not None and light_direct_values.numel() > 0:
        mean_light = light_direct_values.mean(-1, keepdim=True).expand_as(light_direct_values)
        loss_light = F.l1_loss(light_direct_values, mean_light)
    else:
        loss_light = gt_rgb.new_tensor(0.0)

    if lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg_ir.get("base_color_linear", render_pkg["albedo"])
        loss_base_color_smooth = _masked_edge_aware_loss(rendered_base_color, gt_rgb, reg_mask)
    else:
        loss_base_color_smooth = gt_rgb.new_tensor(0.0)

    if lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg_ir.get("roughness", render_pkg["roughness"])
        loss_roughness_smooth = _masked_edge_aware_loss(rendered_roughness, gt_rgb, reg_mask)
    else:
        loss_roughness_smooth = gt_rgb.new_tensor(0.0)

    if lambda_normal_smooth > 0:
        rendered_normal = render_pkg_ir.get("rend_normal", render_pkg["normal"])
        loss_normal_smooth = _masked_edge_aware_loss(rendered_normal, gt_rgb, reg_mask)
    else:
        loss_normal_smooth = gt_rgb.new_tensor(0.0)

    if lambda_light_smooth > 0:
        loss_light_smooth = _compute_env_smooth_loss(gaussians, viewpoint_camera, gt_rgb)
    else:
        loss_light_smooth = gt_rgb.new_tensor(0.0)

    total = (
        loss_pbr
        + lambda_lam * loss_lam
        + lambda_sops * loss_sops
        + lambda_d2n * loss_d2n
        + lambda_mask * loss_mask
        + lambda_light * loss_light
        + lambda_base_color_smooth * loss_base_color_smooth
        + lambda_roughness_smooth * loss_roughness_smooth
        + lambda_normal_smooth * loss_normal_smooth
        + lambda_light_smooth * loss_light_smooth
    )

    stats = {
        "loss_total": total,
        "loss_pbr": loss_pbr,
        "loss_lam": loss_lam,
        "loss_sops": loss_sops,
        "loss_d2n": loss_d2n,
        "loss_mask": loss_mask,
        "loss_light": loss_light,
        "loss_base_color_smooth": loss_base_color_smooth,
        "loss_roughness_smooth": loss_roughness_smooth,
        "loss_normal_smooth": loss_normal_smooth,
        "loss_light_smooth": loss_light_smooth,
        "loss_sops_indirect": loss_sops_indirect,
        "loss_sops_occlusion": loss_sops_occlusion,
        "pbr_l1": pbr_l1,
        "pbr_ssim": pbr_ssim,
        "shading_points": gt_rgb.new_tensor(float(shading_idx.numel())),
        "shading_valid_ratio": gt_rgb.new_tensor(float(shading_idx.numel()) / float(max(valid_idx.numel(), 1))),
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
            "indirect": indirect_map,
            "occlusion": occlusion_map,
        },
        "query_indices": shading_idx,
        "query_rgb_values": pbr_rgb,
        "supervision_mask": supervision_mask,
    }
    return total, stats, aux
