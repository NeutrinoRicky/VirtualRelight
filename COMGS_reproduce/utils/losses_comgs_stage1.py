import torch
import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim


def compute_rgb_loss(pred_rgb, gt_rgb):
    l1 = l1_loss(pred_rgb, gt_rgb)
    ssim_term = 1.0 - ssim(pred_rgb, gt_rgb)
    loss = l1 + 0.2 * ssim_term
    return loss, {"rgb_l1": l1, "rgb_ssim": ssim_term}


def _depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height

    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1],
    ], dtype=torch.float, device=depthmap.device).T

    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=depthmap.device).float(),
        torch.arange(H, device=depthmap.device).float(),
        indexing='xy',
    )
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

    rays_d = pixels @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth, eps=1e-6):
    """
    Convert a depth map [1, H, W] to normal map [3, H, W] in world coordinates.
    """
    points = _depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    normals = torch.zeros_like(points)

    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    n = torch.cross(dx, dy, dim=-1)
    n = F.normalize(n, dim=-1, eps=eps)

    normals[1:-1, 1:-1, :] = n
    normals = normals.permute(2, 0, 1)
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    return normals


def compute_d2n_loss(normal, depth_unbiased, viewpoint_camera, weight=None, weight_threshold=1e-4, eps=1e-6):
    n_pred = F.normalize(normal, dim=0, eps=eps)
    n_depth = depth_to_normal(viewpoint_camera, depth_unbiased, eps=eps)
    n_depth = F.normalize(n_depth, dim=0, eps=eps)

    valid = torch.isfinite(depth_unbiased)
    if weight is not None:
        valid = valid & (weight > weight_threshold)
    valid = valid & torch.isfinite(n_pred.sum(dim=0, keepdim=True)) & torch.isfinite(n_depth.sum(dim=0, keepdim=True))

    dot = (n_pred * n_depth).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    d2n_map = 1.0 - dot

    if int(valid.sum().item()) == 0:
        loss = d2n_map.new_tensor(0.0)
    else:
        loss = d2n_map[valid].mean()

    return loss, n_depth, valid


def compute_mask_loss(weight, gt_mask, eps=1e-6):
    if gt_mask is None:
        return weight.new_tensor(0.0)

    k = gt_mask
    if k.shape[0] != 1:
        k = k[:1]
    if k.shape[-2:] != weight.shape[-2:]:
        k = F.interpolate(k.unsqueeze(0), size=weight.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

    k = torch.clamp(k, 0.0, 1.0)
    w = torch.clamp(weight, min=eps, max=1.0 - eps)

    loss = -k * torch.log(w) - (1.0 - k) * torch.log(1.0 - w)
    return loss.mean()


def compute_stage1_loss(
    render_pkg,
    gt_rgb,
    viewpoint_camera,
    lambda_d2n=0.05,
    lambda_mask=0.05,
    use_mask_loss=False,
    weight_threshold=1e-4,
    eps=1e-6,
):
    rgb_loss, rgb_terms = compute_rgb_loss(render_pkg["render"], gt_rgb)

    d2n_loss, depth_normal, valid_mask = compute_d2n_loss(
        normal=render_pkg["normal"],
        depth_unbiased=render_pkg["depth_unbiased"],
        viewpoint_camera=viewpoint_camera,
        weight=render_pkg["weight"],
        weight_threshold=weight_threshold,
        eps=eps,
    )

    if use_mask_loss and getattr(viewpoint_camera, "gt_alpha_mask", None) is not None:
        mask_loss = compute_mask_loss(render_pkg["weight"], viewpoint_camera.gt_alpha_mask, eps=eps)
    else:
        mask_loss = gt_rgb.new_tensor(0.0)

    total = rgb_loss + lambda_d2n * d2n_loss + lambda_mask * mask_loss

    stats = {
        "loss_total": total,
        "loss_rgb": rgb_loss,
        "loss_d2n": d2n_loss,
        "loss_mask": mask_loss,
        "rgb_l1": rgb_terms["rgb_l1"],
        "rgb_ssim": rgb_terms["rgb_ssim"],
        "d2n_valid_ratio": valid_mask.float().mean(),
    }

    aux = {
        "depth_normal": depth_normal,
        "d2n_valid_mask": valid_mask,
    }

    return total, stats, aux
