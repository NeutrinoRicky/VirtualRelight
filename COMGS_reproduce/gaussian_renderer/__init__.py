#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh


def _make_screenspace_points(pc: GaussianModel):
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def _resolve_covariance(viewpoint_camera, pc: GaussianModel, pipe, scaling_modifier):
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor(
            [
                [width / 2, 0, 0, (width - 1) / 2],
                [0, height / 2, 0, (height - 1) / 2],
                [0, 0, far - near, near],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        ).T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    return scales, rotations, cov3D_precomp


def _resolve_color_inputs(viewpoint_camera, pc: GaussianModel, pipe, override_color):
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return shs, colors_precomp


def _compute_regularization_outputs(allmap, viewpoint_camera, pipe):
    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)

    render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)

    render_depth_expected = allmap[0:1] / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal = surf_normal * render_alpha.detach()

    return {
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    }


def _rasterize_scene(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    features=None,
):
    screenspace_points = _make_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales, rotations, cov3D_precomp = _resolve_covariance(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        scaling_modifier=scaling_modifier,
    )
    shs, colors_precomp = _resolve_color_inputs(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_color=override_color,
    )

    _, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "rendered_image": rendered_image,
        "rendered_features": rendered_features,
        "allmap": allmap,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    raster_out = _rasterize_scene(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
    )

    rets = {
        "render": raster_out["rendered_image"],
        "viewspace_points": raster_out["viewspace_points"],
        "visibility_filter": raster_out["visibility_filter"],
        "radii": raster_out["radii"],
    }
    rets.update(_compute_regularization_outputs(raster_out["allmap"], viewpoint_camera, pipe))
    return rets


def render_multitarget(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, eps=1e-6):
    """
    Stage1-oriented multi-target renderer.

    Returns at least:
    - render / RGB
    - weight
    - depth
    - depth_unbiased
    - normal
    - albedo
    - roughness
    - metallic
    """

    material_features = torch.cat([pc.get_albedo, pc.get_roughness, pc.get_metallic], dim=1)
    raster_out = _rasterize_scene(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        features=material_features,
    )

    base = {
        "render": raster_out["rendered_image"],
        "viewspace_points": raster_out["viewspace_points"],
        "visibility_filter": raster_out["visibility_filter"],
        "radii": raster_out["radii"],
    }
    base.update(_compute_regularization_outputs(raster_out["allmap"], viewpoint_camera, pipe))

    weight = torch.nan_to_num(base["rend_alpha"], nan=0.0, posinf=0.0, neginf=0.0)
    depth_unbiased = torch.nan_to_num(base["surf_depth"], nan=0.0, posinf=0.0, neginf=0.0)
    depth = depth_unbiased * weight
    depth_unbiased = depth / torch.clamp(weight, min=eps)
    depth_unbiased = torch.nan_to_num(depth_unbiased, nan=0.0, posinf=0.0, neginf=0.0)

    normal_map = base["rend_normal"] / torch.clamp(weight, min=eps)
    normal = torch.nn.functional.normalize(normal_map, dim=0, eps=eps)
    valid = (weight > eps).float()
    normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0) * valid

    albedo, roughness, metallic = raster_out["rendered_features"].split((3, 1, 1), dim=0)

    out = {
        "render": base["render"],
        "weight": weight,
        "depth": depth,
        "depth_unbiased": depth_unbiased,
        "normal": normal,
        "albedo": torch.clamp(albedo, 0.0, 1.0),
        "roughness": torch.clamp(roughness, 0.0, 1.0),
        "metallic": torch.clamp(metallic, 0.0, 1.0),
        "viewspace_points": base["viewspace_points"],
        "visibility_filter": base["visibility_filter"],
        "radii": base["radii"],
        "rend_alpha": base["rend_alpha"],
        "rend_normal": base["rend_normal"],
        "rend_dist": base["rend_dist"],
        "surf_depth": base["surf_depth"],
        "surf_normal": base["surf_normal"],
    }
    return out
