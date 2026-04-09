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
import torch
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal, depths_to_points
from utils.graphics_utils import rotation_between_z, fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel2
from .ref_gaussian import render_initial, render_surfel, render_volume, render_surfel2
import numpy as np
from utils.system_utils import Timing
import trimesh
import nvdiffrast.torch as dr
import kornia
from torchvision.utils import save_image


def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median / render_alpha, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    render_var = render_depth_median - render_depth_expected.square()
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'render_var': render_var,
    }



def render_ir(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opt=None, iteration=-1, training=False, relight=False, base_color_scale=None, material_only=False):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    base_color = pc.get_base_color
    roughness = pc.get_rough
    metallic = pc.get_metallic
    
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    
    shs = pc.get_features
    colors_precomp = None
    
    features = torch.cat([base_color, roughness, metallic], dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    points = surf_depth.permute(1, 2, 0) * viewpoint_camera.rays_d_hw_unnormalized + viewpoint_camera.camera_center
    
    surf_normal = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    surf_normal[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)  
    normal_map = F.normalize(normal_map, dim=-1)

    rendered_base_color, rendered_roughness, rendered_metallic = rendered_features.split([3, 1, 1], dim=0)
    if base_color_scale is not None:
        rendered_base_color = rendered_base_color * base_color_scale[:, None, None]

    if material_only:
        results = {
            "metallic": rendered_metallic * render_alpha,
            "roughness": rendered_roughness * render_alpha,
            "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
            "base_color_linear": rendered_base_color * render_alpha,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        }
        return results
    
    if training:
        if opt.train_ray:
            mask_alpha = render_alpha[0] > 0
            mask_sum = mask_alpha.sum()
            
            num_pixels = opt.trace_num_rays // (pipe.diffuse_sample_num + pipe.light_sample_num)
            if num_pixels > mask_sum:
                ray_ids = torch.arange(mask_sum, device='cuda')
            else:
                ray_ids = torch.multinomial(torch.ones(mask_sum, device=mask_sum.device), num_pixels, replacement=False)

            mask_=mask_alpha[mask_alpha]
            mask_[ray_ids]=False
            mask = torch.zeros_like(mask_alpha)
            mask[mask_alpha]=~mask_
        else:
            mask = render_alpha[0] > 0
    else:
        mask = render_alpha[0] > 0
        
    rays_d = viewpoint_camera.rays_d_hw
    w_o = -rays_d
    if training:
        render_results = rendering_equation(rendered_base_color.permute(1, 2, 0)[mask], rendered_roughness.permute(1, 2, 0)[mask], rendered_metallic.permute(1, 2, 0)[mask], normal_map[mask], points[mask], w_o[mask], pc, pipe=pipe, training=training, camera_center=viewpoint_camera.camera_center)
    else:
        render_results = rendering_equation_chunk(rendered_base_color.permute(1, 2, 0)[mask], rendered_roughness.permute(1, 2, 0)[mask], rendered_metallic.permute(1, 2, 0)[mask], normal_map[mask], points[mask], w_o[mask], pc, pipe=pipe, training=training, relight=relight, camera_center=viewpoint_camera.camera_center)
        
    diffuse = render_results['diffuse']
    specular = render_results['specular']
    light_direct = render_results['light_direct']
    
    rendered_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_diffuse[mask] = diffuse
    rendered_diffuse = rendered_diffuse.permute(2, 0, 1)
        
    rendered_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)
    rendered_specular[mask] = specular
    rendered_specular = rendered_specular.permute(2, 0, 1)
    rendered_full = rgb_to_srgb(rendered_diffuse + rendered_specular)
    final_image = rendered_full * render_alpha + bg_color[:, None, None] * (1 - render_alpha)
        
    final_image_sh = rgb_to_srgb(rendered_image) + bg_color[:, None, None] * (1 - render_alpha)
    
    direct_lights = rgb_to_srgb(pc.get_envmap(rays_d, mode='pure_env').permute(2,0,1))
    env_only = direct_lights
    
    results = {
        "render": final_image,
        "env_only": env_only,
        "render_sh": final_image_sh,
        "diffuse": rgb_to_srgb(rendered_diffuse),
        "specular": rgb_to_srgb(rendered_specular),
        "mask": mask,
        "metallic": rendered_metallic * render_alpha,
        "roughness": rendered_roughness * render_alpha,
        "base_color": rgb_to_srgb(rendered_base_color) * render_alpha,
        "base_color_linear": rendered_base_color * render_alpha,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        ## normal, accum alpha, dist, depth map
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "ray_light_direct": light_direct,
    }
    
    if opt is not None and training and opt.train_ray:
        alpha = render_alpha.permute(1,2,0)[mask]
        full = diffuse + specular
        full = rgb_to_srgb(full)
        ray_rgb = full * alpha + bg_color[None, :] * (1 - alpha)
        results.update({
            "ray_rgb": ray_rgb,
        })
        
    if not training:
        
        visibility = render_results['visibility']
        light = render_results['light']
        light_indirect = render_results['light_indirect']
        
        rendered_visibility = torch.zeros_like(rendered_image[:1]).permute(1, 2, 0)
        rendered_visibility[mask] = visibility
        rendered_visibility = rendered_visibility.permute(2, 0, 1) * render_alpha
        
        rendered_light = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light[mask] = light
        rendered_light = rendered_light.permute(2, 0, 1) * render_alpha
        
        rendered_light_indirect = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_indirect[mask] = light_indirect
        rendered_light_indirect = rendered_light_indirect.permute(2, 0, 1) * render_alpha
        
        rendered_light_direct = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_light_direct[mask] = light_direct
        rendered_light_direct = rendered_light_direct.permute(2, 0, 1) * render_alpha
        
        final_image_env = rendered_full * render_alpha + direct_lights * (1 - render_alpha)
        
        results.update({
            "render_env": final_image_env,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "visibility": rendered_visibility,
            "light": rgb_to_srgb(rendered_light),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
        })

    return results

def _broadcast_to_target(value, target):
    value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    if value.ndim == 0:
        return value
    while value.ndim < target.ndim:
        value = value.unsqueeze(-2)
    return value


def rendering_equation_chunk(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training=False, f0=0.04, relight=False, chunk_size=2**20, camera_center=None, image_sh=None, **kwargs):
    chunk_size = chunk_size // (pipe.diffuse_sample_num + pipe.light_sample_num)
    if base_color.shape[0] <= chunk_size:
        return rendering_equation(base_color, roughness, metallic, normal, position, w_o, pc, pipe, training, f0, relight=relight, camera_center=camera_center, **kwargs)
    else:
        results = []
        for i in range(0, base_color.shape[0], chunk_size):
            results.append(rendering_equation(base_color[i:i+chunk_size], roughness[i:i+chunk_size], metallic[i:i+chunk_size], normal[i:i+chunk_size], position[i:i+chunk_size], w_o[i:i+chunk_size], pc, pipe, training, f0, relight=relight, camera_center=camera_center, **kwargs))
        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}
    
def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

def rendering_equation(base_color, roughness, metallic, normals, position, viewdirs, pc, pipe, training=False, f0=0.04, relight=False, camera_center=None, **kwargs):
    B = base_color.shape[0]
    envmap = pc.get_envmap
    
    if pipe.diffuse_sample_num > 0 and pipe.light_sample_num == 0:
        incident_dirs, incident_areas = sample_incident_rays(normals, training, pipe.diffuse_sample_num)
    elif pipe.diffuse_sample_num > 0 and pipe.light_sample_num > 0:
        p_diffuse = pipe.diffuse_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
        p_light = pipe.light_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
    
        diffuse_directions, diffuse_areas = sample_incident_rays(normals, training, pipe.diffuse_sample_num)
        diffuse_pdfs = 1 / diffuse_areas
        
        light_directions, light_pdfs = pc.get_envmap.sample_light_directions(B, pipe.light_sample_num, training)
    
        diffuse_pdfs_light = 1 / (2 * np.pi)
        light_pdfs_diffuse = pc.get_envmap.light_pdf(diffuse_directions)
        
        diffuse_pdfs = diffuse_pdfs * p_diffuse + light_pdfs_diffuse * p_light
        light_pdfs = diffuse_pdfs_light * p_diffuse + light_pdfs * p_light
        
        incident_dirs = torch.cat([diffuse_directions, light_directions], dim=1)
        incident_pdfs = torch.cat([diffuse_pdfs, light_pdfs], dim=1)
        incident_areas = 1 / incident_pdfs.clamp_min(1e-6)
    else:
        raise NotImplementedError
    global_incident_lights = envmap(incident_dirs, mode='pure_env')
    dielectric_f0 = _broadcast_to_target(f0, base_color)
    
    if relight:
        features = torch.cat([pc.get_base_color, pc.get_rough, pc.get_metallic], dim=1)
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, features=features, camera_center=camera_center)
        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_alpha
        trace_feature = trace_outputs['feature'] / trace_alpha.clamp_min(1e-6)
        trace_normal = F.normalize(trace_outputs['normal'], dim=-1)
        trace_base_color, trace_roughness, trace_metallic = trace_feature.split([3, 1, 1], dim=-1)
        trace_diffuse = (1 - trace_metallic) * trace_base_color * envmap(trace_normal, mode='diffuse')
        trace_wi = -incident_dirs
        trace_NdotV = (trace_normal * trace_wi).sum(-1, keepdim=True)
        trace_reflected = F.normalize(trace_NdotV * trace_normal * 2 - trace_wi, dim=-1)
        fg_uv = torch.cat([trace_NdotV, trace_roughness], -1).clamp(0, 1)
        fg = dr.texture(pc.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(*fg_uv.shape)
        trace_dielectric_f0 = _broadcast_to_target(f0, trace_base_color)
        trace_specular_f0 = trace_dielectric_f0 * (1 - trace_metallic) + trace_base_color * trace_metallic
        trace_specular = envmap(trace_reflected, roughness=trace_roughness, mode='specular') * (trace_specular_f0 * fg[..., 0:1] + fg[..., 1:2])
        local_incident_lights = (trace_diffuse + trace_specular) * trace_alpha
        if pipe.wo_indirect_relight:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    else:
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, camera_center=camera_center)
        incident_visibility = 1 - trace_outputs['alpha'][..., None]
        local_incident_lights = trace_outputs['color']
        if pipe.wo_indirect:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        if pipe.detach_indirect:
            incident_visibility = incident_visibility.detach()
            local_incident_lights = local_incident_lights.detach()
    incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    
    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    specular_f0 = dielectric_f0 * (1 - metallic) + base_color * metallic
    f_d = (1 - metallic)[:, None] * base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=specular_f0)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    diffuse = ((f_d) * transport).mean(dim=-2)
    specular = ((f_s) * transport).mean(dim=-2)

    if training:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "light_direct": global_incident_lights.mean(dim=1),
        }
    else:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
        }
    return results

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    fresnel = _broadcast_to_target(fresnel, alpha[:, None, :])
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec