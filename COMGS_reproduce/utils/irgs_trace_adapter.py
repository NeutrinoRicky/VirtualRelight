from typing import Dict, Optional

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation, build_scaling_rotation


def _safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def _flip_align_view(normal: torch.Tensor, viewdir: torch.Tensor) -> torch.Tensor:
    dotprod = torch.sum(normal * -viewdir, dim=-1, keepdim=True)
    return normal * torch.where(dotprod >= 0, 1, -1)


def _build_trace_features(gaussians, feature_mode: str = "comgs_pbr") -> torch.Tensor:
    if feature_mode == "comgs_pbr":
        return torch.cat(
            [
                gaussians.get_albedo,
                gaussians.get_roughness,
                gaussians.get_metallic,
            ],
            dim=1,
        )
    if feature_mode == "irgs_base_rough":
        return torch.cat(
            [
                gaussians.get_albedo,
                gaussians.get_roughness,
            ],
            dim=1,
        )
    raise ValueError(f"Unsupported trace feature mode: {feature_mode}")


def _split_trace_feature_outputs(
    feature_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if feature_tensor.shape[-1] < 4:
        raise ValueError(
            "Trace feature tensor must contain at least 4 channels "
            f"(got shape={tuple(feature_tensor.shape)})."
        )

    out_device = device if device is not None else feature_tensor.device
    out_dtype = dtype if dtype is not None else feature_tensor.dtype
    albedo = feature_tensor[..., :3]
    roughness = feature_tensor[..., 3:4]
    if feature_tensor.shape[-1] >= 5:
        metallic = feature_tensor[..., 4:5]
    else:
        metallic = torch.zeros(
            (*feature_tensor.shape[:-1], 1),
            device=out_device,
            dtype=out_dtype,
        )
    return albedo, roughness, metallic


class IRGS2DGaussianTraceAdapter:
    """
    Thin adapter around the IRGS-style 2D Gaussian ray tracing backend.

    It exposes the low-level traced alpha/color/feature/normal outputs without
    pulling in the rest of the IRGS training or shading pipeline.
    """

    def __init__(self, gaussians, alpha_min: float = 1.0 / 255.0, transmittance_min: float = 0.03):
        import trimesh
        from surfel_tracer import GaussianTracer

        self.gaussians = gaussians
        self.alpha_min = alpha_min
        self.transmittance_min = transmittance_min
        self.gaussian_tracer = GaussianTracer(transmittance_min=transmittance_min)

        icosahedron = trimesh.creation.icosahedron()
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()
        self._bvh_built = False

    @torch.no_grad()
    def rebuild(self, gaussians=None):
        if gaussians is not None:
            self.gaussians = gaussians

        mu = self.gaussians.get_xyz
        opacity = self.gaussians.get_opacity
        scale = torch.cat(
            [self.gaussians.get_scaling, torch.full_like(self.gaussians.get_scaling[:, :1], 1e-6)],
            dim=-1,
        )
        L = build_scaling_rotation(scale, self.gaussians._rotation)
        radius = torch.sqrt(torch.clamp(2.0 * torch.log(opacity / self.alpha_min), min=0.0))[:, None]
        vertices = radius * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device=mu.device)[:, None, None] * self.unit_icosahedron_vertices.shape[0]
        gs_idx = torch.arange(mu.shape[0], device=mu.device)[:, None].expand(-1, faces.shape[1])

        if self._bvh_built:
            self.gaussian_tracer.update_bvh(vertices.reshape(-1, 3), faces.reshape(-1, 3), gs_idx.reshape(-1))
        else:
            self.gaussian_tracer.build_bvh(vertices.reshape(-1, 3), faces.reshape(-1, 3), gs_idx.reshape(-1))
            self._bvh_built = True

    def trace(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        feature_mode: str = "comgs_pbr",
        camera_center: Optional[torch.Tensor] = None,
        back_culling: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self._bvh_built:
            self.rebuild(self.gaussians)

        prefix = ray_origins.shape[:-1]
        rays_o = ray_origins.reshape(-1, 3)
        rays_d = F.normalize(ray_directions.reshape(-1, 3), dim=-1, eps=1e-6)

        means3D = self.gaussians.get_xyz
        opacity = self.gaussians.get_opacity
        inv_scale = 1.0 / self.gaussians.get_scaling
        rotation = build_rotation(self.gaussians._rotation)
        ru = rotation[:, :, 0] * inv_scale[:, 0:1]
        rv = rotation[:, :, 1] * inv_scale[:, 1:2]
        splat2world = self.gaussians.get_covariance()
        normals_raw = splat2world[:, 2, :3]
        if camera_center is not None:
            normals_raw = _flip_align_view(normals_raw, means3D - camera_center.view(1, 3))
        normals = _safe_normalize(normals_raw)

        if features is None:
            features = _build_trace_features(self.gaussians, feature_mode=feature_mode)

        shs = self.gaussians.get_features
        color, normal, feature, depth, alpha = self.gaussian_tracer.trace(
            rays_o,
            rays_d,
            means3D,
            opacity,
            ru,
            rv,
            normals,
            features,
            shs,
            alpha_min=self.alpha_min,
            deg=self.gaussians.active_sh_degree,
            back_culling=back_culling,
        )

        alpha_ = alpha.unsqueeze(-1).clamp_min(1e-6)
        need_normalize = alpha.unsqueeze(-1) >= (1.0 - self.transmittance_min)
        color = torch.where(need_normalize, color / alpha_, color)
        normal = torch.where(need_normalize, normal / alpha_, normal)
        feature = torch.where(need_normalize, feature / alpha_, feature)
        depth = torch.where(alpha >= (1.0 - self.transmittance_min), depth / alpha.clamp_min(1e-6), depth)
        alpha = torch.where(alpha >= (1.0 - self.transmittance_min), torch.ones_like(alpha), alpha)

        positions = rays_o + depth.unsqueeze(-1) * rays_d
        hit_mask = alpha > 1e-4
        albedo, roughness, metallic = _split_trace_feature_outputs(
            feature,
            device=feature.device,
            dtype=feature.dtype,
        )

        outputs = {
            "color": color.view(*prefix, 3),
            "normal": normal.view(*prefix, 3),
            "feature": feature.view(*prefix, feature.shape[-1]),
            "depth": depth.view(*prefix),
            "alpha": alpha.view(*prefix),
            "position": positions.view(*prefix, 3),
            "hit_mask": hit_mask.view(*prefix),
            "albedo": torch.clamp(albedo, 0.0, 1.0).view(*prefix, 3),
            "roughness": torch.clamp(roughness, 0.0, 1.0).view(*prefix, 1),
            "metallic": torch.clamp(metallic, 0.0, 1.0).view(*prefix, 1),
        }
        return outputs
