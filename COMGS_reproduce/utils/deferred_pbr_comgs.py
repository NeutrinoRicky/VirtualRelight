import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sop_utils import build_octahedral_direction_grid, dir_to_oct_uv, oct_uv_to_dir, sample_octahedral_texture


DEFAULT_OCT_ENV_HEIGHT = 256
DEFAULT_OCT_ENV_WIDTH = 256


def depths_to_points(view, depthmap: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map [1, H, W] to world-space points [H, W, 3].
    """
    rays_d_hw = getattr(view, "rays_d_hw_unnormalized", None)
    camera_center = getattr(view, "camera_center", None)
    if rays_d_hw is not None and camera_center is not None:
        rays_d_hw = rays_d_hw.to(device=depthmap.device, dtype=depthmap.dtype)
        camera_center = camera_center.to(device=depthmap.device, dtype=depthmap.dtype)
        points = depthmap.permute(1, 2, 0) * rays_d_hw + camera_center
        return torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    c2w = (view.world_view_transform.T).inverse()
    width, height = view.image_width, view.image_height
    device = depthmap.device
    dtype = depthmap.dtype

    ndc2pix = torch.tensor(
        [
            [width / 2.0, 0.0, 0.0, width / 2.0],
            [0.0, height / 2.0, 0.0, height / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    ).T

    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(height, device=device, dtype=dtype),
        indexing="xy",
    )
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

    rays_d = pixels @ torch.inverse(intrins).T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    points = points.reshape(height, width, 3)
    return torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def recover_shading_points(
    view,
    depth_unbiased: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    weight_threshold: float = 1e-4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    points = depths_to_points(view, depth_unbiased)
    valid = torch.isfinite(depth_unbiased)
    valid = valid & (depth_unbiased > eps)
    if weight is not None:
        valid = valid & (weight > weight_threshold)
    return points, valid.squeeze(0)


def compute_view_directions(points: torch.Tensor, camera_center: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(camera_center.view(1, 3) - points, dim=-1, eps=eps)


def rgb_to_srgb(x: torch.Tensor, clip: bool = True) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = torch.tensor(0.0031308, device=y.device, dtype=y.dtype)
    y = torch.where(
        y > threshold,
        torch.pow(torch.max(y, threshold), 1.0 / 2.4) * 1.055 - 0.055,
        12.92 * y,
    )
    if clip:
        y = torch.clamp(y, 0.0, 1.0)
    return y


def _rgb_to_srgb_for_vis(x: torch.Tensor, clip: bool = True) -> torch.Tensor:
    return rgb_to_srgb(x.detach(), clip=clip)


def _tonemap_hdr_for_vis(x: torch.Tensor) -> torch.Tensor:
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.clamp(y, min=0.0)
    y = y / (1.0 + y)
    return _rgb_to_srgb_for_vis(y, clip=True)


def _radical_inverse_vdc(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(dtype=torch.int64)
    bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
    bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
    bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return bits.to(dtype=torch.float32) * 2.3283064365386963e-10


def hammersley_sequence(num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(num_samples, device=device, dtype=torch.int64)
    u = idx.to(dtype=dtype) / float(max(num_samples, 1))
    v = _radical_inverse_vdc(idx).to(dtype=dtype)
    return torch.stack([u, v], dim=-1)


def _build_local_frame(normals: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    up_z = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype).view(1, 3)
    up_y = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype).view(1, 3)
    use_y = torch.abs(normals[:, 2:3]) > 0.999
    up = torch.where(use_y, up_y, up_z)
    tangent = F.normalize(torch.cross(up.expand_as(normals), normals, dim=-1), dim=-1, eps=eps)
    bitangent = F.normalize(torch.cross(normals, tangent, dim=-1), dim=-1, eps=eps)
    return tangent, bitangent


def sample_hemisphere_hammersley(
    normals: torch.Tensor,
    num_samples: int,
    randomized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uniform hemisphere sampling with Hammersley points.

    Returns:
    - directions: [N, S, 3]
    - pdf: [N, S, 1]
    - solid_angle_per_sample: scalar tensor
    """
    if normals.numel() == 0:
        empty_dirs = normals.new_zeros((0, num_samples, 3))
        empty_pdf = normals.new_zeros((0, num_samples, 1))
        return empty_dirs, empty_pdf, normals.new_tensor(0.0)

    nrm = F.normalize(normals, dim=-1, eps=1e-6)
    seq = hammersley_sequence(num_samples, nrm.device, nrm.dtype)
    if randomized:
        shift = torch.rand((nrm.shape[0], 1, 2), device=nrm.device, dtype=nrm.dtype)
        seq = (seq.unsqueeze(0) + shift) % 1.0
    else:
        seq = seq.unsqueeze(0).expand(nrm.shape[0], -1, -1)

    phi = 2.0 * math.pi * seq[..., 0]
    cos_theta = seq[..., 1].clamp(0.0, 1.0)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))

    local_dirs = torch.stack(
        [
            torch.cos(phi) * sin_theta,
            torch.sin(phi) * sin_theta,
            cos_theta,
        ],
        dim=-1,
    )

    tangent, bitangent = _build_local_frame(nrm)
    world_dirs = (
        tangent[:, None, :] * local_dirs[..., 0:1]
        + bitangent[:, None, :] * local_dirs[..., 1:2]
        + nrm[:, None, :] * local_dirs[..., 2:3]
    )
    world_dirs = F.normalize(world_dirs, dim=-1, eps=1e-6)

    pdf = torch.full(
        (nrm.shape[0], num_samples, 1),
        1.0 / (2.0 * math.pi),
        device=nrm.device,
        dtype=nrm.dtype,
    )
    solid_angle = nrm.new_tensor(2.0 * math.pi / float(max(num_samples, 1)))
    return world_dirs, pdf, solid_angle


def _rotation_between_z(vec: torch.Tensor) -> torch.Tensor:
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1.0).clamp_min(1e-7)
    rot = torch.zeros(vec.shape[:-1] + (3, 3), dtype=vec.dtype, device=vec.device)
    rot[..., 0, 0] = 1.0 + (-v33 - v22) / cos_p_1
    rot[..., 0, 1] = -v3 + v12 / cos_p_1
    rot[..., 0, 2] = v2 + v13 / cos_p_1
    rot[..., 1, 0] = v3 + v12 / cos_p_1
    rot[..., 1, 1] = 1.0 + (-v33 - v11) / cos_p_1
    rot[..., 1, 2] = -v1 + v23 / cos_p_1
    rot[..., 2, 0] = -v2 + v13 / cos_p_1
    rot[..., 2, 1] = v1 + v23 / cos_p_1
    rot[..., 2, 2] = 1.0 + (-v22 - v11) / cos_p_1
    neg_eye = -torch.eye(3, dtype=vec.dtype, device=vec.device).expand_as(rot)
    return torch.where((vec[..., 2] + 1.0 > 0.0)[..., None, None], rot, neg_eye)


def sample_incident_rays_irgs(
    normals: torch.Tensor,
    training: bool = False,
    sample_num: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pre_shape = normals.shape[:-1]
    flat_normals = normals.reshape(-1, 3) if len(pre_shape) > 1 else normals
    flat_normals = F.normalize(flat_normals, dim=-1, eps=1e-6)

    delta = math.pi * (3.0 - math.sqrt(5.0))
    idx = torch.arange(sample_num, dtype=flat_normals.dtype, device=flat_normals.device)[None]
    denom = float(max(2 * sample_num - 1, 1))
    min_z = math.sin(math.radians(10.0))
    z = (1.0 - 2.0 * idx / denom).clamp_min(min_z)
    rad = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = delta * idx
    if training:
        theta = torch.rand((flat_normals.shape[0], 1), device=flat_normals.device, dtype=flat_normals.dtype) * (2.0 * math.pi) + theta
    else:
        theta = theta.expand(flat_normals.shape[0], -1)
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    rotation = _rotation_between_z(flat_normals)
    incident_dirs = rotation @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2, eps=1e-6).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs[..., 0:1]) * (2.0 * math.pi)

    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas


class LatLongEnvMap(nn.Module):
    def __init__(
        self,
        height: int = 32,
        width: Optional[int] = None,
        init_value: float = 0.5,
        activation: str = "exp",
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width) if width is not None else int(height) * 2
        self.activation_name = activation

        base = torch.full((self.height, self.width, 3), float(init_value), dtype=torch.float32)
        if activation == "exp":
            self.base = nn.Parameter(torch.log(torch.clamp(base, min=1e-4)))
            self._activation = torch.exp
        elif activation == "softplus":
            self.base = nn.Parameter(torch.log(torch.exp(base) - 1.0 + 1e-6))
            self._activation = F.softplus
        elif activation == "none":
            self.base = nn.Parameter(base)
            self._activation = lambda x: x
        else:
            raise ValueError(f"Unsupported envmap activation: {activation}")

        self.register_buffer("_pdf", torch.empty(0), persistent=False)
        self.update_pdf()

    def activated_map(self) -> torch.Tensor:
        return torch.clamp(self._activation(self.base), min=0.0)

    def update_pdf(self):
        with torch.no_grad():
            env = self.activated_map().detach()
            theta = torch.linspace(
                0.5 / self.height,
                1.0 - 0.5 / self.height,
                self.height,
                device=env.device,
                dtype=env.dtype,
            ) * math.pi
            sin_theta = torch.sin(theta).view(self.height, 1)
            luminance = env.max(dim=-1).values.clamp_min(1e-8)
            pdf = luminance * sin_theta
            pdf = pdf / pdf.sum().clamp_min(1e-8)
            self._pdf = pdf

    def sample_light_directions(self, batch_size: int, sample_num: int, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._pdf.numel() == 0 or self._pdf.shape[0] != self.height or self._pdf.shape[1] != self.width:
            self.update_pdf()

        pdf_flat = self._pdf.reshape(-1)
        idx = torch.multinomial(pdf_flat, batch_size * sample_num, replacement=True)
        x = (idx % self.width).to(dtype=self.base.dtype)
        y = (idx // self.width).to(dtype=self.base.dtype)

        u = (x + 0.5) / float(self.width)
        v = (y + 0.5) / float(self.height)
        if training:
            u = (u + (torch.rand_like(u) - 0.5) / float(self.width)).remainder(1.0)
            v = (v + (torch.rand_like(v) - 0.5) / float(self.height)).clamp(1e-5, 1.0 - 1e-5)

        phi = (u - 0.5) * (2.0 * math.pi)
        theta = v * math.pi
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        directions = torch.stack(
            [
                sin_theta * torch.sin(phi),
                cos_theta,
                -sin_theta * torch.cos(phi),
            ],
            dim=-1,
        )
        directions = directions.reshape(batch_size, sample_num, 3)
        probability = self.light_pdf(directions)
        return directions, probability

    def light_pdf(self, directions: torch.Tensor) -> torch.Tensor:
        if self._pdf.numel() == 0 or self._pdf.shape[0] != self.height or self._pdf.shape[1] != self.width:
            self.update_pdf()

        flat_dirs = directions.reshape(-1, 3)
        u = torch.atan2(flat_dirs[:, 0], -flat_dirs[:, 2]).nan_to_num() / (2.0 * math.pi) + 0.5
        v = torch.acos(flat_dirs[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / math.pi

        x = torch.clamp((u * self.width).long(), min=0, max=self.width - 1)
        y = torch.clamp((v * self.height).long(), min=0, max=self.height - 1)
        flat_idx = y * self.width + x
        pdf_flat = self._pdf.reshape(-1)
        area_pdf = torch.take_along_dim(pdf_flat, flat_idx, dim=0)
        sin_theta = torch.sin(v * math.pi).clamp_min(1e-6)
        direction_pdf = area_pdf * (self.height * self.width) / (2.0 * math.pi * math.pi * sin_theta)
        return direction_pdf.reshape(*directions.shape[:-1], 1)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        env = self.activated_map().permute(2, 0, 1).unsqueeze(0)
        shape = directions.shape[:-1]
        flat_dirs = directions.reshape(-1, 3)

        u = torch.atan2(flat_dirs[:, 0], -flat_dirs[:, 2]).nan_to_num() / math.pi
        v = 2.0 * (torch.acos(flat_dirs[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / math.pi) - 1.0
        grid = torch.stack([u, v], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(env, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.view(3, -1).T.reshape(*shape, 3)
        return torch.clamp(sampled, min=0.0)

    def capture(self) -> Dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "height": self.height,
            "width": self.width,
            "activation": self.activation_name,
        }

    @classmethod
    def from_capture(cls, payload: Dict[str, object]) -> "LatLongEnvMap":
        envmap = cls(
            height=payload["height"],
            width=payload["width"],
            activation=payload["activation"],
        )
        envmap.load_state_dict(payload["state_dict"])
        envmap.update_pdf()
        return envmap

    def visualization(self) -> torch.Tensor:
        env = _tonemap_hdr_for_vis(self.activated_map())
        return env.permute(2, 0, 1)


class OctahedralEnvMap(nn.Module):
    def __init__(
        self,
        height: int = 32,
        width: Optional[int] = None,
        init_value: float = 0.5,
        activation: str = "exp",
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width) if width is not None else int(height)
        self.activation_name = activation

        base = torch.full((self.height, self.width, 3), float(init_value), dtype=torch.float32)
        if activation == "exp":
            self.base = nn.Parameter(torch.log(torch.clamp(base, min=1e-4)))
            self._activation = torch.exp
        elif activation == "softplus":
            self.base = nn.Parameter(torch.log(torch.exp(base) - 1.0 + 1e-6))
            self._activation = F.softplus
        elif activation == "none":
            self.base = nn.Parameter(base)
            self._activation = lambda x: x
        else:
            raise ValueError(f"Unsupported envmap activation: {activation}")

        self.register_buffer("_pdf", torch.empty(0), persistent=False)
        self.update_pdf()

    def activated_map(self) -> torch.Tensor:
        return torch.clamp(self._activation(self.base), min=0.0)

    def update_pdf(self):
        with torch.no_grad():
            env = self.activated_map().detach()
            luminance = env.max(dim=-1).values.clamp_min(1e-8)
            pdf = luminance / luminance.sum().clamp_min(1e-8)
            self._pdf = pdf

    def sample_light_directions(self, batch_size: int, sample_num: int, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._pdf.numel() == 0 or self._pdf.shape[0] != self.height or self._pdf.shape[1] != self.width:
            self.update_pdf()

        pdf_flat = self._pdf.reshape(-1)
        idx = torch.multinomial(pdf_flat, batch_size * sample_num, replacement=True)
        x = (idx % self.width).to(dtype=self.base.dtype)
        y = (idx // self.width).to(dtype=self.base.dtype)

        u = (x + 0.5) / float(self.width)
        v = (y + 0.5) / float(self.height)
        if training:
            u = torch.clamp(u + (torch.rand_like(u) - 0.5) / float(self.width), 1e-6, 1.0 - 1e-6)
            v = torch.clamp(v + (torch.rand_like(v) - 0.5) / float(self.height), 1e-6, 1.0 - 1e-6)

        uv = torch.stack([u, v], dim=-1)
        directions = oct_uv_to_dir(uv).reshape(batch_size, sample_num, 3)
        probability = self.light_pdf(directions)
        return directions, probability

    def light_pdf(self, directions: torch.Tensor) -> torch.Tensor:
        if self._pdf.numel() == 0 or self._pdf.shape[0] != self.height or self._pdf.shape[1] != self.width:
            self.update_pdf()

        flat_dirs = directions.reshape(-1, 3)
        uv = dir_to_oct_uv(flat_dirs)
        x = torch.clamp((uv[:, 0] * self.width).long(), min=0, max=self.width - 1)
        y = torch.clamp((uv[:, 1] * self.height).long(), min=0, max=self.height - 1)
        flat_idx = y * self.width + x
        pdf_flat = self._pdf.reshape(-1)
        texel_pdf = torch.take_along_dim(pdf_flat, flat_idx, dim=0)
        # Approximate each oct texel with equal solid angle to keep runtime sampling simple.
        texel_solid_angle = (4.0 * math.pi) / float(max(self.height * self.width, 1))
        direction_pdf = texel_pdf / max(texel_solid_angle, 1e-8)
        return direction_pdf.reshape(*directions.shape[:-1], 1)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        env = self.activated_map()
        sampled = sample_octahedral_texture(env, directions)
        return torch.clamp(sampled, min=0.0)

    def capture(self) -> Dict[str, object]:
        return {
            "projection": "octahedral",
            "state_dict": self.state_dict(),
            "height": self.height,
            "width": self.width,
            "activation": self.activation_name,
        }

    @classmethod
    def from_capture(cls, payload: Dict[str, object]) -> "OctahedralEnvMap":
        envmap = cls(
            height=payload["height"],
            width=payload["width"],
            activation=payload["activation"],
        )
        envmap.load_state_dict(payload["state_dict"])
        envmap.update_pdf()
        return envmap

    @classmethod
    def from_latlong_envmap(
        cls,
        latlong_envmap: LatLongEnvMap,
        height: int,
        width: Optional[int] = None,
    ) -> "OctahedralEnvMap":
        oct_width = int(width) if width is not None else int(height)
        env = latlong_envmap.activated_map().detach()
        device = env.device
        dtype = env.dtype
        oct_dirs = build_octahedral_direction_grid(
            tex_h=int(height),
            tex_w=oct_width,
            device=device,
            dtype=dtype,
        )
        oct_tex = latlong_envmap(oct_dirs).detach()
        envmap = cls(height=int(height), width=oct_width, activation="none")
        envmap = envmap.to(device=device, dtype=dtype)
        with torch.no_grad():
            envmap.base.copy_(oct_tex)
        envmap.update_pdf()
        return envmap

    def visualization(self) -> torch.Tensor:
        env = _tonemap_hdr_for_vis(self.activated_map())
        return env.permute(2, 0, 1)


def convert_envmap_to_octahedral(
    envmap: Union[LatLongEnvMap, OctahedralEnvMap],
    height: int = DEFAULT_OCT_ENV_HEIGHT,
    width: int = DEFAULT_OCT_ENV_WIDTH,
) -> OctahedralEnvMap:
    target_h = int(height)
    target_w = int(width)
    if isinstance(envmap, OctahedralEnvMap) and envmap.height == target_h and envmap.width == target_w:
        return envmap

    if isinstance(envmap, LatLongEnvMap):
        return OctahedralEnvMap.from_latlong_envmap(envmap, height=target_h, width=target_w)

    device = envmap.base.device
    dtype = envmap.base.dtype
    oct_dirs = build_octahedral_direction_grid(
        tex_h=target_h,
        tex_w=target_w,
        device=device,
        dtype=dtype,
    )
    oct_tex = envmap(oct_dirs).detach()
    oct_envmap = OctahedralEnvMap(height=target_h, width=target_w, activation="none").to(device=device, dtype=dtype)
    with torch.no_grad():
        oct_envmap.base.copy_(oct_tex)
    oct_envmap.update_pdf()
    return oct_envmap


def load_envmap_capture_as_octahedral(
    payload: Dict[str, object],
    height: int = DEFAULT_OCT_ENV_HEIGHT,
    width: int = DEFAULT_OCT_ENV_WIDTH,
) -> OctahedralEnvMap:
    projection = str(payload.get("projection", "latlong")).lower()
    if projection == "octahedral":
        envmap = OctahedralEnvMap.from_capture(payload)
    else:
        envmap = LatLongEnvMap.from_capture(payload)
    return convert_envmap_to_octahedral(envmap, height=height, width=width)


def fresnel_schlick(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta.clamp(0.0, 1.0), 5.0)


def ggx_specular_irgs(
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    roughness: torch.Tensor,
    fresnel: float = 0.04,
    eps: float = 1e-6,
) -> torch.Tensor:
    light = F.normalize(lightdirs, dim=-1, eps=eps)
    view = F.normalize(viewdirs, dim=-1, eps=eps)
    half = F.normalize((light + view[:, None, :]) / 2.0, dim=-1, eps=eps)
    normal = F.normalize(normals, dim=-1, eps=eps)

    n_dot_v = torch.sum(view * normal, dim=-1, keepdim=True)
    normal = normal * n_dot_v.sign()

    n_dot_l = torch.sum(normal[:, None, :] * light, dim=-1, keepdim=True).clamp_(eps, 1.0)
    n_dot_v = torch.sum(normal * view, dim=-1, keepdim=True).clamp_(eps, 1.0)
    n_dot_h = torch.sum(normal[:, None, :] * half, dim=-1, keepdim=True).clamp_(eps, 1.0)
    v_dot_h = torch.sum(view[:, None, :] * half, dim=-1, keepdim=True).clamp_(eps, 1.0)

    alpha = roughness * roughness
    alpha2 = alpha * alpha
    k = (alpha + 2.0 * roughness + 1.0) / 8.0
    fmi = ((-5.55473) * v_dot_h - 6.98316) * v_dot_h
    frac0 = fresnel + (1.0 - fresnel) * torch.pow(2.0, fmi)

    frac = frac0 * alpha2[:, None, :]
    denom0 = n_dot_h * n_dot_h * (alpha2[:, None, :] - 1.0) + 1.0
    denom1 = n_dot_v * (1.0 - k) + k
    denom2 = n_dot_l * (1.0 - k[:, None, :]) + k[:, None, :]
    denom = (4.0 * math.pi * denom0 * denom0 * denom1[:, None, :] * denom2).clamp_(eps, 4.0 * math.pi)
    return frac / denom


def evaluate_microfacet_brdf(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    nrm = F.normalize(normals, dim=-1, eps=eps)
    v = F.normalize(viewdirs, dim=-1, eps=eps)[:, None, :]
    l = F.normalize(lightdirs, dim=-1, eps=eps)
    h = F.normalize(v + l, dim=-1, eps=eps)

    n = nrm[:, None, :]
    n_dot_l = (n * l).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    n_dot_v = (nrm * viewdirs).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)[:, None, :]
    n_dot_h = (n * h).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    v_dot_h = (v * h).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)

    rough = roughness.clamp(0.04, 1.0)[:, None, :]
    alpha = rough * rough
    alpha2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0
    D = alpha2 / (math.pi * denom * denom + eps)

    k = ((rough + 1.0) * (rough + 1.0)) / 8.0
    G_v = n_dot_v / (n_dot_v * (1.0 - k) + k + eps)
    G_l = n_dot_l / (n_dot_l * (1.0 - k) + k + eps)
    G = G_v * G_l

    base_f0 = torch.full_like(albedo, 0.04)
    f0 = base_f0 * (1.0 - metallic) + albedo * metallic
    F_term = fresnel_schlick(v_dot_h, f0[:, None, :])

    diffuse_color = albedo * (1.0 - metallic)
    diffuse = diffuse_color[:, None, :] / math.pi
    specular = (D * G) * F_term / (4.0 * n_dot_v * n_dot_l + eps)

    return diffuse + specular, {
        "diffuse": diffuse,
        "specular": specular,
        "n_dot_l": n_dot_l,
        "n_dot_v": n_dot_v,
    }


def integrate_incident_radiance(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    incident_radiance: torch.Tensor,
    sample_solid_angle: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    brdf, brdf_aux = evaluate_microfacet_brdf(
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
    )
    transport = incident_radiance * brdf_aux["n_dot_l"] * sample_solid_angle
    diffuse = (brdf_aux["diffuse"] * transport).sum(dim=1)
    specular = (brdf_aux["specular"] * transport).sum(dim=1)
    shaded = diffuse + specular
    return shaded, {
        "diffuse": diffuse,
        "specular": specular,
        "n_dot_l": brdf_aux["n_dot_l"],
        "brdf": brdf,
    }


def integrate_incident_radiance_irgs(
    base_color: torch.Tensor,
    roughness: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    incident_radiance: torch.Tensor,
    incident_areas: torch.Tensor,
    fresnel: float = 0.04,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lightdirs = F.normalize(lightdirs, dim=-1, eps=eps)
    normals = F.normalize(normals, dim=-1, eps=eps)
    n_dot_l = (normals[:, None, :] * lightdirs).sum(-1, keepdim=True).clamp(min=0.0)
    diffuse_brdf = base_color[:, None, :] / math.pi
    specular_brdf = ggx_specular_irgs(
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
        roughness=roughness,
        fresnel=fresnel,
        eps=eps,
    )
    transport = incident_radiance * incident_areas * n_dot_l
    diffuse = (diffuse_brdf * transport).mean(dim=1)
    specular = (specular_brdf * transport).mean(dim=1)
    shaded = diffuse + specular
    return shaded, {
        "diffuse": diffuse,
        "specular": specular,
        "n_dot_l": n_dot_l,
    }


def integrate_incident_radiance_importance(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    incident_radiance: torch.Tensor,
    light_pdf: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    brdf, brdf_aux = evaluate_microfacet_brdf(
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
        eps=eps,
    )
    safe_pdf = torch.clamp(torch.nan_to_num(light_pdf, nan=0.0, posinf=0.0, neginf=0.0), min=eps)
    transport = incident_radiance * brdf_aux["n_dot_l"] / safe_pdf
    diffuse = (brdf_aux["diffuse"] * transport).mean(dim=1)
    specular = (brdf_aux["specular"] * transport).mean(dim=1)
    shaded = diffuse + specular
    return shaded, {
        "diffuse": diffuse,
        "specular": specular,
        "n_dot_l": brdf_aux["n_dot_l"],
        "brdf": brdf,
        "light_pdf": safe_pdf,
    }


def shade_secondary_points(
    envmap,
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    num_samples: int,
    randomized: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(
        normals=normals,
        num_samples=num_samples,
        randomized=randomized,
    )
    env_radiance = envmap(lightdirs)
    shaded, aux = integrate_incident_radiance(
        albedo=albedo,
        roughness=roughness,
        metallic=metallic,
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
        incident_radiance=env_radiance,
        sample_solid_angle=sample_solid_angle,
    )
    aux.update({"env_radiance": env_radiance, "lightdirs": lightdirs})
    return shaded, aux
