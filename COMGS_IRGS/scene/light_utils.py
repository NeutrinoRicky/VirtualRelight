import os
import numpy as np

import torch
import nvdiffrast.torch as dr


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def uv_grid(width, height, center_x=0.5, center_y=0.5, device='cuda'):
    y, x = torch.meshgrid(
        (torch.arange(0, height, dtype=torch.float32, device=device) + center_y) / height,
        (torch.arange(0, width, dtype=torch.float32, device=device) + center_x) / width,
        indexing='ij',
    )
    return torch.stack((x, y), dim=-1)


def sign_not_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def direction_to_latlong_uv(direction: torch.Tensor) -> torch.Tensor:
    direction = safe_normalize(direction)
    u = torch.atan2(direction[..., 0:1], -direction[..., 2:3]).nan_to_num() / (2.0 * np.pi) + 0.5
    v = torch.acos(direction[..., 1:2].clamp(-1.0, 1.0)) / np.pi
    return torch.cat((u, v), dim=-1)


def latlong_uv_to_direction(uv: torch.Tensor) -> torch.Tensor:
    gx = uv[..., 0:1] * 2.0 - 1.0
    gy = uv[..., 1:2]
    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    return torch.cat((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)


def direction_to_octahedral_uv(direction: torch.Tensor) -> torch.Tensor:
    direction = safe_normalize(direction)
    denom = torch.abs(direction).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    projected = direction[..., :2] / denom
    projected_fold = (1.0 - torch.abs(projected[..., [1, 0]])) * sign_not_zero(projected)
    projected = torch.where(direction[..., 2:3] >= 0, projected, projected_fold)
    return projected * 0.5 + 0.5


def octahedral_uv_to_direction(uv: torch.Tensor) -> torch.Tensor:
    projected = uv * 2.0 - 1.0
    normal = torch.cat((projected, 1.0 - torch.abs(projected[..., 0:1]) - torch.abs(projected[..., 1:2])), dim=-1)
    fold = (-normal[..., 2:3]).clamp_min(0.0)
    xy = torch.where(normal[..., :2] >= 0, normal[..., :2] - fold, normal[..., :2] + fold)
    normal = torch.cat((xy, normal[..., 2:3]), dim=-1)
    return safe_normalize(normal)


def octahedral_solid_angle_jacobian(direction: torch.Tensor) -> torch.Tensor:
    direction = safe_normalize(direction)
    l1_norm = torch.abs(direction).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return 4.0 * l1_norm ** 3


def cube_to_dir(s, x, y):
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
            indexing='ij',
        )
        direction = safe_normalize(cube_to_dir(s, gx, gy))
        texcoord = direction_to_latlong_uv(direction)
        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


def latlong_to_octahedral(latlong_map, res, device='cuda'):
    uv = uv_grid(res[1], res[0], device=device)
    direction = octahedral_uv_to_direction(uv)
    texcoord = direction_to_latlong_uv(direction)
    return dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]


def octahedral_to_cubemap(octahedral_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], octahedral_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
            indexing='ij',
        )
        direction = safe_normalize(cube_to_dir(s, gx, gy))
        texcoord = direction_to_octahedral_uv(direction)
        cubemap[s, ...] = dr.texture(octahedral_map[None, ...], texcoord[None, ...], filter_mode='linear', boundary_mode='clamp')[0]
    return cubemap


def cubemap_to_latlong(cubemap, res, device='cuda'):
    uv = uv_grid(res[1], res[0], device=device)
    direction = latlong_uv_to_direction(uv)
    return dr.texture(cubemap[None, ...], direction[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


def cubemap_to_octahedral(cubemap, res, device='cuda'):
    uv = uv_grid(res[1], res[0], device=device)
    direction = octahedral_uv_to_direction(uv)
    return dr.texture(cubemap[None, ...], direction[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device),
                indexing='ij',
            )
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out
