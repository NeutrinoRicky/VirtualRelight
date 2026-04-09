import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from utils.deferred_pbr_comgs import ggx_specular_irgs


def ggx_specular_irgs_compat(
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    roughness: torch.Tensor,
    fresnel: float = 0.04,
    eps: float = 1e-6,
) -> torch.Tensor:
    # IRGS-compatible compatibility path: keep the fixed-F0 GGX semantics
    # isolated from the legacy COMGS metallic workflow.
    return ggx_specular_irgs(
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
        roughness=roughness,
        fresnel=fresnel,
        eps=eps,
    )


def compose_incident_lights_irgs_compat(
    trace_alpha: torch.Tensor,
    trace_color: torch.Tensor,
    direct_env_radiance: torch.Tensor,
) -> torch.Tensor:
    safe_alpha = torch.nan_to_num(trace_alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    safe_trace_color = torch.nan_to_num(trace_color, nan=0.0, posinf=0.0, neginf=0.0)
    safe_direct_env = torch.nan_to_num(direct_env_radiance, nan=0.0, posinf=0.0, neginf=0.0)
    return (1.0 - safe_alpha) * safe_direct_env + safe_trace_color


def integrate_incident_radiance_irgs_compat(
    base_color: torch.Tensor,
    roughness: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    lightdirs: torch.Tensor,
    incident_radiance: torch.Tensor,
    incident_weights_or_areas: torch.Tensor,
    fresnel: float = 0.04,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    normals = F.normalize(normals, dim=-1, eps=eps)
    lightdirs = F.normalize(lightdirs, dim=-1, eps=eps)
    n_dot_l = (normals[:, None, :] * lightdirs).sum(dim=-1, keepdim=True).clamp(min=0.0)

    diffuse_brdf = base_color[:, None, :] / math.pi
    specular_brdf = ggx_specular_irgs_compat(
        normals=normals,
        viewdirs=viewdirs,
        lightdirs=lightdirs,
        roughness=roughness,
        fresnel=fresnel,
        eps=eps,
    )

    sample_weight = torch.nan_to_num(incident_weights_or_areas, nan=0.0, posinf=0.0, neginf=0.0)
    transport = incident_radiance * sample_weight * n_dot_l
    diffuse = (diffuse_brdf * transport).mean(dim=1)
    specular = (specular_brdf * transport).mean(dim=1)
    shaded = diffuse + specular
    return shaded, {
        "diffuse": diffuse,
        "specular": specular,
        "n_dot_l": n_dot_l,
        "transport": transport,
    }
