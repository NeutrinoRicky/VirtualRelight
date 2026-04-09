import math
from typing import Dict

import torch

from utils.deferred_pbr_comgs import sample_incident_rays_irgs


def _empty_sampling_outputs(
    normals: torch.Tensor,
    sample_num: int,
) -> Dict[str, torch.Tensor]:
    return {
        "incident_dirs": normals.new_zeros((normals.shape[0], sample_num, 3)),
        "incident_pdf": normals.new_zeros((normals.shape[0], sample_num, 1)),
        "sample_weight": normals.new_zeros((normals.shape[0], sample_num, 1)),
        "incident_areas": normals.new_zeros((normals.shape[0], sample_num, 1)),
    }


def sample_incident_dirs_diffuse_irgs_compat(
    normals: torch.Tensor,
    sample_num: int,
    randomized: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    # IRGS-compatible compatibility path: keep diffuse sampling semantics close
    # to IRGS while isolating the new path from the legacy COMGS code.
    if sample_num <= 0 or normals.numel() == 0:
        return _empty_sampling_outputs(normals, max(sample_num, 0))

    incident_dirs, incident_areas = sample_incident_rays_irgs(
        normals=normals,
        training=randomized,
        sample_num=sample_num,
    )
    incident_pdf = 1.0 / incident_areas.clamp_min(eps)
    return {
        "incident_dirs": incident_dirs,
        "incident_pdf": incident_pdf,
        "sample_weight": incident_areas,
        "incident_areas": incident_areas,
    }


def sample_incident_dirs_env_irgs_compat(
    normals: torch.Tensor,
    envmap,
    sample_num: int,
    randomized: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    if sample_num <= 0:
        return _empty_sampling_outputs(normals, 0)

    if not hasattr(envmap, "sample_light_directions") or not hasattr(envmap, "light_pdf"):
        raise AttributeError("envmap must provide sample_light_directions(...) and light_pdf(...) for IRGS-compatible sampling")

    batch_size = normals.shape[0]
    incident_dirs, incident_pdf = envmap.sample_light_directions(
        batch_size=batch_size,
        sample_num=sample_num,
        training=randomized,
    )
    sample_weight = 1.0 / incident_pdf.clamp_min(eps)
    return {
        "incident_dirs": incident_dirs,
        "incident_pdf": incident_pdf,
        "sample_weight": sample_weight,
        "incident_areas": sample_weight,
    }


def sample_incident_dirs_mixture_irgs_compat(
    normals: torch.Tensor,
    envmap,
    diffuse_sample_num: int,
    light_sample_num: int,
    randomized: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    # IRGS-compatible compatibility path: combine diffuse hemisphere sampling
    # with env importance sampling using a mixture PDF.
    total_samples = int(max(diffuse_sample_num, 0) + max(light_sample_num, 0))
    if total_samples <= 0 or normals.numel() == 0:
        return _empty_sampling_outputs(normals, max(total_samples, 0))

    diffuse_outputs = sample_incident_dirs_diffuse_irgs_compat(
        normals=normals,
        sample_num=max(diffuse_sample_num, 0),
        randomized=randomized,
        eps=eps,
    )
    light_outputs = sample_incident_dirs_env_irgs_compat(
        normals=normals,
        envmap=envmap,
        sample_num=max(light_sample_num, 0),
        randomized=randomized,
        eps=eps,
    )

    if diffuse_sample_num <= 0:
        return light_outputs
    if light_sample_num <= 0:
        return diffuse_outputs

    p_diffuse = float(diffuse_sample_num) / float(total_samples)
    p_light = float(light_sample_num) / float(total_samples)

    diffuse_dirs = diffuse_outputs["incident_dirs"]
    light_dirs = light_outputs["incident_dirs"]
    diffuse_pdf_uniform = diffuse_outputs["incident_pdf"]
    light_pdf_env = light_outputs["incident_pdf"]

    diffuse_pdf_env = envmap.light_pdf(diffuse_dirs)
    light_pdf_uniform = torch.full_like(light_pdf_env, 1.0 / (2.0 * math.pi))

    diffuse_pdf = diffuse_pdf_uniform * p_diffuse + diffuse_pdf_env * p_light
    light_pdf = light_pdf_uniform * p_diffuse + light_pdf_env * p_light

    incident_dirs = torch.cat([diffuse_dirs, light_dirs], dim=1)
    incident_pdf = torch.cat([diffuse_pdf, light_pdf], dim=1)
    sample_weight = 1.0 / incident_pdf.clamp_min(eps)
    return {
        "incident_dirs": incident_dirs,
        "incident_pdf": incident_pdf,
        "sample_weight": sample_weight,
        "incident_areas": sample_weight,
    }
