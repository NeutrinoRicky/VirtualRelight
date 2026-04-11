from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torchvision
from plyfile import PlyData

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.sop_utils import build_octahedral_direction_grid


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tonemap_for_vis(x: torch.Tensor) -> torch.Tensor:
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(y / (1.0 + y), 0.0, 1.0)


def _repeat_single_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 1:
        return x.repeat(1, 1, 1, 3)
    return x


def _set_gaussian_material_defaults(gaussians: GaussianModel, opt) -> None:
    gaussians.init_base_color_value = getattr(opt, "init_base_color_value", gaussians.init_base_color_value)
    gaussians.init_metallic_value = getattr(opt, "init_metallic_value", gaussians.init_metallic_value)
    gaussians.init_roughness_value = getattr(opt, "init_roughness_value", gaussians.init_roughness_value)


def _load_checkpoint_into_gaussians(gaussians: GaussianModel, opt, checkpoint_path: str) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=torch.device("cuda"), weights_only=False)
    if isinstance(payload, dict) and "gaussians" in payload:
        gaussians.restore(payload["gaussians"], opt)
        return {
            "source": "checkpoint_dict",
            "path": checkpoint_path,
            "iteration": int(payload.get("iteration", 0)),
            "format": str(payload.get("format", "checkpoint_dict")),
        }

    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

    model_params = payload[0]
    iteration = int(payload[1]) if len(payload) >= 2 else 0
    if isinstance(model_params, (tuple, list)) and len(model_params) in {16}:
        gaussians.restore(model_params, opt)
        label = "irgs_checkpoint"
    elif isinstance(model_params, (tuple, list)) and len(model_params) in {19, 26}:
        gaussians.restore_from_refgs(model_params, opt)
        label = "irgs_refgs"
    else:
        raise RuntimeError(f"Unsupported checkpoint model args len={len(model_params) if isinstance(model_params, (tuple, list)) else 'n/a'}")

    return {
        "source": "checkpoint_tuple",
        "path": checkpoint_path,
        "iteration": iteration,
        "format": label,
    }


def _prepare_scene_and_gaussians(args, dataset, pipe, opt) -> Tuple[Scene, GaussianModel, Dict[str, object]]:
    gaussians = GaussianModel(dataset.sh_degree)
    _set_gaussian_material_defaults(gaussians, opt)

    load_iteration = None if args.checkpoint else args.iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)

    if args.checkpoint:
        loaded_from = _load_checkpoint_into_gaussians(gaussians, opt, args.checkpoint)
    else:
        loaded_from = {
            "source": "scene_iteration",
            "iteration": int(scene.loaded_iter) if scene.loaded_iter is not None else 0,
        }

    if scene.light_rotate:
        transform = torch.tensor(
            [
                [0, -1, 0],
                [0, 0, 1],
                [-1, 0, 0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        gaussians.env_map.set_transform(transform)

    gaussians.alpha_min = float(args.native_alpha_min)
    gaussians.gaussian_tracer.transmittance_min = float(args.native_transmittance_min)
    gaussians.build_bvh()
    return scene, gaussians, loaded_from


def _resolve_probe_file(path_str: str, model_path: str) -> Path:
    if path_str:
        path = Path(path_str)
    else:
        default_root = Path(model_path) / "SOP_phase1"
        npz_path = default_root / "probe_init_data.npz"
        ply_path = default_root / "probe_offset_points.ply"
        if npz_path.exists():
            return npz_path
        return ply_path

    if path.is_dir():
        npz_path = path / "probe_init_data.npz"
        ply_path = path / "probe_offset_points.ply"
        if npz_path.exists():
            return npz_path
        if ply_path.exists():
            return ply_path
    return path


def _load_probe_data(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Probe file not found: {path}")

    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        if "probe_points" not in payload or "probe_normals" not in payload:
            raise KeyError(f"{path} does not contain probe_points/probe_normals")
        probe_xyz = np.asarray(payload["probe_points"], dtype=np.float32)
        probe_normal = np.asarray(payload["probe_normals"], dtype=np.float32)
        info = {
            "source": "npz",
            "path": str(path),
            "num_probes": int(probe_xyz.shape[0]),
        }
        return probe_xyz, probe_normal, info

    if path.suffix.lower() == ".ply":
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        required = ["x", "y", "z", "nx", "ny", "nz"]
        for name in required:
            if name not in vertex.dtype.names:
                raise KeyError(f"{path} is missing PLY field {name}")
        probe_xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        probe_normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
        info = {
            "source": "ply",
            "path": str(path),
            "num_probes": int(probe_xyz.shape[0]),
        }
        return probe_xyz, probe_normal, info

    raise ValueError(f"Unsupported probe file type: {path}")


@torch.no_grad()
def init_sop_textures_from_irgs_trace(
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    gaussians: GaussianModel,
    tex_h: int,
    tex_w: int,
    probe_chunk_size: int,
    trace_bias: float,
    probe_normal_bias: float,
    wo_indirect: bool,
    init_albedo_value: float,
    init_roughness_value: float,
    init_metallic_value: float,
) -> Dict[str, torch.Tensor]:
    device = probe_xyz.device
    dtype = probe_xyz.dtype
    probe_xyz = probe_xyz.reshape(-1, 3)
    probe_normal = torch.nn.functional.normalize(probe_normal.reshape(-1, 3), dim=-1, eps=1e-6)

    num_probes = int(probe_xyz.shape[0])
    dirs_grid = build_octahedral_direction_grid(
        tex_h=tex_h,
        tex_w=tex_w,
        device=device,
        dtype=dtype,
    )
    flat_dirs = dirs_grid.reshape(-1, 3)
    num_texels = flat_dirs.shape[0]

    probe_lin_tex = torch.zeros((num_probes, tex_h, tex_w, 3), device=device, dtype=dtype)
    probe_occ_tex = torch.zeros((num_probes, tex_h, tex_w, 1), device=device, dtype=dtype)
    probe_albedo_tex = torch.full((num_probes, tex_h, tex_w, 3), float(init_albedo_value), device=device, dtype=dtype)
    probe_roughness_tex = torch.full((num_probes, tex_h, tex_w, 1), float(init_roughness_value), device=device, dtype=dtype)
    probe_metallic_tex = torch.full((num_probes, tex_h, tex_w, 1), float(init_metallic_value), device=device, dtype=dtype)

    material_features = torch.cat([gaussians.get_base_color, gaussians.get_rough, gaussians.get_metallic], dim=1)
    chunk_size = max(1, int(probe_chunk_size))

    for start in range(0, num_probes, chunk_size):
        end = min(start + chunk_size, num_probes)
        xyz_chunk = probe_xyz[start:end]
        normal_chunk = probe_normal[start:end]
        ray_dirs = flat_dirs.unsqueeze(0).expand(end - start, num_texels, 3)
        ray_origins = xyz_chunk[:, None, :] + ray_dirs * float(trace_bias) + normal_chunk[:, None, :] * float(probe_normal_bias)

        trace_outputs = gaussians.trace(
            ray_origins,
            ray_dirs,
            features=material_features,
            camera_center=None,
        )
        trace_alpha = torch.clamp(trace_outputs["alpha"][..., None], 0.0, 1.0)
        local_radiance = torch.clamp_min(trace_outputs["color"], 0.0)
        if wo_indirect:
            local_radiance = torch.zeros_like(local_radiance)

        hit_feature = trace_outputs["feature"] / trace_alpha.clamp_min(1e-6)
        hit_albedo, hit_roughness, hit_metallic = hit_feature.split([3, 1, 1], dim=-1)
        hit_mask = trace_alpha > 1e-4

        probe_lin_tex[start:end] = local_radiance.reshape(end - start, tex_h, tex_w, 3)
        probe_occ_tex[start:end] = trace_alpha.reshape(end - start, tex_h, tex_w, 1)
        probe_albedo_tex[start:end] = torch.where(
            hit_mask.expand_as(hit_albedo),
            torch.clamp(hit_albedo, 0.0, 1.0),
            probe_albedo_tex[start:end].reshape(end - start, num_texels, 3),
        ).reshape(end - start, tex_h, tex_w, 3)
        probe_roughness_tex[start:end] = torch.where(
            hit_mask,
            torch.clamp(hit_roughness, 0.0, 1.0),
            probe_roughness_tex[start:end].reshape(end - start, num_texels, 1),
        ).reshape(end - start, tex_h, tex_w, 1)
        probe_metallic_tex[start:end] = torch.where(
            hit_mask,
            torch.clamp(hit_metallic, 0.0, 1.0),
            probe_metallic_tex[start:end].reshape(end - start, num_texels, 1),
        ).reshape(end - start, tex_h, tex_w, 1)

    return {
        "probe_lin_tex": torch.clamp_min(probe_lin_tex, 0.0),
        "probe_occ_tex": torch.clamp(probe_occ_tex, 0.0, 1.0),
        "probe_albedo_tex": torch.clamp(probe_albedo_tex, 0.0, 1.0),
        "probe_roughness_tex": torch.clamp(probe_roughness_tex, 0.0, 1.0),
        "probe_metallic_tex": torch.clamp(probe_metallic_tex, 0.0, 1.0),
        "oct_dirs": dirs_grid,
    }


def _save_probe_texture_previews(
    output_root: Path,
    textures: Dict[str, torch.Tensor],
    preview_count: int,
) -> None:
    preview_dir = _ensure_dir(output_root / "previews")
    count = min(int(preview_count), int(textures["probe_lin_tex"].shape[0]))
    if count <= 0:
        return

    lin = _tonemap_for_vis(textures["probe_lin_tex"][:count]).permute(0, 3, 1, 2)
    occ = _repeat_single_channel(textures["probe_occ_tex"][:count]).permute(0, 3, 1, 2)
    albedo = textures["probe_albedo_tex"][:count].permute(0, 3, 1, 2)
    roughness = _repeat_single_channel(textures["probe_roughness_tex"][:count]).permute(0, 3, 1, 2)
    metallic = _repeat_single_channel(textures["probe_metallic_tex"][:count]).permute(0, 3, 1, 2)

    torchvision.utils.save_image(lin, str(preview_dir / "probe_lin_tex.png"), nrow=min(4, count))
    torchvision.utils.save_image(occ, str(preview_dir / "probe_occ_tex.png"), nrow=min(4, count))
    torchvision.utils.save_image(albedo, str(preview_dir / "probe_albedo_tex.png"), nrow=min(4, count))
    torchvision.utils.save_image(roughness, str(preview_dir / "probe_roughness_tex.png"), nrow=min(4, count))
    torchvision.utils.save_image(metallic, str(preview_dir / "probe_metallic_tex.png"), nrow=min(4, count))


def _save_outputs(
    output_root: Path,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    textures: Dict[str, torch.Tensor],
    summary: Dict[str, object],
) -> None:
    payload = {
        "format": "sop_query_init_v1",
        "probe_xyz": probe_xyz.detach().cpu(),
        "probe_normal": probe_normal.detach().cpu(),
        "probe_lin_tex": textures["probe_lin_tex"].detach().cpu(),
        "probe_occ_tex": textures["probe_occ_tex"].detach().cpu(),
        "probe_albedo_tex": textures["probe_albedo_tex"].detach().cpu(),
        "probe_roughness_tex": textures["probe_roughness_tex"].detach().cpu(),
        "probe_metallic_tex": textures["probe_metallic_tex"].detach().cpu(),
        "oct_dirs": textures["oct_dirs"].detach().cpu(),
        "summary": summary,
    }
    torch.save(payload, output_root / "sop_query_init.pt")
    np.savez(
        output_root / "sop_query_init.npz",
        probe_xyz=probe_xyz.detach().cpu().numpy().astype(np.float32),
        probe_normal=probe_normal.detach().cpu().numpy().astype(np.float32),
        probe_lin_tex=textures["probe_lin_tex"].detach().cpu().numpy().astype(np.float32),
        probe_occ_tex=textures["probe_occ_tex"].detach().cpu().numpy().astype(np.float32),
        probe_albedo_tex=textures["probe_albedo_tex"].detach().cpu().numpy().astype(np.float32),
        probe_roughness_tex=textures["probe_roughness_tex"].detach().cpu().numpy().astype(np.float32),
        probe_metallic_tex=textures["probe_metallic_tex"].detach().cpu().numpy().astype(np.float32),
        oct_dirs=textures["oct_dirs"].detach().cpu().numpy().astype(np.float32),
    )
    with open(output_root / "sop_query_init_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


@torch.no_grad()
def initialize_sop_query_textures(args):
    if not torch.cuda.is_available():
        raise RuntimeError("SOP query initialization currently requires CUDA because it uses COMGS_IRGS Gaussian tracing.")

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)
    _ensure_dir(Path(dataset.model_path))

    _, gaussians, loaded_from = _prepare_scene_and_gaussians(args, dataset, pipe, opt)

    probe_file = _resolve_probe_file(args.probe_file, dataset.model_path)
    probe_xyz_np, probe_normal_np, probe_info = _load_probe_data(probe_file)
    probe_xyz = torch.from_numpy(probe_xyz_np).cuda()
    probe_normal = torch.from_numpy(probe_normal_np).cuda()

    output_root = _ensure_dir(Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "SOP_query_init")
    trace_bias = float(args.trace_bias) if args.trace_bias >= 0.0 else float(pipe.light_t_min)
    init_albedo_value = float(args.init_albedo_value) if args.init_albedo_value >= 0.0 else float(opt.init_base_color_value)
    textures = init_sop_textures_from_irgs_trace(
        probe_xyz=probe_xyz,
        probe_normal=probe_normal,
        gaussians=gaussians,
        tex_h=args.tex_h,
        tex_w=args.tex_w,
        probe_chunk_size=args.probe_chunk_size,
        trace_bias=trace_bias,
        probe_normal_bias=args.probe_normal_bias,
        wo_indirect=bool(pipe.wo_indirect),
        init_albedo_value=init_albedo_value,
        init_roughness_value=opt.init_roughness_value,
        init_metallic_value=opt.init_metallic_value,
    )

    num_probes = probe_xyz.shape[0]
    summary = {
        "format": "irgs_sop_query_init_trace_v1",
        "loaded_from": loaded_from,
        "probe_info": probe_info,
        "trace_backend": "comgs_irgs_gaussian_trace",
        "probe_lin_tex_semantics": "local_incident_lights_trace_color",
        "probe_occ_tex_semantics": "trace_alpha_occlusion",
        "train_equation_alignment": "rendering_equation/rendering_equation_sop use incident=(1-alpha)*env+local; env is not baked into probe_lin_tex",
        "trace_bias": float(trace_bias),
        "trace_bias_source": "args.trace_bias" if args.trace_bias >= 0.0 else "pipe.light_t_min",
        "probe_normal_bias": float(args.probe_normal_bias),
        "wo_indirect": bool(pipe.wo_indirect),
        "init_albedo_value": float(init_albedo_value),
        "init_roughness_value": float(opt.init_roughness_value),
        "init_metallic_value": float(opt.init_metallic_value),
        "native_alpha_min": float(args.native_alpha_min),
        "native_transmittance_min": float(args.native_transmittance_min),
        "tex_h": int(args.tex_h),
        "tex_w": int(args.tex_w),
        "probe_lin_tex_mean": float(textures["probe_lin_tex"].mean().item()),
        "probe_lin_tex_max": float(textures["probe_lin_tex"].max().item()),
        "probe_occ_tex_mean": float(textures["probe_occ_tex"].mean().item()),
        "probe_occ_tex_max": float(textures["probe_occ_tex"].max().item()),
        "probe_albedo_tex_mean": float(textures["probe_albedo_tex"].mean().item()),
        "probe_roughness_tex_mean": float(textures["probe_roughness_tex"].mean().item()),
        "probe_metallic_tex_mean": float(textures["probe_metallic_tex"].mean().item()),
        "args": dict(vars(args)),
    }

    _save_probe_texture_previews(output_root, textures, preview_count=args.preview_count)
    _save_outputs(output_root, probe_xyz, probe_normal, textures, summary)

    print(f"[SOP-QueryInit] Probes: {num_probes}")
    print("[SOP-QueryInit] Trace backend: COMGS_IRGS GaussianModel.trace")
    print(f"[SOP-QueryInit] Probe source: {probe_file}")
    print(f"[SOP-QueryInit] Output root: {output_root}")
    return summary


def _build_parser():
    parser = ArgumentParser(description="Initialize SOP query textures from COMGS_IRGS Gaussian tracing")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--probe_file", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--tex_h", default=16, type=int)
    parser.add_argument("--tex_w", default=16, type=int)
    parser.add_argument("--probe_chunk_size", default=128, type=int)
    parser.add_argument("--trace_bias", default=-1.0, type=float)
    parser.add_argument("--probe_normal_bias", default=5e-4, type=float)
    parser.add_argument("--native_alpha_min", default=1.0 / 255.0, type=float)
    parser.add_argument("--native_transmittance_min", default=0.03, type=float)
    parser.add_argument("--preview_count", default=16, type=int)
    parser.add_argument("--init_lin_value", default=0.0, type=float)
    parser.add_argument("--init_occ_value", default=0.0, type=float)
    parser.add_argument("--init_albedo_value", default=-1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = get_combined_args(parser)
    if not getattr(args, "model_path", ""):
        raise RuntimeError("SOP query initialization requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("SOP query initialization requires --source_path/-s, or a cfg_args under --model_path.")
    _ensure_dir(Path(args.model_path))
    with open(Path(args.model_path) / "cfg_args", "w") as cfg_log_f:
        cfg_log_f.write(str(args))
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialize_sop_query_textures(args)


if __name__ == "__main__":
    main()
