from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torchvision
from plyfile import PlyData

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from utils.deferred_pbr_comgs import OctahedralEnvMap, load_envmap_capture_as_octahedral
from utils.general_utils import safe_state
from utils.sop_utils import build_octahedral_direction_grid, query_sops
from utils.tracing_comgs import TraceBackendConfig, build_trace_backend


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


def _load_gaussians_from_checkpoint(gaussians: GaussianModel, opt, checkpoint_path: str) -> Tuple[int, str]:
    payload = torch.load(checkpoint_path)
    if isinstance(payload, dict) and "gaussians" in payload:
        model_params = payload["gaussians"]
        iteration = int(payload.get("iteration", 0))
        label = payload.get("format", "checkpoint_dict")
    elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
        model_params = payload[0]
        iteration = int(payload[1])
        label = "stage1_checkpoint"
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    gaussians.training_setup(opt)
    gaussians.restore(model_params, opt)
    return iteration, label


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


def _load_envmap(args) -> Tuple[OctahedralEnvMap, Dict[str, object]]:
    if args.stage2_trace_ckpt:
        payload = torch.load(args.stage2_trace_ckpt)
        if not isinstance(payload, dict) or payload.get("format") != "comgs_stage2_trace_v1" or "envmap" not in payload:
            raise RuntimeError(f"{args.stage2_trace_ckpt} is not a valid Stage2 Trace checkpoint with envmap")
        envmap = load_envmap_capture_as_octahedral(
            payload["envmap"],
            height=args.envmap_height,
            width=args.envmap_width,
        ).cuda()
        return envmap, {
            "source": "stage2_trace_ckpt",
            "path": args.stage2_trace_ckpt,
            "iteration": int(payload.get("iteration", -1)),
            "projection": "octahedral",
            "height": int(envmap.height),
            "width": int(envmap.width),
        }

    if args.envmap_capture:
        capture = torch.load(args.envmap_capture)
        envmap = load_envmap_capture_as_octahedral(
            capture,
            height=args.envmap_height,
            width=args.envmap_width,
        ).cuda()
        return envmap, {
            "source": "envmap_capture",
            "path": args.envmap_capture,
            "projection": "octahedral",
            "height": int(envmap.height),
            "width": int(envmap.width),
        }

    envmap = OctahedralEnvMap(
        height=args.envmap_height,
        width=args.envmap_width,
        init_value=args.envmap_init_value,
        activation=args.envmap_activation,
    ).cuda()
    return envmap, {
        "source": "default_constant",
        "projection": "octahedral",
        "height": int(args.envmap_height),
        "width": int(args.envmap_width),
        "init_value": float(args.envmap_init_value),
        "activation": str(args.envmap_activation),
    }


@torch.no_grad()
def init_sop_textures(
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    tracer,
    envmap: OctahedralEnvMap,
    tex_h: int = 16,
    tex_w: int = 16,
    probe_chunk_size: int = 128,
    trace_bias: float = 1e-3,
    probe_normal_bias: float = 5e-4,
    secondary_num_samples: int = 16,
    randomized_secondary: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Initialize per-probe octahedral textures with traced incident radiance,
    occlusion, and hit-point material attributes.
    """
    device = probe_xyz.device
    dtype = probe_xyz.dtype
    probe_xyz = probe_xyz.reshape(-1, 3)
    probe_normal = torch.nn.functional.normalize(probe_normal.reshape(-1, 3), dim=-1, eps=1e-6)

    num_probes = int(probe_xyz.shape[0])
    tex_h = int(tex_h)
    tex_w = int(tex_w)
    dirs_grid = build_octahedral_direction_grid(tex_h, tex_w, device=device, dtype=dtype)
    flat_dirs = dirs_grid.view(-1, 3)
    num_texels = flat_dirs.shape[0]

    probe_lin_tex = torch.zeros((num_probes, tex_h, tex_w, 3), device=device, dtype=dtype)
    probe_occ_tex = torch.zeros((num_probes, tex_h, tex_w, 1), device=device, dtype=dtype)
    probe_albedo_tex = torch.zeros((num_probes, tex_h, tex_w, 3), device=device, dtype=dtype)
    probe_roughness_tex = torch.zeros((num_probes, tex_h, tex_w, 1), device=device, dtype=dtype)
    probe_metallic_tex = torch.zeros((num_probes, tex_h, tex_w, 1), device=device, dtype=dtype)

    for start in range(0, num_probes, max(1, int(probe_chunk_size))):
        end = min(start + max(1, int(probe_chunk_size)), num_probes)
        xyz_chunk = probe_xyz[start:end]
        normal_chunk = probe_normal[start:end]
        ray_dirs = flat_dirs.unsqueeze(0).expand(end - start, num_texels, 3)
        ray_origins = xyz_chunk[:, None, :] + ray_dirs * float(trace_bias) + normal_chunk[:, None, :] * float(probe_normal_bias)

        trace_outputs = tracer.trace(
            ray_origins=ray_origins,
            ray_directions=ray_dirs,
            envmap=envmap,
            secondary_num_samples=secondary_num_samples,
            randomized_secondary=randomized_secondary,
            camera_center=None,
        )

        probe_lin_tex[start:end] = trace_outputs["incident_radiance"].reshape(end - start, tex_h, tex_w, 3)
        probe_occ_tex[start:end] = trace_outputs["occlusion"].reshape(end - start, tex_h, tex_w, 1)
        probe_albedo_tex[start:end] = trace_outputs["hit_albedo"].reshape(end - start, tex_h, tex_w, 3)
        probe_roughness_tex[start:end] = trace_outputs["hit_roughness"].reshape(end - start, tex_h, tex_w, 1)
        probe_metallic_tex[start:end] = trace_outputs["hit_metallic"].reshape(end - start, tex_h, tex_w, 1)

    return {
        "probe_lin_tex": probe_lin_tex,
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


def _run_query_smoke_test(
    output_root: Path,
    probe_xyz: torch.Tensor,
    probe_normal: torch.Tensor,
    textures: Dict[str, torch.Tensor],
    args,
) -> Dict[str, object]:
    if not args.run_query_smoke_test:
        return {"enabled": False}

    offset = float(args.query_smoke_test_offset)
    if offset <= 0.0:
        offset = max(float(args.trace_bias) * 4.0, float(args.probe_normal_bias) * 2.0, 1e-4)

    x_world = probe_xyz - probe_normal * offset
    lin_x, occ_x = query_sops(
        x_world=x_world,
        probe_xyz=probe_xyz,
        probe_normal=probe_normal,
        probe_lin_tex=textures["probe_lin_tex"],
        probe_occ_tex=textures["probe_occ_tex"],
        radius=(float(args.query_radius) if args.query_radius > 0.0 else None),
        topk=args.query_topk,
        chunk_size=args.query_chunk_size,
    )
    np.savez(
        output_root / "query_smoke_test.npz",
        x_world=x_world.detach().cpu().numpy().astype(np.float32),
        lin_x=lin_x.detach().cpu().numpy().astype(np.float32),
        occ_x=occ_x.detach().cpu().numpy().astype(np.float32),
    )
    return {
        "enabled": True,
        "num_points": int(x_world.shape[0]),
        "offset": float(offset),
        "lin_mean": float(lin_x.mean().item()) if lin_x.numel() > 0 else 0.0,
        "occ_mean": float(occ_x.mean().item()) if occ_x.numel() > 0 else 0.0,
    }


@torch.no_grad()
def initialize_sop_query_textures(args) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("SOP query initialization currently requires CUDA because it reuses the tracing backend.")

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    checkpoint_path = getattr(args, "checkpoint", "")
    if checkpoint_path:
        scene = Scene(dataset, gaussians, shuffle=False)
        ckpt_iteration, ckpt_label = _load_gaussians_from_checkpoint(gaussians, opt, checkpoint_path)
        loaded_from = {
            "source": "checkpoint",
            "path": checkpoint_path,
            "iteration": ckpt_iteration,
            "format": ckpt_label,
        }
    else:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        loaded_from = {
            "source": "point_cloud",
            "iteration": int(scene.loaded_iter),
        }

    probe_file = _resolve_probe_file(args.probe_file, dataset.model_path)
    probe_xyz_np, probe_normal_np, probe_info = _load_probe_data(probe_file)
    probe_xyz = torch.from_numpy(probe_xyz_np).cuda()
    probe_normal = torch.from_numpy(probe_normal_np).cuda()

    envmap, envmap_info = _load_envmap(args)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    trace_config = TraceBackendConfig(
        backend=args.trace_backend,
        trace_bias=args.trace_bias,
        secondary_num_samples=args.secondary_num_samples,
        rebuild_every=0,
        open3d_voxel_size=args.trace_voxel_size,
        open3d_sdf_trunc=args.trace_sdf_trunc,
        open3d_depth_trunc=args.trace_depth_trunc,
        open3d_mask_background=args.trace_mask_background,
        native_alpha_min=args.native_alpha_min,
        native_transmittance_min=args.native_transmittance_min,
    )
    tracer = build_trace_backend(trace_config, scene, gaussians, pipe, background)

    output_dir = Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "SOP_query_init"
    output_root = _ensure_dir(output_dir)
    textures = init_sop_textures(
        probe_xyz=probe_xyz,
        probe_normal=probe_normal,
        tracer=tracer,
        envmap=envmap,
        tex_h=args.tex_h,
        tex_w=args.tex_w,
        probe_chunk_size=args.probe_chunk_size,
        trace_bias=args.trace_bias,
        probe_normal_bias=args.probe_normal_bias,
        secondary_num_samples=args.secondary_num_samples,
        randomized_secondary=not args.disable_sample_jitter,
    )
    smoke_stats = _run_query_smoke_test(
        output_root=output_root,
        probe_xyz=probe_xyz,
        probe_normal=probe_normal,
        textures=textures,
        args=args,
    )

    summary = {
        "loaded_from": loaded_from,
        "probe_info": probe_info,
        "envmap": envmap_info,
        "trace_backend": tracer.backend_name,
        "num_probes": int(probe_xyz.shape[0]),
        "tex_h": int(args.tex_h),
        "tex_w": int(args.tex_w),
        "probe_lin_tex_mean": float(textures["probe_lin_tex"].mean().item()),
        "probe_lin_tex_max": float(textures["probe_lin_tex"].max().item()),
        "probe_occ_tex_mean": float(textures["probe_occ_tex"].mean().item()),
        "probe_occ_tex_max": float(textures["probe_occ_tex"].max().item()),
        "probe_albedo_tex_mean": float(textures["probe_albedo_tex"].mean().item()),
        "probe_roughness_tex_mean": float(textures["probe_roughness_tex"].mean().item()),
        "probe_metallic_tex_mean": float(textures["probe_metallic_tex"].mean().item()),
        "smoke_test": smoke_stats,
        "args": dict(vars(args)),
    }

    _save_probe_texture_previews(output_root, textures, preview_count=args.preview_count)
    _save_outputs(output_root, probe_xyz, probe_normal, textures, summary)

    print(f"[SOP-QueryInit] Probes: {probe_xyz.shape[0]}")
    print(f"[SOP-QueryInit] Trace backend: {tracer.backend_name}")
    print(f"[SOP-QueryInit] Probe source: {probe_file}")
    print(f"[SOP-QueryInit] Output root: {output_root}")
    return summary


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Initialize SOP octahedral query textures with trace backend")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--probe_file", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--tex_h", default=16, type=int)
    parser.add_argument("--tex_w", default=16, type=int)
    parser.add_argument("--probe_chunk_size", default=128, type=int)
    parser.add_argument("--trace_backend", type=str, default="auto", choices=["auto", "irgs", "irgs_adapter", "irgs_native", "open3d", "open3d_mesh"])
    parser.add_argument("--trace_bias", type=float, default=1e-3)
    parser.add_argument("--probe_normal_bias", type=float, default=5e-4)
    parser.add_argument("--secondary_num_samples", type=int, default=16)
    parser.add_argument("--disable_sample_jitter", dest="disable_sample_jitter", action="store_true")
    parser.add_argument("--enable_sample_jitter", dest="disable_sample_jitter", action="store_false")
    parser.add_argument("--trace_voxel_size", type=float, default=0.004)
    parser.add_argument("--trace_sdf_trunc", type=float, default=0.02)
    parser.add_argument("--trace_depth_trunc", type=float, default=0.0)
    parser.add_argument("--trace_mask_background", dest="trace_mask_background", action="store_true")
    parser.add_argument("--no_trace_mask_background", dest="trace_mask_background", action="store_false")
    parser.add_argument("--native_alpha_min", type=float, default=1.0 / 255.0)
    parser.add_argument("--native_transmittance_min", type=float, default=0.03)
    parser.add_argument("--stage2_trace_ckpt", default="", type=str)
    parser.add_argument("--envmap_capture", default="", type=str)
    parser.add_argument("--envmap_height", default=256, type=int)
    parser.add_argument("--envmap_width", default=256, type=int)
    parser.add_argument("--envmap_init_value", default=0.5, type=float)
    parser.add_argument("--envmap_activation", default="exp", choices=["exp", "softplus", "none"])
    parser.add_argument("--preview_count", default=16, type=int)
    parser.add_argument("--run_query_smoke_test", action="store_true")
    parser.add_argument("--query_smoke_test_offset", default=0.0, type=float)
    parser.add_argument("--query_topk", default=8, type=int)
    parser.add_argument("--query_radius", default=0.0, type=float)
    parser.add_argument("--query_chunk_size", default=4096, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(disable_sample_jitter=True, trace_mask_background=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = get_combined_args(parser)
    safe_state(args.quiet)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initialize_sop_query_textures(args)


if __name__ == "__main__":
    main()
