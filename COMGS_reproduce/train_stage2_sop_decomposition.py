import json
import os
import sys
from pathlib import Path
from random import randint
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_multitarget
from scene import Scene, GaussianModel
from train_stage1_comgs import prepare_output_and_logger
from utils.deferred_pbr_comgs import (
    DEFAULT_OCT_ENV_HEIGHT,
    DEFAULT_OCT_ENV_WIDTH,
    OctahedralEnvMap,
    load_envmap_capture_as_octahedral,
    recover_shading_points,
)
from utils.general_utils import safe_state
from utils.losses_comgs_stage2_sop import compute_stage2_sop_loss
from utils.sop_utils import build_view_sop_neighbor_cache
from utils.tracing_comgs import TraceBackendConfig, build_trace_backend


def _normalize_single_channel_for_vis(x, valid_mask=None, eps=1e-6):
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        mask = valid_mask.bool()
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    if int(mask.sum().item()) > 0:
        values = y[mask]
        vmin = values.min()
        vmax = values.max()
    else:
        vmin = y.min()
        vmax = y.max()

    denom = torch.clamp(vmax - vmin, min=eps)
    vis = (y - vmin) / denom
    if valid_mask is not None:
        vis = vis * mask.float()
    return torch.clamp(vis, 0.0, 1.0)


def _tonemap_for_vis(x):
    y = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(y / (1.0 + y), 0.0, 1.0)


def _set_requires_grad(group, value: bool):
    for param in group["params"]:
        param.requires_grad_(value)


def configure_stage2_gaussian_optimizer(gaussians: GaussianModel, args):
    geometry_names = {"xyz", "scaling", "rotation", "opacity"}
    color_names = {"f_dc", "f_rest"}
    material_names = {"albedo", "roughness", "metallic"}

    feature_lr = getattr(args, "feature_lr", 0.0025)
    material_lr = getattr(args, "material_lr", None)
    albedo_lr = getattr(args, "albedo_lr", material_lr if material_lr is not None else feature_lr)
    roughness_lr = getattr(args, "roughness_lr", material_lr if material_lr is not None else feature_lr)
    metallic_lr = getattr(args, "metallic_lr", material_lr if material_lr is not None else feature_lr)
    default_group_lrs = {
        "f_dc": feature_lr,
        "f_rest": feature_lr / 20.0,
        "opacity": args.opacity_lr,
        "scaling": args.scaling_lr,
        "rotation": args.rotation_lr,
        "albedo": albedo_lr,
        "roughness": roughness_lr,
        "metallic": metallic_lr,
    }

    for group in gaussians.optimizer.param_groups:
        name = group["name"]
        if name in geometry_names:
            _set_requires_grad(group, not args.freeze_geometry)
            if args.freeze_geometry:
                group["lr"] = 0.0
            elif name != "xyz":
                group["lr"] = default_group_lrs[name]
        elif name in color_names:
            _set_requires_grad(group, not args.freeze_color)
            if args.freeze_color:
                group["lr"] = 0.0
            else:
                group["lr"] = default_group_lrs[name]
        elif name in material_names:
            _set_requires_grad(group, True)
            group["lr"] = default_group_lrs[name]


class Stage2SOPState(nn.Module):
    def __init__(self, payload: Dict[str, object]):
        super().__init__()
        required = ["probe_xyz", "probe_normal", "probe_lin_tex", "probe_occ_tex"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"SOP payload is missing required fields: {missing}")

        probe_xyz = payload["probe_xyz"].float().cuda()
        probe_normal = F.normalize(payload["probe_normal"].float().cuda(), dim=-1, eps=1e-6)
        probe_lin_tex = payload["probe_lin_tex"].float().cuda()
        probe_occ_tex = payload["probe_occ_tex"].float().cuda()

        self.register_buffer("probe_xyz", probe_xyz)
        self.register_buffer("probe_normal", probe_normal)
        self.probe_lin_tex = nn.Parameter(probe_lin_tex.requires_grad_(True))
        self.probe_occ_tex = nn.Parameter(probe_occ_tex.requires_grad_(True))

        optional_tensor_keys = [
            "probe_albedo_tex",
            "probe_roughness_tex",
            "probe_metallic_tex",
            "oct_dirs",
        ]
        self._optional_tensor_keys = []
        for key in optional_tensor_keys:
            if key in payload and torch.is_tensor(payload[key]):
                self.register_buffer(key, payload[key].float().cuda())
                self._optional_tensor_keys.append(key)

        self.source_format = str(payload.get("format", "unknown"))
        self.source_summary = payload.get("summary", None)

    @property
    def lin_tex(self) -> torch.Tensor:
        return torch.clamp(self.probe_lin_tex, min=0.0)

    @property
    def occ_tex(self) -> torch.Tensor:
        return torch.clamp(self.probe_occ_tex, 0.0, 1.0)

    @property
    def num_probes(self) -> int:
        return int(self.probe_xyz.shape[0])

    def set_texture_training(self, enabled: bool):
        self.probe_lin_tex.requires_grad_(enabled)
        self.probe_occ_tex.requires_grad_(enabled)

    def capture(self) -> Dict[str, object]:
        payload = {
            "format": self.source_format,
            "probe_xyz": self.probe_xyz.detach().cpu(),
            "probe_normal": self.probe_normal.detach().cpu(),
            "probe_lin_tex": self.lin_tex.detach().cpu(),
            "probe_occ_tex": self.occ_tex.detach().cpu(),
        }
        for key in self._optional_tensor_keys:
            payload[key] = getattr(self, key).detach().cpu()
        if self.source_summary is not None:
            payload["summary"] = self.source_summary
        return payload


def _resolve_sop_init_path(path_str: str, model_path: str) -> Path:
    if path_str:
        path = Path(path_str)
    else:
        path = Path(model_path) / "SOP_query_init" / "sop_query_init.pt"

    if path.is_dir():
        candidate = path / "sop_query_init.pt"
        if candidate.exists():
            return candidate
    return path


def _load_sop_payload(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"SOP init file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"SOP init payload must be a dict: {path}")
    return payload


def _cuda_checkpoint_map_location():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda")


def _clone_tensor(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _clone_trainable_parameter(x: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(x.detach().clone().requires_grad_(True))


def _filled_trainable_parameter_like(x: torch.Tensor, value: float) -> nn.Parameter:
    return _clone_trainable_parameter(torch.full_like(x, float(value)))


def _encode_visible_material_like(gaussians: GaussianModel, template: torch.Tensor, value: float, material_name: str) -> nn.Parameter:
    visible = torch.full_like(template, float(value))
    if material_name == "albedo":
        raw = gaussians._encode_visible_albedo(visible)
    elif material_name == "roughness":
        raw = gaussians._encode_visible_roughness(visible)
    elif material_name == "metallic":
        raw = gaussians._encode_visible_metallic(visible)
    else:
        raise ValueError(f"Unsupported material name: {material_name}")
    return _clone_trainable_parameter(raw)


def _describe_model_args(model_args) -> str:
    if isinstance(model_args, (tuple, list)):
        return f"{type(model_args).__name__}(len={len(model_args)})"
    return type(model_args).__name__


def _convert_irgs_refgs_model_args(model_args, fallback_optimizer_state, gaussians: GaussianModel):
    # Discard RefGS material tensors and reinitialize stage2 materials.
    # Keep the material hyperparameters in a similar range instead of pushing
    # metallic all the way to zero.
    irgs_init_albedo = 0.3
    irgs_init_roughness = 0.7
    irgs_init_metallic = 0.2

    if not isinstance(model_args, (tuple, list)) or len(model_args) != 19:
        raise RuntimeError(
            "IRGS refgs checkpoint should store model params as a tuple/list with len=19, "
            f"but got {_describe_model_args(model_args)}."
        )

    (
        active_sh_degree,
        xyz,
        metallic_raw,
        roughness_raw,
        base_color_raw,
        features_dc,
        features_rest,
        _indirect_dc,
        _indirect_rest,
        scaling,
        rotation,
        opacity,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        _opt_dict,
        _env_map_1,
        _env_map_2,
        spatial_lr_scale,
    ) = model_args

    return (
        int(active_sh_degree),
        _clone_trainable_parameter(xyz),
        _clone_trainable_parameter(features_dc),
        _clone_trainable_parameter(features_rest),
        _clone_trainable_parameter(scaling),
        _clone_trainable_parameter(rotation),
        _clone_trainable_parameter(opacity),
        _encode_visible_material_like(gaussians, base_color_raw, irgs_init_albedo, "albedo"),
        _encode_visible_material_like(gaussians, roughness_raw, irgs_init_roughness, "roughness"),
        _encode_visible_material_like(gaussians, metallic_raw, irgs_init_metallic, "metallic"),
        _clone_tensor(max_radii2D),
        _clone_tensor(xyz_gradient_accum),
        _clone_tensor(denom),
        fallback_optimizer_state,
        float(spatial_lr_scale),
    )


def _resolve_stage1_model_args(gaussians: GaussianModel, checkpoint_path: str, stage1_ckpt_format: str):
    payload = torch.load(checkpoint_path, map_location=_cuda_checkpoint_map_location())
    if isinstance(payload, dict) and "gaussians" in payload:
        return payload["gaussians"], int(payload.get("iteration", 0)), payload.get("format", "checkpoint_dict")

    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload type for {checkpoint_path}: {type(payload).__name__}")

    model_args = payload[0]
    iteration = int(payload[1]) if len(payload) >= 2 else 0

    if stage1_ckpt_format == "comgs":
        if not isinstance(model_args, (tuple, list)) or len(model_args) not in {12, 15}:
            raise RuntimeError(
                "Expected a COMGS checkpoint with model params len=12 or len=15, "
                f"but got {_describe_model_args(model_args)} from {checkpoint_path}."
            )
        return model_args, iteration, "comgs_stage1"

    if stage1_ckpt_format == "irgs_refgs":
        return (
            _convert_irgs_refgs_model_args(model_args, gaussians.optimizer.state_dict(), gaussians),
            iteration,
            "irgs_refgs_compat",
        )

    if isinstance(model_args, (tuple, list)) and len(model_args) in {12, 15}:
        return model_args, iteration, "comgs_stage1"

    if isinstance(model_args, (tuple, list)) and len(model_args) == 19:
        return (
            _convert_irgs_refgs_model_args(model_args, gaussians.optimizer.state_dict(), gaussians),
            iteration,
            "irgs_refgs_compat",
        )

    raise RuntimeError(
        "Unable to auto-detect stage1 checkpoint format for "
        f"{checkpoint_path}: got {_describe_model_args(model_args)}. "
        "Try --stage1_ckpt_format comgs or --stage1_ckpt_format irgs_refgs."
    )


def _load_gaussians_from_checkpoint(
    gaussians: GaussianModel,
    opt,
    checkpoint_path: str,
    stage1_ckpt_format: str = "auto",
) -> Tuple[int, str]:
    model_params, iteration, label = _resolve_stage1_model_args(gaussians, checkpoint_path, stage1_ckpt_format)
    gaussians.restore(model_params, opt)
    return iteration, label


def _load_envmap_from_args(args) -> Tuple[OctahedralEnvMap, Dict[str, object]]:
    if args.stage2_trace_ckpt:
        payload = torch.load(args.stage2_trace_ckpt, map_location="cpu")
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
        capture = torch.load(args.envmap_capture, map_location="cpu")
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


def load_initial_stage2_sop_state(
    gaussians: GaussianModel,
    opt,
    stage1_ckpt: str,
    model_path: str,
    sop_init_path: str,
    start_checkpoint: Optional[str] = None,
    stage1_ckpt_format: str = "auto",
    envmap_height: int = DEFAULT_OCT_ENV_HEIGHT,
    envmap_width: int = DEFAULT_OCT_ENV_WIDTH,
):
    if start_checkpoint:
        payload = torch.load(start_checkpoint, map_location=_cuda_checkpoint_map_location())
        if not isinstance(payload, dict) or payload.get("format") != "comgs_stage2_sop_v1":
            raise RuntimeError(f"{start_checkpoint} is not a valid Stage2 SOP checkpoint")

        gaussians.restore(payload["gaussians"], opt)
        envmap = load_envmap_capture_as_octahedral(
            payload["envmap"],
            height=envmap_height,
            width=envmap_width,
        ).cuda()
        sop_state = Stage2SOPState(payload["sop"])
        info = {
            "loaded_from": {
                "source": "stage2_sop_checkpoint",
                "path": start_checkpoint,
                "iteration": int(payload.get("iteration", 0)),
            },
            "sop_source": {
                "source": "stage2_sop_checkpoint",
                "path": start_checkpoint,
            },
            "envmap_source": {
                "source": "stage2_sop_checkpoint",
                "path": start_checkpoint,
            },
        }
        return (
            int(payload.get("iteration", 0)),
            envmap,
            payload.get("env_optimizer"),
            sop_state,
            payload.get("sop_optimizer"),
            info,
        )

    ckpt_iteration, ckpt_label = _load_gaussians_from_checkpoint(
        gaussians,
        opt,
        stage1_ckpt,
        stage1_ckpt_format=stage1_ckpt_format,
    )
    sop_path = _resolve_sop_init_path(sop_init_path, model_path=model_path)
    sop_payload = _load_sop_payload(sop_path)
    sop_state = Stage2SOPState(sop_payload)
    info = {
        "loaded_from": {
            "source": "checkpoint",
            "path": stage1_ckpt,
            "iteration": int(ckpt_iteration),
            "format": ckpt_label,
        },
        "sop_source": {
            "source": "sop_query_init",
            "path": str(sop_path),
            "format": str(sop_payload.get("format", "unknown")),
        },
        "envmap_source": None,
    }
    return 0, None, None, sop_state, None, info


def save_stage2_sop_checkpoint(path, gaussians, envmap, env_optimizer, sop_state, sop_optimizer, iteration, args, stage1_ckpt, sop_init_path, trace_backend_name):
    payload = {
        "format": "comgs_stage2_sop_v1",
        "iteration": int(iteration),
        "stage1_checkpoint": stage1_ckpt,
        "stage1_checkpoint_format": getattr(args, "stage1_ckpt_format", "auto"),
        "sop_init_path": sop_init_path,
        "trace_backend": trace_backend_name,
        "args": dict(vars(args)),
        "gaussians": gaussians.capture(),
        "envmap": envmap.capture(),
        "env_optimizer": env_optimizer.state_dict() if env_optimizer is not None else None,
        "sop": sop_state.capture(),
        "sop_optimizer": sop_optimizer.state_dict() if sop_optimizer is not None else None,
    }
    torch.save(payload, path)


def save_envmap_artifacts(envmap, output_root: str, stem: str):
    os.makedirs(output_root, exist_ok=True)
    torchvision.utils.save_image(envmap.visualization(), os.path.join(output_root, f"{stem}.png"))
    torch.save(envmap.capture(), os.path.join(output_root, f"{stem}.pt"))


def save_sop_artifacts(sop_state: Stage2SOPState, output_root: str, stem: str):
    os.makedirs(output_root, exist_ok=True)
    torch.save(sop_state.capture(), os.path.join(output_root, f"{stem}.pt"))


def save_stage2_sop_debug(output_root, iteration, view_name, gt_rgb, render_pkg, aux, envmap):
    iter_dir = os.path.join(output_root, f"iter_{iteration:06d}")
    os.makedirs(iter_dir, exist_ok=True)

    valid = render_pkg["weight"] > 1e-4
    depth_vis = _normalize_single_channel_for_vis(render_pkg["depth_unbiased"], valid_mask=valid)
    depth_normal_vis = torch.clamp(aux["depth_normal"].detach() * 0.5 + 0.5, 0.0, 1.0)

    torchvision.utils.save_image(torch.clamp(aux["pbr_render"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_pbr.png"))
    torchvision.utils.save_image(torch.clamp(gt_rgb, 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_gt.png"))
    torchvision.utils.save_image(torch.clamp(render_pkg["albedo"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_albedo.png"))
    torchvision.utils.save_image(torch.clamp(render_pkg["roughness"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_roughness.png"))
    torchvision.utils.save_image(torch.clamp(render_pkg["metallic"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_metallic.png"))
    torchvision.utils.save_image(torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_normal.png"))
    torchvision.utils.save_image(depth_vis, os.path.join(iter_dir, f"{view_name}_depth.png"))
    torchvision.utils.save_image(depth_normal_vis, os.path.join(iter_dir, f"{view_name}_depth_normal.png"))
    torchvision.utils.save_image(torch.clamp(aux["query_selection"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_query_selection.png"))
    torchvision.utils.save_image(torch.clamp(aux["query_occlusion"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_query_occlusion.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["query_direct"]), os.path.join(iter_dir, f"{view_name}_query_direct.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["query_indirect"]), os.path.join(iter_dir, f"{view_name}_query_indirect.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["pbr_diffuse"]), os.path.join(iter_dir, f"{view_name}_pbr_diffuse.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["pbr_specular"]), os.path.join(iter_dir, f"{view_name}_pbr_specular.png"))
    torchvision.utils.save_image(torch.clamp(aux["sop_supervision_selection"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_sop_supervision_selection.png"))
    torchvision.utils.save_image(envmap.visualization(), os.path.join(iter_dir, "envmap.png"))
    torch.save(envmap.capture(), os.path.join(iter_dir, "envmap.pt"))


def write_stage2_sop_summary(path: str, args, state_info: Dict[str, object], envmap_info: Dict[str, object], sop_state: Stage2SOPState, iteration: int, ema_stats: Dict[str, float], trace_backend_name: Optional[str]):
    lin_tex = sop_state.lin_tex.detach()
    occ_tex = sop_state.occ_tex.detach()
    summary = {
        "format": "comgs_stage2_sop_summary_v1",
        "iteration": int(iteration),
        "loaded_from": state_info.get("loaded_from"),
        "sop_source": state_info.get("sop_source"),
        "envmap_source": envmap_info,
        "trace_backend": trace_backend_name,
        "num_probes": sop_state.num_probes,
        "probe_texture_shape": list(lin_tex.shape),
        "probe_lin_tex_mean": float(lin_tex.mean().item()),
        "probe_lin_tex_max": float(lin_tex.max().item()),
        "probe_occ_tex_mean": float(occ_tex.mean().item()),
        "probe_occ_tex_max": float(occ_tex.max().item()),
        "ema": ema_stats,
        "args": dict(vars(args)),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _add_boolean_toggle(parser: ArgumentParser, name: str, default: bool, help_text: str):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def _build_view_neighbor_cache_key(viewpoint_cam, sop_query_radius: float, sop_query_topk: int, weight_threshold: float):
    uid = getattr(viewpoint_cam, "uid", None)
    image_name = getattr(viewpoint_cam, "image_name", "view")
    image_height = getattr(viewpoint_cam, "image_height", None)
    image_width = getattr(viewpoint_cam, "image_width", None)
    radius_key = float(sop_query_radius) if sop_query_radius and sop_query_radius > 0.0 else -1.0
    return (
        uid,
        image_name,
        image_height,
        image_width,
        radius_key,
        int(sop_query_topk),
        float(weight_threshold),
    )


def training_stage2_sop(dataset, opt, pipe, stage2_args):
    tb_writer = prepare_output_and_logger(stage2_args)
    dataset.model_path = stage2_args.model_path

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    first_iter, resumed_envmap, env_optimizer_state, sop_state, sop_optimizer_state, state_info = load_initial_stage2_sop_state(
        gaussians=gaussians,
        opt=opt,
        stage1_ckpt=stage2_args.stage1_ckpt,
        model_path=stage2_args.model_path,
        sop_init_path=stage2_args.sop_init,
        start_checkpoint=stage2_args.start_checkpoint,
        stage1_ckpt_format=stage2_args.stage1_ckpt_format,
        envmap_height=stage2_args.envmap_height,
        envmap_width=stage2_args.envmap_width,
    )
    if not gaussians.get_xyz.is_cuda:
        raise RuntimeError("Loaded Gaussian parameters are not on CUDA. Please check checkpoint loading and current CUDA visibility.")
    configure_stage2_gaussian_optimizer(gaussians, stage2_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if resumed_envmap is None:
        envmap, envmap_info = _load_envmap_from_args(stage2_args)
    else:
        envmap = resumed_envmap
        envmap_info = state_info.get("envmap_source") or {
            "source": "stage2_sop_checkpoint",
            "path": stage2_args.start_checkpoint,
            "iteration": int(first_iter),
        }

    for param in envmap.parameters():
        param.requires_grad_(stage2_args.train_envmap)
    env_optimizer = torch.optim.Adam(envmap.parameters(), lr=stage2_args.envmap_lr if stage2_args.train_envmap else 0.0)
    if env_optimizer_state is not None:
        try:
            env_optimizer.load_state_dict(env_optimizer_state)
        except ValueError:
            print("[Stage2-SOP][Warn] Envmap optimizer state is incompatible; reinitializing envmap optimizer.")

    sop_state.set_texture_training(stage2_args.train_sop_textures)
    if stage2_args.train_sop_textures:
        sop_optimizer = torch.optim.Adam(
            [sop_state.probe_lin_tex, sop_state.probe_occ_tex],
            lr=stage2_args.sop_texture_lr,
        )
        if sop_optimizer_state is not None:
            try:
                sop_optimizer.load_state_dict(sop_optimizer_state)
            except ValueError:
                print("[Stage2-SOP][Warn] SOP optimizer state is incompatible; reinitializing SOP optimizer.")
    else:
        sop_optimizer = None

    if stage2_args.use_sop_supervision and stage2_args.lambda_sops > 0.0:
        trace_config = TraceBackendConfig(
            backend=stage2_args.trace_backend,
            trace_bias=stage2_args.trace_bias,
            secondary_num_samples=stage2_args.secondary_num_samples,
            rebuild_every=stage2_args.trace_rebuild_every,
            open3d_voxel_size=stage2_args.trace_voxel_size,
            open3d_sdf_trunc=stage2_args.trace_sdf_trunc,
            open3d_depth_trunc=stage2_args.trace_depth_trunc,
            open3d_mask_background=stage2_args.trace_mask_background,
            native_alpha_min=stage2_args.native_alpha_min,
            native_transmittance_min=stage2_args.native_transmittance_min,
        )
        tracer = build_trace_backend(trace_config, scene, gaussians, pipe, background)
        trace_backend_name = tracer.backend_name
    else:
        tracer = None
        trace_backend_name = None

    debug_dir = os.path.join(scene.model_path, "debug_stage2_sop")
    canonical_ckpt_path = os.path.join(scene.model_path, "object_step2_sop.ckpt")
    summary_path = os.path.join(scene.model_path, "object_step2_sop_summary.json")
    resolved_sop_source_path = state_info.get("sop_source", {}).get("path", stage2_args.sop_init)
    view_neighbor_cache_store = {} if stage2_args.freeze_geometry else None
    if view_neighbor_cache_store is not None:
        print("[Stage2-SOP] Per-view SOP neighbor cache enabled because geometry is frozen.")

    ema_total = 0.0
    ema_pbr = 0.0
    ema_lam = 0.0
    ema_sops = 0.0
    ema_d2n = 0.0
    ema_mask = 0.0

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Stage2-SOP training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if tracer is not None and stage2_args.trace_rebuild_every > 0 and iteration > first_iter and (iteration % stage2_args.trace_rebuild_every == 0):
            tracer.rebuild(scene=scene, gaussians=gaussians, pipe=pipe, background=background)

        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
        gt_rgb = viewpoint_cam.original_image.cuda()
        current_view_neighbor_cache = None
        if view_neighbor_cache_store is not None:
            weight_threshold = 1e-4
            cache_key = _build_view_neighbor_cache_key(
                viewpoint_cam=viewpoint_cam,
                sop_query_radius=stage2_args.sop_query_radius,
                sop_query_topk=stage2_args.sop_query_topk,
                weight_threshold=weight_threshold,
            )
            current_view_neighbor_cache = view_neighbor_cache_store.get(cache_key)
            if current_view_neighbor_cache is None:
                with torch.no_grad():
                    cached_points, cached_valid_mask = recover_shading_points(
                        view=viewpoint_cam,
                        depth_unbiased=render_pkg["depth_unbiased"],
                        weight=render_pkg["weight"],
                        weight_threshold=weight_threshold,
                    )
                    current_view_neighbor_cache = build_view_sop_neighbor_cache(
                        points=cached_points,
                        valid_mask=cached_valid_mask,
                        probe_xyz=sop_state.probe_xyz,
                        probe_normal=sop_state.probe_normal,
                        radius=float(stage2_args.sop_query_radius) if stage2_args.sop_query_radius and stage2_args.sop_query_radius > 0.0 else None,
                        topk=stage2_args.sop_query_topk,
                        chunk_size=stage2_args.sop_query_chunk_size,
                        storage_device=torch.device("cpu"),
                    )
                view_neighbor_cache_store[cache_key] = current_view_neighbor_cache

        total_loss, loss_stats, aux = compute_stage2_sop_loss(
            render_pkg=render_pkg,
            gt_rgb=gt_rgb,
            viewpoint_camera=viewpoint_cam,
            envmap=envmap,
            probe_xyz=sop_state.probe_xyz,
            probe_normal=sop_state.probe_normal,
            probe_lin_tex=sop_state.lin_tex,
            probe_occ_tex=sop_state.occ_tex,
            lambda_lam=stage2_args.lambda_lam,
            lambda_sops=stage2_args.lambda_sops,
            lambda_d2n=stage2_args.lambda_d2n,
            lambda_mask=stage2_args.lambda_mask,
            use_mask_loss=stage2_args.use_mask_loss,
            num_shading_samples=stage2_args.num_shading_samples,
            max_shading_points=stage2_args.max_shading_points,
            sop_query_radius=stage2_args.sop_query_radius,
            sop_query_topk=stage2_args.sop_query_topk,
            sop_query_chunk_size=stage2_args.sop_query_chunk_size,
            use_sop_supervision=stage2_args.use_sop_supervision,
            tracer=tracer,
            max_sop_supervision_points=stage2_args.max_sop_supervision_points,
            trace_bias=stage2_args.trace_bias,
            secondary_num_samples=stage2_args.secondary_num_samples,
            randomized_samples=not stage2_args.disable_sample_jitter,
            view_neighbor_cache=current_view_neighbor_cache,
        )

        total_loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            env_optimizer.step()
            env_optimizer.zero_grad(set_to_none=True)
            envmap.update_pdf()

            if sop_optimizer is not None:
                sop_optimizer.step()
                sop_optimizer.zero_grad(set_to_none=True)

            ema_total = 0.4 * total_loss.item() + 0.6 * ema_total
            ema_pbr = 0.4 * loss_stats["loss_pbr"].item() + 0.6 * ema_pbr
            ema_lam = 0.4 * loss_stats["loss_lam"].item() + 0.6 * ema_lam
            ema_sops = 0.4 * loss_stats["loss_sops"].item() + 0.6 * ema_sops
            ema_d2n = 0.4 * loss_stats["loss_d2n"].item() + 0.6 * ema_d2n
            ema_mask = 0.4 * loss_stats["loss_mask"].item() + 0.6 * ema_mask

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "total": f"{ema_total:.5f}",
                        "pbr": f"{ema_pbr:.5f}",
                        "lam": f"{ema_lam:.5f}",
                        "sops": f"{ema_sops:.5f}",
                        "d2n": f"{ema_d2n:.5f}",
                        "mask": f"{ema_mask:.5f}",
                        "pts": f"{int(loss_stats['shading_points'].item())}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar("train_stage2_sop/loss_total", ema_total, iteration)
                tb_writer.add_scalar("train_stage2_sop/loss_pbr", ema_pbr, iteration)
                tb_writer.add_scalar("train_stage2_sop/loss_lam", ema_lam, iteration)
                tb_writer.add_scalar("train_stage2_sop/loss_sops", ema_sops, iteration)
                tb_writer.add_scalar("train_stage2_sop/loss_d2n", ema_d2n, iteration)
                tb_writer.add_scalar("train_stage2_sop/loss_mask", ema_mask, iteration)
                tb_writer.add_scalar("train_stage2_sop/shading_points", loss_stats["shading_points"].item(), iteration)
                tb_writer.add_scalar("train_stage2_sop/shading_valid_ratio", loss_stats["shading_valid_ratio"].item(), iteration)
                tb_writer.add_scalar("train_stage2_sop/sop_supervision_points", loss_stats["sop_supervision_points"].item(), iteration)

            if stage2_args.save_debug_every > 0 and iteration % stage2_args.save_debug_every == 0:
                save_stage2_sop_debug(
                    output_root=debug_dir,
                    iteration=iteration,
                    view_name=viewpoint_cam.image_name,
                    gt_rgb=gt_rgb,
                    render_pkg=render_pkg,
                    aux=aux,
                    envmap=envmap,
                )

            if iteration in stage2_args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians, envmap, and SOP textures")
                scene.save(iteration)
                save_envmap_artifacts(envmap, scene.model_path, f"envmap_iter_{iteration:06d}")
                save_sop_artifacts(sop_state, scene.model_path, f"sop_iter_{iteration:06d}")

            if iteration in stage2_args.checkpoint_iterations:
                ckpt_path = os.path.join(scene.model_path, f"object_step2_sop_iter_{iteration:06d}.ckpt")
                print(f"\n[ITER {iteration}] Saving checkpoint to {ckpt_path}")
                save_stage2_sop_checkpoint(
                    path=ckpt_path,
                    gaussians=gaussians,
                    envmap=envmap,
                    env_optimizer=env_optimizer,
                    sop_state=sop_state,
                    sop_optimizer=sop_optimizer,
                    iteration=iteration,
                    args=stage2_args,
                    stage1_ckpt=stage2_args.stage1_ckpt,
                    sop_init_path=resolved_sop_source_path,
                    trace_backend_name=trace_backend_name,
                )

            if stage2_args.summary_every > 0 and iteration % stage2_args.summary_every == 0:
                write_stage2_sop_summary(
                    path=summary_path,
                    args=stage2_args,
                    state_info=state_info,
                    envmap_info=envmap_info,
                    sop_state=sop_state,
                    iteration=iteration,
                    ema_stats={
                        "loss_total": ema_total,
                        "loss_pbr": ema_pbr,
                        "loss_lam": ema_lam,
                        "loss_sops": ema_sops,
                        "loss_d2n": ema_d2n,
                        "loss_mask": ema_mask,
                    },
                    trace_backend_name=trace_backend_name,
                )

    scene.save(opt.iterations)
    save_envmap_artifacts(envmap, scene.model_path, "object_step2_sop_envmap")
    save_sop_artifacts(sop_state, scene.model_path, "object_step2_sop_probes")
    save_stage2_sop_checkpoint(
        path=canonical_ckpt_path,
        gaussians=gaussians,
        envmap=envmap,
        env_optimizer=env_optimizer,
        sop_state=sop_state,
        sop_optimizer=sop_optimizer,
        iteration=opt.iterations,
        args=stage2_args,
        stage1_ckpt=stage2_args.stage1_ckpt,
        sop_init_path=resolved_sop_source_path,
        trace_backend_name=trace_backend_name,
    )
    write_stage2_sop_summary(
        path=summary_path,
        args=stage2_args,
        state_info=state_info,
        envmap_info=envmap_info,
        sop_state=sop_state,
        iteration=opt.iterations,
        ema_stats={
            "loss_total": ema_total,
            "loss_pbr": ema_pbr,
            "loss_lam": ema_lam,
            "loss_sops": ema_sops,
            "loss_d2n": ema_d2n,
            "loss_mask": ema_mask,
        },
        trace_backend_name=trace_backend_name,
    )
    print(f"[Stage2-SOP] Final checkpoint written to {canonical_ckpt_path}")


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="COMGS object-only Step2 SOP decomposition training")
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)
    parser.set_defaults(iterations=2000, position_lr_max_steps=2000)

    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument(
        "--stage1_ckpt_format",
        type=str,
        default="auto",
        choices=["auto", "comgs", "irgs_refgs"],
        help="Checkpoint format for --stage1_ckpt. Use irgs_refgs to load IRGS refgs checkpoints into COMGS stage2.",
    )
    parser.add_argument("--sop_init", type=str, default="")
    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--lambda_lam", type=float, default=0.001)
    parser.add_argument("--lambda_sops", type=float, default=1.0)
    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--max_shading_points", type=int, default=4096)
    parser.add_argument("--max_sop_supervision_points", type=int, default=1024)
    parser.add_argument("--trace_bias", type=float, default=1e-3)
    parser.add_argument("--secondary_num_samples", type=int, default=16)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=False)
    parser.add_argument("--sop_query_radius", type=float, default=0.0)
    parser.add_argument("--sop_query_topk", type=int, default=8)
    parser.add_argument("--sop_query_chunk_size", type=int, default=1024)

    parser.add_argument("--save_debug_every", type=int, default=500)
    parser.add_argument("--summary_every", type=int, default=500)
    parser.add_argument("--material_lr", type=float, default=None)
    parser.add_argument("--albedo_lr", type=float, default=0.0075)
    parser.add_argument("--roughness_lr", type=float, default=0.005)
    parser.add_argument("--metallic_lr", type=float, default=0.005)
    parser.add_argument("--envmap_lr", type=float, default=0.01)
    parser.add_argument("--sop_texture_lr", type=float, default=0.001)
    parser.add_argument("--envmap_height", type=int, default=256)
    parser.add_argument("--envmap_width", type=int, default=256)
    parser.add_argument("--envmap_init_value", type=float, default=1.5)
    parser.add_argument("--envmap_activation", type=str, default="exp", choices=["exp", "softplus", "none"])
    parser.add_argument("--stage2_trace_ckpt", default="", type=str)
    parser.add_argument("--envmap_capture", default="", type=str)

    parser.add_argument("--trace_backend", type=str, default="auto", choices=["auto", "irgs", "irgs_adapter", "irgs_native", "open3d", "open3d_mesh"])
    parser.add_argument("--trace_rebuild_every", type=int, default=0)
    parser.add_argument("--trace_voxel_size", type=float, default=0.004)
    parser.add_argument("--trace_sdf_trunc", type=float, default=0.02)
    parser.add_argument("--trace_depth_trunc", type=float, default=0.0)
    parser.add_argument("--native_alpha_min", type=float, default=1.0 / 255.0)
    parser.add_argument("--native_transmittance_min", type=float, default=0.03)

    _add_boolean_toggle(parser, "freeze_geometry", default=True, help_text="freeze geometry-related Gaussian parameters during stage2")
    _add_boolean_toggle(parser, "freeze_color", default=True, help_text="freeze SH color parameters during stage2")
    _add_boolean_toggle(parser, "train_envmap", default=True, help_text="optimize the learnable environment map during stage2")
    _add_boolean_toggle(parser, "train_sop_textures", default=True, help_text="optimize SOP radiance and occlusion textures during stage2")
    _add_boolean_toggle(parser, "use_sop_supervision", default=True, help_text="supervise queried SOP radiance and occlusion with traced targets during stage2")
    _add_boolean_toggle(parser, "trace_mask_background", default=True, help_text="mask object background while extracting the Open3D trace mesh")
    return parser


def main():
    parser = _build_parser()
    args = get_combined_args(parser)

    if not getattr(args, "model_path", ""):
        raise RuntimeError("Stage2 SOP training requires --model_path/-m so the scene output directory and cfg_args can be resolved.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("Stage2 SOP training requires --source_path/-s, or a cfg_args under --model_path that already records source_path.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage2 SOP decomposition currently requires CUDA because it reuses the Gaussian renderer and optional tracing backend.")

    if args.lambda_mask > 0.0 and not args.use_mask_loss:
        print("[Stage2-SOP][Warn] --lambda_mask > 0 but --use_mask_loss is not set. Mask loss is disabled and will stay 0.")
    if args.use_mask_loss and args.lambda_mask <= 0.0:
        print("[Stage2-SOP][Warn] --use_mask_loss is set but --lambda_mask <= 0. Mask loss contributes 0 to total loss.")
    if args.lambda_sops > 0.0 and not args.use_sop_supervision:
        print("[Stage2-SOP][Warn] --lambda_sops > 0 but --use_sop_supervision is not set. SOP supervision loss is disabled and will stay 0.")
    if args.use_sop_supervision and args.lambda_sops <= 0.0:
        print("[Stage2-SOP][Warn] --use_sop_supervision is set but --lambda_sops <= 0. SOP supervision contributes 0 to total loss.")
    if args.use_sop_supervision and not args.train_sop_textures:
        print("[Stage2-SOP][Warn] SOP supervision is enabled while SOP textures are frozen. L_sops will only be logged and will not update probe textures.")

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)

    training_stage2_sop(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        stage2_args=args,
    )

    print("\nStage2 SOP decomposition training complete.")


if __name__ == "__main__":
    main()
