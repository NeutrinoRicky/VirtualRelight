import os
from pathlib import Path
from random import randint
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_sop_gbuffer
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.losses_comgs_stage2_sop import compute_stage2_sop_loss
from utils.image_utils import visualize_depth
from utils.graphics_utils import rgb_to_srgb


class Stage2SOPState(nn.Module):
    def __init__(self, payload):
        super().__init__()
        required = ["probe_xyz", "probe_normal", "probe_lin_tex", "probe_occ_tex"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"SOP payload is missing required fields: {missing}")

        probe_xyz = payload["probe_xyz"].float().cuda()
        probe_normal = torch.nn.functional.normalize(payload["probe_normal"].float().cuda(), dim=-1, eps=1e-6)
        probe_lin_tex = payload["probe_lin_tex"].float().cuda()
        probe_occ_tex = payload["probe_occ_tex"].float().cuda()

        self.register_buffer("probe_xyz", probe_xyz)
        self.register_buffer("probe_normal", probe_normal)
        self.probe_lin_tex = nn.Parameter(probe_lin_tex.requires_grad_(True))
        self.probe_occ_tex = nn.Parameter(probe_occ_tex.requires_grad_(True))

        self._optional_tensor_keys = []
        for key in ["probe_albedo_tex", "probe_roughness_tex", "probe_metallic_tex", "oct_dirs"]:
            if key in payload and torch.is_tensor(payload[key]):
                self.register_buffer(key, payload[key].float().cuda())
                self._optional_tensor_keys.append(key)

        self.source_format = str(payload.get("format", "unknown"))
        self.source_summary = payload.get("summary")

    @property
    def lin_tex(self):
        return torch.clamp(self.probe_lin_tex, min=0.0)

    @property
    def occ_tex(self):
        return torch.clamp(self.probe_occ_tex, 0.0, 1.0)

    def capture(self):
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


def set_gaussian_para(gaussians, opt):
    gaussians.init_base_color_value = opt.init_base_color_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.init_roughness_value = opt.init_roughness_value


def _cuda_mem_debug_enabled(debug_cfg) -> bool:
    return bool(debug_cfg) and bool(debug_cfg.get("enabled", False)) and torch.cuda.is_available()


def _log_cuda_mem(debug_cfg, label: str) -> None:
    if not _cuda_mem_debug_enabled(debug_cfg):
        return
    torch.cuda.synchronize()
    prefix = debug_cfg.get("prefix", "")
    if prefix:
        label = f"{prefix} | {label}"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[CUDA-MEM] {label}: alloc={allocated:.1f} MiB peak={peak:.1f} MiB reserved={reserved:.1f} MiB")


def _freeze_sh_gradients(gaussians) -> None:
    for tensor_name in ("_features_dc", "_features_rest"):
        tensor = getattr(gaussians, tensor_name, None)
        if tensor is None:
            continue
        tensor.requires_grad_(False)
        tensor.grad = None

    if gaussians.optimizer is not None:
        for param_group in gaussians.optimizer.param_groups:
            if param_group.get("name") in {"f_dc", "f_rest"}:
                param_group["lr"] = 0.0

    print("[Stage2-SOP] Frozen SH gradients for f_dc and f_rest.")


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


def _cli_has_flag(flag_name: str) -> bool:
    prefix = f"--{flag_name}"
    for arg in sys.argv[1:]:
        if arg == prefix or arg.startswith(prefix + "="):
            return True
    return False


def _load_sop_payload(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"SOP init file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"SOP init payload must be a dict: {path}")
    return payload


def _load_initial_state(gaussians, opt, args):
    if args.start_checkpoint:
        payload = torch.load(args.start_checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != "irgs_stage2_sop_v1":
            raise RuntimeError(f"{args.start_checkpoint} is not a valid Stage2 SOP checkpoint")

        gaussians.restore(payload["gaussians"], opt)
        sop_state = Stage2SOPState(payload["sop"])
        return int(payload.get("iteration", 0)), sop_state, payload.get("sop_optimizer"), payload.get("source_info", {})

    if not args.start_checkpoint_refgs:
        raise RuntimeError("Stage2 SOP requires --start_checkpoint_refgs when --start_checkpoint is not provided.")

    refgs_payload = torch.load(args.start_checkpoint_refgs, weights_only=False)
    if not isinstance(refgs_payload, (tuple, list)) or len(refgs_payload) < 1:
        raise RuntimeError(f"Unsupported refgs checkpoint payload: {type(refgs_payload).__name__}")

    model_params = refgs_payload[0]
    gaussians.restore_from_refgs(model_params, opt)

    sop_path = _resolve_sop_init_path(args.sop_init, args.model_path)
    sop_state = Stage2SOPState(_load_sop_payload(sop_path))
    source_info = {
        "refgs_checkpoint": args.start_checkpoint_refgs,
        "sop_init": str(sop_path),
    }
    return 0, sop_state, None, source_info


def _save_stage2_sop_checkpoint(path, gaussians, sop_state, sop_optimizer, iteration, args, source_info):
    payload = {
        "format": "irgs_stage2_sop_v1",
        "iteration": int(iteration),
        "args": dict(vars(args)),
        "source_info": source_info,
        "gaussians": gaussians.capture(),
        "sop": sop_state.capture(),
        "sop_optimizer": sop_optimizer.state_dict() if sop_optimizer is not None else None,
    }
    torch.save(payload, path)


def _repeat_to_rgb(x):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


@torch.no_grad()
def save_training_vis(viewpoint_cam, gaussians, background, pipe, opt, sop_state, args, iteration):
    render_pkg = render_sop_gbuffer(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        opt=opt,
        iteration=iteration,
        training=False,
    )
    gt_rgb = viewpoint_cam.original_image.cuda()
    _, _, aux = compute_stage2_sop_loss(
        render_pkg=render_pkg,
        gt_rgb=gt_rgb,
        viewpoint_camera=viewpoint_cam,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        training=False,
        probe_xyz=sop_state.probe_xyz,
        probe_normal=sop_state.probe_normal,
        probe_lin_tex=sop_state.lin_tex,
        probe_occ_tex=sop_state.occ_tex,
        opt=opt,
        lambda_lam=args.lambda_lam,
        lambda_sops=0.0,
        lambda_d2n=args.lambda_d2n,
        lambda_mask=args.lambda_mask,
        use_mask_loss=args.use_mask_loss,
        num_shading_samples=args.num_shading_samples,
        max_shading_points=0,
        sop_query_radius=args.sop_query_radius,
        sop_query_topk=args.sop_query_topk,
        sop_query_chunk_size=args.sop_query_chunk_size,
        randomized_samples=False,
        cuda_mem_debug=None,
    )

    error_map = torch.abs(gt_rgb - aux["pbr_render"])
    visualization_list = [
        gt_rgb,
        torch.clamp(aux["pbr_render"], 0.0, 1.0),
        torch.clamp(render_pkg["albedo"], 0.0, 1.0),
        _repeat_to_rgb(torch.clamp(render_pkg["roughness"], 0.0, 1.0)),
        _repeat_to_rgb(torch.clamp(render_pkg["metallic"], 0.0, 1.0)),
        _repeat_to_rgb(torch.clamp(render_pkg["weight"], 0.0, 1.0)),
        visualize_depth(render_pkg["depth_unbiased"][None]),
        torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0),
        torch.clamp(aux["depth_normal"] * 0.5 + 0.5, 0.0, 1.0),
        _repeat_to_rgb(torch.clamp(aux["query_selection"], 0.0, 1.0)),
        _repeat_to_rgb(torch.clamp(aux["query_occlusion"], 0.0, 1.0)),
        torch.clamp(aux["query_direct"], 0.0, 1.0),
        torch.clamp(aux["query_indirect"], 0.0, 1.0),
        torch.clamp(aux["pbr_diffuse"], 0.0, 1.0),
        torch.clamp(aux["pbr_specular"], 0.0, 1.0),
        torch.clamp(error_map, 0.0, 1.0),
    ]

    grid = torch.stack(visualization_list, dim=0)
    grid = make_grid(grid, nrow=4)
    scale = grid.shape[-2] / 1600
    if scale > 1.0:
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
    save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

    env_dict = gaussians.render_env_map()
    if "env1" in env_dict and "env2" in env_dict:
        env_grid = [
            rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)),
            rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)),
        ]
        env_grid = make_grid(env_grid, nrow=1, padding=10)
        save_image(env_grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))
    else:
        env_image = rgb_to_srgb(env_dict["env"].permute(2, 0, 1))
        save_image(env_image, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))


def training_stage2_sop(dataset, opt, pipe, args):
    lr_scale = opt.lr_scale
    opt.position_lr_init *= lr_scale
    opt.opacity_lr *= lr_scale
    opt.scaling_lr *= lr_scale
    opt.rotation_lr *= lr_scale

    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    first_iter, sop_state, sop_optimizer_state, source_info = _load_initial_state(gaussians, opt, args)
    _freeze_sh_gradients(gaussians)

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

    sop_optimizer = torch.optim.Adam(
        [sop_state.probe_lin_tex, sop_state.probe_occ_tex],
        lr=args.sop_texture_lr,
    )
    if sop_optimizer_state is not None:
        sop_optimizer.load_state_dict(sop_optimizer_state)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_total = 0.0
    ema_pbr = 0.0
    ema_lam = 0.0
    ema_d2n = 0.0
    ema_mask = 0.0
    ema_light = 0.0
    ema_base_color_smooth = 0.0
    ema_roughness_smooth = 0.0
    ema_normal_smooth = 0.0
    ema_light_smooth = 0.0

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Stage2 SOP training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        debug_this_iter = args.cuda_mem_debug_iters > 0 and (iteration - first_iter) < args.cuda_mem_debug_iters
        cuda_mem_debug = None
        if debug_this_iter and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cuda_mem_debug = {
                "enabled": True,
                "prefix": f"iter={iteration}",
                "rendering_equation_sop_logs_remaining": 1,
            }
            print("[Stage2-SOP] CUDA memory debug is active; SOP activation checkpoint is disabled for this iteration.")
            _log_cuda_mem(cuda_mem_debug, "iter_start")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_sop_gbuffer(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            opt=opt,
            iteration=iteration,
            training=True,
        )
        _log_cuda_mem(cuda_mem_debug, "after render_sop_gbuffer")
        gt_rgb = viewpoint_cam.original_image.cuda()
        total_loss, loss_stats, _aux = compute_stage2_sop_loss(
            render_pkg=render_pkg,
            gt_rgb=gt_rgb,
            viewpoint_camera=viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            opt=opt,
            training=True,
            probe_xyz=sop_state.probe_xyz,
            probe_normal=sop_state.probe_normal,
            probe_lin_tex=sop_state.lin_tex,
            probe_occ_tex=sop_state.occ_tex,
            lambda_lam=args.lambda_lam,
            lambda_sops=args.lambda_sops,
            lambda_d2n=args.lambda_d2n,
            lambda_mask=args.lambda_mask,
            use_mask_loss=args.use_mask_loss,
            num_shading_samples=args.num_shading_samples,
            max_shading_points=args.max_shading_points,
            sop_query_radius=args.sop_query_radius,
            sop_query_topk=args.sop_query_topk,
            sop_query_chunk_size=args.sop_query_chunk_size,
            randomized_samples=not args.disable_sample_jitter,
            cuda_mem_debug=cuda_mem_debug,
        )
        _log_cuda_mem(cuda_mem_debug, "before loss.backward")
        total_loss.backward()
        _log_cuda_mem(cuda_mem_debug, "after loss.backward")

        with torch.no_grad():
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                sop_optimizer.step()
                sop_optimizer.zero_grad(set_to_none=True)

            if args.visualize_every > 0 and (iteration % args.visualize_every == 0 or iteration == first_iter):
                save_training_vis(viewpoint_cam, gaussians, background, pipe, opt, sop_state, args, iteration)

            ema_total = 0.4 * total_loss.item() + 0.6 * ema_total
            ema_pbr = 0.4 * loss_stats["loss_pbr"].item() + 0.6 * ema_pbr
            ema_lam = 0.4 * loss_stats["loss_lam"].item() + 0.6 * ema_lam
            ema_d2n = 0.4 * loss_stats["loss_d2n"].item() + 0.6 * ema_d2n
            ema_mask = 0.4 * loss_stats["loss_mask"].item() + 0.6 * ema_mask
            ema_light = 0.4 * loss_stats["loss_light"].item() + 0.6 * ema_light
            ema_base_color_smooth = 0.4 * loss_stats["loss_base_color_smooth"].item() + 0.6 * ema_base_color_smooth
            ema_roughness_smooth = 0.4 * loss_stats["loss_roughness_smooth"].item() + 0.6 * ema_roughness_smooth
            ema_normal_smooth = 0.4 * loss_stats["loss_normal_smooth"].item() + 0.6 * ema_normal_smooth
            ema_light_smooth = 0.4 * loss_stats["loss_light_smooth"].item() + 0.6 * ema_light_smooth

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "total": f"{ema_total:.5f}",
                        "pbr": f"{ema_pbr:.5f}",
                        "lam": f"{ema_lam:.5f}",
                        "d2n": f"{ema_d2n:.5f}",
                        "mask": f"{ema_mask:.5f}",
                        "light": f"{ema_light:.5f}",
                        "bc_sm": f"{ema_base_color_smooth:.5f}",
                        "r_sm": f"{ema_roughness_smooth:.5f}",
                        "n_sm": f"{ema_normal_smooth:.5f}",
                        "l_sm": f"{ema_light_smooth:.5f}",
                        "pts": f"{int(loss_stats['shading_points'].item())}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in args.checkpoint_iterations:
                ckpt_path = os.path.join(scene.model_path, f"object_step2_sop_iter_{iteration:06d}.ckpt")
                print(f"\n[ITER {iteration}] Saving Checkpoint to {ckpt_path}")
                _save_stage2_sop_checkpoint(
                    path=ckpt_path,
                    gaussians=gaussians,
                    sop_state=sop_state,
                    sop_optimizer=sop_optimizer,
                    iteration=iteration,
                    args=args,
                    source_info=source_info,
                )

    scene.save(opt.iterations)
    final_ckpt = os.path.join(scene.model_path, "object_step2_sop.ckpt")
    _save_stage2_sop_checkpoint(
        path=final_ckpt,
        gaussians=gaussians,
        sop_state=sop_state,
        sop_optimizer=sop_optimizer,
        iteration=opt.iterations,
        args=args,
        source_info=source_info,
    )
    print(f"[Stage2-SOP] Final checkpoint written to {final_ckpt}")


def _build_parser():
    parser = ArgumentParser(description="Stage2 SOP training for COMGS_IRGS")
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--start_checkpoint_refgs", type=str, default="")
    parser.add_argument("--sop_init", type=str, default="")

    parser.add_argument("--lambda_lam", type=float, default=0.0)
    parser.add_argument("--lambda_sops", type=float, default=0.0)
    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--max_shading_points", type=int, default=4096)
    parser.add_argument("--sop_query_radius", type=float, default=0.0)
    parser.add_argument("--sop_query_topk", type=int, default=4)
    parser.add_argument("--sop_query_chunk_size", type=int, default=1024)
    parser.add_argument("--sop_texture_lr", type=float, default=0.001)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=False)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--cuda_mem_debug_iters", type=int, default=1)
    return parser


def main():
    parser = _build_parser()
    args = get_combined_args(parser)

    if not _cli_has_flag("diffuse_sample_num"):
        args.diffuse_sample_num = 128

    if not getattr(args, "model_path", ""):
        raise RuntimeError("Stage2 SOP training requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("Stage2 SOP training requires --source_path/-s, or a cfg_args under --model_path.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage2 SOP training currently requires CUDA.")

    os.makedirs(args.model_path, exist_ok=True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    with open(os.path.join(args.model_path, "cmd.txt"), "w") as cmd_f:
        cmd_f.write(full_cmd)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(args))

    args.visualize_path = os.path.join(args.model_path, "visualize")
    os.makedirs(args.visualize_path, exist_ok=True)

    print("Optimizing " + args.model_path)
    print("Visualization folder: {}".format(args.visualize_path))
    print("Stage2 diffuse_sample_num: {}".format(args.diffuse_sample_num))
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)

    training_stage2_sop(dataset, opt, pipe, args)
    print("\nStage2 SOP training complete.")


if __name__ == "__main__":
    main()
