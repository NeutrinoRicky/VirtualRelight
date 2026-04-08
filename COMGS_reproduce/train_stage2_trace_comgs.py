import os
import sys
from random import randint
from typing import Optional

import torch
import torchvision
from tqdm import tqdm

from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_multitarget
from scene import Scene, GaussianModel
from train_stage1_comgs import prepare_output_and_logger
from utils.deferred_pbr_comgs import OctahedralEnvMap, load_envmap_capture_as_octahedral
from utils.general_utils import safe_state
from utils.losses_comgs_stage2_trace import compute_stage2_trace_loss
from utils.tracing_comgs import TraceBackendConfig, build_trace_backend


def _cuda_checkpoint_map_location():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda")


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


def _clone_tensor(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _clone_trainable_parameter(x: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(x.detach().clone().requires_grad_(True))


def _filled_trainable_parameter_like(x: torch.Tensor, value: float) -> torch.nn.Parameter:
    return _clone_trainable_parameter(torch.full_like(x, float(value)))


def _encode_visible_material_like(gaussians: GaussianModel, template: torch.Tensor, value: float, material_name: str) -> torch.nn.Parameter:
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


def _resolve_stage1_model_args(gaussians: GaussianModel, stage1_ckpt: str, stage1_ckpt_format: str):
    payload = torch.load(stage1_ckpt, map_location=_cuda_checkpoint_map_location())
    if isinstance(payload, dict) and "gaussians" in payload:
        return payload["gaussians"], payload.get("format", "checkpoint_dict")

    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload type for {stage1_ckpt}: {type(payload).__name__}")

    model_args = payload[0]
    if stage1_ckpt_format == "comgs":
        if not isinstance(model_args, (tuple, list)) or len(model_args) not in {12, 15}:
            raise RuntimeError(
                "Expected a COMGS checkpoint with model params len=12 or len=15, "
                f"but got {_describe_model_args(model_args)} from {stage1_ckpt}."
            )
        return model_args, "comgs_stage1"

    if stage1_ckpt_format == "irgs_refgs":
        return (
            _convert_irgs_refgs_model_args(model_args, gaussians.optimizer.state_dict(), gaussians),
            "irgs_refgs_compat",
        )

    if isinstance(model_args, (tuple, list)) and len(model_args) in {12, 15}:
        return model_args, "comgs_stage1"

    if isinstance(model_args, (tuple, list)) and len(model_args) == 19:
        return (
            _convert_irgs_refgs_model_args(model_args, gaussians.optimizer.state_dict(), gaussians),
            "irgs_refgs_compat",
        )

    raise RuntimeError(
        "Unable to auto-detect stage1 checkpoint format for "
        f"{stage1_ckpt}: got {_describe_model_args(model_args)}. "
        "Try --stage1_ckpt_format comgs or --stage1_ckpt_format irgs_refgs."
    )



def load_initial_checkpoint(
    gaussians: GaussianModel,
    opt,
    stage1_ckpt: str,
    start_checkpoint: Optional[str] = None,
    stage1_ckpt_format: str = "auto",
):
    first_iter = 0
    if start_checkpoint is not None:
        payload = torch.load(start_checkpoint, map_location=_cuda_checkpoint_map_location())
        if isinstance(payload, dict) and payload.get("format") == "comgs_stage2_trace_v1":
            gaussians.restore(payload["gaussians"], opt)
            first_iter = int(payload.get("iteration", 0))
            envmap = load_envmap_capture_as_octahedral(
                payload["envmap"],
                height=256,
                width=256,
            ).cuda()
            env_optimizer_state = payload.get("env_optimizer")
            print(f"[Stage2-Trace] Resumed stage2 checkpoint from {start_checkpoint} @ iter {first_iter}.")
            return first_iter, envmap, env_optimizer_state

    model_params, detected_format = _resolve_stage1_model_args(gaussians, stage1_ckpt, stage1_ckpt_format)
    gaussians.restore(model_params, opt)
    print(f"[Stage2-Trace] Loaded {detected_format} checkpoint from {stage1_ckpt}.")
    return first_iter, None, None



def save_stage2_trace_checkpoint(path, gaussians, envmap, env_optimizer, iteration, args, stage1_ckpt, trace_backend_name):
    payload = {
        "format": "comgs_stage2_trace_v1",
        "iteration": int(iteration),
        "stage1_checkpoint": stage1_ckpt,
        "stage1_checkpoint_format": getattr(args, "stage1_ckpt_format", "auto"),
        "trace_backend": trace_backend_name,
        "args": dict(vars(args)),
        "gaussians": gaussians.capture(),
        "envmap": envmap.capture(),
        "env_optimizer": env_optimizer.state_dict(),
    }
    torch.save(payload, path)



def save_envmap_artifacts(envmap, output_root: str, stem: str):
    os.makedirs(output_root, exist_ok=True)
    torchvision.utils.save_image(envmap.visualization(), os.path.join(output_root, f"{stem}.png"))
    torch.save(envmap.capture(), os.path.join(output_root, f"{stem}.pt"))



def save_stage2_trace_debug(output_root, iteration, view_name, gt_rgb, render_pkg, aux, envmap):
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
    torchvision.utils.save_image(torch.clamp(aux["trace_selection"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_trace_selection.png"))
    torchvision.utils.save_image(torch.clamp(aux["trace_occlusion"], 0.0, 1.0), os.path.join(iter_dir, f"{view_name}_trace_occlusion.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["trace_direct"]), os.path.join(iter_dir, f"{view_name}_trace_direct.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["trace_indirect"]), os.path.join(iter_dir, f"{view_name}_trace_indirect.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["pbr_diffuse"]), os.path.join(iter_dir, f"{view_name}_pbr_diffuse.png"))
    torchvision.utils.save_image(_tonemap_for_vis(aux["pbr_specular"]), os.path.join(iter_dir, f"{view_name}_pbr_specular.png"))
    torchvision.utils.save_image(envmap.visualization(), os.path.join(iter_dir, "envmap.png"))
    torch.save(envmap.capture(), os.path.join(iter_dir, "envmap.pt"))



def training_stage2_trace(dataset, opt, pipe, stage2_args):
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    first_iter, resumed_envmap, env_optimizer_state = load_initial_checkpoint(
        gaussians=gaussians,
        opt=opt,
        stage1_ckpt=stage2_args.stage1_ckpt,
        start_checkpoint=stage2_args.start_checkpoint,
        stage1_ckpt_format=stage2_args.stage1_ckpt_format,
    )
    configure_stage2_gaussian_optimizer(gaussians, stage2_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if resumed_envmap is None:
        envmap = OctahedralEnvMap(
            height=stage2_args.envmap_height,
            width=stage2_args.envmap_width,
            init_value=stage2_args.envmap_init_value,
            activation=stage2_args.envmap_activation,
        ).cuda()
    else:
        envmap = resumed_envmap

    env_optimizer = torch.optim.Adam(envmap.parameters(), lr=stage2_args.envmap_lr)
    if env_optimizer_state is not None:
        try:
            env_optimizer.load_state_dict(env_optimizer_state)
        except ValueError:
            print("[Stage2-Trace][Warn] Envmap optimizer state is incompatible; reinitializing envmap optimizer.")
    for group in env_optimizer.param_groups:
        group["lr"] = stage2_args.envmap_lr

    if (not stage2_args.freeze_geometry) and stage2_args.trace_rebuild_every <= 0:
        stage2_args.trace_rebuild_every = 1
        print("[Stage2-Trace] Enabled --trace_rebuild_every=1 for joint geometry optimization.")

    trace_config = TraceBackendConfig(
        backend=stage2_args.trace_backend,
        trace_bias=stage2_args.trace_bias,
        secondary_num_samples=stage2_args.secondary_num_samples,
        rebuild_every=stage2_args.trace_rebuild_every,
        open3d_voxel_size=stage2_args.trace_voxel_size,
        open3d_sdf_trunc=stage2_args.trace_sdf_trunc,
        open3d_depth_trunc=stage2_args.trace_depth_trunc,
        open3d_mask_background=stage2_args.trace_mask_background,
    )
    trace_backend = build_trace_backend(trace_config, scene, gaussians, pipe, background)

    debug_dir = os.path.join(scene.model_path, "debug_stage2_trace")
    canonical_ckpt_path = os.path.join(scene.model_path, "object_step2_trace.ckpt")
    ema_total = 0.0
    ema_raster = 0.0
    ema_pbr = 0.0
    ema_lam = 0.0
    ema_d2n = 0.0
    ema_mask = 0.0

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Stage2-Trace training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if stage2_args.trace_rebuild_every > 0 and iteration > first_iter and (iteration % stage2_args.trace_rebuild_every == 0):
            trace_backend.rebuild(scene=scene, gaussians=gaussians, pipe=pipe, background=background)

        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        need_debug_maps = stage2_args.save_debug_every > 0 and iteration % stage2_args.save_debug_every == 0

        render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
        gt_rgb = viewpoint_cam.original_image.cuda()

        total_loss, loss_stats, aux = compute_stage2_trace_loss(
            render_pkg=render_pkg,
            gt_rgb=gt_rgb,
            viewpoint_camera=viewpoint_cam,
            background=background,
            envmap=envmap,
            tracer=trace_backend,
            iteration=iteration,
            normal_loss_start=stage2_args.normal_loss_start,
            lambda_raster=stage2_args.lambda_raster,
            lambda_dssim=opt.lambda_dssim,
            lambda_lam=stage2_args.lambda_lam,
            lambda_d2n=stage2_args.lambda_d2n,
            lambda_mask=stage2_args.lambda_mask,
            lambda_base_color_smooth=stage2_args.lambda_base_color_smooth,
            lambda_roughness_smooth=stage2_args.lambda_roughness_smooth,
            lambda_metallic_smooth=stage2_args.lambda_metallic_smooth,
            lambda_normal_smooth=stage2_args.lambda_normal_smooth,
            lambda_light=stage2_args.lambda_light,
            lambda_light_smooth=stage2_args.lambda_light_smooth,
            use_mask_loss=stage2_args.use_mask_loss,
            num_shading_samples=stage2_args.num_shading_samples,
            secondary_num_samples=stage2_args.secondary_num_samples,
            max_trace_points=(
                stage2_args.max_trace_points
                if stage2_args.max_trace_points > 0
                else max(stage2_args.trace_num_rays // max(stage2_args.num_shading_samples, 1), 1)
            ),
            trace_bias=stage2_args.trace_bias,
            randomized_samples=not stage2_args.disable_sample_jitter,
            collect_debug_maps=need_debug_maps,
        )

        total_loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            env_optimizer.step()
            env_optimizer.zero_grad(set_to_none=True)
            envmap.update_pdf()

            ema_total = 0.4 * total_loss.item() + 0.6 * ema_total
            ema_raster = 0.4 * loss_stats["loss_raster"].item() + 0.6 * ema_raster
            ema_pbr = 0.4 * loss_stats["loss_pbr"].item() + 0.6 * ema_pbr
            ema_lam = 0.4 * loss_stats["loss_lam"].item() + 0.6 * ema_lam
            ema_d2n = 0.4 * loss_stats["loss_d2n"].item() + 0.6 * ema_d2n
            ema_mask = 0.4 * loss_stats["loss_mask"].item() + 0.6 * ema_mask

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "total": f"{ema_total:.5f}",
                        "raster": f"{ema_raster:.5f}",
                        "pbr": f"{ema_pbr:.5f}",
                        "lam": f"{ema_lam:.5f}",
                        "d2n": f"{ema_d2n:.5f}",
                        "mask": f"{ema_mask:.5f}",
                        "trace": f"{int(loss_stats['trace_points'].item())}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar("train_stage2_trace/loss_total", ema_total, iteration)
                tb_writer.add_scalar("train_stage2_trace/loss_raster", ema_raster, iteration)
                tb_writer.add_scalar("train_stage2_trace/loss_pbr", ema_pbr, iteration)
                tb_writer.add_scalar("train_stage2_trace/loss_lam", ema_lam, iteration)
                tb_writer.add_scalar("train_stage2_trace/loss_d2n", ema_d2n, iteration)
                tb_writer.add_scalar("train_stage2_trace/loss_mask", ema_mask, iteration)
                tb_writer.add_scalar("train_stage2_trace/trace_points", loss_stats["trace_points"].item(), iteration)
                tb_writer.add_scalar("train_stage2_trace/trace_valid_ratio", loss_stats["trace_valid_ratio"].item(), iteration)

            if need_debug_maps:
                save_stage2_trace_debug(
                    output_root=debug_dir,
                    iteration=iteration,
                    view_name=viewpoint_cam.image_name,
                    gt_rgb=gt_rgb,
                    render_pkg=render_pkg,
                    aux=aux,
                    envmap=envmap,
                )

            if iteration in stage2_args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians and envmap state")
                scene.save(iteration)
                save_envmap_artifacts(envmap, scene.model_path, f"envmap_iter_{iteration:06d}")

            if iteration in stage2_args.checkpoint_iterations:
                ckpt_path = os.path.join(scene.model_path, f"object_step2_trace_iter_{iteration:06d}.ckpt")
                print(f"\n[ITER {iteration}] Saving checkpoint to {ckpt_path}")
                save_stage2_trace_checkpoint(
                    path=ckpt_path,
                    gaussians=gaussians,
                    envmap=envmap,
                    env_optimizer=env_optimizer,
                    iteration=iteration,
                    args=stage2_args,
                    stage1_ckpt=stage2_args.stage1_ckpt,
                    trace_backend_name=trace_backend.backend_name,
                )

    scene.save(opt.iterations)
    save_envmap_artifacts(envmap, scene.model_path, "object_step2_trace_envmap")
    save_stage2_trace_checkpoint(
        path=canonical_ckpt_path,
        gaussians=gaussians,
        envmap=envmap,
        env_optimizer=env_optimizer,
        iteration=opt.iterations,
        args=stage2_args,
        stage1_ckpt=stage2_args.stage1_ckpt,
        trace_backend_name=trace_backend.backend_name,
    )
    print(f"[Stage2-Trace] Final checkpoint written to {canonical_ckpt_path}")



def _add_boolean_freeze_flags(parser: ArgumentParser, name: str, default: bool, help_text: str):
    parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    parser.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


if __name__ == "__main__":
    parser = ArgumentParser(description="COMGS object-only Step2 Trace training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.set_defaults(iterations=2000, position_lr_max_steps=2000, feature_lr=0.0075)

    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument(
        "--stage1_ckpt_format",
        type=str,
        default="auto",
        choices=["auto", "comgs", "irgs_refgs"],
        help="Checkpoint format for --stage1_ckpt. Use irgs_refgs to load IRGS refgs checkpoints into COMGS stage2.",
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--lambda_raster", type=float, default=1.0)
    parser.add_argument("--lambda_lam", type=float, default=0.001)
    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--normal_loss_start", type=int, default=1000)
    parser.add_argument("--lambda_mask", type=float, default=0.01)
    parser.add_argument("--lambda_base_color_smooth", type=float, default=2.0)
    parser.add_argument("--lambda_roughness_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_metallic_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_normal_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_light", type=float, default=0.5)
    parser.add_argument("--lambda_light_smooth", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--num_shading_samples", type=int, default=256)
    parser.add_argument("--secondary_num_samples", type=int, default=16)
    parser.add_argument("--trace_num_rays", type=int, default=2**18)
    parser.add_argument("--max_trace_points", type=int, default=0)
    parser.add_argument("--trace_bias", type=float, default=0.05)
    parser.add_argument("--save_debug_every", type=int, default=500)
    parser.add_argument("--material_lr", type=float, default=None)
    parser.add_argument("--albedo_lr", type=float, default=0.0075)
    parser.add_argument("--roughness_lr", type=float, default=0.005)
    parser.add_argument("--metallic_lr", type=float, default=0.005)
    parser.add_argument("--envmap_lr", type=float, default=0.01)
    parser.add_argument("--envmap_height", type=int, default=128)
    parser.add_argument("--envmap_width", type=int, default=128)
    parser.add_argument("--envmap_init_value", type=float, default=1.5)
    parser.add_argument("--envmap_activation", type=str, default="exp", choices=["exp", "softplus", "none"])
    parser.add_argument("--trace_backend", type=str, default="irgs_adapter", choices=["auto", "irgs", "irgs_adapter", "irgs_native", "open3d", "open3d_mesh"])
    parser.add_argument("--trace_rebuild_every", type=int, default=0)
    parser.add_argument("--trace_voxel_size", type=float, default=0.004)
    parser.add_argument("--trace_sdf_trunc", type=float, default=0.02)
    parser.add_argument("--trace_depth_trunc", type=float, default=0.0)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=False)

    _add_boolean_freeze_flags(parser, "freeze_geometry", default=True, help_text="Freeze geometry-related Gaussian parameters during stage2")
    _add_boolean_freeze_flags(parser, "freeze_color", default=False, help_text="Freeze SH color parameters during stage2")
    _add_boolean_freeze_flags(parser, "trace_mask_background", default=True, help_text="Mask object background while extracting the Open3D trace mesh")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training_stage2_trace(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        stage2_args=args,
    )

    print("\nStage2 Trace training complete.")
