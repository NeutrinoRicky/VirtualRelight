import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_sop_gbuffer
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from train_stage2_sop import Stage2SOPState, set_gaussian_para, _load_sop_payload, _resolve_sop_init_path
from utils.general_utils import safe_state
from utils.graphics_utils import rgb_to_srgb
from utils.image_utils import psnr, visualize_depth
from utils.loss_utils import ssim
from utils.losses_comgs_stage2_sop import compute_stage2_sop_loss


def select_views(views, first_k=-1):
    if first_k is None or first_k <= 0 or first_k >= len(views):
        return views, ""
    return views[:first_k], f"_first{first_k}"


def _repeat_to_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def _view_output_mask(view, ref: torch.Tensor) -> Optional[torch.Tensor]:
    mask = getattr(view, "mask", None)
    if mask is None:
        mask = getattr(view, "gt_alpha_mask", None)
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=ref.device, dtype=ref.dtype)
    return _repeat_to_rgb(mask)


def _resolve_stage2_checkpoint(args) -> Optional[Path]:
    if getattr(args, "start_checkpoint", ""):
        return Path(args.start_checkpoint)

    model_path = Path(args.model_path)
    if args.iteration is not None and int(args.iteration) > 0:
        iter_ckpt = model_path / f"object_step2_sop_iter_{int(args.iteration):06d}.ckpt"
        if iter_ckpt.exists():
            return iter_ckpt

    final_ckpt = model_path / "object_step2_sop.ckpt"
    if final_ckpt.exists():
        return final_ckpt
    return None


def _load_render_state(gaussians: GaussianModel, args):
    stage2_ckpt = _resolve_stage2_checkpoint(args)
    if stage2_ckpt is not None:
        if not stage2_ckpt.exists():
            raise FileNotFoundError(f"Stage2 SOP checkpoint not found: {stage2_ckpt}")
        payload = torch.load(stage2_ckpt, map_location="cuda")
        if not isinstance(payload, dict) or payload.get("format") != "irgs_stage2_sop_v1":
            raise RuntimeError(f"{stage2_ckpt} is not a valid Stage2 SOP checkpoint")

        gaussians.restore(payload["gaussians"], None)
        sop_state = Stage2SOPState(payload["sop"])
        source_info = dict(payload.get("source_info", {}))
        source_info["stage2_checkpoint"] = str(stage2_ckpt)
        loaded_iter = int(payload.get("iteration", 0))
        return loaded_iter, sop_state, source_info

    if not getattr(args, "start_checkpoint_refgs", ""):
        raise RuntimeError(
            "render_sop.py needs a Stage2 SOP checkpoint under --model_path "
            "(object_step2_sop.ckpt / object_step2_sop_iter_xxxxxx.ckpt), "
            "or an explicit --start_checkpoint, or --start_checkpoint_refgs with --sop_init."
        )

    refgs_payload = torch.load(args.start_checkpoint_refgs, map_location="cuda")
    if not isinstance(refgs_payload, (tuple, list)) or len(refgs_payload) < 1:
        raise RuntimeError(f"Unsupported refgs checkpoint payload: {type(refgs_payload).__name__}")

    gaussians.restore_from_refgs(refgs_payload[0], None)
    sop_path = _resolve_sop_init_path(args.sop_init, args.model_path)
    sop_state = Stage2SOPState(_load_sop_payload(sop_path))
    source_info = {
        "refgs_checkpoint": args.start_checkpoint_refgs,
        "sop_init": str(sop_path),
    }
    fallback_iter = int(args.iteration) if int(args.iteration) > 0 else 0
    return fallback_iter, sop_state, source_info


@torch.no_grad()
def render_stage2_sop_view(viewpoint_cam, gaussians, background, pipe, opt, sop_state, args, iteration):
    render_pkg = render_sop_gbuffer(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        opt=opt,
        iteration=iteration,
        training=False,
    )
    gt_rgb = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)
    _, stats, aux = compute_stage2_sop_loss(
        render_pkg=render_pkg,
        gt_rgb=gt_rgb,
        viewpoint_camera=viewpoint_cam,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        opt=opt,
        training=False,
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
        max_shading_points=0,
        sop_query_radius=args.sop_query_radius,
        sop_query_topk=args.sop_query_topk,
        sop_query_chunk_size=args.sop_query_chunk_size,
        randomized_samples=False,
        cuda_mem_debug=None,
    )

    outputs = {
        "render": torch.clamp(aux["pbr_render"], 0.0, 1.0),
        "albedo": torch.clamp(render_pkg["albedo"], 0.0, 1.0),
        "roughness": _repeat_to_rgb(torch.clamp(render_pkg["roughness"], 0.0, 1.0)),
        "metallic": _repeat_to_rgb(torch.clamp(render_pkg["metallic"], 0.0, 1.0)),
        "weight": _repeat_to_rgb(torch.clamp(render_pkg["weight"], 0.0, 1.0)),
        "depth": visualize_depth(render_pkg["depth_unbiased"][None]),
        "normal": torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0),
        "depth_normal": torch.clamp(aux["depth_normal"] * 0.5 + 0.5, 0.0, 1.0),
        "query_selection": _repeat_to_rgb(torch.clamp(aux["query_selection"], 0.0, 1.0)),
        "query_occlusion": _repeat_to_rgb(torch.clamp(aux["query_occlusion"], 0.0, 1.0)),
        "query_direct": torch.clamp(aux["query_direct"], 0.0, 1.0),
        "query_indirect": torch.clamp(aux["query_indirect"], 0.0, 1.0),
        "diffuse": torch.clamp(aux["pbr_diffuse"], 0.0, 1.0),
        "specular": torch.clamp(aux["pbr_specular"], 0.0, 1.0),
    }
    return outputs, gt_rgb, stats


def render_set(model_path, name, iteration, views, gaussians, sop_state, pipeline, opt, background, args, source_info, subset_suffix=""):
    output_name = f"{name}{subset_suffix}"
    if len(views) == 0:
        print(f"No views found for {output_name}, skipping.")
        return

    output_root = os.path.join(model_path, output_name)
    path_prefix = os.path.join(model_path, output_name, f"ours_{iteration}")
    gts_path = os.path.join(path_prefix, "gt")
    keys = [
        "render",
        "albedo",
        "roughness",
        "metallic",
        "weight",
        "depth",
        "normal",
        "depth_normal",
        "query_selection",
        "query_occlusion",
        "query_direct",
        "query_indirect",
        "diffuse",
        "specular",
    ]

    os.makedirs(output_root, exist_ok=True)
    if not args.no_save:
        os.makedirs(gts_path, exist_ok=True)
        for key in keys:
            os.makedirs(os.path.join(path_prefix, key), exist_ok=True)
        env_dict = gaussians.render_env_map()
        env_image = rgb_to_srgb(env_dict["env"].permute(2, 0, 1))
        torchvision.utils.save_image(env_image, os.path.join(path_prefix, "env.png"))

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0

    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress ({output_name})")):
        outputs, gt_image, _stats = render_stage2_sop_view(
            viewpoint_cam=view,
            gaussians=gaussians,
            background=background,
            pipe=pipeline,
            opt=opt,
            sop_state=sop_state,
            args=args,
            iteration=iteration,
        )

        pred_image = outputs["render"]
        psnr_avg += psnr(pred_image, gt_image).mean().double().item()
        ssim_avg += ssim(pred_image, gt_image).mean().double().item()
        if not args.no_lpips:
            lpips_avg += lpips(pred_image, gt_image, net_type="vgg").mean().double().item()

        if args.no_save:
            continue

        save_mask = _view_output_mask(view, gt_image)
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, f"{idx:05d}.png"))
        for key in keys:
            out = outputs[key]
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            if save_mask is not None:
                out = out * save_mask
            torchvision.utils.save_image(out, os.path.join(path_prefix, key, f"{idx:05d}.png"))

    psnr_avg /= len(views)
    ssim_avg /= len(views)
    if not args.no_lpips:
        lpips_avg /= len(views)

    results_dict = {
        "num_views": len(views),
        "iteration": int(iteration),
        "psnr_avg": psnr_avg,
        "ssim_avg": ssim_avg,
        "lpips_avg": lpips_avg,
        "lpips_enabled": not args.no_lpips,
        "source_info": source_info,
    }
    print(f"\n[ITER {iteration}] Evaluating {output_name} set: PSNR {psnr_avg} SSIM {ssim_avg} LPIPS {lpips_avg}")
    with open(os.path.join(model_path, output_name, "nvs_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(model_path, output_name, "nvs_results.json"))


def render_sets(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        set_gaussian_para(gaussians, opt)

        point_cloud_root = os.path.join(dataset.model_path, "point_cloud")
        load_iteration = -1 if os.path.isdir(point_cloud_root) else None
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

        render_iteration, sop_state, source_info = _load_render_state(gaussians, args)

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

        if gaussians.env_map is not None and hasattr(gaussians.env_map, "update_pdf"):
            gaussians.env_map.update_pdf()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            train_views, train_suffix = select_views(scene.getTrainCameras(), args.first_k)
            render_set(
                dataset.model_path,
                "train_sop",
                render_iteration,
                train_views,
                gaussians,
                sop_state,
                pipeline,
                opt,
                background,
                args,
                source_info,
                subset_suffix=train_suffix,
            )

        if not args.skip_test:
            test_views, test_suffix = select_views(scene.getTestCameras(), args.first_k)
            render_set(
                dataset.model_path,
                "test_sop",
                render_iteration,
                test_views,
                gaussians,
                sop_state,
                pipeline,
                opt,
                background,
                args,
                source_info,
                subset_suffix=test_suffix,
            )


def _build_parser():
    parser = ArgumentParser(description="Stage2 SOP rendering parameters")
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", default=False, action="store_true")
    parser.add_argument("--no_lpips", default=False, action="store_true")
    parser.add_argument("--first_k", default=-1, type=int)

    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--start_checkpoint_refgs", type=str, default="")
    parser.add_argument("--sop_init", type=str, default="")

    parser.add_argument("--lambda_lam", type=float, default=0.001)
    parser.add_argument("--lambda_sops", type=float, default=0.0)
    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--num_shading_samples", type=int, default=128)
    parser.add_argument("--max_shading_points", type=int, default=4096)
    parser.add_argument("--sop_query_radius", type=float, default=0.0)
    parser.add_argument("--sop_query_topk", type=int, default=4)
    parser.add_argument("--sop_query_chunk_size", type=int, default=1024)
    parser.add_argument("--disable_sample_jitter", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = get_combined_args(parser)
    print("Rendering Stage2 SOP from " + args.model_path)

    if not getattr(args, "model_path", ""):
        raise RuntimeError("render_sop.py requires --model_path/-m.")
    if not getattr(args, "source_path", ""):
        raise RuntimeError("render_sop.py requires --source_path/-s, or a cfg_args under --model_path.")
    if not torch.cuda.is_available():
        raise RuntimeError("render_sop.py currently requires CUDA.")

    safe_state(args.quiet)

    dataset = ModelParams(ArgumentParser(), sentinel=True).extract(args)
    opt = OptimizationParams(ArgumentParser()).extract(args)
    pipe = PipelineParams(ArgumentParser()).extract(args)
    render_sets(dataset, opt, pipe, args)
