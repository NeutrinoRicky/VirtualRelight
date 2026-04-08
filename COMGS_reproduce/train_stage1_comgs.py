import os
import sys
import uuid
from random import randint

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render, render_multitarget, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from utils.loss_utils import l1_loss
from utils.losses_comgs_stage1 import compute_stage1_loss

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


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


def save_stage1_gbuffers(output_root, iteration, view_name, render_pkg):
    os.makedirs(output_root, exist_ok=True)
    prefix = f"iter_{iteration:06d}_{view_name}"

    rgb = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
    weight = torch.clamp(torch.nan_to_num(render_pkg["weight"].detach(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    valid = weight > 1e-4
    depth_unbiased_vis = _normalize_single_channel_for_vis(render_pkg["depth_unbiased"], valid_mask=valid)

    normal_vis = torch.clamp(render_pkg["normal"].detach() * 0.5 + 0.5, 0.0, 1.0)
    albedo = torch.clamp(render_pkg["albedo"].detach(), 0.0, 1.0)
    roughness = torch.clamp(render_pkg["roughness"].detach(), 0.0, 1.0)
    metallic = torch.clamp(render_pkg["metallic"].detach(), 0.0, 1.0)

    torchvision.utils.save_image(rgb, os.path.join(output_root, f"{prefix}_rgb.png"))
    torchvision.utils.save_image(weight, os.path.join(output_root, f"{prefix}_weight.png"))
    torchvision.utils.save_image(depth_unbiased_vis, os.path.join(output_root, f"{prefix}_depth_unbiased.png"))
    torchvision.utils.save_image(normal_vis, os.path.join(output_root, f"{prefix}_normal.png"))
    torchvision.utils.save_image(albedo, os.path.join(output_root, f"{prefix}_albedo.png"))
    torchvision.utils.save_image(roughness, os.path.join(output_root, f"{prefix}_roughness.png"))
    torchvision.utils.save_image(metallic, os.path.join(output_root, f"{prefix}_metallic.png"))


@torch.no_grad()
def training_report_stage1(tb_writer, iteration, elapsed, testing_iterations, scene: Scene, render_func, render_args):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {'name': 'test', 'cameras': scene.getTestCameras()},
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
    )

    for config in validation_configs:
        if not config['cameras'] or len(config['cameras']) == 0:
            continue

        l1_test = 0.0
        psnr_test = 0.0

        for idx, viewpoint in enumerate(config['cameras']):
            render_pkg = render_func(viewpoint, scene.gaussians, *render_args)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if tb_writer and (idx < 5):
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                depth_vis = _normalize_single_channel_for_vis(render_pkg["depth_unbiased"], valid_mask=(render_pkg["weight"] > 1e-4))
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/depth_unbiased", depth_vis[None], global_step=iteration)
                normal_vis = torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0)
                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/normal", normal_vis[None], global_step=iteration)
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_train_for_best(scene: Scene, gaussians: GaussianModel, pipe, background):
    cameras = scene.getTrainCameras()
    if cameras is None or len(cameras) == 0:
        return None, None

    l1_avg = 0.0
    psnr_avg = 0.0

    for viewpoint in cameras:
        render_pkg = render_multitarget(viewpoint, gaussians, pipe, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        l1_avg += l1_loss(image, gt_image).mean().double()
        psnr_avg += psnr(image, gt_image).mean().double()

    l1_avg /= len(cameras)
    psnr_avg /= len(cameras)
    torch.cuda.empty_cache()
    return float(l1_avg), float(psnr_avg)


def training_stage1(dataset, opt, pipe, stage1_args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_total_for_log = 0.0
    ema_rgb_for_log = 0.0
    ema_d2n_for_log = 0.0
    ema_mask_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Stage1 training")
    first_iter += 1

    gbuffer_dir = os.path.join(dataset.model_path, "stage1_gbuffers")

    best_train_psnr = float("-inf")
    best_ckpt_path = os.path.join(scene.model_path, "chkpnt_best.pth")
    best_info_path = os.path.join(scene.model_path, "best_train_metrics.txt")
    warned_missing_mask = False

    if stage1_args.use_mask_loss:
        train_cams = scene.getTrainCameras()
        cams_with_mask = sum(1 for cam in train_cams if getattr(cam, "gt_alpha_mask", None) is not None)
        print(f"[Stage1] Mask coverage in train cameras: {cams_with_mask}/{len(train_cams)}")
        if cams_with_mask == 0:
            print("[Stage1][Warn] --use_mask_loss is enabled, but no training camera has gt_alpha_mask. Mask loss will stay 0.")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_multitarget(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        if stage1_args.use_mask_loss and (getattr(viewpoint_cam, "gt_alpha_mask", None) is None) and (not warned_missing_mask):
            print("[Stage1][Warn] Current training camera has no gt_alpha_mask. Check masks directory and naming.")
            warned_missing_mask = True

        total_loss, loss_stats, _ = compute_stage1_loss(
            render_pkg=render_pkg,
            gt_rgb=gt_image,
            viewpoint_camera=viewpoint_cam,
            lambda_d2n=stage1_args.lambda_d2n,
            lambda_mask=stage1_args.lambda_mask,
            use_mask_loss=stage1_args.use_mask_loss,
        )

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_total_for_log = 0.4 * total_loss.item() + 0.6 * ema_total_for_log
            ema_rgb_for_log = 0.4 * loss_stats["loss_rgb"].item() + 0.6 * ema_rgb_for_log
            ema_d2n_for_log = 0.4 * loss_stats["loss_d2n"].item() + 0.6 * ema_d2n_for_log
            ema_mask_for_log = 0.4 * loss_stats["loss_mask"].item() + 0.6 * ema_mask_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "total": f"{ema_total_for_log:.5f}",
                    "rgb": f"{ema_rgb_for_log:.5f}",
                    "d2n": f"{ema_d2n_for_log:.5f}",
                    "mask": f"{ema_mask_for_log:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_stage1/loss_total', ema_total_for_log, iteration)
                tb_writer.add_scalar('train_stage1/loss_rgb', ema_rgb_for_log, iteration)
                tb_writer.add_scalar('train_stage1/loss_d2n', ema_d2n_for_log, iteration)
                tb_writer.add_scalar('train_stage1/loss_mask', ema_mask_for_log, iteration)

            training_report_stage1(
                tb_writer=tb_writer,
                iteration=iteration,
                elapsed=iter_start.elapsed_time(iter_end),
                testing_iterations=testing_iterations,
                scene=scene,
                render_func=render_multitarget,
                render_args=(pipe, background),
            )

            if stage1_args.save_gbuffers_every > 0 and iteration % stage1_args.save_gbuffers_every == 0:
                save_stage1_gbuffers(gbuffer_dir, iteration, viewpoint_cam.image_name, render_pkg)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if stage1_args.eval_train_every > 0 and iteration % stage1_args.eval_train_every == 0:
                train_l1, train_psnr = evaluate_train_for_best(scene, gaussians, pipe, background)
                if train_psnr is not None:
                    print(f"\n[ITER {iteration}] Evaluating train: L1 {train_l1:.6f} PSNR {train_psnr:.6f}")
                    if tb_writer is not None:
                        tb_writer.add_scalar('train_eval/l1', train_l1, iteration)
                        tb_writer.add_scalar('train_eval/psnr', train_psnr, iteration)

                    if train_psnr > best_train_psnr:
                        best_train_psnr = train_psnr
                        torch.save((gaussians.capture(), iteration), best_ckpt_path)
                        with open(best_info_path, "w") as f:
                            f.write(f"iter={iteration}\n")
                            f.write(f"psnr={train_psnr:.6f}\n")
                            f.write(f"l1={train_l1:.6f}\n")
                        print(f"[ITER {iteration}] New best train PSNR={train_psnr:.6f}. Saved best checkpoint to {best_ckpt_path}")

        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        gui_render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(gui_render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                        )
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_total_for_log,
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception:
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def apply_tnt_eval_consistency(args):
    args.depth_ratio = 1.0
    args.resolution = 2
    args.test_iterations = [-1]
    args.quiet = True
    print("[TNT-Consistency] Applied tnt_eval.py settings: --depth_ratio 1.0 -r 2 --test_iterations -1 --quiet")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage1 COMGS training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--lambda_d2n", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.05)
    parser.add_argument("--use_mask_loss", action="store_true", default=False)
    parser.add_argument("--save_gbuffers_every", type=int, default=1000)
    parser.add_argument("--eval_train_every", type=int, default=1000,
                        help="Evaluate full train set every N iterations and update chkpnt_best.pth when PSNR improves")
    parser.add_argument("--match_tnt_eval", action="store_true", default=False,
                        help="Force TNT settings from scripts/tnt_eval.py: --depth_ratio 1.0 -r 2 --test_iterations -1 --quiet")

    args = parser.parse_args(sys.argv[1:])

    if args.match_tnt_eval:
        apply_tnt_eval_consistency(args)

    if args.lambda_mask > 0.0 and not args.use_mask_loss:
        print("[Stage1][Warn] --lambda_mask > 0 but --use_mask_loss is not set. Mask loss is disabled and will stay 0.")
    if args.use_mask_loss and args.lambda_mask <= 0.0:
        print("[Stage1][Warn] --use_mask_loss is set but --lambda_mask <= 0. Mask loss contributes 0 to total loss.")

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training_stage1(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        stage1_args=args,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
    )

    print("\nStage1 training complete.")
