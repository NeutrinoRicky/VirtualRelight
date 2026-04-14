#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import calculate_loss2
from gaussian_renderer import render_ir
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import rgb_to_srgb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, checkpoint_refgs, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    lr_scale = opt.lr_scale
    opt.position_lr_init *= lr_scale
    opt.opacity_lr *= lr_scale
    opt.scaling_lr *= lr_scale
    opt.rotation_lr *= lr_scale
    
    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    elif checkpoint_refgs:
        (model_params, _) = torch.load(checkpoint_refgs, weights_only=False)
        gaussians.restore_from_refgs(model_params, opt)
        
    gaussians.build_bvh()
    
    if scene.light_rotate:
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    
    while iteration < opt.iterations + 1:
        iter_start.record()

        # gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_ir(viewpoint_cam, gaussians, pipe, background, opt=opt, iteration=iteration, training=True)

        gt_image = viewpoint_cam.original_image.cuda()
        
        total_loss, tb_dict = calculate_loss2(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss"]

        total_loss.backward()
            
        iter_end.record()

        with torch.no_grad():
            
            # Densification
            is_densify = False
            # if iteration < opt.densify_until_iter:
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                          radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
            #     if iteration % opt.densification_interval == 0:
            #         is_densify = True
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
            #                                     size_threshold)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if lr_scale > 0:
                if is_densify:
                    gaussians.build_bvh()
                else:
                    gaussians.update_bvh()

            if iteration % 500 == 0 or iteration == first_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render_ir, pipe, opt, iteration)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            if opt.train_ray:
                mask = render_pkg["mask"]
                ray_rgb_gt = viewpoint_cam.original_image.cuda().permute(1, 2, 0)[mask]
                ray_rgb = render_pkg["ray_rgb"]
                ema_psnr_for_log = 0.4 * psnr(ray_rgb, ray_rgb_gt).mean().double().item() + 0.6 * ema_psnr_for_log
            else:
                image = render_pkg["render"]
                ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_path = model_path + f"/chkpnt{iteration}.pth"
                torch.save((gaussians.capture(), iteration), save_path)
                
            if iteration in testing_iterations:
                psnr_test = evaluate_psnr(scene, render_ir, {"pipe": pipe, "bg_color": background, "opt": opt}, iteration)
        iteration += 1

def set_gaussian_para(gaussians, opt):
    gaussians.init_base_color_value = opt.init_base_color_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.init_roughness_value = opt.init_roughness_value

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        visualization_list = [
            viewpoint_cam.original_image.cuda(),
            render_pkg["render"],
            render_pkg["diffuse"],
            render_pkg["specular"],
            render_pkg["render_sh"],
            render_pkg["base_color_linear"],
            render_pkg["base_color"],
            render_pkg["metallic"].repeat(3, 1, 1),
            render_pkg["roughness"].repeat(3, 1, 1),
            render_pkg["visibility"].repeat(3, 1, 1),
            render_pkg["light_indirect"],
            render_pkg["light_direct"],
            render_pkg["light"],
            render_pkg["rend_alpha"].repeat(3, 1, 1),
            visualize_depth(render_pkg["surf_depth"]),
            render_pkg["rend_normal"] * 0.5 + 0.5,
            render_pkg["surf_normal"] * 0.5 + 0.5,
            error_map,
            render_pkg["render_env"],
        ]
            
        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 1600
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

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs, iteration):    
    eval_path = os.path.join(scene.model_path, "eval", "ours_{}".format(iteration))
    os.makedirs(eval_path, exist_ok=True)
    psnr_test = 0.0
    if len(scene.getTestCameras()):
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            # save_image(image, os.path.join(eval_path, '{0:05d}'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["diffuse"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_diffuse'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["specular"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_specular'.format(idx) + ".png"))
        psnr_test /= len(scene.getTestCameras())
        print("\n[ITER {}] Evaluating test set: PSNR {}".format(iteration, psnr_test))
        with open(os.path.join(eval_path, "psnr.txt"), 'w') as psnr_f:
            psnr_f.write(str(psnr_test))
    torch.cuda.empty_cache()
    return psnr_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,60000,70000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("-c", "--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_checkpoint_refgs", type=str, default = None)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.save_iterations = args.save_iterations + [i for i in range(5000, args.iterations+1, 5000)]
    args.checkpoint_iterations = args.checkpoint_iterations + [i for i in range(5000, args.iterations+1, 5000)]
    
    # Set up output folder
    os.makedirs(args.model_path, exist_ok = True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    
    with open(os.path.join(args.model_path, "cmd.txt"), 'w') as cmd_f:
        cmd_f.write(full_cmd)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    print("Output folder: {}".format(args.model_path))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint_refgs, args.model_path)

    # All done
    print("\nTraining complete.")
