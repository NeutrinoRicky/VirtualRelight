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

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_ir as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.dataset_readers import load_img_rgb
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
import json

def select_views(views, first_k=-1):
    if first_k is None or first_k <= 0 or first_k >= len(views):
        return views, ""
    return views[:first_k], f"_first{first_k}"


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, subset_suffix=""):
    output_name = f"{name}{subset_suffix}"
    if len(views) == 0:
        print(f"No views found for {output_name}, skipping.")
        return

    path_prefix = os.path.join(model_path, output_name, "ours_{}".format(iteration))
    gts_path = os.path.join(path_prefix, "gt")

    keys = ["render", "render_env", "diffuse", "specular", "roughness", "metallic", "base_color", "base_color_linear", "rend_alpha", "rend_normal", "visibility"]

    makedirs(gts_path, exist_ok=True)
    for key in keys:
        makedirs(os.path.join(path_prefix, key), exist_ok=True)

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress ({output_name})")):
        render_pkg = render(view, gaussians, pipeline, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)

        psnr_avg += psnr(image, gt_image).mean().double().item()
        ssim_avg += ssim(image, gt_image).mean().double().item()
        if not args.no_lpips:
            lpips_avg += lpips(image, gt_image, net_type='vgg').mean().double().item()

        if args.no_save:
            continue

        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        for key in keys:
            out = render_pkg[key]
            if 'normal' in key:
                out = (out + 1) / 2
            if 'position' in key:
                out = (out + 1) / 2
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            torchvision.utils.save_image(out * view.mask.float().cuda(), os.path.join(path_prefix, key, '{0:05d}'.format(idx) + ".png"))

    psnr_avg /= len(views)
    ssim_avg /= len(views)
    lpips_avg /= len(views)
    results_dict = {
        "num_views": len(views),
        "psnr_avg": psnr_avg,
        "ssim_avg": ssim_avg,
        "lpips_avg": lpips_avg,
    }
    print("\n[ITER {}] Evaluating {} set: PSNR {} SSIM {} LPIPS {}".format(iteration, output_name, psnr_avg, ssim_avg, lpips_avg))
    with open(os.path.join(model_path, output_name, "nvs_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(model_path, output_name, "nvs_results.json"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, first_k : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.build_bvh()      
        gaussians.env_map.update_pdf()
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            train_views, train_suffix = select_views(scene.getTrainCameras(), first_k)
            render_set(dataset.model_path, "train", scene.loaded_iter, train_views, gaussians, pipeline, background, subset_suffix=train_suffix)

        if not skip_test:
            test_views, test_suffix = select_views(scene.getTestCameras(), first_k)
            render_set(dataset.model_path, "test", scene.loaded_iter, test_views, gaussians, pipeline, background, subset_suffix=test_suffix)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("--first_k", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.first_k)