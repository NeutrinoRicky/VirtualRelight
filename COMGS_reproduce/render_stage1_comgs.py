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
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage1 COMGS training-view rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    print("Rendering stage1 train views for " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0:
        print("No training cameras found, skip rendering.")
        raise SystemExit(0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
    os.makedirs(train_dir, exist_ok=True)

    gauss_extractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    print("export training images ...")
    gauss_extractor.reconstruction(train_cameras)
    gauss_extractor.export_image(train_dir)
    print("training images saved at {}".format(train_dir))
