import json
import os
from gaussian_renderer import render_ir
import numpy as np
import torch
import torch.nn.functional as F
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.light import EnvMap, EnvLight
from utils.graphics_utils import focal2fov, fov2focal, rgb_to_srgb, srgb_to_rgb
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from scene.dataset_readers import load_img_rgb
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def get_mae(gt_normal, normal):
    mae = (gt_normal*normal).sum(0).clamp(-1, 1).arccos().mean() * 180 / np.pi
    return mae
    
if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    gaussians = GaussianModel(3)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
    gaussians.build_bvh()
        
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    if args.albedo_rescale == 0:
        base_color_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    else:
        with open(os.path.join(args.model_path, "albedo_scale.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict[str(args.albedo_rescale)], dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
        "training": False,
        "relight": False,
        "base_color_scale": base_color_scale,
        "material_only": True,
    }
    
    psnr_albedo = 0.0
    ssim_albedo = 0.0
    lpips_albedo = 0.0
    mae_normal = 0.0
    results_dict = {}
    normal_bg = torch.tensor([0.0, 0.0, 1.0], device='cuda')
    
    for idx, frame in enumerate(tqdm(frames)):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        albedo_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "albedo.png"))
        gt_albedo_np = load_img_rgb(albedo_path)
        gt_albedo = torch.from_numpy(gt_albedo_np[..., :3] * gt_albedo_np[..., 3:4]).permute(2, 0, 1).float().cuda()
        mask = torch.from_numpy(gt_albedo_np[..., 3:]).permute(2, 0, 1).float().cuda()
        
        H = gt_albedo.shape[1]
        W = gt_albedo.shape[2]
        fovy = focal2fov(fov2focal(fovx, W), H)

        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy,
                            image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

        with torch.no_grad():
            render_pkg = render_ir(viewpoint_camera=custom_cam, **render_kwargs)

        render_pkg['base_color_linear'] = render_pkg['base_color_linear'] * mask
        gt_albedo = gt_albedo * mask

        psnr_albedo += psnr(render_pkg['base_color_linear'], gt_albedo).mean().double().item()
        ssim_albedo += ssim(render_pkg['base_color_linear'], gt_albedo).mean().double().item()
        if not args.no_lpips:
            lpips_albedo += lpips(render_pkg['base_color_linear'], gt_albedo, net_type='vgg').mean().double().item()
        
        normal = render_pkg['rend_normal']
        alpha = render_pkg['rend_alpha']
        normal = normal * alpha + normal_bg[:, None, None] * (1.0 - alpha)
        normal = F.normalize(normal, dim=0)
        
        normal_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "normal.png"))
        gt_normal_img = torch.from_numpy(load_img_rgb(normal_path)[..., :3]).float().cuda().permute(2, 0, 1)
        gt_normal = (gt_normal_img - 0.5) * 2.0
        gt_normal = gt_normal * mask + normal_bg[:, None, None] * (1.0 - mask)
        gt_normal = F.normalize(gt_normal, dim=0)
        mae_normal += get_mae(gt_normal, normal).mean().double().item()
                
    psnr_albedo /= len(frames)
    ssim_albedo /= len(frames)
    lpips_albedo /= len(frames)
    mae_normal /= len(frames)
        
    results_dict["psnr_albedo_avg"] = psnr_albedo
    results_dict["ssim_albedo_avg"] = ssim_albedo
    results_dict["lpips_albedo_avg"] = lpips_albedo
    results_dict["mae_normal_avg"] = mae_normal
    print("\nEvaluating AVG: PSNR_ALBEDO {: .2f} SSIM_ALBEDO {: .3f} LPIPS_ALBEDO {: .3f}".format(psnr_albedo, ssim_albedo, lpips_albedo))
    print("\nEvaluating AVG: MAE_NORMAL {: .2f}".format(mae_normal))
    with open(os.path.join(args.model_path, "material_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(args.model_path, "material_results.json"))
    