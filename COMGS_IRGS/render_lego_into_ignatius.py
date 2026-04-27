import json
import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torchvision
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from torch import nn
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent
if not Path("assets/bsdf_256_256.bin").exists() and (REPO_DIR / "assets/bsdf_256_256.bin").exists():
    os.chdir(REPO_DIR)

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from utils.image_utils import visualize_depth
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh


def _ensure_repo_cwd() -> None:
    """GaussianModel loads assets with a repo-relative path."""
    if not Path("assets/bsdf_256_256.bin").exists():
        os.chdir(REPO_DIR)


def _as_parameter(x: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(x.detach().cuda().requires_grad_(False))


def _make_screenspace_points(pc: GaussianModel) -> torch.Tensor:
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def _resolve_covariance(viewpoint_camera: Camera, pc: GaussianModel, pipe: Namespace, scaling_modifier: float):
    scales = None
    rotations = None
    cov3D_precomp = None
    if getattr(pipe, "compute_cov3D_python", False):
        splat2world = pc.get_covariance(scaling_modifier)
        width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor(
            [
                [width / 2, 0, 0, (width - 1) / 2],
                [0, height / 2, 0, (height - 1) / 2],
                [0, 0, far - near, near],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        ).T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]
        ).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    return scales, rotations, cov3D_precomp


def _resolve_color_inputs(viewpoint_camera: Camera, pc: GaussianModel, pipe: Namespace, override_color):
    shs = None
    colors_precomp = None
    if override_color is None:
        if getattr(pipe, "convert_SHs_python", False):
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    return shs, colors_precomp


def _compute_regularization_outputs(allmap: torch.Tensor, viewpoint_camera: Camera, pipe: Namespace) -> dict:
    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median / render_alpha, 0, 0)

    render_depth_expected = allmap[0:1] / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_dist = allmap[6:7]
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal = surf_normal * render_alpha.detach()
    return {
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    }


def _rasterize_scene(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: Namespace,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color=None,
) -> dict:
    screenspace_points = _make_screenspace_points(pc)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    scales, rotations, cov3D_precomp = _resolve_covariance(viewpoint_camera, pc, pipe, scaling_modifier)
    shs, colors_precomp = _resolve_color_inputs(viewpoint_camera, pc, pipe, override_color)

    _contrib, rendered_image, _rendered_features, radii, allmap = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "rendered_image": rendered_image,
        "allmap": allmap,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def _load_legacy_comgs_checkpoint(path: Path, gaussians: GaussianModel) -> int:
    payload = torch.load(path, map_location="cuda")
    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported checkpoint payload at {path}")

    model_args = payload[0]
    loaded_iter = int(payload[1]) if len(payload) > 1 else 0
    if len(model_args) != 15:
        raise RuntimeError(
            f"{path} does not look like a legacy COMGS checkpoint; expected 15 model fields, got {len(model_args)}"
        )

    (
        gaussians.active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        albedo,
        roughness,
        metallic,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        _opt_dict,
        gaussians.spatial_lr_scale,
    ) = model_args

    gaussians._xyz = _as_parameter(xyz)
    gaussians._features_dc = _as_parameter(features_dc)
    gaussians._features_rest = _as_parameter(features_rest)
    gaussians._scaling = _as_parameter(scaling)
    gaussians._rotation = _as_parameter(rotation)
    gaussians._opacity = _as_parameter(opacity)
    gaussians._base_color = _as_parameter(albedo)
    gaussians._roughness = _as_parameter(roughness)
    gaussians._metallic = _as_parameter(metallic)
    gaussians.max_radii2D = max_radii2D.detach().cuda()
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().cuda()
    gaussians.denom = denom.detach().cuda()
    return loaded_iter


def load_ignatius_gaussians(checkpoint_path: Path, sh_degree: int) -> Tuple[GaussianModel, int]:
    gaussians = GaussianModel(sh_degree)
    loaded_iter = _load_legacy_comgs_checkpoint(checkpoint_path, gaussians)
    print(f"[load] Ignatius: {checkpoint_path} ({gaussians.get_xyz.shape[0]} gaussians, iter={loaded_iter})")
    return gaussians, loaded_iter


def load_refgs_as_irgs_gaussians(checkpoint_path: Path, sh_degree: int) -> Tuple[GaussianModel, int]:
    gaussians = GaussianModel(sh_degree)
    payload = torch.load(checkpoint_path, map_location="cuda")
    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise RuntimeError(f"Unsupported refgs checkpoint payload at {checkpoint_path}")
    gaussians.restore_from_refgs(payload[0], None)
    loaded_iter = int(payload[1]) if len(payload) > 1 else 0
    print(f"[load] lego refgs: {checkpoint_path} ({gaussians.get_xyz.shape[0]} gaussians, iter={loaded_iter})")
    return gaussians, loaded_iter


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported Euler axis: {axis}")


def blender_euler_matrix(euler_deg: Iterable[float], order: str = "XYZ") -> np.ndarray:
    angles = [math.radians(float(v)) for v in euler_deg]
    if len(order) != 3 or set(order.upper()) != {"X", "Y", "Z"}:
        raise ValueError(f"Euler order must be a permutation of XYZ, got {order}")
    order = order.upper()
    matrix = np.eye(3, dtype=np.float32)
    for axis, angle in zip(order, angles):
        matrix = _rotation_matrix(axis, angle) @ matrix
    return matrix


def make_transform(translation: Iterable[float], euler_deg: Iterable[float], scale: float, order: str) -> torch.Tensor:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = blender_euler_matrix(euler_deg, order=order) * float(scale)
    transform[:3, 3] = np.asarray(list(translation), dtype=np.float32)
    return torch.from_numpy(transform).cuda()


def apply_gaussian_transform(
    gaussians: GaussianModel,
    translation: Iterable[float],
    euler_deg: Iterable[float],
    scale: float,
    order: str,
) -> torch.Tensor:
    transform = make_transform(translation, euler_deg, scale, order)
    gaussians.set_transform(transform=transform)
    print("[transform] lego -> Ignatius")
    print(transform.detach().cpu().numpy())
    return transform


@torch.no_grad()
def apply_post_transform_scale(gaussians: GaussianModel, post_scale: float, center_mode: str) -> torch.Tensor:
    post_scale = float(post_scale)
    if post_scale <= 0.0:
        raise ValueError(f"--post_scale must be positive, got {post_scale}")
    xyz = gaussians.get_xyz.detach()
    if center_mode == "bbox":
        center = 0.5 * (xyz.amin(dim=0) + xyz.amax(dim=0))
    elif center_mode == "mean":
        center = xyz.mean(dim=0)
    elif center_mode == "origin":
        center = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)
    else:
        raise ValueError(f"Unsupported --post_scale_center {center_mode}")

    gaussians._xyz.data = center + (gaussians._xyz.data - center) * post_scale
    gaussians._scaling.data = gaussians.scaling_inverse_activation(gaussians.get_scaling * post_scale)
    print(f"[post-scale] scale={post_scale} center_mode={center_mode} center={center.detach().cpu().tolist()}")
    return center


def load_camera_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError(f"No cameras found in {path}")
    return cameras


def camera_from_json(cam_json: dict, max_width: int, scale: float) -> Camera:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(cam_json["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(cam_json["position"], dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]

    width = int(cam_json["width"])
    height = int(cam_json["height"])
    render_scale = float(scale)
    if max_width > 0 and width * render_scale > max_width:
        render_scale = float(max_width) / float(width)
    out_w = max(2, int(round(width * render_scale)))
    out_h = max(2, int(round(height * render_scale)))

    fx = float(cam_json["fx"]) * render_scale
    fy = float(cam_json["fy"]) * render_scale
    fovx = focal2fov(fx, out_w)
    fovy = focal2fov(fy, out_h)
    K = np.array(
        [
            [fx, 0.0, out_w / 2.0],
            [0.0, fy, out_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image = torch.zeros(3, out_h, out_w)
    return Camera(
        colmap_id=int(cam_json.get("id", 0)),
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        gt_alpha_mask=None,
        image_name=str(cam_json.get("img_name", cam_json.get("id", "view"))),
        uid=int(cam_json.get("id", 0)),
        data_device="cuda",
        HWK=(out_h, out_w, K),
    )


def select_camera_jsons(cameras: List[dict], first_k: int, camera_ids: List[int]) -> List[dict]:
    if camera_ids:
        id_set = set(camera_ids)
        cameras = [cam for cam in cameras if int(cam.get("id", -1)) in id_set]
    if first_k > 0:
        cameras = cameras[:first_k]
    return cameras


@torch.no_grad()
def render_sh_gbuffer(view: Camera, gaussians: GaussianModel, pipe: Namespace) -> dict:
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    raster_out = _rasterize_scene(
        viewpoint_camera=view,
        pc=gaussians,
        pipe=pipe,
        bg_color=black,
    )
    out = {
        "rgb_premul": torch.clamp(raster_out["rendered_image"], 0.0, 1.0),
        "viewspace_points": raster_out["viewspace_points"],
        "visibility_filter": raster_out["visibility_filter"],
        "radii": raster_out["radii"],
    }
    out.update(_compute_regularization_outputs(raster_out["allmap"], view, pipe))
    out["alpha"] = torch.clamp(torch.nan_to_num(out["rend_alpha"], nan=0.0), 0.0, 1.0)
    out["depth"] = torch.nan_to_num(out["surf_depth"], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def composite_by_depth(
    bg_pkg: dict,
    fg_pkg: dict,
    background: torch.Tensor,
    alpha_thresh: float,
    depth_bias: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bg_alpha = bg_pkg["alpha"]
    fg_alpha = fg_pkg["alpha"]
    bg_depth = bg_pkg["depth"]
    fg_depth = fg_pkg["depth"]

    bg_final = torch.clamp(bg_pkg["rgb_premul"] + background[:, None, None] * (1.0 - bg_alpha), 0.0, 1.0)
    fg_only = torch.clamp(fg_pkg["rgb_premul"] + background[:, None, None] * (1.0 - fg_alpha), 0.0, 1.0)

    bg_valid = (bg_alpha > alpha_thresh) & (bg_depth > 0.0)
    fg_valid = (fg_alpha > alpha_thresh) & (fg_depth > 0.0)
    fg_front = fg_valid & ((~bg_valid) | (fg_depth <= bg_depth + depth_bias))

    fg_over_bg = torch.clamp(fg_pkg["rgb_premul"] + bg_final * (1.0 - fg_alpha), 0.0, 1.0)
    composite = torch.where(fg_front.expand_as(fg_over_bg), fg_over_bg, bg_final)
    return composite, fg_only, fg_front.float()


def save_debug_outputs(
    output_dir: Path,
    stem: str,
    composite: torch.Tensor,
    ignatius_pkg: dict,
    lego_pkg: dict,
    lego_only: torch.Tensor,
    front_mask: torch.Tensor,
    background: torch.Tensor,
) -> None:
    torchvision.utils.save_image(composite, output_dir / "composite" / f"{stem}.png")
    torchvision.utils.save_image(
        torch.clamp(ignatius_pkg["rgb_premul"] + background[:, None, None] * (1.0 - ignatius_pkg["alpha"]), 0.0, 1.0),
        output_dir / "ignatius" / f"{stem}.png",
    )
    torchvision.utils.save_image(lego_only, output_dir / "lego" / f"{stem}.png")
    torchvision.utils.save_image(ignatius_pkg["alpha"].repeat(3, 1, 1), output_dir / "ignatius_alpha" / f"{stem}.png")
    torchvision.utils.save_image(lego_pkg["alpha"].repeat(3, 1, 1), output_dir / "lego_alpha" / f"{stem}.png")
    torchvision.utils.save_image(front_mask.repeat(3, 1, 1), output_dir / "lego_front_mask" / f"{stem}.png")
    torchvision.utils.save_image(visualize_depth(ignatius_pkg["depth"]), output_dir / "ignatius_depth" / f"{stem}.png")
    torchvision.utils.save_image(visualize_depth(lego_pkg["depth"]), output_dir / "lego_depth" / f"{stem}.png")


def make_output_dirs(output_dir: Path) -> None:
    for name in [
        "composite",
        "ignatius",
        "lego",
        "ignatius_alpha",
        "lego_alpha",
        "lego_front_mask",
        "ignatius_depth",
        "lego_depth",
    ]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = ArgumentParser(description="Render a transformed lego refgs model into Ignatius cameras and depth-composite it.")
    parser.add_argument(
        "--lego_checkpoint",
        default=str(REPO_DIR / "outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth"),
    )
    parser.add_argument(
        "--ignatius_checkpoint",
        default=str(REPO_DIR / "outputs/tnt/Ignatius/chkpnt_best.pth"),
    )
    parser.add_argument(
        "--ignatius_cameras",
        default=str(REPO_DIR / "outputs/tnt/Ignatius/cameras.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_DIR / "outputs/composite/lego_into_ignatius"),
    )
    parser.add_argument("--translation", nargs=3, type=float, default=[-2.37, 1.451, -0.45])
    parser.add_argument("--rotation_deg", nargs=3, type=float, default=[83.57, 190.7, 0.254])
    parser.add_argument("--rotation_order", default="XYZ")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--post_scale",
        type=float,
        default=1.0,
        help="Uniformly scale lego after the translation/rotation transform, keeping its transformed center fixed.",
    )
    parser.add_argument(
        "--post_scale_center",
        choices=["bbox", "mean", "origin"],
        default="bbox",
        help="Pivot used by --post_scale in the transformed Ignatius coordinate system.",
    )
    parser.add_argument("--max_width", type=int, default=1600)
    parser.add_argument("--resolution_scale", type=float, default=1.0)
    parser.add_argument("--first_k", type=int, default=-1)
    parser.add_argument("--camera_ids", default="", help="Comma-separated camera ids to render, e.g. 0,10,42")
    parser.add_argument("--alpha_thresh", type=float, default=1e-3)
    parser.add_argument("--depth_bias", type=float, default=0.0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _ensure_repo_cwd()
    safe_state(args.quiet)

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=bool(args.debug),
    )
    background_value = 1.0 if args.white_background else 0.0
    background = torch.tensor([background_value, background_value, background_value], dtype=torch.float32, device="cuda")

    ignatius, ignatius_iter = load_ignatius_gaussians(Path(args.ignatius_checkpoint), sh_degree=3)
    lego, lego_iter = load_refgs_as_irgs_gaussians(Path(args.lego_checkpoint), sh_degree=3)
    transform = apply_gaussian_transform(lego, args.translation, args.rotation_deg, args.scale, args.rotation_order)
    post_scale_center = apply_post_transform_scale(lego, args.post_scale, args.post_scale_center)

    cameras_json = select_camera_jsons(
        load_camera_json(Path(args.ignatius_cameras)),
        first_k=int(args.first_k),
        camera_ids=parse_int_list(args.camera_ids),
    )
    if not cameras_json:
        raise RuntimeError("No cameras selected.")

    output_dir = Path(args.output_dir)
    make_output_dirs(output_dir)
    metadata = {
        "lego_checkpoint": str(Path(args.lego_checkpoint).resolve()),
        "lego_iteration": lego_iter,
        "ignatius_checkpoint": str(Path(args.ignatius_checkpoint).resolve()),
        "ignatius_iteration": ignatius_iter,
        "ignatius_cameras": str(Path(args.ignatius_cameras).resolve()),
        "num_cameras": len(cameras_json),
        "translation": list(map(float, args.translation)),
        "rotation_deg": list(map(float, args.rotation_deg)),
        "rotation_order": args.rotation_order,
        "scale": float(args.scale),
        "post_scale": float(args.post_scale),
        "post_scale_center": args.post_scale_center,
        "post_scale_center_xyz": post_scale_center.detach().cpu().tolist(),
        "transform_matrix": transform.detach().cpu().tolist(),
        "max_width": int(args.max_width),
        "resolution_scale": float(args.resolution_scale),
        "alpha_thresh": float(args.alpha_thresh),
        "depth_bias": float(args.depth_bias),
        "white_background": bool(args.white_background),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    for idx, cam_json in enumerate(tqdm(cameras_json, desc="Rendering lego into Ignatius")):
        view = camera_from_json(cam_json, max_width=args.max_width, scale=args.resolution_scale)
        ignatius_pkg = render_sh_gbuffer(view, ignatius, pipe)
        lego_pkg = render_sh_gbuffer(view, lego, pipe)
        composite, lego_only, front_mask = composite_by_depth(
            bg_pkg=ignatius_pkg,
            fg_pkg=lego_pkg,
            background=background,
            alpha_thresh=float(args.alpha_thresh),
            depth_bias=float(args.depth_bias),
        )
        stem = f"{idx:05d}_{cam_json.get('img_name', cam_json.get('id', idx))}"
        save_debug_outputs(output_dir, stem, composite, ignatius_pkg, lego_pkg, lego_only, front_mask, background)

        del view, ignatius_pkg, lego_pkg, composite, lego_only, front_mask
        torch.cuda.empty_cache()

    print(f"[done] Saved {len(cameras_json)} composite views to {output_dir}")


if __name__ == "__main__":
    main()
