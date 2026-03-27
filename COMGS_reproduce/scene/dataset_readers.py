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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _resolve_masks_folder(scene_path):
    for masks_dir_name in ("masks", "mask"):
        masks_folder = os.path.join(scene_path, masks_dir_name)
        if os.path.isdir(masks_folder):
            return masks_folder
    return None


def _build_mask_stem_candidates(frame_stem):
    candidates = [frame_stem]
    try:
        frame_idx = int(frame_stem)
        candidates.extend([
            f"{frame_idx:05d}",
            f"{frame_idx:08d}",
            f"{frame_idx:05d}_obj1",
            f"{frame_idx:08d}_obj1",
            f"{frame_idx}_obj1",
        ])
    except ValueError:
        candidates.append(frame_stem + "_obj1")

    deduped = []
    seen = set()
    for name in candidates:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _stem_sort_key(stem):
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _mask_extensions():
    return (
        ".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF",
    )


def _find_mask_path_by_stem(masks_folder, frame_stem):
    if masks_folder is None:
        return None

    for stem_candidate in _build_mask_stem_candidates(frame_stem):
        for ext in _mask_extensions():
            candidate = os.path.join(masks_folder, stem_candidate + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def _load_binary_mask(mask_path):
    if mask_path.lower().endswith(".npy"):
        mask_np = np.load(mask_path)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = (mask_np > 0).astype(np.uint8) * 255
    else:
        mask_np = np.asarray(Image.open(mask_path).convert("L"))
        mask_np = (mask_np > 127).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def _collect_ordered_masks(masks_folder):
    if masks_folder is None or not os.path.isdir(masks_folder):
        return []

    per_root = {}
    ext_priority = {ext: i for i, ext in enumerate(_mask_extensions())}
    for file_name in os.listdir(masks_folder):
        full_path = os.path.join(masks_folder, file_name)
        if not os.path.isfile(full_path):
            continue

        stem, ext = os.path.splitext(file_name)
        if ext not in ext_priority:
            continue
        if stem.endswith("_vis"):
            continue

        is_obj1 = stem.endswith("_obj1")
        root = stem[:-5] if is_obj1 else stem
        priority = (1 if is_obj1 else 0, ext_priority[ext])

        prev = per_root.get(root)
        if prev is None or priority < prev[0]:
            per_root[root] = (priority, full_path)

    ordered = sorted(per_root.items(), key=lambda kv: _stem_sort_key(kv[0]))
    return [path for _, (_, path) in ordered]


def _build_colmap_mask_lookup(masks_folder, image_stems):
    image_stems = sorted(set(image_stems), key=_stem_sort_key)
    if masks_folder is None:
        return {}

    direct = {}
    for stem in image_stems:
        mask_path = _find_mask_path_by_stem(masks_folder, stem)
        if mask_path is not None:
            direct[stem] = mask_path

    if len(direct) == len(image_stems):
        return direct

    ordered_mask_paths = _collect_ordered_masks(masks_folder)
    if len(ordered_mask_paths) == len(image_stems):
        return {img_stem: mask_path for img_stem, mask_path in zip(image_stems, ordered_mask_paths)}

    return direct


def _load_colmap_mask(mask_lookup, masks_folder, frame_stem, image_size):
    mask_path = None
    if mask_lookup is not None:
        mask_path = mask_lookup.get(frame_stem)
    if mask_path is None:
        mask_path = _find_mask_path_by_stem(masks_folder, frame_stem)
    if mask_path is None:
        return None

    mask_img = _load_binary_mask(mask_path)
    if mask_img.size != image_size:
        nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        mask_img = mask_img.resize(image_size, nearest)
    return mask_img


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder=None):
    image_stems = [os.path.splitext(os.path.basename(cam_extrinsics[k].name))[0] for k in cam_extrinsics]
    mask_lookup = _build_colmap_mask_lookup(masks_folder, image_stems)

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image_rgb = Image.open(image_path).convert("RGB")

        mask_img = _load_colmap_mask(mask_lookup, masks_folder, image_name, image_rgb.size)
        if mask_img is not None:
            image = image_rgb.copy()
            image.putalpha(mask_img)
        else:
            image = image_rgb

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    masks_folder = _resolve_masks_folder(path)
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        masks_folder=masks_folder,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}
