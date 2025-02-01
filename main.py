#!/usr/bin/env python3
import os
from os import path as osp

import math
import random
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import hydra
from omegaconf import DictConfig
from midastouch.render.digit_renderer import digit_renderer
from midastouch.modules.misc import remove_and_mkdir, save_heightmaps, save_contactmasks, save_images
from midastouch.modules.pose import pose_from_vertex_normal

from tqdm import tqdm


def sample_point_in_triangle(face_idx, mesh):
    verts = mesh.vertices[mesh.faces[face_idx]]
    r1, r2 = np.random.rand(2)
    sqrt_r1 = math.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    return verts[0]*w0 + verts[1]*w1 + verts[2]*w2


def poisson_disk_on_mesh(mesh, min_dist, max_samples, max_attempts):
    """
    간단한 브리드슨(Bridson)식 Poisson Disk 샘플링 (표면).
    """
    faces_area = np.cumsum(mesh.area_faces)
    faces_area /= faces_area[-1]  # [0..1]

    selected_points = []
    selected_faces = []
    tree = None

    accepted = 0
    attempts = 0

    def random_point_on_mesh():
        r = random.random()
        f_idx = np.searchsorted(faces_area, r)
        pt = sample_point_in_triangle(f_idx, mesh)
        return pt, f_idx

    while attempts < max_attempts and accepted < max_samples:
        attempts += 1
        p, fidx = random_point_on_mesh()

        if accepted == 0:
            selected_points.append(p)
            selected_faces.append(fidx)
            accepted += 1
            tree = cKDTree(np.array(selected_points))
            continue

        dist, idx_ = tree.query(p, k=1)
        if dist >= min_dist:
            selected_points.append(p)
            selected_faces.append(fidx)
            accepted += 1
            tree = cKDTree(np.array(selected_points))

    print(f"[PoissonDisk] attempts={attempts}, accepted={accepted}")
    return np.array(selected_points), np.array(selected_faces)


def compute_face_normals(mesh, face_idxs):
    norms = mesh.face_normals[face_idxs]
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True)+1e-12)
    return norms


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Config 예시 (config.yaml):

    obj_model: "035_power_drill"
    min_dist: 0.001
    max_samples: 2000
    max_attempts: 200000
    render:
      shear_mag: 5.0
      randomize: True
      headless: False
    """

    obj_model = cfg.obj_model
    min_dist = cfg.min_dist
    max_samples = cfg.max_samples
    max_attempts = cfg.max_attempts
    render_cfg = cfg.render

    # 1) Load mesh
    # mesh_path = f"/home/wireless/Tactile/MidasTouch/obj_models/{obj_model}/nontextured.stl"
    mesh_path = osp.join("obj_models",obj_model,"nontextured.stl")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    print(f"[INFO] Loaded {obj_model}")
    print(f" #faces={len(mesh.faces)}, #verts={len(mesh.vertices)}")
    print(f" extents={mesh.extents}, bounds={mesh.bounds}")

    # 2) Poisson Disk
    print(f"[INFO] Start Poisson disk sampling: min_dist={min_dist}, max_samples={max_samples}")
    points, faces = poisson_disk_on_mesh(mesh, min_dist, max_samples, max_attempts)
    print(f" -> final accepted points: {len(points)}")

    # 3) compute normals -> poses
    normals = compute_face_normals(mesh, faces)
    import math
    shear_deg = render_cfg.shear_mag
    shear_rad = math.radians(shear_deg)
    delta = np.random.uniform(0.0, 2*math.pi, size=len(points))
    poses = pose_from_vertex_normal(points, normals, shear_rad, delta)

    # 4) Make output folder
    data_path = os.path.join("data", "sim", obj_model, "full_coverage")
    remove_and_mkdir(data_path)
    image_path = os.path.join(data_path, "tactile_images")
    heightmap_path = os.path.join(data_path, "gt_heightmaps")
    contactmasks_path = os.path.join(data_path, "gt_contactmasks")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(heightmap_path, exist_ok=True)
    os.makedirs(contactmasks_path, exist_ok=True)

    # 5) digit_renderer
    renderer = digit_renderer(
        cfg=render_cfg,
        obj_path=mesh_path,
        headless=render_cfg.headless
    )

    # 6) Batch rendering
    BATCH_SIZE = 500
    total_poses = len(poses)
    all_hm, all_cm, all_img = [], [], []
    start_idx = 0

    tqdm.write(f"[INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}")
    with tqdm(total=total_poses, desc="Rendering Progress", unit="poses") as pbar:

        while start_idx < total_poses:
            end_idx = min(start_idx + BATCH_SIZE, total_poses)
            batch_poses = poses[start_idx:end_idx]

            if render_cfg.randomize:
                # optional: re-init each batch
                renderer = digit_renderer(
                    cfg=render_cfg,
                    obj_path=mesh_path,
                    headless=render_cfg.headless
                )
            hm, cm, imgs, _, _, _ = renderer.render_sensor_trajectory(p=batch_poses, mNoise=cfg.noise)
            all_hm.extend(hm)
            all_cm.extend(cm)
            all_img.extend(imgs)
            pbar.update(len(batch_poses))
            start_idx = end_idx

    print("[INFO] Saving final results...")
    save_heightmaps(all_hm, heightmap_path)
    save_contactmasks(all_cm, contactmasks_path)
    save_images(all_img, image_path)

    # 샘플링 완료 후 경로 저장
    np.save(os.path.join(data_path, "sampled_points.npy"), points)  # 샘플링된 점
    np.save(os.path.join(data_path, "sensor_poses.npy"), poses)  # 센서 포즈

    print(f"[DONE] coverage scan done, total poses={total_poses}")
    print(f"Results in {data_path}")


if __name__ == "__main__":
    main()
