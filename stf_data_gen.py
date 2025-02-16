#!/usr/bin/env python3
import os
from os import path as osp
import math
import random
import numpy as np
import trimesh
import csv

import time
import hydra
from omegaconf import DictConfig
from midastouch.render.digit_renderer import digit_renderer
from midastouch.modules.misc import remove_and_mkdir, save_heightmaps, save_contactmasks, save_images
from midastouch.modules.pose import pose_from_vertex_normal

from tqdm import tqdm

from main import sample_point_in_triangle, generate_candidate_points, compute_candidate_density, farthest_point_sampling, compute_face_normals


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    obj_model = cfg.obj_model
    max_samples = cfg.max_samples
    candidate_multiplier = cfg.get("candidate_multiplier", 10)
    render_cfg = cfg.render
    density_boost = cfg.get("density_boost", 1.0)
    area_exponent = cfg.get("area_exponent", 0.5)

    # 1) 메쉬 로드
    mesh_path = osp.join("obj_models", obj_model, "nontextured.stl")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    print(f"[INFO] Loaded {obj_model}")
    print(f" #faces={len(mesh.faces)}, #verts={len(mesh.vertices)}")
    print(f" extents={mesh.extents}, bounds={mesh.bounds}")

    # 2) 후보점 생성 및 FPS 적용
    candidate_count = max_samples * candidate_multiplier
    print(f"[INFO] Generating {candidate_count} candidate points for sampling...")
    # area_exponent를 반영하여 후보점 생성
    candidates, candidate_faces = generate_candidate_points(mesh, candidate_count, area_exponent=area_exponent)

    # density_radius 기본값: bounding box diagonal / density_radius_factor
    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    density_radius = diag / cfg.get("density_radius_factor", 50.0)

    print(f"[INFO] Applying Farthest Point Sampling (with density boost) to select {max_samples} points...")
    points, faces = farthest_point_sampling(candidates, candidate_faces, max_samples,
                                            density_radius=density_radius,
                                            density_boost=density_boost)
    print(f" -> Final selected points: {len(points)}")

    # 3) 센서 포즈 계산 (선택된 face의 노멀 정보 사용)
    normals = compute_face_normals(mesh, faces)
    shear_rad = math.radians(render_cfg.shear_mag)
    delta = np.random.uniform(0.0, 2 * math.pi, size=len(points))
    poses = pose_from_vertex_normal(points, normals, shear_rad, delta)

    # 4) 출력 폴더 생성
    data_path = os.path.join("data", "sim", obj_model, "stiffness")
    remove_and_mkdir(data_path)


    # 5) 렌더러 초기화 (randomize 옵션이 False이면 한 번만 생성)
    renderer = digit_renderer(
        cfg=render_cfg,
        obj_path=mesh_path,
        headless=render_cfg.headless
    )

    # 6) 배치 렌더링 처리
    BATCH_SIZE = cfg.get("batch_size", 500)
    total_poses = len(poses)

    k_config = cfg.render.k

    k_max = k_config.max
    k_min = k_config.min
    k_interval = 500

    k_values = list(range(k_min,k_max,k_interval))

    for k in k_values:
        tqdm.write(f"------------stiffness: {k}--------------")

        renderer.set_obj_stiffness(k)
        per_stf_path = os.path.join(data_path, f"stiffness_{k}")
        os.makedirs(per_stf_path, exist_ok=True)

        image_path = os.path.join(per_stf_path, "tactile_images")
        heightmap_path = os.path.join(per_stf_path, "gt_heightmaps")
        contactmasks_path = os.path.join(per_stf_path, "gt_contactmasks")
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(heightmap_path, exist_ok=True)
        os.makedirs(contactmasks_path, exist_ok=True)

        all_hm, all_cm, all_img, cam_poses = [], [], [], []
        start_idx = 0

        tqdm.write(f"[INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}")
        time.sleep(1)
        with tqdm(total=total_poses, desc="Rendering Progress", unit="poses") as pbar:
            while start_idx < total_poses:
                end_idx = min(start_idx + BATCH_SIZE, total_poses)
                batch_poses = poses[start_idx:end_idx]

                hm, cm, imgs,cam_p , _, _ = renderer.render_sensor_trajectory(p=batch_poses, mNoise=cfg.noise)
                all_hm.extend(hm)
                all_cm.extend(cm)
                all_img.extend(imgs)
                cam_poses.extend(cam_p)


                pbar.update(len(batch_poses))
                start_idx = end_idx

        print("[INFO] Saving final results...")
        save_heightmaps(all_hm, heightmap_path)
        save_contactmasks(all_cm, contactmasks_path)
        save_images(all_img, image_path)

        # 최종 결과 저장
        np.save(os.path.join(per_stf_path, "sampled_points.npy"), points)
        np.save(os.path.join(per_stf_path, "sensor_poses.npy"), poses)
        np.save(os.path.join(per_stf_path, "cam_poses.npy"), cam_poses)

        print(f"[DONE] Coverage scan done, total poses={total_poses}")
        print(f"Results in {per_stf_path}")

    output_csv = os.path.join(data_path, "combined_result.csv")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "stiffness"])  # 헤더 작성
        for s in k_values:
            base_dir = os.path.join(data_path, f"s_{s}")
            img_dir = os.path.join(base_dir, "gt_heightmaps")
            if not os.path.exists(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(".jpg"):
                    file_path = os.path.join(img_dir, fname)
                    writer.writerow([file_path, s])

    print(f"CSV 파일이 생성되었습니다: {output_csv}")

if __name__ == "__main__":
    main()
