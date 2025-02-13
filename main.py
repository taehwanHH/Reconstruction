#!/usr/bin/env python3
import os
from os import path as osp
import math
import random
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import time
import hydra
from omegaconf import DictConfig
from midastouch.render.digit_renderer import digit_renderer
from midastouch.modules.misc import remove_and_mkdir, save_heightmaps, save_contactmasks, save_images
from midastouch.modules.pose import pose_from_vertex_normal

from tqdm import tqdm


def sample_point_in_triangle(face_idx, mesh):
    """
    삼각형 내부에서 균일하게 한 점을 샘플링.
    """
    verts = mesh.vertices[mesh.faces[face_idx]]
    r1, r2 = np.random.rand(2)
    sqrt_r1 = math.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    return verts[0] * w0 + verts[1] * w1 + verts[2] * w2


def generate_candidate_points(mesh, num_candidates, area_exponent=0.5):
    """
    메쉬의 면적에 area_exponent(예, 0.5)를 적용하여,
    면적이 작은(얇은) 영역도 충분히 후보점이 생성되도록 num_candidates 개의 후보점을 생성.
    각 후보점은 해당 face에서 샘플링되며, 원래 샘플에 사용된 face 인덱스도 함께 저장함.
    """
    # 기존 면적 대신 면적^area_exponent를 사용한 가중치 계산
    face_areas = mesh.area_faces
    weighted_areas = face_areas ** area_exponent
    cum_weights = np.cumsum(weighted_areas)
    cum_weights /= cum_weights[-1]  # [0,1] 범위 누적 분포

    candidate_points = []
    candidate_faces = []

    for _ in range(num_candidates):
        r = random.random()
        f_idx = np.searchsorted(cum_weights, r)
        pt = sample_point_in_triangle(f_idx, mesh)
        candidate_points.append(pt)
        candidate_faces.append(f_idx)

    return np.array(candidate_points), np.array(candidate_faces)


def compute_candidate_density(points, radius):
    """
    각 후보점 주변의 점 개수를 radius 내에서 계산하여 local density 배열을 반환.
    """
    tree = cKDTree(points)
    densities = np.array([len(tree.query_ball_point(pt, radius)) - 1 for pt in points])
    return densities


def farthest_point_sampling(points, candidate_faces, num_samples, density_radius=None, density_boost=1.0):
    """
    후보점 집합(points)에서 Farthest Point Sampling (FPS)을 수행하여
    num_samples 개의 점과 이에 해당하는 face 인덱스를 반환합니다.
    density_radius: 각 후보점의 local density를 계산할 반경. None이면 기본값 사용.
    density_boost: 밀도가 낮은 영역에 부여할 보정 계수.
    """
    N = points.shape[0]
    sampled_indices = []
    distances = np.full(N, np.inf)

    # density_radius 기본값 설정 (메쉬 bounding box 크기에 따라 조정)
    if density_radius is None:
        diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        density_radius = diag / 50.0  # 조정 가능한 파라미터

    # 각 후보점의 local density 계산
    densities = compute_candidate_density(points, density_radius)
    dens_min = densities.min()
    dens_max = densities.max()
    # normalized_density: 값이 낮을수록 주변에 후보점이 적은 영역 (얇은 영역)로 간주
    normalized_density = (densities - dens_min) / (dens_max - dens_min + 1e-12)

    # 첫 번째 점은 랜덤 선택
    first_idx = random.randint(0, N - 1)
    sampled_indices.append(first_idx)

    last_point = points[first_idx]
    dists = np.linalg.norm(points - last_point, axis=1)
    distances = np.minimum(distances, dists)

    for i in range(1, num_samples):
        # 지수 함수 기반 보정: 밀도가 낮은 영역에 대해 더 강하게 보정.
        effective_distance = distances * np.exp(density_boost * (1 - normalized_density))
        next_idx = np.argmax(effective_distance)
        sampled_indices.append(next_idx)
        last_point = points[next_idx]
        dists = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dists)

    sampled_points = points[sampled_indices]
    sampled_faces = candidate_faces[sampled_indices]
    return sampled_points, sampled_faces


def compute_face_normals(mesh, face_idxs):
    """
    선택된 face 인덱스에 해당하는 면 노멀을 계산하여 정규화.
    """
    norms = mesh.face_normals[face_idxs]
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-12)
    return norms


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Config 예시 (config.yaml):

    obj_model: "035_power_drill"  # 바나나 형태의 모델로 대체 가능
    max_samples: 2000
    candidate_multiplier: 10       # 후보점은 max_samples의 몇 배로 생성할지 결정
    area_exponent: 0.5             # 면적 가중치의 지수 (0.5이면 모든 face가 거의 균일하게 선택됨)
    render:
      shear_mag: 5.0
      randomize: False
      headless: False
    noise: 0.0
    batch_size: 500
    density_boost: 1.0           # 밀도 보정을 위한 파라미터 (얇은 영역 강화)
    density_radius_factor: 50.0  # 기본 반경 조정을 위한 인자 (bounding box diagonal / factor)
    """
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
    data_path = os.path.join("data", "sim", obj_model, "full_coverage")
    remove_and_mkdir(data_path)
    image_path = os.path.join(data_path, "tactile_images")
    heightmap_path = os.path.join(data_path, "gt_heightmaps")
    contactmasks_path = os.path.join(data_path, "gt_contactmasks")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(heightmap_path, exist_ok=True)
    os.makedirs(contactmasks_path, exist_ok=True)

    # 5) 렌더러 초기화 (randomize 옵션이 False이면 한 번만 생성)
    renderer = digit_renderer(
        cfg=render_cfg,
        obj_path=mesh_path,
        headless=render_cfg.headless
    )

    # 6) 배치 렌더링 처리
    BATCH_SIZE = cfg.get("batch_size", 500)
    total_poses = len(poses)
    all_hm, all_cm, all_img, cam_poses = [], [], [],[]
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
    np.save(os.path.join(data_path, "sampled_points.npy"), points)
    np.save(os.path.join(data_path, "sensor_poses.npy"), poses)
    np.save(os.path.join(data_path, "cam_poses.npy"), cam_poses)

    print(f"[DONE] Coverage scan done, total poses={total_poses}")
    print(f"Results in {data_path}")


if __name__ == "__main__":
    main()
