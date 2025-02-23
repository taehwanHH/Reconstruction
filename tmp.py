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

from module.TactileUtil import Stiffness
from module.TSN import TSN_COPY

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import cached_property


class Sensing(Stiffness):
    def __init__(self, obj, config):
        super().__init__(config)
        self.obj = obj
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.sensing_cfg = config.sensing
        self.render_cfg = config.render
        self.max_samples = config.sesning.max_samples
        self._init_paths()
        self.model = TSN_COPY(num_k = self.k_num)
        self.all_hm, self.all_cm, self.all_img = [], [], []



    def _init_paths(self):
        base = osp.join("data", "sim", self.obj, "full_coverage")
        self.base = base
        remove_and_mkdir(base)
        self.image_path = osp.join(base, "tactile_images")
        self.heightmap_dir = osp.join(base, "gt_heightmaps")
        self.mask_dir = osp.join(base, "gt_contactmasks")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.heightmap_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        self.mesh_path = osp.join("obj", self.obj, "nontextured.stl")

    @cached_property
    def mesh(self):
        return trimesh.load(self.mesh_path)

    @cached_property
    def renderer(self):
        return digit_renderer(cfg=self.render_cfg, obj_path=self.mesh_path,
                              headless=self.render_cfg.headless)

    @cached_property
    def mesh_bounds(self):
        min_bound, max_bound = self.mesh.bounds
        bounds = (min_bound, max_bound)
        return bounds

    @cached_property
    def num_candidates(self):
        return self.sensing_cfg.max_samples * self.sensing_cfg.candidate_multiplier

    def candidate_points(self):
        # 면적 가중치 계산: 각 면적에 area_exponent를 적용
        face_areas = self.mesh.area_faces
        weighted_areas = face_areas ** self.sensing_cfg.area_exponent
        cum_weights = np.cumsum(weighted_areas)
        cum_weights /= cum_weights[-1]  # 누적 분포 [0,1]

        # num_candidates 개수만큼 난수를 한 번에 생성하고, 해당하는 face 인덱스를 vectorized하게 계산
        rs = np.random.random(self.num_candidates)
        candidate_faces = np.searchsorted(cum_weights, rs)

        # 각 face에 대해 점을 샘플링 (sample_point_in_triangle은 각 면마다 개별 처리)
        candidate_points = np.array([sample_point_in_triangle(f_idx, self.mesh)
                                     for f_idx in candidate_faces])
        return candidate_points, candidate_faces

    def farthest_point_sampling(self,points, candidate_faces):
        """
        후보점 집합(points)에서 Farthest Point Sampling (FPS)을 수행하여
        num_samples 개의 점과 이에 해당하는 face 인덱스를 반환합니다.
        density_radius: 각 후보점의 local density를 계산할 반경. None이면 기본값 사용.
        density_boost: 밀도가 낮은 영역에 부여할 보정 계수.
        """
        print(f"[INFO] Applying Farthest Point Sampling (with density boost) to select {self.sensing_cfg.max_samples} points...")

        # density_radius 기본값: bounding box diagonal / density_radius_factor
        diag = np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0])
        density_radius = diag /self.sensing_cfg.density_radius_factor
        density_boost = 1.0

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
        # first_idx = random.randint(0, N - 1)
        first_idx = 0
        sampled_indices.append(first_idx)

        last_point = points[first_idx]
        dists = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dists)

        for i in range(1, self.max_samples):
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

    def get_points_poses(self):
        print(f"[INFO] Generating {self.num_candidates} candidate points for sampling...")
        candidates, candidate_faces = self.candidate_points()
        print(f"[INFO] Applying Farthest Point Sampling (with density boost) to select {self.max_samples} points...")
        points, faces = self.farthest_point_sampling(candidates, candidate_faces)
        print(f" -> Final selected points: {len(points)}")

        normals = compute_face_normals(self.mesh, faces)
        shear_rad = math.radians(self.render_cfg.shear_mag)
        delta = np.random.uniform(0.0, 2 * math.pi, size=len(points))
        poses = pose_from_vertex_normal(points, normals, shear_rad, delta)

        return points, poses

    def sensing(self,poses):
        total_poses = len(poses)
        BATCH_SIZE = self.sensing_cfg.batch_size
        start_idx = 0

        print((f"[INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}"))
        while start_idx < total_poses:
            end_idx = min(start_idx + BATCH_SIZE, total_poses)
            batch_poses = poses[start_idx:end_idx]
            positions = batch_poses[:, :3, 3]
            stiffness_values = get_local_stiffness(positions, self.mesh_bounds, self.k_values)

            hm, cm, imgs, _, _, _ = self.renderer.render_sensor_trajectory(p=batch_poses, k_values=stiffness_values,
                                                                          mNoise=self.cfg.noise)
            self.all_hm.extend(hm)
            self.all_cm.extend(cm)
            self.all_img.extend(imgs)

            start_idx = end_idx

    def save_results(self):

def sample_point_in_triangle(face_idx, mesh):
    verts = mesh.vertices[mesh.faces[face_idx]]
    r1, r2 = np.random.rand(2)
    sqrt_r1 = math.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    return verts[0] * w0 + verts[1] * w1 + verts[2] * w2


def compute_candidate_density(points, radius):
    """
    각 후보점 주변의 점 개수를 radius 내에서 계산하여 local density 배열을 반환.
    """
    tree = cKDTree(points)
    densities = np.array([len(tree.query_ball_point(pt, radius)) - 1 for pt in points])
    return densities



def compute_face_normals(mesh, face_idxs):
    """
    선택된 face 인덱스에 해당하는 면 노멀을 계산하여 정규화.
    """
    norms = mesh.face_normals[face_idxs]
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-12)
    return norms


def get_local_stiffness(positions, bounds, k_values):
    min_bound, max_bound = bounds
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)

    # 각 축의 구간 길이 계산
    z_interval = (max_bound[2] - min_bound[2]) / 2.0
    x_interval = (max_bound[0] - min_bound[0]) / 2.0
    y_interval = (max_bound[1] - min_bound[1]) / 2.0

    # 각 contact point의 인덱스 계산 (배치 연산)
    z_idx = np.floor((positions[:, 2] - min_bound[2]) / z_interval).astype(np.int32)
    z_idx = np.clip(z_idx, 0, 1)

    x_idx = np.floor((positions[:, 0] - min_bound[0]) / x_interval).astype(np.int32)
    x_idx = np.clip(x_idx, 0, 1)

    y_idx = np.floor((positions[:, 1] - min_bound[1]) / y_interval).astype(np.int32)
    y_idx = np.clip(y_idx, 0, 1)

    xy_idx = y_idx * 2 + x_idx
    region_idx = (z_idx * 4 + xy_idx)%8  # 0 ~ 15

    stiffness_candidates = np.array(k_values)
    return stiffness_candidates[region_idx]


def color_mesh_by_stiffness(mesh, sample_positions, sample_stiffness, colormap_name='viridis'):
    """
    mesh: trimesh 객체
    sample_positions: (N, 3) numpy array, 각 contact point의 위치
    sample_stiffness: (N,) numpy array, 각 contact point에 해당하는 stiffness 값
    colormap_name: 사용할 matplotlib colormap 이름 (예: 'viridis')

    메쉬의 각 버텍스에 대해, 가장 가까운 contact point의 stiffness 값을 할당한 후,
    이를 colormap을 통해 RGB로 변환하여 버텍스 컬러로 지정.
    """
    # KD-tree를 구축하여 각 버텍스에 대해 가장 가까운 contact point 찾기
    tree = cKDTree(sample_positions)
    vertex_positions = mesh.vertices  # (M, 3)
    distances, indices = tree.query(vertex_positions)

    # 각 버텍스에 대해 stiffness 값 할당
    vertex_stiffness = sample_stiffness[indices]

    # 정규화: 0 ~ 1 범위로
    norm_stiffness = (vertex_stiffness - vertex_stiffness.min()) / (
                vertex_stiffness.max() - vertex_stiffness.min() + 1e-8)

    # colormap 적용 (RGBA -> RGB 추출)
    cmap = plt.get_cmap(colormap_name)
    vertex_colors = cmap(norm_stiffness)[:, :3]  # (M, 3), 값은 0~1

    # trimesh에서는 vertex_colors가 0~255 uint8 형식이어야 함
    mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    return mesh



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
    min_bound, max_bound = mesh.bounds
    bounds = (min_bound, max_bound)
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

    k_config = cfg.render.k
    k_max = k_config.max
    k_min = k_config.min
    k_interval = k_config.interval

    k_values = list(range(k_min, k_max + 1, k_interval))

    tqdm.write(f"[INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}")
    time.sleep(1)
    with tqdm(total=total_poses, desc="Rendering Progress", unit="poses") as pbar:
        while start_idx < total_poses:
            end_idx = min(start_idx + BATCH_SIZE, total_poses)
            batch_poses = poses[start_idx:end_idx]
            positions = batch_poses[:,:3, 3]
            stiffness_values = get_local_stiffness(positions,bounds,k_values)

            hm, cm, imgs,cam_p , _, _ = renderer.render_sensor_trajectory(p=batch_poses,k_values=stiffness_values, mNoise=cfg.noise)
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
    # np.save(os.path.join(data_path, "cam_poses.npy"), cam_poses)


    print(f"[DONE] Coverage scan done, total poses={total_poses}")


    print(f"Results in {data_path}")

    # print(f"[INFO] Stiffness map creating...")
    # positions = poses[:, :3, 3]
    # org_stiffness = get_local_stiffness(positions, bounds, k_values)
    # # stiffness 값을 [0, 1] 범위로 정규화 (heatmap 시각화를 위해)
    # # norm_res = (org_stiffness - org_stiffness.min()) / (org_stiffness.max() - org_stiffness.min() + 1e-8)
    #
    # colored_mesh1 = color_mesh_by_stiffness(mesh, positions, org_stiffness, colormap_name='viridis')
    # colored_mesh1.show()
    # print(f"[DONE] Stiffness map saved")

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = create_resnet_for_classification(num_input_channels=1, num_classes=len(k_values))
    # model.to(device)
    # model.load_state_dict(
    #     torch.load("model_weights/resnet_TSN.pth", map_location=device, weights_only=True))
    # model.eval()
    #
    # k_l=[]
    # for i in range(4000):
    #     hm_path = osp.join(heightmap_path,f"{i}.jpg")
    #     hm = Image.open(hm_path)
    #     hm_img = transform(hm.convert('L'))
    #     with torch.no_grad():
    #         hm_in = hm_img.unsqueeze(0).to(device)
    #         output = model(hm_in)
    #         _, predicted_k = torch.max(output, 1)
    #         k_l.append(predicted_k.item())
    # k_values = np.array(k_values)
    # k_list = k_values[k_l]
    # positions = poses[:, :3, 3]
    # norm_res = (k_list - k_list.min()) / (k_list.max() - k_list.min() + 1e-8)
    #
    # colored_mesh2 = color_mesh_by_stiffness(mesh, positions, norm_res,
    #                                        colormap_name='viridis')
    # # colored_mesh2.show()  # 내장 뷰어로 확인하거나
    # # colored_mesh2.apply_translation([0.2, 0, 0])
    # #
    # # # 두 메쉬를 하나의 씬에 추가
    # # scene = trimesh.Scene([colored_mesh1, colored_mesh2])
    # # scene.show()
if __name__ == "__main__":
    main()
