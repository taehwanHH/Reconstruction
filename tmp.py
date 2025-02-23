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
from module.TSN import TSN_COPY, TRANSFORM
from PIL import Image

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
        self.max_samples = config.sensing.max_samples
        self._init_paths()
        self.model = TSN_COPY(num_k = self.k_num)
        self.all_hm, self.all_cm, self.all_img = [], [], []



    def _init_paths(self):
        base = osp.join("data", "sim", self.obj, "full_coverage")
        self.base = base
        remove_and_mkdir(base)
        self.image_dir = osp.join(base, "tactile_images")
        self.heightmap_dir = osp.join(base, "gt_heightmaps")
        self.mask_dir = osp.join(base, "gt_contactmasks")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.heightmap_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        self.mesh_path = osp.join("obj_models", self.obj, "nontextured.stl")

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

        return  poses

    def sensing(self,poses):
        total_poses = len(poses)
        BATCH_SIZE = self.sensing_cfg.batch_size
        start_idx = 0

        print((f"[INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}"))
        while start_idx < total_poses:
            end_idx = min(start_idx + BATCH_SIZE, total_poses)
            batch_poses = poses[start_idx:end_idx]
            positions = batch_poses[:, :3, 3]
            stiffness_values = self.get_local_stiffness(positions, self.mesh_bounds)

            hm, cm, imgs, _, _, _ = self.renderer.render_sensor_trajectory(p=batch_poses, k_values=stiffness_values,
                                                                          mNoise=self.cfg.noise)
            self.all_hm.extend(hm)
            self.all_cm.extend(cm)
            self.all_img.extend(imgs)

            start_idx = end_idx

        print(f"[DONE] Coverage scan done, total poses={total_poses}")

    def save_results(self, poses):
        points = poses[:,:3, 3]
        print("[INFO] Saving final results...")
        save_heightmaps(self.all_hm, self.heightmap_dir)
        save_contactmasks(self.all_cm, self.mask_dir)
        save_images(self.all_img, self.image_dir)
        np.save(os.path.join(self.base, "sampled_points.npy"), points)
        np.save(os.path.join(self.base, "sensor_poses.npy"), poses)
        print(f"[DONE] Results in {self.base}")

    def show_heatmap(self, poses, mode=None, visible=False):
        print(f"[INFO] Stiffness map creating...")
        points = poses[:,:3, 3]
        if mode=="origin":
            org_k = self.get_local_stiffness(points, self.mesh_bounds)
            norm_k = self.k_normalize(org_k)
        else:
            k_l = []
            for i in range(self.max_samples):
                hm_path = osp.join(self.heightmap_dir, f"{i}.jpg")
                hm = Image.open(hm_path)
                hm_img = TRANSFORM(hm.convert('L'))
                with torch.no_grad():
                    hm_in = hm_img.unsqueeze(0).to(self.device)
                    output = self.model(hm_in)
                    _, predicted_k = torch.max(output, 1)
                    k_l.append(predicted_k.item())

            model_k = self.k_values[k_l]
            norm_k = self.k_normalize(model_k)

        heatmap = self.color_mesh_by_stiffness(self.mesh, points, norm_k)

        if visible:
            heatmap.show()


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


