#!/usr/bin/env python3
import os
from os import path as osp
import math
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


from midastouch.render.digit_renderer import digit_renderer
from midastouch.modules.misc import remove_and_mkdir, save_heightmaps, save_contactmasks, save_images
# from midastouch.modules.pose import pose_from_vertex_normal
from midastouch.data_gen.utils import random_geodesic_poses, random_manual_poses

from module.Stiffness import Stiffness
# from module.model.Classifier.Classifier import TSN_COPY, TRANSFORM
from PIL import Image

import torch
from functools import cached_property


class Sensing(Stiffness):
    def __init__(self, config, output_base, mkdir=True):
        super().__init__(config)
        self.obj = config.obj_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.sensing_cfg = config.sensing
        self.render_cfg = config.render
        # self.samples = config.sensing.samples
        self.base = output_base
        self._init_paths(mkdir)
        # self.samples = int(self.num_faces()/150)
        self.samples = self.sensing_cfg.num_samples
        self.num_candidates = self.get_num_candidates()
        # self.model = TSN_COPY(num_k = self.k_num)
        self.all_hm, self.all_cm, self.all_img = [], [], []



    def _init_paths(self, mkdir=True):
        base = self.base
        self.image_dir = osp.join(base, "tactile_images")
        self.heightmap_dir = osp.join(base, "gt_heightmaps")
        self.mask_dir = osp.join(base, "gt_contactmasks")
        if mkdir:
            remove_and_mkdir(base)
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.heightmap_dir, exist_ok=True)
            os.makedirs(self.mask_dir, exist_ok=True)
        self.mesh_path = osp.join("obj_models", self.obj, "nontextured.stl")
        self.points_path = osp.join(self.base, "sampled_points.npy")
        self.poses_path = osp.join(self.base, "sensor_poses.npy")

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

    def get_num_candidates(self):
        return self.samples * self.sensing_cfg.candidate_multiplier

    def num_faces(self):
        return self.mesh.faces.shape[0]

    # 후보점 생성 함수 (candidate_points)
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
        # 향후 Poisson Disk Sampling이나 다른 기법으로 후보점 집합을 후처리할 수 있음
        return candidate_points, candidate_faces

    # FPS (Farthest Point Sampling) 함수 수정
    def farthest_point_sampling(self, points, candidate_faces):
        """
        후보점 집합(points)에서 FPS를 수행해 num_samples 개의 점과 해당하는 face 인덱스를 반환합니다.
        각 후보점 주변의 local density 정보를 반영해 센싱 밀도가 낮은 영역이 더 많이 선택되도록 합니다.
        """
        # 기본 density_radius: 메쉬 bounding box 대각선 길이 / density_radius_factor
        diag = np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0])
        density_radius = diag / self.sensing_cfg.density_radius_factor
        density_boost = 1.0  # (추후 파라미터 튜닝에 활용 가능)

        N = points.shape[0]
        sampled_indices = []
        distances = np.full(N, np.inf)

        # density_radius가 None인 경우: 메쉬 bounding box의 크기에 따라 조정 (여기서는 거의 사용되지 않음)
        if density_radius is None:
            diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            density_radius = diag / 50.0

        # 각 후보점의 local density 계산
        densities = compute_candidate_density(points, density_radius)
        dens_min = densities.min()
        dens_max = densities.max()
        # normalized_density: 값이 낮을수록 주변 후보점이 적은 (빈) 영역이라고 판단
        normalized_density = (densities - dens_min) / (dens_max - dens_min + 1e-12)

        # 첫 번째 점은 고정 혹은 랜덤 선택 (여기서는 0번 인덱스)
        first_idx = 0
        sampled_indices.append(first_idx)

        last_point = points[first_idx]
        dists = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dists)

        # 작은 epsilon 값으로 0 나누기 방지 (보통 1e-6 정도)
        epsilon = 1e-6

        for i in range(1, self.samples):
            # inverse weighting: 밀도가 낮은 영역에서는 normalized_density 값이 작으므로 effective_distance가 커짐
            effective_distance = distances / (normalized_density + epsilon)
            next_idx = np.argmax(effective_distance)
            sampled_indices.append(next_idx)
            last_point = points[next_idx]
            dists = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dists)

        sampled_points = points[sampled_indices]
        sampled_faces = candidate_faces[sampled_indices]
        return sampled_points, sampled_faces
    def get_random_trajectory(self):
        return random_geodesic_poses(self.mesh, self.render_cfg.shear_mag,N=self.sensing_cfg.samples)

    def get_manual_trajectory(self,poses=None):
        points = None
        if poses is not None:
            points = poses[:,:3,3]
        return random_manual_poses(self.mesh_path, self.render_cfg.shear_mag, points,lc=0.001)

    def get_points_poses(self):
        print(f" [INFO] Generating {self.num_candidates} candidate points for sampling...")
        candidates, candidate_faces = self.candidate_points()
        print(f" [INFO] Applying Farthest Point Sampling (with density boost) to select {self.samples} points...")
        points, faces = self.farthest_point_sampling(candidates, candidate_faces)
        print(f"  -> Final selected points: {len(points)}")

        normals = compute_face_normals(self.mesh, faces)
        shear_rad = math.radians(self.render_cfg.shear_mag)
        delta = np.random.uniform(0.0, 2 * math.pi, size=len(points))
        # poses = pose_from_vertex_normal(points, normals, shear_rad, delta)
        poses = pose_from_vertex_normal(points,normals)
        return  poses

    def sensing(self,poses):
        total_poses = len(poses)
        BATCH_SIZE = self.sensing_cfg.batch_size
        start_idx = 0

        print(f" [INFO] Rendering {total_poses} poses with batch_size={BATCH_SIZE}")
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

        print(f" [DONE] Coverage scan done, total poses={total_poses}")

    def save_results(self, poses, idx_offset):
        points = poses[:,:3, 3]
        print(" [INFO] Saving final results...")
        save_heightmaps(self.all_hm, self.heightmap_dir, idx_offset)
        save_contactmasks(self.all_cm, self.mask_dir, idx_offset)
        save_images(self.all_img, self.image_dir, idx_offset)
        np.save(self.points_path, points)
        np.save(self.poses_path, poses)
        print(f" [DONE] Results in {self.base}\n")

    def show_heatmap(self, poses, visible=False):
        print(f" [INFO] Stiffness map creating...")
        points = poses[:,:3, 3]
        org_k = self.get_local_stiffness(points, self.mesh_bounds)
        norm_k = self.k_normalize(org_k)

        heatmap = self.show_colored_with_pyvista(self.mesh, points, norm_k)

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


def skew_matrix(v: np.ndarray) -> np.ndarray:
    """
    주어진 3D 벡터 v (shape: (3,) 또는 (N,3))에 대한 skew-symmetric matrix를 반환합니다.
    """
    v = np.atleast_2d(v)
    N = v.shape[0]
    # 각 벡터에 대해 3x3 skew matrix 생성
    mat = np.zeros((N, 3, 3))
    mat[:, 0, 1] = -v[:, 2]
    mat[:, 0, 2] = v[:, 1]
    mat[:, 1, 0] = v[:, 2]
    mat[:, 1, 2] = -v[:, 0]
    mat[:, 2, 0] = -v[:, 1]
    mat[:, 2, 1] = v[:, 0]
    # 만약 단일 벡터면 3x3 반환
    if N == 1:
        return mat[0]
    return mat


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues 공식을 이용해 주어진 축(axis)과 각도(angle, 라디안)에 대한 3x3 회전 행렬을 계산합니다.
    """
    axis = axis / np.linalg.norm(axis)
    K = skew_matrix(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def rotation_matrix_from_normal(normal: np.ndarray, ref: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """
    주어진 법선 벡터(normal)를 기준 벡터(ref, 기본적으로 [0,0,1])에 맞추는 결정론적 회전 행렬을 계산합니다.
    만약 법선과 기준 벡터가 거의 동일하면 항등행렬을, 반대면 180도 회전 행렬을 반환합니다.
    """
    normal = normal / np.linalg.norm(normal)
    ref = ref / np.linalg.norm(ref)
    dot = np.dot(normal, ref)
    # 거의 동일하면 항등행렬 반환
    if dot > 0.9999:
        return np.eye(3)
    # 거의 반대이면, 임의의 수직축을 선택해 180도 회전
    if dot < -0.9999:
        axis = np.cross(ref, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(ref, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return rotation_matrix(axis, np.pi)
    # 일반적인 경우: 회전축과 회전각 결정
    axis = np.cross(ref, normal)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return rotation_matrix(axis, angle)


def pose_from_vertex_normal(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    주어진 vertices와 normals로부터 각 샘플에 대해 결정론적 SE(3) 변환 행렬 T를 생성합니다.

    입력:
        vertices: (N, 3) 배열 – 각 센서의 위치
        normals: (N, 3) 배열 – 각 센서의 표면 법선
    출력:
        T: (N, 4, 4) 배열 – 각 샘플에 대한 SE(3) 변환 행렬
           T[:3, :3]는 회전 행렬 (법선이 [0,0,1]과 일치하도록 결정론적으로 계산),
           T[:3, 3]는 translation (즉, vertex 위치)
    """
    vertices = np.atleast_2d(vertices)
    normals = np.atleast_2d(normals)
    num_samples = vertices.shape[0]
    T = np.zeros((num_samples, 4, 4))
    T[:, 3, 3] = 1
    T[:, :3, 3] = vertices

    # 각 샘플에 대해 결정론적 회전 행렬 계산:
    R_all = []
    for i in range(num_samples):
        R = rotation_matrix_from_normal(normals[i], ref=np.array([0, 0, 1]))
        R_all.append(R)
    R_all = np.stack(R_all, axis=0)  # (N, 3, 3)
    T[:, :3, :3] = R_all
    return T