import numpy as np
import torch
from PIL import Image
import open3d as o3d
from os import path as osp
from module.TSN import TRANSFORM, TSN_COPY
from midastouch.render.digit_renderer import digit_renderer
from functools import cached_property

import os
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


class Stiffness:
    def __init__(self, stiff_config):
        self.k_cfg = stiff_config.render.k

    @cached_property
    def k_values(self):
        k_min, k_max, k_interval= self.k_cfg.min, self.k_cfg.max, self.k_cfg.interval
        return np.array(list(range(k_min, k_max + 1, k_interval)))
    @cached_property
    def k_num(self):
        return self.k_values.shape[0]

    def k_normalize(self,k):
        return (k - self.k_values.min()) / (self.k_values.max() - self.k_values.min() + 1e-8)


    def get_local_stiffness(self,positions, bounds):
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
        region_idx = (z_idx * 4 + xy_idx) % 8  # 0 ~ 15

        stiffness_candidates = self.k_values
        return stiffness_candidates[region_idx]

    def color_mesh_by_stiffness(self, mesh, sample_positions, sample_stiffness, colormap_name='viridis'):
        # KD-tree를 구축하여 각 버텍스에 대해 가장 가까운 contact point 찾기
        tree = cKDTree(sample_positions)
        vertex_positions = mesh.vertices  # (M, 3)
        distances, indices = tree.query(vertex_positions)

        # 각 버텍스에 대해 stiffness 값 할당
        vertex_stiffness = sample_stiffness[indices]


        # colormap 적용 (RGBA -> RGB 추출)
        cmap = plt.get_cmap(colormap_name)
        vertex_colors = cmap(vertex_stiffness)[:, :3]  # (M, 3), 값은 0~1

        # trimesh에서는 vertex_colors가 0~255 uint8 형식이어야 함
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        return mesh


class TactileMap(Stiffness):
    def __init__(self, obj, config):
        super().__init__(config)
        self.obj = obj
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.num_samples = config.sensing.num_samples
        self._init_paths()
        self.model = TSN_COPY(num_k = self.k_num)


    def _init_paths(self):
        base = osp.join("data", "sim", self.obj, "full_coverage")
        stl = osp.join(base, "recon_stl")
        os.makedirs(stl, exist_ok=True)
        self.data_path = base
        self.stl_path = stl
        self.points_path = osp.join(base, "sampled_points.npy")
        self.sensor_poses_path = osp.join(base, "sensor_poses.npy")
        self.heightmap_dir = osp.join(base, "gt_heightmaps")
        self.mask_dir = osp.join(base, "gt_contactmasks")
        self.stl_filename = osp.join(stl, f"{self.num_samples}_re_mesh.stl")


    @cached_property
    def points_np(self):
        return np.load(self.points_path)

    @cached_property
    def points(self):
        return torch.tensor(self.points_np, device=self.device)

    @cached_property
    def total_frames(self):
        return len(self.points)

    @cached_property
    def poses(self):
        if os.path.exists(self.sensor_poses_path):
            return np.load(self.sensor_poses_path)
        return None

    @cached_property
    def transform(self):
        # 만약 TRANSFORM을 사용한다면 그대로 사용하고,
        # 아니라면 필요한 transform을 정의하세요.
        return TRANSFORM

    @cached_property
    def renderer(self):
        return digit_renderer(cfg=self.cfg.render, obj_path=None, headless=self.cfg.render.headless)

    @cached_property
    def sampled_indices(self):
        return self.fps_torch(self.num_samples)

    def fps_torch(self,num_samples):
        """
        PyTorch를 이용해 GPU에서 빠르게 FPS를 수행하는 간단한 구현.
        points: (N, 3) tensor (device가 GPU여야 함)
        returns: 선택된 인덱스 tensor (int64)
        """
        N, _ = self.points.shape
        if num_samples >= N:
            return torch.arange(N, device=self.device)

        selected = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        distances = torch.full((N,), float('inf'), device=self.device)

        # 첫 번째 점은 랜덤 선택
        # selected[0] = torch.randint(0, N, (1,), device=points.device)
        selected[0] = 0

        last = self.points[selected[0]].unsqueeze(0)  # (1,3)
        # 초기 거리 계산
        distances = torch.norm(self.points - last, dim=1)

        for i in range(1, num_samples):
            # 다음 점: 현재까지의 최소 거리 중 최대인 값 선택
            selected[i] = torch.argmax(distances)
            last = self.points[selected[i]].unsqueeze(0)
            new_dists = torch.norm(self.points - last, dim=1)
            distances = torch.minimum(distances, new_dists)

        return selected


    def process_frame(self, idx):
        """
        하나의 프레임(idx)에 대해:
          - heightmap과 mask를 로드
          - 모델 추론을 통한 stiffness 예측 (predicted_k는 리스트에 추가)
          - local point cloud 생성 및 센서 좌표계 -> global 좌표 변환
        반환: global point cloud (tensor)
        """
        hm_path = osp.join(self.heightmap_dir, f"{idx}.jpg")
        mask_path = osp.join(self.mask_dir, f"{idx}.jpg")
        try:
            hm = Image.open(hm_path)
            cm = Image.open(mask_path)
            heightmap_np = np.array(hm)
            mask_np = np.array(cm).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Failed to load images for frame {idx}: {e}")
            return None

        # 모델 추론 (stiffness 예측)
        hm_img = TRANSFORM(hm.convert('L'))
        with torch.no_grad():
            output = self.model(hm_img.unsqueeze(0).to(self.device))
            _, predicted_k = torch.max(output, 1)
        # (predicted_k 값은 필요 시 저장)

        # Tensor 변환
        heightmap = torch.tensor(heightmap_np, device=self.device)
        contact_mask = torch.tensor(mask_np, device=self.device).float()
        contact_mask = (contact_mask > 128).float()

        # Local point cloud 생성 (renderer 내 함수)
        local_points = self.renderer.heightmap2Pointcloud(heightmap, contact_mask)

        # 센서 중앙 좌표 계산
        H, W = heightmap.shape
        center_y, center_x = H // 2, W // 2
        depth_corr = self.renderer.correct_image_height_map(heightmap, output_frame="cam")
        depth_center = depth_corr[center_y, center_x]
        f = self.renderer.renderer.f
        w_img = self.renderer.renderer.width / 2.0
        h_img = self.renderer.renderer.height / 2.0
        center_point = torch.tensor([
            ((center_x - w_img) / f) * depth_center,
            -((center_y - h_img) / f) * depth_center,
            -depth_center
        ], device=self.device, dtype=local_points.dtype)
        local_points_centered = local_points - center_point

        # Global transformation (sensor pose 적용)
        if self.poses is not None:
            pose = self.poses[idx]  # 4x4 matrix (NumPy)
            sensor_pose = torch.tensor(pose, device=self.device, dtype=local_points.dtype)
            R = sensor_pose[:3, :3]
            Rz = torch.tensor([[-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0]], device=self.device, dtype=local_points.dtype)
            local_points_rotated = (Rz @ local_points_centered.T).T
            rotated_points = (R @ local_points_rotated.T).T
            contact_pt = self.points[idx]
            global_points = rotated_points + contact_pt
        else:
            contact_pt = torch.tensor(self.points[idx], device=self.device, dtype=local_points.dtype)
            global_points = local_points_centered + contact_pt

        return global_points, predicted_k


    def pcd2mesh(self, all_points, alpha):
        p = merge_points(all_points)
        if p is None:
            print("[ERROR] No frames were processed.")
            return
        pcd = prepare_point_cloud(all_points=p,cfg=self.cfg)
        mesh = reconstruct_mesh_from_pcd(pcd, alpha)
        return mesh

    def stl_export(self, mesh, visible=False):
        o3d.io.write_triangle_mesh(self.stl_filename, mesh)
        print(f"[INFO] Mesh has been saved to {self.stl_filename}.")

        if visible:
            o3d.visualization.draw_geometries([mesh], window_name="3D Reconstruction Mesh",width=900, height=600)

    def stiff_map(self,pred_k,visible=False):
        k_list = self.k_values[pred_k]
        norm_res = (k_list - self.k_values.min()) / (self.k_values.max() - self.k_values.min() + 1e-8)
        rec_shape = trimesh.load_mesh(self.stl_filename)
        colored_mesh = self.color_mesh_by_stiffness(rec_shape, self.points[self.sampled_indices].cpu().numpy(), norm_res,
                                               colormap_name='viridis')
        if visible:
            colored_mesh.show()

def merge_points(all_points_list):
    """
    여러 프레임에서 얻은 global point들을 병합 및 필터링.
    """
    if not all_points_list:
        return None
    all_points = torch.cat([p.cpu() for p in all_points_list], dim=0)
    all_points = all_points[torch.all(torch.isfinite(all_points), dim=1)]
    return all_points


def prepare_point_cloud(all_points, cfg):

    max_points = cfg.get("max_points", 80_000_000)
    if all_points.shape[0] > max_points:
        indices = torch.randperm(all_points.shape[0])[:max_points]
        all_points = all_points[indices]
    # print(f"[INFO] Total points after filtering: {all_points.shape[0]}")
    all_points_np = all_points.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_np)
    voxel_size = cfg.get("voxel_size", 0.00025)
    pcd = pcd.voxel_down_sample(voxel_size)
    nb_neighbors = cfg.get("nb_neighbors", 20)
    std_ratio = cfg.get("std_ratio", 2.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[INFO] Number of points after downsampling and outlier removal: {len(pcd.points)}")
    return pcd

def reconstruct_mesh_from_pcd(pcd, alpha):
    print(f"[INFO] Performing Alpha Shape Reconstruction (alpha={alpha})...")

    try:
        mesh_rec = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if len(mesh_rec.vertices) == 0:
            print("[WARN] The reconstructed mesh is empty. Increasing alpha value and retrying.")
            alpha *= 2
            mesh_rec = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    except Exception as e:
        print(f"[ERROR] Alpha Shape Reconstruction failed: {e}")
        return None
    if len(mesh_rec.vertices) == 0:
        print("[ERROR] The reconstructed mesh is still empty. Please adjust the alpha parameter.")
        return None
    mesh_rec.remove_degenerate_triangles()
    mesh_rec.compute_vertex_normals()
    return mesh_rec