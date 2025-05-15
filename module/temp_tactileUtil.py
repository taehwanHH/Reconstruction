import numpy as np
import torch
from PIL import Image
import open3d as o3d
from os import path as osp
from module.model.Classifier import build_classifier_model
from midastouch.render.digit_renderer import digit_renderer
from functools import cached_property

import os
import trimesh
import pyvista as pv
from scipy.interpolate import Rbf
from skimage import measure
from collections import defaultdict, deque

import pymeshlab as ml

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
    def __init__(self, config):
        super().__init__(config)
        self.obj = config.obj_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.num_samples = config.sensing.num_samples
        self.base = config.data_dir
        self._init_paths()
        self._init_poses()


        self.classifier,_ = build_classifier_model()


    def _init_paths(self):
        base = self.base
        stl = osp.join(base, "recon_stl")
        os.makedirs(stl, exist_ok=True)
        self.data_path = base
        self.stl_path = stl
        self.points_path = osp.join(base, "sampled_points.npy")
        self.sensor_poses_path = osp.join(base, "sensor_poses.npy")
        self.heightmap_dir = osp.join(base, "recon")
        self.mask_dir = osp.join(base, "recon")
        self.stl_filename = osp.join(stl, f"{self.num_samples}_re_mesh.stl")

    def _init_poses(self):
        if os.path.exists(self.sensor_poses_path):
            self.poses = np.load(self.sensor_poses_path)
            self.total_frames=self.poses.shape[0]


    @property
    def points(self):
        points_np = self.poses[:,:3, 3]
        return torch.tensor(points_np, device=self.device)

    # @cached_property
    # def points_np(self):
    #     return np.load(self.points_path)
    #
    # @cached_property
    # def points(self):
    #     return torch.tensor(self.points_np, device=self.device)



    @cached_property
    def transform(self):
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
        hm_path = osp.join(self.heightmap_dir, f"{idx:04d}.jpg")
        mask_path = osp.join(self.mask_dir, f"{idx:04d}.jpg")
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
        # with torch.no_grad():
        #     output = self.model(hm_img.unsqueeze(0).to(self.device))
        #     _, predicted_k = torch.max(output, 1)
        # (predicted_k 값은 필요 시 저장)


        # Tensor 변환
        heightmap = torch.tensor(heightmap_np, device=self.device)
        contact_mask = torch.tensor(mask_np, device=self.device).float()
        contact_mask = (contact_mask > 0).float()

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

        return global_points

    def process_all_frame(self):
        all_gp = []
        pred_k = []
        for frame in range(self.total_frames):
            gp, k_hat= self.process_frame(frame)
            all_gp.append(gp)
            # pred_k.append(k_hat.item())
        all_gp_tensor = merge_points(all_gp)
        return all_gp_tensor

    def process_part_frame(self, indices):
        all_gp = []
        pred_k = []
        for idx in indices:
            gp= self.process_frame(idx)
            all_gp.append(gp)
            # pred_k.append(k_hat.item())
        all_gp_tensor = merge_points(all_gp)
        return all_gp_tensor

    def pcd2mesh(self, all_points_tensor, depth,down_sample=None):
        # p = merge_points(all_points)
        # if p is None:
        #     print("[ERROR] No frames were processed.")
        #     return
        pcd = self.prepare_point_cloud(all_points=all_points_tensor, down_sample=down_sample)
        # mesh = reconstruct_mesh_from_pcd(pcd, alpha=0.01)


        mesh = poisson_pcd_to_mesh(pcd,depth=depth)
        # mesh = reconstruct_mesh_bpa(pcd)
        # vertices = np.asarray(mesh.vertices)
        # faces = np.asarray(mesh.triangles)
        # mesh_trimesh = trimesh.Trimesh(vertices, faces)

        return mesh



    def stl_export(self, mesh, visible=False):
        # o3d.io.write_triangle_mesh(self.stl_filename, mesh)
        mesh.export(self.stl_filename)
        print(f" [INFO] Mesh has been saved to {self.stl_filename}.")

        if visible:
            # o3d.visualization.draw_geometries([mesh], window_name="3D Reconstruction Mesh",width=900, height=600)
            dargs = dict(
                color="grey",
                ambient=0.5,
                opacity=1,
                smooth_shading=True,

                specular=1.0,
                show_scalar_bar=False,
                render=False,
            )

            plotter = pv.Plotter()
            plotter.add_mesh(mesh, **dargs)
            plotter.add_axes()
            plotter.show(title="STL Model Viewer")

    def stiff_map(self,pred_k,visible=False):
        k_list = self.k_values[pred_k]
        norm_res = (k_list - self.k_values.min()) / (self.k_values.max() - self.k_values.min() + 1e-8)
        rec_shape = trimesh.load_mesh(self.stl_filename)
        colored_mesh = self.color_mesh_by_stiffness(rec_shape, self.points[self.sampled_indices].cpu().numpy(), norm_res,
                                               colormap_name='viridis')
        if visible:
            colored_mesh.show()

    def prepare_point_cloud(self, all_points,down_sample=None):

        max_points = self.cfg.get("max_points", 80_000_000)
        if all_points.shape[0] > max_points:
            indices = torch.randperm(all_points.shape[0])[:max_points]
            all_points = all_points[indices]
        # print(f" [INFO] Total points after filtering: {all_points.shape[0]}")
        all_points_np = all_points.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points_np)
        if down_sample is not None:
            return pcd
        voxel_size = self.cfg.get("voxel_size", 0.00015)
        # voxel_size = cfg.get("voxel_size", 0.0002)
        pcd = pcd.voxel_down_sample(voxel_size)
        nb_neighbors = self.cfg.get("nb_neighbors", 10)
        std_ratio = self.cfg.get("std_ratio", 2.0)
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 중요! 메쉬 생성 전 법선(normal) 계산 필요
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30)
        )

        # 법선 방향을 메쉬에 적합하도록 정렬
        pcd.orient_normals_consistent_tangent_plane(100)

        print(f" [INFO] Number of points after downsampling and outlier removal: {len(pcd.points)}")
        return pcd

def merge_points(all_points_list):
    """
    여러 프레임에서 얻은 global point들을 병합 및 필터링.
    """
    if not all_points_list:
        return None
    all_points = torch.cat([p.cpu() for p in all_points_list], dim=0)
    all_points = all_points[torch.all(torch.isfinite(all_points), dim=1)]
    return all_points




def reconstruct_mesh_from_pcd(pcd, alpha):
    print(f" [INFO] Performing Alpha Shape Reconstruction (alpha={alpha})...")

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

    if is_watertight(mesh_rec):
        print("Reconstructed mesh is watertight!")
    return mesh_rec

def poisson_pcd_to_mesh(pcd, depth=9):

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(pcd.points),
                        v_normals_matrix=np.asarray(pcd.normals)))

    ms.generate_surface_reconstruction_screened_poisson(depth=depth)

    mesh_pl = ms.current_mesh()

    # PyMeshLab mesh에서 vertices와 faces 배열을 추출
    vertices = mesh_pl.vertex_matrix()  # numpy array, shape (N, 3)
    faces = mesh_pl.face_matrix()  # numpy array, shape (M, 3)

    # trimesh 객체로 변환
    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh_tm

def pcd_to_mesh(pcd, depth=11 , density_threshold=0.01):
    """
    다운샘플된 포인트 클라우드를 입력받아 Poisson Surface Reconstruction 기반으로 메쉬를 생성합니다.

    Args:
        pcd (open3d.geometry.PointCloud): 다운샘플된 포인트 클라우드.
        depth (int): Poisson 재구성의 octree 깊이. 값이 높을수록 세밀한 디테일 반영.
        density_threshold (float): 0~1 사이의 값으로, 낮은 밀도에 해당하는 영역을 제거하기 위한 백분위수.
                                     (예를 들어 0.05이면 하위 5%의 밀도값을 가진 정점들이 제거됩니다.)

    Returns:
        mesh (open3d.geometry.TriangleMesh): 필터링 후 최종 메쉬.
    """
    # 1. 노말 추정: Poisson 재구성은 정교한 노말 정보가 필요합니다.
    print(" [INFO] Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 2. Poisson Surface Reconstruction 수행
    print(f" [INFO] Performing Poisson Surface Reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)

    # 3. 밀도 기반 필터링: 낮은 밀도(신뢰도가 낮은) 정점 제거
    densities = np.asarray(densities)
    density_quantile = np.quantile(densities, density_threshold)
    print(
        f" [INFO] Removing vertices with density lower than the {density_threshold * 100:.1f}th percentile ({density_quantile:.6f})...")
    vertices_to_remove = densities < density_quantile
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4. 후처리: 삼각형 중복 제거, 노말 재계산
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    mesh.filter_smooth_taubin()


    return mesh



def is_watertight(mesh):
    """
    주어진 Open3D TriangleMesh가 watertight(즉, 경계(edge)가 없는지) 여부를 판단합니다.
    모든 edge가 2개의 삼각형에 공유되어야 합니다.
    """
    triangles = np.asarray(mesh.triangles)
    # 각 삼각형의 3개 edge를 추출합니다.
    edges = np.concatenate([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ], axis=0)
    # 각 edge의 순서를 정렬하여 동일 edge가 다른 순서로 나오는 문제를 해결합니다.
    edges = np.sort(edges, axis=1)
    # 고유 edge와 각 edge가 등장한 횟수를 구합니다.
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    # watertight하면 모든 고유 edge의 count가 2가 되어야 합니다.
    return np.all(counts == 2)


