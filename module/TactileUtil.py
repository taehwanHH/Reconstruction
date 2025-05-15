import numpy as np
import torch
from torch.utils.data import TensorDataset
from PIL import Image
import open3d as o3d
from os import path as osp
from module.model import build_model, build_classifier_model
from module.Stiffness import Stiffness
from midastouch.render.digit_renderer import digit_renderer
from functools import cached_property

import os
import trimesh

import pyvista as pv

import pymeshlab as ml




class TactileMap(Stiffness):
    def __init__(self, config, stl_filename=None):
        super().__init__(config)
        self.obj = config.obj_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.num_samples = config.sensing.num_samples
        self.base = config.data_dir
        self.stl_filename = stl_filename
        self.scheme = config.sim.scheme
        self._init_paths()
        self._init_poses()
        self._init_classifier()


    def _init_paths(self):
        base = self.base
        self.data_path = base
        self.points_path = osp.join(base, "sampled_points.npy")
        self.sensor_poses_path = osp.join(base, "sensor_poses.npy")
        self.heightmap_dir = osp.join(base, "gt_heightmaps")
        self.mask_dir = osp.join(base, "gt_contactmasks")

    def _init_poses(self):
        if os.path.exists(self.sensor_poses_path):
            self.poses = np.load(self.sensor_poses_path)
            self.total_frames=self.poses.shape[0]

    def _init_classifier(self):
        _fe_model_cfg = self.cfg.feat_extractor.model
        _cls_model_cfg = self.cfg.k_classifier.model
        _model, _ = build_model(_fe_model_cfg)
        _encoder = _model.get_encoder()

        self.classifier,_ = build_classifier_model(_encoder, _cls_model_cfg)
        self.classifier.load_state_dict(torch.load(_cls_model_cfg.saved_path))
        self.classifier.eval()
        print(f" [INFO] Classifier model loaded from {_cls_model_cfg.saved_path}")



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



    # @cached_property
    # def transform(self):
    #     return TRANSFORM

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
            # heightmap_np = np.array(transforms.Resize((320,240),interpolation=transforms.InterpolationMode.BICUBIC)(hm))
            # mask_np = np.array(transforms.Resize((320,240),interpolation=transforms.InterpolationMode.BICUBIC)(cm)).astype(np.uint8)
            heightmap_np = np.array(hm.resize((240,320),resample=Image.Resampling.BICUBIC))
            mask_np = np.array(cm.resize((240,320),resample=Image.Resampling.BICUBIC)).astype(np.uint8)

        except Exception as e:
            print(f"[ERROR] Failed to load images for frame {idx}: {e}")
            return None

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

        return global_points

    def process_all_frame(self):
        all_gp = []
        for frame in range(self.total_frames):
            gp = self.process_frame(frame)
            all_gp.append(gp)
        all_gp_tensor = merge_points(all_gp)

        return all_gp_tensor

    def process_part_frame(self, indices):
        all_gp = []
        for idx in indices:
            gp = self.process_frame(idx)
            all_gp.append(gp)
        all_gp_tensor = merge_points(all_gp)

        return all_gp_tensor


    def pcd2mesh(self, all_points_tensor,  depth, down_sample=None):
        # p = merge_points(all_points)
        # if p is None:
        #     print("[ERROR] No frames were processed.")
        #     return
        pcd = self.prepare_point_cloud(all_points=all_points_tensor, down_sample=down_sample)

        print(f" [INFO] Mesh reconstruction started. This may take a while...")
        mesh = poisson_pcd_to_mesh(pcd, depth=depth)
        print(f" [INFO] Mesh reconstruction finished.")

        # if all_ks is not None:
        #     norm_res = self.k_normalize(all_ks)
        #     sampled_points = self.points[self.sampled_indices].cpu().numpy()

            # colored_mesh = self.show_colored_with_pyvista(mesh, sampled_points , norm_res)
        return mesh

    def stiffness_tuple(self,ks_tensor):
        sampled_points = self.points[self.sampled_indices].cpu()
        pred_k_indices =  ks_tensor[self.sampled_indices].cpu()
        sampled_pred_k = np.asarray([self.k_values[i] for i in pred_k_indices])

        pred_k_tuple = (sampled_points, sampled_pred_k)

        return pred_k_tuple


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
        voxel_size = self.cfg.get("voxel_size", 0.0002)
        # voxel_size = cfg.get("voxel_size", 0.0002)
        pcd = pcd.voxel_down_sample(voxel_size)
        nb_neighbors = self.cfg.get("nb_neighbors", 10)
        std_ratio = self.cfg.get("std_ratio", 2.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        print(f" [INFO] Number of points after downsampling and outlier removal: {len(pcd.points)}")

        # 중요! 메쉬 생성 전 법선(normal) 계산 필요
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30)
        )

        # 법선 방향을 메쉬에 적합하도록 정렬
        pcd.orient_normals_consistent_tangent_plane(100)

        return pcd

    def show_stiffness_map(self, all_ks, visible=False):
        if visible:
            norm_res = self.k_normalize(all_ks)
            rec_mesh = trimesh.load_mesh(self.stl_filename)
            colored_mesh = self.show_colored_with_pyvista(rec_mesh,)


def merge_points(all_points_list):
    """
    여러 프레임에서 얻은 global point들을 병합 및 필터링.
    """
    if not all_points_list:
        return None
    all_points = torch.cat([p.cpu() for p in all_points_list], dim=0)
    all_points = all_points[torch.all(torch.isfinite(all_points), dim=1)]
    return all_points



def poisson_pcd_to_mesh(pcd, depth=9):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(pcd.points),
                        v_normals_matrix=np.asarray(pcd.normals)))

    ms.generate_surface_reconstruction_screened_poisson(depth=depth, fulldepth=6, cgdepth=3)

    mesh_pl = ms.current_mesh()

    # PyMeshLab mesh에서 vertices와 faces 배열을 추출
    vertices = mesh_pl.vertex_matrix()  # numpy array, shape (N, 3)
    faces = mesh_pl.face_matrix()  # numpy array, shape (M, 3)

    # trimesh 객체로 변환
    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh_tm


