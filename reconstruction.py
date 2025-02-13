#!/usr/bin/env python3
import hydra
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from os import path as osp
from tqdm import tqdm
from omegaconf import DictConfig
from midastouch.render.digit_renderer import digit_renderer  # digit_renderer 임포트
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import trimesh
matplotlib.use('TkAgg')  # 인터랙티브 백엔드 명시 (환경에 따라 변경)

from mpl_toolkits.mplot3d import Axes3D  # 이 줄은 3D 프로젝션을 활성화함

# sensor STL 모델을 한 번만 로드 (경로는 환경에 맞게 수정)
sensor_model_path = osp.join("STL","digit.STL")  # 예: 현재 디렉토리의 digit.STL
if osp.exists(sensor_model_path):
    sensor_mesh = trimesh.load(sensor_model_path)
    # sensor_mesh.vertices: (N,3) NumPy array
    sensor_vertices = torch.tensor(sensor_mesh.vertices, device="cuda", dtype=torch.float)
    print(f"[INFO] digit.STL 파일을 성공적으로 로드하였습니다. (정점 수: {sensor_vertices.shape[0]})")

else:
    sensor_vertices = None
    print("[WARN] Sensor STL 파일을 찾을 수 없습니다.")


def merge_points_with_sensor_direction(points, sensor_dirs, merge_voxel_size):
    """
    (후처리 옵션) 각 포인트를 merge_voxel_size 크기의 voxel로 그룹화하여,
    그룹 내에서 글로벌 z-축([0,0,1])과의 내적(수직성)이 가장 높은 점을 대표값으로 선택.
    """
    voxel_indices = np.floor(points / merge_voxel_size).astype(np.int32)
    groups = {}
    for i, voxel in enumerate(voxel_indices):
        key = tuple(voxel)
        groups.setdefault(key, []).append(i)
    merged_points = []
    merged_sensor_dirs = []
    ref = np.array([0, 0, 1])
    for key, indices in groups.items():
        if len(indices) == 1:
            merged_points.append(points[indices[0]])
            merged_sensor_dirs.append(sensor_dirs[indices[0]])
        else:
            best_idx = indices[0]
            best_score = np.dot(sensor_dirs[indices[0]], ref)
            for idx in indices[1:]:
                score = np.dot(sensor_dirs[idx], ref)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            merged_points.append(points[best_idx])
            merged_sensor_dirs.append(sensor_dirs[best_idx])
    return np.array(merged_points), np.array(merged_sensor_dirs)


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def reconstruction(cfg: DictConfig):
    # 데이터 경로 설정
    obj_model = cfg.obj_model
    data_path = osp.join("data", "sim", obj_model, "full_coverage")
    points_path = osp.join(data_path, "sampled_points.npy")
    sensor_poses_path = osp.join(data_path, "sensor_poses.npy")
    heightmap_dir = osp.join(data_path, "gt_heightmaps")
    mask_dir = osp.join(data_path, "gt_contactmasks")
    # heightmap_dir = osp.join(data_path, "tdn_hm")
    # mask_dir = osp.join(data_path, "tdn_cm")
    cam_poses_path = osp.join(data_path, "cam_poses.npy")

    # 저장된 sampled_points.npy를 GPU tensor로 로드 (각 프레임의 contact point; shape: (num_frames, 3))
    sampled_points = torch.tensor(np.load(points_path), device="cuda")
    num_frames = len(sampled_points)

    # sensor_poses가 존재하면 로드 (shape: (num_frames, 4, 4)); 이 pose는 센서가 표면을 바라보는 각도를 담고 있음.
    if osp.exists(sensor_poses_path):
        sensor_poses = np.load(sensor_poses_path)
    else:
        sensor_poses = None

    # cam_poses.npy 파일이 있다면, 이를 로드합니다.
    if osp.exists(cam_poses_path):
        cam_poses = np.load(cam_poses_path)  # shape: (num_frames, 4, 4)
    else:
        cam_poses = None
        print("[WARN] gel_poses.npy 파일을 찾을 수 없습니다. cam_poses를 사용할 수도 있습니다.")

    # 인터랙티브 모드 활성화 (기본적으로 백엔드가 인터랙티브한 경우)
    plt.ion()

    # figure와 3D axis 생성
    fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.set_xlim(-0.001, 0.001)
    # ax.set_ylim(-0.001, 0.001)
    # ax.set_zlim(-0.001, 0.001)
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # # ax.set_xlim(-0.001, 0.001)
    # # ax.set_ylim(-0.001, 0.001)
    # # ax.set_zlim(-0.001, 0.001)
    ax3= fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-0.001, 0.001)
    # ax.set_ylim(-0.001, 0.001)
    # ax.set_zlim(-0.001, 0.001)

    print(f"[INFO] {obj_model} reconstruction 시작...")
    time.sleep(1)


    # digit_renderer 인스턴스 생성 (headless 옵션에 따라)
    renderer = digit_renderer(cfg=cfg.render, obj_path=None, headless=cfg.render.headless)

    # 모든 프레임의 global point cloud를 저장할 리스트
    all_points_list = []

    for i in tqdm(range(num_frames), desc="프레임 처리", unit="frame"):
        hm_path = osp.join(heightmap_dir, f"{i}.jpg")
        mask_path = osp.join(mask_dir, f"{i}.jpg")
        try:
            heightmap_np = np.array(Image.open(hm_path))
            mask_np = np.array(Image.open(mask_path)).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Frame {i} 이미지 로드 실패: {e}")
            continue

        # # --- 2D 중심 보정 종료 ---
        # heightmap 및 contact mask를 GPU tensor로 변환
        heightmap = torch.tensor(heightmap_np, device="cuda")
        contact_mask = torch.tensor(mask_np, device="cuda").float()
        contact_mask = (contact_mask > 128).float()

        # local point cloud 생성 (renderer.heightmap2Pointcloud 내부에서 이미 masking된 점은 제거됨)
        local_points = renderer.heightmap2Pointcloud(heightmap, contact_mask)
        
        # --- 센서 출력 이미지 중앙 (3D 좌표) 계산 및 보정 ---
        # 센서 출력 이미지의 크기를 이용하여 중앙 픽셀 좌표 계산
        H, W = heightmap.shape
        center_y, center_x = H // 2, W // 2

        # 보정된 depth 이미지 (미터 단위)를 통해 중앙 픽셀의 depth 값 추출
        depth_corr = renderer.correct_image_height_map(heightmap, output_frame="cam")
        depth_center = depth_corr[center_y, center_x]

        # 카메라(센서) 내재 파라미터
        f = renderer.renderer.f
        w_img = renderer.renderer.width / 2.0
        h_img = renderer.renderer.height / 2.0

        # 센서 좌표계에서 중앙 픽셀의 3D 좌표 계산
        # (x = ((x - w_img)/f)*depth, y = -((y - h_img)/f)*depth, z = -depth)
        center_point = torch.tensor([
            ((center_x - w_img) / f) * depth_center,
            -((center_y - h_img) / f) * depth_center,
            -depth_center
        ], device=local_points.device, dtype=local_points.dtype)

        # 모든 local point cloud의 점들에서 센터 포인트(center_point)를 빼서, 해당 픽셀이 (0,0,0)에 오도록 함.
        local_points_centered = local_points - center_point

        # 90도 (반시계방향) 회전각 (라디안 단위)
        theta = np.deg2rad(90)

        # 회전 행렬 R_z 정의 (xy 평면 기준)
        R_z = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], device=local_points_centered.device, dtype=local_points_centered.dtype)

        # # local_points_centered에 R_z를 적용하여 회전
        # local_points_centered_rot = (R_z @ local_points_centered.T).T

        # --- 보정 종료 ---

        # --- Global 변환 ---
        # 단계 1: sensor_pose에 의한 회전(및 flip)
        # sensor_pose는 센서가 표면을 바라보는 각도를 담고 있음.
        # Flip 행렬(F = diag(1,1,-1))을 먼저 적용하여, local point cloud가 센서가 바라보는 방향의 반대쪽으로 회전.
        if sensor_poses is not None:
            sensor_pose_np = sensor_poses[i]  # 4x4 행렬 (NumPy)
            sensor_pose = torch.tensor(sensor_pose_np, device=local_points.device, dtype=local_points.dtype)
            R = sensor_pose[:3, :3]
            # 먼저, local_points_centered에 대해 z축 기준 180도 회전 행렬을 적용
            Rz = torch.tensor([[-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 1.0]], device=local_points.device, dtype=local_points.dtype)
            local_points_rotated = (Rz @ local_points_centered.T).T
            # flip 행렬 없이 R만 사용:
            composite_rot = R

            rotated_points = (composite_rot @ local_points_rotated.T).T
            contact_pt = sampled_points[i]
            global_points = rotated_points + contact_pt
        else:
            contact_pt = torch.tensor(sampled_points[i], device=local_points.device, dtype=local_points.dtype)
            global_points = local_points_centered + contact_pt

        # --- Global 변환 종료 ---
        # sensor 모델 변환: sensor_vertices를 현재 프레임의 composite_rot와 contact_pt를 이용하여 변환
        # cam_poses가 있다면 cam_poses를 사용하여 센서 모델을 변환합니다.
        # 만약 cam_poses와 sensor_vertices가 모두 존재하면 cam_poses를 사용하여 센서 모델 변환
        # if cam_poses is not None and sensor_vertices is not None:
        #     # cam_poses는 [x,y,z,qx,qy,qz,qw] 형태로 저장되어 있음.
        #     cam_pose_vec = cam_poses[i]  # [x,y,z,qx,qy,qz,qw]
        #     cam_translation = torch.tensor(cam_pose_vec[:3], device="cuda", dtype=local_points.dtype)
        #     cam_quat = cam_pose_vec[3:]  # [qx, qy, qz, qw]
        #     # 쿼터니언을 회전행렬로 변환 (scipy 사용)
        #     from scipy.spatial.transform import Rotation as R
        #     R_cam_np = R.from_quat(cam_quat).as_matrix()  # (3,3); quaternion order: [qx,qy,qz,qw]
        #     R_cam = torch.tensor(R_cam_np, device="cuda", dtype=local_points.dtype)
        #     # flip 행렬 없이 저장된 데이터 그대로 사용 (즉, composite_rot_cam = R_cam)
        #     composite_rot_cam = R_cam
        #     # 센서 모델 변환: sensor_vertices에 composite_rot_cam를 적용한 후, cam_translation을 더함.
        #     transformed_sensor = (composite_rot_cam @ sensor_vertices.T).T + cam_translation
        #     sensor_pts = transformed_sensor.cpu().numpy()
        #     ax3.scatter(sensor_pts[:, 0], sensor_pts[:, 1], sensor_pts[:, 2],
        #                 s=5, c='green', marker='^', label="Sensor Model (Cam Pose)")
        # elif sensor_vertices is not None:
        #     # cam_poses가 없으면, fallback으로 sensor_poses를 사용 (flip 없이)
        #     sensor_pose_np = sensor_poses[i]
        #     sensor_pose = torch.tensor(sensor_pose_np, device="cuda", dtype=local_points.dtype)
        #     R_sensor = sensor_pose[:3, :3]
        #     composite_rot_sensor = R_sensor  # flip 없이 그대로 사용
        #     transformed_sensor = (composite_rot_sensor @ sensor_vertices.T).T + contact_pt
        #     sensor_pts = transformed_sensor.cpu().numpy()
        #     ax3.scatter(sensor_pts[:, 0], sensor_pts[:, 1], sensor_pts[:, 2],
        #                 s=5, c='green', marker='^', label="Sensor Model (Sensor Pose)")

        # # --- 센서 모델 시각화 종료 ---
        # # --- 디버깅: 매 프레임 local_points_centered의 3D 산점도 표시 ---
        # ax.cla()  # 현재 축 초기화
        # ax2.cla()
        #
        # pts = local_points_centered.cpu().numpy()
        # pts2 = local_points.cpu().numpy()
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='blue', label="Local Points")
        # ax2.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], s=1, c='blue', label="Local Points")
        # # (0,0,0) 위치 강조: 빨간 별로 표시하고, 크기와 레이블 추가
        # ax.scatter(0, 0, 0, s=50, c='red', marker='*', label="Origin (0,0,0)")
        # center_point = center_point.cpu().numpy()
        #
        # ax2.scatter(center_point[0], center_point[1],center_point[2], c='red', marker='*', label="Center")
        #
        # ax.set_title(f"Frame {i}: Local Point Cloud (3D)")
        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_zlabel("Z (m)")
        # ax2.set_xlabel("X (m)")
        # ax2.set_ylabel("Y (m)")
        # ax2.set_zlabel("Z (m)")
        #
        # ax.legend()
        # ax2.legend()
        # ax.set_xlim(-0.02,0.02)
        # ax.set_ylim(-0.02, 0.02)
        # ax.view_init(elev=90, azim=-90)  # 원하는 시점으로 조정
        # ax2.view_init(elev=30, azim=0)  # 원하는 시점으로 조정
        # # plt.draw()
        # # plt.pause(0.1)
        #
        # ax3.view_init(elev=30, azim=0)  # 원하는 시점으로 조정
        #
        # pts3 = global_points.cpu().numpy()
        # ax3.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=20, c='blue', label="Local Points")
        #
        # ax3.set_title(f"Frame {i}: Global Point Cloud (3D)")
        # ax3.set_xlabel("X (m)")
        # ax3.set_ylabel("Y (m)")
        # ax3.set_zlabel("Z (m)")
        # # ax3.legend()
        # plt.draw()
        # plt.pause(0.0001)
        all_points_list.append(global_points)


    if not all_points_list:
        print("[ERROR] 처리된 프레임이 없습니다.")
        return

    print("[INFO] 모든 프레임의 포인트를 병합하는 중...")
    all_points = torch.cat(all_points_list, dim=0)
    all_points = all_points[torch.all(torch.isfinite(all_points), dim=1)]
    print(f"[INFO] 병합 후 총 포인트 수: {all_points.shape[0]}")

    max_points = cfg.get("max_points", 40_000_000)
    if all_points.shape[0] > max_points:
        indices = torch.randperm(all_points.shape[0])[:max_points]
        all_points = all_points[indices]
    print(f"[INFO] 필터링 후 총 포인트 수: {all_points.shape[0]}")

    all_points_np = all_points.cpu().numpy()

    # Open3D 포인트 클라우드 생성 및 후처리
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_np)
    voxel_size = cfg.get("voxel_size", 0.0005)
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"[INFO] Downsampling 후 포인트 수: {len(pcd.points)}")
    # nb_neighbors = cfg.get("nb_neighbors", 20)
    # std_ratio = cfg.get("std_ratio", 2.0)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[INFO] 이상치 제거 후 포인트 수: {len(pcd.points)}")

    alpha = cfg.get("alpha", 0.02)
    print(f"[INFO] Alpha Shape Reconstruction 수행 중 (alpha={alpha})...")
    try:
        mesh_rec = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if len(mesh_rec.vertices) == 0:
            print("[WARN] 재구성된 메쉬가 비어 있습니다. alpha 값을 증가시켜 재시도합니다.")
            alpha *= 2
            mesh_rec = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    except Exception as e:
        print(f"[ERROR] Alpha Shape Reconstruction 실패: {e}")
        return

    if len(mesh_rec.vertices) == 0:
        print("[ERROR] 재구성된 메쉬가 여전히 비어 있습니다. alpha 파라미터를 재조정하세요.")
        return

    mesh_rec.remove_degenerate_triangles()
    mesh_rec.compute_vertex_normals()

    stl_filename = osp.join(data_path, "reconstructed_mesh.stl")
    o3d.io.write_triangle_mesh(stl_filename, mesh_rec)
    print(f"[DONE] Mesh가 {stl_filename}에 저장되었습니다.")
    o3d.visualization.draw_geometries([mesh_rec], window_name="3D Reconstruction Mesh",width=900, height=600)
    print("[DONE] 3D Reconstruction 작업 완료.")

if __name__ == "__main__":
    reconstruction()
