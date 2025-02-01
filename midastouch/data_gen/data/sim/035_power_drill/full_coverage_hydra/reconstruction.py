import numpy as np
import trimesh
import pyvista as pv

# 데이터 로드
sampled_points = np.load("sampled_points.npy")  # 접촉점 좌표
sensor_poses = np.load("sensor_poses.npy")  # 센서 포즈 (4x4 변환 행렬)
heightmaps = [np.load(f"gt_heightmaps/{i}.npy") for i in range(len(sampled_points))]
contact_masks = [np.load(f"gt_contactmasks/{i}.npy") for i in range(len(sampled_points))]

# 카메라 파라미터 (예시 값)
fx, fy = 600.0, 600.0  # 초점 거리
cx, cy = 320.0, 240.0  # 카메라 중심

# 3D 포인트 클라우드 생성
all_points = []

for i, (heightmap, mask, pose) in enumerate(zip(heightmaps, contact_masks, sensor_poses)):
    h, w = heightmap.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = heightmap * mask  # 접촉 영역만 사용
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 로컬 포인트 클라우드
    local_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    local_points = local_points[mask.flatten() > 0]  # 마스크 적용

    # 전역 좌표계로 변환
    rot = pose[:3, :3]  # 회전 행렬
    trans = pose[:3, 3]  # 변환 벡터
    global_points = local_points @ rot.T + trans
    all_points.append(global_points)

# 모든 포인트 병합
all_points = np.vstack(all_points)

# 3D 시각화
point_cloud = pv.PolyData(all_points)
point_cloud.plot()
