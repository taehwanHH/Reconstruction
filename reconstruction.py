import hydra
import numpy as np
from PIL import Image
import pyvista as pv
from os import path as osp
from tqdm import tqdm

from omegaconf import DictConfig


def reconstruction(cfg:DictConfig):
    # 데이터 경로
    obj_model = cfg.obj_model
    data_path = osp.join("data", "sim", obj_model, "full_coverage")
    points_path = osp.join(data_path, "sampled_points.npy")
    heightmap_dir = osp.join(data_path, "gt_heightmaps")
    mask_dir = osp.join(data_path, "gt_contactmasks")

    # 데이터 로드
    sampled_points = np.load(points_path)  # 샘플링된 접촉점
    num_frames = len(sampled_points)

    # 카메라 파라미터
    fx, fy = 600.0, 600.0  # 초점 거리
    cx, cy = 320.0, 240.0  # 이미지 중심

    # 물체 크기
    scale_factor = 0.001  # 높이맵을 물리적 단위로 변환

    # 3D 포인트 클라우드 생성
    all_points = []
    print(f"[INFO] {obj_model} reconstruction...")


    for i in tqdm(range(num_frames), desc="Processing", unit="step"):
        # Heightmap 및 Contact Mask 로드
        heightmap_path = osp.join(heightmap_dir, f"{i}.jpg")
        mask_path = osp.join(mask_dir, f"{i}.jpg")
        heightmap = np.array(Image.open(heightmap_path))
        contact_mask = np.array(Image.open(mask_path))
        contact_mask = (contact_mask > 128).astype(np.uint8)

        # 스케일링 적용
        heightmap = heightmap * scale_factor

        # 픽셀 좌표를 물리적 좌표로 변환
        h, w = heightmap.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = heightmap * contact_mask
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # 로컬 포인트 클라우드 생성
        local_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)/20
        local_points = local_points[contact_mask.flatten() > 0]  # 마스크로 필터링

        # 샘플링된 점 기준으로 Heightmap 이동
        global_points = local_points + sampled_points[i]  # 중심점 이동
        all_points.append(global_points)

    # 모든 포인트 병합
    all_points = np.vstack(all_points)

    # 3D 시각화
    point_cloud = pv.PolyData(all_points)
    point_cloud.plot()
    print("[DONE] 3D Reconstruction finished.")



@hydra.main(version_base="1.1",config_path="config", config_name="config")
def main(cfg: DictConfig):
    reconstruction(cfg)

if __name__ == "__main__":
    main()