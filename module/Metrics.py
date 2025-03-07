
import hydra
from omegaconf import DictConfig
import os.path as osp
import trimesh
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import io
import trimesh.transformations as tf




def render_mesh(mesh, rot, resolution=(256, 256)):
    rot_matrix = tf.rotation_matrix(rot, [0, 1, 0], point=mesh.centroid)
    mesh.apply_transform(rot_matrix)
    scene = trimesh.Scene(mesh)
    # 명시적으로 카메라 설정 업데이트
    scene.camera.resolution = resolution
    scene.camera.fov = [60, 60]  # 예: 수평, 수직 시야각

    png_bytes = scene.save_image(resolution=resolution, visible=True)
    if png_bytes is None:
        raise RuntimeError("Rendering failed. Check your OpenGL/pyglet settings.")
    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(image)




def evaluate_rendered_images(mesh_gt, mesh_recon,  resolution=(256,256)):
    """
    각 시점에서 원본 메쉬와 재구성 메쉬를 렌더링한 후,
    PSNR, SSIM, 그리고 MSE를 계산하여 평균 값을 반환합니다.
    """
    psnr_list = []
    ssim_list =[]
    mse_list =[]
    for _ in range(4):
        rot = np.pi / 2
        img_gt = render_mesh(mesh_gt, rot=rot, resolution=resolution)
        img_recon = render_mesh(mesh_recon,rot=rot, resolution=resolution)
        data_range = float(img_gt.max() - img_gt.min())
        psnr_val = peak_signal_noise_ratio(img_gt, img_recon, data_range=data_range)
        ssim_val = structural_similarity(img_gt, img_recon, data_range=data_range)
        mse_val = np.mean((img_gt - img_recon) ** 2)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mse_list.append(mse_val)


    metrics = {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "MSE": np.mean(mse_list)
    }

    for key, value in metrics.items():
        # key를 대문자로 변환, 소수점 4자리까지 출력
        print(f"\033[1;36m{key.upper():<4}: {value:.4f}\033[0m")
    return metrics



@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    obj = cfg.obj_model
    original_mesh_path = osp.join("obj_models", obj, "nontextured.stl")
    reconstructed_mesh_path = osp.join("data", "sim", obj, "full_coverage", "recon_stl","1500_recon_mesh.stl")


    mesh_gt = trimesh.load(original_mesh_path)
    mesh_recon = trimesh.load(reconstructed_mesh_path)


    metrics = evaluate_rendered_images(mesh_gt, mesh_recon, resolution=(256, 256))

if __name__ == '__main__':
    main()