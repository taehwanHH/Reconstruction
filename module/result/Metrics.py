import hydra
from omegaconf import DictConfig
import os.path as osp
import trimesh
import numpy as np

from scipy.spatial import cKDTree
from typing import Dict
from module.SensingPart import Sensing



def load_mesh(stl_file):
    """
    STL 파일을 읽어 trimesh 메쉬 객체로 반환합니다.
    """
    mesh = trimesh.load_mesh(stl_file)
    if not mesh.is_watertight:
        print(f"Warning: {stl_file} is not watertight!")
    return mesh

def sample_mesh(mesh, num_samples=100000):
    """
    메쉬 표면에서 num_samples 개수만큼 점을 균일하게 샘플링합니다.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points

def compute_metro_metrics(mesh_ref, mesh_test, num_samples=100000):
    """
    원본 메쉬(mesh_ref)와 복원 메쉬(mesh_test)의 Metro 스타일 지표를 계산합니다.
    두 메쉬 모두에서 num_samples 만큼 표면 샘플링하여,
    각 샘플 점에 대해 상대 메쉬까지의 최근접 거리를 구한 뒤,
    평균, 중앙값, 최대값(하우스도르프) 및 95퍼센타일 값을 반환합니다.
    """
    # 참고 메쉬와 테스트 메쉬에 대해 proximity query 객체 생성
    proximity_ref = trimesh.proximity.ProximityQuery(mesh_ref)
    proximity_test = trimesh.proximity.ProximityQuery(mesh_test)

    # 복원 메쉬 표면에서 샘플링한 점들에 대해 원본 메쉬까지의 거리 계산
    pts_test = sample_mesh(mesh_test, num_samples)
    dists_test = np.abs(proximity_ref.signed_distance(pts_test))

    # 원본 메쉬 표면에서 샘플링한 점들에 대해 복원 메쉬까지의 거리 계산
    pts_ref = sample_mesh(mesh_ref, num_samples)
    dists_ref = np.abs(proximity_test.signed_distance(pts_ref))

    # 양쪽에서 구한 거리들을 합칩니다.
    all_dists = np.concatenate([dists_test, dists_ref])

    metrics = {
        'mean_distance': np.mean(all_dists),
        'median_distance': np.median(all_dists),
        'hausdorff_distance': np.max(all_dists),
        '95th_percentile': np.percentile(all_dists, 95)
    }
    return metrics




def compute_accuracy(
    original_mesh: trimesh.Trimesh,
    recon_mesh:    trimesh.Trimesh,
    pred_k_ds:     tuple,        # (sampled_points (N,3), pred_ks (N,))
    cfg:           DictConfig,
    num_points:    int   = 1000000,
    threshold:     float = 0.0005
):
    """
    Returns:
      outlier_count: fraction of recon_points farther than threshold from orig_mesh
      avg_distance:   mean distance of recon_points → orig_mesh
      recon_point_ks: (num_points,) stiffness at each sampled recon_point
    """

    _DIGIT = Sensing(cfg,output_base="",mkdir=False)


    # 1) sample recon / orig points
    recon_points, _ = trimesh.sample.sample_surface(recon_mesh, num_points)
    orig_points, _ = trimesh.sample.sample_surface(original_mesh, num_points)

    # 2) distance → orig_mesh
    tree_orig = cKDTree(orig_points)
    distances, _ = tree_orig.query(recon_points, k=1)

    outlier_count = np.sum(distances > threshold)

    stiffness_accuracy = 0
    if pred_k_ds is not None:
        # unpack your predictions
        sampled_points, pred_ks = pred_k_ds
        # 1) build once
        tree_k = cKDTree(sampled_points, leafsize=32)

        # 2) query in parallel
        dists, idxs = tree_k.query(recon_points, k=3, workers=-1)

        # 3) vectorized majority vote
        neigh_ks = pred_ks[idxs]  # (M,3)
        a, b, c = neigh_ks[:, 0], neigh_ks[:, 1], neigh_ks[:, 2]
        recon_ks = np.where(a == b, a,
                            np.where(a == c, a,
                                     np.where(b == c, b, a)))

        # ───────────────────────────────────────────────────────────────
        # 4) exclude points near a bin boundary (Approach #3)
        #    compute normalized z and distance to the nearest bin edge
        min_b, max_b = _DIGIT.mesh_bounds
        z = recon_points[:, 2]
        z_min, z_max = min_b[2], max_b[2]
        z_norm = (z - z_min) / (z_max - z_min)
        z_norm = np.clip(z_norm, 0.0, 1.0)

        k_num = _DIGIT.k_num
        # interior bin edges in normalized [0,1]
        boundaries = np.linspace(0.0, 1.0, k_num + 1)[1:-1]  # shape (k_num-1,)
        # distance of each point to nearest boundary
        dist_to_edge = np.min(np.abs(z_norm[:, None] - boundaries[None, :]), axis=1)

        # threshold: exclude points within half a bin's width
        delta = 0.5 / (2 * k_num)
        mask = dist_to_edge > delta

        # 5) compare only on "unambiguous" points
        true_ks = _DIGIT.get_local_stiffness(recon_points, _DIGIT.mesh_bounds)
        recon_clean = recon_ks[mask]
        true_clean = true_ks[mask]

        assert recon_clean.shape == true_clean.shape, "shape mismatch after masking!"

        stiffness_accuracy = (recon_clean == true_clean).mean()

    return outlier_count, stiffness_accuracy



def compute_surface_metrics(original_mesh_path, reconstructed_mesh_path, pred_k_tuple, cfg, threshold=0.0007, num_points=1000000 ):
    """
    원본 메쉬와 재구축된 메쉬 간의 표면 Completeness와 Accuracy를 평가합니다.

    """
    original_mesh = trimesh.load(original_mesh_path)
    recon_mesh = trimesh.load(reconstructed_mesh_path)
    # completeness_ratio, avg_completeness_distance = compute_completeness(original_mesh, recon_mesh, num_points,
    #                                                                      threshold)
    outlier_count, stiffness_accuracy = compute_accuracy(
        original_mesh, recon_mesh, pred_k_tuple, cfg, num_points, threshold
    )
    metric = {
        'outlier_count': outlier_count,
        'stiffness_accuracy': stiffness_accuracy
    }

    return metric


def sample_points(mesh: trimesh.Trimesh, n_points: int = 1000000) -> np.ndarray:
    """
    Uniformly sample points on the mesh surface.
    """
    return mesh.sample(n_points)

def compute_chamfer_dist(
        original_mesh_path, reconstructed_mesh_path
) -> float:
    """
    Compute Chamfer and Hausdorff distances between two point clouds
    and return them as a dictionary.
    """
    original_mesh = trimesh.load(original_mesh_path)
    recon_mesh = trimesh.load(reconstructed_mesh_path)

    pts_tgt = sample_points(original_mesh)
    pts_src = sample_points(recon_mesh)

    tree_tgt = cKDTree(pts_tgt)
    tree_src = cKDTree(pts_src)

    d_src_to_tgt, _ = tree_tgt.query(pts_src)
    d_tgt_to_src, _ = tree_src.query(pts_tgt)

    # mean_src_to_tgt = float(d_src_to_tgt.mean())
    # mean_tgt_to_src = float(d_tgt_to_src.mean())
    # chamfer_dist    = 0.5 * (mean_src_to_tgt + mean_tgt_to_src)
    hausdorff_dist  = float(max(d_src_to_tgt.max(), d_tgt_to_src.max()))

    # return {
    #     "chamfer": chamfer_dist,
    #     "mean_src_to_tgt": mean_src_to_tgt,
    #     "mean_tgt_to_src": mean_tgt_to_src,
    # }
    return hausdorff_dist

@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    obj = cfg.obj_model
    original_mesh_path = osp.join("obj_models", obj, "nontextured.stl")
    # reconstructed_mesh_path = osp.join("data", "sim","recon_stl",f"{cfg.sensing.num_samples}_re_mesh.stl")
    reconstructed_mesh_path = "data/sim/results/num_sample_sweep/scheme2/recon_stl/scheme2_100_re_mesh.stl"

    metrics = compute_chamfer_dist(original_mesh_path, reconstructed_mesh_path)
    print(metrics)

if __name__ == '__main__':
    main()