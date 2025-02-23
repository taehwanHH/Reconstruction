#!/usr/bin/env python3
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import time

from module.TactileUtil import TactileMap

@hydra.main(version_base="1.1", config_path="config", config_name="config")
def reconstruction(cfg: DictConfig):
    obj_model = cfg.obj_model

    TM = TactileMap(obj=obj_model,config=cfg)

    # FPS 방식으로 num_samples만큼 프레임 인덱스 선택
    indices = TM.sampled_indices
    print(f"[INFO] Selected {len(indices)} frames using FPS.")

    print(f"[INFO] Starting reconstruction for {obj_model}...")
    time.sleep(1)

    # 프레임별 처리
    all_gp = []
    pred_k = []
    for idx in tqdm(indices, desc="Processing frames", unit="frame"):
        gp, k_hat = TM.process_frame(idx)
        if gp is not None:
            all_gp.append(gp)
        pred_k.append(k_hat.item())

    mesh_rec =TM.pcd2mesh(all_points=all_gp,alpha=0.003)

    TM.stl_export(mesh_rec, visible=False)

    print("[DONE] 3D Reconstruction completed.")

    TM.stiff_map(pred_k, visible=True)




if __name__ == "__main__":
    reconstruction()
