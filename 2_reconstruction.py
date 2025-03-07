#!/usr/bin/env python3
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import time

from module.TactileUtil import TactileMap

@hydra.main(version_base="1.1", config_path="config", config_name="config")
def reconstruction(cfg: DictConfig):
    TM = TactileMap(config=cfg)

    # FPS 방식으로 num_samples만큼 프레임 인덱스 선택
    indices = TM.sampled_indices
    print(f" [INFO] Selected {len(indices)} frames using FPS.")

    print(f" [INFO] Starting reconstruction for {TM.obj}...")
    time.sleep(1)

    all_gp,pred_k = TM.process_part_frame(indices)


    mesh_rec =TM.pcd2mesh(all_points_tensor=all_gp,alpha=0.005)

    TM.stl_export(mesh_rec, visible=True)

    print(" [DONE] 3D Reconstruction completed.")

    TM.stiff_map(pred_k, visible=False)




if __name__ == "__main__":
    reconstruction()
