#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
from tmp import Sensing


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    obj_model = cfg.obj_model
    DIGIT = Sensing(obj_model,cfg)

    poses = DIGIT.get_points_poses()

    DIGIT.sensing(poses)

    DIGIT.save_results(poses)

    DIGIT.show_heatmap(poses,"origin", True)

if __name__ == "__main__":
    main()
