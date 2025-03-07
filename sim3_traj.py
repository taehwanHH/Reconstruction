#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
from module.SensingPart import Sensing

@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    cfg.sensing.max_samples = 40
    DIGIT = Sensing(cfg)


    # traj_poses = DIGIT.get_random_trajectory()
    # traj_poses = DIGIT.get_points_poses()
    # traj_poses = DIGIT.get_manual_trajectory(poses=traj_poses)
    traj_poses = DIGIT.get_manual_trajectory()
    DIGIT.sensing(traj_poses)

    DIGIT.save_results(traj_poses)

    DIGIT.show_heatmap(traj_poses,"origin", False)


if __name__ == "__main__":
    main()
