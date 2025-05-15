from os import path as osp
import threading
import numpy as np
from module.result.visualizer import Viz

from omegaconf import DictConfig
from midastouch.modules.misc import (
    load_images, images_to_video, remove_and_mkdir
)

import hydra
import time
from module.SensingPart import Sensing


def plotting(cfg: DictConfig, viz: Viz) -> None:
    DIGIT = Sensing(cfg, mkdir=False)

    plot_path = osp.join(DIGIT.base, "sensed_result")
    remove_and_mkdir(plot_path)

    points = np.load(DIGIT.points_path)
    imgs, hms, cms = load_images(DIGIT.image_dir), load_images(DIGIT.heightmap_dir), load_images(DIGIT.mask_dir)

    if viz:
        viz.init_variables(
            obj_model=DIGIT.obj,
            mesh_path=DIGIT.mesh_path,
        )
    for i, point in enumerate(points):

        image = imgs[i]
        heightmap = hms[i]
        mask = (cms[i] > 123)

        if viz is not None:
            viz.update(
                point,
                mask,
                heightmap,
                image,
                i,
                image_path=osp.join(plot_path, f"{i}.jpg")
            )
        time.sleep(0.5)
    if viz is not None:
        viz.close()

    images_to_video(plot_path)  # convert saved images to .mp4
    return


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig, viz=None):
    if cfg.expt.render:
        viz = Viz(off_screen=cfg.expt.off_screen, zoom=1.0, window_size=0.25)

    # Viz 업데이트 스레드 시작
    t = threading.Thread(name="plotting", target=plotting, args=(cfg, viz))
    t.start()

    if viz:
        viz.plotter.app.exec_()
    t.join()


if __name__ == "__main__":
    main()