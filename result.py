import os
from os import path as osp
import threading
import numpy as np
from module.visualizer import Viz
import pyvista as pv
from omegaconf import DictConfig
from midastouch.modules.misc import (
    load_images,save_heightmaps, save_contactmasks, save_images,    images_to_video, remove_and_mkdir
)
from midastouch.contrib.tdn_fcrn.tdn import TDN
from midastouch.render.digit_renderer import digit_renderer

import hydra


def plotting(cfg: DictConfig, viz: Viz) -> None:
    expt_cfg, tcn_cfg, tdn_cfg = cfg.expt, cfg.tcn, cfg.tdn
    obj_model = expt_cfg.obj_model
    # 데이터 로드
    data_path = osp.join("data", "sim", obj_model, "full_coverage")
    points_path = osp.join(data_path,"sampled_points.npy")
    poses_path = osp.join(data_path, "sensor_poses.npy")
    image_path = osp.join(data_path, "tactile_images")
    # height_path = osp.join(data_path,"gt_heightmaps")
    # contact_path = osp.join(data_path, "gt_contactmasks")
    cm_path = osp.join(data_path,"tdn_cm")
    hm_path = osp.join(data_path,"tdn_hm")
    plot_path = osp.join(data_path,"sensed_result")
    remove_and_mkdir(plot_path)

    os.makedirs(cm_path, exist_ok=True)
    os.makedirs(hm_path, exist_ok=True)

    mesh_path = osp.join("obj_models",obj_model,"nontextured.stl")

    # 메쉬 및 데이터 로드
    mesh = pv.read(mesh_path)
    points = np.load(points_path)
    poses = np.load(poses_path)
    tactile_images= load_images(image_path)
    # mask_images = load_images(contact_path)
    # height_images= load_images(height_path)
    tac_render = digit_renderer(cfg=tdn_cfg.render, obj_path=mesh_path)

    digit_tdn = TDN(tdn_cfg, bg=tac_render.get_background(frame="gel"))

    if viz:
        viz.init_variables(
            obj_model=obj_model,
            mesh_path=mesh_path,
        )
    all_hm,all_cm =[],[]
    for i, point in enumerate(points):

        image = tactile_images[i]

        # image to heightmap
        heightmap = digit_tdn.image2heightmap(image)  # expensive
        mask = digit_tdn.heightmap2mask(heightmap,True)

        # heightmap = height_images[i]
        # mask = (mask_images[i]>123)

        all_hm.append(heightmap.cpu().numpy())
        all_cm.append(mask.cpu().numpy())
        if viz is not None:
            viz.update(
                point,
                mask,
                heightmap,
                image,
                i,
                image_path=osp.join(plot_path,f"{i}.jpg")
            )
    if viz is not None:
        viz.close()
    save_heightmaps(all_hm, hm_path)
    save_contactmasks(all_cm, cm_path)
    images_to_video(plot_path)  # convert saved images to .mp4
    return



@hydra.main(version_base="1.1",config_path="config", config_name="config")
def main(cfg: DictConfig, viz=None):
    if cfg.expt.render:
        viz = Viz(off_screen=cfg.expt.off_screen, zoom=1.0, window_size=0.25)

    # Viz 업데이트 스레드 시작
    t = threading.Thread(name="plotting",target=plotting, args=(cfg,viz))
    t.start()

    if viz:
        viz.plotter.app.exec_()
    t.join()

if __name__ == "__main__":
    main()