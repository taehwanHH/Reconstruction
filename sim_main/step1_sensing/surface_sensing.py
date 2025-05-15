#!/usr/bin/env python3
import torch
from torch.utils.data import TensorDataset

import numpy as np
import os
import os.path as osp
import hydra
from omegaconf import DictConfig

from module.data_module.data_utils import collect_image_data, get_image
from midastouch.modules.misc import  save_heightmaps, save_contactmasks

@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def whole_sensing(cfg: DictConfig):
    sim_config = cfg.sim
    obj_model, save_base = sim_config.obj_model, sim_config.save_base

    cfg.obj_model = obj_model
    collect_image_data(cfg, save_base)

    # image to dataset
    img, pos = get_image(save_base)

    dataset_each_obj = TensorDataset(img, pos)
    torch.save(dataset_each_obj, osp.join(save_base, f"img_dataset.pt"))
    masked_hm_path = osp.join(save_base, "origin_masked_hm")
    os.makedirs(masked_hm_path)
    save_heightmaps(img.squeeze(1).numpy(), masked_hm_path, 0)

    origin_masks_tensor = (img>0).to(torch.uint8)
    origin_masks = [origin_masks_tensor[i].squeeze().cpu().numpy().astype(np.float32) for i in range(origin_masks_tensor.size(0))]
    origin_mask_save_path = osp.join(save_base, "origin_mask")
    os.makedirs(origin_mask_save_path)
    save_contactmasks(origin_masks, origin_mask_save_path, idx_offset=0)



if __name__ == "__main__":
    whole_sensing()
