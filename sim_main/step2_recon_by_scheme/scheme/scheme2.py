import torch
from torch_geometric.loader import DataLoader
import numpy as np

import os
import os.path as osp
from omegaconf import DictConfig, OmegaConf
import hydra
from functools import cached_property

from module.data_module import NormalizedImageDatasets, image_reconstruction
from module.model import build_model, build_classifier_model
from module.comm import Channel
from midastouch.modules.misc import save_heightmaps, save_contactmasks, remove_and_mkdir

from .base.systemModel import SystemModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scheme2(SystemModel):
    def __init__(self, cfg: DictConfig):
        super(Scheme2,self).__init__(cfg)
        self._init_path()
        self.channel = Channel(channel_type=self.comm_cfg.channel_type,
                               snr=self.comm_cfg.snr,
                               iscomplex=self.comm_cfg.iscomplex)

    def _init_path(self):
        _sweep_type = self.sim_cfg.sweep
        if _sweep_type == "num_samples":
            self.result_dir = self.sim_cfg.smp_sweep.base_dir
        elif _sweep_type == "snr":
            self.result_dir = self.sim_cfg.snr_sweep.base_dir
        else:
            raise ValueError("Invalid sweep type")

    def _additional_processing(self):
        return

    def _image_reconstruction(self):
        print(" [INFO] Image reconstruction...")
        recon_image = self.fe_decoder(self.embedding.to(device))

        denorm_s = (recon_image * 0.5 + 0.5).clamp(min=0, max=1)
        images_tensor = denorm_s.mul(255).byte()
        masks_tensor = (images_tensor > 0).to(torch.uint8)

        recon_images = [images_tensor[i].squeeze().cpu().numpy().astype(np.float32) for i in
                        range(images_tensor.size(0))]
        masks = [masks_tensor[i].squeeze().cpu().numpy().astype(np.float32) for i in range(masks_tensor.size(0))]

        image_save_dir = osp.join(self.result_dir, "recon_image")
        mask_save_dir = osp.join(self.result_dir, "recon_mask")

        if os.path.exists(image_save_dir):
            remove_and_mkdir(image_save_dir)
        else:
            os.makedirs(image_save_dir)

        if os.path.exists(mask_save_dir):
            remove_and_mkdir(mask_save_dir)
        else:
            os.makedirs(mask_save_dir)

        save_heightmaps(recon_images, image_save_dir, idx_offset=0)
        save_contactmasks(masks, mask_save_dir, idx_offset=0)
        print(" [Done] Image reconstruction done")

        self.image_dir = image_save_dir
        self.mask_dir = osp.join(self.sim_cfg.save_base, "origin_mask")
        return


