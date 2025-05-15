import torch
import numpy as np

import os
import os.path as osp
from omegaconf import DictConfig, OmegaConf
import hydra
from functools import cached_property

from module.data_module import to_graph
from module.comm import Channel
from midastouch.modules.misc import save_heightmaps, save_contactmasks, remove_and_mkdir
from module.model import  build_graph_model

from .base.systemModel import SystemModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scheme3(SystemModel):
    def __init__(self, cfg: DictConfig):
        super(Scheme3,self).__init__(cfg)
        self._init_path()
        self.channel = Channel(channel_type=self.comm_cfg.channel_type,
                               snr=self.comm_cfg.snr,
                               iscomplex=self.comm_cfg.iscomplex)

        self._init_gmae()


    def _init_path(self):
        _sweep_type = self.sim_cfg.sweep
        if _sweep_type == "num_samples":
            self.result_dir = self.sim_cfg.smp_sweep.base_dir
        elif _sweep_type == "snr":
            self.result_dir = self.sim_cfg.snr_sweep.base_dir
        else:
            raise ValueError("Invalid sweep type")

    def _init_gmae(self):
        self.gmae = build_graph_model(self.gmae_model_cfg)
        if self.gmae_model_cfg.pretrained:
            self.gmae.load_state_dict(torch.load(self.gmae_model_cfg.saved_path))
        else:
            print(" [ERROR] Pretrained graph model not found")
            exit()
        self.gmae.eval()
        self.gmae.to(device)

    def _additional_processing(self):
        print(" [INFO] Additional processing...")
        g = to_graph(self.embedding, self.points, k=5).to(device)
        self.pred_embedding,_ = self.gmae.reconstruct_random_mask(g, mask_ratio=0.3)
        return


    def _image_reconstruction(self):
        print(" [INFO] Image reconstruction...")
        recon_image = self.fe_decoder(self.pred_embedding.to(device))

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


