import torch

import os.path as osp
from omegaconf import DictConfig

from module.model import build_model, build_classifier_model
from module.comm import Channel

from .base.systemModel import SystemModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scheme1(SystemModel):
    def __init__(self, cfg: DictConfig):
        super(Scheme1,self).__init__(cfg)
        self._init_path()
        self.channel = Channel(channel_type="ideal",
                               snr=None,
                               iscomplex=True)

    def _init_path(self):
        _sweep_type = self.sim_cfg.sweep
        if _sweep_type == "num_samples":
            self.result_dir = self.sim_cfg.smp_sweep.base_dir
        else:
            raise ValueError("Invalid sweep type")


    def _init_model(self):
        #
        self.fe_model,_ = build_model(self.fe_model_cfg, self.comm_cfg)
        if self.fe_model_cfg.pretrained:
            self.fe_model.load_state_dict(torch.load(self.fe_model_cfg.saved_path))
        else:
            print(" [ERROR] Pretrained feature extractor model not found")
            exit()
        self.fe_model.eval()
        self.fe_encoder = self.fe_model.encoder.to(device)
        self.fe_decoder = self.fe_model.decoder.to(device)
        self.fe_encoder.eval()
        self.fe_decoder.eval()

        #
        self.k_predictor,_ = build_classifier_model(self.fe_encoder, self.cls_model_cfg)
        if self.cls_model_cfg.pretrained:
            self.k_predictor.load_state_dict(torch.load(self.cls_model_cfg.saved_path))
        else:
            print(" [ERROR] Pretrained classifier model not found")
            exit()
        self.classifier = self.k_predictor.classifier.to(device)
        self.classifier.eval()

    def _additional_processing(self):
        return

    def _image_reconstruction(self):
        print(" [INFO] No need to reconstruct image.")
        self.image_dir = osp.join(self.sim_cfg.save_base, "origin_masked_hm")
        self.mask_dir = osp.join(self.sim_cfg.save_base, "origin_mask")
        if self.sim_cfg.sweep == "num_samples":
            self.result_dir = self.sim_cfg.smp_sweep.base_dir
        else:
            raise ValueError("Invalid sweep type")
        return



