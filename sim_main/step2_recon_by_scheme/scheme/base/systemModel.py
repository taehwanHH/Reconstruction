import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import os
import os.path as osp
from omegaconf import DictConfig
import abc
import time

from functools import cached_property

from module.TactileUtil import TactileMap
from module.data_module import NormalizedImageDatasets, image_reconstruction
from module.model import build_model, build_classifier_model, build_graph_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SystemModel(metaclass=abc.ABCMeta):
    def __init__(self, cfg: DictConfig):
        self._init_config(cfg)
        self._init_model()
        self.scheme = self.sim_cfg.scheme

        self.embedding = None
        self.channel = None

        self.image_dir = None
        self.mask_dir = None
        self.result_dir = None

        self.poses = None

    def _init_config(self, cfg: DictConfig):
        self.cfg = cfg
        self.sim_cfg = cfg.sim
        self.comm_cfg = cfg.sim.comm
        self.fe_model_cfg = cfg.feat_extractor.model
        self.cls_model_cfg = cfg.k_classifier.model
        self.gmae_model_cfg = cfg.graph_mae.model


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


    def run(self):
        self._image_processing()
        self._additional_processing()
        self._image_reconstruction()
        self._stiffness_predict()


    @cached_property
    def image_loader(self):
        print(" [INFO] Loading image data...")
        imgs, self.poses = torch.load(osp.join(self.sim_cfg.save_base, "img_dataset.pt")).tensors

        img_dataset = NormalizedImageDatasets(imgs)
        return DataLoader(img_dataset, batch_size=128, shuffle=False, num_workers=4)


    def _image_processing(self):
        if self.channel is not None:
            self.channel.channel_param_print()

        ys = []
        with torch.no_grad():
            for batch in self.image_loader:
                batch = batch.to(device)
                feat = self.fe_encoder(batch)
                y = self.channel.transmit(feat)
                ys.append(y)

        self.embedding = torch.cat(ys, dim=0)
        print(f" [INFO] Image embedding shape: {self.embedding.shape}")


    def _stiffness_predict(self):
        ys = self.embedding.to(device)
        logits = self.classifier(ys)
        probs = F.softmax(logits, dim=1)  # [B, n_classes]
        self.k_indices = probs.argmax(dim=1)  # [B]


    def map_reconstruction(self):
        stl_save_dir = osp.join(self.result_dir,"recon_stl")
        os.makedirs(stl_save_dir,exist_ok=True)
        stl_path = osp.join(stl_save_dir, self.sim_cfg.stl_filename)

        TM = TactileMap(config=self.cfg, stl_filename= stl_path)
        TM.heightmap_dir = self.image_dir
        TM.mask_dir = self.mask_dir

        indices = TM.sampled_indices
        print(f" [INFO] Selected {len(indices)} frames using FPS.")

        print(f" [INFO] Starting reconstruction for {TM.obj}...")
        time.sleep(1)

        all_gp = TM.process_part_frame(indices)
        mesh_rec = TM.pcd2mesh(all_points_tensor=all_gp, depth=7)
        TM.stl_export(mesh_rec, visible=False)


        pred_k_tuple = TM.stiffness_tuple(self.k_indices)

        print(" [DONE] 3D Reconstruction completed.\n")

        return pred_k_tuple

    def _image_reconstruction(self):
        pass

    def _additional_processing(self):
        pass

