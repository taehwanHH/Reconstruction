import torch

import hydra
from omegaconf import DictConfig

from module.data_module.data_utils import *
from module.model import build_model, build_classifier_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image_datasets(train_base, train_object, test_base, test_object, cfg):
    output_base = osp.join(train_base, "image")
    remove_and_mkdir(output_base)
    print("\033[1;31mCollecting train image data...\033[0m")
    save_image_datasets(train_object, output_base, cfg)

    output_base = osp.join(test_base, "image")
    remove_and_mkdir(output_base)
    print("\033[1;31mCollecting test image data...\033[0m")
    save_image_datasets(test_object, output_base, cfg)

def make_graph_datasets(train_base, test_base, cfg):
    #
    fe_model_cfg = cfg.feat_extractor.model
    cls_model_cfg = cfg.k_classifier.model

    fe_model, _ = build_model(fe_model_cfg, None)
    if fe_model_cfg.pretrained:
        fe_model.load_state_dict(torch.load(fe_model_cfg.saved_path))
    else:
        print(" [ERROR] Pretrained feature extractor model not found")
        exit()
    fe_encoder = fe_model.encoder.to(device)
    #
    k_predictor, _ = build_classifier_model(fe_encoder, cls_model_cfg)
    if cls_model_cfg.pretrained:
        k_predictor.load_state_dict(torch.load(cls_model_cfg.saved_path))
    else:
        print(" [ERROR] Pretrained classifier model not found")
        exit()

    feature_extractor = k_predictor.encoder.to(device)
    channel = k_predictor.ideal_channel
    # Data configuration
    data_cfg = cfg.data_config

    print("\033[1;31mCollecting train graph data...\033[0m")
    save_graph_datasets(base=train_base, model=feature_extractor, channel= channel, cfg=data_cfg.graph)

    print("\033[1;31mCollecting test graph data...\033[0m")
    save_graph_datasets(base=test_base, model=feature_extractor, channel= channel, cfg=data_cfg.graph)

def make_stiffness_datasets(train_base, train_object, test_base, test_object, cfg):
    output_base = osp.join(train_base, "k_labeled_image")
    objs = train_object
    remove_and_mkdir(output_base)
    print("\033[1;31mCollecting stiffness labeled image data for train...\033[0m")
    save_stiffness_labeled_datasets(objs, output_base, cfg)

    output_base = osp.join(test_base, "k_labeled_image")
    objs = test_object[:2]
    remove_and_mkdir(output_base)
    print("\033[1;31mCollecting stiffness labeled image data for test...\033[0m")
    save_stiffness_labeled_datasets(objs, output_base, cfg)



@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    # Data configuration
    data_cfg = cfg.data_config

    train_object = data_cfg.train.obj
    test_object = data_cfg.test.obj

    train_base = osp.join(data_cfg.train.dir,"image")
    test_base = osp.join(data_cfg.test.dir,"image")

    if cfg.mode == "all":
        make_image_datasets(train_base, train_object, test_base, test_object, cfg)
        make_graph_datasets(train_base, test_base, cfg)

    elif cfg.mode == "image":
        make_image_datasets(train_base, train_object, test_base, test_object, cfg)

    elif cfg.mode == "graph":
        make_graph_datasets(train_base, test_base, cfg)

    elif cfg.mode == "stiffness":
        make_stiffness_datasets(train_base, train_object, test_base, test_object, cfg)

    else:
        raise ValueError(f"Unknown mode {cfg.mode}")



if __name__ == "__main__":
    main()
