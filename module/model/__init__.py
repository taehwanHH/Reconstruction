from .Graph.MAE import *
from .AutoEncoder.enc_dec import *
from .Graph.ResMAE import *

from .AutoEncoder import build_model
from .Classifier import build_classifier_model
from .Graph import build_graph_model
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classifier_setting(cfg):
    _fe_model_cfg = cfg.feat_extractor.model
    _comm_cfg = cfg.sim.comm
    _model,_ = build_model(_fe_model_cfg, comm_cfg=_comm_cfg)
    _model.load_state_dict(torch.load(_fe_model_cfg.saved_path))

    _encoder= _model.get_encoder()
    _encoder.eval()

    classifier_cfg = cfg.k_classifier.model
    train_cfg = cfg.k_classifier.train

    classifier, trainer = build_classifier_model(_encoder, classifier_cfg)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()

    from module.data_module import get_stiffness_image_dataset
    train_loader, test_loader = get_stiffness_image_dataset(cfg.data_config)

    # 1) encoder 파라미터: 아주 작은 LR, 작은 weight decay
    enc_params = list(classifier.encoder.parameters())
    # 2) classifier 파라미터: 상대적으로 큰 LR, weight decay도 조금 높게
    clf_params = list(classifier.classifier.parameters())

    optimizer = torch.optim.Adam([
        {'params': enc_params, 'lr': 1e-5, 'weight_decay': 1e-6},
        {'params': clf_params, 'lr': 1e-3, 'weight_decay': 1e-4},
    ])

    return classifier, trainer, criterion, (train_cfg, classifier_cfg), (train_loader, test_loader), optimizer

def autoencoder_setting(cfg):
    model_cfg = cfg.feat_extractor.model
    train_cfg = cfg.feat_extractor.train
    comm_cfg = cfg.sim.comm

    fe_model, fe_trainer = build_model(model_cfg, comm_cfg=comm_cfg)
    fe_model.to(device)
    criterion = nn.MSELoss()

    from module.data_module import get_normalized_dataset
    train_loader, test_loader = get_normalized_dataset(cfg=cfg.data_config)

    return fe_model, fe_trainer, criterion, (train_cfg, model_cfg), (train_loader, test_loader), None


def image_model_train_setup(cfg):
    flag = cfg.data_config.image.flag
    if flag=="CL":
        print(" [INFO] Loading classifier model...")
        return classifier_setting(cfg)
    elif flag=="AE":
        print(" [INFO] Loading autoencoder model...")
        return autoencoder_setting(cfg)
    else:
        raise ValueError("Invalid flag")




__all__ =['image_model_train_setup', 'build_model', 'build_classifier_model']