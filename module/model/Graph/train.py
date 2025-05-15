import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

from module.data_module.data_utils.graph_utils import get_loader, graph_loader
from module.model import ResGraphMAE, GraphMAE
from module.model.Graph.loss_func  import sce_loss,sig_loss



def cosine_loss(x_recon, x_target):
    cos_sim = F.cosine_similarity(x_recon, x_target, dim=1)
    loss = 1 - cos_sim
    return loss.mean()

def generate_mask(batch, mask_ratio, device):
    num_nodes = batch.num_nodes
    mask_vector = torch.ones(batch.x.size(0), device=device)
    num_masked = int(mask_ratio * num_nodes)  # 정확히 mask_ratio 만큼 마스킹
    perm = torch.randperm(num_nodes, device=device)  # 전체 노드 인덱스 무작위 섞기
    mask_indices = perm[:num_masked]
    mask_vector[mask_indices] = 0
    return mask_vector, num_masked

def collect_features(loader, device):
    features_list = []
    for batch in loader:
        batch = batch.to(device)
        # batch.x가 (N, feature_dim) 형태라면
        features_list.append(batch.x.cpu().numpy())
    features = np.concatenate(features_list, axis=0)
    return features



def train(data_loader, model, optimizer,  criterion, mask_ratio, device):
    model.train()
    epoch_loss = 0
    count = 0
    orig_feature_list=[]
    recon_feature_list=[]
    for batch in data_loader:
        batch = batch.to(device)
        mask_vector,num_masked = generate_mask(batch, mask_ratio, device)
        optimizer.zero_grad()
        recon_feat= model(batch.x, batch.edge_index, mask_vector)
        mask = (mask_vector.squeeze()==0)
        observed = ~mask
        # loss_obs = criterion(recon_feat[observed], batch.x[observed])
        # loss_mask = F.smooth_l1_loss(recon_feat[mask], batch.x[mask],beta=0.1)
        loss_mask = criterion(recon_feat[mask], batch.x[mask])
        loss = loss_mask
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item() *num_masked
        count += num_masked

        # orig_feature_list.append(batch.x[mask].cpu().detach().numpy())
        # recon_feature_list.append(recon_feat[mask].cpu().detach().numpy())
    # f = np.concatenate(orig_feature_list, axis=0)
    # features = np.concatenate(recon_feature_list, axis=0)
    # plt.clf()
    #
    # plt.subplot(1,2,1)
    # plt.hist(f.flatten(), bins=100, density=True, alpha=0.7)
    # plt.xlabel("Feature value")
    # plt.ylabel("Density")
    # plt.title("Distribution of data.x features")
    # # plt.xlim(0, 1)
    # plt.subplot(1, 2, 2)
    # plt.hist(features.flatten(), bins=100, density=True, alpha=0.7)
    # plt.xlabel("Feature value")
    # plt.ylabel("Density")
    # plt.title("Distribution of reconstructed features")
    # # plt.xlim(0,1)
    #
    # plt.draw()
    #
    # plt.pause(0.001)
    # np.concatenate(recon_feature_list, axis=0)
    avg_loss = epoch_loss / count
    torch.cuda.empty_cache()

    return avg_loss

def evaluate(data_loader, model, criterion, mask_ratio, device):
    model.eval()
    epoch_loss = 0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mask_vector,num_masked = generate_mask(batch, mask_ratio, device)
            recon_feat = model(batch.x, batch.edge_index, mask_vector)
            # loss_obs = criterion(recon_heightmaps[observed], batch.x[observed])

            mask = (mask_vector.squeeze()==0)
            # loss_mask = F.smooth_l1_loss(recon_feat[mask], batch.x[mask],beta=0.1)
            loss_mask = criterion(recon_feat[mask], batch.x[mask])

            loss = loss_mask
            epoch_loss += loss.item() * num_masked
            count += num_masked
        avg_test_loss = epoch_loss / count
    torch.cuda.empty_cache()

    return avg_test_loss

@hydra.main(version_base="1.1", config_path="../../../config", config_name="config")
def train_GMAE(cfg:DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.graph_mae.model
    train_cfg = cfg.graph_mae.train

    model = ResGraphMAE(cfg=model_cfg).to(device)
    # model = GraphMAE(cfg=model_cfg).to(device)

    num_epochs = train_cfg.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    if train_cfg.use_scheduler:
        scheduler = lambda _epoch: (1 + np.cos(_epoch * np.pi / num_epochs)) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    criterion = sce_loss

    # train_loader, test_loader = get_loader(cfg=cfg.data_config)
    train_loader, test_loader = graph_loader(cfg=cfg.data_config)
    # plt.figure(figsize=(8, 6))
    # plt.ion()  # Interactive mode on


    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False, ncols=90,
                    dynamic_ncols=False)
        # Training
        avg_train_loss = train(pbar, model, optimizer, criterion, model_cfg.mask_ratio, device)
        # Validation
        avg_test_loss = evaluate(test_loader, model, criterion, model_cfg.mask_ratio, device)

        print(f"\033[1;33mEpoch {epoch + 1:03d}\033[0m: (lr:{optimizer.param_groups[0]['lr']:05f})Train Loss = {avg_train_loss:.6f} | Validation Loss = {avg_test_loss:.6f}")

        if scheduler is not None:
            scheduler.step()

    model_path = model_cfg.saved_path
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    train_GMAE()

