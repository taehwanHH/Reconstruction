import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from torch_geometric.loader import NeighborLoader
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data  # Data is used for graph data
from Comm.model import EndToEndGraphMAE
from module.TactileUtil import load_all_heightmaps
from tqdm import tqdm
import os.path as osp
import os
import numpy as np
from omegaconf import DictConfig
import hydra

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


# 5. Build edge_index from sensor coordinates using k-NN
def build_edge_index(coords, k=10):
    """
    coords: [N, D] (e.g., [3000, 3]) sensor coordinates
    Returns: edge_index: [2, num_edges] torch.long tensor
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    edge_list = []
    N = coords.shape[0]
    for i in range(N):
        for j in indices[i][1:]:  # Exclude self
            edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def train_end_to_end(model, loader, optimizer, criterion):

    model.train()
    epoch_loss = 0
    count = 0
    for batch in tqdm(loader,desc="Batch"):
        optimizer.zero_grad()
        # Each mini-batch 'batch' has its own x, edge_index, y, and mask fields.
        recon_heightmaps,mask = model(batch.x, batch.edge_index)
        # Compute loss:
        observed = (mask == 0).squeeze()
        masked = (mask == 1).squeeze()
        loss_obs = criterion(recon_heightmaps[observed], batch.x[observed][:, :1, :, :])
        loss_mask = criterion(recon_heightmaps[masked], batch.y[masked])
        loss = loss_obs + loss_mask
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_nodes
        count += batch.num_nodes
    avg_loss = epoch_loss / count

    torch.cuda.empty_cache()

    return avg_loss


def train_epoch(sensor_images, sensor_coords,
                     num_epochs=50, mask_ratio=0.3, k=10, batch_size=64, device='cuda', loss_fn="sce", alpha_l=1.0):
    """
    sensor_images: [N, 1, 160, 120] tensor – input sensor images (heightmap images, grayscale)
    sensor_coords: [N, 3] tensor – sensor location coordinates
    ground_truth_heightmaps: [N, 1, 160, 120] tensor – ground truth heightmaps
    """
    sensor_images = sensor_images.to(device)

    # Build global edge_index from sensor coordinates
    sensor_coords_np = sensor_coords.cpu().numpy()
    global_edge_index = build_edge_index(sensor_coords_np, k=k).to(device)

    # Create a fixed mask vector (e.g., 10% observed, 90% masked)
    N = sensor_images.size(0)

    # Create a PyG Data object with global graph structure
    data = Data(x=sensor_images, edge_index=global_edge_index, y=sensor_images)
    data.num_nodes = N  # ensure num_nodes is set

    # Use NeighborLoader to create mini-batches with proper re-indexing
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],  # adjust based on the number of GCN layers
        batch_size=batch_size,
        input_nodes=torch.arange(N, device=device)
    )

    model = EndToEndGraphMAE(feature_dim=128, graph_hidden=64, graph_latent=32,
                              mask_ratio=mask_ratio, height=80, width=60).to(device)
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_loss = train_end_to_end(model, loader, optimizer, criterion)
        print(f"Epoch {epoch}: Average Loss = {train_loss:.6f}")
        torch.cuda.empty_cache()


    # model.eval()
    # with torch.no_grad():
    #     full_mask = torch.ones(N, 1, device=device)
    #     recon_heightmaps = model(sensor_images, global_edge_index, full_mask)
    #     print("Inference completed. Reconstructed heightmaps shape:", recon_heightmaps.shape)

    return model