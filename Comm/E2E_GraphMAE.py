import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # Data is used for graph data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os.path as osp
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from module.TactileUtil import load_all_heightmaps  # Ensure these images are resized to 160x120

# 1. ResNet-based Feature Extractor (modified for grayscale input)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, num_input_channels=1):
        """
        Args:
            feature_dim: Dimension of the extracted feature vector.
            num_input_channels: Number of channels in the input image.
        """
        super(ResNetFeatureExtractor, self).__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify first conv layer for grayscale input:
        if num_input_channels == 1:
            weight = model.conv1.weight  # shape: [64, 3, 7, 7]
            new_weight = weight[:, 0:1, :, :].clone()  # Extract weights for first channel
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                model.conv1.weight.copy_(new_weight)
        self.features = nn.Sequential(*list(model.children())[:-1])
        in_features = model.fc.in_features
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        # x: [B, num_input_channels, 160, 120]
        x = self.features(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        features = self.fc(x)  # [B, feature_dim]
        return features


# 2. Graph Masked Autoencoder (GraphMAE)
class GraphMAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, mask_ratio=0.3):
        """
        in_dim: Dimension of input node features (e.g., 128)
        mask_ratio: Ratio of nodes to mask during training
        """
        super(GraphMAE, self).__init__()
        self.mask_ratio = mask_ratio
        # Encoder: 2-layer GCN
        self.enc1 = GCNConv(in_dim, hidden_dim)
        self.enc2 = GCNConv(hidden_dim, latent_dim)
        # Decoder: 2-layer MLP
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, in_dim)

    def generate_mask(self, num_nodes, device):
        mask = torch.rand(num_nodes, device=device) < self.mask_ratio
        return mask

    def encode(self, x, edge_index):
        x = F.relu(self.enc1(x, edge_index))
        z = self.enc2(x, edge_index)
        return z

    def decode(self, z):
        x_hat = F.relu(self.dec1(z))
        x_hat = self.dec2(x_hat)
        return x_hat

    def forward(self, x, edge_index, mask_vector=None):
        if mask_vector is None:
            mask = self.generate_mask(x.size(0), x.device)  # True means masked
            mask_vector = (~mask).float().unsqueeze(1)  # Observed nodes are 1, masked nodes 0
            x_input = x.clone()
            x_input[mask] = 0.0
        else:
            x_input = x.clone()
            missing = (mask_vector == 0).squeeze()
            x_input[missing] = 0.0

        z = self.encode(x_input, edge_index)
        x_recon = self.decode(z)
        return x_recon, z


# 3. Heightmap Decoder (reconstruct flattened heightmap)
class HeightmapDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim):
        """
        in_dim: Dimension of latent features from GraphMAE (e.g., latent feature dimension)
        output_dim: Flattened dimension of the final heightmap (e.g., 160*120)
        hidden_dim: Hidden layer dimension
        """
        super(HeightmapDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 4. End-to-End Model: From sensor images to reconstructed heightmaps
class EndToEndGraphMAE(nn.Module):
    def __init__(self, feature_dim=128, graph_hidden=64, graph_latent=128,
                 mask_ratio=0.3, decoder_hidden=256, height=160, width=120):
        super(EndToEndGraphMAE, self).__init__()
        self.height = height
        self.width = width
        self.feature_extractor = ResNetFeatureExtractor(feature_dim=feature_dim, num_input_channels=1)
        self.graph_mae = GraphMAE(in_dim=feature_dim, hidden_dim=graph_hidden,
                                  latent_dim=graph_latent, mask_ratio=mask_ratio)
        self.heightmap_decoder = HeightmapDecoder(in_dim=graph_latent,
                                                  hidden_dim=decoder_hidden,
                                                  output_dim=height * width)

    def forward(self, images, edge_index, mask_vector):
        """
        images: [N, 1, 160, 120] sensor images
        mask_vector: [N, 1] tensor indicating which nodes are observed (1) and which are masked (0).
        """
        # For nodes not observed, use a default background (zeros)
        default_background = torch.zeros_like(images)
        images_modified = images.clone()
        missing = (mask_vector == 0).squeeze()
        images_modified[missing] = default_background[missing]

        # 1. Extract node features from images
        node_features = self.feature_extractor(images_modified)  # [N, feature_dim]

        # 2. Reconstruct node features via GraphMAE
        recon_node_features, latent = self.graph_mae(node_features, edge_index, mask_vector)

        # 3. Decode latent features into flattened heightmap
        recon_heightmaps_flat = self.heightmap_decoder(latent)  # [N, height*width]
        recon_heightmaps = recon_heightmaps_flat.view(-1, 1, self.height, self.width)

        return recon_heightmaps


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


# 6. End-to-End Training Function using NeighborLoader for mini-batch graph sampling
def train_end_to_end(sensor_images, sensor_coords, ground_truth_heightmaps,
                     num_epochs=50, mask_ratio=0.3, k=10, batch_size=64, device='cuda'):
    """
    sensor_images: [N, 1, 160, 120] tensor – input sensor images (heightmap images, grayscale)
    sensor_coords: [N, 3] tensor – sensor location coordinates
    ground_truth_heightmaps: [N, 1, 160, 120] tensor – ground truth heightmaps
    """
    sensor_images = sensor_images.to(device)
    ground_truth_heightmaps = ground_truth_heightmaps.to(device)

    # Build global edge_index from sensor coordinates
    sensor_coords_np = sensor_coords.cpu().numpy()
    global_edge_index = build_edge_index(sensor_coords_np, k=k).to(device)

    # Create a fixed mask vector (e.g., 10% observed, 90% masked)
    N = sensor_images.size(0)
    mask = (torch.rand(N, device=device) < mask_ratio).float().unsqueeze(1)
    mask_vector = mask  # [N, 1]

    # Create a PyG Data object with global graph structure
    data = Data(x=sensor_images, edge_index=global_edge_index, y=ground_truth_heightmaps, mask=mask_vector)
    data.num_nodes = N  # ensure num_nodes is set

    # Use NeighborLoader to create mini-batches with proper re-indexing
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],  # adjust based on the number of GCN layers
        batch_size=batch_size,
        input_nodes=torch.arange(N, device=device)
    )

    model = EndToEndGraphMAE(feature_dim=128, graph_hidden=64, graph_latent=128,
                              mask_ratio=mask_ratio, decoder_hidden=256,
                              height=160, width=120).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for batch in loader:
            optimizer.zero_grad()
            # Each mini-batch 'batch' has its own x, edge_index, y, and mask fields.
            recon_heightmaps = model(batch.x, batch.edge_index, batch.mask)
            # Compute loss:
            observed = (batch.mask == 1).squeeze()
            masked = (batch.mask == 0).squeeze()
            loss_obs = criterion(recon_heightmaps[observed], batch.x[observed][:, :1, :, :])
            loss_mask = criterion(recon_heightmaps[masked], batch.y[masked])
            loss = loss_obs + loss_mask
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_nodes
            count += batch.num_nodes
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        full_mask = torch.ones(N, 1, device=device)
        recon_heightmaps = model(sensor_images, global_edge_index, full_mask)
        print("Inference completed. Reconstructed heightmaps shape:", recon_heightmaps.shape)

    return model
