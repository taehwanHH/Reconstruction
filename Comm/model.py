import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torchvision.models import resnet18, ResNet18_Weights
from torch_geometric.loader import NeighborLoader
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data  # Data is used for graph data

# 1. ResNet-based Feature Extractor (modified for grayscale input)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, num_input_channels=1):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained ResNet18
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # If input is grayscale, modify the first conv layer to accept 1 channel
        if num_input_channels == 1:
            # Original conv1 weight shape: [64, 3, 7, 7]
            weight = model.conv1.weight  # [64, 3, 7, 7]
            new_weight = weight[:, 0:1, :, :].clone()  # [64, 1, 7, 7]
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                model.conv1.weight.copy_(new_weight)
        # Use all layers except the last fc layer and the final pooling
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512  # ResNet18 마지막 conv layer 출력 채널 수
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        # x: [B, 1, 160, 120]
        x = self.features(x)  # -> [B, 512, H, W] (여기서 H, W는 입력 크기에 따라 달라짐)
        x = self.avgpool(x)  # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # -> [B, 512]
        z = self.fc(x)  # -> [B, feature_dim]
        return z


class HeightmapDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=1, output_size=(160, 120)):
        super(HeightmapDecoder, self).__init__()
        self.output_size = output_size
        # fc: latent vector를 feature map (예: [B, 512, 5, 5])로 확장
        self.fc = nn.Linear(latent_dim, 512 * 5 * 5)
        # Transposed Convolution layers for upsampling
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # assuming input images normalized to [0,1]
        )

    def forward(self, z):
        # z: [B, latent_dim]
        x = self.fc(z)  # -> [B, 512*5*5]
        x = x.view(-1, 512, 5, 5)  # reshape to feature map [B, 512, 5, 5]
        x = self.deconv(x)  # upsample through convtranspose layers
        # x의 크기가 다를 경우, 마지막에 원하는 output_size로 보간(resize)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x



class GNNEncoder(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        latent_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self,  in_dim, hidden_dim, latent_dim,  JK="last", drop_ratio=0, gnn_type="gcn"):
        super(GNNEncoder, self).__init__()
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]

        self.num_layer = len(hidden_dim) + 1
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        dims = [in_dim] + hidden_dim + [latent_dim]

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            in_d = dims[layer]
            out_d = dims[layer + 1]
            if gnn_type == "gin":
                self.gnns.append(GINConv(in_d, out_d, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(in_d, out_d))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_d, out_d))


        ###List of batch-normalization
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(dims[layer+1]))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 2:
            x, edge_index= argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index = data.x, data.edge_index
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNNDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, gnn_type="gcn"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv(in_dim, out_dim, aggr="add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim, aggr="add")
        elif gnn_type == "linear":
            self.dec = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = nn.Parameter(torch.zeros([1, in_dim]))
        self.enc_to_dec = nn.Linear(in_dim, in_dim, bias=False)
        self.activation = nn.PReLU()
        self.temp = 0.2

    def forward(self, x, edge_index, mask_vector):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            missing = (mask_vector == 1).squeeze()
            x[missing] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index)
            # out = F.softmax(out, dim=-1) / self.temp
        return out


class GraphMAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, JK="last", drop_ratio=0, gnn_type="gcn"):
        super(GraphMAE, self).__init__()
        self.encoder = GNNEncoder(in_dim,hidden_dim, latent_dim, JK, drop_ratio, gnn_type)
        self.decoder = GNNDecoder(latent_dim, in_dim, gnn_type)

    def forward(self,x, edge_index, mask_vector=None):
        # if mask_vector is None:
        #     mask = self.generate_mask(x.size(0), x.device)  # True means masked
        #     mask_vector = (~mask).float().unsqueeze(1)  # Observed nodes are 1, masked nodes 0
        #     x_input = x.clone()
        #     x_input[mask] = 0.0
        # else:
        #     x_input = x.clone()
        #     missing = (mask_vector == 0).squeeze()
        #     x_input[missing] = 0.0

        z = self.encoder(x, edge_index)
        x_recon = self.decoder(z, edge_index, mask_vector)
        return x_recon


class EndToEndGraphMAE(nn.Module):
    def __init__(self, feature_dim=128, graph_hidden=64, graph_latent=128, mask_ratio=0.3,
                 height=160, width=120):
        super(EndToEndGraphMAE, self).__init__()
        self.feature_extractor= ResNetFeatureExtractor(feature_dim=feature_dim,num_input_channels=1)
        self.graph_mae = GraphMAE(in_dim=feature_dim,hidden_dim=graph_hidden,latent_dim=graph_latent,
                                  gnn_type="gcn")
        self.heightmap_decoder= HeightmapDecoder(latent_dim=feature_dim,output_channels=1, output_size=(height, width))
        self.mask_ratio = mask_ratio

    def generate_mask(self, num_nodes, device):
        # mask = torch.rand(num_nodes, device=device) < self.mask_ratio
        num_masked = int(self.mask_ratio * num_nodes)  # 정확히 mask_ratio 만큼 마스킹
        perm = torch.randperm(num_nodes, device=device)  # 전체 노드 인덱스 무작위 섞기
        mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        mask[perm[:num_masked]] = True
        return mask

    def forward(self, images, edge_index, mask_vector=None):
        if mask_vector is None:
            mask = self.generate_mask(images.size(0), images.device)  # True means masked
            mask_vector = mask.unsqueeze(1)  # Observed nodes are 1, masked nodes 0

        # For nodes not observed, use a default background (zeros)
        default_background = torch.zeros_like(images)
        images_modified = images.clone()
        missing = (mask_vector == 1).squeeze()
        images_modified[missing] = default_background[missing]

        # 1. Extract node features from images
        node_features = self.feature_extractor(images_modified)  # [N, feature_dim]

        # 2. Reconstruct node features via GraphMAE
        recon_node_features = self.graph_mae(node_features, edge_index, mask_vector)

        # 3. Decode latent features into flattened heightmap
        recon_heightmaps = self.heightmap_decoder(recon_node_features)  # [N, height*width]

        return recon_heightmaps, mask_vector


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

    # Create a PyG Data object with global graph structure
    data = Data(x=sensor_images, edge_index=global_edge_index, y=ground_truth_heightmaps)
    data.num_nodes = N  # ensure num_nodes is set

    # Use NeighborLoader to create mini-batches with proper re-indexing
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],  # adjust based on the number of GCN layers
        batch_size=batch_size,
        input_nodes=torch.arange(N, device=device)
    )

    model = EndToEndGraphMAE(feature_dim=128, graph_hidden=64, graph_latent=128,
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
