import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter, LeakyReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, GATv2Conv, GraphNorm
# from torch_geometric.nn.conv import MessagePassing




class ResidualGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, norm_type='batch', gnn_type="gcn"):
        super(ResidualGNNLayer, self).__init__()
        if gnn_type == "gcn":
            self.gnn = GCNConv(in_dim, out_dim)
        elif gnn_type == "gat":
            self.gnn = GATConv(in_dim, out_dim)
            # self.gnn = GATv2Conv(in_dim, out_dim, aggr="mean")
        elif gnn_type == "gin":
            self.gnn = GINConv(in_dim, out_dim, aggr="add")
        else:
            raise NotImplementedError(f"{gnn_type} not implemented")

        # 정규화: BatchNorm 또는 LayerNorm 선택
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type == 'graph_norm':
            self.norm = GraphNorm(out_dim)
        else:
            raise ValueError("norm_type should be 'batch' or 'layer'")

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual 연결: 입력과 출력 차원이 다르면 선형변환으로 맞춤
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = None

    def forward(self, x, edge_index):
        # GNN layer 처리
        out = self.gnn(x, edge_index)
        # 정규화 (보통 GNN 출력은 (N, out_dim) 형태)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        # Residual 연결: 입력 차원이 다르면 선형 변환 적용
        if self.residual is not None:
            x = self.residual(x)
        return out + x


# Residual GNN Encoder: 여러 ResidualGNNLayer를 쌓아서 깊은 네트워크를 구성
class ResidualGNNEncoder(nn.Module):
    def __init__(self, in_dim,  latent_dim, num_layers=2,dropout=0.3, norm_type='batch', gnn_type="gcn"):
        """
        hidden_dims: list of hidden dimensions for 중간 레이어들
        latent_dim: 마지막 레이어의 출력 차원
        """
        super(ResidualGNNEncoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            ResidualGNNLayer(in_dim, latent_dim, dropout=dropout, norm_type=norm_type, gnn_type=gnn_type)
        )
        for i in range(num_layers-1):
            self.layers.append(
                ResidualGNNLayer(latent_dim, latent_dim, dropout=dropout, norm_type=norm_type, gnn_type=gnn_type))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# Residual 기반의 Decoder
class ResidualGNNDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim, num_layers=2, dropout=0.3, norm_type='batch', gnn_type="gcn",
                 use_dec_token=True):
        """
        Args:
            latent_dim: Encoder에서 나온 latent 차원 (Decoder의 입력 차원)
            out_dim: 최종 복원해야 할 feature 차원 (원본 feature 차원)
            num_layers: Residual GNN 레이어의 수
            dropout: Dropout 비율
            norm_type: 'batch' 또는 'layer'
            gnn_type: 사용하고자 하는 GNN 레이어 종류 (예: "gcn", "gat", "gin")
            use_dec_token: True이면 마스킹된 노드 위치에 learnable token을 삽입
        """
        super(ResidualGNNDecoder, self).__init__()
        self.use_dec_token = use_dec_token
        # 마스킹된 노드에 대체할 learnable token
        self.dec_token = nn.Parameter(torch.zeros([1, latent_dim]))

        # 여러 residual GNN 레이어를 쌓음
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ResidualGNNLayer(latent_dim, latent_dim, dropout=dropout, norm_type=norm_type, gnn_type=gnn_type))

        # 최종 projection: latent_dim -> out_dim
        self.projection = nn.Linear(latent_dim, out_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, z, edge_index, mask_vector):
        """
        Args:
            z: [N, latent_dim] Encoder에서 나온 노드 임베딩
            edge_index: [2, E] 그래프 연결 정보
            mask_vector: 마스킹된 노드의 인덱스 (1D tensor)
        """
        # 마스킹된 노드에 대해 dec_token을 삽입하거나 0으로 대체
        z_dec = z.clone()
        if self.use_dec_token:
            z_dec[mask_vector.squeeze()==0] = self.dec_token
        else:
            z_dec[mask_vector.squeeze()==0] = 0.0

        # Residual GNN 레이어를 통과
        out = z_dec
        for layer in self.layers:
            out = layer(out, edge_index)

        # 최종 projection 후 출력 활성화 적용
        out = self.projection(out)
        out = self.output_activation(out)
        return out

class ResGraphMAE(nn.Module):
    def __init__(self, cfg):
        super(ResGraphMAE, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        in_dim = cfg.in_dim
        latent_dim = cfg.latent_dim
        num_layers = cfg.num_layers
        drop_ratio = cfg.drop_ratio
        norm_type = cfg.norm_type
        gnn_type = cfg.gnn_type
        self.enc_token = nn.Parameter(torch.zeros([1, in_dim]))

        self.mask_ratio = cfg.mask_ratio
        self.encoder = ResidualGNNEncoder(in_dim,latent_dim, num_layers,drop_ratio, norm_type, gnn_type)
        self.decoder = ResidualGNNDecoder(latent_dim, in_dim, num_layers, drop_ratio, norm_type, gnn_type, use_dec_token=True)

    def generate_mask(self,batch, mask_ratio, device):
        num_nodes = batch.size(0)
        mask_vector = torch.ones(num_nodes, device=device)
        num_masked = int(mask_ratio * num_nodes)  # 정확히 mask_ratio 만큼 마스킹
        perm = torch.randperm(num_nodes, device=device)  # 전체 노드 인덱스 무작위 섞기
        mask_indices = perm[:num_masked]
        mask_vector[mask_indices] = 0
        return mask_vector

    def forward(self,x, edge_index, mask_vector=None):
        if mask_vector is None:
            mask_vector = self.generate_mask(x,self.mask_ratio, self.device) # True means masked
            x_input = x.clone()
            x_input[mask_vector.squeeze()==0] = self.enc_token
        else:
            x_input = x.clone()
            x_input[mask_vector.squeeze()==0] = self.enc_token

        z = self.encoder(x_input, edge_index)
        x_recon = self.decoder(z, edge_index, mask_vector)
        return x_recon

