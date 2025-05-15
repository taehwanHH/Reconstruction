import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv



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

    def __init__(self,  in_dim, hidden_dim, latent_dim, JK="last",drop_ratio=0, gnn_type="gcn"):
        super(GNNEncoder, self).__init__()
        if not isinstance(hidden_dim, list):
            hidden_dim = list(hidden_dim)

        self.JK = JK
        self.num_layer = len(hidden_dim) + 1
        self.drop_ratio = drop_ratio

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
            self.batch_norms.append(nn.LayerNorm(dims[layer+1]))

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
            self.conv = GINConv(in_dim, in_dim, aggr="add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(in_dim, in_dim, aggr="add")
        elif gnn_type == "gat":
            self.conv = GATConv(in_dim, in_dim, aggr="add")
        elif gnn_type == "linear":
            self.dec = nn.Linear(in_dim, in_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")

        # self.bn = nn.BatchNorm1d(in_dim)
        self.bn = nn.LayerNorm(in_dim)

        self.conv2 = GCNConv(in_dim, in_dim, aggr="add")
        # self.bn2 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.LayerNorm(in_dim)
        self.dec_token = nn.Parameter(torch.zeros([1, in_dim]))

        self.enc_to_dec = nn.Linear(in_dim, in_dim, bias=False)
        self.activation = nn.PReLU()


        self.output_fc = nn.Linear(in_dim, out_dim)
        self.output_activation = nn.Identity()
        # self.output_fc = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.Sigmoid()
        # )

    def forward(self, x, edge_index, mask_vector):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.enc_to_dec(x)
            x_input = x.clone()

            # x_input[mask_vector] = 0
            x_input[mask_vector.squeeze()==0] = self.dec_token

            # out_ = self.activation(x_input)
            out_ = self.conv(x_input, edge_index)
            out_ = self.bn(out_)
            out_ = self.activation(out_)

            out_ = self.conv2(out_, edge_index)
            out_ = self.bn2(out_)
            out_ = self.activation(out_)
            out_= self.output_fc(out_)
            out = self.output_activation(out_)
        return out


class GraphMAE(nn.Module):
    def __init__(self, cfg):
        super(GraphMAE, self).__init__()
        in_dim = cfg.in_dim
        hidden_dim = cfg.hidden_dim
        latent_dim = cfg.latent_dim
        JK = cfg.JK
        drop_ratio = cfg.drop_ratio
        gnn_type = cfg.gnn_type
        self.mask_ratio = cfg.mask_ratio
        self.dec_token = nn.Parameter(torch.zeros([1, in_dim]))

        self.encoder = GNNEncoder(in_dim,hidden_dim, latent_dim, JK,drop_ratio, gnn_type)
        self.decoder = GNNDecoder(latent_dim, in_dim, gnn_type)

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
            x_input[mask_vector.squeeze()==0] = 0
        else:
            x_input = x.clone()
            x_input[mask_vector.squeeze()==0] = 0

        z = self.encoder(x_input, edge_index)
        x_recon = self.decoder(z, edge_index, mask_vector)
        return x_recon
