import os
import os.path as osp
import glob

from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader, GraphSAINTNodeSampler
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler


def build_edge_index(coords, k=10) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    edge_list = []
    N = coords.shape[0]
    for i in range(N):
        for j in indices[i][1:]:  # self 제외
            edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def feature_extraction(model, img_dataset, channel) -> torch.Tensor:
    print(f" [INFO] Loading {len(img_dataset)} images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(img_dataset, batch_size=64, shuffle=False)

    model.eval()
    feature_list =[]
    print(f" [INFO] Feature extraction starting...")
    with torch.no_grad():
        for batch in loader:
            images, pos = batch
            images = images.to(device)
            features= model(images)
            ch_out = channel.transmit(features)
            feature_list.append(ch_out.cpu())
    all_feature = torch.cat(feature_list, dim=0)
    return all_feature


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def to_graph(x, pos, k):
    edge = build_edge_index(pos.numpy(),k=k)
    data = Data(x=x, pos=pos, edge_index = edge)
    data.num_nodes = x.size(0)
    return data


def save_graph_datasets(base:str, model, channel, cfg) -> None:
    # cfg: graph data config
    pattern = os.path.join(base, "*", "img_dataset_*.pt")
    dataset_files = sorted(glob.glob(pattern))
    # print("Found dataset files:", dataset_files)
    graphs =[]

    for i,f in enumerate(dataset_files):
        print(f" [INFO] Creating graph data...")

        ds = torch.load(f)
        _normed_ds = NormalizedImageDataset(ds)
        _feat = feature_extraction(model, _normed_ds, channel)
        # _feat = scale_feats(_feat)
        # _mean = _feat.mean(dim=0, keepdim=True)
        # _std = _feat.std(dim=0, keepdim=True)
        # _feat = (_feat - _mean) / (_std + 1e-8)

        _pos = ds.tensors[1]
        _edge = build_edge_index(_pos.numpy(), k=cfg.k)
        data = Data(x=_feat, edge_index = _edge, pos=_pos)
        data.num_nodes = _feat.size(0)

        saved_path = osp.join(osp.dirname(f), f"graph_dataset_{i}.pt")
        torch.save(data,saved_path)
        print(f" [DONE] Data is saved to {saved_path}.\n")

        graphs.append(data)
    torch.save(graphs, osp.join(base, "graph_datasets.pt"))
    merged_graph = Batch.from_data_list(graphs)
    torch.save(merged_graph, osp.join(base,"merged_graph.pt"))
    # sampler = GraphSAINTNodeSampler(
    #     merged_graph,
    #     batch_size=1024,  # 한 서브그래프에 포함할 노드 수 (예: 1024)
    #     num_steps=10,  # 한 에포크 동안 10번 샘플링
    #     sample_coverage=1000  # 전체 노드가 평균적으로 1000번 등장하도록 조절
    # )

    # graph_datasets = GraphDatasets(graphs)
    # torch.save(graph_datasets, osp.join(base, f"{cfg.file_name}.pt"))


class NormalizedImageDataset(Dataset):
    def __init__(self, tensor_dataset):
        """
        tensor_dataset: TensorDataset(img, pos)와 같이 구성되어 있다고 가정
        transform: 이미지에 적용할 torchvision transform (PIL 이미지 입력을 기대)
        """
        super().__init__()
        self.tensor_dataset = tensor_dataset
        self.img_normalize = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, index):
        img, pos = self.tensor_dataset[index]

        img = self.img_normalize(img)
        return img, pos

    def __len__(self):
        return len(self.tensor_dataset)


class GraphDatasets(InMemoryDataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super(GraphDatasets, self).__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)


#######################################################################################
# For Dataloader (Graph MAE training)
#######################################################################################
def dataset2loader(dataset, cfg, shuffle=True):
    loader = NeighborLoader(
        Data(x=dataset.x, edge_index=dataset.edge_index, pos=dataset.pos),
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.batch_size,
        input_nodes=torch.arange(dataset.x.size(0)),
        shuffle=shuffle,
        drop_last=True
    )

    return loader


def get_loader(cfg):
    train_path = osp.join(cfg.train.dir, f"{cfg.graph.file_name}.pt")
    test_path = osp.join(cfg.test.dir, f"{cfg.graph.file_name}.pt")

    train_datasets = torch.load(train_path)
    test_datasets = torch.load(test_path)

    train_loader = dataset2loader(train_datasets, cfg.graph, shuffle=True)
    test_loader = dataset2loader(test_datasets, cfg.graph, shuffle=False)

    return train_loader, test_loader
#######################################################################################
# For Dataloader (Graph MAE training)
#######################################################################################


# def graphsaintsampler(dataset):
#     sampler = GraphSAINTNodeSampler(
#         dataset,
#         batch_size=128,  # 한 서브그래프에 포함할 노드 수 (예: 1024)
#         num_steps=256,  # 한 에포크 동안 10번 샘플링
#         sample_coverage=100 # 전체 노드가 평균적으로 1000번 등장하도록 조절
#     )
#     return sampler

# 노드 feature 정규화 함수
def normalize_graph_list(data_list, iscomplex=False):
    for data in data_list:
        x = data.x            # [num_nodes, feat_dim]
        # 실제 norm 차원을 d or d/2 로 계산
        d = x.size(1)//2 if iscomplex else x.size(1)
        # 노드별 스칼라 크기 r_i = ||x_i||_2
        r = x.norm(p=2, dim=1, keepdim=True)         # [num_nodes,1]
        # 방향단위 벡터 u_i = x_i / r_i
        u = x.div(r + 1e-9)                           # [num_nodes,feat_dim]
        # (원한다면) 스케일을 sqrt(d) 로 맞추는 경우
        # scale = math.sqrt(d)
        # u = u * scale

        # 복원 시 필요하다면 r도 같이 보관
        data.r = r                                    # [num_nodes,1]
        data.x = u                                    # 이제 x 는 단위벡터
    return data_list

def graph_loader(cfg):
    train_data_path = osp.join(cfg.train.dir,"image/graph_datasets.pt")
    test_data_path = osp.join(cfg.test.dir,"image/graph_datasets.pt")

    train_datasets = normalize_graph_list(torch.load(train_data_path))
    test_datasets = normalize_graph_list(torch.load(test_data_path))

    train_loader = DataLoader(train_datasets, batch_size=cfg.graph.batch_size, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=cfg.graph.batch_size, shuffle=False)

    return train_loader, test_loader


#######################################################################################
# For Dataloader (Graph MAE training)
#######################################################################################


def get_one_obj(cfg):
    train_path = osp.join(cfg.train.dir, "2", f"graph_dataset_2.pt")
    test_path = osp.join(cfg.test.dir, "1",  f"graph_dataset_1.pt")

    train_datasets = torch.load(train_path)
    test_datasets = torch.load(test_path)
    #
    # train_loader = graphsaintsampler(train_datasets)
    # test_loader = graphsaintsampler(test_datasets)

    return train_datasets, test_datasets