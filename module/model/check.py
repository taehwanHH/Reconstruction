import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from module.data_gen.data_load import graph_loader

def compute_edge_similarity(data):
    """
    data: PyG Data 객체, data.x는 노드 feature (shape: [N, d]),
          data.edge_index는 [2, E] 형태의 edge 정보
    반환: 각 edge에 대해 코사인 유사도를 계산한 tensor (shape: [E])
    """
    src = data.edge_index[0]
    dst = data.edge_index[1]
    sim = F.cosine_similarity(data.x[src], data.x[dst], dim=-1)
    return sim


def compute_random_similarity(data, num_samples=None):
    """
    data: PyG Data 객체
    num_samples: 샘플링할 랜덤 노드 쌍의 수 (없으면 edge 수와 동일하게 샘플링)
    반환: 각 랜덤 쌍에 대해 코사인 유사도를 계산한 tensor (shape: [num_samples])
    """
    N = data.x.shape[0]
    if num_samples is None:
        num_samples = data.edge_index.shape[1]
    src = torch.randint(0, N, (num_samples,), device=data.x.device)
    dst = torch.randint(0, N, (num_samples,), device=data.x.device)
    sim = F.cosine_similarity(data.x[src], data.x[dst], dim=-1)
    return sim


def analyze_subgraph_similarities(loader):
    """
    loader: DataLoader로 불러온 subgraph Data 객체들
    각 subgraph마다 연결된 edge 쌍과 랜덤 노드 쌍의 코사인 유사도를 계산하여, 전체 평균 및 분포를 비교합니다.
    """
    edge_sim_list = []
    rand_sim_list = []

    # DataLoader로부터 각 subgraph Data 객체를 순회합니다.
    for data in loader:
        # DataLoader가 개별 Data 객체를 반환한다고 가정 (batch_size=1 또는 collate_fn이 따로 설정된 경우)
        if data.x is None or data.x.size(0) < 2:
            continue

        # 연결된 노드 쌍의 유사도 계산
        sim_edges = compute_edge_similarity(data)
        # 랜덤 노드 쌍의 유사도 계산
        sim_rand = compute_random_similarity(data)

        edge_sim_list.append(sim_edges.cpu().numpy())
        rand_sim_list.append(sim_rand.cpu().numpy())

    if len(edge_sim_list) > 0:
        edge_sims = np.concatenate(edge_sim_list)
        rand_sims = np.concatenate(rand_sim_list)
    else:
        print("No valid subgraphs found.")
        return

    print("Mean edge similarity: {:.3f}".format(edge_sims.mean()))
    print("Mean random similarity: {:.3f}".format(rand_sims.mean()))

    plt.figure(figsize=(8, 6))
    plt.hist(edge_sims, bins=50, alpha=0.7, label="Edge Similarity")
    plt.hist(rand_sims, bins=50, alpha=0.7, label="Random Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Subgraph Node Feature Similarity Comparison")
    plt.legend()
    plt.show()


@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def train_GMAE(cfg:DictConfig):

    train_loader, test_loader = graph_loader(cfg=cfg.GraphMAE)
    analyze_subgraph_similarities(train_loader)

if __name__ == "__main__":
    train_GMAE()