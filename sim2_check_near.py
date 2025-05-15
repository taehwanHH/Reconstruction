import torch

from omegaconf import DictConfig
import hydra
from module.data_module.data_utils.image_utils import get_image
from module.data_module.data_utils.graph_utils import NormalizedImageDataset,feature_extraction
from sklearn.neighbors import NearestNeighbors
from module.model import MobileNetAutoencoder
import torch.nn.functional as F
import torchvision.transforms as transforms
import os.path as osp

transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg:DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Feature extractor model configuration
    fe_model_cfg = cfg.feat_extractor.model
    fe_model = MobileNetAutoencoder(fe_model_cfg).to(device)
    fe_model.load_state_dict(torch.load(fe_model_cfg.saved_path))

    feature_extractor = fe_model.encoder
    feature_extractor.eval()


    ds=torch.load(osp.join(cfg.data_config.train.dir,"2","img_dataset_2.pt"))
    _normed_ds = NormalizedImageDataset(ds)
    _feat = feature_extraction(feature_extractor, _normed_ds)
    masked_hm, pos = _normed_ds.tensor_dataset.tensors


    im = []
    # for i in range(masked_hm.shape[0]):
    #     im.append(transform1(masked_hm[i]).unsqueeze(0))
    # masked_hm = transform1(masked_hm)




    k=20
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    point_idx = 53
    neighbors_indices = indices[point_idx][1:]
    print(neighbors_indices)


    images = []

    x = _feat[point_idx]
    p_ = pos[point_idx]
    # neighbors_indices = [[1,2,3]]
    # b=masked_hm[neighbors_indices]
    # Stack all images into a single numpy array of shape (N, height, width)
    p = pos[neighbors_indices]

    distance = torch.norm(p-p_, dim=1)




    # Y= feature_extractor(b)
    Y= _feat[neighbors_indices]
    # sim = F.cosine_similarity(a_, a, dim=-1)

    # x와 Y의 각 행에 대해 평균을 계산
    x_mean = x.mean(dim=0, keepdim=True)  # (1, 1)
    Y_mean = Y.mean(dim=1, keepdim=True)  # (10, 1)

    # 중심화: 각 텐서에서 평균을 뺍니다.
    x_centered = x - x_mean  # (1, 16)
    Y_centered = Y - Y_mean  # (10, 16)

    # 분자: x_centered와 Y_centered의 각 원소의 곱을 행별로 합산 (각 행의 내적)
    numerator = (Y_centered * x_centered).sum(dim=1)  # (10,)

    # 분모: 각 텐서의 제곱합의 제곱근을 계산하여 곱합니다.
    denom = torch.sqrt((Y_centered ** 2).sum(dim=1) * (x_centered ** 2).sum())  # (10,)

    # Pearson 상관계수 계산
    corr = numerator / denom  # (10,)
    print("Pearson correlation for each row:", corr)
    a=point_idx
#
if __name__ == "__main__":
    main()