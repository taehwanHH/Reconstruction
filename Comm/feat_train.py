import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import numpy as np
import torch
import os.path as osp

from module.TactileUtil import load_all_heightmaps

from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader, TensorDataset


# Encoder: MobileNet 기반 Feature Extractor
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, latent_dim=128, num_input_channels=1):
        """
        Args:
            latent_dim (int): 최종 출력 feature vector 차원 (예: 128)
            num_input_channels (int): 입력 이미지 채널 수 (그레이스케일이면 1)
        """
        super(MobileNetFeatureExtractor, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        # 입력이 그레이스케일이면 첫 conv layer 수정
        if num_input_channels == 1:
            conv = model.features[0][0]  # 원래 conv layer
            new_conv = nn.Conv2d(1, conv.out_channels, kernel_size=conv.kernel_size,
                                 stride=conv.stride, padding=conv.padding, bias=conv.bias)
            with torch.no_grad():
                new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            model.features[0][0] = new_conv
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 1280  # MobileNet_v2 마지막 출력 채널 수
        self.fc = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        # x: [B, num_input_channels, H, W] (예: [B, 1, 160, 120])
        x = self.features(x)  # [B, 1280, H', W']
        x = self.avgpool(x)  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]
        z = self.fc(x)  # [B, latent_dim]
        return z


# Decoder: MobileNetDecoder
class MobileNetDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=1, output_size=(160, 120)):
        """
        Args:
            latent_dim (int): 입력 latent vector의 차원 (Encoder 출력 차원, 예: 128)
            output_channels (int): 복원된 이미지의 채널 수 (일반적으로 입력과 동일, 예: 1)
            output_size (tuple): 최종 복원 이미지 크기 (예: (160, 120))
        """
        super(MobileNetDecoder, self).__init__()
        self.output_size = output_size
        # latent vector를 feature map으로 확장: 예를 들어 [B, 256, 5, 5] (256*5*5 = 6400)
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        # ConvTranspose2d 레이어를 통해 업샘플링
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 입력 이미지가 [0,1] 범위라고 가정
        )

    def forward(self, z):
        # z: [B, latent_dim]
        x = self.fc(z)  # -> [B, 256*5*5]
        x = x.view(-1, 256, 5, 5)  # -> [B, 256, 5, 5]
        x = self.deconv(x)  # 업샘플링 진행, 대략 [B, output_channels, H, W]
        # 최종 크기가 output_size와 다를 경우 보간하여 맞춤
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


# 전체 Autoencoder: Encoder와 Decoder 결합
class MobileNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, num_input_channels=1, output_channels=1, output_size=(160, 120)):
        super(MobileNetAutoencoder, self).__init__()
        self.encoder = MobileNetFeatureExtractor(latent_dim=latent_dim, num_input_channels=num_input_channels)
        self.decoder = MobileNetDecoder(latent_dim=latent_dim, output_channels=output_channels, output_size=output_size)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    base = osp.join("data", "sim", cfg.obj_model, "full_coverage")
    heightmap_dir = osp.join(base, "gt_heightmaps")
    mask_dir = osp.join(base, "gt_contactmasks")

    points_path = osp.join(base, "sampled_points.npy")

    # Load all heightmaps from the directory. Ensure they are returned as a [N, 1, 160, 120] tensor.
    hm = load_all_heightmaps(heightmap_dir, height=320,
                             width=240)  # Modify load_all_heightmaps to output resized images if needed.
    cm = load_all_heightmaps(mask_dir, height=320, width=240)
    cm = ((cm > 128).float())
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)

    masked_hm = hm * cm / 255

    coords = torch.tensor(np.load(points_path), device="cuda")

    model = MobileNetAutoencoder(latent_dim=64, num_input_channels=1, output_channels=1, output_size=(320, 240)).to("cuda")

    # Training 설정
    num_epochs = 50
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = TensorDataset(masked_hm)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for batch in train_loader:
            images_batch = batch[0].to("cuda")  # [B, 1, 80, 60]
            optimizer.zero_grad()
            recon = model(images_batch)
            loss = criterion(recon, images_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images_batch.size(0)
            count += images_batch.size(0)
        avg_loss = epoch_loss / count
        print(f"\033[1;33mEpoch {epoch:2d}\033[0m: Average Training Loss = {avg_loss:.6f}")
        torch.cuda.empty_cache()


    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in train_loader:
            images_batch = batch[0].to("cuda")  # [B, 1, 80, 60]
            recon_batch = model(images_batch)
            loss = criterion(recon_batch, images_batch)
            total_loss += loss.item() * images_batch.size(0)
            count += images_batch.size(0)
    avg_loss = total_loss / count
    print("Average Reconstruction MSE Loss:", avg_loss)
    # trained_model = train_epoch(masked_hm, coords, num_epochs=200, loss_fn="mse", mask_ratio=0.3, batch_size=64)
    # torch.save(trained_model.state_dict(), "model_weights/E2E_GraphMAE.pth")
    torch.save(model.state_dict(), osp.join("model_weights", "MobileNetAutoencoder.pth"))

if __name__ == '__main__':
    main()
