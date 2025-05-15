import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) pretrained ResNet56 불러오기
pretrained_resnet56 = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet56',
    pretrained=True
)

class ResNet56Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        backbone = pretrained_resnet56

        # --- 1. grayscale용 conv1 교체 (3→1채널) ---
        conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            1, conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False
        )
        # 기존 RGB 가중치 합으로 초기화 (sum 유지)
        with torch.no_grad():
            new_conv1.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))
        backbone.conv1 = new_conv1

        # --- 2. 분류 헤드(fc) 떼고 피처 추출부만 남기기 ---
        # CIFAR ResNet56도 layer1, layer2, layer3만 가집니다.
        self.features = nn.Sequential(
            backbone.conv1,    # 1→16, 32×32→32×32
            backbone.bn1,
            backbone.relu,
            backbone.layer1,   # 16→16, 32×32→32×32
            backbone.layer2,   # 16→32, 32×32→16×16
            backbone.layer3,   # 32→64, 16×16→8×8
            nn.AdaptiveAvgPool2d((1,1)),  # → [B,64,1,1]
            nn.Flatten(),                  # → [B,64]
        )
        # latent로 매핑
        self.fc_latent = nn.Linear(64, latent_dim)

    def forward(self, x):
        # x: [B,1,32,32]
        feat = self.features(x)
        z = self.fc_latent(feat)
        return z  # [B, latent_dim]


class ResNet56Decoder(nn.Module):
    def __init__(self, latent_dim=128, drop_p=0.2):
        super().__init__()
        # latent → 64×4×4
        self.fc_up = nn.Linear(latent_dim, 64 * 4 * 4)

        # ① 4×4 → 8×8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Dropout2d(drop_p),
        )
        # ② 8×8 → 16×16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout2d(drop_p),
        )
        # ③ 16×16 → 32×32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Dropout2d(drop_p),
        )
        # 최종 1채널 재투영
        self.final = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # 입력 스케일에 따라 Sigmoid로 변경 가능
        )

    def forward(self, z):
        # z: [B, latent_dim]
        x = self.fc_up(z)
        x = x.view(-1, 64, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        recon = self.final(x)
        return recon  # [B,1,32,32]

