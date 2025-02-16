import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm
# 최대 stiffness 값 (라벨 스케일링에 사용)
MAX_STIFFNESS = 20000.0


class TactileDatasetClassification(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        CSV 파일에는 "filename"과 "stiffness" 컬럼이 있다고 가정합니다.
        stiffness 값은 예를 들어 500, 1000, ... , 20000 의 후보값입니다.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        # 후보 stiffness 값을 정렬한 리스트
        self.candidates = sorted(self.df['stiffness'].unique())
        # 각 stiffness 값을 클래스 인덱스로 매핑 (예: {500:0, 1000:1, ...})
        self.mapping = {val: idx for idx, val in enumerate(self.candidates)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # CSV에 저장된 전체 경로를 사용
        img_path = self.df.iloc[idx]["filename"]
        image = Image.open(img_path).convert('L')  # Grayscale (heightmap)
        if self.transform:
            image = self.transform(image)
        # stiffness 값을 클래스 인덱스로 변환
        stiffness_val = self.df.iloc[idx]["stiffness"]
        label = torch.tensor(self.mapping[stiffness_val], dtype=torch.long)
        return image, label


def create_resnet_for_classification(num_input_channels=1, num_classes=40):
    # 최신 방식으로 사전 학습된 ResNet18 로드
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Grayscale 입력일 경우: 첫 conv layer 수정
    if num_input_channels == 1:
        # 기존 conv1의 weight shape: [64, 3, 7, 7]
        weight = model.conv1.weight
        new_weight = weight[:, 0:1, :, :].clone()  # 첫 채널만 복사
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(new_weight)

    # 마지막 fully-connected 레이어 교체: 출력 뉴런 수를 num_classes로 변경
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_model_classification(model, dataloader, criterion, optimizer, device, epochs=20):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # tqdm progress bar with fixed width
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=80, dynamic_ncols=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")
    return model


if __name__ == '__main__':
    # CSV 파일에 저장된 경로가 전체 경로라 가정합니다.
    csv_file = 'data/sim/011_banana/stiffness/combined_result.csv'

    # ResNet은 보통 224x224 해상도를 사용합니다.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = TactileDatasetClassification(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 후보 stiffness 클래스 개수는 CSV에서 파악한 후보 개수 (예: 40개)
    num_classes = len(dataset.mapping)
    model = create_resnet_for_classification(num_input_channels=1, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = train_model_classification(model, dataloader, criterion, optimizer, device, epochs=20)

    torch.save(model.state_dict(), "resnet_classification_stiffness.pth")
