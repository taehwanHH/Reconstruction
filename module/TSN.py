import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig


# 최대 stiffness 값 (라벨 스케일링에 사용)
MAX_STIFFNESS = 10000.0

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


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
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Grayscale 입력일 경우: 첫 conv layer 수정
    if num_input_channels == 1:
        weight = model.conv1.weight
        new_weight = weight[:, 0:1, :, :].clone()  # 첫 채널만 복사
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(new_weight)
    # 마지막 fully-connected 레이어 교체: 출력 뉴런 수를 num_classes로 변경
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model_classification(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False, ncols=80,
                    dynamic_ncols=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / len(train_loader.dataset)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return model


# Test evaluation 코드
def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = test_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def TSN_COPY(num_k):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet_for_classification(num_input_channels=1, num_classes=num_k)
    model.to(device)
    model.load_state_dict(torch.load("model_weights/resnet_TSN.pth", map_location=device, weights_only=True))
    model.eval()
    return model

@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mode = cfg.mode
    obj = cfg.obj_model
    if mode == "train":
        # CSV 파일 경로 (여기서는 전체 경로가 CSV에 저장되어 있다고 가정)
        csv_file = f"data/sim/{obj}/stiffness/train/combined_result.csv"

        # ResNet은 보통 224x224 해상도를 사용합니다.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 전체 데이터셋 생성
        dataset = TactileDatasetClassification(csv_file, transform=transform)
        # 데이터셋 인덱스를 train/validation으로 분리 (예: 80:20)
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        num_classes = len(dataset.mapping)
        model = create_resnet_for_classification(num_input_channels=1, num_classes=num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = train_model_classification(model, train_loader, val_loader, criterion, optimizer, device, epochs=100)
        torch.save(model.state_dict(), "../model_weights/resnet_TSN.pth")

    elif mode == "test":
        # Test CSV 파일 경로 (학습과는 별개로 test 데이터셋 CSV 경로 지정)
        test_csv_file = f'data/sim/{obj}/stiffness/test/combined_result.csv'
        # test 데이터셋에 대한 전처리 (224x224 크기로 resize, ToTensor, Normalize)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        test_dataset = TactileDatasetClassification(test_csv_file, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        # 학습 시 사용했던 모델과 동일한 구조 생성 (클래스 수는 test CSV에 따라 자동 결정)
        num_classes = len(test_dataset.mapping)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_resnet_for_classification(num_input_channels=1, num_classes=num_classes)
        model.to(device)

        # 저장된 모델 weight 로드 (학습 시 저장한 파일 경로 사용)
        model.load_state_dict(torch.load("../model_weights/resnet_TSN.pth", map_location=device, weights_only=True))

        # 모델 평가
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
   main()