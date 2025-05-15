import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
import hydra
from tqdm import tqdm

from module.model import MobileNetAutoencoder
from module.data_module.data_utils.image_utils import get_normalized_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def simsiam_loss(p, z):
    # p: predictor output, z: target (stop-gradient 적용)
    z = z.detach()  # target은 업데이트 하지 않음
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    # Cosine similarity는 내적을 사용하므로, loss는 -내적의 평균으로 정의합니다.
    return - (p * z).sum(dim=1).mean()


# Predictor 모듈 (SimSiam에서 사용하는 예측기)
class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



def nt_xent_loss_hard_negative(z1, z2, temperature=0.5, threshold=0.7):
    """
    z1, z2: [B, d] 임베딩. 같은 이미지의 두 증강 뷰는 positive pair로 간주합니다.
    temperature: 온도 스케일링 파라미터.
    threshold: negative pair로 취급할 raw cosine similarity 임계값.
               이 값 이상이면 해당 negative pair는 손실 계산에서 제외됩니다.
    """
    B = z1.size(0)
    # 두 배치를 결합하여 2B x d 텐서 생성 후 L2 정규화
    z = torch.cat([z1, z2], dim=0)  # [2B, d]
    z = F.normalize(z, dim=1)

    # cosine similarity 행렬 계산: [2B, 2B]
    sim_matrix = torch.matmul(z, z.T)

    # 자기 자신과의 유사도를 매우 낮은 값으로 마스킹
    identity_mask = torch.eye(2 * B, device=z.device).bool()
    sim_matrix.masked_fill_(identity_mask, -9e15)

    # logits: 온도 스케일링 적용 (각 원소를 temperature로 나눔)
    logits = sim_matrix / temperature

    # 각 샘플의 positive pair 인덱스 계산:
    # for i in [0, B-1]: positive index = i+B; for i in [B, 2B-1]: positive index = i-B.
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = torch.where(pos_idx < B, pos_idx + B, pos_idx - B)
    pos_logits = logits[torch.arange(2 * B), pos_idx]  # [2B]

    # 하드 네거티브 마이닝: 각 행에서, negative pair로 고려할지 여부 결정
    # raw cosine similarity(sim_matrix)는 threshold와 비교합니다.
    negatives_mask = sim_matrix < threshold  # True: negative pair를 손실 계산에 포함
    # 하지만 positive pair는 항상 포함시켜야 하므로,
    pos_mask = torch.zeros_like(negatives_mask)
    pos_mask[torch.arange(2 * B), pos_idx] = True
    final_mask = negatives_mask | pos_mask  # 최종 마스크 (True이면 해당 쌍을 사용)

    # 분모 계산: 각 행의 final_mask에 해당하는 원소들에 대해 exp(logits)를 합산
    exp_logits = torch.exp(logits)
    denom = exp_logits.masked_fill(~final_mask, 0).sum(dim=1)  # [2B]

    # 각 샘플에 대해 NT-Xent loss: -log(exp(pos_logit) / denom)
    loss = -torch.log(torch.exp(pos_logits) / denom)
    return loss.mean()


# def nt_xent_loss(z1, z2, temperature=0.5):
#     batch_size = z1.shape[0]
#     z = torch.cat([z1, z2], dim=0)
#     z = F.normalize(z, dim=1)
#     sim_matrix = torch.matmul(z, z.T)  # [2B, 2B]
#     mask = torch.eye(2 * batch_size, device=z.device).bool()
#     sim_matrix = sim_matrix.masked_fill(mask, -9e15)
#     # 온도 스케일링 한 logits
#     logits = sim_matrix / temperature
#     # Positive pairs: logits의 대각선 (오프셋 batch_size)
#     pos_sim = torch.cat([torch.diag(logits, batch_size), torch.diag(logits, -batch_size)], dim=0)
#     loss = -torch.log(torch.exp(pos_sim) / torch.exp(logits).sum(dim=1))
#     return loss.mean()


def train_joint(model, predictor, dataloader, optimizer, pre_optimizer, lambda_contrastive=1.0):
    model.train()
    predictor.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cont_loss = 0.0
    for view1, view2 in dataloader:
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        pre_optimizer.zero_grad()

        # 두 뷰에 대해 latent vector와 재구성 결과를 얻습니다.
        recon1, z1 = model(view1)
        recon2, z2 = model(view2)
        p1 = predictor(z1)
        p2 = predictor(z2)

        # Reconstruction loss (예: MSE)
        loss_recon = F.mse_loss(recon1, view1) + F.mse_loss(recon2, view2)

        # Contrastive loss (NT-Xent Loss)
        # loss_contrastive = nt_xent_loss_hard_negative(z1, z2, temperature=0.5)
        loss_simsiam = simsiam_loss(p1, z2) + simsiam_loss(p2, z1)


        # Joint loss: reconstruction + weighted contrastive loss
        loss = loss_recon + lambda_contrastive * loss_simsiam
        loss.backward()
        optimizer.step()
        pre_optimizer.step()


        total_loss += loss.item()
        total_recon_loss += loss_recon.item()
        total_cont_loss += loss_simsiam.item()
    torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_cont_loss = total_cont_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_cont_loss

def evaluate_joint(model,predictor, dataloader, lambda_contrastive=1.0):
    model.eval()
    predictor.eval()
    total_loss = 0.0
    with torch.no_grad():
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)

            # 두 뷰에 대해 latent vector와 재구성 결과를 얻습니다.
            recon1, z1 = model(view1)
            recon2, z2 = model(view2)

            p1 = predictor(z1)
            p2 = predictor(z2)
            # Reconstruction loss (예: MSE)
            loss_recon = F.mse_loss(recon1, view1) + F.mse_loss(recon2, view2)

            # Contrastive loss (NT-Xent Loss)
            loss_simsiam = simsiam_loss(p1, z2) + simsiam_loss(p2, z1)

            # Joint loss: reconstruction + weighted contrastive loss
            loss = loss_recon + lambda_contrastive * loss_simsiam

            total_loss += loss.item()
    torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)

    return avg_loss



def train(data_loader, model, optimizer, criterion):
    model.train()

    epoch_loss = 0
    count = 0
    for batch in data_loader:
        optimizer.zero_grad()

        images_batch = batch.to(device)
        recon,_ = model(images_batch)
        loss = criterion(recon, images_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images_batch.size(0)
        count += images_batch.size(0)

    avg_loss = epoch_loss / count
    # print(f"\033[1;33mEpoch {epoch:02d}\033[0m: Average Training Loss = {avg_loss:.6f}")
    torch.cuda.empty_cache()
    return avg_loss
#
def evaluate(data_loader, model, criterion):
    model.eval()

    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for batch in data_loader:
            images_batch = batch.to(device)
            recon,_ = model(images_batch)
            loss = criterion(recon, images_batch)
            val_loss += loss.item() * images_batch.size(0)
            val_count += images_batch.size(0)
    avg_val_loss = val_loss / val_count
    torch.cuda.empty_cache()
    return avg_val_loss
#


@hydra.main(version_base="1.1", config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    model_cfg = cfg.feat_extractor.model
    train_cfg = cfg.feat_extractor.train
    # train_loader, val_loader, test_loader= image_loader(cfg.MobileNetAE.data)
    train_loader, test_loader = get_normalized_dataset(cfg.data_config)

    model = MobileNetAutoencoder(model_cfg).to(device)
    predictor = Predictor(in_dim=model_cfg.latent_dim, hidden_dim=model_cfg.latent_dim*2, out_dim=model_cfg.latent_dim).to(device)

    # Training 설정
    num_epochs = train_cfg.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    pre_optimizer = torch.optim.Adam(predictor.parameters(), lr=train_cfg.lr)
    criterion = nn.MSELoss()


    # Training loop
    for epoch in range(num_epochs):
        # pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False, ncols=90,
        #             dynamic_ncols=False)
        # Training
        # avg_train_loss, avg_recon_loss, avg_cont_loss = train_joint(model,predictor, pbar, optimizer,pre_optimizer,device, lambda_contrastive=0.1)
        avg_train_loss = train(train_loader, model, optimizer, criterion)

        # Validation
        # avg_val_loss = evaluate_joint(model,predictor,pbar, device, lambda_contrastive=0.1)
        avg_val_loss = evaluate(test_loader, model, criterion)
        print(f"\033[1;33mEpoch {epoch + 1:02d}\033[0m: Train Loss = {avg_train_loss:.5f} || Val Loss = {avg_val_loss:.5f}")
        # print(f"\033[1;33mEpoch {epoch + 1:02d}\033[0m: Train Loss = {avg_train_loss:.5f} , Recon Loss = {avg_recon_loss:.5f} , Simsiam Loss = {avg_cont_loss:.5f} || Val Loss = {avg_val_loss:.5f}")


    print("[INFO] Check loss with test data")
    # avg_test_loss = evaluate_joint(model, predictor,test_loader, device,lambda_contrastive=0.1)
    avg_test_loss = evaluate(test_loader, model, criterion)

    print("[DONE] Average Reconstruction MSE Loss:", avg_test_loss)

    model_path = model_cfg.saved_path
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")

if __name__ == '__main__':
    main()
