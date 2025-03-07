import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import os

from Comm.train import train_epoch
from Comm.feat_train import MobileNetAutoencoder
from module.TactileUtil import load_all_heightmaps

from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader, TensorDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



@hydra.main(version_base="1.1", config_path="config", config_name="config")
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
