import torch
from module.model.Graph import build_graph_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

from module.model.utils import (
    create_optimizer,
    set_random_seed,
)
from module.data_module.data_utils.graph_utils import graph_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, data_loader, optimizer):
    model.train()
    epoch_loss = 0
    count = 0

    for graph in data_loader:
        graph = graph.to(device)
        optimizer.zero_grad()

        loss = model(graph)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss.item()* model.masked_node_num
        count += model.masked_node_num

    avg_epoch_loss = epoch_loss / count
    torch.cuda.empty_cache()

    return avg_epoch_loss

def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0
    count = 0
    with torch.no_grad():
        for graph in data_loader:
            graph = graph.to(device)
            loss = model(graph)
            epoch_loss += loss.item()* model.masked_node_num
            count += model.masked_node_num
        avg_epoch_loss = epoch_loss / count
    torch.cuda.empty_cache()

    return avg_epoch_loss



@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg:DictConfig):
    model_cfg = cfg.graph_mae.model
    train_cfg = cfg.graph_mae.train

    num_epochs, lr, weight_decay = train_cfg.epochs, train_cfg.lr, train_cfg.weight_decay


    model = build_graph_model(model_cfg).to(device)
    delayed_ema_epoch = model.delayed_ema_epoch
    optimizer = create_optimizer(opt="adam", model=model, lr=lr)

    if train_cfg.use_scheduler:
        scheduler = lambda _epoch: (1 + np.cos(_epoch * np.pi / num_epochs)) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    train_loader, test_loader = graph_loader(cfg.data_config)

    best_val_loss = float('inf')
    patience = train_cfg.early_stop_patience  # 예: 20
    min_delta = train_cfg.early_stop_min_delta  # 예: 1e-4
    trigger_times = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        if epoch >= delayed_ema_epoch:
            model.ema_update()

        test_loss = evaluate(model, test_loader)

        print(f"\033[1;33mEpoch {epoch + 1:03d}\033[0m: (lr:{optimizer.param_groups[0]['lr']:05f})Train Loss = {train_loss:.6f} | Validation Loss = {test_loss:.6f}")

        # Early stopping check
        # 만약 validation loss가 이전 best_val_loss보다 min_delta 만큼 개선되면 업데이트
        if best_val_loss - test_loss > min_delta:
            best_val_loss = test_loss
            trigger_times = 0
            best_model_state = model.state_dict()  # best model 저장
        else:
            trigger_times += 1
            # print(f' [INFO] No improvement in validation loss for {trigger_times} epochs.')
            if trigger_times >= patience:
                print(f" [DONE] Early stopping triggered. Stopping training. Best validation loss={best_val_loss:.6f}")
                break

        if scheduler is not None:
            scheduler.step()

    # 가능한 경우 best_model_state로 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = model_cfg.saved_path
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")



if __name__ == "__main__":
    main()