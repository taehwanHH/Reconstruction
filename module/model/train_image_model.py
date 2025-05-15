import torch

from omegaconf import DictConfig
import hydra

from module.model.utils import (
    create_optimizer,
    set_random_seed,
    get_scheduler
)
from module.model import image_model_train_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base="1.1", config_path="../../config", config_name="config")
def main(cfg:DictConfig):
    model, trainer, criterion, config, loader, optimizer = image_model_train_setup(cfg)
    train_cfg, model_cfg = config
    train_loader, test_loader = loader

    num_epochs, lr, gamma = train_cfg.num_epochs, train_cfg.lr, train_cfg.gamma

    if model_cfg.pretrained:
        model.load_state_dict(torch.load(model_cfg.saved_path))

    else:
        if optimizer is None:
            optimizer = create_optimizer(opt="adam", model=model, lr=lr)

        if train_cfg.use_scheduler:
            scheduler = get_scheduler(optimizer=optimizer, gamma=gamma, min_lr=1e-5)
        else:
            scheduler = None

        early_stop = train_cfg.early_stop
        best_val_loss = float('inf')
        patience = train_cfg.early_stop_patience  # 예: 20
        min_delta = train_cfg.early_stop_min_delta  # 예: 1e-4
        trigger_times = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = trainer.train(train_loader,optimizer, criterion)

            val_loss = trainer.validate(test_loader, criterion)

            current_lr = optimizer.param_groups[0]['lr']

            if early_stop:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    trigger_times = 0
                    best_model_state = model.state_dict()  # best model 저장
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f" [DONE] Early stopping triggered. Stopping training. Best validation loss={best_val_loss:.6f}")
                        break

            if scheduler is not None:
                scheduler.step()

            print(f"\033[1;33mEpoch {epoch + 1:02d}\033[0m: Train Loss = {train_loss:.5f} || Val Loss = {val_loss:.5f} (lr = {current_lr:.6f})")

        # 가능한 경우 best_model_state로 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.save_model()

if __name__ == "__main__":
    main()