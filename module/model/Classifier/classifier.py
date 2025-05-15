import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os.path as osp

from module.Stiffness import Stiffness
from module.comm import Channel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StiffnessClassifier(nn.Module):
    def __init__(self, encoder, model_cfg):
        super(StiffnessClassifier, self).__init__()
        self.encoder = encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        emb_dim = model_cfg.emb_dim
        n_classes = model_cfg.n_classes
        dropout = model_cfg.drop_p

        self.ideal_channel = Channel(channel_type="ideal",
                               snr=None,
                               iscomplex=True)

        self._init_img_transform()
        model_saved_path = model_cfg.saved_path
        base_path, ext = osp.splitext(model_saved_path)
        idx = 1
        while osp.exists(model_saved_path):
            model_saved_path = f"{base_path}_{idx}{ext}"
            idx += 1

        self.model_saved_path = model_saved_path

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,  n_classes),
        )


    def _init_img_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),                      # → [0,1]
            transforms.Normalize((0.5,), (0.5,)),       # → [-1,1]
        ])

    def forward(self, x,channel=None):
        # self.encoder.eval()
        # with torch.no_grad():
        z = self.encoder(x)
        if channel is None:
            z = self.ideal_channel.transmit(z)
        else:
            z= self.channel.transmit(z)
        logits = self.classifier(z)
        return logits

    def predict_stiffness(self, x, k_values):
        x = self.transform(x)
        logits = self(x.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).cpu().tolist()

        pred_k = k_values[idx]
        return pred_k


    def save_model(self):
        torch.save(self.state_dict(), self.model_saved_path)
        print(f"Model saved to {self.model_saved_path}.")


class Trainer:
    def __init__(self, model: torch.nn.Module):
        """
        model     : your autoencoder (or any torch.nn.Module)
        """
        self.model     = model.to(device)

    def train(self, data_loader: torch.utils.data.DataLoader,optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module) -> float:
        """
        Run one full training epoch over data_loader.
        Returns the average loss over all samples.
        """
        self.model.train()
        # self.model.classifier.train()
        running_loss = 0.0
        running_count = 0

        for batch in data_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self.model(imgs)
            loss  = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            bsize = imgs.size(0)
            running_loss  += loss.item() * bsize
            running_count += bsize

        return running_loss / running_count

    def validate(self, data_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> float:
        """
        Run one full validation pass (no grads).
        Returns the average loss over all samples.
        """
        self.model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in data_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = self.model(imgs)
                loss  = criterion(logits, labels)

                bsize = imgs.size(0)
                val_loss  += loss.item() * bsize
                val_count += bsize

        return val_loss / val_count

    def inference(self, data_loader, main_cfg):
        stiffness = Stiffness(main_cfg)
        k_values = stiffness.k_values

        result_save_dir = main_cfg.k_classifier.train.infer_dir

        self.model.eval()
        origins = []
        preds = []
        with torch.no_grad():
            for batch in data_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                pred_k = self.model.predict_stiffness(imgs, k_values)

                origins.extend(labels.cpu().tolist())
                preds.extend(pred_k)

        return preds
