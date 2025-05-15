import torch
import torch.nn as nn

import os.path as osp

from itertools import chain
from module.comm.channel import Channel
from .enc_dec import setup_enc_dec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ImageAutoEncoder(nn.Module):
    def __init__(self, model_cfg, channel_cfg=None):
        super(ImageAutoEncoder, self).__init__()

        backbone = model_cfg.backbone
        latent_dim = model_cfg.latent_dim
        drop_p = model_cfg.drop_p

        self.encoder, self.decoder = setup_enc_dec(backbone, latent_dim, drop_p)

        model_saved_path = model_cfg.saved_path
        base_path, ext = osp.splitext(model_saved_path)
        idx = 1
        while osp.exists(model_saved_path):
            model_saved_path = f"{base_path}_{idx}{ext}"
            idx+=1

        self.model_saved_path = model_saved_path

        if channel_cfg is None:
            self.channel = Channel(channel_type="ideal",
                                   snr=None,
                                   iscomplex=True)
        else:
            _channel_type = channel_cfg.channel_type
            _snr = channel_cfg.snr
            _iscomplex = channel_cfg.iscomplex
            self.channel = Channel(channel_type=_channel_type,
                                   snr=_snr,
                                   iscomplex=_iscomplex)



    def forward(self,x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def forward_with_channel(self,x):
        z  = self.encoder(x)
        y = self.channel.transmit(z)
        out = self.decoder(y)
        return out,y

    def embed(self,x):
        z = self.encoder(x)
        return z

    def channel_output(self,x):
        z = self.encoder(x)
        y = self.channel.transmit(z)
        return y

    def get_encoder(self):
        return self.encoder

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters(), self.dec_fc.parameters()])

    def save_model(self):
        torch.save(self.state_dict(), self.model_saved_path)
        print(f"Model saved to {self.model_saved_path}.")


class Trainer:
    def __init__(self, model: torch.nn.Module):
        """
        model     : your autoencoder (or any torch.nn.Module)
        device    : torch.device('cuda') or torch.device('cpu')
        """
        self.model     = model.to(device)
        self.device    = device

    def train(self, data_loader: torch.utils.data.DataLoader,optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module) -> float:
        """
        Run one full training epoch over data_loader.
        Returns the average loss over all samples.
        """
        self.model.train()
        running_loss = 0.0
        running_count = 0

        for batch in data_loader:
            imgs = batch.to(self.device)   # if loader yields tensors
            optimizer.zero_grad()
            recon = self.model.forward_with_channel(imgs)
            loss  = criterion(recon, imgs)
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
                imgs = batch.to(self.device)
                recon = self.model(imgs)
                loss  = criterion(recon, imgs)

                bsize = imgs.size(0)
                val_loss  += loss.item() * bsize
                val_count += bsize

        return val_loss / val_count
