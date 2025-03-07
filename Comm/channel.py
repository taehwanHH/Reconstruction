import os, sys, time, glob, shutil
import torch
from torch.distributions import Normal

# Communication Utils: Channel Model, error rate, etc.

class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def norm_tx(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.5 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)


class Channel(Normlize_tx):
    def __init__(self, cfg):
        _iscomplex = cfg.iscomplex
        super().__init__(_iscomplex)
        self.channel_type = cfg.channel_type
        self.snr = cfg.snr
        self.iscomplex = _iscomplex

    def _apply_awgn(self, _input):
        _std = (10 ** (-self.snr / 10.) / 2) ** 0.5 if self.iscomplex else (10 ** (-self.snr / 10.)) ** 0.5
        return _input + torch.randn_like(_input) * _std

    def _apply_fading(self, _input):
        if self.iscomplex:
            _shape = _input.shape
            _dim = _shape[1] // 2
            _std = (10 ** (-self.snr / 10.) / 2) ** 0.5
            _mul = torch.abs(torch.randn(_shape[0], 2) / (2 ** 0.5))  # 복소수 채널의 Rayleigh fading 적용
            _input_ = _input.clone()
            _input_[:, :_dim] *= _mul[:, 0].view(-1, 1)
            _input_[:, _dim:] *= _mul[:, 1].view(-1, 1)
            _input = _input_
        else:
            _std = (10 ** (-self.snr / 10.)) ** 0.5
            _input = _input * torch.abs(torch.randn(_input.shape[0], 1)).to(_input)
        return _input + torch.randn_like(_input) * _std

    def _apply_phase_invariant_fading(self, _input):
        _std = (10 ** (-self.snr / 10.) / 2) ** 0.5 if self.iscomplex else (10 ** (-self.snr / 10.)) ** 0.5
        if self.iscomplex:
            _mul = (torch.randn(_input.shape[0], 1) ** 2 / 2. + torch.randn(_input.shape[0], 1) ** 2 / 2.) ** 0.5
        else:
            _mul = (torch.randn(_input.shape[0], 1) ** 2 + torch.randn(_input.shape[0], 1) ** 2) ** 0.5
        return _input * _mul.to(_input) + torch.randn_like(_input) * _std

    def transmit(self, _input):
        _input = self.norm_tx(_input)
        if self.channel_type == 'ideal':
            return _input
        elif self.channel_type == 'awgn':
            return self._apply_awgn(_input)
        elif self.channel_type == 'fading':
            return self._apply_fading(_input)
        elif self.channel_type == 'phase_invariant_fading':
            return self._apply_phase_invariant_fading(_input)
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")