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
    def __init__(self, channel_type, snr, iscomplex=True):
        _iscomplex = iscomplex
        super().__init__(_iscomplex)
        self.channel_type = channel_type
        self.snr = snr
        self.iscomplex = _iscomplex

    def channel_param_print(self):
        print(f"    -> Channel Type: {self.channel_type}, SNR: {self.snr}dB, iscomplex: {self.iscomplex} ->")

    def _apply_awgn(self, _input):
        _std = (10 ** (-self.snr / 10.) / 2) ** 0.5 if self.iscomplex else (10 ** (-self.snr / 10.)) ** 0.5
        return _input + torch.randn_like(_input) * _std

    def _apply_fading_and_zf(self, x):
        # x: [B, 2*d] or [B, d] if real-valued
        device = x.device
        dtype = x.dtype

        if self.iscomplex:
            B, C2 = x.shape
            d = C2 // 2

            # split real/imag parts
            xr, xi = x[:, :d], x[:, d:]

            # 1) generate Rayleigh fading coefficients h = h_real + j*h_imag
            #    each ~ CN(0,1)
            hr = torch.randn(B, device=device, dtype=dtype) * (1 / 2 ** 0.5)
            hi = torch.randn(B, device=device, dtype=dtype) * (1 / 2 ** 0.5)

            # 2) apply complex fading: y = h * x
            yr = hr.view(-1, 1) * xr - hi.view(-1, 1) * xi
            yi = hr.view(-1, 1) * xi + hi.view(-1, 1) * xr

            # 3) add AWGN noise
            sigma = (10 ** (-self.snr / 10) / 2) ** 0.5
            nr = torch.randn_like(yr) * sigma
            ni = torch.randn_like(yi) * sigma
            yrn, yin = yr + nr, yi + ni

            # 4) zero-forcing equalization: x_hat = y_noisy / h = conj(h)/|h|^2 * y_noisy
            denom = hr.pow(2) + hi.pow(2) + 1e-9
            inv_r = hr / denom
            inv_i = -hi / denom

            xh_r = inv_r.view(-1, 1) * yrn - inv_i.view(-1, 1) * yin
            xh_i = inv_r.view(-1, 1) * yin + inv_i.view(-1, 1) * yrn

            # 5) concatenate real and imag back
            return torch.cat([xh_r, xh_i], dim=1)

        else:
            # real-valued channel: fading ~ |N(0,1)|
            B, D = x.shape
            h = torch.abs(torch.randn(B, 1, device=device, dtype=dtype))

            # apply fading and noise
            y = x * h
            sigma = (10 ** (-self.snr / 10)) ** 0.5
            y_noisy = y + torch.randn_like(y) * sigma

            # zero-forcing: divide out h
            return y_noisy / (h + 1e-9)
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
            return self._apply_fading_and_zf(_input)
        elif self.channel_type == 'phase_invariant_fading':
            return self._apply_phase_invariant_fading(_input)
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")