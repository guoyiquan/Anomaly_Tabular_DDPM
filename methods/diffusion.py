import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_named_beta_schedule


# 扩散过程类
class GaussianDiffusion:
    def __init__(
        self,
        device,
        model,
        schedule_name,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        self.model = model
        self.timesteps = timesteps

        self.schedule_name = schedule_name
        self.device = device
        self.betas = torch.tensor(
            get_named_beta_schedule(schedule_name, timesteps), dtype=torch.float32
        ).to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # 预计算扩散参数
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.register_buffer = lambda name, val: None  # 兼容性占位

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

    def p_losses(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(noise, predicted_noise)

    def p_sample(self, x, t, t_index, noise=None):
        batch_size = x.shape[0]
        betas_t = self.betas[t].view(batch_size, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(
            batch_size, 1
        )
        sqrt_recip_alpha_t = torch.sqrt(1.0 / (1 - betas_t))

        # 预测噪声并计算均值
        predicted_noise = self.model(x, t)
        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance = betas_t
            if noise is None:
                noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise

    def p_sample_loop(self, shape, noise=None):
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps // 2)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, noise)
        return x
