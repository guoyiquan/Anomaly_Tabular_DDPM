import torch
import torch.nn as nn


def matrix_kl_divergence(mat1, mat2, reduction="batchmean", dim=-1):

    p = torch.nn.functional.softmax(mat1, dim=dim)
    q = torch.nn.functional.softmax(mat2, dim=dim)

    loss = torch.nn.KLDivLoss(reduction=reduction)
    kl = loss(torch.log(q), p)
    return kl


def vec_kl_divergence(vec1, vec2, dim=-1):

    p = torch.nn.functional.softmax(vec1, dim=dim)
    q = torch.nn.functional.softmax(vec2, dim=dim)
    loss = torch.nn.KLDivLoss(reduction="batchmean")
    kl = loss(torch.log(q), p)
    return kl


def mse_distance(
    mat1,
    mat2,
):

    loss = torch.nn.MSELoss()
    return loss(mat1, mat2)


class AnomalyLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        sigma_mu=1,
        sigma_sigma=1,
        dim=5,
        device="cpu",
        decay_type="exp",
        cov_epsilon=1e-6,
        num_mask=1,
    ):
        super(AnomalyLoss, self).__init__()

        self.device = device
        self.dim = dim
        self.cov_epsilon = cov_epsilon
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma
        self.alpha = alpha
        self.beta = beta
        self.num_mask = num_mask
        self.decay_type = decay_type
        self.p = [1.0, 1.0, 1.0]
        self.register_buffer("step", torch.tensor(1))

    def get_decay(self):
        
        decay_rate = 0.995
        t = self.step.float()
        if self.decay_type == "exp":
            return decay_rate**t

    def forward(self, x_hat, x):

        decay_factor = self.get_decay()
        batch_size = x.size(0)
        mu_true = x.mean(dim=0)

        x_centered = x - mu_true
        sigma_true = (x_centered.T @ x_centered) / (batch_size - 1)
        sigma_true += self.cov_epsilon * torch.eye(self.dim, device=self.device)
        sigma_true = sigma_true

        mu_pred = x_hat.mean(dim=0)
        x_hat_centered = x_hat - mu_pred
        sigma_pred = (x_hat_centered.T @ x_hat_centered) / (batch_size - 1)
        sigma_pred += self.cov_epsilon * torch.eye(self.dim, device=self.device)

        current_alpha = self.alpha / decay_factor
        current_beta = self.beta / decay_factor
        self.step += 1

        rate1 = torch.rand(1)[0].item() * decay_factor
        rate2 = torch.rand(1)[0].item() * decay_factor

        M_mu = torch.bernoulli(rate1 * torch.ones(self.dim, device=self.device))

        M_sigma = torch.bernoulli(
            rate2 * torch.ones(self.dim, self.dim, device=self.device)
        )

        mode = torch.multinomial(torch.tensor(self.p, device=self.device), 1).item()

        if mode == 0:
            mu_pred = mu_pred
            mu_true = mu_true
            sigma_pred = M_sigma * sigma_pred
            sigma_true = M_sigma * (
                sigma_true + torch.randn_like(sigma_pred) * decay_factor
            )

        elif mode == 1:
            mu_pred = M_mu * mu_pred
            mu_true = M_mu * (mu_true + torch.randn_like(mu_pred) * decay_factor)
            sigma_pred = sigma_pred
            sigma_true = sigma_true

        elif mode == 2:
            mu_pred = M_mu * mu_pred
            mu_true = M_mu * (mu_true + torch.randn_like(mu_pred) * decay_factor)
            sigma_pred = M_sigma * sigma_pred
            sigma_true = M_sigma * (
                sigma_true + torch.randn_like(sigma_pred) * decay_factor
            )
        loss_mu = mse_distance(mu_pred, mu_true)
        loss_sigma = mse_distance(sigma_pred, sigma_true)

        return current_alpha * loss_mu + current_beta * loss_sigma
