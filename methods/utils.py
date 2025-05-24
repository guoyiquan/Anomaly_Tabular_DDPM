import math
import torch

import numpy as np
import torch.nn.functional as F


def find_sim(A, B, p=0.05):
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)

    sim_matrix = torch.mm(A_norm, B_norm.T)  
    max_sim_values, _ = torch.max(sim_matrix, dim=0)  

    k = int(p * len(B))
    topk_values, topk_indices = torch.topk(max_sim_values, k=k)

    mask = torch.ones(B.size(0), dtype=torch.bool, device=B.device)
    mask[topk_indices] = False
    remaining_B = B[mask]

    A = torch.cat([A, B[topk_indices]], dim=0)
    B = remaining_B
    
    return A, B


def get_timestep_embedding(timesteps, embedding_dim):

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []

    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
