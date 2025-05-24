import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor
from .utils import get_timestep_embedding


class MLPBlock(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.LayerNorm(d_out)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        bias: Union[bool, List[bool]],
        d_out: int,
    ):
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        if isinstance(bias, bool):
            bias = [bias] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert len(d_layers) == len(bias)

        self.blocks = nn.ModuleList(
            [
                MLPBlock(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=bia,
                    dropout=dropout,
                )
                for i, (d, dropout, bia) in enumerate(zip(d_layers, dropouts, bias))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class TabDDPM1(nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LeakyReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, data_dim),
        )


    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embed(t_emb)
        x = torch.cat([x, t_emb], dim=1)
        return self.model(x)


class TabDDPM(nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        self.attn = nn.MultiheadAttention(data_dim + time_emb_dim, num_heads=1)
        
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
        self.norm1 = nn.LayerNorm(data_dim + time_emb_dim)

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embed(t_emb)
        x = torch.cat([x, t_emb], dim=1)
        
        attn_out, _ = self.attn(x, x, x)
        
        x = self.norm1(x + attn_out)
        
        return self.model(x)
