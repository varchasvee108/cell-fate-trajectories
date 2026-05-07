import torch
import torch.nn as nn
import math

from core.config import Config


class CellProjection(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.proj1 = nn.Linear(config.data.input_cell_dim, config.model.n_embd)
        self.norm = nn.LayerNorm(config.model.n_embd)

    def forward(self, x):
        return self.norm(self.proj1(x))


class PositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.model.n_embd

    def forward(self, pseudotime: torch.Tensor):
        device = pseudotime.device
        half_dim = self.dim // 2

        indices = torch.arange(half_dim, device=device)
        emb_scale = math.log(1000) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * indices)
        emb = pseudotime.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.n_embd,
            nhead=config.model.n_heads,
            dim_feedforward=config.model.hidden_dim,
            dropout=config.model.dropout,
            activation="silu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.model.n_layers
        )

    def forward(self, x, mask=None):
        return self.encoder(x, mask=mask)


class QuantileHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.output_dim = config.data.input_cell_dim
        self.n_quantiles = len(config.model.quantiles)

        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, self.output_dim * self.n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)

        out = out.view(out.shape[0], out.shape[1], self.output_dim, self.n_quantiles)
        return out


class ClusterHead(nn.Module):
    def __init__(self, n_clusters, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, n_clusters),
        )

    def forward(self, x):
        return self.net(x)
