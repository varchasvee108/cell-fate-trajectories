import torch
import torch.nn as nn
import math
from core.config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class CellProjection(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.proj = nn.Linear(config.data.input_cell_dim, config.model.n_embd)
        self.norm = RMSNorm(config.model.n_embd)

    def forward(self, x):
        return self.norm(self.proj(x))


class PositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.model.n_embd

    def forward(self, pseudotime: torch.Tensor):
        device = pseudotime.device
        half_dim = self.dim // 2
        indices = torch.arange(half_dim, device=device)
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * indices)
        emb = pseudotime.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.hidden_dim, config.model.n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.model.n_embd,
            num_heads=config.model.n_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.model.n_embd)
        self.norm2 = RMSNorm(config.model.n_embd)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.model.n_layers)]
        )
        self.norm = RMSNorm(config.model.n_embd)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class QuantileHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.output_dim = config.data.input_cell_dim
        self.n_quantiles = len(config.model.quantiles)
        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, self.output_dim * self.n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.shape[0], out.shape[1], self.output_dim, self.n_quantiles)


class ClusterHead(nn.Module):
    def __init__(self, n_clusters, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, n_clusters),
        )

    def forward(self, x):
        return self.net(x)
