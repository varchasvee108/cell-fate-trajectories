import torch
import torch.nn as nn

from core.config import Config
from model.layers import (
    CellProjection,
    ClusterHead,
    QuantileHead,
    PositionalEncoding,
    TransformerBlock,
)


class WaddingtonModel(nn.Module):
    def __init__(
        self,
        config: Config,
        n_clusters: int,
    ):
        super().__init__()

        self.config = config

        self.cell_projection = CellProjection(config)

        self.time_embeddings = PositionalEncoding(config)

        self.transformer = TransformerBlock(config)

        self.cluster_head = ClusterHead(
            config=config,
            n_clusters=n_clusters,
        )

        self.quantile_head = QuantileHead(config)

    def generate_causal_mask(
        self,
        seq_len: int,
        device,
    ):

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
            ),
            diagonal=1,
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf"),
        )

        return mask

    def forward(
        self,
        x,
        pseudotime,
    ):

        seq_len = x.shape[1]

        x = self.cell_projection(x)

        time_emb = self.time_embeddings(pseudotime)

        x = x + time_emb

        causal_mask = self.generate_causal_mask(
            seq_len=seq_len,
            device=x.device,
        )

        x = self.transformer(
            x,
            mask=causal_mask,
        )

        cluster_logits = self.cluster_head(x)

        quantile_preds = self.quantile_head(x)

        return {
            "cluster_logits": cluster_logits,
            "quantile_preds": quantile_preds,
        }
