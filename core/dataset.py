from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import torch

from scipy.sparse import issparse
from torch.utils.data import Dataset


class WaddingtonDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int = 128,
        n_pcs: int = 50,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_pcs = n_pcs

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        adata = sc.read_h5ad(path)

        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=2000,
            subset=True,
        )

        X: Any = adata.X

        if issparse(X):
            X = X.toarray()

        X = np.asarray(X).astype(np.float32)

        self.X = torch.from_numpy(X)

        clusters = adata.obs["clusters"]

        if clusters.dtype.name == "category":
            clusters = clusters.cat.codes

        clusters = np.asarray(clusters).astype(np.int64)

        self.cluster_y = torch.from_numpy(clusters)

        pseudotime = np.asarray(adata.obs["dpt_pseudotime"]).astype(np.float32)

        self.pseudotime_y = torch.from_numpy(pseudotime)

        self.n_genes = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
            "x": self.X[index],
            "cluster_y": self.cluster_y[index],
            "pseudotime_y": self.pseudotime_y[index],
        }
