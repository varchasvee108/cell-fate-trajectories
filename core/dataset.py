from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import scanpy as sc
import scipy.sparse as sp


class WaddingtonDataset(Dataset):
    def __init__(
        self, file_path: str, block_size=128, n_pcs=50, target_col: str | None = None
    ):
        super().__init__()
        self.block_size = block_size
        self.n_pcs = n_pcs
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found at {file_path}")
        adata = sc.read_h5ad(file_path)

        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

        X = np.asarray(adata.X)

        self.X = torch.tensor(X, dtype=torch.float32)

        if target_col is not None and target_col in adata.obs:
            y = adata.obs[target_col]

            if y.dtype.name == "category":
                y = y.cat.codes

            self.y = torch.tensor(y.values, dtype=torch.float32)
        self.adata = adata
        self.n_genes = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        return x, y
