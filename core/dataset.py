from pathlib import Path
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset


class WaddingtonDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int = 128,
        n_pcs: int = 50,
        look_ahead: int = 5,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_pcs = n_pcs
        self.look_ahead = look_ahead

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        adata = sc.read_h5ad(path)
        adata = adata[adata.obs["dpt_pseudotime"].argsort()].copy()

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.pca(adata, n_comps=n_pcs)

        X = adata.obsm["X_pca"].astype(np.float32)
        self.X = torch.from_numpy(X)

        clusters = adata.obs["clusters"]
        self.n_clusters = len(clusters.unique())

        if clusters.dtype.name == "category":
            clusters = clusters.cat.codes
        self.clusters = torch.from_numpy(clusters.values.astype(np.int64))

        self.pseudotime = torch.from_numpy(
            adata.obs["dpt_pseudotime"].values.astype(np.float32)
        )

    def __len__(self):
        return len(self.X) - self.block_size - self.look_ahead

    def __getitem__(self, index):
        x_seq = self.X[index : index + self.block_size]

        next_seq = self.X[
            index + self.look_ahead : index + self.block_size + self.look_ahead
        ]

        cluster_seq = self.clusters[index : index + self.block_size]
        pseudotime_seq = self.pseudotime[index : index + self.block_size]

        return {
            "x": x_seq,
            "next_state": next_seq,
            "clusters": cluster_seq,
            "pseudotime": pseudotime_seq,
        }
