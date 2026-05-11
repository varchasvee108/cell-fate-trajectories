import scanpy as sc

adata = sc.read_h5ad("data/pancreas.h5ad")
print(adata.obs.columns.tolist())

sc.pp.pca(adata)

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

sc.tl.diffmap(adata)
adata.uns["iroot"] = 0
sc.tl.dpt(adata)
print(adata.obs["dpt_pseudotime"])
adata.write_h5ad("data/pancreas.h5ad")
