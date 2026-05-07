import scvelo as scv
from pathlib import Path


def download():
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Pancreas dataset...")
    adata = scv.datasets.pancreas()

    path = data_dir / "pancreas.h5ad"
    adata.write_h5ad(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    download()
