import torch
from pathlib import Path

from core.config import Config
from core.factory import get_device
from core.dataset import WaddingtonDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import WaddingtonModel


def main():
    config = Config.load_config("config/config.toml")
    device = get_device()

    dataset = WaddingtonDataset(
        file_path="data/pancreas.h5ad",
        block_size=config.data.block_size,
        n_pcs=config.data.input_cell_dim,
    )
    model = WaddingtonModel(config, n_clusters=dataset.n_clusters).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = Path("checkpoints/best.pt")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path} - running with random weights")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['best_val_loss']:.4f})"
        )

    model.eval()

    all_cluster_logits = []
    all_quantile_preds = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference"):
            x = batch["x"].to(device)
            pseudotime = batch["pseudotime"].to(device)

            output = model(x, pseudotime)

            all_cluster_logits.append(output["cluster_logits"].cpu())
            all_quantile_preds.append(output["quantile_preds"].cpu())

    cluster_logits = torch.cat(all_cluster_logits, dim=0)
    quantile_preds = torch.cat(all_quantile_preds, dim=0)

    cluster_preds = cluster_logits.argmax(dim=-1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "cluster_logits": cluster_logits,
            "cluster_preds": cluster_preds,
            "quantile_preds": quantile_preds,
        },
        output_dir / "predictions.pt",
    )

    print(f"Saved predictions to {output_dir / 'predictions.pt'}")
    print(f"  cluster_logits:  {tuple(cluster_logits.shape)}")
    print(f"  quantile_preds:  {tuple(quantile_preds.shape)}")


if __name__ == "__main__":
    main()
