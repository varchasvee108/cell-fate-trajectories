import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from core.config import Config
from core.dataset import WaddingtonDataset
from core.factory import get_device
from torch.utils.data import DataLoader
from model.model import WaddingtonModel

from plots.style import set_style
from plots.landscape import plot_landscape
from plots.fan_chart import plot_fan_chart
from plots.phase_portrait import plot_phase_portrait
from plots.combined_report import plot_combined
from plots.umap_flow import generate_umap_flow_plot
from plots.generate_3d_html import generate_3d_html
from plots.uncertainty_map import plot_uncertainty_map
from plots.sankey_fate import plot_sankey_fate
from plots.stream_umap import plot_stream_umap

def main():
    config = Config.load_config("config/config.toml")
    device = get_device()

    dataset = WaddingtonDataset(
        file_path="data/pancreas.h5ad",
        block_size=config.data.block_size,
        n_pcs=config.data.input_cell_dim,
    )

    n_clusters = int(dataset.clusters.max().item()) + 1

    model = WaddingtonModel(config=config, n_clusters=n_clusters).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = Path("checkpoints/best.pt")
    if not checkpoint_path.exists():
        print(f"\nWARNING: No checkpoint at {checkpoint_path} — using random weights\n")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"\nLoaded checkpoint — epoch {ckpt['epoch']} (val_loss={ckpt['best_val_loss']:.4f})\n"
        )

    model.eval()

    all_inputs = []
    all_pseudotime = []
    all_next_state = []
    all_cluster_logits = []
    all_quantile_preds = []
    all_clusters_gt = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Running inference"):
            x = batch["x"].to(device)
            pseudotime = batch["pseudotime"].to(device)

            output = model(x, pseudotime)

            all_inputs.append(x.cpu())
            all_pseudotime.append(pseudotime.cpu())
            all_next_state.append(batch["next_state"].cpu())
            all_clusters_gt.append(batch["clusters"].cpu())
            all_cluster_logits.append(output["cluster_logits"].cpu())
            all_quantile_preds.append(output["quantile_preds"].cpu())

    inputs = torch.cat(all_inputs, dim=0)
    pseudotime = torch.cat(all_pseudotime, dim=0)
    cluster_logits_tsr = torch.cat(all_cluster_logits, dim=0)
    quantile_preds_tsr = torch.cat(all_quantile_preds, dim=0)
    cluster_preds_tsr = cluster_logits_tsr.argmax(dim=-1)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "inputs": inputs,
            "pseudotime": pseudotime,
            "cluster_logits": cluster_logits_tsr,
            "cluster_preds": cluster_preds_tsr,
            "quantile_preds": quantile_preds_tsr,
        },
        output_dir / "predictions.pt",
    )

    x_all = inputs.numpy()  # (N, T, 50)
    pseudotime_all = pseudotime.numpy()
    next_state_all = torch.cat(all_next_state, dim=0).numpy()
    clusters_gt_all = torch.cat(all_clusters_gt, dim=0).numpy()
    cluster_logits = cluster_logits_tsr.numpy()
    quantile_preds = quantile_preds_tsr.numpy()

    cluster_preds = cluster_logits.argmax(axis=-1)
    cluster_probs = np.exp(cluster_logits) / np.exp(cluster_logits).sum(
        axis=-1, keepdims=True
    )

    next_state_flat = next_state_all.reshape(-1, 50)
    x_flat = x_all.reshape(-1, 50)
    pseudotime_flat = pseudotime_all.reshape(-1)
    cluster_preds_flat = cluster_preds.reshape(-1)
    cluster_probs_flat = cluster_probs.reshape(-1, cluster_probs.shape[-1])
    clusters_gt_flat = clusters_gt_all.reshape(-1)

    non_pad_mask = clusters_gt_flat >= 0

    x_flat = x_flat[non_pad_mask]
    next_state_flat = next_state_flat[non_pad_mask]
    pseudotime_flat = pseudotime_flat[non_pad_mask]
    cluster_preds_flat = cluster_preds_flat[non_pad_mask]
    cluster_probs_flat = cluster_probs_flat[non_pad_mask]

    set_style()

    print("\nGenerating plots:\n")

    plot_landscape(
        pca=x_flat,
        pseudotime=pseudotime_flat,
        cluster_preds=cluster_preds_flat,
        n_clusters=n_clusters,
        save_path=output_dir / "waddington_landscape.png",
    )

    plot_phase_portrait(
        pca=x_flat,
        pseudotime=pseudotime_flat,
        cluster_probs=cluster_probs_flat,
        next_state=next_state_flat,
        current_state=x_flat,
        save_path=output_dir / "phase_portrait.png",
    )

    plot_fan_chart(
        pseudotime_seq=pseudotime_all[0],
        next_state_seq=next_state_all[0],
        quantile_preds_seq=quantile_preds[0],
        dims=[0, 1, 2],
        save_path=output_dir / "quantile_fan_chart.png",
    )

    plot_combined(
        pca=x_flat,
        pseudotime=pseudotime_flat,
        cluster_preds=cluster_preds_flat,
        cluster_probs=cluster_probs_flat,
        pseudotime_seq=pseudotime_all,
        next_state_seq=next_state_all,
        quantile_preds_seq=quantile_preds,
        n_clusters=n_clusters,
        save_path=output_dir / "inference_report.png",
    )

    generate_umap_flow_plot(
        predictions_path=str(output_dir / "predictions.pt"),
        output_path=str(output_dir / "umap_flow.png"),
    )

    generate_3d_html(
        predictions_path=str(output_dir / "predictions.pt"),
        output_path=str(output_dir / "waddington_3d.html"),
    )

    plot_uncertainty_map(
        predictions_path=str(output_dir / "predictions.pt"),
        output_path=str(output_dir / "uncertainty_map.png"),
    )

    plot_sankey_fate(
        predictions_path=str(output_dir / "predictions.pt"),
        output_path=str(output_dir / "sankey_fate.png"),
    )

    plot_stream_umap(
        predictions_path=str(output_dir / "predictions.pt"),
        output_path=str(output_dir / "stream_umap.png"),
    )

    accuracy = (cluster_preds_flat == clusters_gt_flat[non_pad_mask]).mean()

    print(f"\n{'─' * 50}")
    print(f"  Clusters:        {n_clusters}")
    print(f"  Cells evaluated:  {len(x_flat):,}")
    print(f"  Cluster accuracy: {accuracy:.4f}")
    print(f"  Outputs saved to:  {output_dir.resolve()}/")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
