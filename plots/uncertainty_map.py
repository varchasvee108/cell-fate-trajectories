import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import griddata
from matplotlib.colors import Normalize


def plot_uncertainty_map(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]
    cluster_preds = preds["cluster_preds"]

    x = inputs[:, -1].numpy()
    q10 = quantile_preds[:, -1, :, 0].numpy()
    q90 = quantile_preds[:, -1, :, 2].numpy()
    pseudo = pseudotime[:, -1].numpy()
    clusters = cluster_preds[:, -1].numpy()

    uncertainty = np.mean(q90 - q10, axis=1)

    reducer = umap.UMAP(n_components=2, random_state=42)
    x_emb = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(14, 11))

    scatter = ax.scatter(
        x_emb[:, 0], x_emb[:, 1],
        c=uncertainty,
        cmap="magma",
        s=12,
        alpha=0.75,
        edgecolors="none",
        norm=Normalize(
            vmin=np.percentile(uncertainty, 2),
            vmax=np.percentile(uncertainty, 98),
        ),
    )

    grid_n = 80
    xi = np.linspace(x_emb[:, 0].min(), x_emb[:, 0].max(), grid_n)
    yi = np.linspace(x_emb[:, 1].min(), x_emb[:, 1].max(), grid_n)
    zi = griddata(
        (x_emb[:, 0], x_emb[:, 1]),
        uncertainty,
        (xi[None, :], yi[:, None]),
        method="cubic",
    )

    n_levels = 8
    levels = np.linspace(
        np.nanpercentile(uncertainty, 10),
        np.nanpercentile(uncertainty, 90),
        n_levels,
    )
    ax.contour(
        xi, yi, zi,
        levels=levels,
        colors="#c9d1d9",
        linewidths=0.5,
        alpha=0.35,
    )

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(
        "Prediction Uncertainty (q90 minus q10)",
        fontsize=12,
        color="#c9d1d9",
    )
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.outline.set_edgecolor("#30363d")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

    ax.set_title(
        "Uncertainty Landscape",
        fontsize=16,
        fontweight="bold",
        pad=15,
        color="#c9d1d9",
    )
    ax.set_xlabel("UMAP 1", fontsize=12, color="#c9d1d9")
    ax.set_ylabel("UMAP 2", fontsize=12, color="#c9d1d9")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.12, color="#21262d")

    plt.tight_layout()
    plt.savefig(
        output_path,
        facecolor=fig.get_facecolor(),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  uncertainty -> {output_path}")
