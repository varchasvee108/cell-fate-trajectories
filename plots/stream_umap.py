import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import griddata


def plot_stream_umap(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]
    cluster_preds = preds["cluster_preds"]

    x = inputs[:, -1].numpy()
    pred_next = quantile_preds[:, -1, :, 1].numpy()
    pseudo = pseudotime[:, -1].numpy()
    clusters = cluster_preds[:, -1].numpy()
    n_clusters = int(clusters.max()) + 1

    reducer = umap.UMAP(n_components=2, random_state=42)
    x_emb = reducer.fit_transform(x)
    pred_emb = reducer.transform(pred_next)

    flow_x = pred_emb[:, 0] - x_emb[:, 0]
    flow_y = pred_emb[:, 1] - x_emb[:, 1]

    grid_n = 60
    gx = np.linspace(x_emb[:, 0].min(), x_emb[:, 0].max(), grid_n)
    gy = np.linspace(x_emb[:, 1].min(), x_emb[:, 1].max(), grid_n)
    gxx, gyy = np.meshgrid(gx, gy)

    gu = griddata(
        (x_emb[:, 0], x_emb[:, 1]), flow_x, (gxx, gyy),
        method="cubic",
    )
    gv = griddata(
        (x_emb[:, 0], x_emb[:, 1]), flow_y, (gxx, gyy),
        method="cubic",
    )

    # Fill NaN cells with nearest-neighbor interpolation
    nan_mask = np.isnan(gu) | np.isnan(gv)
    if nan_mask.any():
        gu_fill = griddata(
            (x_emb[:, 0], x_emb[:, 1]), flow_x, (gxx, gyy),
            method="nearest",
        )
        gv_fill = griddata(
            (x_emb[:, 0], x_emb[:, 1]), flow_y, (gxx, gyy),
            method="nearest",
        )
        gu[nan_mask] = gu_fill[nan_mask]
        gv[nan_mask] = gv_fill[nan_mask]

    magnitude = np.sqrt(gu ** 2 + gv ** 2)
    magnitude = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, n_clusters))
    rgba = np.array([colors[c] for c in clusters])

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.scatter(
        x_emb[:, 0], x_emb[:, 1],
        c=rgba, s=8, alpha=0.5, edgecolors="none", zorder=1,
    )

    stream = ax.streamplot(
        gx, gy, gu, gv,
        color="#c9d1d9",
        linewidth=1.0,
        density=2.0,
        arrowsize=1.0,
        arrowstyle="->",
        zorder=2,
    )
    stream.lines.set_alpha(0.6)
    stream.arrows.set_alpha(0.4)

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Legend
    handles = [plt.scatter([], [], color=colors[c], s=40, label=f"Cluster {c}") for c in range(n_clusters)]
    ax.legend(
        handles=handles,
        loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
        fontsize=9, markerscale=1,
    )

    ax.set_title("Developmental Flow Field", fontsize=16, fontweight="bold", pad=15, color="#c9d1d9")
    ax.set_xlabel("UMAP 1", fontsize=12, color="#c9d1d9")
    ax.set_ylabel("UMAP 2", fontsize=12, color="#c9d1d9")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.1, color="#21262d")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  stream      -> {output_path}")
