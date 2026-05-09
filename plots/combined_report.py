import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

def plot_combined(
    pca: np.ndarray,
    pseudotime: np.ndarray,
    cluster_preds: np.ndarray,
    cluster_probs: np.ndarray,
    pseudotime_seq: np.ndarray,
    next_state_seq: np.ndarray,
    quantile_preds_seq: np.ndarray,
    n_clusters: int,
    save_path: Path,
):
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.85])

    # ---- Top Left: 3D Landscape ----
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, n_clusters))
    for c in range(n_clusters):
        mask = cluster_preds == c
        ax1.scatter(
            pseudotime[mask],
            pca[mask, 0],
            pca[mask, 1],
            c=[colors[c]],
            s=1,
            alpha=0.7,
            label=f"Cluster {c}",
        )
    ax1.set_title("Waddington Landscape", fontsize=14, fontweight="bold", pad=18)
    ax1.set_xlabel("Pseudotime", fontsize=10)
    ax1.set_ylabel("PC 1", fontsize=10)
    ax1.set_zlabel("PC 2", fontsize=10)
    ax1.legend(
        loc="upper left",
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        fontsize=7,
        markerscale=3,
    )
    ax1.view_init(elev=25, azim=-60)
    ax1.xaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax1.yaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax1.zaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax1.grid(True, alpha=0.12)

    # ---- Top Right: Phase Portrait ----
    ax2 = fig.add_subplot(gs[0, 1])
    confidence = cluster_probs.max(axis=1)
    color_idx = cluster_probs.argmax(axis=1)
    rgba_colors = np.array([colors[i] for i in color_idx])
    rgba_colors[:, 3] = np.clip(confidence / confidence.max(), 0.15, 1.0)
    ax2.scatter(pca[:, 0], pca[:, 1], c=rgba_colors, s=6, edgecolors="none")
    vel = next_state_seq.reshape(-1, pca.shape[1]) - pca
    stride = max(1, len(pca) // 300)
    idx = np.arange(0, len(pca), stride)
    ax2.quiver(
        pca[idx, 0],
        pca[idx, 1],
        vel[idx, 0],
        vel[idx, 1],
        color="#c9d1d9",
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=2.0,
        width=0.002,
        headwidth=3,
        headlength=4,
    )
    ax2.set_title("Phase Portrait — Fate Flow", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("PC 1", fontsize=10)
    ax2.set_ylabel("PC 2", fontsize=10)
    ax2.grid(True, alpha=0.08)

    # ---- Bottom Left: Fan Chart ----
    ax3 = fig.add_subplot(gs[1, 0])
    T = len(pseudotime_seq[0])
    x_axis = np.arange(T)
    lower = quantile_preds_seq[0, :, 0, 0]
    median = quantile_preds_seq[0, :, 0, 1]
    upper = quantile_preds_seq[0, :, 0, 2]
    ax3.fill_between(
        x_axis, lower, upper, alpha=0.25, color="#58a6ff", label="10%–90% PI"
    )
    ax3.plot(x_axis, median, color="#58a6ff", linewidth=2, label="Median")
    ax3.scatter(
        x_axis,
        next_state_seq[0, :, 0],
        s=10,
        color="#f78166",
        alpha=0.8,
        zorder=5,
        label="Actual",
    )
    ax3.set_title("Quantile Forecast — PC 1", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel("Sequence Step", fontsize=10)
    ax3.set_ylabel("PC 1 Value", fontsize=10)
    ax3.legend(
        loc="upper right",
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        fontsize=8,
    )
    ax3.grid(True, alpha=0.1)

    # ---- Bottom Right: Confidence by Cluster ----
    ax4 = fig.add_subplot(gs[1, 1])
    cluster_labels = [f"Cluster {c}" for c in range(n_clusters)]
    mean_conf = [
        cluster_probs[cluster_preds == c, c].mean()
        if (cluster_preds == c).sum() > 0
        else 0
        for c in range(n_clusters)
    ]
    sizes = [(cluster_preds == c).sum() for c in range(n_clusters)]
    bar_colors = np.array(colors)
    ax4.bar(
        cluster_labels,
        mean_conf,
        color=bar_colors,
        alpha=0.85,
        edgecolor="#30363d",
        linewidth=0.8,
    )
    for i, (mc, sz) in enumerate(zip(mean_conf, sizes)):
        ax4.text(i, mc + 0.01, f"n={sz}", ha="center", fontsize=9, color="#8b949e")
    ax4.set_title("Mean Cluster Confidence", fontsize=14, fontweight="bold", pad=12)
    ax4.set_ylabel("Mean Softmax Confidence", fontsize=10)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.1, axis="y")

    fig.suptitle(
        "Waddington Transformer — Inference Report",
        fontsize=17,
        fontweight="bold",
        y=0.985,
        color="#e6edf3",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  combined   → {save_path}")
