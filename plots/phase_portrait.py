import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_phase_portrait(
    pca: np.ndarray,
    pseudotime: np.ndarray,
    cluster_probs: np.ndarray,
    next_state: np.ndarray,
    current_state: np.ndarray,
    save_path: Path,
):
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111)

    confidence = cluster_probs.max(axis=1)
    color_idx = cluster_probs.argmax(axis=1)
    n_clusters = cluster_probs.shape[1]

    cmap = plt.get_cmap("viridis")
    cluster_colors = cmap(np.linspace(0.05, 0.95, n_clusters))
    rgba_colors = np.array([cluster_colors[i] for i in color_idx])
    rgba_colors[:, 3] = np.clip(confidence / confidence.max(), 0.15, 1.0)

    ax.scatter(
        pca[:, 0],
        pca[:, 1],
        c=rgba_colors,
        s=8,
        edgecolors="none",
    )

    vel = next_state - current_state
    stride = max(1, len(pca) // 400)
    idx = np.arange(0, len(pca), stride)

    magnitudes = np.linalg.norm(vel[idx], axis=1)
    mag_norm = magnitudes / (magnitudes.max() + 1e-8)

    ax.quiver(
        pca[idx, 0],
        pca[idx, 1],
        vel[idx, 0],
        vel[idx, 1],
        color="#c9d1d9",
        alpha=np.clip(mag_norm * 0.6 + 0.2, 0.2, 0.8),
        angles="xy",
        scale_units="xy",
        scale=1.5,
        width=0.0015,
        headwidth=3,
        headlength=4,
    )

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, n_clusters - 1))
    sm.set_array([])

    ax.set_xlabel("PC 1", fontsize=13, labelpad=8)
    ax.set_ylabel("PC 2", fontsize=13, labelpad=8)
    ax.set_title(
        "Phase Portrait — Fate Flow & Cluster Confidence",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.1)

    cax = fig.add_axes((0.92, 0.55, 0.015, 0.3))
    cb = fig.colorbar(
        sm,
        cax=cax,
        ticks=range(n_clusters),
    )
    cb.set_label("Cluster", fontsize=10, color="#c9d1d9")
    cb.ax.tick_params(colors="#8b949e")
    cb.outline.set_edgecolor("#30363d")  # pyright: ignore[reportCallIssue]

    ptime_ax = fig.add_axes((0.92, 0.15, 0.015, 0.3))
    ptime_cb = fig.colorbar(
        ScalarMappable(
            cmap="magma",
            norm=Normalize(pseudotime.min(), pseudotime.max()),
        ),
        cax=ptime_ax,
    )
    ptime_cb.set_label("Pseudotime", fontsize=10, color="#c9d1d9")
    ptime_cb.ax.tick_params(colors="#8b949e")
    ptime_cb.outline.set_edgecolor("#30363d")  # pyright: ignore[reportCallIssue]

    fig.subplots_adjust(right=0.90)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  phase      → {save_path}")
