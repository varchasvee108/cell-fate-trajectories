import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_landscape(
    pca: np.ndarray,
    pseudotime: np.ndarray,
    cluster_preds: np.ndarray,
    n_clusters: int,
    save_path: Path,
):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, n_clusters))

    for c in range(n_clusters):
        mask = cluster_preds == c
        ax.scatter(
            pseudotime[mask],
            pca[mask, 0],
            pca[mask, 1],
            c=[colors[c]],
            s=1,
            alpha=0.7,
            label=f"Cluster {c}",
        )

    ax.set_xlabel("Pseudotime", fontsize=12, labelpad=10)
    ax.set_ylabel("PC 1", fontsize=12, labelpad=10)
    ax.set_zlabel("PC 2", fontsize=12, labelpad=10)
    ax.set_title("Waddington Landscape", fontsize=16, fontweight="bold", pad=20)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        fontsize=9,
        markerscale=4,
    )
    ax.view_init(elev=25, azim=-60)
    ax.xaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax.yaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax.zaxis.pane.fill = False  # pyright: ignore[reportAttributeAccessIssue]
    ax.grid(True, alpha=0.15)

    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  landscape  → {save_path}")
