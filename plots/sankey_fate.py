import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def _bezier_segment(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def plot_sankey_fate(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    pseudotime = preds["pseudotime"]
    cluster_preds = preds["cluster_preds"]

    pseudo = pseudotime.numpy().reshape(-1)
    clusters = cluster_preds.numpy().reshape(-1)

    n_clusters = int(clusters.max()) + 1

    n_bins = 5
    edges = np.linspace(pseudo.min(), pseudo.max(), n_bins + 1)
    labels = [f"{edges[i]:.2f}\u2013{edges[i+1]:.2f}" for i in range(n_bins)]

    counts = np.zeros((n_bins, n_clusters))
    for b in range(n_bins):
        mask = (pseudo >= edges[b]) & (pseudo < edges[b + 1])
        if b == n_bins - 1:
            mask = (pseudo >= edges[b]) & (pseudo <= edges[b + 1])
        for c in range(n_clusters):
            counts[b, c] = (clusters[mask] == c).sum()

    props = counts / counts.sum(axis=1, keepdims=True)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, n_clusters))

    fig, ax = plt.subplots(figsize=(16, 10))

    bar_width = 0.12
    gap = 0.18
    x_positions = np.linspace(0.05, 0.95, n_bins)

    y_cum = np.zeros(n_bins)
    bar_patches = []
    for b in range(n_bins):
        for c in range(n_clusters):
            if props[b, c] == 0:
                continue
            h = props[b, c]
            rect = plt.Rectangle(
                (x_positions[b] - bar_width / 2, y_cum[b]),
                bar_width, h,
                facecolor=colors[c], edgecolor="#161b22", linewidth=0.8, alpha=0.9,
            )
            ax.add_patch(rect)
            bar_patches.append((b, c, y_cum[b], h))
            y_cum[b] += h

    for left_bin in range(n_bins - 1):
        left_cum = np.zeros(n_clusters)
        right_cum = np.zeros(n_clusters)
        for c in range(n_clusters):
            if props[left_bin, c] == 0:
                continue
            left_top = sum(props[left_bin, :c]) + props[left_bin, c]
            left_bottom = sum(props[left_bin, :c])
            right_top = sum(props[left_bin + 1, :c]) + props[left_bin + 1, c]
            right_bottom = sum(props[left_bin + 1, :c])

            lx0 = x_positions[left_bin] + bar_width / 2
            lx1 = lx0 + gap * 0.6
            rx0 = x_positions[left_bin + 1] - bar_width / 2
            rx1 = rx0 - gap * 0.6

            p0 = np.array([lx0, left_bottom])
            p1 = np.array([lx1, left_bottom])
            p2 = np.array([rx1, right_bottom])
            p3 = np.array([rx0, right_bottom])

            bot_curve = _bezier_segment(p0, p1, p2, p3)

            p0_top = np.array([lx0, left_top])
            p1_top = np.array([lx1, left_top])
            p2_top = np.array([rx1, right_top])
            p3_top = np.array([rx0, right_top])

            top_curve = _bezier_segment(p0_top, p1_top, p2_top, p3_top)

            verts = np.vstack([bot_curve, top_curve[::-1], bot_curve[0:1]])
            codes = (
                [Path.MOVETO]
                + [Path.LINETO] * (len(bot_curve) - 1)
                + [Path.LINETO] * len(top_curve)
                + [Path.CLOSEPOLY]
            )
            path = Path(verts, codes)
            patch = PathPatch(
                path, facecolor=colors[c], edgecolor="none", alpha=0.35, lw=0,
            )
            ax.add_patch(patch)

    for b in range(n_bins):
        ax.text(
            x_positions[b], 0.5, f"n={int(counts[b].sum())}",
            ha="center", va="center", fontsize=8, color="#8b949e",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9, color="#8b949e")
    ax.set_xlabel("Pseudotime Stage", fontsize=13, color="#c9d1d9")
    ax.set_ylabel("Fate Proportion", fontsize=13, color="#c9d1d9")
    ax.set_title("Fate Flow \u2014 Cluster Proportions Across Pseudotime", fontsize=16, fontweight="bold", pad=15, color="#c9d1d9")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[c], edgecolor="#161b22", linewidth=0.8) for c in range(n_clusters)]
    cluster_labels = [f"Cluster {c}" for c in range(n_clusters)]
    ax.legend(
        handles, cluster_labels,
        loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
        fontsize=9, ncol=2,
    )

    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.08, axis="y", color="#21262d")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  sankey      -> {output_path}")
