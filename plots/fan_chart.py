import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_fan_chart(
    pseudotime_seq: np.ndarray,
    next_state_seq: np.ndarray,
    quantile_preds_seq: np.ndarray,
    dims: list[int],
    save_path: Path,
):
    T = len(pseudotime_seq)
    x_axis = np.arange(T)
    n_dims = len(dims)

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    pt_labels = [f"{pseudotime_seq[i]:.2f}" for i in range(0, T, max(1, T // 6))]

    for idx, dim in enumerate(dims):
        ax = axes[idx]

        lower = quantile_preds_seq[:, dim, 0]
        median = quantile_preds_seq[:, dim, 1]
        upper = quantile_preds_seq[:, dim, 2]

        ax.fill_between(
            x_axis,
            lower,
            upper,
            alpha=0.25,
            color="#58a6ff",
            edgecolor="none",
            label="10%–90% PI",
        )
        ax.fill_between(
            x_axis,
            lower,
            median,
            alpha=0.15,
            color="#58a6ff",
            edgecolor="none",
        )
        ax.fill_between(
            x_axis,
            median,
            upper,
            alpha=0.15,
            color="#58a6ff",
            edgecolor="none",
        )

        ax.plot(x_axis, median, color="#58a6ff", linewidth=1.8, label="Median (q=0.5)")
        ax.plot(
            x_axis,
            lower,
            color="#58a6ff",
            linewidth=0.6,
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            x_axis,
            upper,
            color="#58a6ff",
            linewidth=0.6,
            linestyle="--",
            alpha=0.5,
        )

        ax.scatter(
            x_axis,
            next_state_seq[:, dim],
            s=8,
            color="#f78166",
            alpha=0.8,
            zorder=5,
            label="Actual next state",
        )

        ax.set_ylabel(f"PC {dim + 1}", fontsize=12, color="#c9d1d9")
        ax.set_title(
            f"Quantile Forecast — PC {dim + 1}",
            fontsize=13,
            fontweight="bold",
            color="#c9d1d9",
        )
        ax.grid(True, alpha=0.12)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="#161b22",
            edgecolor="#30363d",
            fontsize=8,
        )

    axes[-1].set_xticks(range(0, T, max(1, T // 6)))
    axes[-1].set_xticklabels(pt_labels, rotation=30, ha="right", fontsize=8)
    axes[-1].set_xlabel(
        "Sequence Index (pseudotime ticks)", fontsize=12, color="#c9d1d9"
    )

    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  fan chart  → {save_path}")
