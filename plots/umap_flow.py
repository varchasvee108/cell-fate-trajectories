import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

def generate_umap_flow_plot(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path)
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]

    # Use ONLY the last token in sequence
    x = inputs[:, -1].numpy()
    # Use median prediction only (0.5 quantile -> index 1)
    pred_next = quantile_preds[:, -1, :, 1].numpy()
    pseudo = pseudotime[:, -1].numpy()

    # Fit ONE shared UMAP reducer
    reducer = umap.UMAP(n_components=2, random_state=42)
    x_emb = np.asarray(reducer.fit_transform(x))
    
    # Transform predictions using SAME reducer
    pred_emb = np.asarray(reducer.transform(pred_next))

    # Subsample arrows
    n_points = x.shape[0]
    n_arrows = min(1000, n_points)
    np.random.seed(42)
    indices = np.random.choice(n_points, size=n_arrows, replace=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # UMAP scatter of current cell states, points colored by pseudotime
    scatter = ax.scatter(
        x_emb[:, 0], 
        x_emb[:, 1], 
        c=pseudo, 
        cmap='viridis', 
        s=10, 
        alpha=0.6,
        label="Current State"
    )

    # arrows from current state -> predicted next state
    ax.quiver(
        x_emb[indices, 0],
        x_emb[indices, 1],
        pred_emb[indices, 0] - x_emb[indices, 0],
        pred_emb[indices, 1] - x_emb[indices, 1],
        angles='xy', scale_units='xy', scale=1,
        color='#c9d1d9', alpha=0.3, width=0.002,
        label="Predicted Flow"
    )

    ax.set_title("UMAP Developmental Flow", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Pseudotime", fontsize=12)

    ax.legend(loc="upper right", frameon=True, facecolor="#161b22", edgecolor="#30363d", fontsize=10)
    
    # Styling
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.outline.set_edgecolor("#30363d")  # pyright: ignore[reportCallIssue]
    cbar.ax.yaxis.label.set_color("#c9d1d9")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#8b949e")
    ax.grid(True, alpha=0.15, color="#21262d")
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    print(f"  umap flow  → {output_path}")
