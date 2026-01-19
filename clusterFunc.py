import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import umap
from matplotlib.cm import get_cmap
import json

def plot_umap_feature(Z_umap, feature, feature_name, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        Z_umap[:, 0], Z_umap[:, 1],
        c=feature,
        cmap=cmap,
        s=4,
        alpha=1.0
    )
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=15)
    plt.title(f"UMAP colored by {feature_name}")
    plt.xlabel("UMAP-1", fontsize=15)
    plt.ylabel("UMAP-2", fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_clusters(Z_umap, labels):
    # Unique label set (sorted)
    unique_labels = np.unique(labels)

    # Define a simple categorical palette
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow']
    
    # Build mapping: noise -> gray, clusters -> base colors
    color_map = {-1: 'gray'}
    for i, lab in enumerate(unique_labels):
        if lab == -1:
            continue
        color_map[lab] = base_colors[i % len(base_colors)]

    # Map labels → actual colors
    colors = [color_map[l] for l in labels]

    plt.figure(figsize=(10, 8))
    plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=colors, s=4, alpha=1.0)

    # --- Create legend ---
    handles = []
    for lab in unique_labels:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=("Noise" if lab == -1 else f"Cluster {lab}"),
                markerfacecolor=color_map[lab],
                markersize=8
            )
        )
    plt.legend(handles=handles, loc="best", fontsize=20)

    plt.xlabel("UMAP-1", fontsize=20)
    plt.ylabel("UMAP-2", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def main(latent_space_path, featureset_path, clusters_path, saveClusters):
    print("Clustering the latent space with HDBSCAN.")
    
    Z = np.load(latent_space_path)

    scaler = StandardScaler()
    Z_norm = scaler.fit_transform(Z)

    u = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric="euclidean", random_state=42)
    Z2 = u.fit_transform(Z_norm)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=1000,
        min_samples=5,
        metric='euclidean'
    )

    labels = clusterer.fit_predict(Z2)

    plot_clusters(Z2, labels)
    #probs = clusterer.probabilities_
    #plot_umap_feature(Z2, probs, "Cluster Confidence", cmap='viridis')

    if saveClusters:
        df = pd.read_csv(featureset_path)
        trajectory_ids = list(df.groupby("trajectory_id").groups.keys())
        clusters = {}
        for tid, lab in zip(trajectory_ids, labels):
            clusters.setdefault(int(lab), []).append(tid)

        with open(clusters_path, "w") as f:
            json.dump(clusters, f, indent=4)
    return

if __name__ == "__main__":
    main()

