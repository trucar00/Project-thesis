import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import umap
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import json
import seaborn as sns

# TODO: Cluster on 10D

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
    #plt.title(f"UMAP colored by {feature_name}")
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
    #plt.xlim([7.5, 17.5])
    #plt.ylim([4.9, 17.5])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

Z = np.load("../Latent/lat_3h_64_sin.npy")

scaler = StandardScaler()
Z_norm = scaler.fit_transform(Z)

u = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric="euclidean", random_state=42)
Z2 = u.fit_transform(Z_norm)

# min_cluster_size = 30, min_samples = 5, no leaf -> nice two clusters
# 6h: 2500, 5 gave 3 clusters, clear distinction between fishing/non fishing
# 6h: 1500, 5 gave 4 clusters
# 6h: 2000, 10, -> cluster in center of fishin region, is it more fishing in the middle of this big region?

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1000,
    min_samples=5,
    #cluster_selection_method="leaf", 
    metric='euclidean'
)

labels = clusterer.fit_predict(Z2)
#np.save("cluster_labels.npy", labels)

# --- remap labels ---
labels_mapped = labels.copy()
labels_mapped[labels_mapped == -1] = 0
labels_mapped[labels_mapped > 0] += 1

# --- build vivid colormap ---
unique_clusters = np.unique(labels)
num_real_clusters = np.sum(unique_clusters >= 0)

vivid = sns.color_palette("tab10", num_real_clusters)  # or "hls"
palette = [(0.7, 0.7, 0.7)] + vivid   # gray first (index 0)

custom_cmap = mcolors.ListedColormap(palette)

plot_clusters(Z2, labels)
probs = clusterer.probabilities_
plot_umap_feature(Z2, probs, "Cluster Confidence", cmap='viridis')

mapTraj = False

if mapTraj:
    df = pd.read_csv("../../Featureset/2024FeatsNorm6h.csv")
    trajectory_ids = list(df.groupby("trajectory_id").groups.keys())
    clusters = {}
    for tid, lab in zip(trajectory_ids, labels):
        clusters.setdefault(int(lab), []).append(tid)

    with open("Final/6h_64_z_5.json", "w") as f:
        json.dump(clusters, f, indent=4)


# 3h_63_3: min_clust = 1000, min_samples = 5 -> 3 clusters
# 3h_63_3: min_clust = 1000, min_samples = 10 -> 3 clusters
# 3h_32_5: min_clust = 2000, min_samples = 5 -> 5 clusters
#3h_64_val_3: min_clust = 


# 3h_64_z : 1000, 5

