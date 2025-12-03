import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import umap
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import json

# TODO: Cluster on 10D

# Make noise gray
cmap = get_cmap('tab20')
colors = list(cmap.colors)
colors.insert(0, (0.7, 0.7, 0.7))  # gray for noise
custom_cmap = mcolors.ListedColormap(colors)

def plot_umap_feature(Z_umap, feature, feature_name, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        Z_umap[:, 0], Z_umap[:, 1],
        c=feature,
        cmap=cmap,
        s=3,
        alpha=0.7
    )
    plt.colorbar(sc, label=feature_name)
    plt.title(f"UMAP colored by {feature_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

Z = np.load("../Latent/latent_vectors_3.npy")

scaler = StandardScaler()
Z_norm = scaler.fit_transform(Z)

u = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric="euclidean", random_state=42)
Z2 = u.fit_transform(Z_norm)

# min_cluster_size = 30, min_samples = 5, no leaf -> nice two clusters

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1500,
    min_samples=5,
    #cluster_selection_method="leaf", 
    metric='euclidean'
)
labels = clusterer.fit_predict(Z2)
np.save("cluster_labels3.npy", labels)

plot_umap_feature(Z2, labels, "HDBSCAN clusters", cmap=custom_cmap)
probs = clusterer.probabilities_
plot_umap_feature(Z2, probs, "Cluster Confidence", cmap='viridis')

mapTraj = True

if mapTraj:
    df = pd.read_csv("../../Featureset/2024FeatsNorm.csv")
    trajectory_ids = list(df.groupby("trajectory_id").groups.keys())
    clusters = {}
    for tid, lab in zip(trajectory_ids, labels):
        clusters.setdefault(int(lab), []).append(tid)

    with open("clusters3.json", "w") as f:
        json.dump(clusters, f, indent=4)



