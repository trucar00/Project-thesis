import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import json


""" data = np.random.rand(50, 2)  # 50 random 2D points
clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
labels = clusterer.fit_predict(data)

x = data[:, 0]
y = data[:, 1]

mask_clusters = labels != -1
mask_outliers = labels == -1

plt.figure(figsize=(6,6))

# Plot clustered points
plt.scatter(x[mask_clusters], y[mask_clusters], c=labels[mask_clusters],
            cmap='tab10', s=50, edgecolor='k', label='Clustered')

# Plot outliers
plt.scatter(x[mask_outliers], y[mask_outliers], c='black', s=50,
            label='Outliers')

plt.title('HDBSCAN Clustering with Outliers')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show() """



Z = np.load("latent_vectors.npy")
df = pd.read_csv("../Featureset/2024Feats.csv") # Featureset\2024Feats.csv

trajectory_ids = df["trajectory_id"].unique()

assert len(trajectory_ids) == Z.shape[0]

scaler = StandardScaler()
Z_norm = scaler.fit_transform(Z)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric='euclidean'
)

labels = clusterer.fit_predict(Z_norm)

label_dict = {}

for traj_id, label in zip(trajectory_ids, labels):
    label_dict.setdefault(label,[]).append(traj_id)

print(label_dict)

#with open("traj_clusters.json", "w") as f:
#    json.dump(label_dict, f, indent=4)

for label, traj_list in label_dict.items():
    print(f"{label} : {traj_list[:5]}... ({len(traj_list)} trajectories)")