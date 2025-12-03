import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

def dfForPlot():
    df = pd.read_csv("../../Featureset/2024FeatsNorm.csv")

    df_traj = df.groupby("trajectory_id").agg({
        "avg_speed": "first",
        "std_speed": "first"
    }).reset_index()

   
    
    delcog_mean_abs = df.groupby("trajectory_id")["z_del_cog"] \
                        .apply(lambda x: np.mean(np.abs(x))) \
                        .reset_index(name="delcog_mean_abs")
    
    delcog_sum_abs = df.groupby("trajectory_id")["z_del_cog"] \
                        .apply(lambda x: np.sum(np.abs(x))) \
                        .reset_index(name="delcog_sum_abs")

    delcog_activity = df.groupby("trajectory_id")["z_del_cog"] \
                        .apply(lambda x: np.mean(np.abs(x) > 0.2)) \
                        .reset_index(name="delcog_activity")

    df_traj = df_traj.merge(delcog_mean_abs, on="trajectory_id")
    df_traj = df_traj.merge(delcog_sum_abs, on="trajectory_id")
    df_traj = df_traj.merge(delcog_activity, on="trajectory_id")

    print(df_traj.head())
    return df_traj


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

def plotPureLatent():
    Z = np.load("../Latent/latent_vectors_3.npy")
    scaler = StandardScaler()
    Z_norm = scaler.fit_transform(Z)

    # Load your feature file (or any file with trajectory_id order)

    u = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric="euclidean", random_state=42)
    Z2 = u.fit_transform(Z_norm)

    # Plotting pure latent space
    plt.scatter(Z2[:,0], Z2[:,1], s=3)
    plt.title("UMAP of latent space")
    plt.show()
    return Z2


if __name__ == "__main__":
    df_traj = dfForPlot()
    Z2 = plotPureLatent()
    plot_umap_feature(Z2, df_traj["avg_speed"], "avg_speed")
    plot_umap_feature(Z2, df_traj["std_speed"], "std_speed")
    plot_umap_feature(Z2, df_traj["delcog_mean_abs"], "delcog mean abs")
    plot_umap_feature(Z2, df_traj["delcog_sum_abs"], "delcog sum abs")
    plot_umap_feature(Z2, df_traj["delcog_activity"], "delcog activity")