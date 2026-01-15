import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def dfForPlot(featureset_path):
    df = pd.read_csv(featureset_path)

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

    #print(df_traj.head())
    return df_traj


def plot_umap_feature(Z_umap, feature, feature_name, cmap='viridis'):
    plt.figure(figsize=(11, 8))
    sc = plt.scatter(
        Z_umap[:, 0], Z_umap[:, 1],
        c=feature,
        cmap=cmap,
        s=5,
        alpha=0.7
    )
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=30)
    plt.xlabel("UMAP-1", fontsize=30)
    plt.ylabel("UMAP-2", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plotPureLatent(latent_space_path):
    Z = np.load(latent_space_path)
    print("Any NaNs?:", np.isnan(Z).any())
    print("NaN count:", np.isnan(Z).sum())
    print("Any inf?:", np.isinf(Z).any())
    scaler = StandardScaler()
    Z_norm = scaler.fit_transform(Z)

    # Load your feature file (or any file with trajectory_id order)

    u = umap.UMAP(n_neighbors=60, n_components=2, min_dist=0.1, metric="euclidean", random_state=42)
    Z2 = u.fit_transform(Z_norm)

    # Plotting pure latent space
    fig, ax = plt.subplots(figsize=(10,8), constrained_layout=True)

    ax.scatter(Z2[:,0], Z2[:,1], s=4)
    ax.set_xlabel("UMAP-1", fontsize=30)
    ax.set_ylabel("UMAP-2", fontsize=30)

    ax.tick_params(labelsize=30)
    ax.grid(True, linestyle='--', alpha=0.5)
    #plt.title("128 dim")
    plt.show()
    return Z2


def main(latent_space_path, featureset_path):
    df_traj = dfForPlot(featureset_path)
    Z2 = plotPureLatent(latent_space_path)
    plot_umap_feature(Z2, df_traj["avg_speed"], "avg_speed")
    plot_umap_feature(Z2, df_traj["std_speed"], "std_speed")
    plot_umap_feature(Z2, df_traj["delcog_activity"], "delcog activity")

if __name__ == "__main__":
    main(latent_space_path="../../Latent/2024norm6h.csv", featureset_path="../../Featureset/2024norm6h.csv")
