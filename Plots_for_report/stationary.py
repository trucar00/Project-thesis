import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

# Load datasets
df_raw = pd.read_parquet("../Processed_AIS/Concatenated/2024-01.parquet", engine="pyarrow")
df_mov = pd.read_csv("../Processed_AIS/Cleaned3h/2024-01.csv")

# MMSIs you want to plot separately
interesting = [257549800, 257656000, 257733500]

for mmsi in interesting:
    # Filter raw + cleaned for this MMSI
    d_raw = df_raw[df_raw["mmsi"] == mmsi]
    d_clean = df_mov[df_mov["mmsi"] == mmsi]

    # Convert to GeoDataFrame
    gdf_raw = gpd.GeoDataFrame(
        d_raw, geometry=gpd.points_from_xy(d_raw["lon"], d_raw["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    gdf_clean = gpd.GeoDataFrame(
        d_clean, geometry=gpd.points_from_xy(d_clean["lon"], d_clean["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Create a figure for this MMSI
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot raw + cleaned
    gdf_raw.plot(ax=ax, color="red", markersize=5, label="Raw")
    gdf_clean.plot(ax=ax, color="blue", markersize=5, label="Cleaned")

    # Add basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldGrayCanvas,
        zoom=8
    )

    ax.set_title(f"AIS Tracks for MMSI {mmsi}")
    ax.legend()
    plt.tight_layout()
    plt.show()
