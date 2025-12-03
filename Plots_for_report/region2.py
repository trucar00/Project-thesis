import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString

# --- Define region edges ---
lon_min, lon_max = 0, 40
lat_min = 62
lat_max = 80   # only used to draw vertical lines

# Create three lines:
bottom = LineString([(lon_min, lat_min), (lon_max, lat_min)])
left   = LineString([(lon_min, lat_min), (lon_min, lat_max)])
right  = LineString([(lon_max, lat_min), (lon_max, lat_max)])

gdf_lines = gpd.GeoDataFrame(
    {"name": ["bottom", "left", "right"]},
    geometry=[bottom, left, right],
    crs="EPSG:4326"
).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(8, 8))

# Plot only the three boundary lines
gdf_lines.plot(ax=ax, edgecolor="red", linewidth=2)

# Expand extent based on all lines
minx, miny, maxx, maxy = gdf_lines.total_bounds
pad_x = (maxx - minx) * 0.2
pad_y = (maxy - miny) * 0.2
ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy)

# Add ESRI basemap
ctx.add_basemap(
    ax,
    source=ctx.providers.Esri.WorldGrayCanvas,
    zoom=4
)

ax.axis("off")
plt.tight_layout()
plt.show()
