import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString
import numpy as np

# ----------------------------------------------------------
# Constants for Web Mercator
# ----------------------------------------------------------
R = 6378137.0  # radius used in Web Mercator

def lon_to_x(lon):
    # lon in degrees -> x in EPSG:3857
    return lon * np.pi * R / 180.0

def lat_to_y(lat):
    # lat in degrees -> y in EPSG:3857
    lat_rad = np.radians(lat)
    return R * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))

# ----------------------------------------------------------
# Region edges (in degrees)
# ----------------------------------------------------------
lon_min, lon_max = 0, 40
lat_min, lat_max = 62, 80   # top just for the vertical edges

# Create three lines: bottom, left, right in lon/lat
bottom = LineString([(lon_min, lat_min), (lon_max, lat_min)])
left   = LineString([(lon_min, lat_min), (lon_min, lat_max)])
right  = LineString([(lon_max, lat_min), (lon_max, lat_max)])

gdf_lines = gpd.GeoDataFrame(
    {"name": ["bottom", "left", "right"]},
    geometry=[bottom, left, right],
    crs="EPSG:4326"
).to_crs(epsg=3857)

bottomb = LineString([(lon_min, 56), (lon_max, 56)])
leftb   = LineString([(lon_min, 56), (lon_min, lat_max)])
rightb  = LineString([(lon_max, 56), (lon_max, lat_max)])
gdf_bound = gpd.GeoDataFrame(
    {"name": ["bottom", "left", "right"]},
    geometry=[bottomb, leftb, rightb],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# ----------------------------------------------------------
# Figure and main axis
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))

# Plot the three boundary lines (now in EPSG:3857)
gdf_lines.plot(ax=ax, edgecolor="red", linewidth=1.5, linestyle="dashed", zorder=5)

# Extent from lines, with padding
minx, miny, maxx, maxy = gdf_bound.total_bounds
print(miny)
pad_x = (maxx - minx) * 0.2
pad_y = (maxy - miny) * 0
ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)

# ----------------------------------------------------------
# Basemap (EPSG:3857)
# ----------------------------------------------------------
ctx.add_basemap(
    ax,
    source=ctx.providers.Esri.WorldGrayCanvas,
    zoom=5
)

# ----------------------------------------------------------
# Longitude / latitude ticks & grid (in degrees)
# ----------------------------------------------------------
# Choose tick locations in degrees
lon_ticks_deg = np.arange(lon_min, lon_max + 1e-6, 10)   # every 5°
lat_ticks_deg = np.arange(60, 81, 2)                    # every 2° from 60–80

# Convert to Web Mercator coordinates for placement
lon_ticks_merc = [lon_to_x(l) for l in lon_ticks_deg]
lat_ticks_merc = [lat_to_y(l) for l in lat_ticks_deg]

ax.set_xticks(lon_ticks_merc)
ax.set_yticks(lat_ticks_merc)

# Build labels with degree + E/W, N/S
lon_labels = [f"{abs(l):.0f}°{'E' if l >= 0 else 'W'}" for l in lon_ticks_deg]
lat_labels = [f"{abs(l):.0f}°{'N' if l >= 0 else 'S'}" for l in lat_ticks_deg]

ax.set_xticklabels(lon_labels)
ax.set_yticklabels(lat_labels)

# Style ticks and grid
ax.tick_params(color="gray", labelsize=12)
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
