import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely

df = pd.read_csv("AIS_data/ais_2025_09_01.csv")


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

test = gdf.iloc[0]
testPoint = shapely.Point(10.7, 59.599815) 
# Svalbard: 78.71337730864367, 17.522977113299284

svb = shapely.Point(17.5229, 78.7133)
svb2 = shapely.Point(17.5229, 77)
print(testPoint)

gdf_kontinentalSokkel = gpd.read_file("maritime_borders_norway.gml", layer="Kontinentalsokkel").to_crs("EPSG:4326")
gdf_territorialWaters = gpd.read_file("maritime_borders_norway.gml", layer="Territorialfarvann").to_crs("EPSG:4326")

both = pd.concat([gdf_kontinentalSokkel.geometry, gdf_territorialWaters.geometry]) # Stacks them into one geoseries

combined_geom = both.union_all() # Creating one polygon from all the polygons inside both territorial and continental

gdf_combinedWaters = gpd.GeoDataFrame(geometry=[combined_geom])
print("Ship: ", test.geometry.within(combined_geom))
print("Test: ", testPoint.within(combined_geom))
print("Svalbard: ", svb.within(combined_geom))
print("SvalbardOut: ", svb2.within(combined_geom))

#print(test.geometry.x) # The test.geometry is a point of the coordinates

ax = gdf_combinedWaters.plot(edgecolor="blue", facecolor="none", linewidth=0.8)
ax.set_ylim(55, 88)
ax.set_xlim(-15, 43)
ax.plot([], [], label="Norway's continental shelf", color="blue")
ax.plot([0, 0, 40, 40], [88, 62, 62, 88], color="red", linewidth=1.2, label="Region")

""" ax.scatter(test.geometry.x, test.geometry.y, color="blue", s=10)
ax.scatter(testPoint.x, testPoint.y, color="black", s=10)
ax.scatter(svb.x, svb.y, color="green", s=10)
ax.scatter(svb2.x, svb2.y, color="yellow", s=10) """
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="upper left")
plt.show()
#plt.savefig("Figures/region.eps")