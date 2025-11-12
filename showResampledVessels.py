import pandas as pd
import folium

df = pd.read_csv("Processed_AIS/Resampled/2024-05.csv")
#df = pd.read_csv("resampled_traj_inter.csv")

#print(df["trajectory_id"].unique())

df1 = df.loc[df["trajectory_id"] == "257046470-0"]
df2 = df.loc[df["trajectory_id"] == "257070440-0"]
df3 = df.loc[df["trajectory_id"] == "257149000-0"] 
df4 = df.loc[df["trajectory_id"] == "257386000-0"]
df5 = df.loc[df["trajectory_id"] == "257506600-0"] 

# Plotting on a map
m = folium.Map(location=[df['lat'].iloc[0], df['lon'].iloc[0]], zoom_start=12)

# Add markers for each position
for i, row in df1.iterrows():
    
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="red").add_to(m)

for i, row in df2.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="blue").add_to(m)

for i, row in df3.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="orange").add_to(m)

for i, row in df4.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="green").add_to(m)

for i, row in df5.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color="yellow").add_to(m)

#folium.CircleMarker(location=[70.06204, 20.918283], radius=4, color="black").add_to(m)
#folium.CircleMarker(location=[70.06133, 20.91314], radius=4, color="black").add_to(m)


m.save("Maps/resampled_may.html")
m

#20.918283,70.06204
# 20.91314,70.06133
