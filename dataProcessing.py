import pandas as pd
import os
import numpy as np

TYPE_CODE_FISHING_VESSEL = 30 # AIS code for fishing vessels, filtering out fishing vessels

REGION_LAT = 62 # We want all vessels north of 62 degrees north
REGION_LON_EAST = 40
REGION_LON_WEST = 0

def readParquetFile(filename):
    df = pd.read_parquet(filename, engine='pyarrow')
    return df

def extractFishingVessels(df):
    print("Extracting fishing vessels based on type code: ", TYPE_CODE_FISHING_VESSEL)
    fishingVessels = df.loc[df["ship_type"] == TYPE_CODE_FISHING_VESSEL].copy()
    return fishingVessels

def filterRegion(df):
    print("Filtering out all vessels that are not within the region of interest.")

    df.drop("geometry_wkt", axis=1, inplace=True) # Delete the geoemtry_wkt column

    insideRegion = df.loc[(df["lat"] >= REGION_LAT) & (df["lon"] >= REGION_LON_WEST) & (df["lon"] <= REGION_LON_EAST)]

    return insideRegion

def saveToCSV(filename, df):
    df.to_csv(filename)
    print("Succesfully saved to: ", filename)

def saveToParquet(outfile, df):
    df.to_parquet(outfile, engine="pyarrow", compression="snappy")
    print("Successfully saved to ", outfile)


def readFilterSave(filename, saveFilename):
    df = readParquetFile(filename)
    fishingVessels = extractFishingVessels(df)
    insideRegion = filterRegion(fishingVessels)
    #saveToCSV(saveFilename, insideRegion)
    saveToParquet(saveFilename, insideRegion)


def resample(df, step="1min"):
    print("Resampling")
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(by=["mmsi", "date_time_utc"])
    df = df.set_index("date_time_utc")
    #print(df.head())

    resampled = (
        df.groupby("mmsi", group_keys=False)
          .apply(lambda g: g.resample(step, origin=g.index.min()).first())
    )

    try:

        resampled["mmsi"] = resampled["mmsi"].astype("int64")
    except:
        print("Nah")
    #print(df.head())
    return resampled



""" directory = "Z:date_utc=2024-01-01"

for entry in os.scandir(directory):
    if entry.is_file() and entry.name.endswith(".parquet"):
        print(entry.path)
        try:
            readFilterSave(entry.path, "Processed_AIS/test_2024_01_01.parquet")
            #df = pd.read_parquet(entry.path, engine="pyarrow")
            #print(df.head())
        except:
            print(entry.path, " is not a parquet file.") """

