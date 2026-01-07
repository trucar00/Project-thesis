import getData
import concatParquets
import cleanAIS
import downSample
import buildTrainingSets
import os

# Need folder paths, anc creation of these folders if they dont exist?

# Define the time period we are interested in
YEAR = 2024
MONTHS = 12
DAYS = 5

AIS_PATH = f"Z:date_utc={YEAR}"
FILTERED_PATH = f"Processed_AIS/Parquets/{YEAR}"
CONCAT_PATH = f"Processed_AIS/Concatenated/{YEAR}"
CLEAN_PATH = f"Processed_AIS/Cleaned/{YEAR}"
RESAMPLE_PATH = f"Processed_AIS/Resampled/{YEAR}"

# Define the time period we are interested in
YEAR = 2024
MONTHS = 12
DAYS = 5

print(AIS_PATH)

def main():
    getData.main() # year, months, dates
    concatParquets.main()
    cleanAIS.main()
    downSample.main()
    buildTrainingSets.main() # need fixing

    # MODEL
        # AE
        # UMAP
        # Clustering
        # Plot clustering?