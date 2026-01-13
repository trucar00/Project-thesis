import getData
import concatParquets
import cleanAIS
import downSample
import buildTrainingSets
import autoencoder
from pathlib import Path

# Need folder paths, anc creation of these folders if they dont exist?

# Define the time period we are interested in
YEAR = 2024
MONTHS = 12
DAYS = 5

AIS_PATH = f"Z:date_utc={YEAR}"
FILTERED_PATH = f"Processed_AIS_{YEAR}/Parquets/"
CONCAT_PATH = f"Processed_AIS_{YEAR}/Concatenated/"
CLEAN_PATH = f"Processed_AIS_{YEAR}/Cleaned/"
RESAMPLE_PATH = f"Processed_AIS_{YEAR}/Resampled/"
TRAINING_SETS_PATH = f"Training_sets_{YEAR}/"

folder_paths = [FILTERED_PATH, CONCAT_PATH, CLEAN_PATH, RESAMPLE_PATH, TRAINING_SETS_PATH]

for p in folder_paths:
    path = Path(p)

    if path.exists():
        print(f"[EXISTS]  {path}")
    else:
        path.mkdir(parents=True)
        print(f"[CREATED] {path}")

def main():
    getData.main() # year, months, dates
    concatParquets.main()
    cleanAIS.main()
    downSample.main()
    buildTrainingSets.main() # need fixing
    autoencoder.main()

    # MODEL
        # AE
        # UMAP
        # Clustering
        # Plot clustering?