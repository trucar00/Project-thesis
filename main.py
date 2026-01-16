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

TRAJECTORY_LENGTH = 3
RESAMPLE_STEP = "30s"

AIS_PATH = f"Z:date_utc={YEAR}"
FILTERED_PATH = f"Processed_AIS_{YEAR}/Parquets_{TRAJECTORY_LENGTH}h/"
CONCAT_PATH = f"Processed_AIS_{YEAR}/Concatenated_{TRAJECTORY_LENGTH}h/"
CLEAN_PATH = f"Processed_AIS_{YEAR}/Cleaned_{TRAJECTORY_LENGTH}h/"
RESAMPLE_PATH = f"Processed_AIS_{YEAR}/Resampled_{TRAJECTORY_LENGTH}h/"
TRAINING_SETS_PATH = f"Training_sets_{YEAR}/{TRAJECTORY_LENGTH}h/"

def main():
    folder_paths = [FILTERED_PATH, CONCAT_PATH, CLEAN_PATH, RESAMPLE_PATH, TRAINING_SETS_PATH]

    for p in folder_paths:
        path = Path(p)

        if path.exists():
            print(f"[EXISTS]  {path}")
        else:
            path.mkdir(parents=True)
            print(f"[CREATED] {path}")

    #getData.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH)
    #concatParquets.main(months=MONTHS, days=DAYS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)
    cleanAIS.main(months=MONTHS, concat_path=CONCAT_PATH, cleaned_path=CLEAN_PATH, traj_length=TRAJECTORY_LENGTH)
    downSample.main(cleaned_path=CLEAN_PATH, resmapled_path=RESAMPLE_PATH, step=RESAMPLE_STEP)
    buildTrainingSets.main(resampled_path=RESAMPLE_PATH, training_path=TRAINING_SETS_PATH, months=MONTHS, traj_length=TRAJECTORY_LENGTH)
    #autoencoder.main()
    #latent_plot
    #cluster_plot
    #example trajectories
    # MODEL
        # AE
        # UMAP
        # Clustering
        # Plot clustering?


if __name__ == "__main__":
    main()