# Loading in packages

import os
import pandas as pd
import xarray as xr

# Defining some settings and variables

cumulative_df = None
data_directory = "eReefs-monthly-data/" # Include the slash
depth_index = 16                        # Corresponds to a depth of -0.5 metres
latitude_downsample = 4
longitude_downsample = 4

# Looping through each data file

for file_name in os.listdir(data_directory):
    print(f"Currently processing {file_name}")
    split_results = file_name.split(".")[0].split("-")
    current_year = int(split_results[-2])  # May change depending on type of file
    current_month = int(split_results[-1]) # May change depending on type of file
    current_df = xr.open_dataset(f"{data_directory}{file_name}").to_dataframe().reset_index()
    selected_latitudes = current_df["latitude"].unique()[::latitude_downsample]
    selected_longitudes = current_df["longitude"].unique()[::longitude_downsample]
    current_df_processed = current_df[current_df["latitude"].isin(selected_latitudes) & \
                                      current_df["longitude"].isin(selected_longitudes) & \
                                      (current_df["k"] == depth_index) & \
                                      current_df["eta"].notna()].drop(columns = ["time", "k", "zc"]) \
                                                                .reset_index(drop = True)
    current_df_processed["year"] = current_year
    current_df_processed["month"] = current_month
    cumulative_df = pd.concat([cumulative_df, current_df_processed])

# Exporting the concatenated data
cumulative_df.to_csv("eReefs-aggregated-monthly-data.csv", index = False)
print("Finished processing all data files")
