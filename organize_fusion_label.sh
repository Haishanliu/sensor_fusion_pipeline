#!/bin/bash

# Define the base directory where your Run folders are located
base_dir="../datasets/validation_data_full"  # Change this to your actual base directory
target_dir='./camera_fused_label'  # Change this to your actual target directory in the sensor_fusion_pipline directory

target_saving_dir="$target_dir/fused_label_lidar12_cam24"
# all_fusion_dir="$target_dir/fused_label_lidar12_cam24_full"

# Create the All_fusion_table directory
mkdir -p "$target_saving_dir"

# Loop through each Run folder in the base directory
for run_folder in "$base_dir"/Run_*; do
    # Extract the run number from the folder name
    run_number=$(basename "$run_folder")
    
    origin_fusion_label_dir="$run_folder/masked_fusion_label_coco"

    # Check if the fusion_label directory exists in the current run folder
    if [ -d "$origin_fusion_label_dir" ]; then
        # Copy the fusion_label directory to the target fusion_table directory
        cp -r "$origin_fusion_label_dir" "$target_saving_dir"
        echo "Copied fusion_label of $run_number to $target_saving_dir"
    else
        echo "fusion_label not found in $run_folder"
    fi
done

echo "All fusion tables directories have been created."
