#!/bin/bash

# Source directory containing the CSV files
SRC_DIR="../camera_fused_label/fused_label_lidar12_cam24/masked_fusion_label_coco"

# Destination directory where the renamed files will be copied
DST_DIR="../sample_submission"

# Create the destination directory if it doesn't exist
if [ ! -d "$DST_DIR" ]; then
    mkdir -p "$DST_DIR"
fi

# Loop through all matching CSV files in the source directory
for file in "${SRC_DIR}"/Run_*_detections_fusion_lidar12_camera_search-based.csv; do
    # Extract the base name of the file
    filename=$(basename "$file")

    # Use sed to extract the Run number from the filename
    Y=$(echo "$filename" | sed -n 's/^Run_\([0-9]\+\)_.*$/\1/p')

    # Check if the Run number was successfully extracted
    if [ -z "$Y" ]; then
        echo "Could not extract Run number from $filename"
        continue
    fi

    # Construct the destination filename
    DEST_FILE="Detection_Classification_Localization_Submission_Run_${Y}.csv"

    # Copy and rename the file to the destination directory
    if cp "$file" "${DST_DIR}/${DEST_FILE}"; then
        echo "Copied and renamed $filename to ${DEST_FILE}"
    else
        echo "Failed to copy $filename"
    fi
done
