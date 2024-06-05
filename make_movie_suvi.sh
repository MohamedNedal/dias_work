#!/bin/bash

# Ensure the script exits on error
set -e

# Check if the data directory and channel number are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <data_directory> <channel_number>"
    exit 1
fi

# Define the data directory and channel number
DATA_DIR=$1
CHANNEL=$2

# Navigate to the data directory
cd "$DATA_DIR"

# Check if the first image is already in the desired format
first_image=$(ls -1v *.png | head -n 1)
if [[ "$first_image" =~ ^[0-9]{5}\.png$ && "$first_image" =~ ^00001\.png$ ]]; then
    echo "Images are already in the desired format."
else
    echo "Renaming images to a numerical sequence..."
    # Temporarily rename your files to a numerical sequence
    a=1
    for i in *.png; do
        new=$(printf "%05d.png" ${a}) # e.g., 00001.png, 00002.png, etc.
        mv -- "$i" "$new"
        let a=a+1
    done
fi

## List the files to verify they have been renamed correctly
# echo "Listing files in $DATA_DIR:"
# ls -1v *.png

# Resize images to ensure dimensions are even
echo "Resizing images to ensure dimensions are even..."
for img in *.png; do
    convert "$img" -resize '2620x2574!' "$img"
done

# Define the output movie filename
OUTPUT_MOVIE="output_${CHANNEL}"

# Run ffmpeg to create the movie
echo "Running ffmpeg..."
ffmpeg -framerate 10 -i %05d.png -c:v libx264 -pix_fmt yuv420p $OUTPUT_MOVIE.mp4

# Print success message
echo "Movie created successfully at $DATA_DIR/$OUTPUT_MOVIE"



