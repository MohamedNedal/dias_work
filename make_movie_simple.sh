#!/bin/bash

# Exit on error
set -e

# Check if folder path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_with_pngs>"
    exit 1
fi

# Get the data directory and its base name
DATA_DIR="$1"
BASENAME=$(basename "$DATA_DIR")

# Go to the data directory
cd "$DATA_DIR"

# Get the first image file
first_image=$(ls -1v *.png | head -n 1)

# Rename images if not already in desired format
if [[ "$first_image" =~ ^[0-9]{5}\.png$ && "$first_image" == "00001.png" ]]; then
    echo "Images are already in the correct format."
else
    echo "Renaming images to sequential format..."
    a=1
    for i in *.png; do
        new=$(printf "%05d.png" "$a")
        mv -- "$i" "$new"
        ((a++))
    done
fi

# Define output file name
OUTPUT_MOVIE="output_movie.mp4"

# Create the movie
echo "Making movie..."
ffmpeg -framerate 10 -i %05d.png -c:v libx264 -pix_fmt yuv420p "$OUTPUT_MOVIE"

echo "Movie created successfully at $DATA_DIR/$OUTPUT_MOVIE"
