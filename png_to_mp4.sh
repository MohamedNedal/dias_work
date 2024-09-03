#!/bin/bash

# Change the names of the folders to the desired location
images_folder = 'aia' #'suvi/20240514'
channel = '193A'
# 131A, 171A, 195A, 284A, 304A, 94A

# Navigate to the directory containing your images
cd ~/data/png/'$images_folder'/'$channel'

# List the images in natural sort order and save to images.txt
ls -1v *.png > images.txt
# ls -1v *.png | sed 's|^|file |' > images.txt

# Prepend 'file ' to each line in images.txt
sed 's|^|file |' images.txt > formatted_images.txt

# Create the video using ffmpeg
ffmpeg -f concat -safe 0 -i formatted_images.txt -c:v libx264 -r 12 -pix_fmt yuv420p output_AIA'$channel'_12fps.mp4
