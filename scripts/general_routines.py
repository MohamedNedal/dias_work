##!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Frequently used Python functions.'''
# @Author: David Long

import numpy as np
import cv2 
import os


## Function to get closest element in array to defined value
def closest(list, value):
    """
    Get the closest element to value in list

    Parameters
    ----------
    list: Numpy array
        Array of numbers in which to find the value
    value: Number
        Number to find in list
    
    Return: Integer of the number in list closest to value
    """
    ind = np.abs([elem - value for elem in list])
    return ind.argmin(0)

## Function to save image files as a movie
def convert_frames_to_movie(path_in, path_out, fps):
    """
    Convert folder of PNG images to a MP4 movie

    Parameters
    ----------
    path_in: String
        Location of folder with png images
    path_out: String
        Location of mp4 movie
    fps: int
        Frames per second (typically 15)

    Return: Saved mp4 movie
    """
    frame_array = []
    files = [img for img in os.listdir(path_in) if (img.endswith(".png") or img.endswith(".tiff"))]
    files.sort()

    mean_width = 0
    mean_height = 0

    for i in range(len(files)):
        frame = cv2.imread(os.path.join(path_in, files[i]))
        height, width, layers = frame.shape
        mean_width += width
        mean_height += height
        frame_array.append(frame)

    mean_width = int(mean_width / len(files))
    mean_height = int(mean_height / len(files))
    size = (mean_width, mean_height)

    out = cv2.VideoWriter(path_out,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    for i in range(len(frame_array)):
        resized = cv2.resize(frame_array[i], size)
        out.write(resized)
    out.release()

    cv2.destroyAllWindows()
