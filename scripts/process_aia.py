#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
import os
import glob
from datetime import datetime
from sunpy.map import Map
from astropy import units as u
from astropy.coordinates import SkyCoord
from aiapy.calibrate.prep import correct_degradation
from aiapy.calibrate import register, update_pointing
import aiapy.psf
from tqdm import tqdm

data_dir = '/home/mnedal/data'
# passbands = [94, 131, 171, 193, 211, 335]
# nf = len(passbands)

channel = 171
date_time_str = '2024-05-14 17:36:05.0' # Your target date and time

target_datetime = datetime.strptime(f'{date_time_str}', '%Y-%m-%d %H:%M:%S.%f')

os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv1', exist_ok=True)
os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)


def extract_datetime(filename):
    """
    Function to extract the datetime from a filename.
    """
    # Split the filename and extract the date and time parts
    date_time_part = filename.split('/')[-1]                            # Extracts '2024_05_14T18_49_59.12'
    date_part = date_time_part.split('T')[0].split(f'{channel}A_')[-1]  # Extracts '2024_05_14'
    time_part = date_time_part.split('T')[1].split('Z')[0]              # Extracts '18_49_59.12'
    
    # Reformat date and time to standard datetime format
    date_str = date_part.replace('_', '-')  # '2024-05-14'
    time_str = time_part.replace('_', ':')  # '18:49:59.12'
    
    # Combine date and time and convert to datetime object
    return datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M:%S.%f')


def find_closest_filename(filenames, target_datetime):
    """
    Function to find the index of the filename with the closest datetime to a given target.
    """
    closest_index = None
    min_time_diff = None
    
    for i, filename in enumerate(filenames):
        file_datetime = extract_datetime(filename)
        
        # Calculate the absolute time difference
        time_diff = abs(file_datetime - target_datetime)
        
        # Update the closest file if this one is closer
        if min_time_diff is None or time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_index = i
    
    return closest_index





# find the file index with the nearest datetime to the given one above
files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
closest_index = find_closest_filename(files, target_datetime)
print(f'\nclosest index to the given date: {closest_index}\n')

aia_file = files[closest_index]

# load the file as a sunpy map
m = Map(aia_file)

print(f'Upgrade AIA {channel}A {aia_file.split("/")[-1]} map to lv1.5 and deconvolve with PSF ..\n')

# # crop the region of interest
# top_right   = SkyCoord(-840*u.arcsec, 420*u.arcsec, frame=m.coordinate_frame)
# bottom_left = SkyCoord(-920*u.arcsec, 300*u.arcsec, frame=m.coordinate_frame)
# submap      = m.submap(bottom_left, top_right=top_right)
# print(f'submap shape: {submap.data.shape}')

psf                      = aiapy.psf.psf(m.wavelength)
aia_map_deconvolved      = aiapy.psf.deconvolve(m, psf=psf)
print('Deconvolution is finished')
aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
print('Updating pointing is finished')
aia_map_registered       = register(aia_map_updated_pointing)
print('Registration is finished')
aia_map_corrected        = correct_degradation(aia_map_registered)
print('Degradation correction is finished')
aia_map_norm             = aia_map_corrected / aia_map_corrected.exposure_time
print('Exposure time correction is finished')

output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{aia_file.split("/")[-1].replace("lev1", "lev15")}'
aia_map_norm.save(output_filename, filetype='auto', overwrite=True)

print('Images prepared and exporeted with the region of interest selected')
