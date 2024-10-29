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
channel = 335
single_frame = False
# target_datetime = '2024-05-14 17:36:05.0' # Your target date and time

# start_time = ...
# end_time   = ...

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
        # print(type(file_datetime))
        # print(type(target_datetime))
        
        # Calculate the absolute time difference
        time_diff = abs(file_datetime - target_datetime)
        
        # Update the closest file if this one is closer
        if min_time_diff is None or time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_index = i
    
    return closest_index


def extract_datetime_v1(filename):
    import re
    # Regular expression to capture the date-time part of the filename
    pattern = r'_(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}\.\d{2})Z'
    
    # Search the filename for the pattern
    match = re.search(pattern, filename)
    
    if match:
        # Replace underscores with colons and hyphens to format as a standard date-time string
        datetime_str = match.group(1).replace('_', '-', 2).replace('_', ':').replace('T', ' ')
        return datetime_str
    else:
        return []  # Return empty list if no match found


def do_process(date_time_str):
    target_datetime = datetime.strptime(f'{date_time_str}', '%Y-%m-%d %H:%M:%S.%f')
    
    # find the file index with the nearest datetime to the given one above
    files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
    
    closest_index = find_closest_filename(files, target_datetime)
    
    print(f'\nclosest index to the given date: {closest_index}\n')
    
    # load the file as a sunpy map
    aia_file = files[closest_index]
    
    output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{aia_file.split("/")[-1].replace("lev1", "lev15")}'
    if os.path.exists(output_filename):
        print(f'{output_filename} exists and processed already.')
        pass
    else:
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
        
        aia_map_norm.save(output_filename, filetype='auto', overwrite=True) # overwrite bc I already have lv1.5 but without PSF deconvolve.
        
        print('Images prepared and exporeted with the region of interest selected')

# ====================================================================================================
# START FROM HERE ...
# ====================================================================================================

# datetime_list = []
# if single_frame:
#     datetime_list.append(target_datetime) 
# else:
#     files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
#     for file in files:
#         filename = file.split('/')[-1]
#         datetime_list.append(extract_datetime(filename))

# if len(datetime_list) == 1:
#     date_time_str = datetime_list[0]
# else:
#     for date_time_str in datetime_list:
#         print(f'Doing frame {date_time_str} now ..')
#         do_process(date_time_str)

datetime_list = []
if single_frame:
    datetime_list.append(target_datetime) 
else:
    files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
    for file in files:
        filename = file.split('/')[-1]
        datetime_list.append(extract_datetime_v1(filename))

if len(datetime_list) == 1:
    date_time_str = datetime_list[0]
else:
    for date_time_str in datetime_list:
        print(f'Doing frame {date_time_str} now ..')
        do_process(date_time_str)


