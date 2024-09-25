#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import glob
import sunpy.map
from sunkit_instruments import suvi
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, LogStretch

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'




# Set up argument parser
parser = argparse.ArgumentParser(description='Process input attributes.')
parser.add_argument('date', type=str, help='The date in YYYY-mm-dd format')
parser.add_argument('channel', type=int, help='The channel number (e.g., 304A)')
parser.add_argument('goes_sat_number', type=int, help='The GOES satellite number (e.g., 18)')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
date = args.date
channel = args.channel
goes_sat_number = args.goes_sat_number

start_hour = '00'
start_minute = '00'
end_hour = '23'
end_minute = '59'

YEAR, MONTH, DAY = date.split('-')

data_dir = '/home/mnedal/data'

try:
    os.makedirs(f'{data_dir}/output_png/suvi/{YEAR}{MONTH}{DAY}/{channel}A/RD', exist_ok=True)
except:
    pass

# read the files names in order
if channel == 94:
    suvi_fits = sorted(glob.glob(f'{data_dir}/SUVI/{YEAR}{MONTH}{DAY}/{channel}A/*{channel}*g{goes_sat_number}*{YEAR}{MONTH}{DAY}*.fits'))
else:
    suvi_fits = sorted(glob.glob(f'{data_dir}/SUVI/{YEAR}{MONTH}{DAY}/{channel}A/*ci{channel}*g{goes_sat_number}*{YEAR}{MONTH}{DAY}*.fits'))

print(f'\nNumber of fetched SUVI files: {len(suvi_fits)}\n')


# import the files as maps and append them to a list
print('\nConvert fits files to sunpy maps ..\n')
suvi_maps = []
for file in suvi_fits:
    suvi_maps.append(suvi.files_to_map(file))


print('\nNormalize the maps ..\n')

min_range = 0

if channel == 94:
    max_range = 20
elif channel == 171:
    max_range = 20
elif channel == 131:
    max_range = 20
elif channel == 195:
    max_range = 50
elif channel == 284:
    max_range = 50
elif channel == 304:
    max_range = 100

for m in suvi_maps:
    m.plot_settings['norm'] = ImageNormalize(vmin=min_range, vmax=max_range, stretch=LogStretch())


print('\nApply running difference ..\n')

# Apply running difference
rundiff = [m - prev_m.quantity for m, prev_m in zip(suvi_maps[1:], suvi_maps[:-1])]
m_seq_running = sunpy.map.Map(rundiff, sequence=True)

# normalize the intensity ranges
low_thresh = -0.05
high_thresh = 0.05

print('\nExporting SUVI running-difference png images ..\n')

for i, m in enumerate(m_seq_running):
    output_map_filename = f'{data_dir}/output_png/suvi/{YEAR}{MONTH}{DAY}/{channel}A/RD/{i:05d}_{channel}.png'
    
    if os.path.exists(output_map_filename):
        print(f'\nMap {i} already exists')
    
    else:
        # normalize the intensity ranges
        m.plot_settings['norm'] = colors.Normalize(vmin=low_thresh, vmax=high_thresh)
        m.plot_settings['cmap'] = 'Greys_r'
        
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111, projection=m)
        m.plot(axes=ax)
        ax.grid(False)
        fig.savefig(f'{data_dir}/output_png/suvi/{YEAR}{MONTH}{DAY}/{channel}A/RD/{i:05d}_{channel}.png', dpi=300, format='png', facecolor='w')
        plt.close()
        print(f'\nMap {i} is done')


