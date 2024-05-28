#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import astropy.io.fits as fits
import numpy as np
import astropy.units as u
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunkit_instruments import suvi
from PIL import Image
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, LogStretch

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


'''
* 9.4 nm (FeXVIII)
* 13.1 nm (FeXXI)
* 17.1 nm (FeIX/X)
* 19.5 nm (FeXII)
* 28.4 nm (FeXV)
* 30.4 nm (HeII)
'''


available_channels = ['9.4 nm (FeXVIII)','13.1 nm (FeXXI)','17.1 nm (FeIX/X)',
                      '19.5 nm (FeXII)','28.4 nm (FeXV)','30.4 nm (HeII)\n']

print('The available SUVI channels are:')
print(*available_channels, sep='\n')

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

YEAR, MONTH, DAY = date.split('-')

# set the date and the data directory path
# YEAR = '2024'
# MONTH = '05'
# DAY = '14'
start_hour = '00'
start_minute = '00'
end_hour = '23'
end_minute = '59'
# channel = 304
# goes_sat_number = 18

data_dir = '/home/mnedal/data'

try:
    os.makedirs(f'{data_dir}/SUVI/{YEAR}{MONTH}{DAY}/{channel}A', exist_ok=True)
    os.makedirs(f'{data_dir}/output_png/suvi/{YEAR}{MONTH}{DAY}/{channel}A', exist_ok=True)
except:
    pass


start_time = pd.Timestamp(f'{YEAR}-{MONTH}-{DAY} {start_hour}:{start_minute}:00')
end_time = pd.Timestamp(f'{YEAR}-{MONTH}-{DAY} {end_hour}:{end_minute}:59')

results = Fido.search(a.Time(start_time, end_time),
                      a.Instrument('suvi'),
                      a.goes.SatelliteNumber(goes_sat_number),
                      a.Wavelength(channel*u.angstrom),
                      a.Level(2))

print(f'\nDownloading SUVI {channel}A data for {YEAR}-{MONTH}-{DAY} ..\n')

data = Fido.fetch(results, path=f'{data_dir}/SUVI/{YEAR}{MONTH}{DAY}/{channel}A/')
print('\nSUVI data is fetched sccessfully\n')

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


print('\nExporting SUVI png images ..\n')

for i, seq in enumerate(suvi_maps):
    output_map_filename = f'{data_dir}/output_png/suvi/{YEAR}{MONTH}{DAY}/{channel}A/{i:05d}_{channel}.png'
    
    if os.path.exists(output_map_filename):
        print(f'\nMap {i} already exists')
    
    else:
        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111, projection=seq)
        img = seq.plot(axes=ax)
        ax.grid(False)
        fig.colorbar(img, shrink=0.8, pad=0.02, label=seq.meta['bunit'])
        fig.savefig(output_map_filename, dpi=300, format='png')
        plt.close()
        print(f'\nMap {i} is done')

print('\nAll maps are exported successfully\n')


