#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import os
import glob
import sunpy.map
from sunpy.time import parse_time
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
# Show Plot in The Notebook 
import matplotlib
#matplotlib.use('nbagg')
# from aiapy.calibrate import normalize_exposure
from aiapy.calibrate import register, update_pointing
import astropy.io.fits as fits
import matplotlib.colors as colors
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
import matplotlib.animation as animation

import sunpy.map
from sunpy.net import Fido, attrs as a

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'



YEAR = '2021'
MONTH = '09'
DAY = '18'
start_hour = '00'
start_minute = '00'
end_hour = '23'
end_minute = '59'
channel = 211
local_dir = '/home/mnedal/data'

print(f'\nApplying running-difference to AIA {channel}A images on {YEAR}-{MONTH}-{DAY}\n')



aia_fits = sorted(glob.glob(f'{local_dir}/AIA/*_211*.fits'))
print(f'Number of imported files: {len(aia_fits)}')


# ignore images with exposure time less than a threshold
aia_maps = []

print('\nAppending maps ..\n')

for i, file in enumerate(aia_fits):
# for file in aia_fits:
    print(f'Appending fits file number {i} now ..')
    m = sunpy.map.Map(file)
    if not m.meta['exptime'] < 2.9:
        aia_maps.append(m)



m_seq = sunpy.map.Map(aia_maps, sequence=True)

del aia_maps

# define a common normalization to use in the animation
print('\nNormalizing maps ..\n')

for i, m in enumerate(m_seq):
    print(f'Normalize map number {i} now ..')
    m.plot_settings['norm'] = ImageNormalize(vmin=0,
                                             vmax=2e3,
                                             stretch=SqrtStretch())


# Running-Difference

# make nested directories recursively
try:
    os.makedirs(f'{local_dir}/output_png/aia/RD/', exist_ok=True)
except:
    pass

print('\nMaking running-difference maps ..\n')

rundiff = [m - prev_m.quantity for m, prev_m in zip(m_seq[1:], m_seq[:-1])]
m_seq_running = sunpy.map.Map(rundiff, sequence=True)

del rundiff

# normalize the intensity ranges
low_thresh = -10
high_thresh = 10
norm = colors.Normalize(vmin=low_thresh, vmax=high_thresh)

print('\nNormalizing the running-difference maps ..\n')

for i, m in enumerate(m_seq_running):
    print(f'Normalizing RD map number {i} now ..')
    m.plot_settings['norm'] = norm
    m.plot_settings['cmap'] = 'Greys_r'





# Extract individual frames and export them as PNG images
print('\nExporting the running-difference images ..\n')

for i, m in enumerate(m_seq_running):
    print(f'Exporting map number {i} now ..')
    
    fig = plt.figure(figsize=[5,5])
    ax = fig.add_subplot(111, projection=m_seq_running[0])
    m.plot()
    # m.draw_limb()
    # m.draw_grid()
    ax.grid(False)
    fig.savefig(f"{local_dir}/output_png/aia/RD/{m.meta['date-obs']}.png", dpi=300, format='png', facecolor='w')
    plt.close()

print('\nDone successfully\n')


