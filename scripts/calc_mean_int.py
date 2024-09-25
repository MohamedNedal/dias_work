#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
import glob
import numpy as np
import pandas as pd
from sunpy.sun import constants as const
import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.coordinates import RotatedSunFrame
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst, get_horizons_coord
from matplotlib.patches import ConnectionPatch
import matplotlib
import numpy as np
import astropy.io.fits as fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from aiapy.calibrate import normalize_exposure, register, update_pointing

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300




year = '2021'
month = '09'
day = '18'
channel = 211

aia_files = sorted(glob.glob(f'./AIA/*{channel}*_{year}_{month}_{day}*.fits'))





# store all the mean intensities of the region of interest
list_mean_int = []

# load and calibrate the map
aiamap = sunpy.map.Map(aia_files[0])
aiamap_updated_pointing = update_pointing(aiamap)
aiamap_registered = register(aiamap_updated_pointing)
aiamap_normalized = normalize_exposure(aiamap_registered)

# We can differentially rotate this coordinate by using `RotatedSunFrame` with an array of observation times. Let’s define a daily cadence for +/- five days.
center_point = SkyCoord(560*u.arcsec, 6*u.arcsec, frame=aiamap_normalized.coordinate_frame)
increment = 40*u.arcsec
duration = range(1, len(aia_files), 1)*u.minute

top_left  = SkyCoord(center_point.Tx - increment, center_point.Ty + increment, frame=aiamap_normalized.coordinate_frame)
top_right = SkyCoord(center_point.Tx + increment, center_point.Ty + increment, frame=aiamap_normalized.coordinate_frame)

bottom_left  = SkyCoord(center_point.Tx - increment, center_point.Ty - increment, frame=aiamap_normalized.coordinate_frame)
bottom_right = SkyCoord(center_point.Tx + increment, center_point.Ty - increment, frame=aiamap_normalized.coordinate_frame)

diffrot_center_point = SkyCoord(RotatedSunFrame(base=center_point, duration=duration))

diffrot_topleft_point  = SkyCoord(RotatedSunFrame(base=top_left, duration=duration))
diffrot_topright_point = SkyCoord(RotatedSunFrame(base=top_right, duration=duration))

diffrot_bottomleft_point  = SkyCoord(RotatedSunFrame(base=bottom_left, duration=duration))
diffrot_bottomright_point = SkyCoord(RotatedSunFrame(base=bottom_right, duration=duration))

# plot the map with the points
fig = plt.figure(figsize=[15,10])
ax = fig.add_subplot(121, projection=aiamap_normalized)
aiamap_normalized.plot(axes=ax, vmin=0)
ax.grid(b=False)

ax.plot_coord(center_point, 'rx', fillstyle='none')
ax.plot_coord(top_left, 'bo', fillstyle='none')
ax.plot_coord(top_right, 'bo', fillstyle='none')
ax.plot_coord(bottom_left, 'bo', fillstyle='none')
ax.plot_coord(bottom_right, 'bo', fillstyle='none')

# rotate a submap instead of a point
aia_sub = aiamap_normalized.submap(bottom_left, top_right=top_right)
ax = fig.add_subplot(122, projection=aia_sub)
im = aia_sub.plot(axes=ax, vmin=0)
plt.colorbar(im, shrink=0.6, pad=0.02, label=aia_sub.meta['bunit'])
ax.grid(b=False)
fig.tight_layout()

image_path = f"./plots/aia_maps/{aia_files[0].split('/')[-1][4:-10]}.png"
if not os.path.exists(image_path):
    fig.savefig(image_path, format='png', dpi=100, bbox_inches='tight')

mean_int = np.nanmean(aia_sub.data)
obs_time = aia_sub.meta['date-obs'].replace('T', ' ')
list_mean_int.append((obs_time, mean_int))
plt.close()



# Run the above in a for loop over all the AIA maps for the day to track the bright spot
# I have to run the loop in batches because I run on my local computer!

for i, map in enumerate(aia_files[1:]):

    # load and calibrate the map
    aiamap = sunpy.map.Map(map)
    aiamap_updated_pointing = update_pointing(aiamap)
    aiamap_registered = register(aiamap_updated_pointing)
    aiamap_normalized = normalize_exposure(aiamap_registered)

    print(f'Working on file number: {i}')
    print(f"{aiamap_normalized.meta['date-obs'].replace('T', ' ')}\n")

    # plot the map with the rotated points
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(121, projection=aiamap_normalized)
    aiamap_normalized.plot(axes=ax, vmin=0)
    ax.grid(b=False)

    ax.plot_coord(diffrot_center_point[i], 'rx', fillstyle='none')
    ax.plot_coord(diffrot_topleft_point[i], 'bo', fillstyle='none')
    ax.plot_coord(diffrot_topright_point[i], 'bo', fillstyle='none')
    ax.plot_coord(diffrot_bottomleft_point[i], 'bo', fillstyle='none')
    ax.plot_coord(diffrot_bottomright_point[i], 'bo', fillstyle='none')

    # rotate the submap
    aia_sub = aiamap_normalized.submap(diffrot_bottomleft_point[i], top_right=diffrot_topright_point[i])

    ax = fig.add_subplot(122, projection=aia_sub)
    im = aia_sub.plot(axes=ax, vmin=0)
    plt.colorbar(im, shrink=0.6, pad=0.02, label=aia_sub.meta['bunit'])
    ax.grid(b=False)
    fig.tight_layout()

    image_path = f"./plots/aia_maps/{map.split('/')[-1][4:-10]}.png"
    if not os.path.exists(image_path):
        fig.savefig(image_path, format='png', dpi=100, bbox_inches='tight')
    
    mean_int = np.nanmean(aia_sub.data)
    obs_time = aia_sub.meta['date-obs'].replace('T', ' ')
    list_mean_int.append((obs_time, mean_int))
    
    plt.close()




df = pd.DataFrame(list_mean_int)
df.to_csv(f'./mean_intensities_{channel}a.csv')
print('Table exported successfully!')
