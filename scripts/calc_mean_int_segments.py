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
channel = 94

aia_files = sorted(glob.glob(f'./AIA/*{channel}*_{year}_{month}_{day}*.fits'))





# Initialize an empty list to store dictionaries
data = []

for i, file in enumerate(aia_files):
    print(f"{i} \t Working on {file.split('/')[-1]} ...")

    # load and calibrate the AIA map
    aiamap = sunpy.map.Map(file)
    aiamap_updated_pointing = update_pointing(aiamap)
    aiamap_registered = register(aiamap_updated_pointing)
    aiamap_normalized = normalize_exposure(aiamap_registered)

    # Divide the solar disk into segments
    top_right = SkyCoord(-500*u.arcsec, 1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-1000*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub1 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(0*u.arcsec, 1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-500*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub2 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(500*u.arcsec, 1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(0*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub3 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(1000*u.arcsec, 1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(500*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub4 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(-500*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-1000*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub5 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(0*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-500*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub6 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(500*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub7 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(1000*u.arcsec, 500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(500*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub8 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(-500*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-1000*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub9 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-500*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub10 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(500*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(0*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub11 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(1000*u.arcsec, 0*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(500*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub12 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(-500*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-1000*u.arcsec, -1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub13 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(0*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(-500*u.arcsec, -1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub14 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(500*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(0*u.arcsec, -1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub15 = aiamap.submap(bottom_left, top_right=top_right)

    top_right = SkyCoord(1000*u.arcsec, -500*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    bottom_left = SkyCoord(500*u.arcsec, -1000*u.arcsec, frame=aiamap_normalized.coordinate_frame)
    aia_sub16 = aiamap.submap(bottom_left, top_right=top_right)

    int_dict = {
        'datetime': aiamap_normalized.meta['date-obs'].replace('T', ' '),
        'avg_int_seg1': np.nanmean(aia_sub1.data),
        'avg_int_seg2': np.nanmean(aia_sub2.data),
        'avg_int_seg3': np.nanmean(aia_sub3.data),
        'avg_int_seg4': np.nanmean(aia_sub4.data),
        'avg_int_seg5': np.nanmean(aia_sub5.data),
        'avg_int_seg6': np.nanmean(aia_sub6.data),
        'avg_int_seg7': np.nanmean(aia_sub7.data),
        'avg_int_seg8': np.nanmean(aia_sub8.data),
        'avg_int_seg9': np.nanmean(aia_sub9.data),
        'avg_int_seg10': np.nanmean(aia_sub10.data),
        'avg_int_seg11': np.nanmean(aia_sub11.data),
        'avg_int_seg12': np.nanmean(aia_sub12.data),
        'avg_int_seg13': np.nanmean(aia_sub13.data),
        'avg_int_seg14': np.nanmean(aia_sub14.data),
        'avg_int_seg15':np.nanmean(aia_sub15.data),
        'avg_int_seg16': np.nanmean(aia_sub16.data)
    }

    # Append dictionary to the list
    data.append(int_dict)



df = pd.DataFrame(data)
df.to_csv(f'./mean_intensities_segments_{channel}a.csv')
print('Table exported successfully!')
