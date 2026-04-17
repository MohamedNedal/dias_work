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

mydate = '2025-10-06'
data_dir = '/home/mnedal/data'
passbands = [94, 131, 171, 193, 211, 304, 335]

print('=====================================\n Process AIA image from lv1 to lv1.5 with deconvolution \n =====================================')

def do_process(aia_file):
    """
    Process AIA image from lv1 to lv1.5 with deconvolution.
    """
    m = Map(aia_file)
    print(f'Upgrade AIA {channel}A {aia_file.split("/")[-1]} map to lv1.5 and deconvolve with PSF ..\n')
    psf                      = aiapy.psf.psf(m.wavelength)
    aia_map_deconvolved      = aiapy.psf.deconvolve(m, psf=psf, iterations=10)
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


mydate = mydate.replace('-','_')

for channel in passbands:
    os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)
    files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/aia.lev1.{channel}A_{mydate}T*.fits'))

    with tqdm(total=len(files), desc=f'Process AIA images ...') as pbar:
        for aia_file in files:
            do_process(aia_file)
            pbar.update(1)
    
    print(f'Processing of channel {channel}A is finished successfully')

# for channel in passbands:
#     os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)
#     files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/aia.lev1.{channel}A_{mydate}T*.fits'))

#     with tqdm(total=len(files), desc=f'Process AIA images ...') as pbar:
#         for aia_file in files:
#             output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{aia_file.split("/")[-1].replace("lev1", "lev15")}'
#             if os.path.exists(output_filename):
#                 print(f'{output_filename} exists and processed already')
#                 pass
#             else:
#                 do_process(aia_file)
#             pbar.update(1)
    
#     print(f'Processing of channel {channel}A is finished successfully')
