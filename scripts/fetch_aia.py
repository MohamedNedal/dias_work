#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import logging
import sunpy
sunpy.log.setLevel(logging.WARNING) # Set SunPy's logger to only show WARNING or above

import os
from astropy import units as u
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

data_dir = '/home/mnedal/data'


date = '2025-05-11'
start_time = '00:00:00'
end_time   = '15:00:00'

# passbands = [94, 131, 171, 193, 211, 335]
passbands = [304]

with tqdm(total=len(passbands), desc=f'Fetching AIA data ...') as pbar:
    
    for channel in passbands:
        
        print(f'Downloading data for AIA channel {channel} on {date} from {start_time} to {end_time} ..')
        
        os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv1', exist_ok=True)
        
        aia_result = Fido.search(a.Time(f'{date}T{start_time}', f'{date}T{end_time}'),
                                         a.Instrument('AIA'),
                                         a.Wavelength(channel*u.angstrom),
                                         a.Sample(12*u.second))
        
        # aia_files = Fido.fetch(aia_result, path='/home/mnedal/data/{instrument}/{file}')
        aia_files = Fido.fetch(aia_result, path=f'{data_dir}/AIA/{channel}A/highres/lv1')
        print(f'AIA {channel} data is downloaded successfully')
        pbar.update(1)


