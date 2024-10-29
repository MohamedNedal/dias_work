#!/usr/bin/env python
# coding: utf-8

import os
from astropy import units as u
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

data_dir = '/home/mnedal/data'


date = '2024-05-14'
start_time = '17:00:00'
end_time   = '19:00:00'

passbands = [94, 131, 171, 193, 211, 335]

with tqdm(total=len(passbands), desc=f'Fetching AIA data ...') as pbar:
    for channel in passbands:
        os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv1', exist_ok=True)
        
        aia_result = Fido.search(a.Time(f'{date}T{start_time}', f'{date}T{end_time}'),
                                         a.Instrument('AIA'),
                                         a.Wavelength(channel*u.angstrom),
                                         a.Sample(12*u.second))
        
        # aia_files = Fido.fetch(aia_result, path='/home/mnedal/data/{instrument}/{file}')
        aia_files = Fido.fetch(aia_result, path=f'{data_dir}/AIA/{channel}A/highres/lv1')
        print(f'AIA {channel} data is downloaded successfully')
        pbar.update(1)