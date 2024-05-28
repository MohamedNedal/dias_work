#!/usr/bin/env python
# coding: utf-8

from astropy import units as u
from sunpy.net import Fido, attrs as a




year = '2021'
month = '09'
start_day = '18'
end_day = '19'
channel = 211

aia_result = Fido.search(a.Time(f'{year}-{month}-{start_day}', f'{year}-{month}-{end_day}'),
                                 a.Instrument('AIA'),
                                 a.Wavelength(channel*u.angstrom),
                                 a.Sample(1*u.min))

aia_files = Fido.fetch(aia_result, path='/Users/mnedal/DIAS/data/{instrument}/{file}')

print('AIA data is fetched sccessfully!')
