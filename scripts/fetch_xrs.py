#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sunpy.net import Fido, attrs as a


df = pd.read_csv('./DATA/SN_d_tot_V2.0.csv')                             # Import sunspot number data
df['total_sunspot_num'] = df['total_sunspot_num'].replace(-1, np.nan)    # Replace -1 with NaN
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.drop(columns=['year', 'month', 'day'])

# Filter the DataFrame to include only data from 2019 onwards
# sub_df = df[df['date'] >= '2019-01-01']

# Filter the DataFrame to include only data within a range
sub_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-12-31')]

zero_sunspot_days = sub_df[sub_df['total_sunspot_num'] == 0]
print(len(zero_sunspot_days))


for t in zero_sunspot_days['date']:
    try:
        goes_result = Fido.search(a.Time(f'{t.date()}', f'{t.date()}'),
                                  a.Instrument('XRS'),
                                  a.goes.SatelliteNumber(16) | a.goes.SatelliteNumber(17))
        goes_files = Fido.fetch(goes_result, path='./{instrument}/{file}')
        print(f'{t} is fetched sccessfully!')
    except:
        pass
