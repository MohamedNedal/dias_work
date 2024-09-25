#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import requests
import datetime
import numpy as np
import pandas as pd


df = pd.read_csv('./DATA/SN_d_tot_V2.0.csv')                             # Import sunspot number data
df['total_sunspot_num'] = df['total_sunspot_num'].replace(-1, np.nan)    # Replace -1 with NaN
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.drop(columns=['year', 'month', 'day'])

# Filter the DataFrame to include only data from 2019 onwards
sub_df = df[df['date'] >= '2019-01-01']
zero_sunspot_days = sub_df[sub_df['total_sunspot_num'] == 0]
print(len(zero_sunspot_days))


for t in zero_sunspot_days['date']:
    YEAR = t.date().year
    MONTH = t.date().month
    DAY = t.date().day

    if DAY < 9:
        DAY = f'0{DAY}'
    else:
        DAY = f'{DAY}'

    if MONTH < 9:
        MONTH = f'0{MONTH}'
    else:
        MONTH = f'{MONTH}'
    
    try:
        URL = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf/{YEAR}/stereo_level2_swaves_{YEAR}{MONTH}{DAY}_v02.cdf'
        response = requests.get(URL)
        open(os.path.join('./stereo_data', f'stereo_level2_swaves_{YEAR}{MONTH}{DAY}_v02.cdf'), 'wb').write(response.content)

        print(f'{t} is fetched sccessfully!')
    except:
        pass
