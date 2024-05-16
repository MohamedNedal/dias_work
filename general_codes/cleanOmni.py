# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:37:08 2021

@author: Mohamed Nedal 
""" 

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth needs to be the same across GPUs 
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

basedir = '/home/mnedal/TEC/omni/'

# In[]: Import excel files of yearly data 
Years = ['2014','2015','2016','2017','2018','2019','2020','2021']

df = []
for year in Years:
    df.append(pd.read_excel(basedir + 'omni_min_' + year + '.xlsx').astype('float64'))

# concatenate all lists together 
names = ['Year','Day','Hour','Minute',
                        'Bavg','Bx','By','Bz',
                        'Vsw','Vx','Vy','Vz',
                        'np','T','P','E','beta','MA','MsonicM','Dst']
dfall = pd.DataFrame(np.concatenate(df), columns=names)

# In[]: clean gaps with NaNs 
dfall['Dst'] = dfall['Dst'].replace(99.0, np.nan)
dfall = dfall.replace(99.9, np.nan)
dfall = dfall.replace(999.9, np.nan)
dfall = dfall.replace(99.99, np.nan)
dfall = dfall.replace(999.99, np.nan)
dfall = dfall.replace(9999.99, np.nan)
dfall = dfall.replace(99999.9, np.nan)
dfall = dfall.replace(9999999, np.nan)

# dfall = dfall.replace(np.nan, 0)

# define a new column at first position 
dfall.insert(0, 'Datetime', '')

# In[]: Convert from DOY to datetime 
for i in range(len(dfall)):
    year = dfall['Year'][i]
    doy = dfall['Day'][i]
    hh = dfall['Hour'][i]
    MM = dfall['Minute'][i]
    new_date = datetime.strptime(str(int(year))+str(' ')+str(int(doy)), '%Y %j')
    new_date.strftime('%Y/%m/%d')
    new_time = timedelta(hours=int(hh), minutes=int(MM))
    dfall['Datetime'][i] = new_date + new_time


dfall.drop({'Year','Day','Hour','Minute'}, inplace=True)

# In[]: export dataset 
dfall.to_csv(basedir + 'omni_2014_2021.csv', index=False)
