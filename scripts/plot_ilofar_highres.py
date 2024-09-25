#!/usr/bin/env python
# coding: utf-8

print('Importing packages')

import warnings
warnings.filterwarnings('ignore')

from sigpyproc.readers import FilReader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import dates
import matplotlib.dates as mdates
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ImageNormalize, PercentileInterval

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

path = '/home/mnedal/data/ilofar'




def freq_axis(freqs):
    '''
    Introduce gaps in the frequency axis of I-LOFAR REALTA data.
    '''
    gap1 = np.flipud(freqs[288]+(np.arange(59)*0.390625))
    gap2 = np.flipud(freqs[88]+(np.arange(57)*0.390625))
    ax_shape = 59+57-1
    new_freq = np.zeros(ax_shape+freqs.shape[0])
    
    new_freq[0:88] = freqs[0:88]
    new_freq[88:145]  = gap2[:57]
    new_freq[145:345] = freqs[88:288]
    new_freq[345:404] = gap1[:59]
    new_freq[404:] = freqs[289:]
    
    return new_freq







# filename = f'{path}/Sun357_20240514_stokesI.fil'

# a = FilReader(filename) # header
# header = a.header.to_dict()

# tstart_obs_str = Time(a.header.tstart, format='mjd').iso
# tstart = datetime.strptime(tstart_obs_str, '%Y-%m-%d %H:%M:%S.%f')

# tstart_str = '2024-05-14 16:35:00'
# tend_str   = '2024-05-14 17:50:00'
# datetime1 = datetime.strptime(tstart_str, '%Y-%m-%d %H:%M:%S')
# datetime2 = datetime.strptime(tend_str, '%Y-%m-%d %H:%M:%S')

# time_difference1 = datetime1 - tstart
# time_difference2 = datetime2 - datetime1

# secondsFromStart = time_difference1.total_seconds()
# totalTime = time_difference2.total_seconds()

# # downsampling the data to time resolution of 0.5 s. the time resolution to 0.5 s > 500 ms. And 500/1.31 ms = ~381. So tfactor should be 381.
# data = a.read_block(int(secondsFromStart / a.header.tsamp), int(totalTime / a.header.tsamp))

# Tres = 2                  # time resolution of 2 millisecond
# tres = a.header.tsamp*1e3 # time resolution in header

# # downsampling data applied by summing sequential sample/channels and averaging the data
# data2 = data.downsample(tfactor=int(Tres/tres))

# import the data files
print('Loading data')

Tres = 2
data = np.load(f'{path}/res_{Tres}ms_ilofar.npy')
time = np.load(f'{path}/time_{Tres}ms.npy', allow_pickle=True)
freqs = np.load(f'{path}/freq_{Tres}ms.npy')

print('Data loaded')

new_freq = freq_axis(freqs)

data = np.log10(data)
data[np.where(np.isinf(data)==True)] = 0.0

data2 = np.empty((new_freq.shape[0], data.shape[1]))    
data2[:] = np.NaN
data2[0:88] = data[0:88]
data2[145:345] = data[88:288]
data2[404:] = data[289:]

times_mpl = [mdates.date2num(t) for t in time]

freq_mode3 = np.linspace(10, 90, 199)
freq_mode5 = np.linspace(110, 190, 200)
freq_mode7 = np.linspace(210, 270, 88)

mode3 = {'time': times_mpl,
         'freq': freq_mode3[::-1],
         'data': data2[404:]}

mode5 = {'time': times_mpl,
         'freq': freq_mode5[::-1],
         'data': data2[145:345]}

mode7 = {'time': times_mpl,
         'freq': freq_mode7[::-1],
         'data': data2[0:88]}

del data
del data2
del new_freq
del times_mpl

fig_title = str(time[0].date()).replace("-","")
del time
del freqs

print('Data prepared')


fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(111)
ax.pcolormesh(mode3['time'], mode3['freq'], mode3['data'],
                    vmin=np.percentile(mode3['data'], 1), vmax=np.percentile(mode3['data'], 99),
                    cmap='RdYlBu_r')
ax.pcolormesh(mode5['time'], mode5['freq'], mode5['data'],
                    vmin=np.percentile(mode5['data'], 1), vmax=np.percentile(mode5['data'], 99),
                    cmap='RdYlBu_r')
ax.pcolormesh(mode7['time'], mode7['freq'], mode7['data'],
                    vmin=np.percentile(mode7['data'], 1), vmax=np.percentile(mode7['data'], 99),
                    cmap='RdYlBu_r')
ax.set_yscale('log')

# Define the custom ticks
custom_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
ax.set_yticks(custom_ticks)
ax.set_yticklabels([str(tick) for tick in custom_ticks])

ax.set_xlabel('Time (UT)')
ax.set_ylabel('Frequency (MHz)')
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylim(ax.get_ylim()[::-1])
fig.savefig(f'{path}/plots/realta_ilofar_{fig_title}.png', format='png', bbox_inches='tight')
fig.savefig(f'{path}/plots/realta_ilofar_{fig_title}.pdf', format='pdf', bbox_inches='tight')
plt.close()

print('Image is exported')





# remove the const background
mode3_new = mode3['data'] - np.tile(np.mean(mode3['data'],0), (mode3['data'].shape[0],1))
mode5_new = mode5['data'] - np.tile(np.mean(mode5['data'],0), (mode5['data'].shape[0],1))
mode7_new = mode7['data'] - np.tile(np.mean(mode7['data'],0), (mode7['data'].shape[0],1))

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(111)
ax.pcolormesh(mode3['time'], mode3['freq'], mode3_new,
                    vmin=np.percentile(mode3_new, 10), vmax=np.percentile(mode3_new, 98),
                    cmap='RdYlBu_r')
ax.pcolormesh(mode5['time'], mode5['freq'], mode5_new,
                    vmin=np.percentile(mode5_new, 10), vmax=np.percentile(mode5_new, 98),
                    cmap='RdYlBu_r')
ax.pcolormesh(mode7['time'], mode7['freq'], mode7_new,
                    vmin=np.percentile(mode7_new, 10), vmax=np.percentile(mode7_new, 98),
                    cmap='RdYlBu_r')
ax.set_yscale('log')

# Define the custom ticks
custom_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
ax.set_yticks(custom_ticks)
ax.set_yticklabels([str(tick) for tick in custom_ticks])

ax.set_xlabel('Time (UT)')
ax.set_ylabel('Frequency (MHz)')
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylim(ax.get_ylim()[::-1])
fig.savefig(f'{path}/plots/realta_ilofar_{fig_title}_backSubtract.png', format='png', bbox_inches='tight')
fig.savefig(f'{path}/plots/realta_ilofar_{fig_title}_backSubtract.pdf', format='pdf', bbox_inches='tight')
plt.close()

print('Code finished successfully')





