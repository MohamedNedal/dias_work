#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
# os.environ['CDF_LIB'] = '/home/peijin/cdf/cdf38_0-dist/lib'

import sys
import time
import requests
import argparse
import datetime
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

from astropy import units as u
from astropy.visualization import ImageNormalize, SqrtStretch, PercentileInterval

import sunpy.map
from sunpy import timeseries as ts
from sunpy.net import Fido, attrs as a

from scipy.ndimage import gaussian_filter
from solarmach import SolarMACH
import pyspedas
# from pytplot import tplot
# from pytplot import options
from pytplot import get_data
from spacepy import pycdf

mpl.rcParams['date.epoch'] = '1970-01-01T00:00:00'
try:
    mdates.set_epoch('1970-01-01T00:00:00')
except:
    pass

#plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'







# Check if a variable is defined either locally or globally
def is_defined(variable_name):
    local_vars = locals()
    global_vars = globals()
    return variable_name in local_vars or variable_name in global_vars





def process_events(event=None, start_date=None, end_date=None, individual=True):
    if individual:
        print(f'Processing individual event: {event} ..')

        # make sure that the date is a pd.Timestamp object
        if not isinstance(event, pd.Timestamp):
            YEAR, MONTH, DAY = split_date(dt=pd.Timestamp(t))
        
        try:
            ######################### S/C locations #########################
            fetch_locations(dt=t)

            ######################### XRS #########################            
            goes = fetch_xrs(dt=t)
            
            ######################### AIA #########################
            m = fetch_aia(dt=t)

            ######################### WIND #########################            
            wind_time, wind_freq, wind_data, wind_norm = fetch_waves(dt=t)

            ######################### STA ##########################
            time_ste, freq_ste, data_ste_A, sta_norm = fetch_STAswaves(year=YEAR, month=MONTH, day=DAY)

            ######################### PSP #########################
            df_psp, psp_norm = fetch_PSPfields(year=YEAR, month=MONTH, day=DAY)
        
        except:
            print('Error: Missing data files.')
        
        ######################### BIG PLOT #########################            
        fig = plt.figure(figsize=[20,10])
        gs = GridSpec(4, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.pcolormesh(df_psp.columns, df_psp.index, df_psp.values, norm=psp_norm, cmap='RdYlBu_r')
        ax1.set_ylabel('Frequency (MHz)')
        ax1.set_title(f'PSP/FIELDS: {df_psp.index[0]*1e3:.2f} kHz $-$ {df_psp.index[-1]:.2f} MHz')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(data_ste_A, aspect='auto', origin='lower', norm=sta_norm,
                    extent=[mdates.date2num(time_ste[0]), mdates.date2num(time_ste[-1]),
                            freq_ste[0]/1e3, freq_ste[-1]/1e3], cmap='RdYlBu_r')
        ax2.grid(False)
        ax2.set_ylabel('Frequency (MHz)')
        ax2.set_title(f'STEREO/SWAVES: {freq_ste[0]:.2f} kHz $-$ {freq_ste[-1]/1e3:.2f} MHz')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.imshow(wind_data, aspect='auto', origin='lower', norm=wind_norm,
                    extent=[mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(wind_time)[0], '%Y-%m-%d %H:%M:%S.%f')), 
                            mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(wind_time)[-1], '%Y-%m-%d %H:%M:%S.%f')), 
                            wind_freq[0]/1e3, wind_freq[-1]/1e3], cmap='RdYlBu_r')
        ax3.grid(False)
        ax3.xaxis_date()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.set_ylabel('Frequency (MHz)')
        ax3.set_title(f'Wind/WAVES: {wind_freq[0]:.2f} kHz $-$ {wind_freq[-1]/1e3:.2f} MHz')
        
        ax4 = fig.add_subplot(gs[3, 0])
        goes[0].plot(axes=ax4)
        ax4.set_xlabel('Time (UT)')
        ax4.set_title(f'{goes.observatory}/{goes[0].source.upper()}')
        
        # Load the PNG image
        img = mpimg.imread(f'./DIAS/locations/{event}.png')
        ax5 = fig.add_subplot(gs[:, 1])
        ax5.imshow(img)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[:, 2], projection=m)
        m.plot(axes=ax6, clip_interval=(1, 99.99)*u.percent)
        ax6.grid(False)
        ax6.set_xlabel('Helioprojective Lonitude (Solar-X)')
        ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')

        fig.tight_layout()
        fig.savefig(f'./plots/summary_plot_{event}.png', format='png', bbox_inches='tight')
        plt.close()

        print(f'Summary plot for {event} has been exported.\n')




    else:
        # ### Find the spotless days
        # https://www.sidc.be/SILSO/dayssnplot
        df = pd.read_csv('./DATA/SN_d_tot_V2.0.csv')                             # Import sunspot number data
        df['total_sunspot_num'] = df['total_sunspot_num'].replace(-1, np.nan)    # Replace -1 with NaN
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df.drop(columns=['year', 'month', 'day'])
        sub_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]       # Filter the DataFrame to include only data within a range
        zero_sunspot_days = sub_df[sub_df['total_sunspot_num'] == 0]             # Filter the DataFrame to find the days where the sunspot number was zero        
        print(f'\nProcessing list of {len(zero_sunspot_days)} spotless days between {start_date} and {end_date} ..\n')


        for t in zero_sunspot_days['date']:
            print(f'Doing {t} ..\n')

            YEAR, MONTH, DAY = split_date(dt=t)

            #################################################################
            ######################### S/C locations #########################
            #################################################################

            try:
                fetch_locations(dt=t)
                print('Locations data are loaded.')
            except:
                print('Locations map is missing!')

            #######################################################
            ######################### XRS #########################
            #######################################################
            
            try:
                goes = fetch_xrs(dt=t, year=YEAR, month=MONTH, day=DAY)
                print('GOES data is loaded.')
            except:
                print('GOES data is missing!')
                goes = []
            
            #######################################################
            ######################### AIA #########################
            #######################################################

            try:
                m = fetch_aia(dt=t)
                print('AIA data is loaded.')
            except:
                print('AIA data is missing!')
                m = []

            ########################################################
            ######################### WIND #########################
            ########################################################
            
            wind_result = fetch_waves(dt=t, year=YEAR, month=MONTH, day=DAY)

            if wind_result is not None:
                wind_time, wind_freq, wind_data, wind_norm = wind_result
                print('Wind data is loaded.')
            else:
                print('Wind data is missing!')
                wind_time = []
                wind_freq = []
                wind_data = []
                wind_norm = []
            
            ########################################################
            ######################### STA ##########################
            ########################################################

            steA_result = fetch_STAswaves(year=YEAR, month=MONTH, day=DAY)

            if steA_result is not None:
                time_ste, freq_ste, data_ste_A, ste_norm = steA_result
                print('STEREO data is loaded.')
            else:
                print('STEREO data is missing!')
                time_ste = []
                freq_ste = []
                data_ste_A = []
                ste_norm = []

            #######################################################
            ######################### PSP #########################
            #######################################################

            psp_result = fetch_PSPfields(year=YEAR, month=MONTH, day=DAY, data_version=3)

            if psp_result is not None:
                df_psp, psp_norm = psp_result
                print('PSP data is loaded.')
            else:
                print('PSP data is missing!')
                df_psp = []
                psp_norm = []
            
            #######################################################

            if is_defined('goes'):
                raise ValueError('`goes` is undefined in the `data_dict object!')
                
            elif is_defined('m'):
                raise ValueError('`m` is undefined in the `data_dict object!')
                
            elif is_defined('wind_time') or is_defined('wind_freq') or is_defined('wind_data') or is_defined('wind_norm'):
                raise ValueError('One of the `wind` variables is undefined in the `data_dict object!')
            
            elif is_defined('time_ste') or is_defined('freq_ste') or is_defined('data_ste_A') or is_defined('ste_norm'):
                raise ValueError('One of the `stereo` variables is undefined in the `data_dict object!')
                
            elif is_defined('df_psp') or is_defined('psp_norm'):
                raise ValueError('One of the `psp` variables is undefined in the `data_dict object!')
            
            else:
                print('Making the panels ..')

                ############################################################
                ######################### BIG PLOT #########################
                ############################################################

                fig = plt.figure(figsize=[20,10])
                gs = GridSpec(4, 3, figure=fig)
                
                if any(df_psp):
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.pcolormesh(df_psp.columns,
                                    df_psp.index,
                                    df_psp.values,
                                    norm=psp_norm,
                                    cmap='RdYlBu_r')
                    ax1.set_ylabel('Frequency (MHz)')
                    ax1.set_title(f'PSP/FIELDS: {df_psp.index[0]*1e3:.2f} kHz $-$ {df_psp.index[-1]:.2f} MHz')
                    ax1.xaxis_date()
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax1.set_yscale('log')
                    ax1.yaxis.set_major_formatter(custom_formatter)
                    print('PSP plot is ready.')
                # else:
                #     print('PSP data is missing!')
                
                if len(time_ste)>1 and len(freq_ste)>1 and len(data_ste_A)>1:
                    ax2 = fig.add_subplot(gs[1, 0])
                    ax2.pcolormesh(time_ste,
                                    freq_ste/1e3,
                                    data_ste_A,
                                    norm=ste_norm,
                                    cmap='RdYlBu_r')
                    ax2.grid(False)
                    ax2.set_ylabel('Frequency (MHz)')
                    ax2.set_title(f'STEREO/SWAVES: {freq_ste[0]:.2f} kHz $-$ {freq_ste[-1]/1e3:.2f} MHz')
                    ax2.xaxis_date()
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax2.set_yscale('log')
                    ax2.yaxis.set_major_formatter(custom_formatter)
                    print('STEREO plot is ready.')
                # else:
                #     print('STEREO data is missing!')
                
                if len(wind_time)>1 and len(wind_freq)>1 and len(wind_data)>1:
                    ax3 = fig.add_subplot(gs[2, 0])
                    ax3.pcolormesh(wind_time,
                                    wind_freq/1e3,
                                    wind_data,
                                    norm=wind_norm,
                                    cmap='RdYlBu_r')
                    ax3.set_ylabel('Frequency (MHz)')
                    ax3.set_title(f'Wind/WAVES: {wind_freq[0]:.2f} kHz $-$ {wind_freq[-1]/1e3:.2f} MHz')
                    ax3.xaxis_date()
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax3.set_yscale('log')
                    ax3.yaxis.set_major_formatter(custom_formatter)
                    print('Wind plot is ready.')
                # else:
                #     print('Wind data is missing!')
                
                if goes:
                    if isinstance(goes, list):
                        goes = goes[0] # goes[0]: 1-s; goes[1]: 1-m resolution.
                    ax4 = fig.add_subplot(gs[3, 0])
                    goes.plot(axes=ax4)
                    ax4.set_xlabel('Time (UT)')
                    ax4.set_title(f'{goes.observatory}/{goes.source.upper()}')
                    print('GOES plot is ready.')
                # else:
                #     print('GOES data is missing!')

                # Load the PNG image
                if os.path.exists(f'./locations/{t.date()}.png'):
                    img = mpimg.imread(f'./locations/{t.date()}.png')
                    ax5 = fig.add_subplot(gs[:, 1])
                    ax5.imshow(img)
                    ax5.axis('off')
                    print('Locations map is ready.')
                # else:
                #     print('Locations map is missing!')
                
                if m:
                    ax6 = fig.add_subplot(gs[:, 2], projection=m)
                    m.plot(axes=ax6, clip_interval=(1, 99.99)*u.percent)
                    ax6.grid(False)
                    ax6.set_xlabel('Helioprojective Lonitude (Solar-X)')
                    ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')
                    print('AIA map is ready.')
                # else:
                #     print('AIA map is missing!')

                print(f'Exporting the figure at: ./plots/summary_plot_{t.date()}.png')
                fig.tight_layout()
                fig.savefig(f'./plots/summary_plot_{t.date()}.png', format='png', dpi=100, bbox_inches='tight')
                print(f'Summary plot for {t} has been exported successfully.\n')
                plt.close()











# Define custom tick formatter for y-axis
def custom_formatter(x, pos):
    return f'{x:.2f}'



def minmax_normalize(arr=None):
    '''
    Min-Max normalization.
    '''
    min_val = np.min(arr)
    max_val = np.max(arr)
    arr_norm = (arr - min_val) / (max_val - min_val)
    return arr_norm



def split_date(dt=None):
    '''
    Split a datetime object into year, month, and day, and return them as strings.
        dt: Datetime/Timestamp object.
    '''
    YEAR = dt.date().year
    MONTH = dt.date().month
    DAY = dt.date().day
    
    if DAY < 10:
        DAY = f'0{DAY}'
    else:
        DAY = f'{DAY}'
    
    if MONTH < 10:
        MONTH = f'0{MONTH}'
    else:
        MONTH = f'{MONTH}'
    
    return YEAR, MONTH, DAY




def fetch_locations(dt=None):
    '''
    Download the locations of instruments.
        dt: Datetime/Timestamp object.
    '''
    if dt.year == 2019:
        body_list = ['Earth', 'Mars', 'STEREO-A', 'PSP']
    else:
        body_list = ['Earth', 'Mars', 'STEREO-A', 'PSP', 'Solo']
    
    vsw_list = [] # [400, 400, 400, 400, 400]        # leave empty to obtain measurements
    date_solmach = str(dt)

    coord_sys = 'Stonyhurst'                         # 'Carrington' (default) or 'Stonyhurst'
    reference_long = None                            # longitude of reference (None to omit)
    reference_lat = None                             # latitude of reference (None to omit)
    # reference_vsw = 400                              # define solar wind speed at reference
    long_offset = 270                                # longitudinal offset for polar plot; defines where Earth's longitude is (by default 270, i.e., at "6 o'clock")
    plot_spirals = True                              # plot Parker spirals for each body
    plot_sun_body_line = False                       # plot straight line between Sun and body
    return_plot_object = False                       # figure and axis object of matplotib are returned, allowing further adjustments to the figure
    transparent = True                               # make output figure background transparent
    numbered_markers = True                          # plot each body with a numbered marker

    # initialize
    sm = SolarMACH(date_solmach, body_list, vsw_list, reference_long, reference_lat, coord_sys)

    # make plot
    sm.plot(plot_spirals=plot_spirals,
            plot_sun_body_line=plot_sun_body_line,
            # reference_vsw=reference_vsw,
            transparent=transparent,
            numbered_markers=numbered_markers,
            long_offset=long_offset,
            return_plot_object=return_plot_object,
            figsize=[17,10],
            outfile=f'./locations/{dt.date()}.png')




def fetch_xrs(dt=None, year=None, month=None, day=None):
    '''
    Download GOES/XRS data.
        dt: Datetime/Timestamp object.
        Inputs: year, month, day.
    '''
    if year == 2019:
        # Check if the file doesn't exist
        file_path = f'./XRS/sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'

        if not os.path.exists(file_path):
            # get the data file
            URL = f'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes17/l2/data/xrsf-l2-flx1s_science/{year}/{month}/sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'
            response = requests.get(URL)
            open(os.path.join('./XRS', f'sci_xrsf-l2-flx1s_g17_d{year}{month}{day}_v2-2-0.nc'), 'wb').write(response.content)

        goes = ts.TimeSeries(file_path)
    else:
        file_path = f'./XRS/sci_xrsf-l2-avg1m_g16_d{year}{month}{day}_v2-2-0.nc'
        
        if not os.path.exists(file_path):
            goes_result = Fido.search(a.Time(f'{dt.date()}', f'{dt.date()}'), a.Instrument('XRS'), a.goes.SatelliteNumber(16))
            goes_file = Fido.fetch(goes_result, path='./{instrument}/{file}')
            goes = ts.TimeSeries(goes_file)
        else:
            goes = ts.TimeSeries(file_path)
    return goes





def fetch_aia(dt=None, year=None, month=None, day=None):
    '''
    Download SDO/AIA data.
        dt: Datetime/Timestamp object.
        Inputs: year, month, day.
    '''
    file_path = f'./AIA/aia_lev1_193a_{year}_{month}_{day}t00_00_04_84z_image_lev1.fits'
    
    if not os.path.exists(file_path):
        result = Fido.search(a.Time(f'{dt}', f'{dt + datetime.timedelta(minutes=1)}'),
                            a.Instrument('AIA'),
                            a.Wavelength(193*u.angstrom),
                            a.Sample(1*u.min))
        aia_file = Fido.fetch(result[0], path='./{instrument}/{file}')
        m = sunpy.map.Map(aia_file)
    else:
        m = sunpy.map.Map(file_path)
    return m



def fetch_waves(dt=None, year=None, month=None, day=None):
    '''
    Download Wind/WAVES data.
        dt: Datetime/Timestamp object.
        Inputs: year, month, day.
    '''
    file_path = f'./wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf'
    
    if not os.path.exists(file_path):
        print('Wind/WAVES data file not exist!')
        print('Proceed to fetch data from the server ..')
        
        # get the data file
        URL = f'https://spdf.gsfc.nasa.gov/pub/data/wind/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf'
        response = requests.get(URL)
        open(os.path.join(f'./wind_data/waves/wav_h1/{year}', f'wi_h1_wav_{year}{month}{day}_v01.cdf'), 'wb').write(response.content)
        
        print('Wind/WAVES data is downloaded locally.')
        wind = pycdf.CDF(f'./wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf')
    else:
        print('Wind/WAVES data exists locally.')
        wind = pycdf.CDF(f'./wind_data/waves/wav_h1/{year}/wi_h1_wav_{year}{month}{day}_v01.cdf')
        
    wind_time = [mdates.date2num(tm) for tm in pd.to_datetime(wind.get('Epoch'))]
    RAD1_freq = np.array(wind.get('Frequency_RAD1'))
    RAD2_freq = np.array(wind.get('Frequency_RAD2'))
    TNR_freq = np.array(wind.get('Frequency_TNR'))
    RAD1_int = np.array(wind.get('E_VOLTAGE_RAD1'))
    RAD2_int = np.array(wind.get('E_VOLTAGE_RAD2'))
    TNR_int = np.array(wind.get('E_VOLTAGE_TNR'))

    wind_freq = np.concatenate((TNR_freq, RAD1_freq, RAD2_freq))
    wind_data = np.concatenate((TNR_int, RAD1_int, RAD2_int), axis=1)

    # rescale the data
    wind_scaled = minmax_normalize(arr=wind_data.T)

    # Apply Gaussian smoothing
    smoothed_wind = gaussian_filter(wind_scaled, sigma=1)
        
    wind_norm = ImageNormalize(smoothed_wind, interval=PercentileInterval(97), clip=True)
    
    return wind_time, wind_freq, smoothed_wind, wind_norm



def fetch_STAswaves(year=None, month=None, day=None):
    '''
    Download STEREO/SWAVES data.
        Inputs: year, month, day.
    '''
    file_path = f'./stereo_data/stereo_level2_swaves_{year}{month}{day}_v02.cdf'

    if not os.path.exists(file_path):
        # get the data file
        URL = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf/{year}/stereo_level2_swaves_{year}{month}{day}_v02.cdf'
        response = requests.get(URL)
        open(os.path.join('./stereo_data', f'stereo_level2_swaves_{year}{month}{day}_v02.cdf'), 'wb').write(response.content)
    
    # read the data values
    cdf_stereo = pycdf.CDF(f'./stereo_data/stereo_level2_swaves_{year}{month}{day}_v02.cdf')
    time_ste = np.array(cdf_stereo.get('Epoch'))
    freq_ste = np.array(cdf_stereo.get('frequency'))
    data_ste_A = np.array(cdf_stereo.get('avg_intens_ahead'))

    # calculate the mean value in each row (freq channel)
    # and subtract it from each corresponding row
    df_ste_A = pd.DataFrame(data_ste_A.T)
    df_ste_mean = df_ste_A.mean(axis=1)
    data_ste_A = df_ste_A.sub(df_ste_mean, axis=0)

    # rescale the data
    data_ste_A_norm = minmax_normalize(arr=data_ste_A)

    # Apply Gaussian smoothing
    smoothed_ste_A = gaussian_filter(data_ste_A_norm, sigma=1)

    ste_norm = ImageNormalize(smoothed_ste_A, interval=PercentileInterval(97), clip=True)

    cdf_stereo.close()

    return time_ste, freq_ste, smoothed_ste_A, ste_norm




def fetch_PSPfields(year=None, month=None, day=None, data_version=2):
    '''
    Download PSP/FIELDS data.
        Inputs: year, month, day.
    '''
    hfr_file_path = f'psp_fld_l2_rfs_hfr_{year}{month}{day}_v0{data_version}.cdf'
    lfr_file_path = f'psp_fld_l2_rfs_lfr_{year}{month}{day}_v0{data_version}.cdf'
    
    HFR_URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_hfr/{year}/{month}/{hfr_file_path}'
    LFR_URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_lfr/{year}/{month}/{lfr_file_path}'

    hfr_response = requests.head(HFR_URL)
    lfr_response = requests.head(LFR_URL)

    if hfr_response.status_code == 200 and lfr_response.status_code == 200:
        # Files are available, proceed with download
        if not os.path.exists(hfr_file_path):
            response = requests.get(HFR_URL)
            open(os.path.join('./psp_data', hfr_file_path), 'wb').write(response.content)
        
        if not os.path.exists(lfr_file_path):
            response = requests.get(LFR_URL)
            open(os.path.join('./psp_data', lfr_file_path), 'wb').write(response.content)
        
        # load the PSP data
        cdf_psp_hfr = pycdf.CDF(os.path.join('./psp_data', hfr_file_path))
        cdf_psp_lfr = pycdf.CDF(os.path.join('./psp_data', lfr_file_path))

        # the min power scaled power spectral density (PSD) of 1e-16 is used as a threshold
        # convert pixels values to dB
        arr_lfr = np.array(cdf_psp_lfr.get('psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2'))
        Lp_lfr = 10*np.log10(arr_lfr/10**-16) # z-axis
        tm_lfr = np.array(cdf_psp_lfr.get('epoch_lfr')) # x-axis
        freq_lfr = np.array(cdf_psp_lfr.get('frequency_lfr_auto_averages_ch0_V1V2'))/10**6 # y-axis

        # convert pixels values to dB
        arr_hfr = np.array(cdf_psp_hfr.get('psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2'))
        Lp_hfr = 10*np.log10(arr_hfr/10**-16) # z-axis
        tm_hfr = np.array(cdf_psp_hfr.get('epoch_hfr')) # x-axis
        freq_hfr = np.array(cdf_psp_hfr.get('frequency_hfr_auto_averages_ch0_V1V2'))/10**6 # y-axis

        # clean the dyspec by subtracting the Mean intensity from each freq channel
        df_psp_hfr = pd.DataFrame(Lp_hfr.T)
        df_psp_lfr = pd.DataFrame(Lp_lfr.T)

        # concat the two arrays of both bands
        df_lfr = pd.DataFrame(df_psp_lfr)
        df_lfr.insert(loc=0, column='frequency', value=freq_lfr[0])
        df_lfr.set_index(['frequency'], inplace=True)

        df_hfr = pd.DataFrame(df_psp_hfr)
        df_hfr.insert(loc=0, column='frequency', value=freq_hfr[0])
        df_hfr.set_index(['frequency'], inplace=True)

        # drop the overlapped rows, take only the first row of the duplicated group
        df_psp = pd.concat([df_lfr, df_hfr])
        df_psp = df_psp.sort_index(axis=0)
        df_psp = df_psp[~df_psp.index.duplicated(keep='first')]
        df_psp.columns = max([tm_lfr, tm_hfr], key=len)

        # calculate the mean value in each row (freq channel)
        # and subtract it from each corresponding row
        df_psp_mean = df_psp.mean(axis=1)
        df_psp_submean = df_psp.sub(df_psp_mean, axis=0)

        # rescale the data
        psp_submean_norm = minmax_normalize(arr=df_psp_submean)

        # Apply Gaussian smoothing
        psp_submean_norm_smooth = gaussian_filter(psp_submean_norm, sigma=1)
        df_smoothed_psp_norm = pd.DataFrame(psp_submean_norm_smooth, index=df_psp.index, columns=df_psp.columns)

        psp_norm = ImageNormalize(df_smoothed_psp_norm.values, interval=PercentileInterval(97), clip=True)
        
        cdf_psp_hfr.close()
        cdf_psp_lfr.close()

        return df_smoothed_psp_norm, psp_norm
    
    else:
        # Files are not available
        print('PSP/FIELD data is not available on the website!\n')
        return None






def main():
    # # Check if there are enough arguments
    # if len(sys.argv) != 2:
    #     print('Usage: python3 script.py start_date end_date')
    #     return

    # # Extract arguments
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]

    parser = argparse.ArgumentParser(description='Process events')
    parser.add_argument('--list', action='store_true', help='Process a list of events')
    parser.add_argument('--individual', action='store_true', help='Process an individual event')
    args = parser.parse_args()

    if args.list and args.individual:
        print('Error: Both --list and --individual options cannot be used together.')
        return
    
    elif args.individual:
        event = input('Enter the date (YYYY-mm-DD): ')
        # Record the start time
        start_time = time.time()
        process_events(event=event)
    
    elif args.list:
        start_date = input('Enter start date (YYYY-mm-DD): ')
        end_date = input('Enter end date (YYYY-mm-DD): ')
        # Record the start time
        start_time = time.time()
        process_events(start_date=start_date, end_date=end_date, individual=False)
    
    else:
        print('Error: Please specify either --list or --individual option.')

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print(f'Execution time: {execution_time} seconds')




if __name__ == '__main__':
    main()
