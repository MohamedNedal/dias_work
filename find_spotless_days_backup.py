#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CDF_LIB'] = '/home/peijin/cdf/cdf38_0-dist/lib'

import sys
import requests
import argparse
import datetime
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.dates as mdates

from astropy import units as u
from astropy.visualization import ImageNormalize, SqrtStretch, PercentileInterval

import sunpy.map
from sunpy import timeseries as ts
from sunpy.net import Fido, attrs as a

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





def process_events(event=None, start_date=None, end_date=None, individual=True):
    if individual:
        print(f'Processing individual event: {event} ..')

        # make sure that the date is a pd.Timestamp object
        if not isinstance(event, pd.Timestamp):
            YEAR, MONTH, DAY = split_date(dt=pd.Timestamp(t))
        
        try:
            ######################### S/C locations #########################
            fig_solmach, ax_solmach = fetch_locations(dt=t)

            ######################### XRS #########################            
            goes = fetch_xrs(dt=t)
            
            ######################### AIA #########################
            m = fetch_aia(dt=t)

            ######################### WIND #########################            
            TNR_times, wind_freq, wind_data, wind_norm = fetch_waves(dt=t)

            ######################### STA ##########################
            time_ste, freq_ste, data_ste_A, sta_norm = fetch_STAswaves(year=YEAR, month=MONTH, day=DAY)

            ######################### PSP #########################
            tm_hfr, df_psp, psp_norm = fetch_PSPfields(year=YEAR, month=MONTH, day=DAY)
        
        except:
            print('Error: Missing data files.')
        
        ######################### BIG PLOT #########################            
        fig = plt.figure(figsize=[20,10])
        gs = GridSpec(4, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.pcolormesh(tm_hfr, df_psp.index, df_psp.values, norm=psp_norm, cmap='RdYlBu_r')
        ax1.set_ylabel('Frequency (MHz)')
        ax1.set_title(f'PSP/FIELDS: {df_psp.index[0]*1e3:.2f} kHz $-$ {df_psp.index[-1]:.2f} MHz')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(data_ste_A, aspect='auto', origin='lower', norm=sta_norm,
                    extent=[mdates.date2num(time_ste[0]), mdates.date2num(time_ste[-1]),
                            freq_ste[0]/1e3, freq_ste[-1]/1e3], cmap='RdYlBu_r')
        ax2.grid(False)
        ax2.set_ylabel('Frequency (MHz)')
        ax2.set_title(f'STEREO/SWAVES: {freq_ste[0]:.2f} kHz $-$ {dfreq_ste[-1]/1e3:.2f} MHz')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.imshow(wind_data, aspect='auto', origin='lower', norm=wind_norm,
                    extent=[mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(TNR_times)[0], '%Y-%m-%d %H:%M:%S.%f')), 
                            mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(TNR_times)[-1], '%Y-%m-%d %H:%M:%S.%f')), 
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
        
        ax5 = fig.add_subplot(gs[:, 1])
        ax5.imshow(fig_solmach.canvas.renderer.buffer_rgba())
        ax5.axes.xaxis.set_visible(False)
        ax5.axes.yaxis.set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)

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

        # start_date = arg1 # ex. '2020-01-01'   # arg1
        # end_date   = arg2 # ex. '2020-12-31'   # arg2

        sub_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]       # Filter the DataFrame to include only data within a range
        zero_sunspot_days = sub_df[sub_df['total_sunspot_num'] == 0]             # Filter the DataFrame to find the days where the sunspot number was zero
        # print(f'Number of spotless days between {start_date} and {end_date} is {len(zero_sunspot_days)}\n')
        
        print(f'Processing list of {len(zero_sunspot_days)} spotless days between {start_date} and {end_date} ..')

        data_dict = {
            'date': [],
            'wind_time': [],
            'wind_freq': [],
            'wind_spect': [],
            'wind_spectNorm': [],
            'stereo_time': [],
            'stereo_freq': [],
            'stereo_spect': [],
            'stereo_spectNorm': [],
            'psp_time': [],
            'psp_freq': [],
            'psp_spect': [],
            'psp_spectNorm': [],
            'locations': [],
            'aia_map': [],
            'goes_xrs': []
        }

        for t in zero_sunspot_days['date'].iloc[30:32]: #[30:-5]:
            print(t)
            
            YEAR, MONTH, DAY = split_date(dt=t)
            
            try:
                #################################################################
                ######################### S/C locations #########################
                #################################################################

                fig_solmach, ax_solmach = fetch_locations(dt=t)

                #######################################################
                ######################### XRS #########################
                #######################################################
                
                goes = fetch_xrs(dt=t)
                
                #######################################################
                ######################### AIA #########################
                #######################################################

                m = fetch_aia(dt=t)

                ########################################################
                ######################### WIND #########################
                ########################################################
                
                TNR_times, wind_freq, wind_data, wind_norm = fetch_waves(dt=t)

                ########################################################
                ######################### STA ##########################
                ########################################################

                time_ste, freq_ste, data_ste_A, sta_norm = fetch_STAswaves(year=YEAR, month=MONTH, day=DAY)

                #######################################################
                ######################### PSP #########################
                #######################################################

                tm_hfr, df_psp, psp_norm = fetch_PSPfields(year=YEAR, month=MONTH, day=DAY)

                # store the data in a dict object
                data_dict['date'].append(t)

                data_dict['wind_time'].append(TNR_times)
                data_dict['wind_freq'].append(wind_freq)
                data_dict['wind_spect'].append(wind_data)
                data_dict['wind_spectNorm'].append(wind_norm)

                data_dict['stereo_time'].append(time_ste)
                data_dict['stereo_freq'].append(freq_ste)
                data_dict['stereo_spect'].append(data_ste_A)
                data_dict['stereo_spectNorm'].append(sta_norm)

                data_dict['psp_time'].append(tm_hfr)
                data_dict['psp_freq'].append(df_psp.index)
                data_dict['psp_spect'].append(df_psp.values)
                data_dict['psp_spectNorm'].append(psp_norm)

                data_dict['locations'].append(fig_solmach)
                data_dict['aia_map'].append(m)
                data_dict['goes_xrs'].append(goes[0]) # goes[0]: 1-s; goes[1]: 1-m resolution.

                print(f'Day {t} has been analyzed.')
            
            except:
                pass
        
        ############################################################
        ######################### BIG PLOT #########################
        ############################################################

        for i in range(len(data_dict['date'])):
            print(str(data_dict['date'][i]))
            
            fig = plt.figure(figsize=[20,10])
            gs = GridSpec(4, 3, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.pcolormesh(data_dict['psp_time'][i], data_dict['psp_freq'][i], data_dict['psp_spect'][i],
                        norm=data_dict['psp_spectNorm'][i], cmap='RdYlBu_r')
            ax1.set_ylabel('Frequency (MHz)')
            ax1.set_title(f"PSP/FIELDS: {data_dict['psp_freq'][i][0]*1e3:.2f} kHz $-$ {data_dict['psp_freq'][i][-1]:.2f} MHz")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.imshow(data_dict['stereo_spect'][i], aspect='auto', origin='lower', norm=data_dict['stereo_spectNorm'][i],
                        extent=[mdates.date2num(data_dict['stereo_time'][i][0]), mdates.date2num(data_dict['stereo_time'][i][-1]),
                                data_dict['stereo_freq'][i][0]/1e3, data_dict['stereo_freq'][i][-1]/1e3], cmap='RdYlBu_r')
            ax2.grid(False)
            ax2.set_ylabel('Frequency (MHz)')
            ax2.set_title(f"STEREO/SWAVES: {data_dict['stereo_freq'][i][0]:.2f} kHz $-$ {data_dict['stereo_freq'][i][-1]/1e3:.2f} MHz")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            ax3 = fig.add_subplot(gs[2, 0])
            ax3.imshow(data_dict['wind_spect'][i].T, aspect='auto', origin='lower', norm=data_dict['wind_spectNorm'][i],
                        extent=[mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(data_dict['wind_time'][i])[0], '%Y-%m-%d %H:%M:%S.%f')), 
                                mdates.date2num(datetime.datetime.strptime(pyspedas.time_string(data_dict['wind_time'][i])[-1], '%Y-%m-%d %H:%M:%S.%f')), 
                                data_dict['wind_freq'][i][0]/1e3, data_dict['wind_freq'][i][-1]/1e3], cmap='RdYlBu_r')
            ax3.grid(False)
            ax3.xaxis_date()
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax3.set_ylabel('Frequency (MHz)')
            ax3.set_title(f"Wind/WAVES: {data_dict['wind_freq'][i][0]:.2f} kHz $-$ {data_dict['wind_freq'][i][-1]/1e3:.2f} MHz")
            
            ax4 = fig.add_subplot(gs[3, 0])
            data_dict['goes_xrs'][i].plot(axes=ax4)
            ax4.set_xlabel('Time (UT)')
            ax4.set_title(f"{data_dict['goes_xrs'][i].observatory}/{data_dict['goes_xrs'][i].source.upper()}")
            
            ax5 = fig.add_subplot(gs[:, 1])
            ax5.imshow(data_dict['locations'][i].canvas.renderer.buffer_rgba())
            ax5.axes.xaxis.set_visible(False)
            ax5.axes.yaxis.set_visible(False)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.spines['left'].set_visible(False)

            ax6 = fig.add_subplot(gs[:, 2], projection=data_dict['aia_map'][i])
            data_dict['aia_map'][i].plot(axes=ax6, clip_interval=(1, 99.99)*u.percent)
            ax6.grid(False)
            ax6.set_xlabel('Helioprojective Lonitude (Solar-X)')
            ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')

            fig.tight_layout()
            fig.savefig(f"./plots/summary_plot_{data_dict['date'][i].date()}.png", format='png', bbox_inches='tight')
            plt.close()

            print(f"Summary plot for {data_dict['date'][i]} has been exported.\n")






def split_date(dt=None):
    '''
    Split a datetime object into year, month, and day, and return them as strings.
        dt: Datetime/Timestamp object.
    '''
    YEAR = dt.date().year
    MONTH = dt.date().month
    DAY = dt.date().day
    
    if DAY < 9:
        DAY = f'0{DAY}'
    else:
        DAY = f'{DAY}'
    
    if MONTH < 9:
        MONTH = f'0{MONTH}'
    else:
        MONTH = f'{MONTH}'
    
    return YEAR, MONTH, DAY




def fetch_locations(dt=None):
    '''
    Download the locations of instruments.
        dt: Datetime/Timestamp object.
    '''
    body_list = ['Earth', 'Mars', 'STEREO-A', 'PSP', 'Solo']
    vsw_list = [] # [400, 400, 400, 400, 400]        # leave empty to obtain measurements
    date_solmach = str(dt)

    coord_sys = 'Stonyhurst'                         # 'Carrington' (default) or 'Stonyhurst'
    reference_long = None                            # longitude of reference (None to omit)
    reference_lat = None                             # latitude of reference (None to omit)
    reference_vsw = 400                              # define solar wind speed at reference
    long_offset = 270                                # longitudinal offset for polar plot; defines where Earth's longitude is (by default 270, i.e., at "6 o'clock")
    plot_spirals = True                              # plot Parker spirals for each body
    plot_sun_body_line = False                       # plot straight line between Sun and body
    return_plot_object = True                        # figure and axis object of matplotib are returned, allowing further adjustments to the figure
    transparent = True                               # make output figure background transparent
    numbered_markers = True                          # plot each body with a numbered marker

    # initialize
    sm = SolarMACH(date_solmach, body_list, vsw_list, reference_long, reference_lat, coord_sys)

    # make plot
    fig_solmach, ax_solmach = sm.plot(plot_spirals=plot_spirals,
                                    plot_sun_body_line=plot_sun_body_line,
                                    reference_vsw=reference_vsw,
                                    transparent=transparent,
                                    numbered_markers=numbered_markers,
                                    long_offset=long_offset,
                                    return_plot_object=return_plot_object,
                                    figsize=[17,10])
    return fig_solmach, ax_solmach




def fetch_xrs(dt=None):
    '''
    Download GOES/XRS data.
        dt: Datetime/Timestamp object.
    '''
    goes_result = Fido.search(a.Time(f'{dt.date()}', f'{dt.date()}'), a.Instrument('XRS'), a.goes.SatelliteNumber(16))
    goes_file = Fido.fetch(goes_result, path='./{instrument}/{file}')
    goes = ts.TimeSeries(goes_file)
    return goes



def fetch_aia(dt=None):
    '''
    Download SDO/AIA data.
        dt: Datetime/Timestamp object.
    '''
    result = Fido.search(a.Time(f'{dt}', f'{dt + datetime.timedelta(minutes=1)}'),
                        a.Instrument('AIA'),
                        a.Wavelength(193*u.angstrom),
                        a.Sample(1*u.min))
    aia_file = Fido.fetch(result[0], path='./{instrument}/{file}')
    m = sunpy.map.Map(aia_file)
    return m



def fetch_waves(dt=None):
    '''
    Download Wind/WAVES data.
        dt: Datetime/Timestamp object.
    '''
    time_range = [f'{dt.date()}', f'{dt.date() + datetime.timedelta(days=1)}']
    wind_vars = pyspedas.wind.waves(trange=time_range)

    RAD2_times, RAD2_int, RAD2_freq = get_data('E_VOLTAGE_RAD2')
    RAD1_times, RAD1_int, RAD1_freq = get_data('E_VOLTAGE_RAD1')
    TNR_times, TNR_int, TNR_freq = get_data('E_VOLTAGE_TNR')

    wind_freq = np.concatenate((TNR_freq, RAD1_freq, RAD2_freq))
    wind_data = np.concatenate((TNR_int, RAD1_int, RAD2_int), axis=1)

    wind_norm = ImageNormalize(wind_data, interval=PercentileInterval(90))
    
    return TNR_times, wind_freq, wind_data, wind_norm



def fetch_STAswaves(year=None, month=None, day=None):
    '''
    Download STEREO/SWAVES data.
        Inputs: year, month, day.
    '''
    # get the data file
    URL = f'https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf/{year}/stereo_level2_swaves_{year}{month}{day}_v02.cdf'
    response = requests.get(URL)
    open(os.path.join('./stereo_data', f'stereo_level2_swaves_{year}{month}{day}_v02.cdf'), 'wb').write(response.content)

    # read the data values
    cdf_stereo = pycdf.CDF(f'./stereo_data/stereo_level2_swaves_{year}{month}{day}_v02.cdf')
    time_ste = np.array(cdf_stereo.get('Epoch'))
    freq_ste = np.array(cdf_stereo.get('frequency'))
    data_ste_A = np.array(cdf_stereo.get('avg_intens_ahead'))

    # subtract the Mean value
    # calculate the mean intensity in each row (freq channel)
    df_ste_A = pd.DataFrame(data_ste_A.T)
    df_ste_mean = df_ste_A.mean(axis=1)

    # subtract that mean value from each corresponding row
    data_ste_A = df_ste_A.sub(df_ste_mean, axis=0)

    sta_norm = ImageNormalize(data_ste_A, interval=PercentileInterval(90))

    return time_ste, freq_ste, data_ste_A, sta_norm




def fetch_PSPfields(year=None, month=None, day=None):
    '''
    Download PSP/FIELDS data.
        Inputs: year, month, day.
    '''
    # HFR
    URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_hfr/{year}/{month}/psp_fld_l2_rfs_hfr_{year}{month}{day}_v02.cdf'
    response = requests.get(URL)
    open(os.path.join('./psp_data', f'psp_fld_l2_rfs_hfr_{year}{month}{day}_v02.cdf'), 'wb').write(response.content)

    # LFR
    URL = f'http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/rfs_lfr/{year}/{month}/psp_fld_l2_rfs_lfr_{year}{month}{day}_v02.cdf'
    response = requests.get(URL)
    open(os.path.join('./psp_data', f'psp_fld_l2_rfs_lfr_{year}{month}{day}_v02.cdf'), 'wb').write(response.content)

    # load the PSP data
    cdf_psp_hfr = pycdf.CDF(os.path.join('./psp_data', f'psp_fld_l2_rfs_hfr_{year}{month}{day}_v02.cdf'))
    cdf_psp_lfr = pycdf.CDF(os.path.join('./psp_data', f'psp_fld_l2_rfs_lfr_{year}{month}{day}_v02.cdf'))

    tmin_lfr = cdf_psp_lfr['epoch_lfr'].meta['SCALEMIN']
    tmax_lfr = cdf_psp_lfr['epoch_lfr'].meta['SCALEMAX']

    tmin_hfr = cdf_psp_hfr['epoch_hfr'].meta['SCALEMIN']
    tmax_hfr = cdf_psp_hfr['epoch_hfr'].meta['SCALEMAX']

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

    df_psp_hfr_mean = df_psp_hfr.mean(axis=1)
    df_psp_lfr_mean = df_psp_lfr.mean(axis=1)

    # subtract that mean value from each corresponding row
    df_psp_hfr = df_psp_hfr.sub(df_psp_hfr_mean, axis=0)
    df_psp_lfr = df_psp_lfr.sub(df_psp_hfr_mean, axis=0)

    # concat the two arrays of both bands
    df_lfr = pd.DataFrame(df_psp_lfr)
    df_lfr.insert(loc=0, column='frequency', value=freq_lfr[1])
    df_lfr.set_index(['frequency'], inplace=True)

    df_hfr = pd.DataFrame(df_psp_hfr)
    df_hfr.insert(loc=0, column='frequency', value=freq_hfr[1])
    df_hfr.set_index(['frequency'], inplace=True)

    # drop the overlapped rows, take only the first row of the duplicated group
    df_psp = pd.concat([df_lfr, df_hfr])
    df_psp = df_psp.sort_index(axis=0)
    df_psp = df_psp[~df_psp.index.duplicated(keep='first')]

    psp_norm = ImageNormalize(df_psp.values, interval=PercentileInterval(90))

    return tm_hfr, df_psp, psp_norm
    








def main():
    # # Check if there are enough arguments
    # if len(sys.argv) != 2:
    #     print('Usage: python3 script.py start_date end_date')
    #     return

    # # Extract arguments
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]

    parser = argparse.ArgumentParser(description="Process events")
    parser.add_argument("--list", action="store_true", help="Process a list of events")
    parser.add_argument("--individual", action="store_true", help="Process an individual event")
    args = parser.parse_args()

    if args.list and args.individual:
        print('Error: Both --list and --individual options cannot be used together.')
        return
    
    elif args.individual:
        event = input('Enter the date (YYYY-mm-DD): ')
        process_events(event=event)
    
    elif args.list:
        start_date = input('Enter start date (YYYY-mm-DD): ')
        end_date = input('Enter start date (YYYY-mm-DD): ')
        process_events(start_date=start_date, end_date=end_date, individual=False)
        
    else:
        print('Error: Please specify either --list or --individual option.')




if __name__ == "__main__":
    main()


