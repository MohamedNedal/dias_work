#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
from sys import path as sys_path
import os.path
import platform
import datetime as dt
from aiapy.calibrate.prep import correct_degradation
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as fits
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
import scipy.io as io
sys_path.append('/home/mnedal/repos/demreg/python')
from dn2dem_pos import dn2dem_pos
script_path = os.path.abspath('./scripts')
if script_path not in sys_path:
    sys_path.append(script_path)
from general_routines import closest
from aiapy.calibrate import register, update_pointing, estimate_error
import aiapy.psf
import asdf
from bisect import bisect_left, bisect_right
from sunpy.time import parse_time



# Define constants and make the data directories
# data_disk = '/home/mnedal/data/AIA/'
# os.makedirs(data_disk, exist_ok='True')

# Function with event information
# start_time = '2024/05/14 17:00:00'
# end_time   = '2024/05/14 19:00:00'

# ref_file_date = dt.datetime.strftime(dt.datetime.strptime(ref_time,'%Y/%m/%d %H:%M:%S'), '%Y/%m/%d')
# img_file_date = dt.datetime.strftime(dt.datetime.strptime(ref_time,'%Y/%m/%d %H:%M:%S'), '%Y/%m/%d')

# Define and make the output directories
# output_dir = f'{data_disk}/DEM/{img_file_date}/'

# os.makedirs(output_dir, exist_ok='True')
# passband = [94, 131, 171, 193, 211, 335]


def extract_datetime(filename, channel):
    """
    Function to extract the datetime from a filename.
    """
    # Split the filename and extract the date and time parts
    date_time_part = filename.split('/')[-1]                            # Extracts '2024_05_14T18_49_59.12'
    date_part = date_time_part.split('T')[0].split(f'{channel}A_')[-1]  # Extracts '2024_05_14'
    time_part = date_time_part.split('T')[1].split('Z')[0]              # Extracts '18_49_59.12'
    
    # Reformat date and time to standard datetime format
    date_str = date_part.replace('_', '-')  # '2024-05-14'
    time_str = time_part.replace('_', ':')  # '18:49:59.12'
    
    # Combine date and time and convert to datetime object
    return dt.datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M:%S.%f')


def find_closest_filename(filenames, channel, target_datetime):
    """
    Function to find the index of the filename with the closest datetime to a given target.
    """
    closest_index = None
    min_time_diff = None
    
    for i, filename in enumerate(filenames):
        file_datetime = extract_datetime(filename, channel)
        
        # Calculate the absolute time difference
        time_diff = abs(file_datetime - target_datetime)
        
        # Update the closest file if this one is closer
        if min_time_diff is None or time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_index = i
    
    return closest_index


def calculate_dem(map_array, err_array):
    """
    Function to calculate DEM.
    """
    nx, ny      = map_array[0].data.shape
    nf          = len(map_array)
    image_array = np.zeros((nx, ny, nf))
    for img in range(0, nf):
        image_array[:,:,img] = map_array[img].data
    
    if platform.system() == 'Linux':
        trin = io.readsav('/home/mnedal/data/aia_tresp_en.dat')
        
    tresp_logt = np.array(trin['logt'])
    nt         = len(tresp_logt)
    nf         = len(trin['tr'][:])
    trmatrix   = np.zeros((nt,nf))
    for i in range(0,nf):
        trmatrix[:,i] = trin['tr'][i]    
    
    t_space  = 0.1
    t_min    = 5.6
    t_max    = 7.4
    logtemps = np.linspace(t_min, t_max, num=int((t_max-t_min)/t_space)+1)
    temps    = 10**logtemps
    mlogt    = ([np.mean([(np.log10(temps[i])), np.log10((temps[i+1]))]) for i in np.arange(0, len(temps)-1)])
    dem, edem, elogt, chisq, dn_reg = dn2dem_pos(image_array, err_array, trmatrix, tresp_logt, temps, max_iter=15)
    dem = dem.clip(min=0)
    return dem, edem, elogt, chisq, dn_reg, mlogt, logtemps


def make_datetime_range(start_time=None, end_time=None, cadence=None):
    """
    Make a list of datetime strings between two given datetime strings.
    Cadence is in seconds.
    """
    from datetime import datetime, timedelta
    
    # Given start and end times
    st = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    et = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
    
    # Create list of datetime strings with 12-second cadence
    datetime_list = []
    current_time = st
    
    while current_time <= et:
        datetime_list.append(current_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
        current_time += timedelta(seconds=cadence)
    
    return datetime_list


passband = [94, 131, 171, 193, 211, 335]
data_dir = '/home/mnedal/data'


# Your target date and time
# date_time_str = '2024-05-14 17:39:00.00' # single frame

datetime_list = make_datetime_range(start_time='2024-05-14 17:00:00.00',
                                    end_time='2024-05-14 19:00:00.00',
                                    cadence=12) # range of frames

for date_time_str in datetime_list:
    target_datetime = dt.datetime.strptime(f'{date_time_str}', '%Y-%m-%d %H:%M:%S.%f')
    
    farray = []
    for channel in passband:
        files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv15/*.fits'))
        closest_index = find_closest_filename(files, channel, target_datetime)
        aia_file = files[closest_index]
        farray.append(aia_file)
    
    maps = Map(farray)
    
    
    frame_folder = maps[0].meta['t_rec'][:-1].replace('-','').replace(':','')
    err_arr_tit = f'{data_dir}/tornado_files/{frame_folder}/error_data_{frame_folder}.asdf'
    dem_arr_tit = f'{data_dir}/tornado_files/{frame_folder}/dem_data_{frame_folder}.asdf'
    os.makedirs(f'{data_dir}/tornado_files/{frame_folder}', exist_ok='True')
    os.makedirs(f'{data_dir}/tornado_files/png', exist_ok='True')
    
    if os.path.exists(frame_folder):
        if os.path.exists(err_arr_tit) and os.path.exists(dem_arr_tit):
            print(f'{frame_folder} exists and processed already.')
            pass
        else:
            print(f'DEM file is missing from {frame_folder} !')
    else:
        print(f'\n====================\nProcessing {frame_folder} ...\n====================\n')
        top_right   = SkyCoord(-840*u.arcsec, 420*u.arcsec, frame=maps[0].coordinate_frame)
        bottom_left = SkyCoord(-920*u.arcsec, 300*u.arcsec, frame=maps[0].coordinate_frame)
        submap_0    = maps[0].submap(bottom_left, top_right=top_right)
        nx, ny      = submap_0.data.shape
        nf          = len(maps)
        map_arr     = []
        err_array = np.zeros([nx, ny, nf])
        
        for i, m in enumerate(maps):
            # crop the region of interest
            top_right   = SkyCoord(-840*u.arcsec, 420*u.arcsec, frame=m.coordinate_frame)
            bottom_left = SkyCoord(-920*u.arcsec, 300*u.arcsec, frame=m.coordinate_frame)
            submap      = m.submap(bottom_left, top_right=top_right)
            map_arr.append(submap)
            
            num_pix = submap.data.size
            err_array[:,:,i] = estimate_error(submap.data*(u.ct/u.pix), submap.wavelength, n_samples=num_pix)
        
        map_array = Map(map_arr[0], map_arr[1], map_arr[2],
                        map_arr[3], map_arr[4], map_arr[5],
                        sequence=True, sortby=None)
        
        
        map_arr_tit = data_dir + '/tornado_files/' + frame_folder + '/prepped_data_{index:03}.fits'
        map_array.save(map_arr_tit, overwrite='True')
        
        tree = {'err_array': err_array}
        with asdf.AsdfFile(tree) as asdf_file:
            asdf_file.write_to(err_arr_tit, all_array_compression='zlib')
        
        # # export prepped maps as asdf file
        # files = sorted(glob.glob(f'{data_dir}/tornado_files/*.fits'))
        # tree = {}
        # for i, file in enumerate(files):
        #     tree[f'image_array_{i}'] = Map(file)
        # with asdf.AsdfFile(tree) as asdf_file:
        #     asdf_file.write_to(f'{data_dir}/tornado_files/image_arrays_{frame_folder}.asdf', all_array_compression='zlib')
        
        print('Calculating DEM ...')
        dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = calculate_dem(map_array, err_array)
        
        tree = {'dem':dem, 'edem':edem, 'mlogt':mlogt, 'elogt':elogt, 'chisq':chisq, 'logtemps':logtemps}
        with asdf.AsdfFile(tree) as asdf_file:
            asdf_file.write_to(dem_arr_tit, all_array_compression='zlib')
        
        
        
        # Get a submap to have the scales and image properties
        nt     = len(dem[0,0,:])
        nt_new = int(nt/2)
        nc, nr = 3, 3
        plt.rcParams.update({'font.size':12, 'font.family':"DejaVu Sans",\
                             'font.sans-serif':"DejaVu Sans", 'mathtext.default':"regular"})
        
        fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=[12,12], sharex=True, sharey=True, subplot_kw=dict(projection=submap), layout='constrained')
        plt.suptitle('Image time: '+dt.datetime.strftime(submap.date.datetime, "%Y-%m-%d %H:%M:%S"))
        fig.supxlabel('Solar X (arcsec)', y=0.015)
        fig.supylabel('Solar Y (arcsec)', x=0.1)
        cmap = plt.cm.get_cmap('cubehelix_r')
        
        for i, axi in enumerate(axes.flat):
            new_dem = (dem[:,:,i*2]+dem[:,:,i*2+1])/2.
            plotmap = Map(new_dem, submap.meta)
            plotmap.plot(axes=axi,
                         norm=colors.LogNorm(vmin=1e18, vmax=1e24),
                         cmap='RdYlBu_r')
            axi.grid(False)
            
            y = axi.coords[1]
            y.set_axislabel(' ')
            if i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8:
                y.set_ticklabel_visible(False)
            x = axi.coords[0]
            x.set_axislabel(' ')
            if i < 6:
                x.set_ticklabel_visible(False)
        
            axi.set_title(f'Log(T) = {logtemps[i*2]:.2f} - {logtemps[i*2+1+1]:.2f}')
        
        fig.tight_layout(pad=0.1, rect=[0, 0, 1, 0.98])
        plt.colorbar(ax=axes.ravel().tolist(), label='$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$', 
                     aspect=40, pad=0.02)
        fig.savefig(f'{data_dir}/tornado_files/png/dem_tornado_{frame_folder}.png', format='png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{data_dir}/tornado_files/{frame_folder}/dem_tornado_{frame_folder}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
























# # passbands = [94, 131, 171, 193, 211, 335]
# # nf = len(passbands)
# channel = 335
# single_frame = False
# # target_datetime = '2024-05-14 17:36:05.0' # Your target date and time

# # start_time = ...
# # end_time   = ...

# os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv1', exist_ok=True)
# os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)


# def extract_datetime(filename):
#     """
#     Function to extract the datetime from a filename.
#     """
#     # Split the filename and extract the date and time parts
#     date_time_part = filename.split('/')[-1]                            # Extracts '2024_05_14T18_49_59.12'
#     date_part = date_time_part.split('T')[0].split(f'{channel}A_')[-1]  # Extracts '2024_05_14'
#     time_part = date_time_part.split('T')[1].split('Z')[0]              # Extracts '18_49_59.12'
    
#     # Reformat date and time to standard datetime format
#     date_str = date_part.replace('_', '-')  # '2024-05-14'
#     time_str = time_part.replace('_', ':')  # '18:49:59.12'
    
#     # Combine date and time and convert to datetime object
#     return datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M:%S.%f')


# def find_closest_filename(filenames, target_datetime):
#     """
#     Function to find the index of the filename with the closest datetime to a given target.
#     """
#     closest_index = None
#     min_time_diff = None
    
#     for i, filename in enumerate(filenames):
#         file_datetime = extract_datetime(filename)
#         # print(type(file_datetime))
#         # print(type(target_datetime))
        
#         # Calculate the absolute time difference
#         time_diff = abs(file_datetime - target_datetime)
        
#         # Update the closest file if this one is closer
#         if min_time_diff is None or time_diff < min_time_diff:
#             min_time_diff = time_diff
#             closest_index = i
    
#     return closest_index


# def extract_datetime_v1(filename):
#     import re
#     # Regular expression to capture the date-time part of the filename
#     pattern = r'_(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}\.\d{2})Z'
    
#     # Search the filename for the pattern
#     match = re.search(pattern, filename)
    
#     if match:
#         # Replace underscores with colons and hyphens to format as a standard date-time string
#         datetime_str = match.group(1).replace('_', '-', 2).replace('_', ':').replace('T', ' ')
#         return datetime_str
#     else:
#         return []  # Return empty list if no match found


# def do_process(date_time_str):
#     target_datetime = datetime.strptime(f'{date_time_str}', '%Y-%m-%d %H:%M:%S.%f')
    
#     # find the file index with the nearest datetime to the given one above
#     files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
    
#     closest_index = find_closest_filename(files, target_datetime)
    
#     print(f'\nclosest index to the given date: {closest_index}\n')
    
#     # load the file as a sunpy map
#     aia_file = files[closest_index]
    
#     output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{aia_file.split("/")[-1].replace("lev1", "lev15")}'
#     if os.path.exists(output_filename):
#         print(f'{output_filename} exists and processed already.')
#         pass
#     else:
#         m = Map(aia_file)
#         print(f'Upgrade AIA {channel}A {aia_file.split("/")[-1]} map to lv1.5 and deconvolve with PSF ..\n')
        
#         # # crop the region of interest
#         # top_right   = SkyCoord(-840*u.arcsec, 420*u.arcsec, frame=m.coordinate_frame)
#         # bottom_left = SkyCoord(-920*u.arcsec, 300*u.arcsec, frame=m.coordinate_frame)
#         # submap      = m.submap(bottom_left, top_right=top_right)
#         # print(f'submap shape: {submap.data.shape}')
        
#         psf                      = aiapy.psf.psf(m.wavelength)
#         aia_map_deconvolved      = aiapy.psf.deconvolve(m, psf=psf)
#         print('Deconvolution is finished')
#         aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
#         print('Updating pointing is finished')
#         aia_map_registered       = register(aia_map_updated_pointing)
#         print('Registration is finished')
#         aia_map_corrected        = correct_degradation(aia_map_registered)
#         print('Degradation correction is finished')
#         aia_map_norm             = aia_map_corrected / aia_map_corrected.exposure_time
#         print('Exposure time correction is finished')
        
#         aia_map_norm.save(output_filename, filetype='auto', overwrite=True) # overwrite bc I already have lv1.5 but without PSF deconvolve.
        
#         print('Images prepared and exporeted with the region of interest selected')

# # ====================================================================================================
# # START FROM HERE ...
# # ====================================================================================================

# # datetime_list = []
# # if single_frame:
# #     datetime_list.append(target_datetime) 
# # else:
# #     files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
# #     for file in files:
# #         filename = file.split('/')[-1]
# #         datetime_list.append(extract_datetime(filename))

# # if len(datetime_list) == 1:
# #     date_time_str = datetime_list[0]
# # else:
# #     for date_time_str in datetime_list:
# #         print(f'Doing frame {date_time_str} now ..')
# #         do_process(date_time_str)

# datetime_list = []
# if single_frame:
#     datetime_list.append(target_datetime) 
# else:
#     files = sorted(glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/*.fits'))
#     for file in files:
#         filename = file.split('/')[-1]
#         datetime_list.append(extract_datetime_v1(filename))

# if len(datetime_list) == 1:
#     date_time_str = datetime_list[0]
# else:
#     for date_time_str in datetime_list:
#         print(f'Doing frame {date_time_str} now ..')
#         do_process(date_time_str)


