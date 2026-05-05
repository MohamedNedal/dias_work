#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '32'

import os.path
from sys import path as sys_path
sys_path.append('/home/mnedal/repos/demreg/python')
script_path = os.path.abspath('./scripts')
if script_path not in sys_path:
    sys_path.append(script_path)

import re
import asdf
import glob
import numpy as np
import scipy.io as io
from datetime import datetime, timedelta
from sunpy.map import Map
from astropy import units as u
from astropy.coordinates import SkyCoord
from aiapy.calibrate.prep import correct_degradation
from aiapy.calibrate import register, update_pointing, estimate_error
from aiapy.calibrate.prep import correct_degradation
from dn2dem_pos import dn2dem_pos
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────

mydate    = '2025-10-06'
YEAR, MONTH, DAY = mydate.split('-')
data_dir  = '/home/mnedal/data'
passbands = [131, 171, 193, 211, 304]
os.makedirs(f'{data_dir}/DEM_{mydate.replace("-","")}', exist_ok='True')
output_path = f'{data_dir}/DEM_{mydate.replace("-","")}'

# ── Chunk definition: set these per screen ────────────────────────────────────
chunk_start = datetime(int(YEAR), int(MONTH), int(DAY), 8, 30, 0)   # <-- edit per screen
chunk_end   = datetime(int(YEAR), int(MONTH), int(DAY), 8, 45, 0)   # <-- edit per screen
# ─────────────────────────────────────────────────────────────────────────────

print('=====================================\n Apply DME analysis on AIA lv1.5 images \n =====================================')


def parse_timestamp(filepath):
    """Extract datetime from AIA filename."""
    basename = os.path.basename(filepath)
    match = re.search(r'(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})', basename)
    if not match:
        raise ValueError(f'No timestamp pattern found in filename: {basename}')
    return datetime.strptime(match.group(1), '%Y_%m_%dT%H_%M_%S')


def group_files_by_timestep(all_files, tolerance_seconds=7):
    entries = []
    for channel, files in all_files.items():
        for f in files:
            try:
                ts = parse_timestamp(f)
                entries.append((ts, channel, f))
            except ValueError:
                print(f'Warning: Could not parse timestamp from {f}, skipping.')

    entries.sort(key=lambda x: x[0])

    groups = []
    tolerance = timedelta(seconds=tolerance_seconds)

    for ts, channel, filepath in entries:
        matched = None
        for group in reversed(groups):
            if abs(ts - group['timestamp']) <= tolerance:
                matched = group
                break
            if ts - group['timestamp'] > tolerance:
                break

        if matched is None:
            groups.append({'timestamp': ts, 'files': {channel: filepath}})
        else:
            if channel not in matched['files']:
                matched['files'][channel] = filepath
            else:
                existing_ts = parse_timestamp(matched['files'][channel])
                if abs(ts - matched['timestamp']) < abs(existing_ts - matched['timestamp']):
                    matched['files'][channel] = filepath
    return groups


def calculate_dem(map_array, err_array):
    """
    Function to calculate DEM.
    """
    nx, ny      = map_array[0].data.shape
    nf          = len(map_array)
    image_array = np.zeros((nx, ny, nf))
    for img in range(0, nf):
        image_array[:,:,img] = map_array[img].data
    
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


# ── Collect all files ─────────────────────────────────────────────────────────

mydate_fmt = mydate.replace('-', '_')
all_files  = {}
for channel in passbands:
    all_files[channel] = sorted(
        glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv15/aia.lev15.{channel}A_{mydate_fmt}T*.fits')
    )

# ── Inspect total time window ─────────────────────────────────────────────────

all_timestamps = []
for channel, files in all_files.items():
    for f in files:
        try:
            all_timestamps.append(parse_timestamp(f))
        except ValueError:
            pass

all_timestamps.sort()
total_start = all_timestamps[0]
total_end   = all_timestamps[-1]
total_dur   = total_end - total_start

print(f'\n  Total data time window:')
print(f'    Start : {total_start.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'    End   : {total_end.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'    Duration: {total_dur}\n')
print(f'  This screen will process:')
print(f'    Start : {chunk_start.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'    End   : {chunk_end.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'    Duration: {chunk_end - chunk_start}\n')

# ── Group and filter to chunk ─────────────────────────────────────────────────

all_groups    = group_files_by_timestep(all_files, tolerance_seconds=7)
chunk_groups  = [g for g in all_groups if chunk_start <= g['timestamp'] <= chunk_end]

print(f'  Total timesteps in dataset : {len(all_groups)}')
print(f'  Timesteps in this chunk    : {len(chunk_groups)}\n')

# ── Main loop ─────────────────────────────────────────────────────────────────

check_manually = []

with tqdm(total=len(chunk_groups), desc='Timesteps processed') as pbar:
    for group in chunk_groups:
        ts    = group['timestamp']
        files = group['files']
        print(f'\n── Timestep: {ts.strftime("%Y-%m-%dT%H:%M:%S")} '
              f'({len(files)}/{len(passbands)} channels available) ──')
        for key in files:
            print(f'{key}: {files[key]}')
        
        for channel in passbands:
            if channel not in files:
                print(f'  [{channel}A] No file found for this timestep, skipping.')
                continue

            aia_filenames = [ files[key] for key in files ]
            # print(len(aia_filenames), type(aia_filenames))
            # break
            maps = Map(aia_filenames)
            
            frame_folder = maps[0].meta['t_rec'][:-1].replace('-','').replace(':','')
            err_arr_tit  = f'{output_path}/{frame_folder}/error_data_{frame_folder}.asdf'
            dem_arr_tit  = f'{output_path}/{frame_folder}/dem_data_{frame_folder}.asdf'

            if os.path.exists(f'{output_path}/{frame_folder}'):
                if os.path.exists(err_arr_tit) and os.path.exists(dem_arr_tit):
                    print(f'{frame_folder} exists and processed already.')
                    pass
                else:
                    print(f'DEM file is missing from {frame_folder}!\nProcess it manually.')
                    check_manually.append(ts.strftime("%Y-%m-%dT%H:%M:%S"))
            else:
                print(f'\n====================\nProcessing {frame_folder} ...\n====================\n')
                os.makedirs(f'{output_path}/{frame_folder}', exist_ok='True')
                
                nx, ny    = maps[0].data.shape
                nf        = len(maps)
                err_array = np.zeros([nx, ny, nf])
                
                for i, m in enumerate(maps):
                    num_pix = m.data.size
                    err_array[:,:,i] = estimate_error(m.data*(u.ct/u.pix), m.wavelength, n_samples=num_pix)
                
                map_array = Map(maps[0], maps[1], maps[2],
                                maps[3], maps[4], maps[5],
                                sequence=True, sortby=None)
                
                map_arr_tit = f'{output_path}/{frame_folder}/prepped_data_{{index:03}}.fits'
                map_array.save(map_arr_tit, overwrite='True')
                
                tree = {'err_array': err_array}
                with asdf.AsdfFile(tree) as asdf_file:
                    asdf_file.write_to(err_arr_tit, all_array_compression='zlib')
                
                print('Calculating DEM ..')
                dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = calculate_dem(map_array, err_array)
                
                tree = {'dem':dem, 'edem':edem, 'mlogt':mlogt, 'elogt':elogt, 'chisq':chisq, 'logtemps':logtemps}
                with asdf.AsdfFile(tree) as asdf_file:
                    asdf_file.write_to(dem_arr_tit, all_array_compression='zlib')

            print(f'DEM analysis for {channel}A finished successfully.')

            
            # try:
            #     maps = Map(files)
                
            #     frame_folder = maps[0].meta['t_rec'][:-1].replace('-','').replace(':','')
            #     err_arr_tit  = f'{output_path}/{frame_folder}/error_data_{frame_folder}.asdf'
            #     dem_arr_tit  = f'{output_path}/{frame_folder}/dem_data_{frame_folder}.asdf'

            #     if os.path.exists(f'{output_path}/{frame_folder}'):
            #         if os.path.exists(err_arr_tit) and os.path.exists(dem_arr_tit):
            #             print(f'{frame_folder} exists and processed already.')
            #             pass
            #         else:
            #             print(f'DEM file is missing from {frame_folder}!\nProcess it manually.')
            #             check_manually.append(date_time_str)
            #     else:
            #         print(f'\n====================\nProcessing {frame_folder} ...\n====================\n')
            #         os.makedirs(f'{output_path}/{frame_folder}', exist_ok='True')
                    
            #         nx, ny    = maps[0].data.shape
            #         nf        = len(maps)
            #         err_array = np.zeros([nx, ny, nf])
                    
            #         for i, m in enumerate(maps):
            #             num_pix = m.data.size
            #             err_array[:,:,i] = estimate_error(submap.data*(u.ct/u.pix), m.wavelength, n_samples=num_pix)
                    
            #         map_array = Map(maps[0], maps[1], maps[2],
            #                         maps[3], maps[4], maps[5],
            #                         sequence=True, sortby=None)
                    
            #         map_arr_tit = f'{output_path}/{frame_folder}/prepped_data_{{index:03}}.fits'
            #         map_array.save(map_arr_tit, overwrite='True')
                    
            #         tree = {'err_array': err_array}
            #         with asdf.AsdfFile(tree) as asdf_file:
            #             asdf_file.write_to(err_arr_tit, all_array_compression='zlib')
                    
            #         print('Calculating DEM ..')
            #         dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = calculate_dem(map_array, err_array)
                    
            #         tree = {'dem':dem, 'edem':edem, 'mlogt':mlogt, 'elogt':elogt, 'chisq':chisq, 'logtemps':logtemps}
            #         with asdf.AsdfFile(tree) as asdf_file:
            #             asdf_file.write_to(dem_arr_tit, all_array_compression='zlib')

            #     print(f'DEM analysis for {channel}A finished successfully.')
                
            # except Exception as e:
            #     print(f'  [{channel}A] ERROR: {e}')

        pbar.update(1)
