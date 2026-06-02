#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os
print(f"Physical cores available: {os.cpu_count()}")
os.environ['OMP_NUM_THREADS'] = '20' # 8, 20, 32

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
passbands = [94, 131, 171, 193, 211, 335]
os.makedirs(f'{data_dir}/DEM_{mydate.replace("-","")}', exist_ok='True')
output_path = f'{data_dir}/DEM_{mydate.replace("-","")}'

# ── Chunk definition: set these per screen ────────────────────────────────────
chunk_start = datetime(int(YEAR), int(MONTH), int(DAY), 8, 45, 0)   # <-- edit per screen
chunk_end   = datetime(int(YEAR), int(MONTH), int(DAY), 9, 5, 0)   # <-- edit per screen
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
    Function to calculate DEM with shape-safety checks.
    """
    # Use the dimensions from the err_array (which we know are consistent)
    nx, ny, nf = err_array.shape 
    image_array = np.zeros((nx, ny, nf))

    for img in range(nf):
        data = map_array[img].data
        if data.shape != (nx, ny):
            # If the image is the wrong size (e.g., 4094 instead of 4096),
            # we pad it into a correctly sized zero-array.
            padded_data = np.zeros((nx, ny))
            row_limit = min(nx, data.shape[0])
            col_limit = min(ny, data.shape[1])
            padded_data[:row_limit, :col_limit] = data[:row_limit, :col_limit]
            image_array[:,:,img] = padded_data
        else:
            image_array[:,:,img] = data
    
    trin = io.readsav('/home/mnedal/data/aia_tresp_en.dat')
    tresp_logt = np.array(trin['logt'])
    nt = len(tresp_logt)
    nf_resp = len(trin['tr'][:])
    trmatrix = np.zeros((nt, nf_resp))
    for i in range(nf_resp):
        trmatrix[:,i] = trin['tr'][i]    
    
    t_space = 0.1
    t_min = 5.6
    t_max = 7.4
    logtemps = np.linspace(t_min, t_max, num=int((t_max-t_min)/t_space)+1)
    temps = 10**logtemps
    mlogt = ([np.mean([(np.log10(temps[i])), np.log10((temps[i+1]))]) for i in np.arange(0, len(temps)-1)])
    
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

all_groups    = group_files_by_timestep(all_files, tolerance_seconds=12)
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
        
        # 1. Check if ALL required passbands are present for this timestep
        missing_channels = [cp for cp in passbands if cp not in files]
        if missing_channels:
            print(f'  [Timestep {ts}] Missing channels {missing_channels}, skipping.')
            pbar.update(1)
            continue

        frame_folder = ts.strftime("%Y%m%d_%H%M%S")
        err_arr_tit = f'{output_path}/{frame_folder}/error_data_{frame_folder}.asdf'
        dem_arr_tit = f'{output_path}/{frame_folder}/dem_data_{frame_folder}.asdf'

        # 2. Check if already processed
        if os.path.exists(err_arr_tit) and os.path.exists(dem_arr_tit):
            print(f'{frame_folder} already processed.')
            pbar.update(1)
            continue

        print(f'\nProcessing {frame_folder} ...')
        os.makedirs(f'{output_path}/{frame_folder}', exist_ok=True)

        # 3. Load and Prep Maps
        # Ensure files are loaded in the specific order of your passbands list
        ordered_files = [files[pb] for pb in passbands]
        maps = Map(ordered_files)

        # Get target dimensions from the first map
        nx, ny = maps[0].data.shape
        nf = len(maps)
        err_array = np.zeros([nx, ny, nf])
        valid_maps = []

        for i, m in enumerate(maps):
            # If shapes mismatch, we need to handle it. 
            # Here we ensure the data matches nx, ny before inserting into the array.
            if m.data.shape != (nx, ny):
                print(f"Warning: Reshaping {m.wavelength} from {m.data.shape} to {(nx, ny)}")
                # Use submap or simple padding/cropping to fix dimensions
                # For DEM, it's safer to ensure they were registered correctly first
                # If they are off by only 2 pixels, they likely weren't padded during 'register'
                m_data = np.zeros((nx, ny))
                min_x = min(nx, m.data.shape[0])
                min_y = min(ny, m.data.shape[1])
                m_data[:min_x, :min_y] = m.data[:min_x, :min_y]
            else:
                m_data = m.data

            # Calculate error
            err_array[:,:,i] = estimate_error(m_data * (u.ct/u.pix), m.wavelength, n_samples=m_data.size)
            valid_maps.append(m)

        # 4. Create MapSequence and Save
        map_array = Map(valid_maps, sequence=True)
        map_arr_tit_pattern = f'{output_path}/{frame_folder}/prepped_data_{{index:03}}.fits'
        map_array.save(map_arr_tit_pattern, overwrite=True)

        # Save error array
        with asdf.AsdfFile({'err_array': err_array}) as af:
            af.write_to(err_arr_tit, all_array_compression='zlib')
        
        # 5. Calculate and Save DEM
        print('Calculating DEM...')
        dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = calculate_dem(map_array, err_array)
        
        tree = {
            'dem': dem, 
            'edem': edem, 
            'mlogt': mlogt, 
            'elogt': elogt, 
            'chisq': chisq, 
            'logtemps': logtemps
        }
        with asdf.AsdfFile(tree) as af:
            af.write_to(dem_arr_tit, all_array_compression='zlib')
        
        print(f'Step {frame_folder} finished.')
        pbar.update(1)
