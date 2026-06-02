#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os
print(f"Physical cores available: {os.cpu_count()}")
os.environ['OMP_NUM_THREADS'] = '32' # 8, 20, 32

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
chunk_start = datetime(int(YEAR), int(MONTH), int(DAY), 8, 45, 0)    # <-- edit per screen (08:30)
chunk_end   = datetime(int(YEAR), int(MONTH), int(DAY), 9, 5, 0)    # <-- edit per screen (09:05)
# ─────────────────────────────────────────────────────────────────────────────

# ── Region of interest (arcsec, Helioprojective) ─────────────────────────────
# Set all four to None to process the full disk.
ROI_TOP    =  400   # arcsec
ROI_RIGHT  =  500   # arcsec
ROI_BOTTOM = -200   # arcsec
ROI_LEFT   = -120   # arcsec
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



def load_tresp(tresp_path='/home/mnedal/data/aia_tresp_en.dat'):
    """
    Load the AIA temperature response data once and return all derived
    quantities needed by calculate_dem_tiled.  Call this once before the
    main loop so the .dat file is not re-read for every timestep.
    """
    trin       = io.readsav(tresp_path)
    tresp_logt = np.array(trin['logt'])
    nt         = len(tresp_logt)
    nf_resp    = len(trin['tr'][:])
    trmatrix   = np.zeros((nt, nf_resp))
    for i in range(nf_resp):
        trmatrix[:, i] = trin['tr'][i]

    t_space  = 0.1
    t_min    = 5.6
    t_max    = 7.4
    logtemps = np.linspace(t_min, t_max, num=int((t_max - t_min) / t_space) + 1)
    temps    = 10 ** logtemps
    mlogt    = [np.mean([np.log10(temps[i]), np.log10(temps[i + 1])])
                for i in range(len(temps) - 1)]

    return tresp_logt, trmatrix, temps, mlogt, logtemps


def calculate_dem_tiled(map_array, err_array, tresp_logt, trmatrix, temps,
                        mlogt, logtemps, tile_size=512):
    """
    Calculate DEM in spatial tiles to avoid OOM on full 4096×4096 frames.

    Processing the whole image at once allocates ~25–30 GB of working memory
    (image + error + dem/edem/elogt/dn_reg outputs + dn2dem_pos internals).
    The OOM killer silently terminates the process with no traceback.

    With tile_size=512, peak memory per tile is ~140 MB, and the full-frame
    output arrays are assembled incrementally.

    Parameters
    ----------
    map_array   : sunpy MapSequence  (nf maps of shape nx×ny)
    err_array   : np.ndarray         shape (nx, ny, nf)
    tresp_logt / trmatrix / temps / mlogt / logtemps : from load_tresp()
    tile_size   : int  – spatial tile edge in pixels (default 512)
    """
    nx, ny, nf = err_array.shape
    nt_dem     = len(mlogt)

    # ── Build the full image cube (nx,ny,nf) with padding for mis-sized maps ──
    image_array = np.zeros((nx, ny, nf), dtype=np.float64)
    for img in range(nf):
        data = map_array[img].data
        if data.shape != (nx, ny):
            r = min(nx, data.shape[0])
            c = min(ny, data.shape[1])
            image_array[:r, :c, img] = data[:r, :c]
        else:
            image_array[:, :, img] = data

    # ── Allocate full-frame output arrays ─────────────────────────────────────
    dem_full    = np.zeros((nx, ny, nt_dem), dtype=np.float64)
    edem_full   = np.zeros((nx, ny, nt_dem), dtype=np.float64)
    elogt_full  = np.zeros((nx, ny, nt_dem), dtype=np.float64)
    chisq_full  = np.zeros((nx, ny),         dtype=np.float64)
    dn_reg_full = np.zeros((nx, ny, nf),     dtype=np.float64)

    # ── Tile loop ─────────────────────────────────────────────────────────────
    x_starts   = list(range(0, nx, tile_size))
    y_starts   = list(range(0, ny, tile_size))
    total_tiles = len(x_starts) * len(y_starts)

    with tqdm(total=total_tiles, desc='  DEM tiles', leave=False) as tile_pbar:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, nx)
            for y0 in y_starts:
                y1 = min(y0 + tile_size, ny)

                img_tile = image_array[x0:x1, y0:y1, :]   # view, no copy
                err_tile = err_array[x0:x1, y0:y1, :]

                dem_t, edem_t, elogt_t, chisq_t, dn_reg_t = dn2dem_pos(
                    img_tile, err_tile, trmatrix, tresp_logt, temps, max_iter=15
                )
                
                # dn2dem_pos may return arrays as (ny, nx, nt)
                # while our storage convention is (nx, ny, nt)
                
                if dem_t.shape[:2] != (x1 - x0, y1 - y0):
                    dem_t = np.transpose(dem_t, (1, 0, 2))
                
                if edem_t.shape[:2] != (x1 - x0, y1 - y0):
                    edem_t = np.transpose(edem_t, (1, 0, 2))
                
                if elogt_t.shape[:2] != (x1 - x0, y1 - y0):
                    elogt_t = np.transpose(elogt_t, (1, 0, 2))
                
                if dn_reg_t.shape[:2] != (x1 - x0, y1 - y0):
                    dn_reg_t = np.transpose(dn_reg_t, (1, 0, 2))
                
                if chisq_t.shape != (x1 - x0, y1 - y0):
                    chisq_t = chisq_t.T
                
                dem_full   [x0:x1, y0:y1, :] = dem_t.clip(min=0)
                edem_full  [x0:x1, y0:y1, :] = edem_t
                elogt_full [x0:x1, y0:y1, :] = elogt_t
                chisq_full [x0:x1, y0:y1]    = chisq_t
                dn_reg_full[x0:x1, y0:y1, :] = dn_reg_t
                
                tile_pbar.update(1)

    return dem_full, edem_full, elogt_full, chisq_full, dn_reg_full, mlogt, logtemps



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

# ── Load temperature response data once (re-used for every timestep) ─────────

print('Loading AIA temperature response data...')
tresp_logt, trmatrix, temps, mlogt, logtemps = load_tresp()
print(f'  tresp loaded: {len(tresp_logt)} log-T points, '
      f'{trmatrix.shape[1]} passbands\n')

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

        # Crop to ROI if coordinates are defined.
        # Each map is cropped independently using its own coordinate frame so
        # that WCS metadata (CRPIX, CRVAL, etc.) stays consistent in the saved
        # FITS files.  All six channels are co-registered lv1.5 images, so the
        # same arcsec box selects the identical physical region in every map.
        if None not in (ROI_TOP, ROI_RIGHT, ROI_BOTTOM, ROI_LEFT):
            cropped = []
            for m in maps:
                bl = SkyCoord(ROI_LEFT  * u.arcsec, ROI_BOTTOM * u.arcsec,
                              frame=m.coordinate_frame)
                tr = SkyCoord(ROI_RIGHT * u.arcsec, ROI_TOP    * u.arcsec,
                              frame=m.coordinate_frame)
                cropped.append(m.submap(bl, top_right=tr))
            maps = cropped
            print(f'  ROI crop applied: '
                  f'[{ROI_LEFT}", {ROI_BOTTOM}"] -> [{ROI_RIGHT}", {ROI_TOP}"] '
                  f'-> {maps[0].data.shape[1]}x{maps[0].data.shape[0]} px')

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
        print('Calculating DEM (tiled)...')
        dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = calculate_dem_tiled(
            map_array, err_array,
            tresp_logt, trmatrix, temps, mlogt, logtemps,
            tile_size=512
        )
        
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
