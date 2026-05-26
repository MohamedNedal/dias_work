#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')
import os
import glob
import re
from datetime import datetime, timedelta
from sunpy.map import Map
from astropy import units as u
from astropy.coordinates import SkyCoord
from aiapy.calibrate.prep import correct_degradation
from aiapy.calibrate import register, update_pointing
import aiapy.psf
from tqdm import tqdm

mydate    = '2025-10-06'
data_dir  = '/home/mnedal/data'
# passbands = [94, 131, 171, 193, 211, 304, 335]
passbands = [94, 335]

# ── Chunk definition: set these per screen ────────────────────────────────────
chunk_start = datetime(2025, 10, 6, 9, 10,  0)   # <-- edit per screen [Min st: 08:30]
chunk_end   = datetime(2025, 10, 6, 9, 15,  0)   # <-- edit per screen [Max et: 09:15]
# ─────────────────────────────────────────────────────────────────────────────

print('=====================================\n Process AIA image from lv1 to lv1.5 with deconvolution \n =====================================')


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


def do_process(aia_file, channel):
    m = Map(aia_file)
    print(f'  Upgrading AIA {channel}A {os.path.basename(aia_file)} to lv1.5 ...')
    psf                      = aiapy.psf.psf(m.wavelength)
    aia_map_deconvolved      = aiapy.psf.deconvolve(m, psf=psf)
    print('  Deconvolution done')
    aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
    print('  Pointing update done')
    aia_map_registered       = register(aia_map_updated_pointing)
    print('  Registration done')
    aia_map_corrected        = correct_degradation(aia_map_registered)
    print('  Degradation correction done')
    aia_map_norm             = aia_map_corrected / aia_map_corrected.exposure_time
    print('  Exposure correction done')
    output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{os.path.basename(aia_file).replace("lev1", "lev15")}'
    aia_map_norm.save(output_filename, filetype='auto', overwrite=True)


# ── Collect all files ─────────────────────────────────────────────────────────

mydate_fmt = mydate.replace('-', '_')
all_files  = {}
for channel in passbands:
    os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)
    all_files[channel] = sorted(
        glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/aia.lev1.{channel}A_{mydate_fmt}T*.fits')
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
            try:
                do_process(files[channel], channel)
            except Exception as e:
                print(f'  [{channel}A] ERROR: {e}')

        pbar.update(1)

print('\nChunk processed successfully.')












# import warnings
# warnings.simplefilter('ignore')
# import re
# import os
# import glob
# from datetime import datetime, timedelta
# from sunpy.map import Map
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# from aiapy.calibrate.prep import correct_degradation
# from aiapy.calibrate import register, update_pointing
# import aiapy.psf
# from tqdm import tqdm

# mydate = '2025-10-06'
# data_dir = '/home/mnedal/data'
# passbands = [94, 131, 171, 193, 211, 304, 335]

# print('=====================================\n Process AIA image from lv1 to lv1.5 with deconvolution \n =====================================')


# def parse_timestamp(filepath):
#     """Extract datetime from AIA filename."""
#     basename = os.path.basename(filepath)
#     match = re.search(r'(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})', basename)
#     if not match:
#         raise ValueError(f'No timestamp pattern found in filename: {basename}')
#     return datetime.strptime(match.group(1), '%Y_%m_%dT%H_%M_%S')


# def group_files_by_timestep(all_files, tolerance_seconds=30):
#     """
#     Group files from all channels by approximate timestamp.
#     Files within `tolerance_seconds` of each other are considered the same timestep.
#     Returns a list of dicts: [{'timestamp': dt, 'files': {channel: filepath}}, ...]
#     """
#     # Build a flat list of (timestamp, channel, filepath)
#     entries = []
#     for channel, files in all_files.items():
#         for f in files:
#             try:
#                 ts = parse_timestamp(f)
#                 entries.append((ts, channel, f))
#             except ValueError:
#                 print(f'Warning: Could not parse timestamp from {f}, skipping.')

#     # Sort by timestamp
#     entries.sort(key=lambda x: x[0])

#     # Group by proximity: greedily assign to the nearest existing group
#     groups = []  # list of {'timestamp': dt, 'files': {channel: filepath}}
#     tolerance = timedelta(seconds=tolerance_seconds)

#     for ts, channel, filepath in entries:
#         # Try to find an existing group within tolerance
#         matched = None
#         for group in reversed(groups):  # check most recent groups first
#             if abs(ts - group['timestamp']) <= tolerance:
#                 matched = group
#                 break
#             if ts - group['timestamp'] > tolerance:
#                 break  # entries are sorted, no point looking further back

#         if matched is None:
#             groups.append({'timestamp': ts, 'files': {channel: filepath}})
#         else:
#             # Prefer not to overwrite if the channel already exists in this group
#             if channel not in matched['files']:
#                 matched['files'][channel] = filepath
#             else:
#                 # Closer file wins
#                 existing_ts = parse_timestamp(matched['files'][channel])
#                 if abs(ts - matched['timestamp']) < abs(existing_ts - matched['timestamp']):
#                     matched['files'][channel] = filepath

#     return groups


# def do_process(aia_file, channel):
#     """
#     Process AIA image from lv1 to lv1.5 with deconvolution.
#     """
#     m = Map(aia_file)
#     print(f'  Upgrading AIA {channel}A {os.path.basename(aia_file)} to lv1.5 ...')
#     psf                      = aiapy.psf.psf(m.wavelength)
#     aia_map_deconvolved      = aiapy.psf.deconvolve(m, psf=psf)
#     print('  Deconvolution done')
#     aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
#     print('  Pointing update done')
#     aia_map_registered       = register(aia_map_updated_pointing)
#     print('  Registration done')
#     aia_map_corrected        = correct_degradation(aia_map_registered)
#     print('  Degradation correction done')
#     aia_map_norm             = aia_map_corrected / aia_map_corrected.exposure_time
#     print('  Exposure correction done')
#     output_filename = f'{data_dir}/AIA/{channel}A/highres/lv15/{os.path.basename(aia_file).replace("lev1", "lev15")}'
#     aia_map_norm.save(output_filename, filetype='auto', overwrite=True)


# # ── Setup ──────────────────────────────────────────────────────────────────────

# mydate_fmt = mydate.replace('-', '_')

# # Create output directories and collect files per channel
# all_files = {}
# for channel in passbands:
#     os.makedirs(f'{data_dir}/AIA/{channel}A/highres/lv15', exist_ok=True)
#     all_files[channel] = sorted(
#         glob.glob(f'{data_dir}/AIA/{channel}A/highres/lv1/aia.lev1.{channel}A_{mydate_fmt}T*.fits')
#     )

# # Group files across channels by approximate timestamp
# timestep_groups = group_files_by_timestep(all_files, tolerance_seconds=30)
# print(f'Found {len(timestep_groups)} unique timesteps across {len(passbands)} channels.\n')

# # ── Main loop: iterate over timesteps, then channels ──────────────────────────

# with tqdm(total=len(timestep_groups), desc='Timesteps processed') as pbar:
#     for group in timestep_groups:
#         ts   = group['timestamp']
#         files = group['files']
#         print(f'\n── Timestep: {ts.strftime("%Y-%m-%dT%H:%M:%S")} '
#               f'({len(files)}/{len(passbands)} channels available) ──')
#         for key in files:
#             print(f'{key}: {files[key]}')
        
#         for channel in passbands:
#             if channel not in files:
#                 print(f'  [{channel}A] No file found for this timestep, skipping.')
#                 continue
#             try:
#                 do_process(files[channel], channel)
#             except Exception as e:
#                 print(f'  [{channel}A] ERROR: {e}')

#         pbar.update(1)

# print('\nAll timesteps processed successfully.')
