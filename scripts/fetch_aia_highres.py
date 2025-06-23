import warnings
warnings.filterwarnings('ignore')

import logging
import argparse
from pathlib import Path
from datetime import datetime
from astropy import units as u
from sunpy.net import Fido, attrs as a
from tqdm import tqdm

# === EXAMPLE ===
# 
# python fetch_aia_highres.py \
#         -sdt '2011-06-07 06:16'
#         -edt '2011-06-07 06:50'
#         -pb 193 171
#         --sample 12
#         --logfile aia_download.log


# === THIS SCRIPT NEEDS FURTHER INSPECTION, ESPEIALLY THE LAST PART ABOUT CHEING IF FILES EXIST BEFORE DOWNLOADING ===

# === Argument Parser ===
parser = argparse.ArgumentParser(description='Download AIA data from SDO over a time range.')
parser.add_argument('-sdt', '--start_dt', required=True, type=str,
                    help='Start datetime: "YYYY-MM-DD HH:MM[:SS]"')
parser.add_argument('-edt', '--end_dt', required=True, type=str,
                    help='End datetime: "YYYY-MM-DD HH:MM[:SS]"')
parser.add_argument('-pb', '--passbands', required=True, nargs='+', type=int,
                    help='AIA passbands as integers: 193 171 211 ...')
parser.add_argument('--data-dir', type=str, default='/home/mnedal/data',
                    help='Directory to save AIA data (default: /home/mnedal/data)')
# parser.add_argument('--overwrite', action='store_true',
#                     help='Force download even if files already exist')
parser.add_argument('--sample', type=int, default=12,
                    help='Cadence in seconds (default: 12)')
parser.add_argument('--logfile', type=str, default=None,
                    help='Optional log file path')

args = parser.parse_args()

# === Setup Logging ===
log_level = logging.INFO
log_format = '[%(levelname)s] %(message)s'
if args.logfile:
    logging.basicConfig(filename=args.logfile, level=log_level, format=log_format)
else:
    logging.basicConfig(level=log_level, format=log_format)

# === Input Parsing & Validation ===
def parse_datetime_flex(dt_str, label):
    try:
        parts = dt_str.strip().split()
        assert len(parts) == 2, f'{label} must be "YYYY-MM-DD HH:MM[:SS]"'
        if len(parts[1].split(':')) == 2:
            dt_str += ':00'  # Add seconds if missing
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        raise ValueError(f'Invalid {label}: "{dt_str}". Use format "YYYY-MM-DD HH:MM[:SS]"')

start_dt = parse_datetime_flex(args.start_dt, 'start_dt')
end_dt = parse_datetime_flex(args.end_dt, 'end_dt')
assert start_dt < end_dt, 'Start datetime must be before end datetime.'
assert all(isinstance(pb, int) for pb in args.passbands), 'Passbands must be integers.'

# === Data Directory ===
data_dir = Path(args.data_dir)
passbands = args.passbands

# === Download Loop ===
with tqdm(total=len(passbands), desc='Fetching AIA data ...') as pbar:
    for channel in passbands:
        logging.info(f'Downloading AIA {channel}Å from {start_dt} to {end_dt}')
        save_path = data_dir / f'AIA/{channel}A/highres/lv1'
        save_path.mkdir(parents=True, exist_ok=True)

        aia_result = Fido.search(
            a.Time(start_dt.isoformat(), end_dt.isoformat()),
            a.Instrument('AIA'),
            a.Wavelength(channel * u.angstrom),
            a.Sample(args.sample * u.second)
        )

        Fido.fetch(aia_result, path=save_path)
        logging.info(f'AIA {channel}Å data downloaded.')

        # existing_files = list(save_path.glob('*.fits'))
        # if existing_files and not args.overwrite:
        #     logging.info(f'Skipping {channel}Å — data already exists. Use --overwrite to force.')
        # else:
        #     Fido.fetch(aia_result, path=save_path)
        #     logging.info(f'AIA {channel}Å data downloaded.')

        pbar.update(1)
