"""Load the per-slit kinematics CSVs written by SDO_to_STEREO_Jmap.ipynb and
re-plot them with plot_all_slit_speeds / plot_all_slit_accelerations.

The notebook saves either:
  - one file per slit:  {slit_id}_{feature_id}_kinematics.csv
  - a combined file:     all_slits_{feature_id}_kinematics_EUVI.csv
Both carry unit-suffixed columns (e.g. speed_mean_km/s). The plotting functions
expect an `all_kin` dict keyed by slit_id, each value holding the bare-name
arrays (x_time_num, speed_mean, speed_sem, ...), so load_all_kin rebuilds that.

Usage:
    python load_and_plot_kinematics.py /path/to/output_dir
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------------------------------------------------------- #
# Plotting (copied verbatim from the notebook so this script stands alone)
# ----------------------------------------------------------------------------- #
def save_fig(fig, savepath, dpi=300):
    """Publication export: tight bounding box, minimal whitespace. No-op if savepath is None."""
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight', pad_inches=0.02)


def _track_number(track_id):
    """Trailing integer in an id like 'slit_07' or 'arc_10', for numeric ordering."""
    nums = re.findall(r'\d+', str(track_id))
    return int(nums[-1]) if nums else 0


def _plot_all_slits(all_kin, quantity, ylabel, title, errorbar_every=1,
                    show_title=True, savepath=None, dpi=300, annotate_scalar=False):
    """Overlay one kinematic quantity across slits, ordered by slit number.

    With annotate_scalar each legend entry also shows that track's time-averaged
    value +/- mean SEM.
    """
    fig, ax = plt.subplots(figsize=[9, 5])
    ordered = sorted(all_kin.items(), key=lambda kv: _track_number(kv[0]))
    for slit_id, kin in ordered:
        x = np.asarray(kin['x_time_num'])
        idx = np.arange(len(x))[::errorbar_every]
        if annotate_scalar:
            unit = kin.get(f'{quantity}_unit', '')
            m = np.nanmean(kin[f'{quantity}_mean'])
            s = np.nanmean(kin[f'{quantity}_sem'])
            label = f'{slit_id}: {m:.1f} ± {s:.1f} {unit}'
        else:
            label = slit_id
        ax.errorbar(x[idx], kin[f'{quantity}_mean'][idx], yerr=kin[f'{quantity}_sem'][idx],
                    fmt='o-', ms=3, lw=1.2, capsize=2, label=label)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel('Time [UT]')
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_all_slit_speeds(all_kin, errorbar_every=1, show_title=True, savepath=None, dpi=300,
                         annotate_scalar=False):
    return _plot_all_slits(all_kin, 'speed', 'Speed [km/s]', 'Speed along slits',
                           errorbar_every, show_title, savepath, dpi, annotate_scalar)


def plot_all_slit_accelerations(all_kin, errorbar_every=1, show_title=True, savepath=None, dpi=300,
                                annotate_scalar=False):
    return _plot_all_slits(all_kin, 'acceleration', r'Acceleration [m/s$^2$]',
                           'Acceleration along slits', errorbar_every, show_title, savepath, dpi,
                           annotate_scalar)


# ----------------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------------- #
# Bare key -> the column stem to look for (unit suffix, if any, is stripped).
_SCALAR_COLS = ['slit_id', 'feature_id', 'n_repeats']
_ARRAY_STEMS = ['distance_mean', 'distance_std', 'distance_sem',
                'speed_mean', 'speed_std', 'speed_sem',
                'acceleration_mean', 'acceleration_std', 'acceleration_sem']


def _df_to_kin(df):
    """Turn one slit's DataFrame block back into a kinematics dict."""
    kin = {
        'slit_id': df['slit_id'].iloc[0],
        'feature_id': df['feature_id'].iloc[0],
        'n_repeats': int(df['n_repeats'].iloc[0]),
        'x_time': df['time'].to_numpy(),
        'x_time_num': df['time_num'].to_numpy(dtype=float),
        't_seconds': df['t_seconds'].to_numpy(dtype=float),
    }
    # Match each stem to its (possibly unit-suffixed) column, e.g. speed_mean_km/s.
    for stem in _ARRAY_STEMS:
        col = next((c for c in df.columns if c == stem or c.startswith(stem + '_')), None)
        if col is None:
            continue
        kin[stem] = df[col].to_numpy(dtype=float)
        m = re.match(rf'{stem}_(.+)', col)        # recover the unit for axis labels
        if m:
            kin[f'{stem.split("_")[0]}_unit'] = m.group(1)
    return kin


def load_all_kin(output_dir='.', feature_id=None, prefer_combined=True):
    """Build the `all_kin` dict from the notebook's CSV outputs.

    prefer_combined: if a combined all_slits_*_kinematics*.csv exists, read it;
    otherwise fall back to the per-slit *_kinematics.csv files.
    """
    if prefer_combined:
        pat = f'all_slits_{feature_id}_kinematics*.csv' if feature_id else 'all_slits_*_kinematics*.csv'
        combined = sorted(glob.glob(os.path.join(output_dir, pat)))
        if combined:
            df = pd.read_csv(combined[0])
            return {s: _df_to_kin(g) for s, g in df.groupby('slit_id', sort=False)}

    pat = f'*_{feature_id}_kinematics.csv' if feature_id else '*_kinematics.csv'
    files = sorted(f for f in glob.glob(os.path.join(output_dir, pat))
                   if not os.path.basename(f).startswith('all_slits'))
    if not files:
        raise FileNotFoundError(f'No kinematics CSVs found in {output_dir!r} (pattern {pat!r}).')

    all_kin = {}
    for f in files:
        kin = _df_to_kin(pd.read_csv(f))
        all_kin[kin['slit_id']] = kin
    return all_kin


# ----------------------------------------------------------------------------- #
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir', nargs='?', default='.', help='Folder with the kinematics CSVs.')
    p.add_argument('--feature-id', default=None, help='Restrict to one feature_id.')
    p.add_argument('--save', default=None, help='Optional figure path prefix, e.g. ./fig/event.')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()

    all_kin = load_all_kin(args.output_dir, feature_id=args.feature_id)
    print(f'Loaded {len(all_kin)} slit(s): {", ".join(all_kin)}')

    sp_speed = f'{args.save}_speed.png' if args.save else None
    sp_acc = f'{args.save}_acceleration.png' if args.save else None

    plot_all_slit_speeds(all_kin, savepath=sp_speed, dpi=args.dpi)
    plot_all_slit_accelerations(all_kin, savepath=sp_acc, dpi=args.dpi)
    plt.show()
