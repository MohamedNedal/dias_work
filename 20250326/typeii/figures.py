"""Figure helpers shared by the notebook plots.
"""
import os

import numpy as np

from . import config as cfg
from .config import JOINT
from .fitting import accel_is_measured, sg_smooth


def save_fig(fig, name, dpi=300):
    """Save a figure to the run output directory at 300 dpi."""
    path = os.path.join(cfg.OUTDIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print('saved', path)

def has_track(run, key, k):
    """Does this track carry enough finite samples of k to be worth plotting?"""
    # state this step works on
    ALref = run.ALref

    return key in ALref and np.isfinite(ALref[key][k + '_mean']).sum() > 2

def track_panel(run, ax, key_list, k, ylabel, unit, fmt='{:.3g}'):
    """Plot one derived quantity against time for the given tracks, with error bars."""
    # state this step works on
    ALref, A_BIAS = run.ALref, run.A_BIAS
    LANE_COL, t0, tg = run.LANE_COL, run.t0, run.tg

    for key in key_list:
        if not has_track(run, key, k):
            continue
        c = LANE_COL[key]
        m, err = ALref[key][k + '_mean'], ALref[key][k + '_se']
        joint = key == JOINT
        ax.errorbar(tg, m, yerr=err, fmt='o', ms=3, color=c, ecolor=c, elinewidth=0.7,
                    capsize=1.5, alpha=(0.6 if joint else 0.4))
        mval, eval_ = np.nanmean(m), np.sqrt(np.nanmean(err ** 2))
        lbl = f'{key}: {fmt.format(mval)} $\\pm$ {fmt.format(eval_)} {unit}'
        # An acceleration that does not clear its own error bar is still plotted and still
        # labelled with its value: marking it "n.s." says what the reader needs without
        # pretending the lane produced nothing.
        if k == 'a' and not accel_is_measured(mval, eval_, bias=A_BIAS.get(key, 0)):
            lbl += '  (n.s.)'
        ax.plot(tg, sg_smooth(m), '-', color=c, lw=(2.6 if joint else 1.6),
                zorder=(5 if joint else 2), label=lbl)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f'time since {t0.strftime("%H:%M")} UT [s]')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
