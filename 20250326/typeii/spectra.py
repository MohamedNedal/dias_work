"""Dynamic-spectrum handling: windowing, decimation and display.

The tracing layer is Stokes I in dB minus each channel time median. Decimation is
in time only, never in frequency, because the band split is a frequency-domain
feature. METHODS.md section 2.
"""
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.ticker import AutoMinorLocator
from tqdm.auto import tqdm

from . import config as cfg
from .config import (INVERT_FREQ, LAYER_CMAP, LAYER_GAMMA, LAYER_MAX_F, LAYER_MAX_T,
                     LAYER_MODE, LAYER_PHI, LAYER_PLO, PREVIEW_MAX_TCOLS)


def decimate(df, max_t=LAYER_MAX_T, max_f=LAYER_MAX_F, chunk=20000):
    """Block-average a (time x frequency) frame to at most max_t x max_f samples.

    Block centres become the new coordinates. Averaged in time chunks so the full
    frame is never copied whole. Returns (frame, (kt, kf)); a frame already small
    enough is returned unchanged.
    """
    nt, nf = df.shape
    kt = max(1, int(np.ceil(nt / max_t)))
    kf = max(1, int(np.ceil(nf / max_f)))
    if kt == 1 and kf == 1:
        return df, (1, 1)
    nt2, nf2 = (nt // kt) * kt, (nf // kf) * kf
    rows = max(kt, (chunk // kt) * kt)
    blocks = []
    for i0 in tqdm(range(0, nt2, rows), desc='decimating', leave=False):
        i1 = min(i0 + rows, nt2)
        v = df.iloc[i0:i1, :nf2].to_numpy(float)
        blocks.append(np.nanmean(v.reshape((i1 - i0) // kt, kt, nf2 // kf, kf), axis=(1, 3)))
    ti = df.index[:nt2].to_numpy().astype('int64').reshape(-1, kt).mean(axis=1)
    fi = np.asarray(df.columns, float)[:nf2].reshape(-1, kf).mean(axis=1)
    return pd.DataFrame(np.vstack(blocks), index=pd.to_datetime(ti.astype('int64')),
                        columns=fi), (kt, kf)

def window(run, df):
    """Restrict a frame to TYPEII_WINDOW in time and TYPEII_FLIM in frequency."""
    # state this step works on
    TYPEII_WINDOW, TYPEII_FLIM = run.TYPEII_WINDOW, cfg.TYPEII_FLIM

    f = np.asarray(df.columns, float)
    keep = (f >= TYPEII_FLIM[0]) & (f <= TYPEII_FLIM[1])
    return df.loc[max(TYPEII_WINDOW[0], df.index[0]):min(TYPEII_WINDOW[1], df.index[-1]),
                  df.columns[keep]]

def layer_norm(D, plo=None, phi=None, gamma=None):
    """Colour normalisation for the tracing layer, from percentiles plo and phi."""
    plo = LAYER_PLO if plo is None else plo
    phi = LAYER_PHI if phi is None else phi
    gamma = LAYER_GAMMA if gamma is None else gamma
    vmin, vmax = np.nanpercentile(D, plo), np.nanpercentile(D, phi)
    if LAYER_MODE == 'ratio':
        return LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    if gamma != 1:
        return PowerNorm(gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)

def draw_layer(run, ax, plo=2, phi=None, gamma=1, cmap=LAYER_CMAP, tstep=None):
    """Draw the tracing layer on ax and return the QuadMesh.

    plo and gamma override the configured display stretch; pass None to use it.
    Records what was used in LAST_STRETCH so a figure title cannot misreport it.
    """
    # state this step works on
    LAYER_T, LAYER_F, LAYER_D = run.LAYER_T, run.LAYER_F, run.LAYER_D
    EVENT_DATE = run.EVENT_DATE

    step = max(1, LAYER_D.shape[1] // PREVIEW_MAX_TCOLS) if tstep is None else tstep
    LAST_STRETCH.update(plo=(LAYER_PLO if plo is None else plo),
                        phi=(LAYER_PHI if phi is None else phi),
                        gamma=(LAYER_GAMMA if gamma is None else gamma))
    pm = ax.pcolormesh(LAYER_T[::step], LAYER_F, LAYER_D[:, ::step],
                       norm=layer_norm(LAYER_D, plo, phi, gamma), cmap=cmap, rasterized=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if INVERT_FREQ:
        ax.invert_yaxis()
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax.set_xlabel(f'Time (UT) on {EVENT_DATE}')
    ax.set_ylabel('Frequency (MHz)')
    return pm



# Recorded by draw_layer so a figure title cannot misreport the stretch it was drawn with.
LAST_STRETCH = {}
