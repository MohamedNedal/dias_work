"""Flatten the notebook against the REAL NenuFAR data and the REAL recorded picks.

Same idea as make_test_script.py, but nothing here is synthetic: the Stokes I and V/I frames are
the user's own pickles, and the traced lanes are the ones actually recorded in typeii_picks.pkl.
This exercises the code paths a fixture cannot - the true channel count, the true cadence, the
real bandpass, real gaps, and lanes whose curvature was placed by hand.
"""
import json
import sys

SRC, DST, N_KEEP = sys.argv[1], sys.argv[2], int(sys.argv[3])
D = '/sessions/inspiring-youthful-mayer/mnt/DIAS'

STUB = f'''import warnings
warnings.filterwarnings('ignore')
import os, glob, pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
mpl.rcParams['date.epoch'] = '1970-01-01T00:00:00'
outputs = '/tmp/nenufar_real_out'
os.makedirs(outputs, exist_ok=True)
def display(x):
    print(repr(x)[:1500])

_base = '{D}/combined_dyspec_20250326_091312_20250326_095609'
df_int = pd.read_pickle(_base + '_stokesI_typeII.pkl')
df_pol = pd.read_pickle(_base + '_stokesV_over_I_typeII.pkl')
df_int = 10 * np.log10(df_int)
dyspec_subtracted = df_int - np.tile(np.nanmedian(df_int, 0), (df_int.shape[0], 1))
print('=== REAL DATA ===')
print('Stokes I  :', df_int.shape,
      f'{{np.nanmedian(np.diff(df_int.columns) * 1e3):.1f}} kHz,',
      f'{{np.nanmedian(np.diff(df_int.index) / np.timedelta64(1, "ms")):.1f}} ms')
print('Stokes V/I:', df_pol.shape)
print('window    :', df_int.index[0], '->', df_int.index[-1])
print('freq      :', f'{{df_int.columns.min():.2f}} - {{df_int.columns.max():.2f}} MHz')
print('non-finite in Stokes I:', int(np.sum(~np.isfinite(df_int.to_numpy()))))
'''

# the real recorded traces, in place of anything the widget would have produced
INJECT = f'''
_pk = pickle.load(open('{D}/type2_nanufar_20250326/typeii_picks.pkl', 'rb'))
TRACE_STORE.clear()
TRACE_STORE.update(_pk['traces'])
TRACE_HISTORY.clear()
for _l, _v in _pk['traces'].items():
    TRACE_HISTORY.extend([_l] * len(_v))
TRACE_KIND.clear()
for _l in _pk['traces']:
    TRACE_KIND[_l] = 'bezier auto-repeats'
tracer = object.__new__(LaneTracer)
tracer.n_reps = N_REPS
tracer.bands = list(BANDS)
tracer.traces = TRACE_STORE
tracer.history = TRACE_HISTORY
tracer.lane_no = {{b: 1 for b in BANDS}}
print('=== REAL PICKS:', len(TRACE_STORE), 'lanes x',
      [len(v) for v in TRACE_STORE.values()], 'repeats ===')
'''

nb = json.load(open(SRC))
chunks = [STUB]
for i, c in enumerate(nb['cells'][N_KEEP:], start=N_KEEP):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if src.lstrip().startswith('%matplotlib widget'):
        continue
    if src.lstrip().startswith('%matplotlib inline'):
        chunks.append(INJECT)
    src = '\n'.join(l for l in src.splitlines()
                    if not l.strip().startswith(('%matplotlib', 'get_ipython')))
    lines = src.rstrip().splitlines()
    if lines and not lines[-1].startswith((' ', '\t')) and lines[-1].strip() \
            and not lines[-1].rstrip().endswith((':', ')', '}')) and ('=' not in lines[-1]) \
            and not lines[-1].startswith(('print', 'import', 'from', '#', 'raise')):
        lines[-1] = f'print({lines[-1].strip()})'
        src = '\n'.join(lines)
    chunks.append(f'\nprint("\\n########## CELL {i} ##########")\n' + src)

open(DST, 'w').write('\n'.join(chunks))
print('wrote', DST)
