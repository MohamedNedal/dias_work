"""Run the whole notebook without Jupyter, from the archived Bezier control points.

Same call order as plot_nenufar.ipynb, with the tracer replaced by load_controls. Use it to
regenerate every output, to check determinism, or to confirm that a change to typeii/ did what it
was meant to without touching a running kernel.

    python run_headless.py [OUTPUT_DIR]

Writes to OUTPUT_DIR/type2_nenufar/ (default /tmp/typeii_headless).
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)

import typeii as t2
from typeii import config as cfg, pipeline as pl
from typeii.session import Run

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/typeii_headless'
BASE = os.path.join(D, 'combined_dyspec_20250326_091312_20250326_095609')

df_int = 10 * np.log10(pd.read_pickle(BASE + '_stokesI_typeII.pkl'))
df_pol = pd.read_pickle(BASE + '_stokesV_over_I_typeII.pkl')
dyspec_subtracted = df_int - np.tile(np.nanmedian(df_int, 0), (df_int.shape[0], 1))

t2.version()
run = Run(df_int, df_pol, dyspec_subtracted, OUT)

# A.1-A.2
pl.build_layer(run)
pl.plot_stretch_comparison(run)
pl.plot_window(run)
pl.build_model_grid(run)
pl.plot_density_models(run)

# A.3  the lanes, replayed from their control points
run.tracer = pl.load_controls(run, os.path.join(D, 'typeii_bezier_controls.csv'))
pl.collect_traces(run)
pl.plot_traced_lanes(run)          # not optional: it measures LANE_SIGMA

# A.4-A.5
pl.analyse_lanes(run)
pl.polarisation(run)
pl.fh_tests(run)
pl.audit_consistency(run)
pl.plot_fh_checks(run)

# A.6-A.7
pl.plot_kinematics(run)
pl.height_time_fits(run)
pl.plot_fit_comparison(run)
pl.fit_comparison_table(run)

# A.8-A.9   model_sweep writes run.B_tracks, which plot_bfield needs
pl.model_sweep(run)
pl.plot_model_sweep(run)
pl.plot_bfield(run)

# A.10-A.12
pl.characteristics_table(run)
pl.results_text(run)
pl.export_tracks(run)
pl.plot_export_tracks(run)
pl.manifest(run)
