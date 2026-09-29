# -*- coding: utf-8 -*-
"""The state one analysis run accumulates.

Each pipeline step reads what it needs from a Run, does its work, and writes its results back.
Every step declares both lists at its top and bottom, so the data flow through the analysis is
readable without tracing a notebook namespace.
"""
import os

import pandas as pd

from . import config as cfg


class Run:
    """Everything one pass through the analysis produces.

    Attributes appear as the steps create them; reading one that has not been produced yet raises
    rather than returning None, so a step run out of order fails immediately instead of quietly
    working on missing data.
    """

    def __init__(self, df_int, df_pol, dyspec_subtracted, outputs):
        self.df_int = df_int
        self.df_pol = df_pol
        self.dyspec_subtracted = dyspec_subtracted
        self.EVENT_DATE = str(df_int.index[0].date())
        self.OUTDIR = os.path.join(outputs, 'type2_nenufar')
        os.makedirs(self.OUTDIR, exist_ok=True)
        cfg.OUTDIR = self.OUTDIR
        # A marker stamped now. The stale-output check compares file times against it using the
        # same clock and the same call, so timezone and epoch conventions cannot skew it.
        marker = os.path.join(self.OUTDIR, '.run_marker')
        open(marker, 'w').close()
        self.RUN_START = os.path.getmtime(marker)
        self.TYPEII_WINDOW = (pd.Timestamp(f'{self.EVENT_DATE} {cfg.TYPEII_START}'),
                              pd.Timestamp(f'{self.EVENT_DATE} {cfg.TYPEII_END}'))
        cfg.TYPEII_WINDOW = self.TYPEII_WINDOW
        self.t0 = self.TYPEII_WINDOW[0]      # shared time origin for every lane
        self.BEZIER_JITTER = None

    def set(self, **kw):
        """Write results back to the run."""
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def __getattr__(self, name):
        raise AttributeError(
            f'{name!r} has not been produced yet. The steps are ordered: build_layer, '
            f'build_model_grid, load_controls (or the LaneTracer widget), collect_traces, '
            f'plot_traced_lanes, analyse_lanes, polarisation, fh_tests, audit_consistency, '
            f'height_time_fits, model_sweep, characteristics_table, results_text, export_tracks.'
            f'\nplot_traced_lanes is not optional despite the name: it measures the per-lane '
            f'frequency uncertainty (LANE_SIGMA) that every fit downstream is weighted by.')

    def __repr__(self):
        have = [k for k in vars(self) if not k.startswith('_')]
        return f'<Run {self.EVENT_DATE}: {len(have)} attributes, OUTDIR={self.OUTDIR}>'
