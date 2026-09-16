"""A synthetic event with known answers, for the validation suite.

One shock emitting at s = 1 and s = 2. The fundamental drift is quadratic in log f through
73 MHz at 09:20, 33 MHz at 09:29 and 16.5 MHz at 09:50; the harmonic is exactly twice it. Both
bands are split by the same factor and are rendered only where they fall inside the 25-85 MHz
observing band, which reproduces the apparent time offset between them.

The lanes are injected in a deliberately wrong order - downstream branch first, harmonic before
fundamental - so the ordering step has to sort them by frequency rather than by name. The harmonic
lanes go in as fitted Bezier curves and the fundamental ones as clicked points, exercising both
tracing methods.

Nothing here is used by the analysis itself. The notebook contains no synthetic data.
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from typeii import bezier_freq_lane, bezier_indices, draw_bezier, pipeline as pl, tracing
from typeii.config import BEZIER_ANCHORS, HARM, N_REPS
from typeii.session import Run

FLO, FHI, SPLIT = 25.0, 85.0, 1.09
T0 = pd.Timestamp('2025-03-26 09:20:00')
_P = np.polyfit([0, 540, 1800], np.log10([73.0, 33.0, 16.5]), 2)


def f_fund(sec):
    """Fundamental frequency [MHz] at sec seconds after 09:20."""
    return 10 ** np.polyval(_P, np.asarray(sec, float))


def spectra():
    """Synthetic Stokes I (dB) and V/I frames, and the background-subtracted layer."""
    t = pd.date_range('2025-03-26 09:15:00', '2025-03-26 09:55:00', freq='500ms')
    f = np.linspace(20, 90, 900)
    ts = (t - T0).total_seconds().to_numpy()
    I = np.full([len(t), len(f)], 1.0)
    V = np.zeros([len(t), len(f)])
    rng = np.random.default_rng(3)
    for s, amp, pol in [(1, 3.0, 0.12), (2, 2.6, 0.04)]:
        for w, mult in [(1.0, 1.0), (0.8, SPLIT)]:
            track = s * f_fund(ts) * mult
            vis = (track >= FLO) & (track <= FHI) & (ts >= 0) & (ts <= 1800)
            for k in np.flatnonzero(vis):
                g = np.exp(-0.5 * ((f - track[k]) / 1.1) ** 2)
                I[k] += amp * w * g
                V[k] += pol * w * g
    I += rng.normal(0, 0.05, I.shape)
    V += rng.normal(0, 0.01, V.shape)
    df_int = pd.DataFrame(10 * np.log10(np.clip(I, 1e-3, None)), index=t, columns=f)
    df_pol = pd.DataFrame(np.clip(V, -1, 1), index=t, columns=f)
    sub = df_int - np.tile(np.nanmedian(df_int, 0), (df_int.shape[0], 1))
    return df_int, df_pol, sub


def traces(run, quiet=True):
    """The control points and clicks the tracer widget would otherwise have produced."""
    LAYER_T, LAYER_F = run.LAYER_T, run.LAYER_F
    rng = np.random.default_rng(11)
    out = {}
    spec = [('H', 'upper', 'H lane 1', 'bezier'), ('H', 'lower', 'H lane 2', 'bezier'),
            ('F', 'upper', 'F lane 1', 'click'), ('F', 'lower', 'F lane 2', 'click')]
    for b, branch, lab, how in spec:
        mult = SPLIT if branch == 'upper' else 1.0
        s = np.linspace(0, 1800, 4000)
        tr = HARM[b] * f_fund(s) * mult
        sv = s[(tr >= LAYER_F.min()) & (tr <= LAYER_F.max())]
        if branch == 'upper':
            sv = sv[sv >= sv.min() + 0.6 * (sv.max() - sv.min())]      # short downstream lane
        reps = []
        if how == 'click':
            pick = np.linspace(sv.min(), sv.max(), 25)
            for _ in range(N_REPS):
                ft = HARM[b] * f_fund(pick) * mult + rng.normal(0, 0.25, pick.size)
                reps.append({'t': [T0 + pd.Timedelta(seconds=float(v)) for v in pick],
                             'f': [float(v) for v in ft]})
        else:
            ti = np.clip(np.searchsorted(LAYER_T, (T0 + pd.to_timedelta(sv, unit='s')).to_numpy()),
                         0, len(LAYER_T) - 1)
            x0, x1 = int(ti.min()), int(ti.max())
            sec_of = (LAYER_T - T0).total_seconds().to_numpy()

            def yof(i, b=b, mult=mult):
                fq = HARM[b] * f_fund(sec_of[int(np.clip(round(i), 0, len(LAYER_T) - 1))]) * mult
                return float(np.interp(fq, LAYER_F, np.arange(len(LAYER_F))))

            na = BEZIER_ANCHORS
            p0 = np.ravel([[x, yof(x)] for x in np.linspace(x0, x1, na + 2)[1:-1]])

            def cost(p, x0=x0, x1=x1, na=na, yof=yof):
                cs = [(p[2 * k], p[2 * k + 1]) for k in range(na)]
                cu = draw_bezier(x0, yof(x0), x1, yof(x1), cs, na + 1, 60)
                xs = np.clip(cu[:, 0], x0, x1)
                return float(np.mean((cu[:, 1] - np.array([yof(x) for x in xs])) ** 2))

            res = minimize(cost, p0, method='Nelder-Mead',
                           options=dict(maxiter=4000, xatol=0.05, fatol=1e-4))
            if not quiet:
                print(f'  fixture {lab} (bezier): rms {np.sqrt(res.fun):.2f} channels off the lane')
            base = np.array([[x0, yof(x0)], [x1, yof(x1)]]
                            + [[res.x[2 * k], res.x[2 * k + 1]] for k in range(na)], float)
            for k in range(N_REPS):
                pert = base if k == 0 else base + rng.normal(0, run.BEZIER_JITTER, base.shape)
                xi, yi = bezier_indices(pert[0][0], pert[0][1], pert[1][0], pert[1][1],
                                        [list(q) for q in pert[2:]], na + 1)
                reps.append(bezier_freq_lane(xi, yi))
        out[lab] = reps
    return out


class FixtureTracer:
    """Stands in for LaneTracer with the traces already recorded."""

    def __init__(self, recorded):
        self.traces = tracing.TRACE_STORE
        self.n_reps = N_REPS
        self.traces.clear()
        self.traces.update(recorded)
        tracing.TRACE_HISTORY.clear()
        for lab, reps in recorded.items():
            tracing.TRACE_HISTORY.extend([lab] * len(reps))
        tracing.TRACE_KIND.clear()
        for lab in recorded:
            tracing.TRACE_KIND[lab] = ('bezier auto-repeats' if lab.startswith('H')
                                       else 'click')

    @property
    def labels(self):
        return sorted(self.traces)

    @property
    def passes(self):
        n = max(len(v) for v in self.traces.values())
        return [{l: (v[i] if i < len(v) else None) for l, v in self.traces.items()}
                for i in range(n)]

    def colour(self, lab):
        return {'F': 'tab:blue', 'H': 'tab:red'}[lab.split()[0]]

    def summary(self):
        return pd.DataFrame([{'label': l, 'repeats': len(v)} for l, v in self.traces.items()])


def build(outputs='/tmp/nenufar_fixture_out', through='A.5', quiet=True):
    """A Run carried through the analysis as far as `through`."""
    os.makedirs(outputs, exist_ok=True)
    run = Run(*spectra(), outputs)
    pl.build_layer(run)
    pl.build_model_grid(run)
    run.tracer = FixtureTracer(traces(run, quiet=quiet))
    pl.collect_traces(run)
    pl.plot_traced_lanes(run)
    pl.analyse_lanes(run)
    if through >= 'A.5':
        pl.polarisation(run)
    return run
