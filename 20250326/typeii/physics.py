"""Band splitting, the derived shock quantities, and the consistency tests.

X and M_A need no density model; height, speed, v_A and B do. The two averaging
windows are described in METHODS.md section 7 and enforced by lane_windows.
"""
import numpy as np
import pandas as pd

from .config import (AGG_KEYS, BAND_NAME, HARM, JOINT, KIN_N_DENSE, MAKE_JOINT, MU,
                     MU0, M_P, N_MC, POL_DF_MHZ, POL_DT_S, R_SUN_M)
from .models import alfven_mach_from_X, freq_to_density, invert_grid
from .fitting import (eval_lane, kinematics, lane_coeffs, lane_deriv, lane_fit,
                      pass_fits)


def order_lanes(run, labs):
    """Order a band's lanes by frequency over the interval where they overlap.

    The lowest-frequency branch is upstream. Comparing mean frequency over each
    lane's own span instead inverts the split whenever the branches cover
    different stretches of a drifting burst.
    """
    # state this step works on
    passes, LANE_SIGMA = run.passes, run.LANE_SIGMA

    fits = {l: lane_fit(run, passes[0][l], sigma_f=LANE_SIGMA.get(l)) for l in labs}
    lo = max(f['tmin'] for f in fits.values())
    hi = min(f['tmax'] for f in fits.values())
    if hi > lo:
        ts = np.linspace(lo, hi, 40)
        note = f'compared over their {hi - lo:.0f} s of overlap'
    else:
        ts = np.linspace(min(f['tmin'] for f in fits.values()),
                         max(f['tmax'] for f in fits.values()), 40)
        note = ('WARNING: these lanes never overlap in time, so they are ordered on EXTRAPOLATED '
                'fits. Re-trace so the split branches share some time.')
    key = {l: float(np.nanmean(lane_deriv(fits[l], ts)[0])) for l in labs}
    return sorted(labs, key=lambda l: key[l]), key, note

def realise_lanes(run, fits, tg, invert, rng=None, sample=False, mu=MU,
                  kin_method=None):
    """One realisation of r, v, a, v_A, B, X and M_A for every lane.

    Each lane uses its band's harmonic number. B additionally needs that band's
    density jump, so it exists only where both split branches overlap in time.
    """
    # state this step works on
    TRACED, BANDS_TRACED = run.TRACED, run.BANDS_TRACED
    LANE_ORDER, SPLIT_PAIR = run.LANE_ORDER, run.SPLIT_PAIR
    LANE_SIGMA = run.LANE_SIGMA

    rr, ne = invert
    ne_min, ne_max = np.nanmin(ne), np.nanmax(ne)
    # keep the drawn coefficients: the dense per-lane track used for the r(t) fit has to come from
    # the SAME realisation as the gridded one, or the Monte Carlo mixes draws
    cf = {lab: lane_coeffs(fits.get(lab), rng, sample) for lab in TRACED}
    fval = {lab: eval_lane(fits.get(lab), cf[lab], tg) for lab in TRACED}

    def _to_r(f_mhz, band):
        ne_t = freq_to_density(f_mhz * 1e6, harmonic=HARM[band])
        r = np.interp(ne_t, ne[::-1], rr[::-1])
        r[(ne_t > ne_max) | (ne_t < ne_min)] = np.nan
        return r, ne_t

    out = {}
    for b in BANDS_TRACED:
        lo_lab, up_lab = SPLIT_PAIR[b]
        if up_lab is not None:
            with np.errstate(all='ignore'):
                X = (fval[up_lab] / fval[lo_lab]) ** 2
        else:
            X = np.full_like(tg, np.nan)
        MA = alfven_mach_from_X(X)
        for lab in LANE_ORDER[b]:
            r, ne_t = _to_r(fval[lab], b)
            _lf = fits.get(lab)
            t_d = r_d = None
            if _lf is not None:
                t_d = np.linspace(_lf['tmin'], _lf['tmax'], KIN_N_DENSE)
                r_d = _to_r(eval_lane(_lf, cf[lab], t_d), b)[0]
            v, a = kinematics(r, tg, span=np.isfinite(fval[lab]),
                              baseline=(float(_lf['tmax'] - _lf['tmin']) if _lf else None),
                              t_fit=t_d, r_fit=r_d, method=kin_method)
            vA = v / MA
            B = (vA * 1e3) * np.sqrt(MU0 * mu * M_P * (ne_t * 1e6)) * 1e4      # Gauss
            out[lab] = dict(r=r, v=v, a=a, vA=vA, B=B, X=X, MA=MA, ne=ne_t, f=fval[lab])

    # --- combined fundamental + harmonic track -------------------------------------------
    # the two bands are the same shock, so the upstream branches' heights and densities are
    # averaged where both exist and taken singly elsewhere: one trajectory across the whole burst
    ups = [out[SPLIT_PAIR[b][0]] for b in BANDS_TRACED if SPLIT_PAIR[b][0] in out]
    if MAKE_JOINT and len(ups) > 1:
        stack = lambda key: np.nanmean(np.vstack([u[key] for u in ups]), axis=0)
        r_j, ne_j, MA_j = stack('r'), stack('ne'), stack('MA')
        v_j, a_j = kinematics(r_j, tg)
        vA_j = v_j / MA_j
        B_j = (vA_j * 1e3) * np.sqrt(MU0 * mu * M_P * (ne_j * 1e6)) * 1e4
        out[JOINT] = dict(r=r_j, v=v_j, a=a_j, vA=vA_j, B=B_j, X=stack('X'), MA=MA_j,
                          ne=ne_j, f=np.full_like(tg, np.nan))
    return out

def common_mask(d, keys=('r', 'v', 'vA', 'B', 'ne', 'X', 'MA')):
    """Grid samples where every listed quantity of a track exists.

    r, v and n_e span a lane's whole traced length; X, M_A, v_A and B need both
    branches of the split. Averaging each over its own finite samples puts one
    table row on non-overlapping intervals, and the row then fails its own
    identities.
    """
    ms = [np.isfinite(d[k + '_mean']) for k in keys if k + '_mean' in d]
    return np.logical_and.reduce(ms) if ms else None

def grid_scalar(d, key, mask=None):
    """Grid-average of a quantity and its combined standard error.

    Pass mask whenever the result sits in a row beside other quantities.
    """
    m, se = d[key + '_mean'], d[key + '_se']
    good = np.isfinite(m) if mask is None else (np.isfinite(m) & mask)
    if good.sum() == 0:
        return np.nan, np.nan
    return np.nanmean(m[good]), np.sqrt(np.nanmean(se[good] ** 2))

def lane_windows(d, lab, tgrid, t_ref, roles=None, upstream=None):
    """Every scalar a lane supports, on both of its averaging windows.

    span covers everything the lane traced and carries r, v, a and n_e. split is
    the sub-interval where both branches of the band exist, the only place X and
    M_A are defined and therefore the only place v_A and B mean anything; r, v
    and n_e are repeated there so a row closes its own identities.

    v_A and B come back NaN for a downstream branch: v_A from Rankine-Hugoniot
    is the upstream Alfven speed, so pairing it with the downstream density
    gives B_1 sqrt(X), which is neither branch. upstream defaults to reading
    LANE_ROLE; pass it explicitly to avoid that dependency.
    """
    # state this step works on
    LANE_ROLE = roles or {}

    out = {'lane': lab,
           'upstream': (LANE_ROLE.get(lab, '').startswith('upstream') if upstream is None
                        else bool(upstream))}
    fmt = lambda msk: (
        f'{(t_ref + pd.Timedelta(seconds=float(tgrid[msk].min()))).strftime("%H:%M:%S")}-'
        f'{(t_ref + pd.Timedelta(seconds=float(tgrid[msk].max()))).strftime("%H:%M:%S")} UT')
    span = np.isfinite(d['r_mean']) & np.isfinite(d['v_mean'])
    out['span_mask'] = span
    out['n_span'] = int(span.sum())
    out['span_window'] = fmt(span) if span.any() else ''
    for k in ('r', 'v', 'a', 'ne'):
        out[k + '_span'], out[k + '_span_e'] = grid_scalar(d, k, mask=span if span.any() else None)
    cm = common_mask(d)
    out['split_mask'] = cm
    out['n_split'] = int(cm.sum()) if cm is not None else 0
    out['has_split'] = out['n_split'] > 0
    out['split_window'] = fmt(cm) if out['has_split'] else ''
    for k in ('r', 'v', 'ne', 'X', 'MA', 'vA', 'B'):
        if not out['has_split'] or (k in ('vA', 'B') and not out['upstream']):
            out[k + '_split'], out[k + '_split_e'] = np.nan, np.nan
        else:
            out[k + '_split'], out[k + '_split_e'] = grid_scalar(d, k, mask=cm)
    return out

def aggregate_lanes(run, passes, tg, model, n_mc=N_MC, seed=0, kin_method=None):
    """Monte-Carlo aggregation per track, combining the fit and repeat errors."""
    # state this step works on
    ALL_TRACKS = run.ALL_TRACKS
    REPEATS_INDEPENDENT = run.REPEATS_INDEPENDENT

    rng = np.random.default_rng(seed)
    invert = invert_grid(model)
    fitsets = [pass_fits(run, pas) for pas in passes]
    stacks = {k: {kk: [] for kk in AGG_KEYS} for k in ALL_TRACKS}
    for fits in fitsets:
        for _ in range(n_mc):
            res = realise_lanes(run, fits, tg, invert, rng=rng, sample=True,
                            kin_method=kin_method)
            for key, d in res.items():
                for kk in AGG_KEYS:
                    stacks[key][kk].append(d[kk])
    out = {}
    # Dividing the spread by sqrt(N_pass) claims the precision gain that comes from repeating an
    # independent measurement. Jittered Bezier repeats are not that: they are ONE curve displaced
    # N_REPS times, so averaging them recovers the curve you drew and nothing about how well it
    # sits on the ridge. Only divide when the repeats were genuinely re-traced.
    npass = max(len(passes), 1)
    root_n = np.sqrt(npass) if REPEATS_INDEPENDENT else 1
    for key in ALL_TRACKS:
        if not stacks[key]['r']:
            continue
        out[key] = {}
        for kk in AGG_KEYS:
            M = np.vstack(stacks[key][kk])
            out[key][kk + '_mean'] = np.nanmean(M, axis=0)
            out[key][kk + '_sd'] = np.nanstd(M, axis=0)
            out[key][kk + '_se'] = out[key][kk + '_sd'] / root_n
            # how many realisations actually reached this grid point. The passes have slightly
            # different traced spans, so the first and last points of a track are averaged over a
            # subset of them; that makes the mean there noisy enough to send r backwards by more
            # than its own error bar. Downstream code uses this to drop partly covered samples.
            out[key][kk + '_n'] = np.sum(np.isfinite(M), axis=0)
    return out

def scalar_summary(run, passes):
    """Model-independent scalars per band: drift, relative drift, bandwidth,
    X and M_A.

    The quoted error on X combines the scatter between repeats with how much X
    varies along the overlap. The second term usually dominates, and omitting
    it makes two bands of one shock look inconsistent.
    """
    # state this step works on
    BANDS_TRACED, SPLIT_PAIR = run.BANDS_TRACED, run.SPLIT_PAIR
    LANE_SIGMA = run.LANE_SIGMA

    def mse(arr):
        arr = np.asarray(arr, float)
        if arr.size == 0:
            return (np.nan, np.nan)
        se = np.nanstd(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        return np.nanmean(arr), se

    def combine(per_pass_mean, along_lane_sd):
        """Uncertainty on a quantity that is averaged along the overlap.

        The scatter between passes is NOT the whole error, and with jittered Bezier repeats it is
        not even the larger part of it: it only says how far BEZIER_JITTER moved the curve. X is
        quoted as a single number for the whole burst, so how much it actually varies ALONG the
        overlap belongs in its error bar too - and for a real band split that variation dominates.
        Leaving it out is what made two perfectly compatible X values look 10 sigma apart."""
        m, se = mse(per_pass_mean)
        sd = np.nanmean(along_lane_sd) if len(along_lane_sd) else np.nan
        if not np.isfinite(sd):
            return (m, se)
        return (m, float(np.hypot(se if np.isfinite(se) else 0, sd)))

    rows = {}
    for b in BANDS_TRACED:
        lo_lab, up_lab = SPLIT_PAIR[b]
        Xv, MAv, drift, relbw, reldrift = [], [], [], [], []
        Xsd, Xlo, Xhi, rdsd, MAsd = [], [], [], [], []
        Xbad, Xnear4 = [], []
        for pas in passes:
            fl = lane_fit(run, pas.get(lo_lab), sigma_f=LANE_SIGMA.get(lo_lab))
            fu = (lane_fit(run, pas.get(up_lab), sigma_f=LANE_SIGMA.get(up_lab))
                  if up_lab else None)
            if fl and fu:
                lo, hi = max(fl['tmin'], fu['tmin']), min(fl['tmax'], fu['tmax'])
                if hi > lo:
                    gg = np.linspace(lo, hi, 20)
                    fLv, fUv = lane_deriv(fl, gg)[0], lane_deriv(fu, gg)[0]
                    Xg = (fUv / fLv) ** 2
                    Xv.append(np.nanmean(Xg))
                    Xsd.append(np.nanstd(Xg))
                    Xlo.append(np.nanmin(Xg))
                    Xhi.append(np.nanmax(Xg))
                    # How much of the overlap is unusable? alfven_mach_from_X returns NaN outside
                    # 1 <= X < 4, and every mean below is a nanmean, so a split whose branches
                    # cross part-way silently reports M_A from the half that still works while
                    # the mean X stays close to 1 and looks unremarkable.
                    Xbad.append(float(np.mean(~((Xg >= 1) & (Xg < 4)))))
                    Xnear4.append(float(np.mean(Xg > 3.5)))
                    MAg = alfven_mach_from_X(Xg)
                    MAv.append(np.nanmean(MAg))
                    MAsd.append(np.nanstd(MAg))
                    relbw.append(np.nanmean((fUv - fLv) / fLv))
            if fl:
                gg = np.linspace(fl['tmin'], fl['tmax'], 20)
                _, dv, rd = lane_deriv(fl, gg)
                drift.append(np.nanmean(dv))                                  # MHz/s
                reldrift.append(np.nanmean(rd))                               # 1/s
                rdsd.append(np.nanstd(rd))
        rows[b] = {'X': combine(Xv, Xsd), 'M_A': combine(MAv, MAsd), 'drift_MHz_s': mse(drift),
                   'rel_drift_s': combine(reldrift, rdsd), 'rel_bandwidth': mse(relbw),
                   'X_range': (np.nanmean(Xlo) if Xlo else np.nan,
                               np.nanmean(Xhi) if Xhi else np.nan),
                   'X_pass_se': mse(Xv)[1],
                   'X_frac_invalid': float(np.mean(Xbad)) if Xbad else np.nan,
                   'X_frac_near4': float(np.mean(Xnear4)) if Xnear4 else np.nan}
    return rows

def sample_polarisation(run, t_list, f_list, dt_s=POL_DT_S,
                        df_mhz=POL_DF_MHZ):
    """Median Stokes V/I in a +/-POL_DT_S by +/-POL_DF_MHZ box along a lane."""
    # state this step works on
    POL_T, POL_F, POL_V = run.POL_T, run.POL_F, run.POL_V

    tv = pd.to_datetime(pd.Series(t_list)).to_numpy()
    fv = np.asarray(f_list, float)
    half = np.timedelta64(int(dt_s * 1e3), 'ms')
    out = np.full(len(fv), np.nan)
    for k, (tt, ff) in enumerate(zip(tv, fv)):
        # side='right' on the upper edge closes the interval. With the default the box is
        # [t-dt, t+dt), so it holds one fewer sample on the late side than the early side and its
        # centre of mass sits half a sample early - a systematic offset along a drifting lane,
        # not a rounding detail.
        i0 = np.searchsorted(POL_T, tt - half, side='left')
        i1 = max(np.searchsorted(POL_T, tt + half, side='right'), i0 + 1)
        j = np.abs(POL_F - ff) <= df_mhz
        if j.any():
            blk = POL_V[i0:i1][:, j]
            if blk.size:
                out[k] = np.nanmedian(blk)
    return out

def polarisation_caveats(tab, tol=2e-4):
    """Warnings about a polarisation table that the numbers alone hide.

    Chiefly: two lanes agreeing far more closely than either is
    individually determined, which indicates a common instrumental
    offset rather than lane-specific polarisation.
    """
    notes = []
    v = tab.set_index('lane')['V_over_I_mean']
    for a in v.index:
        for b in v.index:
            if a < b and abs(v[a] - v[b]) < tol:
                notes.append(f'{a} and {b} agree to {abs(v[a] - v[b]):.1e} in signed mean V/I, '
                             'much closer than either is individually determined - consistent '
                             'with a common instrumental offset, not lane-specific polarisation')
    return notes

def rel_drift_on_points(run, lab, lo, hi):
    """Relative drift [1/s] from a log-linear fit to traced samples in
    [lo, hi].

    Avoids evaluating the lane cubics near the ends of their spans, where
    a polynomial derivative is least constrained. The F/H overlap is the
    last third of one lane's fit and the first eighth of the other's.
    Returns (mean slope, standard error, points used, pass-to-pass
    spread); the spread measures the jitter for auto-repeats and is not
    used as the error.
    """
    # state this step works on
    passes, t0, LANE_SIGMA = run.passes, run.t0, run.LANE_SIGMA

    slopes, ses, npts = [], [], 0
    for pas in passes:
        p = pas.get(lab)
        if not p:
            continue
        tt = (pd.to_datetime(p['t']) - t0).total_seconds().to_numpy()
        ff = np.asarray(p['f'], float)
        m = (tt >= lo) & (tt <= hi) & np.isfinite(ff) & (ff > 0)
        if m.sum() < 3:
            continue
        x, y = tt[m], np.log(ff[m])
        sig = LANE_SIGMA.get(lab)
        w = (ff[m] / sig) if sig else None               # d(ln f) = df / f
        try:
            c, cv = np.polyfit(x, y, 1, w=w, cov='unscaled') if w is not None \
                else np.polyfit(x, y, 1, cov=True)
            ses.append(float(np.sqrt(cv[0, 0])))
        except (ValueError, np.linalg.LinAlgError):
            c = np.polyfit(x, y, 1)
            ses.append(np.nan)
        slopes.append(float(c[0]))
        npts = int(m.sum())
    if not slopes:
        return np.nan, np.nan, 0, np.nan
    se = float(np.sqrt(np.nanmean(np.asarray(ses) ** 2))) if np.any(np.isfinite(ses)) else np.nan
    spread = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0
    return float(np.mean(slopes)), se, npts, spread

def band_height(run, fit, band, ts):
    """Frequency and heliocentric height of a lane fit at ts, for the given band."""
    # state this step works on
    _rr, _ne = run._INV_REF

    f = lane_deriv(fit, ts)[0]
    ne_t = freq_to_density(f * 1e6, harmonic=HARM[band])
    r = np.interp(ne_t, _ne[::-1], _rr[::-1])
    r[(ne_t > np.nanmax(_ne)) | (ne_t < np.nanmin(_ne))] = np.nan
    return f, r

def compare(a, ea, b, eb):
    """Difference, combined error and significance of two measurements."""
    d = a - b
    ed = np.hypot(ea if np.isfinite(ea) else 0, eb if np.isfinite(eb) else 0)
    return d, ed, (abs(d) / ed if ed > 0 else np.inf)
