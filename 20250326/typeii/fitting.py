"""Lane fits, their derivatives, and height-time kinematics.

A lane is a low-order polynomial in log10 f against time; heights follow from the
density inversion and kinematics from differentiating r(t). METHODS.md sections 4
and 6.
"""
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter

from .config import (FIT_CLIP_TO_TRACED_F, FIT_F_TOL_MHZ, FIT_IN_LOGF, FIT_MIN_DT_S,
                     HARM, KIN_A_SIGMA, KIN_DEG, KIN_MIN_FRAC, KIN_MIN_PTS,
                     KIN_N_DENSE, LANE_DEG, LANE_SIGMA_FLOOR, N_GRID, PLASMA_CONST,
                     POLY_DEG, R_SUN_M)
from .models import freq_to_density


def thin_samples(ts, min_dt=FIT_MIN_DT_S):
    """Indices of one sample per min_dt seconds.

    Samples along a Bezier are not independent measurements, and feeding all of them to
    a fit makes its covariance far too small.
    """
    ts = np.asarray(ts, float)
    keep, last = [], -np.inf
    for i in np.argsort(ts, kind='stable'):
        if ts[i] - last >= min_dt:
            keep.append(int(i))
            last = ts[i]
    return np.asarray(sorted(keep), int)

def lane_fit(run, lane_data, deg=LANE_DEG, sigma_f=None):
    """Fit a lane as a polynomial in log10 f against time.

    sigma_f is the assumed 1-sigma frequency uncertainty per point in MHz. When
    given, the fit is weighted by it and the covariance returned unscaled, so it
    reflects that uncertainty rather than the residual scatter of a smooth curve.
    Returns a dict of coefficients, covariance, span and traced frequency range.
    """
    # state this step works on
    t0 = run.t0

    if not lane_data or len(lane_data['f']) < 2:
        return None
    t = pd.to_datetime(lane_data['t'])
    f = np.asarray(lane_data['f'], float)
    ts = (t - t0).total_seconds().to_numpy()
    o = np.argsort(ts)
    ts, f = ts[o], f[o]
    tmin, tmax = ts.min(), ts.max()
    k = thin_samples(ts)
    if len(k) >= deg + 2:
        ts, f = ts[k], f[k]
    y = np.log10(f) if FIT_IN_LOGF else f
    sy = None
    if sigma_f:
        # d(log10 f) = df / (f ln10)
        sy = (sigma_f / (f * np.log(10))) if FIT_IN_LOGF else np.full_like(f, float(sigma_f))
    d = min(deg, len(ts) - 1)
    cov = None
    try:
        if sy is not None and np.all(np.isfinite(sy)) and np.all(sy > 0) and len(ts) > d + 1:
            p, cov = np.polyfit(ts, y, d, w=1 / sy, cov='unscaled')
        elif len(ts) >= d + 2:
            p, cov = np.polyfit(ts, y, d, cov=True)
        else:
            p = np.polyfit(ts, y, d)
    except (ValueError, np.linalg.LinAlgError):
        p = np.polyfit(ts, y, d)
    return {'p': p, 'cov': cov, 'tmin': tmin, 'tmax': tmax, 'n_fit': len(ts),
            'sigma_f': sigma_f, 'fmin': float(np.nanmin(f)), 'fmax': float(np.nanmax(f))}

def lane_coeffs(fit, rng=None, sample=False):
    """Fit coefficients, or one multivariate-normal draw from their covariance."""
    if fit is None:
        return None
    if sample and rng is not None and fit['cov'] is not None:
        try:
            return rng.multivariate_normal(fit['p'], fit['cov'])
        except np.linalg.LinAlgError:
            return fit['p']
    return fit['p']

def eval_lane(fit, coeffs, tg):
    """Lane frequency [MHz] on a time grid, blanked outside the traced span.

    Blanked in frequency as well as time: a polynomial evaluated at the edge of its
    span, or a Monte-Carlo draw from it, can leave the frequency range actually traced,
    and those samples are extrapolation. FIT_F_TOL_MHZ keeps a legitimate endpoint.
    """
    if fit is None or coeffs is None:
        return np.full_like(tg, np.nan)
    y = np.polyval(coeffs, tg)
    f = 10 ** y if FIT_IN_LOGF else y
    f[(tg < fit['tmin']) | (tg > fit['tmax'])] = np.nan
    if 'fmin' in fit and FIT_CLIP_TO_TRACED_F:
        tol = FIT_F_TOL_MHZ
        f[(f < fit['fmin'] - tol) | (f > fit['fmax'] + tol)] = np.nan
    return f

def lane_deriv(fit, ts):
    """Frequency, drift rate [MHz/s] and relative drift [1/s] of a lane fit at ts.

    With a log fit, df/dt = f ln10 dlog10f/dt, and the relative drift is
    ln10 dlog10f/dt, which is free of the harmonic number.
    """
    dy = np.polyval(np.polyder(fit['p']), ts)
    if FIT_IN_LOGF:
        f = 10 ** np.polyval(fit['p'], ts)
        return f, f * np.log(10) * dy, np.log(10) * dy
    f = np.polyval(fit['p'], ts)
    return f, dy, dy / f

def pass_fits(run, pas):
    """Lane fits for every traced lane of one pass."""
    # state this step works on
    TRACED, LANE_SIGMA = run.TRACED, run.LANE_SIGMA

    return {lab: lane_fit(run, pas.get(lab), sigma_f=LANE_SIGMA.get(lab)) for lab in TRACED}

def lane_fit_quality(run, lane_data, fit):
    """Does the polynomial actually describe the traced curve?

    Reports the rms distance between fit and points, and the time at which
    the fitted drift changes sign inside the traced span, if it does. Every
    downstream quantity comes from the fit and never looks back at the
    points, so neither failure shows up anywhere else.
    """
    # state this step works on
    t0 = run.t0

    if fit is None or not lane_data or len(lane_data['f']) < 3:
        return {'resid_MHz': np.nan, 'resid_over_sigma': np.nan, 'turnover_s': np.nan}
    t = pd.to_datetime(lane_data['t'])
    f = np.asarray(lane_data['f'], float)
    ts = (t - t0).total_seconds().to_numpy()
    o = np.argsort(ts)
    ts, f = ts[o], f[o]
    resid = float(np.sqrt(np.nanmean((f - eval_lane(fit, fit['p'], ts)) ** 2)))
    sig = fit.get('sigma_f') or LANE_SIGMA_FLOOR
    gg = np.linspace(fit['tmin'], fit['tmax'], 200)
    dv = lane_deriv(fit, gg)[1]
    sign_flip = np.flatnonzero(np.diff(np.sign(dv[np.isfinite(dv)])) != 0)
    return {'resid_MHz': resid, 'resid_over_sigma': resid / sig,
            'turnover_s': float(gg[sign_flip[0]]) if len(sign_flip) else np.nan}

def sg_smooth(y, window=9, poly=2):
    """Savitzky-Golay smoothing that fills interior gaps only.

    Returns NaN outside the first and last finite sample rather than clamping to
    the end values.
    """
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 5:
        return y
    idx = np.flatnonzero(m)
    inside = np.arange(idx[0], idx[-1] + 1)
    yy = np.interp(inside, idx, y[m])
    w = min(window, len(inside))
    if w % 2 == 0:
        w -= 1
    out = np.full_like(y, np.nan)
    out[inside] = yy if w < poly + 2 else savgol_filter(yy, w, poly)
    return out

def kin_degree(npts, deg=None):
    """Degree of the r(t) fit a lane can support, from its point count alone.

    A degree-d polynomial needs d + 2 points before its curvature means anything.
    There is deliberately no cut on traced duration: whether a curvature counts as
    measured is decided afterwards by accel_is_measured.
    """
    deg = KIN_DEG if deg is None else deg
    return max(1, min(deg, int(npts) - 2))

def accel_is_measured(a, a_err, n_sigma=KIN_A_SIGMA, bias=0):
    """True when |a| clears both its own error bar and the method bias floor."""
    if not (np.isfinite(a) and np.isfinite(a_err)):
        return False
    if a_err <= 0 or abs(a) < n_sigma * a_err:
        return False
    return not (np.isfinite(bias) and abs(a) < abs(bias))

def accel_bias_floor(run, fit, band, invert, v_ref=None):
    """Acceleration this chain returns for a shock at exactly constant speed.

    A property of the method that no error bar reveals, evaluated on the
    lane's own span and measured speed.
    """
    # state this step works on
    t0 = run.t0

    if fit is None:
        return np.nan
    rr, ne = invert
    tt = np.linspace(fit['tmin'], fit['tmax'], KIN_N_DENSE)
    # Use the speed this lane actually shows, not a fixed 600 km/s. A synthetic shock moving at
    # the wrong speed sweeps a different height range over the same time and returns a bias that
    # is not commensurate with the acceleration it is meant to be compared against - on the slower
    # tracks here that was a factor of two in the swept range.
    if v_ref is None:
        _f0v = eval_lane(fit, fit['p'], tt)
        _r0v = np.interp(freq_to_density(_f0v * 1e6, harmonic=HARM[band]), ne[::-1], rr[::-1])
        _okv = np.isfinite(_r0v)
        v_ref = (abs(np.nanmax(_r0v[_okv]) - np.nanmin(_r0v[_okv])) * R_SUN_M / 1e3
                 / max(tt.max() - tt.min(), 1)) if _okv.sum() > 2 else 600
        v_ref = float(np.clip(v_ref, 50, 3000))
    f_lane = eval_lane(fit, fit['p'], tt)
    ne_lane = freq_to_density(f_lane * 1e6, harmonic=HARM[band])
    r_lane = np.interp(ne_lane, ne[::-1], rr[::-1])
    if not np.isfinite(r_lane).any():
        return np.nan
    # a straight line through the same heights, over the same time, at constant speed
    r_lin = np.nanmin(r_lane) + (v_ref * 1e3 / R_SUN_M) * (tt - tt.min())
    ok = (r_lin >= np.nanmin(rr)) & (r_lin <= np.nanmax(rr))
    if ok.sum() < KIN_MIN_PTS + 2:
        return np.nan
    f_syn = PLASMA_CONST * np.sqrt(np.interp(r_lin[ok], rr, ne)) * HARM[band] / 1e6
    syn = {'t': [t0 + pd.Timedelta(seconds=float(x)) for x in tt[ok]], 'f': list(f_syn)}
    fs = lane_fit(run, syn, sigma_f=fit.get('sigma_f'))
    if fs is None:
        return np.nan
    td = np.linspace(fs['tmin'], fs['tmax'], KIN_N_DENSE)
    tgs = np.linspace(fs['tmin'], fs['tmax'], N_GRID)
    to_r = lambda x: np.interp(freq_to_density(eval_lane(fs, fs['p'], x) * 1e6, harmonic=HARM[band]),
                               ne[::-1], rr[::-1])
    return float(np.nanmean(kinematics(to_r(tgs), tgs, t_fit=td, r_fit=to_r(td))[1]))

def kinematics(r, tg, deg=None, span=None, baseline=None, t_fit=None, r_fit=None,
               method=None):
    """Speed [km/s] and acceleration [m/s^2] from a height track r(t) [Rsun].

    method picks how r(t) is differentiated. None fits a polynomial of degree KIN_DEG, which is
    the default everywhere. A key of FIT_METHODS ("Polynomial", "Gallagher (2003)",
    "Byrne (2013)") uses that model instead, so the speed and acceleration panels can be built
    with any of the three.

    Blanked where r is NaN, and where the speed is inward or above 3000 km/s.
    """
    v = np.full_like(tg, np.nan)
    a = np.full_like(tg, np.nan)
    good = np.isfinite(r)
    span = good if span is None else np.asarray(span, bool)
    # Fit over the lane's own dense samples when they are supplied. r(t) comes from an analytic
    # lane fit, so it can be sampled as finely as the polynomial needs; taking the fit points off
    # the shared grid instead leaves a short lane with a handful of them and makes the curvature
    # far worse determined than the data warrant.
    if t_fit is not None and r_fit is not None:
        tf, rf = np.asarray(t_fit, float), np.asarray(r_fit, float)
        okf = np.isfinite(rf)
        tf, rf = tf[okf], rf[okf]
    else:
        tf, rf = tg[good], r[good]
    deg = kin_degree(len(tf), deg)
    n_span = max(int(span.sum()), 1)
    # Count the points the fit will ACTUALLY use. Counting the shared grid's points instead
    # discards any lane shorter than about three grid steps - it has fewer than KIN_MIN_PTS of
    # them - even though the dense per-lane samples are there and perfectly well behaved. That
    # loses the speed as well as the acceleration, for a lane that was traced without complaint.
    n_used = len(tf)
    frac = (good.sum() / n_span) if (t_fit is None or r_fit is None) else 1
    # measure the height range on the same points the fit uses, for the same reason
    r_range = (np.nanmax(rf) - np.nanmin(rf)) if n_used > 3 else 0
    if (n_used >= KIN_MIN_PTS and frac >= KIN_MIN_FRAC
            and r_range > 0.01 and n_used > deg + 1 and good.any()):
        if method:
            # The published height-time models work in km, and their splines are only defined
            # over the fitted span, so evaluate on a clipped grid and blank outside it below.
            fit = FIT_METHODS[method](tf, rf * (R_SUN_M / 1e3), None)
            tc = np.clip(tg, tf.min(), tf.max())
            v = np.asarray(fit['v'](tc), float)
            a = np.asarray(fit['a'](tc), float)
        else:
            dg = deg
            pr = np.polyfit(tf, rf, dg)
            v = np.polyval(np.polyder(pr, 1), tg) * R_SUN_M / 1e3
            # A straight line has no second derivative to report. Returning the 0.0 that polyder
            # hands back would put a hard "a = 0.0 +/- 0.0" in the tables, which reads as a
            # measured null result instead of the absence of a measurement. NaN says it properly.
            a = (np.polyval(np.polyder(pr, 2), tg) * R_SUN_M if dg >= 2
                 else np.full_like(tg, np.nan))
        # blank the derivatives outside the span this track was traced over: each band covers a
        # different part of the burst, and a cubic extrapolated past its data runs away fast
        v[~good] = np.nan
        a[~good] = np.nan
    return np.where((v > 0) & (v < 3000), v, np.nan), a

def build_grid(run, passes, n=N_GRID):
    """Shared time grid [s from t0] spanning every traced lane."""
    # state this step works on
    t0, TRACED = run.t0, run.TRACED

    secs = []
    for pas in passes:
        for lab in TRACED:
            if pas.get(lab) and pas[lab]['f']:
                secs += list((pd.to_datetime(pas[lab]['t']) - t0).total_seconds())
    if not secs:
        raise RuntimeError('no traced points to build a time grid from')
    return np.linspace(np.nanmin(secs), np.nanmax(secs), n)

def fit_polynomial(t, h, sig):
    """Height-time fit: plain polynomial in t, with analytic v and a."""
    p = np.polyfit(t, h, POLY_DEG, w=(1 / sig if sig is not None else None))
    dp, ddp = np.polyder(p, 1), np.polyder(p, 2)
    return {'h': lambda tt: np.polyval(p, tt),
            'v': lambda tt: np.polyval(dp, tt),               # km/s
            'a': lambda tt: np.polyval(ddp, tt) * 1000}

def gallagher_accel(t, ar, ad, tr, td):
    """Gallagher et al. (2003) reciprocal-sum acceleration profile."""
    return 1 / (1 / (ar * np.exp(t / tr)) + 1 / (ad * np.exp(-t / td)))

def fit_gallagher(t, h, sig):
    """Height-time fit: Gallagher et al. (2003).

    Their a(t) is strictly positive for the required positive amplitudes, so
    this model cannot represent a decelerating shock. Callers must check the
    applicability flag rather than treating it as one of three equal options.
    """
    tgrid = np.linspace(t.min(), t.max(), 400)
    def model(tt, ar, ad, tr, td, h0, v0):
        acc = gallagher_accel(tgrid, ar, ad, tr, td)
        vv = v0 + cumulative_trapezoid(acc, tgrid, initial=0)
        hh = h0 + cumulative_trapezoid(vv, tgrid, initial=0)
        return np.interp(tt, tgrid, hh)
    v0g = (h[-1] - h[0]) / (t[-1] - t[0])
    p0 = [1e-3, 1e-3, 150, 150, h[0], v0g]
    lo = [1e-6, 1e-6, 20, 20, h[0] - 5e4, -2000]
    hi = [1e2, 1e2, 5e3, 5e3, h[0] + 5e4, 3000]
    popt, _ = curve_fit(model, t, h, p0=p0, sigma=sig, absolute_sigma=False,
                        bounds=(lo, hi), maxfev=200000)
    acc = gallagher_accel(tgrid, *popt[:4])
    vv = popt[5] + cumulative_trapezoid(acc, tgrid, initial=0)
    hh = popt[4] + cumulative_trapezoid(vv, tgrid, initial=0)
    return {'h': CubicSpline(tgrid, hh), 'v': CubicSpline(tgrid, vv),
            'a': lambda tt, _g=tgrid, _a=acc: np.interp(tt, _g, _a) * 1000}

def fit_byrne(t, h, sig):
    """Height-time fit: Byrne et al. (2013) Savitzky-Golay, derivatives from the filter."""
    tu = np.linspace(t.min(), t.max(), max(len(t), 21))
    hu = np.interp(tu, t, h)
    dt = tu[1] - tu[0]
    n = len(tu)
    win = min(11, n if n % 2 == 1 else n - 1)
    if win <= POLY_DEG + 1:
        win = POLY_DEG + 3 if (POLY_DEG + 3) % 2 == 1 else POLY_DEG + 2
    hs = savgol_filter(hu, win, POLY_DEG)
    vs = savgol_filter(hu, win, POLY_DEG, deriv=1, delta=dt)
    ac = savgol_filter(hu, win, POLY_DEG, deriv=2, delta=dt)
    return {'h': CubicSpline(tu, hs), 'v': CubicSpline(tu, vs),
            'a': lambda tt, _c=CubicSpline(tu, ac): _c(tt) * 1000}

def hva(fit, t):
    """Height, speed and acceleration from a fitted model on a dense time grid."""
    return fit['h'](t), fit['v'](t), fit['a'](t)


# The three height-time methods, and the colour each is drawn in. A registry rather than per-run
# state: adding a method here is all it takes for the comparison in A.7 to pick it up.
FIT_METHODS = {'Polynomial': fit_polynomial,
               'Gallagher (2003)': fit_gallagher,
               'Byrne (2013)': fit_byrne}
FIT_COLOR = {'Polynomial': 'tab:blue',
             'Gallagher (2003)': 'tab:green',
             'Byrne (2013)': 'tab:red'}

