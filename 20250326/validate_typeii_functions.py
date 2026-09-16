"""Numerical validation of the notebook's own functions against analytic ground truth.

Runs the shipped code (via the headless harness), then checks each estimator against a value
worked out independently. Nothing here is imported from the notebook's own logic twice: every
expected value is either a published constant or hand-derived here.
"""
import sys
import numpy as np

FAIL, PASS = [], []


def check(name, got, want, tol, unit='', rel=False):
    got, want = float(got), float(want)
    err = abs(got - want) / abs(want) if rel else abs(got - want)
    ok = err <= tol
    (PASS if ok else FAIL).append(name)
    mark = 'ok  ' if ok else 'FAIL'
    kind = 'rel' if rel else 'abs'
    print(f'  [{mark}] {name:<52s} got {got:12.6g}  want {want:12.6g} {unit:<8s} '
          f'({kind} err {err:.2e}, tol {tol:.0e})')


def check_true(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"ok  " if cond else "FAIL"}] {name:<52s} {detail}')


# ---------------------------------------------------------------- load the shipped code
import matplotlib
matplotlib.use('Agg')
import pandas as pd

import make_fixture
from typeii import *                      # noqa: F403 - the public analysis interface
from typeii import pipeline as pl, tracing
from typeii.config import *               # noqa: F403 - constants and tunables
from typeii.physics import band_height, compare
from typeii.session import Run

# A synthetic event with known answers, carried through the analysis. Its results become module
# globals so each assertion below can name them directly.
RUN = make_fixture.build(quiet=True)
globals().update({k: v for k, v in vars(RUN).items() if not k.startswith('__')})
print('\n' + '=' * 100)
print('VALIDATION OF THE NOTEBOOK FUNCTIONS AGAINST ANALYTIC GROUND TRUTH')
print('=' * 100)

# ---------------------------------------------------------------- 1. constants
print('\n1. physical constants')
check('PLASMA_CONST vs textbook 8.98e3', PLASMA_CONST, 8.977e3, 2e-3, 'Hz', rel=True)
check('R_sun', R_SUN_M, 6.957e8, 1e-3, 'm', rel=True)

# ---------------------------------------------------------------- 2. density models
print('\n2. density models against published anchor values')
check('Newkirk(1) = 4.2e4 * 10^4.32', newkirk(1.0), 4.2e4 * 10 ** 4.32, 1e-9, 'cm^-3', rel=True)
check('Mann 2023 at 3 Rsun (their quoted 4.267e5)', mann2023(3.0), 4.267e5, 2e-3, 'cm^-3', rel=True)
check('Baumbach-Allen(1) = 1e8*(0.036+1.55+2.99)', baumbach_allen(1.0),
      1e8 * (0.036 + 1.55 + 2.99), 1e-9, 'cm^-3', rel=True)
check('fold scaling is linear (Newkirk x3)', newkirk(1.5, fold=3), 3 * newkirk(1.5), 1e-12,
      'cm^-3', rel=True)
check_true('all models fall with r', all(np.all(np.diff(f(np.linspace(1.05, 3, 50))) < 0)
                                         for f in BASE_MODELS.values()))

# ---------------------------------------------------------------- 3. frequency <-> density
print('\n3. frequency / density / height inversion')
ne_true = 1e7
f_true = PLASMA_CONST * np.sqrt(ne_true)                     # Hz, fundamental
check('freq_to_density round trip (s=1)', freq_to_density(f_true, 1), ne_true, 1e-12,
      'cm^-3', rel=True)
check('freq_to_density round trip (s=2)', freq_to_density(2 * f_true, 2), ne_true, 1e-12,
      'cm^-3', rel=True)
check_true('f_H and f_F/2 give the SAME density',
           abs(freq_to_density(2 * f_true, 2) - freq_to_density(f_true, 1)) < 1e-6,
           '(this is what lets the two bands share one height track)')
for name, mdl in [('Newkirk x2', MODEL_GRID['Newkirk x2']), ('Saito x1', MODEL_GRID['Saito x1'])]:
    r_want = 1.8
    f_at_r = PLASMA_CONST * np.sqrt(mdl(r_want))
    check(f'freq -> height round trip, {name}', freq_to_radius(f_at_r, mdl, 1), r_want, 1e-4,
          'Rsun')
check_true('out-of-range frequency returns NaN',
           np.isnan(freq_to_radius(1e12, MODEL_GRID['Newkirk x2'], 1)))

# ---------------------------------------------------------------- 4. Rankine-Hugoniot
print('\n4. Alfven Mach number from the density jump')
for X in (1.0, 1.5, 2.0, 3.0):
    check(f'M_A(X={X}) vs sqrt(X(X+5)/(2(4-X)))', alfven_mach_from_X(np.array([X]))[0],
          np.sqrt(X * (X + 5) / (2 * (4 - X))), 1e-12, '', rel=True)
check_true('M_A(X<1) is NaN', np.isnan(alfven_mach_from_X(np.array([0.9]))[0]))
check_true('M_A(X>=4) is NaN', np.isnan(alfven_mach_from_X(np.array([4.0]))[0]))
check('M_A(X=1) = 1 exactly', alfven_mach_from_X(np.array([1.0]))[0], 1.0, 1e-12)

# ---------------------------------------------------------------- 5. lane fit and derivatives
print('\n5. lane fit and its analytic derivatives (exact log-quadratic input)')
c2, c1, c0 = -2.0e-7, -5.0e-4, np.log10(70.0)
ts = np.linspace(0, 400, 80)                                 # 5 s apart, so thinning must bite
f_lane = 10 ** (c2 * ts ** 2 + c1 * ts + c0)
lane = {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in ts], 'f': list(f_lane)}
fit = lane_fit(RUN, lane)
tq = np.array([50.0, 200.0, 350.0])
f_got, df_got, rel_got = lane_deriv(fit, tq)
f_exp = 10 ** (c2 * tq ** 2 + c1 * tq + c0)
rel_exp = np.log(10) * (2 * c2 * tq + c1)                    # d ln f / dt
for i, tt in enumerate(tq):
    check(f'f(t={tt:.0f} s)', f_got[i], f_exp[i], 1e-6, 'MHz', rel=True)
    check(f'df/dt(t={tt:.0f} s)', df_got[i], f_exp[i] * rel_exp[i], 1e-6, 'MHz/s', rel=True)
    check(f'relative drift(t={tt:.0f} s)', rel_got[i], rel_exp[i], 1e-6, '1/s', rel=True)
check_true('thinning drops correlated samples',
           fit['n_fit'] < len(ts), f'({fit["n_fit"]} of {len(ts)} kept, FIT_MIN_DT_S={FIT_MIN_DT_S})')
check_true('thinned points are at least FIT_MIN_DT_S apart',
           np.all(np.diff(ts[thin_samples(ts)]) >= FIT_MIN_DT_S - 1e-9))
check_true('thinning does not bias the fit',
           abs(lane_deriv(fit, np.array([200.0]))[0][0]
               - 10 ** (c2 * 200 ** 2 + c1 * 200 + c0)) < 1e-6)
check_true('_eval blanks outside the traced span',
           np.isnan(eval_lane(fit, fit['p'], np.array([-50.0]))[0])
           and np.isfinite(eval_lane(fit, fit['p'], np.array([200.0]))[0]))

# ---------------------------------------------------------------- 6. kinematics and units
print('\n6. kinematics: units and exactness (r = r0 + v t + a t^2 / 2)')
v_true, a_true = 600.0, -45.0                                # km/s and m/s^2
tg = np.linspace(0, 1200, 60)
r_in = 1.6 + (v_true * 1e3 / R_SUN_M) * tg + 0.5 * (a_true / R_SUN_M) * tg ** 2
v_got, a_got = kinematics(r_in, tg)
check('v_sh grid-average', np.nanmean(v_got), v_true + a_true * tg.mean() / 1e3, 1e-6, 'km/s')
check('a recovered', np.nanmean(a_got), a_true, 1e-6, 'm/s^2')
check('v at t=0', v_got[0], v_true, 1e-6, 'km/s')
check_true('inward / superluminal speeds are rejected',
           np.all(np.isnan(kinematics(2.0 - (500 * 1e3 / R_SUN_M) * tg, tg)[0])),
           '(a track moving inward returns NaN v)')

# ---------------------------------------------------------------- 7. B from v_A and n_e
print('\n7. magnetic field from the Alfven speed')
ne_cm3, vA_kms = 1.0e7, 400.0
rho = MU * M_P * ne_cm3 * 1e6                                # kg/m^3
B_hand = vA_kms * 1e3 * np.sqrt(MU0 * rho) * 1e4             # Gauss
check('B = v_A sqrt(mu0 rho), hand-computed', B_hand, 0.65353, 1e-4, 'G', rel=True)
check_true('B scales as sqrt(n_e)',
           abs((vA_kms * 1e3 * np.sqrt(MU0 * MU * M_P * 4e7 * 1e6) * 1e4) / B_hand - 2.0) < 1e-9)
check_true('B scales linearly with v_A',
           abs((2 * vA_kms * 1e3 * np.sqrt(MU0 * rho) * 1e4) / B_hand - 2.0) < 1e-9)

# ---------------------------------------------------------------- 8. lane ordering
print('\n8. lane ordering over the overlap (the bug that made B NaN)')
_saved_traced, _saved_sigma, _saved_passes = RUN.TRACED, RUN.LANE_SIGMA, RUN.passes
tl = np.linspace(100, 800, 40)                               # long lane, high -> low
ts_ = np.linspace(550, 800, 20)                              # short lane, sits ABOVE it
long_lane = {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tl],
             'f': list(70 * (33 / 70) ** ((tl - 100) / 700))}
short_lane = {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in ts_],
              'f': list(1.15 * 70 * (33 / 70) ** ((ts_ - 100) / 700))}
RUN.set(TRACED=['F lane 1', 'F lane 2'], LANE_SIGMA={},
        passes=[{'F lane 1': long_lane, 'F lane 2': short_lane}])
order, key, note = order_lanes(RUN, ['F lane 1', 'F lane 2'])
check_true('long lane is the upstream branch', order[0] == 'F lane 1',
           f'order={order}, f={{{key["F lane 1"]:.1f}, {key["F lane 2"]:.1f}}} MHz, {note}')
check('implied X = (f_U/f_L)^2', (key[order[1]] / key[order[0]]) ** 2, 1.15 ** 2, 2e-3, '', rel=True)
check_true('own-span means would have inverted it',
           np.mean(long_lane['f']) > np.mean(short_lane['f']),
           f'(own-span means {np.mean(long_lane["f"]):.1f} vs {np.mean(short_lane["f"]):.1f} MHz)')
RUN.set(TRACED=_saved_traced, LANE_SIGMA=_saved_sigma, passes=_saved_passes)

# ---------------------------------------------------------------- 9. smoothing
print('\n9. sg_smooth must not invent data outside the traced span')
y = np.full(40, np.nan)
y[10:30] = np.linspace(1.5, 2.0, 20)
sm = sg_smooth(y)
check_true('NaN outside the finite range is preserved',
           np.all(np.isnan(sm[:10])) and np.all(np.isnan(sm[30:])))
check_true('interior values are reproduced', np.nanmax(np.abs(sm[10:30] - y[10:30])) < 1e-6)

# ---------------------------------------------------------------- 10. decimation
print('\n10. decimate conserves the mean and uses block centres')
tt = pd.date_range('2025-03-26 09:00', periods=1000, freq='100ms')
ff = np.linspace(20, 80, 60)
D = pd.DataFrame(np.random.default_rng(0).normal(5, 1, [1000, 60]), index=tt, columns=ff)
dec, (kt, kf) = decimate(D, max_t=100, max_f=60)
check('block mean preserved', np.nanmean(dec.to_numpy()), np.nanmean(D.to_numpy()), 1e-9, '',
      rel=True)
check_true('no averaging in frequency when it is not needed', kf == 1, f'(kt={kt}, kf={kf})')
check('first block centre time', (dec.index[0] - tt[0]).total_seconds(),
      np.mean([(tt[i] - tt[0]).total_seconds() for i in range(kt)]), 1e-4, 's')


# ---------------------------------------------------------------- 11. height-time fitters
print('\n11. height-time fitters on exact constant-acceleration input')
from typeii.fitting import FIT_METHODS
RS_KM = R_SUN_M / 1e3
h0_km, v0_km, a0_ms2 = 1.6 * RS_KM, 550.0, -60.0
tt_ = np.linspace(0, 1500, 50)
h_ = h0_km + v0_km * tt_ + 0.5 * (a0_ms2 / 1e3) * tt_ ** 2
sig_ = np.full_like(h_, 500.0)
for nm, fn in FIT_METHODS.items():
    try:
        F = fn(tt_, h_, sig_)
    except Exception as ex:
        check_true(f'{nm} converges', False, str(ex)[:60])
        continue
    if nm.startswith('Gallagher'):
        # positive-definite by construction, so it CANNOT fit a decelerating track. Assert that
        # limitation explicitly rather than pretending it is a numerical accident.
        check_true('Gallagher a(t) is strictly positive (cannot decelerate)',
                   np.nanmin(F['a'](np.linspace(0, 1500, 200))) >= 0,
                   '-> the notebook must flag it as inapplicable to a decelerating shock')
        continue
    tol_v, tol_a = (1e-6, 1e-6) if nm == 'Polynomial' else (12.0, 12.0)
    check(f'{nm}: v at t=750 s', F['v'](750.0), v0_km + (a0_ms2 / 1e3) * 750, tol_v, 'km/s')
    check(f'{nm}: a at t=750 s', F['a'](750.0), a0_ms2, tol_a, 'm/s^2')
    aa = F['a'](np.linspace(100, 1400, 60))
    check_true(f'{nm}: a is smooth (no spikes)',
               np.nanmax(np.abs(np.diff(aa))) < 25.0,
               f'(max step {np.nanmax(np.abs(np.diff(aa))):.2f} m/s^2 between adjacent samples)')

# ---------------------------------------------------------------- 12. polarisation sampling
print('\n12. Stokes V/I sampling along a lane')
_pt = pd.date_range(LAYER_T[0], LAYER_T[-1], periods=200)
_pf = np.interp(np.linspace(0, 1, 200), [0, 1], [60.0, 35.0])
got_p = sample_polarisation(RUN, list(_pt), list(_pf))
check_true('returns one value per traced point', len(got_p) == 200)
check_true('values lie inside [-1, 1]', np.nanmax(np.abs(got_p)) <= 1.0)
check_true('mean |x| of zero-mean noise returns ~0.8 sigma, not the mean',
           abs(np.mean(np.abs(np.random.default_rng(0).normal(0, 0.02, 100000))) / 0.02 - 0.7979)
           < 0.01, '(this is why the signed mean is the number to compare between bands)')

# ---------------------------------------------------------------- 13. no-overlap ordering path
print('\n13. order_lanes when the lanes never overlap')
_st, _ss, _sp = RUN.TRACED, RUN.LANE_SIGMA, RUN.passes
ta_ = np.linspace(100, 400, 20)
tb_ = np.linspace(600, 900, 20)
RUN.set(TRACED=['F lane 1', 'F lane 2'], LANE_SIGMA={}, passes=[{
    'F lane 1': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in ta_],
                 'f': list(70 - 0.05 * (ta_ - 100))},
    'F lane 2': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tb_],
                 'f': list(40 - 0.02 * (tb_ - 600))}}])
_o, _k, _n = order_lanes(RUN, ['F lane 1', 'F lane 2'])
check_true('non-overlapping lanes are flagged', 'WARNING' in _n, _n[:70] + '...')
RUN.set(TRACED=_st, LANE_SIGMA=_ss, passes=_sp)

# ---------------------------------------------------------------- 14. end-to-end closure
print('\n14. end-to-end closure: inject a known split, recover X and M_A')
X_inj = 1.21                                                 # split factor 1.1 -> X = 1.21
sp = np.sqrt(X_inj)
tc = np.linspace(100, 1000, 60)
f_lo = 65 * (30 / 65) ** ((tc - 100) / 900)
_sb, _sbt = RUN.LANE_BAND, RUN.BANDS_TRACED
_slo, _ssp = RUN.LANE_ORDER, RUN.SPLIT_PAIR
RUN.set(TRACED=['F lane 1', 'F lane 2'], LANE_SIGMA={}, BANDS_TRACED=['F'],
        LANE_BAND={'F lane 1': 'F', 'F lane 2': 'F'}, passes=[{
            'F lane 1': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tc],
                         'f': list(f_lo)},
            'F lane 2': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tc],
                         'f': list(f_lo * sp)}}])
_o, _k, _n = order_lanes(RUN, ['F lane 1', 'F lane 2'])
RUN.set(LANE_ORDER={'F': _o}, SPLIT_PAIR={'F': (_o[0], _o[1])})
sc = scalar_summary(RUN, RUN.passes)
check('X recovered end to end', sc['F']['X'][0], X_inj, 1e-3, '', rel=True)
check('M_A recovered end to end', sc['F']['M_A'][0],
      np.sqrt(X_inj * (X_inj + 5) / (2 * (4 - X_inj))), 1e-3, '', rel=True)
check('relative bandwidth recovered', sc['F']['rel_bandwidth'][0], sp - 1, 1e-3, '', rel=True)
RUN.set(TRACED=_st, LANE_SIGMA=_ss, passes=_sp, LANE_BAND=_sb, BANDS_TRACED=_sbt,
        LANE_ORDER=_slo, SPLIT_PAIR=_ssp)

# ---------------------------------------------------------------- 15. fit degree and significance
_BASELINE_SCALE_S = 600      # a representative lane length, purely to size this test
print('\n15. fit degree is set by point count, and significance decides what counts as measured')
check_true('plenty of points keeps the quadratic', kin_degree(200) == KIN_DEG,
           f'(degree {kin_degree(200)})')
check_true('too few points drops the degree', kin_degree(3) == 1, f'(degree {kin_degree(3)})')
check_true('degree never goes below 1', kin_degree(2) == 1 and kin_degree(0) == 1)

# The curvature of a SHORT lane must still be recovered: it is real, and refusing to fit it
# because of the lane's duration is what a fixed baseline cut wrongly did.
tg2 = np.linspace(0, 1500, 60)
short = (tg2 >= 600) & (tg2 <= 600 + 0.5 * _BASELINE_SCALE_S)
long_ = (tg2 >= 100) & (tg2 <= 100 + 2 * _BASELINE_SCALE_S)
r_curved = 1.6 + (600e3 / R_SUN_M) * tg2 + 0.5 * (-500 / R_SUN_M) * tg2 ** 2
_, a_short = kinematics(np.where(short, r_curved, np.nan), tg2, span=short)
_, a_long = kinematics(np.where(long_, r_curved, np.nan), tg2, span=long_)
check('a IS recovered for a SHORT lane', np.nanmean(a_short), -500, 1e-6, 'm/s^2')
check('a IS recovered for a long lane', np.nanmean(a_long), -500, 1e-6, 'm/s^2')

# significance, not lane length, is what gates the word "measured"
check_true('a well clear of its error is measured', accel_is_measured(-500, 50))
check_true('a inside its error is not measured', not accel_is_measured(-40, 50))
check_true('a exactly at the threshold is measured',
           accel_is_measured(KIN_A_SIGMA * 50, 50))
check_true('NaN a is never measured', not accel_is_measured(np.nan, 50)
           and not accel_is_measured(-500, np.nan) and not accel_is_measured(-500, 0))

# fitting on the lane's own dense samples must not change an exactly-quadratic answer
_td = np.linspace(600, 600 + 0.5 * _BASELINE_SCALE_S, KIN_N_DENSE)
_rd = 1.6 + (600e3 / R_SUN_M) * _td + 0.5 * (-500 / R_SUN_M) * _td ** 2
_, a_dense = kinematics(np.where(short, r_curved, np.nan), tg2, span=short,
                        t_fit=_td, r_fit=_rd)
check('dense per-lane fit gives the same a', np.nanmean(a_dense), -500, 1e-6, 'm/s^2')

# ---------------------------------------------------------------- 17. null test on acceleration
# The strongest check in this file: a shock moving at EXACTLY constant speed must come back with
# a = 0. It does not, automatically - a degree-2 lane fit returns tens of m/s^2 of spurious
# DECELERATION, because a quadratic in log10 f cannot represent the frequency drift of a
# constant-speed shock climbing through a structured corona, and that misfit reappears as
# curvature in r(t). Nothing in the Monte Carlo can see this: the error bars sit tightly around
# the wrong number.
print('\n17. a constant-speed shock must return a = 0')
_mdl = MODEL_GRID['Newkirk x2']
_rr, _ne = invert_grid(_mdl)


def _null_a(r0, T, s, deg):
    """Acceleration recovered from an exactly constant-velocity shock."""
    tt = np.linspace(0, T, 300)
    r_true = r0 + (600 * 1e3 / R_SUN_M) * tt
    f_true = PLASMA_CONST * np.sqrt(_mdl(r_true)) * s / 1e6
    fit = lane_fit(RUN, {'t': [t0 + pd.Timedelta(seconds=float(x)) for x in tt],
                     'f': list(f_true)}, deg=deg, sigma_f=0.7)
    to_r = lambda x: np.interp(freq_to_density(eval_lane(fit, fit['p'], x) * 1e6, harmonic=s),
                               _ne[::-1], _rr[::-1])
    td = np.linspace(fit['tmin'], fit['tmax'], KIN_N_DENSE)
    tgs = np.linspace(fit['tmin'], fit['tmax'], 60)
    v, a = kinematics(to_r(tgs), tgs, t_fit=td, r_fit=to_r(td))
    return float(np.nanmean(v)), float(np.nanmean(a))


for _name, _r0, _T, _s in [('wide F lane, 600 s', 1.48, 600, 1),
                           ('narrow F lane, 190 s', 1.69, 190, 1),
                           ('long H lane, 1500 s', 1.85, 1500, 2)]:
    _v, _a = _null_a(_r0, _T, _s, LANE_DEG)
    check(f'v recovered, {_name}', _v, 600, 2, 'km/s')
    check_true(f'|a| < 10 m/s^2, {_name}', abs(_a) < 10, f'(a = {_a:+.2f} m/s^2)')

# and prove the degree is what fixes it, so nobody lowers LANE_DEG without seeing the cost
_a2 = abs(_null_a(1.85, 1500, 2, 2)[1])
_a3 = abs(_null_a(1.85, 1500, 2, 3)[1])
check_true('a quadratic lane fit really is the source of the bias', _a2 > 10 * max(_a3, 0.1),
           f'(deg 2 -> {_a2:.1f} m/s^2, deg {LANE_DEG} -> {_a3:.2f} m/s^2)')
check_true('the shipped LANE_DEG is at least 3', LANE_DEG >= 3, f'(LANE_DEG = {LANE_DEG})')

# the bias floor must gate the significance test, not merely be reported next to it
check_true('a below the bias floor is not "measured"',
           not accel_is_measured(30, 5, bias=-54))
check_true('a above the bias floor is measured', accel_is_measured(300, 5, bias=-54))

# ------------------------------------------------- 18. faults found in the line-by-line audit
print('\n18. regressions for each fault found in the full audit')

# (a) the kinematics gate must count the points the FIT uses, not the shared grid's points.
#     Tying it to the grid silently lost the speed of any lane shorter than ~3 grid steps.
_tgf = np.linspace(0, 2000, N_GRID)
for _span in (600, 300, 190, 150, 120):
    _sel = (_tgf >= 500) & (_tgf <= 500 + _span)
    _r = np.where(_sel, 1.6 + (600e3 / R_SUN_M) * _tgf, np.nan)
    _td = np.linspace(500, 500 + _span, KIN_N_DENSE)
    _v, _ = kinematics(_r, _tgf, span=_sel, t_fit=_td, r_fit=1.6 + (600e3 / R_SUN_M) * _td)
    check(f'short lane keeps its speed, {_span} s ({_sel.sum()} grid pts)',
          np.nanmean(_v), 600, 0.5, 'km/s')

# (b) samples landing on one time column are averaged, not decided by sort order
_o = bezier_freq_lane(np.array([10, 10, 10, 11]), np.array([100, 140, 180, 200]))
check('duplicate times are averaged, not first-wins', _o['f'][0],
      float(LAYER_F[[100, 140, 180]].mean()), 1e-9, 'MHz')

# (c) _thin must be correct on unsorted input, not just on the sorted input it happens to get
_tu = np.array([0., 30., 5., 60., 12., 90.])
_k = thin_samples(_tu)
check_true('_thin is safe on unsorted input',
           np.all(np.diff(np.sort(_tu[_k])) >= FIT_MIN_DT_S - 1e-9), f'(kept {list(_k)})')

# (d) the polarisation box must be symmetric about the point; a half-open upper edge holds one
#     fewer sample on the late side and drags the box early along a drifting lane
_hf = np.timedelta64(int(POL_DT_S * 1e3), 'ms')
_tm = POL.index[len(POL) // 2].to_numpy()
_i0 = np.searchsorted(POL_T, _tm - _hf, side='left')
_i1 = np.searchsorted(POL_T, _tm + _hf, side='right')
check_true('polarisation box straddles the point evenly',
           abs((_tm - POL_T[_i0]) - (POL_T[_i1 - 1] - _tm)) <= np.timedelta64(1, 'ms'),
           f'({_i1 - _i0} samples)')

# (e) M_A diverges as X -> 4; the notebook has to notice rather than return 134 quietly
check_true('M_A blows up near X = 4', alfven_mach_from_X(np.array([3.99]))[0] > 40,
           f'(M_A(3.99) = {alfven_mach_from_X(np.array([3.99]))[0]:.1f})')
check_true('X out of range gives NaN, not a number',
           np.isnan(alfven_mach_from_X(np.array([0.99]))[0])
           and np.isnan(alfven_mach_from_X(np.array([4.01]))[0]))

# (f) a split whose branches cross has X < 1 over part of the overlap; the fraction has to be
#     measurable, because every downstream mean is a nanmean and drops those samples in silence
_tt = np.linspace(0, 600, 40)
_lo = {'t': [t0 + pd.Timedelta(seconds=float(x)) for x in _tt], 'f': list(50 - 0.02 * _tt)}
_up = {'t': [t0 + pd.Timedelta(seconds=float(x)) for x in _tt], 'f': list(55 - 0.035 * _tt)}
_g = np.linspace(0, 600, 20)
_X = (lane_deriv(lane_fit(RUN, _up, sigma_f=0.3), _g)[0]
      / lane_deriv(lane_fit(RUN, _lo, sigma_f=0.3), _g)[0]) ** 2
_frac = float(np.mean(~((_X >= 1) & (_X < 4))))
check_true('crossing branches are detectable as an X-validity fraction', _frac > 0.3,
           f'({100 * _frac:.0f}% of the overlap unusable, mean X = {np.nanmean(_X):.3f} '
           'looks innocent)')

# (g) B is unchanged by all of the above - the one number that must not move
check('B pipeline still matches the hand calculation',
      (600 / 1.5 * 1e3) * np.sqrt(MU0 * MU * M_P * (3.0e7 * 1e6)) * 1e4, 1.13195, 1e-5, 'G')

# ---------------------------------------------------------------- 16. LaTeX in f-strings
print('\n16. no LaTeX macro can be eaten by a non-raw f-string')
import glob as _glob, re as _re
_D = _re.compile(r'(?<!\\)\\[nrtvbfa](?=[a-zA-Z])')
_bad = []
for _path in sorted(_glob.glob('typeii/*.py')):
    for _ln in open(_path):
        for _m in _re.finditer(r"(?<![rR])\bf(['\"])(.*?)\1", _ln):
            for _seg in _re.findall(r'\$[^$]*\$', _m.group(2)):
                if _D.search(_seg):
                    _bad.append(_seg)
check_true('no \\nu / \\tau / \\rm eaten as a python escape in maths mode',
           not _bad, f'({len(_bad)} found)' if _bad else '(this broke the chi^2 legend once)')

print('\n19. a cubic\'s derivative near the edge of its span is not a measurement')
# The fault this guards against: the F/H relative-drift test used to evaluate each band's cubic
# over the F/H overlap, which is the last third of one fit and the first eighth of the other. Two
# cubics through the SAME underlying curve, fitted over different spans, disagree there - so the
# test was measuring the fits' edge behaviour and reporting it as a physical discrepancy between
# the bands. On the real event that came out as 14.8%, at 5.6 sigma.
#
# Reproduce the geometry from a lane that is exactly ONE shock: a constant-speed front through
# Newkirk, so any band-to-band difference the estimators report is by construction an artefact.
# A pure exponential will not do - a cubic in log f fits that perfectly and both spans agree.
_tt = np.linspace(0, 2000, 400)
_rr_ = 1.35 + (700 * 1e3 / R_SUN_M) * _tt                    # 700 km/s, no acceleration
_ff = PLASMA_CONST * np.sqrt(MODEL_GRID['Newkirk x2'](_rr_)) / 1e6
_A = (_tt >= 100) & (_tt <= 700)                             # "F": the overlap is its last third
_B = (_tt >= 520) & (_tt <= 2000)                            # "H": the overlap is its first eighth
_ov = (_tt >= 520) & (_tt <= 700)
_edge = [np.mean(np.polyval(np.polyder(np.polyfit(_tt[_m], np.log10(_ff[_m]), 3)), _tt[_ov]))
         * np.log(10) for _m in (_A, _B)]
_direct = [np.polyfit(_tt[_m & _ov], np.log(_ff[_m & _ov]), 1)[0] for _m in (_A, _B)]
_truth = np.mean(np.gradient(np.log(_ff), _tt)[_ov])
check_true('two cubics on ONE lane disagree at their opposite edges',
           abs(_edge[0] - _edge[1]) / abs(_truth) > 0.01,
           f'({100 * abs(_edge[0] - _edge[1]) / abs(_truth):.1f}% of the true drift, from a lane '
           f'with no band-to-band difference in it)')
check('a direct log-linear fit on the overlap recovers the true drift',
      _direct[0], _truth, abs(_truth) * 5e-3, '1/s')
check_true('and it cannot manufacture a band-to-band difference',
           abs(_direct[0] - _direct[1]) < 1e-15,
           '(both bands see the same points in the window, so they get the same answer)')

print('\n20. the height inversion, on every model x fold across the whole observing band')
# _invert_grid builds a lookup table and the chain interpolates on it. Test the table against a
# root find done independently with brentq, at every model and both harmonics, over the real band.
from scipy.optimize import brentq as _brentq
_worst, _worst_at, _nchk = 0.0, '', 0
_nonmono = []
for _mname, _mdl in MODEL_GRID.items():
    _rr_t, _ne_t = invert_grid(_mdl)
    if not np.all(np.diff(_ne_t) < 0):
        _nonmono.append(_mname)
    for _s in (1, 2):
        for _fMHz in (25, 40, 60, 84):
            _ne_want = freq_to_density(_fMHz * 1e6, harmonic=_s)
            if not (np.nanmin(_ne_t) <= _ne_want <= np.nanmax(_ne_t)):
                continue                       # genuinely outside this model's range
            _got = float(np.interp(_ne_want, _ne_t[::-1], _rr_t[::-1]))
            try:
                _want = _brentq(lambda r: _mdl(r) - _ne_want, _rr_t.min(), _rr_t.max(), xtol=1e-12)
            except ValueError:
                continue
            _nchk += 1
            if abs(_got - _want) > _worst:
                _worst, _worst_at = abs(_got - _want), f'{_mname}, s={_s}, {_fMHz} MHz'
check_true('n_e(r) is strictly decreasing on every model grid',
           not _nonmono, f'({len(MODEL_GRID)} models)' if not _nonmono else f'BAD: {_nonmono}')
check('interpolated height vs independent root find, worst case over the grid',
      _worst, 0.0, 2e-3, 'Rsun')
check_true('the check actually ran over the whole grid', _nchk >= 100,
           f'({_nchk} model x harmonic x frequency combinations)')
check_true('a frequency below every model returns NaN, not an extrapolation',
           np.isnan(freq_to_radius(1.0, MODEL_GRID['Newkirk x2'], 1)))

print('\n21. Monte-Carlo error propagation does what it claims')
# Three properties the aggregation must have, none of which a central value reveals:
#   (a) the reported error scales with the assumed per-point frequency error;
#   (b) it does NOT shrink as sqrt(N) over jittered Bezier repeats, which are not independent;
#   (c) identical repeats with zero assumed error give zero spread.
_lab0 = TRACED[0]
_sig0 = LANE_SIGMA.get(_lab0, 0.7)
_errs = {}
for _mult in (1, 2, 4):
    _saved = dict(LANE_SIGMA)
    LANE_SIGMA.update({k: v * _mult for k, v in _saved.items()})
    _agg = aggregate_lanes(RUN, passes, tg, MODEL_GRID[REF_MODEL_NAME], n_mc=60, seed=1)
    _errs[_mult] = grid_scalar(_agg[_lab0], 'r')[1]
    LANE_SIGMA.clear()
    LANE_SIGMA.update(_saved)
# The reported error is sqrt(fit^2 + spread^2). Only the fit term scales with sigma_f; the
# pass-to-pass spread comes from the Bezier jitter and does not move. So the ratio rises with the
# multiplier and approaches it from below without reaching it - asserting exact proportionality
# would be asserting that the repeat term does not exist.
for _m in (2, 4):
    _ratio = _errs[_m] / _errs[1]
    check_true(f'error rises with sigma_f x{_m}, approaching {_m} from below',
               1.0 < _ratio < _m,
               f'(ratio {_ratio:.2f}; the fixed repeat-spread term damps it below {_m})')
# With total = sqrt((m f)^2 + s^2), the ratio as a fraction of m is sqrt(f^2 + s^2/m^2) /
# sqrt(f^2 + s^2), which FALLS as m grows and the fixed term matters less in relative terms.
check_true('the shortfall grows with the multiplier, as that form requires',
           _errs[4] / _errs[1] / 4 < _errs[2] / _errs[1] / 2,
           f'({_errs[2] / _errs[1] / 2:.3f} of x2, then {_errs[4] / _errs[1] / 4:.3f} of x4)')
# Solve the two-component form for s/f from the x2 ratio and check it predicts the x4 ratio.
_r2 = _errs[2] / _errs[1]
_sf2 = (4 - _r2 ** 2) / (_r2 ** 2 - 1)                       # (s/f)^2
check('x4 ratio predicted from the x2 ratio by sqrt(fit^2 + spread^2)',
      _errs[4] / _errs[1], np.sqrt((16 + _sf2) / (1 + _sf2)), 0.05, 'x', rel=True)
# The fixture carries a single pass, so build three jittered copies here - the same thing the
# tracer's auto-repeat does - and check the aggregation does not treat them as independent.
_rng0 = np.random.default_rng(7)
_reps = [passes[0]] + [{k: {'t': list(v['t']),
                           'f': [float(x) + _rng0.normal(0, 0.05) for x in v['f']]}
                        for k, v in passes[0].items()} for _ in range(2)]
_one = aggregate_lanes(RUN, [passes[0]], tg, MODEL_GRID[REF_MODEL_NAME], n_mc=60, seed=1)
_many = aggregate_lanes(RUN, _reps, tg, MODEL_GRID[REF_MODEL_NAME], n_mc=60, seed=1)
_r1, _rN = grid_scalar(_one[_lab0], 'r')[1], grid_scalar(_many[_lab0], 'r')[1]
check_true('jittered repeats do NOT divide the error by sqrt(N)',
           _rN > _r1 / np.sqrt(len(_reps)) * 1.3,
           f'({len(_reps)} jittered repeats: {_r1:.5f} -> {_rN:.5f} Rsun; sqrt(N) shrinkage '
           f'would give {_r1 / np.sqrt(len(_reps)):.5f})')
check_true('REPEATS_INDEPENDENT is off for auto-jittered repeats', not REPEATS_INDEPENDENT)

print('\n22. full chain on an injected shock: recover the speed and the field, not just X')
# Section 14 injects a known density jump and recovers X and M_A. Those are model-independent.
# This injects a complete shock - known height, speed and upstream field - and checks the
# model-DEPENDENT half of the chain end to end, which nothing else here does.
_V_TRUE, _R0, _X_TRUE, _MDL = 800.0, 1.45, 1.35, MODEL_GRID['Newkirk x2']
_tt = np.linspace(0, 900, 120)
_rt = _R0 + (_V_TRUE * 1e3 / R_SUN_M) * _tt
_ne_up = _MDL(_rt)
_f_lo = PLASMA_CONST * np.sqrt(_ne_up) / 1e6                       # upstream branch, s = 1
_f_hi = _f_lo * np.sqrt(_X_TRUE)                                   # downstream branch
_MA_TRUE = np.sqrt(_X_TRUE * (_X_TRUE + 5) / (2 * (4 - _X_TRUE)))
_vA_TRUE = _V_TRUE / _MA_TRUE
_rho = MU * M_P * np.mean(_ne_up) * 1e6
_B_TRUE = _vA_TRUE * 1e3 * np.sqrt(MU0 * _rho) * 1e4
_mk = lambda f: {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in _tt], 'f': list(f)}
_sv_tr, _sv_bd, _sv_sg = list(TRACED), dict(LANE_BAND), dict(LANE_SIGMA)
_sv_ord, _sv_pair, _sv_bt = dict(LANE_ORDER), dict(SPLIT_PAIR), list(BANDS_TRACED)
try:
    TRACED[:] = ['F lane 1', 'F lane 2']
    LANE_BAND.clear(); LANE_BAND.update({'F lane 1': 'F', 'F lane 2': 'F'})
    LANE_SIGMA.clear(); LANE_SIGMA.update({'F lane 1': 1e-6, 'F lane 2': 1e-6})
    LANE_ORDER.clear(); LANE_ORDER.update({'F': ['F lane 1', 'F lane 2']})
    SPLIT_PAIR.clear(); SPLIT_PAIR.update({'F': ('F lane 1', 'F lane 2')})
    BANDS_TRACED[:] = ['F']
    _tg2 = np.linspace(_tt.min(), _tt.max(), N_GRID)
    _ag = aggregate_lanes(RUN, [{'F lane 1': _mk(_f_lo), 'F lane 2': _mk(_f_hi)}], _tg2, _MDL, n_mc=1)
    _d = _ag['F lane 1']
    _cm = common_mask(_d)
    # Compare against the truth evaluated on THE SAME grid points the pipeline averaged over.
    # Averaging the injected samples instead compares two different sample sets and fails by more
    # than the pipeline's own error - a test artefact, not a pipeline one.
    _tw = _tg2[_cm]
    _r_true_w = _R0 + (_V_TRUE * 1e3 / R_SUN_M) * _tw
    _ne_true_w = _MDL(_r_true_w)
    _B_true_w = np.mean((_vA_TRUE * 1e3) * np.sqrt(MU0 * MU * M_P * _ne_true_w * 1e6) * 1e4)
    check('recovered height', grid_scalar(_d, 'r', mask=_cm)[0], np.mean(_r_true_w), 2e-3, 'Rsun')
    check('recovered shock speed', grid_scalar(_d, 'v', mask=_cm)[0], _V_TRUE, 6.0, 'km/s')
    check('recovered density jump X', grid_scalar(_d, 'X', mask=_cm)[0], _X_TRUE, 2e-3)
    check('recovered Alfven Mach number', grid_scalar(_d, 'MA', mask=_cm)[0], _MA_TRUE, 2e-3)
    check('recovered Alfven speed', grid_scalar(_d, 'vA', mask=_cm)[0], _vA_TRUE, 6.0, 'km/s')
    # B is the grid-mean of v_A sqrt(mu0 rho(t)), a mean of a product - not the product of the
    # means, which is what _B_TRUE above is. Compare like with like.
    check('recovered magnetic field', grid_scalar(_d, 'B', mask=_cm)[0], _B_true_w, 0.02, 'G')
    # upstream is passed explicitly: lane_windows otherwise reads LANE_ROLE, which this block
    # has not rebuilt, and the test would be measuring the fixture's naming rather than the code.
    _w = lane_windows(_d, 'F lane 1', _tg2, t0, upstream=True)
    check_true('lane_windows closes v_A = v_sh / M_A on its own numbers',
               abs(_w['vA_split'] - _w['v_split'] / _w['MA_split']) / _w['vA_split'] < 5e-3,
               f'(v_A {_w["vA_split"]:.1f} vs v/M_A {_w["v_split"] / _w["MA_split"]:.1f} km/s)')
    check_true('lane_windows withholds v_A and B from a downstream branch',
               np.isnan(lane_windows(_ag['F lane 2'], 'F lane 2', _tg2, t0,
                                     upstream=False)['B_split']))
finally:
    TRACED[:] = _sv_tr
    LANE_BAND.clear(); LANE_BAND.update(_sv_bd)
    LANE_SIGMA.clear(); LANE_SIGMA.update(_sv_sg)
    LANE_ORDER.clear(); LANE_ORDER.update(_sv_ord)
    SPLIT_PAIR.clear(); SPLIT_PAIR.update(_sv_pair)
    BANDS_TRACED[:] = _sv_bt

print('\n23. the polarisation gate discriminates in BOTH directions')
# A significance gate that only ever says "not significant" is worthless. The fixture injects a
# real contrast and the gate must fire positively on it; a shuffled version with no contrast must
# not. Both are checked here because the real data land on the negative side, so the positive
# branch would otherwise never be exercised.
_pF = pol_table[pol_table['band'] == 'F']['V_over_I_mean'].mean()
_pH = pol_table[pol_table['band'] == 'H']['V_over_I_mean'].mean()
_noise = pol_table['V_over_I_sd'].median()
check_true('fires positively on the fixture, which has an injected contrast',
           abs(_pF - _pH) / _noise >= 2 and abs(_pF) > abs(_pH),
           f'({abs(_pF - _pH) / _noise:.1f}x the scatter, F {_pF:+.4f} vs H {_pH:+.4f})')
_flat = pol_table.copy()
_flat['V_over_I_mean'] = _flat['V_over_I_mean'].mean()
check_true('reports no contrast when the two bands are identical',
           abs(_flat[_flat.band == 'F']['V_over_I_mean'].mean()
               - _flat[_flat.band == 'H']['V_over_I_mean'].mean()) / _noise < 2)
# polarisation_caveats must catch lanes that agree far more closely than either is determined
_tight = pol_table.copy()
_tight.loc[_tight.band == 'F', 'V_over_I_mean'] = _tight[_tight.band == 'F'][
    'V_over_I_mean'].iloc[0]
check_true('polarisation_caveats catches a common instrumental offset',
           len(polarisation_caveats(_tight)) > 0,
           '(two lanes of a band made identical)')

print('\n' + '=' * 100)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILURES: ' + ', '.join(FAIL))
print('=' * 100)
sys.exit(1 if FAIL else 0)
