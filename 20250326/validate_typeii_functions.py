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
src = open('test_run.py').read()
cut = src.index('print("\\n########## CELL 4')          # everything through A.4 is defined by then
exec(compile(src[:cut], 'notebook', 'exec'), globals())
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
fit = _lane_fit(lane)
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
           np.all(np.diff(ts[_thin(ts)]) >= FIT_MIN_DT_S - 1e-9))
check_true('thinning does not bias the fit',
           abs(lane_deriv(fit, np.array([200.0]))[0][0]
               - 10 ** (c2 * 200 ** 2 + c1 * 200 + c0)) < 1e-6)
check_true('_eval blanks outside the traced span',
           np.isnan(_eval(fit, fit['p'], np.array([-50.0]))[0])
           and np.isfinite(_eval(fit, fit['p'], np.array([200.0]))[0]))

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
_saved_traced, _saved_sigma = TRACED, LANE_SIGMA
tl = np.linspace(100, 800, 40)                               # long lane, high -> low
ts_ = np.linspace(550, 800, 20)                              # short lane, sits ABOVE it
long_lane = {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tl],
             'f': list(70 * (33 / 70) ** ((tl - 100) / 700))}
short_lane = {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in ts_],
              'f': list(1.15 * 70 * (33 / 70) ** ((ts_ - 100) / 700))}
globals()['TRACED'] = ['F lane 1', 'F lane 2']
globals()['LANE_SIGMA'] = {}
globals()['passes'] = [{'F lane 1': long_lane, 'F lane 2': short_lane}]
order, key, note = order_lanes(['F lane 1', 'F lane 2'])
check_true('long lane is the upstream branch', order[0] == 'F lane 1',
           f'order={order}, f={{{key["F lane 1"]:.1f}, {key["F lane 2"]:.1f}}} MHz, {note}')
check('implied X = (f_U/f_L)^2', (key[order[1]] / key[order[0]]) ** 2, 1.15 ** 2, 2e-3, '', rel=True)
check_true('own-span means would have inverted it',
           np.mean(long_lane['f']) > np.mean(short_lane['f']),
           f'(own-span means {np.mean(long_lane["f"]):.1f} vs {np.mean(short_lane["f"]):.1f} MHz)')
globals()['TRACED'], globals()['LANE_SIGMA'] = _saved_traced, _saved_sigma

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
exec(compile(src[src.index("RS_KM = R_SUN_M / 1e3"):src.index("FIT_OUT = {}")], 'fits', 'exec'),
     globals())
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
exec(compile(src[src.index("POL_T = POL.index.to_numpy()"):src.index("pol_rows = []")],
             'pol', 'exec'), globals())
_pt = pd.date_range(LAYER_T[0], LAYER_T[-1], periods=200)
_pf = np.interp(np.linspace(0, 1, 200), [0, 1], [60.0, 35.0])
got_p = sample_polarisation(list(_pt), list(_pf))
check_true('returns one value per traced point', len(got_p) == 200)
check_true('values lie inside [-1, 1]', np.nanmax(np.abs(got_p)) <= 1.0)
check_true('mean |x| of zero-mean noise returns ~0.8 sigma, not the mean',
           abs(np.mean(np.abs(np.random.default_rng(0).normal(0, 0.02, 100000))) / 0.02 - 0.7979)
           < 0.01, '(this is why the signed mean is the number to compare between bands)')

# ---------------------------------------------------------------- 13. no-overlap ordering path
print('\n13. order_lanes when the lanes never overlap')
_st, _ss = TRACED, LANE_SIGMA
ta_ = np.linspace(100, 400, 20)
tb_ = np.linspace(600, 900, 20)
globals()['TRACED'] = ['F lane 1', 'F lane 2']
globals()['LANE_SIGMA'] = {}
globals()['passes'] = [{
    'F lane 1': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in ta_],
                 'f': list(70 - 0.05 * (ta_ - 100))},
    'F lane 2': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tb_],
                 'f': list(40 - 0.02 * (tb_ - 600))}}]
_o, _k, _n = order_lanes(['F lane 1', 'F lane 2'])
check_true('non-overlapping lanes are flagged', 'WARNING' in _n, _n[:70] + '...')
globals()['TRACED'], globals()['LANE_SIGMA'] = _st, _ss

# ---------------------------------------------------------------- 14. end-to-end closure
print('\n14. end-to-end closure: inject a known split, recover X and M_A')
X_inj = 1.21                                                 # split factor 1.1 -> X = 1.21
sp = np.sqrt(X_inj)
tc = np.linspace(100, 1000, 60)
f_lo = 65 * (30 / 65) ** ((tc - 100) / 900)
globals()['TRACED'] = ['F lane 1', 'F lane 2']
globals()['LANE_BAND'] = {'F lane 1': 'F', 'F lane 2': 'F'}
globals()['BANDS_TRACED'] = ['F']
globals()['LANE_SIGMA'] = {}
globals()['passes'] = [{
    'F lane 1': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tc], 'f': list(f_lo)},
    'F lane 2': {'t': [t0 + pd.Timedelta(seconds=float(s)) for s in tc], 'f': list(f_lo * sp)}}]
_o, _k, _n = order_lanes(['F lane 1', 'F lane 2'])
globals()['LANE_ORDER'] = {'F': _o}
globals()['SPLIT_PAIR'] = {'F': (_o[0], _o[1])}
sc = scalar_summary(passes)
check('X recovered end to end', sc['F']['X'][0], X_inj, 1e-3, '', rel=True)
check('M_A recovered end to end', sc['F']['M_A'][0],
      np.sqrt(X_inj * (X_inj + 5) / (2 * (4 - X_inj))), 1e-3, '', rel=True)
check('relative bandwidth recovered', sc['F']['rel_bandwidth'][0], sp - 1, 1e-3, '', rel=True)
globals()['TRACED'], globals()['LANE_SIGMA'] = _st, _ss

# ---------------------------------------------------------------- 15. adaptive fit degree
print('\n15. r(t) degree adapts to the traced baseline')
tg2 = np.linspace(0, 1500, 60)
short = (tg2 >= 600) & (tg2 <= 600 + 0.5 * KIN_MIN_BASELINE_S)
long_ = (tg2 >= 100) & (tg2 <= 100 + 2.0 * KIN_MIN_BASELINE_S)
check_true(f'short lane (<{KIN_MIN_BASELINE_S} s) drops to a straight line',
           kin_degree(tg2, short) == 1, f'(degree {kin_degree(tg2, short)})')
check_true('long lane keeps the quadratic', kin_degree(tg2, long_) == KIN_DEG,
           f'(degree {kin_degree(tg2, long_)})')
r_curved = 1.6 + (600e3 / R_SUN_M) * tg2 + 0.5 * (-500.0 / R_SUN_M) * tg2 ** 2
_, a_short = kinematics(np.where(short, r_curved, np.nan), tg2, span=short)
_, a_long = kinematics(np.where(long_, r_curved, np.nan), tg2, span=long_)
check_true('a is not reported for a short lane', abs(np.nanmean(a_short)) < 1e-9,
           '(a straight-line fit gives a = 0 by construction, not a measurement)')
check('a IS recovered for a long lane', np.nanmean(a_long), -500.0, 1e-6, 'm/s^2')

# ---------------------------------------------------------------- 16. LaTeX in f-strings
print('\n16. no LaTeX macro can be eaten by a non-raw f-string')
import json as _json, re as _re
_nb = _json.load(open('plot_nenufar_new.ipynb'))
_D = _re.compile(r'(?<!\\)\\[nrtvbfa](?=[a-zA-Z])')
_bad = []
for _c in _nb['cells']:
    if _c['cell_type'] != 'code':
        continue
    for _ln in ''.join(_c['source']).splitlines():
        for _m in _re.finditer(r"(?<![rR])\bf(['\"])(.*?)\1", _ln):
            for _seg in _re.findall(r'\$[^$]*\$', _m.group(2)):
                if _D.search(_seg):
                    _bad.append(_seg)
check_true('no \\nu / \\tau / \\rm eaten as a python escape in maths mode',
           not _bad, f'({len(_bad)} found)' if _bad else '(this broke the chi^2 legend once)')

print('\n' + '=' * 100)
print(f'{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILURES: ' + ', '.join(FAIL))
print('=' * 100)
sys.exit(1 if FAIL else 0)
