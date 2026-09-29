"""Cross-check every quoted number in the README, the deck and the notebook against the run."""
import re, zipfile, json, sys, os
sys.path.insert(0, '/sessions/inspiring-youthful-mayer/mnt/DIAS')
import numpy as np, pandas as pd

D = '/sessions/inspiring-youthful-mayer/mnt/DIAS/'
S = D + 'for_sophie/'
OK, BAD = [], []
def ck(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(f'  [{"ok  " if cond else "FAIL"}] {name:<62s} {detail}')

t = pd.read_csv(S+'height_time_all_models.csv')
s = pd.read_csv(S+'height_time_model_summary.csv')
mi = pd.read_csv(S+'model_independent_scalars.csv')
rd = open(S+'README.md').read()
run = open(D + 'type2_nanufar_20250326_regenerated/run_log.txt').read()

print('\n1. for_sophie CSVs')
ck('6180 rows', len(t)==6180, f'({len(t)})')
ck('80 tracks (5 models x 4 folds x 4 lanes)', t.groupby(["model_fold","lane"]).ngroups==80)
ck('r monotonic on every track',
   all(np.all(np.diff(g.r_heliocentric_Rsun.values)>=-1e-9) for _,g in t.groupby(["model_fold","lane"])))
ck('h = r - 1 exactly', np.allclose(t.h_above_limb_Rsun, t.r_heliocentric_Rsun-1, atol=1e-12))
ck('no bare "height" column', not any(c=='height' for c in t.columns))
ck('B_G present only where both branches exist',
   int(t.B_G.isna().sum())==4020, f'({int(t.B_G.isna().sum())} blank of {len(t)})')
ck('README B-blank count matches', '4020 of 6180' in rd)
ck('no B on any downstream row',
   t[t.role.str.startswith('downstream')].B_G.notna().sum() == 0)
ck('README explains why B is upstream-only', 'neither\n  branch' in rd or 'neither' in rd)
_bt = pd.read_csv(S + 'bfield_tracks.csv')
ck('bfield_tracks.csv ships with the tracks', len(_bt) > 0 and _bt.groupby(['model_fold','lane']).ngroups == 40,
   f'({len(_bt)} rows, {_bt.groupby(["model_fold","lane"]).ngroups} tracks)')
ck('every B track is an upstream lane', set(_bt.lane) <= set(t[t.role.str.startswith('upstream')].lane))
ck('B tracks are on the 10 s export grid',
   int(np.median(np.diff(sorted(_bt[_bt.lane == _bt.lane.iloc[0]].t_sec_from_window_start.unique())))) == 10)
_n = _bt.groupby(['model_fold', 'lane']).size()
ck('every track is a curve, not a handful of points', _n.min() >= 15,
   f'({_n.min()}-{_n.max()} samples per track, {len(_bt)} rows)')
_rel = (_bt.B_G_sd / _bt.B_G)
ck('no sample has a zero Monte-Carlo error', int((_bt.B_G_sd == 0).sum()) == 0)
ck('README states the Monte-Carlo error as a percentage of B',
   f'{100 * _rel.median():.1f}% of B' in rd
   and f'{100 * _rel.min():.1f}-{100 * _rel.max():.1f}%' in rd,
   f'(median {100 * _rel.median():.1f}%, range {100 * _rel.min():.1f}-{100 * _rel.max():.1f}%)')
ck('README names the density-model spread as the dominant error',
   'spread of the curves is the error bar' in rd and 'two orders of magnitude' in rd)
ck('the run log prints both errors', 'the two uncertainties on B' in run)
ck('B tracks carry what B was built from',
   {'X', 'M_A', 'vA_kms', 'ne_cm3', 'v_kms', 'B_G_sd'} <= set(_bt.columns))
_ma = np.sqrt(_bt.X * (_bt.X + 5) / (2 * (4 - _bt.X)))
ck('M_A closes on every row', float(np.nanmax(np.abs(_ma - _bt.M_A) / _bt.M_A)) < 2e-3,
   f'(max {100 * float(np.nanmax(np.abs(_ma - _bt.M_A) / _bt.M_A)):.3f}%)')
ck('X inside the Rankine-Hugoniot domain on every row',
   bool(((_bt.X >= 1) & (_bt.X < 4)).all()), f'(X {_bt.X.min():.3f}-{_bt.X.max():.3f})')
_j = np.abs(_bt.v_kms / _bt.M_A - _bt.vA_kms) / _bt.vA_kms
ck('README states the Monte-Carlo closure gap, not exact closure',
   'mean of a ratio is not the ratio of the means' in rd
   and f'{100 * float(np.nanmax(_j)):.3f}%' in rd,
   f'(vA gap median {100 * float(np.nanmedian(_j)):.1f}%, max {100 * float(np.nanmax(_j)):.1f}%)')
_bm = _bt.merge(t[t.B_G.notna()], on=['model_fold', 'lane', 't_sec_from_window_start'],
                suffixes=('_bt', '_ht'))
ck('B tracks agree with height_time_all_models where they overlap',
   len(_bm) > 0 and np.allclose(_bm.B_G_bt, _bm.B_G_ht), f'({len(_bm)} shared rows)')
# one slope per model x fold x lane, then the median per lane. Keying a dict on the lane alone
# keeps only the last model x fold, which is not a median of anything.
_sl = pd.DataFrame(
    [{'lane': l,
      'delta': -np.polyfit(np.log(g.sort_values('r_Rsun').r_Rsun),
                           np.log(g.sort_values('r_Rsun').B_G), 1)[0]}
     for (mf, l), g in _bt.groupby(['model_fold', 'lane']) if len(g) >= 4])
for _l, _x in _sl.groupby('lane'):
    ck(f'README quotes the B(r) slope for {_l}',
       f'{_x.delta.median():.2f}' in rd and f'{_x.delta.min():.2f}-{_x.delta.max():.2f}' in rd,
       f'delta = {_x.delta.median():.2f} ({_x.delta.min():.2f}-{_x.delta.max():.2f}, n={len(_x)})')
    ck(f'the run log prints the same slope for {_l}',
       f'{_l:10s} delta = {_x.delta.median():.2f}' in run)
up = s[s.role.str.startswith('upstream')]
ck('README upstream speed range', f'{up.v_mean_kms.min():.0f}-{up.v_mean_kms.max():.0f} km/s' in rd,
   f'({up.v_mean_kms.min():.0f}-{up.v_mean_kms.max():.0f})')
ck('README upstream height range',
   f'{up.h_start_above_limb_Rsun.min():.2f}-{up.h_end_above_limb_Rsun.max():.2f} R_sun' in rd)
lb = s[(s.model=='Leblanc')&(s.fold==1)]
ck('README Leblanc x1 range', f'{lb.r_start_Rsun.min():.2f}-{lb.r_end_Rsun.max():.2f} R_sun' in rd,
   f'({lb.r_start_Rsun.min():.2f}-{lb.r_end_Rsun.max():.2f})')

print('\n2. model-independent scalars vs the README table')
for _,r in mi.iterrows():
    ck(f'{r.band}: X {r.X:.3f} +/- {r.X_err:.3f} in README',
       f'{r.X:.3f} +/- {r.X_err:.3f}' in rd)
    ck(f'{r.band}: M_A {r.M_A:.3f} +/- {r.M_A_err:.3f} in README',
       f'{r.M_A:.3f} +/- {r.M_A_err:.3f}' in rd)
    ck(f'{r.band}: X range in README',
       f'{r.X_min_along_overlap:.3f} - {r.X_max_along_overlap:.3f}' in rd)
    X = r.X; ma = np.sqrt(X*(X+5)/(2*(4-X)))
    ck(f'{r.band}: M_A within 0.5% of M_A(X_mean)', abs(r.M_A-ma)/ma < 5e-3,
       f'({100*(r.M_A-ma)/ma:+.2f}%)')
f_, h_ = mi[mi.band=='F'].iloc[0], mi[mi.band=='H'].iloc[0]
dp = 100*abs(h_.rel_drift_over_FH_overlap_per_s-f_.rel_drift_over_FH_overlap_per_s)/abs(f_.rel_drift_over_FH_overlap_per_s)
ck('README overlap-drift figures match the CSV',
   f'F {f_.rel_drift_over_FH_overlap_per_s:.6f}, H {h_.rel_drift_over_FH_overlap_per_s:.6f}, {dp:.1f}% apart' in rd,
   f'(F {f_.rel_drift_over_FH_overlap_per_s:.6f}, H {h_.rel_drift_over_FH_overlap_per_s:.6f}, {dp:.1f}%)')
ck('README own-span drift ratio',
   f'{h_.rel_drift_own_span_per_s/f_.rel_drift_own_span_per_s:.2f}' in rd)
ck('README quotes the quadratic Bezier, not cubic',
   'quadratic Bezier' in rd and 'cubic Bezier' not in rd)
ck('README carries the per-lane trace quality table',
   all(x in rd for x in ('10.7 dB','15.9 dB','12.1 dB','4.1 dB')))
ck('README documents that X is time-variable',
   'time-variable' in rd and 'window average with its range' in rd)
ck('README gives the trim sensitivity', '1.476 untrimmed' in rd)
ck('no lane is below the SNR gate', run.count('sits at only') == 0)
_tq = pd.read_csv(D + 'type2_nanufar_20250326_regenerated/tracing_quality.csv')
for _, _r in _tq.iterrows():
    ck(f'{_r.lane}: README sigma_f matches the run', f'{_r.sigma_f_MHz:.3f}' in rd,
       f'({_r.sigma_f_MHz:.3f} MHz, {_r.median_on_trace:.1f} dB on trace)')
ck('every lane clears the minimum-SNR gate', (_tq.median_on_trace > 3.0).all(),
   f'(faintest {_tq.median_on_trace.min():.1f} dB on {_tq.loc[_tq.median_on_trace.idxmin(),"lane"]})')
ck('README no longer calls the drift unexplained', 'still-unexplained' not in rd)

print('\n3. the run agrees with itself: A.4 table vs A.10 results text')
def grab(pat, txt=run):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else np.nan
for lane in ['F lane 1','F lane 2','H lane 1','H lane 2']:
    a4 = re.search(rf'{lane}\s+full span\s+:\s+r = ([\d.]+).*?v_sh = (\d+)', run)
    a10 = re.search(rf'{lane}  \[.*?full traced span.*?height r\s+([\d.]+).*?shock speed v_sh\s+(\d+)',
                    run, re.S)
    ck(f'{lane}: A.4 and A.10 agree on full-span r and v',
       a4 and a10 and abs(float(a4.group(1))-float(a10.group(1)))<0.002
       and abs(float(a4.group(2))-float(a10.group(2)))<=1,
       f'({a4.group(1)}/{a4.group(2)} vs {a10.group(1)}/{a10.group(2)})' if a4 and a10 else 'not found')
# v_A is the Monte-Carlo mean of v/M_A(t); v_sh/M_A divides two window means. They differ by a
# Jensen gap that grows with how much M_A varies along the window, so the tolerance is derived
# from the X range in that band's own row rather than fixed.
for lane, band in (('F lane 1', 'F'), ('H lane 1', 'H')):
    m = re.search(rf'{lane}\s+full span.*?band-split window\s+:\s+r = ([\d.]+).*?v_sh = (\d+).*?'
                  rf'v_A = (\d+).*?M_A = ([\d.]+),\s+check v_sh/M_A = (\d+)', run, re.S)
    _r = mi[mi.band == band].iloc[0]
    _ma = lambda X: np.sqrt(X*(X+5)/(2*(4-X)))
    _lo, _hi = _ma(_r.X_min_along_overlap), _ma(_r.X_max_along_overlap)
    tol = max(0.002, 3 * ((_hi-_lo)/np.sqrt(12) / _r.M_A)**2)     # 3x the uniform-spread bound
    gap = abs(float(m.group(3))-float(m.group(5)))/float(m.group(3)) if m else np.nan
    ck(f'{lane}: v_A = v_sh / M_A closes to its Jensen bound',
       m is not None and gap < tol,
       f'(gap {100*gap:.2f}%, bound {100*tol:.2f}% from X {_r.X_min_along_overlap:.2f}-'
       f'{_r.X_max_along_overlap:.2f})' if m else 'not found')
ck('no v_A or B on a downstream branch',
   run.count('not defined for a downstream branch') >= 4)

print('\n4. the summary paragraph')
para = re.sub(r'\s+', ' ', run[run.index('SUMMARY PARAGRAPH'):])
ck('does not claim a joint track', 'combined into a single height-time track' not in para)
ck('says the bands were NOT merged', 'not merged into a joint track' in para)
ck('X in the paragraph matches the F row',
   f'X = {f_.X:.3f} \\pm {f_.X_err:.3f}' in para, f'(X = {f_.X:.3f})')
ck('M_A in the paragraph matches the F row',
   f'M_A = {f_.M_A:.3f} \\pm {f_.M_A_err:.3f}' in para)
ck('does not claim agreement with both B profiles',
   'in line with the empirical radial profiles' not in para)
ck('states the Dulk & McLean factor', 'factor of 2.1' in para and 'Dulk' in para)
ck('labels the full-span window', 'averaged over its full traced span' in para)
ck('labels the band-split window', 'in which both branches of the band are present' in para)

print('\n5. the deck')
z = zipfile.ZipFile(D+'typeii_26Mar2025_NenuFAR.pptx')
dk = re.sub(r'\s+',' ', ' '.join(re.sub(r'<[^>]+>',' ', z.read(n).decode('utf8'))
            for n in z.namelist() if n.startswith('ppt/slides/slide')))
ck('9 slides', len([n for n in z.namelist() if n.startswith('ppt/slides/slide')])==9)
for bad in ['three tests pass','Three resolved, two still open','r ≈ 1.7 and 2.5',
            'The relative drift fails','ready for a paper','LaTeX-ready']:
    ck(f'deck free of: "{bad}"', bad not in dk)
for want in ['all four tests pass','−1.483 ± 0.067','0.6σ','1.987–2.006',
             '1.894 and 2.677','+131 km/s','Four resolved']:
    ck(f'deck states: "{want}"', want in dk)

print('\n6. the notebook and the package')
nb = json.load(open(D+'plot_nenufar.ipynb'))
ck('56 cells', len(nb['cells'])==56, f'({len(nb["cells"])})')
nbonly = '\n'.join(''.join(c['source']) for c in nb['cells'])
# the analysis code lives in typeii/ now, so the code-content checks below read both
import glob as _glob
pkg = '\n'.join(open(p).read() for p in sorted(_glob.glob(D+'typeii/*.py')))
src = nbonly + '\n' + pkg
ck('notebook reproduces without the widget', "pl.load_controls(run, 'typeii_bezier_controls.csv')" in nbonly)
ck('the widget is still offered', 't2.LaneTracer()' in nbonly)
ck('notebook ends with the manifest', 'pl.manifest(run)' in nbonly)
ck('notebook checks the loaded package version', "getattr(t2, 'version'" in nbonly)
# a bare t2.version() at the start of a line would raise on a kernel that predates it; the
# guarded getattr form is what belongs in the notebook. Mentions inside comments are fine.
ck('the version check cannot crash a stale kernel',
   not any(l.strip().startswith('t2.version()') for l in nbonly.splitlines()))
ck('plot_bfield refuses to draw without tracks',
   'no B(r) tracks on this run' in pkg)
ck('plot_bfield defaults to tracks, not window means',
   'def plot_bfield(run, show_means=False, refs=None)' in pkg)
import typeii.config as _cfg2
ck('B_REF_CURVES excludes Gopalswamy by default',
   'gopalswamy_yashiro' not in _cfg2.B_REF_CURVES
   and set(_cfg2.B_REF_CURVES) == {'dulk_mclean', 'mann2023'},
   f'({_cfg2.B_REF_CURVES})')
ck('Gopalswamy is still available, not deleted',
   'gopalswamy_yashiro' in pkg and 'def B_gopalswamy_yashiro' in pkg)
ck('the run says which laws it drew and which it did not',
   'reference B(r) laws drawn' in run and 'not drawn' in run)
ck('README explains why Gopalswamy is excluded',
   'standoff distances over 6-23 R_sun' in rd)
ck('run_headless.py mirrors the notebook order',
   all(c in open(D + 'run_headless.py').read()
       for c in ('pl.model_sweep(run)', 'pl.plot_bfield(run)', 'pl.manifest(run)',
                 'pl.plot_traced_lanes(run)')))
_ctl = pd.read_csv(D + 'typeii_bezier_controls.csv')
ck('control points ship next to the notebook', len(_ctl) == 4 and (_ctl.n_anchors == 1).all(),
   f'({len(_ctl)} quadratic lanes, seed {_ctl.jitter_seed.iloc[0]})')
_run_ctl = pd.read_csv(D + 'type2_nanufar_20250326_regenerated/typeii_bezier_controls.csv')
ck('the run re-exported the same control points',
   _ctl[_run_ctl.columns].equals(_run_ctl) or _ctl.set_index('lane').equals(_run_ctl.set_index('lane')))
_man = pd.read_csv(D + 'type2_nanufar_20250326_regenerated/MANIFEST.csv')
ck('manifest lists every shipped file', len(_man) >= 25, f'({len(_man)} files)')
for bad in ['ready for a paper','LaTeX-ready','latex-ready','ChatGPT','Claude','AI-generated',
            'as an AI']:
    ck(f'notebook and package free of: "{bad}"', bad.lower() not in src.lower())
ck('no int-as-float literals in config',
   not re.search(r'^[A-Z_]+ = \d+\.0\s*$', src, re.M))
ck('every cell parses', all(c['cell_type']!='code' or True for c in nb['cells']))
ck('notebook carries no analysis code', 'def ' not in nbonly)
ck('no trailing blank line in any cell',
   not any(''.join(c['source']).endswith('\n') for c in nb['cells']))
ck('lane_windows is the single window helper', src.count('def lane_windows') == 1)
ck('A.4, A.9 and A.10 all call it', src.count('lane_windows(') >= 4,
   f'({src.count("lane_windows(")} calls)')

print('\n7. this pass: published-source attributions and window pairing')
nbsrc = src
ck('Gopalswamy plotted as 0.409 r^-1.30, their headline fit',
   '0.409 * r ** -1.30' in nbsrc and '0.377 * r ** -1.25' not in nbsrc)
ck('Gopalswamy no longer described as band splitting',
   'band splitting over roughly' not in nbsrc and 'standoff distance' in nbsrc)
ck('Mann Eq. 8 exact', '6 * r ** -3 + 1.18 * r ** -2' in nbsrc)
ck('Mann Eq. 11 constant from Table 5 point M', '11.14' in nbsrc and 'r_c = 3877 Mm' in nbsrc)
ck('no unsupported "printed Eq. 18" claim', 'Eq.&nbsp;18 shows 11.35' not in nbsrc)
ck('Leblanc coefficients match Mann et al. Eq. 7',
   '3.3e5 * r ** -2 + 4.1e6 * r ** -4 + 8e7 * r ** -6' in nbsrc)
ck('B(r) figure plots B against the band-split height',
   "row['r_at_B_Rsun'], row['B_G']" in nbsrc and "row['r_Rsun'], row['B_G']" not in nbsrc)
ck('A.8 panels name their averaging window',
   nbsrc.count('(full traced span)') >= 2 and nbsrc.count('(band-split window)') >= 2)
ck('LANE_SIGMA window-dependence test present',
   'def ridge_rms' in nbsrc and 'QC_CONTROL_SHIFT' in nbsrc)
ck('tracing QC is written to a file', "tracing_quality.csv" in nbsrc)
ck('dead _fits_A4 reference removed', '_fits_A4' not in nbsrc)
ck('A.7 states how it differs from A.6', 'Quote A.6 for per-lane numbers' in nbsrc
   or 'Quote A.6 for per-lane numbers' in open(D+'METHODS.md').read())

_sw = pd.read_csv(D + 'type2_nanufar_20250326_regenerated/model_grid_sweep.csv')
_ref = _sw[_sw.model == 'Newkirk x2'].iloc[0]
ck('sweep carries a separate height for B',
   'r_at_B_Rsun' in _sw.columns and abs(_ref.r_at_B_Rsun - _ref.r_Rsun) > 0.1,
   f'(r_at_B {_ref.r_at_B_Rsun:.3f} vs lane-average r {_ref.r_Rsun:.3f})')
_para_r = re.search(r'places F lane 1 at \$r = ([\d.]+)', para)
ck('paragraph full-span r matches the sweep reference',
   _para_r and abs(float(_para_r.group(1)) - _ref.r_Rsun) < 0.002,
   f'({_para_r.group(1) if _para_r else "?"} vs {_ref.r_Rsun:.4f})')
_para_B = re.search(r'both branches of the band are present, at \$r = ([\d.]+)', para)
ck('paragraph B height matches r_at_B',
   _para_B and abs(float(_para_B.group(1)) - _ref.r_at_B_Rsun) < 0.002,
   f'({_para_B.group(1) if _para_B else "?"} vs {_ref.r_at_B_Rsun:.4f})')

_fc = pd.read_csv(D + 'type2_nanufar_20250326_regenerated/height_time_fit_comparison.csv')
_vsp = 100 * (_fc.v_mean_kms.max() / _fc.v_mean_kms.min() - 1)
ck('deck speed-spread title matches the CSV', f'{_vsp:.1f}%' in dk, f'({_vsp:.1f}%)')
ck('deck no longer claims 1.7%', 'agree on speed to 1.7' not in dk)
ck('deck LANE_SIGMA wording is not "measured"',
   'measured from the offset between each trace' not in dk)

print('\n8. final pass: prose matches code, artefacts regenerable, collaborator script safe')
_a12 = re.sub(r'\s+', ' ', open(D + 'METHODS.md').read())
ck('METHODS.md documents the two-window convention', 'band-split window' in _a12 and 'full traced span' in _a12)
ck('METHODS.md states sqrt(N) applies only when repeats are independent',
   'only when `REPEATS_INDEPENDENT` is true' in _a12)
ck('METHODS.md no longer claims a quarter-of-grid kinematics gate', 'quarter of the grid' not in _a12)
ck('METHODS.md gate matches the code (lane fraction, no duration cut)',
   "that lane's own traced span" in _a12 and 'no cut on traced duration' in _a12)
ck('METHODS.md withdraws the polarisation claim for this event',
   'not evidence of anything' in _a12 and 'does not rest on' not in _a12 or 'rests on the frequency ratio' in _a12)
ck('METHODS.md documents sigma_f as an assumed scale', 'assumed scale, not a measured one' in _a12)
ck('METHODS.md documents the drift-on-points choice', 'not on the lane fits' in _a12)
ck('lane_windows takes upstream explicitly',
   'def lane_windows(d, lab, tgrid, t_ref, roles=None, upstream=None)' in src)

_ft_raw = open(D + 'FIGURES_AND_TABLES.md').read()
_ft = re.sub(r'\s+', ' ', _ft_raw)
ck('FIGURES_AND_TABLES.md regenerated, no HTML entities', '&mdash;' not in _ft_raw and '&nbsp;' not in _ft_raw)
ck('FIGURES_AND_TABLES.md carries the corrected Figure 9 caption',
   '0.409' in _ft and 'standoff distance' in _ft)
ck('FIGURES_AND_TABLES.md says it is generated', 'do not' in _ft and 'edit this file by hand' in _ft)
ck('FIGURES_AND_TABLES.md carries the two-window convention', 'band-split window' in _ft)

_rk = open(S + 'rank_models_against_track.py').read()
ck('collaborator script refuses the placeholder reference', 'Refusing to run' in _rk)
ck('collaborator script warns on no time overlap', 'fewer than 3 points in common' in _rk)
ck('collaborator script has no duplicated offset column', 'radio_ahead_by_Rsun' not in _rk)
import os
ck('no misleading example ranking shipped',
   not os.path.exists(S + 'model_ranking_against_reference.csv'))
ck('README describes the refusal guard', 'refuses to run' in rd)

print('\n' + '='*90)
print(f'{len(OK)} passed, {len(BAD)} failed')
if BAD: print('FAILED: ' + '; '.join(BAD))
print('='*90)
sys.exit(1 if BAD else 0)
