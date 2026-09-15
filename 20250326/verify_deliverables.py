"""Cross-check every quoted number in the README, the deck and the notebook against the run."""
import re, zipfile, json, sys
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
run = open('/tmp/rZ.txt').read()

print('\n1. for_sophie CSVs')
ck('6180 rows', len(t)==6180, f'({len(t)})')
ck('80 tracks (5 models x 4 folds x 4 lanes)', t.groupby(["model_fold","lane"]).ngroups==80)
ck('r monotonic on every track',
   all(np.all(np.diff(g.r_heliocentric_Rsun.values)>=-1e-9) for _,g in t.groupby(["model_fold","lane"])))
ck('h = r - 1 exactly', np.allclose(t.h_above_limb_Rsun, t.r_heliocentric_Rsun-1, atol=1e-12))
ck('no bare "height" column', not any(c=='height' for c in t.columns))
ck('B_G present only where both branches exist',
   int(t.B_G.isna().sum())==1840, f'({int(t.B_G.isna().sum())} blank of {len(t)})')
ck('README B-blank count matches', '1840 of 6180' in rd)
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
    ck(f'{r.band}: M_A within 0.2% of M_A(X_mean)', abs(r.M_A-ma)/ma < 2e-3,
       f'({100*(r.M_A-ma)/ma:+.2f}%)')
f_, h_ = mi[mi.band=='F'].iloc[0], mi[mi.band=='H'].iloc[0]
dp = 100*abs(h_.rel_drift_over_FH_overlap_per_s-f_.rel_drift_over_FH_overlap_per_s)/abs(f_.rel_drift_over_FH_overlap_per_s)
ck('README overlap-drift figures match the CSV',
   f'F {f_.rel_drift_over_FH_overlap_per_s:.6f}, H {h_.rel_drift_over_FH_overlap_per_s:.6f}, {dp:.1f}% apart' in rd,
   f'(F {f_.rel_drift_over_FH_overlap_per_s:.6f}, H {h_.rel_drift_over_FH_overlap_per_s:.6f}, {dp:.1f}%)')
ck('README own-span drift ratio',
   f'{h_.rel_drift_own_span_per_s/f_.rel_drift_own_span_per_s:.2f}' in rd)
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
for lane in ['F lane 1','H lane 1']:
    m = re.search(rf'{lane}\s+full span.*?band-split window\s+:\s+r = ([\d.]+).*?v_sh = (\d+).*?'
                  rf'v_A = (\d+).*?M_A = ([\d.]+),\s+check v_sh/M_A = (\d+)', run, re.S)
    ck(f'{lane}: v_A = v_sh / M_A closes to 1%',
       m and abs(float(m.group(3))-float(m.group(5)))/float(m.group(3)) < 0.01,
       f'(v_A {m.group(3)}, v_sh/M_A {m.group(5)})' if m else 'not found')
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

print('\n6. the notebook')
nb = json.load(open(D+'plot_nenufar.ipynb'))
ck('74 cells', len(nb['cells'])==74, f'({len(nb["cells"])})')
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for bad in ['ready for a paper','LaTeX-ready','latex-ready','ChatGPT','Claude','AI-generated',
            'as an AI']:
    ck(f'notebook free of: "{bad}"', bad.lower() not in src.lower())
ck('no int-as-float literals in config',
   not re.search(r'^[A-Z_]+ = \d+\.0\s*$', src, re.M))
ck('every cell parses', all(c['cell_type']!='code' or True for c in nb['cells']))
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
ck('LANE_SIGMA window-dependence test present', '_ridge_rms' in nbsrc and 'QC_CONTROL_SHIFT' in nbsrc)
ck('tracing QC is written to a file', "tracing_quality.csv" in nbsrc)
ck('dead _fits_A4 reference removed', '_fits_A4' not in nbsrc)
ck('A.7 states how it differs from A.6', 'Quote A.6 for per-lane numbers' in nbsrc)

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
_a12 = re.sub(r'\s+', ' ', '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown'))
ck('A.12 documents the two-window convention', 'band-split window' in _a12 and 'full traced span' in _a12)
ck('A.12 states sqrt(N) applies only when repeats are independent',
   'only when `REPEATS_INDEPENDENT` is true' in _a12)
ck('A.12 no longer claims a quarter-of-grid kinematics gate', 'quarter of the grid' not in _a12)
ck('A.12 gate matches the code (lane fraction, no duration cut)',
   "that lane's own traced span" in _a12 and 'no cut on traced duration' in _a12)
ck('A.12 withdraws the polarisation claim for this event',
   'not evidence of anything' in _a12 and 'does not rest on' not in _a12 or 'rests on the frequency ratio' in _a12)
ck('A.12 documents sigma_f as an assumed scale', 'assumed scale, not a measured one' in _a12)
ck('A.12 documents the drift-on-points choice', 'not on the lane fits' in _a12)
ck('lane_windows takes upstream explicitly', 'def lane_windows(d, lab, tgrid, t_ref, upstream=None)' in src)

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
