"""Rank the density models against an independently measured height-time track.

The radio data give f(t) exactly, and f -> n_e exactly. The step that is not determined by the
radio data is n_e -> r, which needs an assumed coronal density profile. An independently measured
height supplies that information from outside, so overplotting an EUV or coronagraph track selects
the density model that is consistent with this event.

Drop your AIA / COR2 points (or the coefficients of your linear fit) into REFERENCE below and run.
Nothing here needs the notebook or the raw spectra; it reads only height_time_all_models.csv.

    python rank_models_against_track.py
"""
import numpy as np
import pandas as pd

TRACKS = 'height_time_all_models.csv'

# ------------------------------------------------------------------ your reference track
# Either give the measured points directly ...
REFERENCE = {
    'time_UT': [],            # e.g. ['2025-03-26T09:12:00', '2025-03-26T11:24:00', ...]
    'height': [],             # matching heights, in the units named by REFERENCE_CONVENTION
}
# ... or, if you would rather use the straight line already fitted through AIA + COR2, give two
# points on that line and leave REFERENCE empty.
#
# The values below are a PLACEHOLDER, not a measurement. The script refuses to run until they are
# replaced, because a wrong reference track still produces a full ranking table that looks like a
# result - there is nothing in the output to tell you the input was invented.
FIT_THROUGH = {
    'time_UT': ['2025-03-26T09:05:00', '2025-03-26T13:50:00'],
    'height': [0, 11],
}
PLACEHOLDER_TIMES = ['2025-03-26T09:05:00', '2025-03-26T13:50:00']
PLACEHOLDER_HEIGHTS = [0, 11]

# WHICH HEIGHT ARE YOUR NUMBERS? This is the one setting that matters most. The two conventions
# differ by exactly 1 Rsun, which is larger than the whole spread between the five density models,
# so getting it wrong will make a model look right for the wrong reason.
#   'above_limb'    h = r - 1 Rsun, zero at the photosphere. Usual for EUV and coronagraph work.
#   'heliocentric'  r measured from Sun centre, 1 Rsun at the photosphere. Usual for radio work.
REFERENCE_CONVENTION = 'above_limb'

# Compare against this lane. The upstream (lower) branch of each band is the shock front proper.
LANES = ['F lane 1', 'H lane 1']


def reference_curve(t_sec_grid, t0):
    """Reference height on the radio track's own time grid, in the ABOVE-LIMB convention."""
    if len(REFERENCE['time_UT']) >= 2:
        tt, hh = REFERENCE['time_UT'], REFERENCE['height']
    else:
        tt, hh = FIT_THROUGH['time_UT'], FIT_THROUGH['height']
    ts = (pd.to_datetime(pd.Series(tt)) - t0).dt.total_seconds().to_numpy()
    h = np.interp(t_sec_grid, ts, np.asarray(hh, float),
                  left=np.nan, right=np.nan) if len(ts) > 2 else \
        np.asarray(hh, float)[0] + (np.asarray(hh, float)[1] - np.asarray(hh, float)[0]) \
        * (t_sec_grid - ts[0]) / (ts[1] - ts[0])
    return h - 1 if REFERENCE_CONVENTION == 'heliocentric' else h


def check_reference_is_real():
    """Stop if the shipped placeholder is still in place.

    A wrong reference track does not fail loudly: it produces a complete, plausible-looking
    ranking table, and nothing in that table records that the input was invented. So the check
    has to happen here, before any of it is computed."""
    if len(REFERENCE['time_UT']) >= 2:
        if len(REFERENCE['time_UT']) != len(REFERENCE['height']):
            raise SystemExit('REFERENCE: time_UT and height have different lengths.')
        return
    if (FIT_THROUGH['time_UT'] == PLACEHOLDER_TIMES
            and list(FIT_THROUGH['height']) == PLACEHOLDER_HEIGHTS):
        raise SystemExit(
            'Refusing to run: FIT_THROUGH still holds the shipped placeholder '
            f'({PLACEHOLDER_TIMES[0]} at h = {PLACEHOLDER_HEIGHTS[0]} to '
            f'{PLACEHOLDER_TIMES[1]} at h = {PLACEHOLDER_HEIGHTS[1]} Rsun), which is a guess and '
            'not a measurement.\nPut your own AIA / COR2 points in REFERENCE, or two points from '
            'your fitted line in FIT_THROUGH, and check REFERENCE_CONVENTION.')


def main():
    check_reference_is_real()
    d = pd.read_csv(TRACKS, parse_dates=['time_UT'])
    t0 = d.time_UT.min()
    out, skipped = [], []
    for lane in LANES:
        sub = d[d.lane == lane]
        if not len(sub):
            continue
        for mf, g in sub.groupby('model_fold'):
            ts = (g.time_UT - t0).dt.total_seconds().to_numpy()
            ref = reference_curve(ts, t0)
            dif = g.h_above_limb_Rsun.to_numpy() - ref
            ok = np.isfinite(dif)
            if ok.sum() < 3:
                skipped.append((lane, mf))
                continue
            out.append({'lane': lane, 'model_fold': mf,
                        # positive = the radio track sits ABOVE the reference
                        'mean_offset_Rsun': float(np.mean(dif[ok])),
                        'rms_Rsun': float(np.sqrt(np.mean(dif[ok] ** 2))),
                        'n_points_compared': int(ok.sum()),
                        'v_radio_kms': float(np.nanmean(g.v_kms))})
    if skipped:
        print(f'WARNING: {len(skipped)} of {len(skipped) + len(out)} lane x model combinations '
              'had fewer than 3 points in common with your reference track.')
        print('         Check that its time range overlaps 09:19-09:52 UT on 2025-03-26 and that')
        print('         REFERENCE_CONVENTION matches the units your heights are in.')
    if not out:
        raise SystemExit('Nothing to rank: the reference track and the radio track do not overlap '
                         'in time.')
    r = pd.DataFrame(out)
    r.to_csv('model_ranking_against_reference.csv', index=False)

    for lane, g in r.groupby('lane'):
        print(f'\n{lane}  —  closest density models to the reference track')
        print(f'{"model x fold":>18}{"rms [Rsun]":>12}{"mean offset":>13}{"v [km/s]":>10}')
        for _, x in g.sort_values('rms_Rsun').head(8).iterrows():
            print(f'{x.model_fold:>18}{x.rms_Rsun:12.3f}{x.mean_offset_Rsun:+13.3f}'
                  f'{x.v_radio_kms:10.0f}')
    print('\nA POSITIVE mean offset means the radio track sits ABOVE the reference, which is what')
    print('a shock running ahead of the CME leading edge should do. Prefer a model that is')
    print('slightly positive and parallel over one that merely minimises the rms.')


if __name__ == '__main__':
    main()
