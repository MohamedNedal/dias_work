def build_layer(run):
    """Build the tracing layer: window, decimate, and record the channel width."""
    # from the run
    (dyspec_subtracted, BEZIER_JITTER, df_pol, df_int) = (
        run.dyspec_subtracted, run.BEZIER_JITTER, run.df_pol, run.df_int
    )

    # Stokes I tracing layer, straight from the background-subtracted frame computed above
    if LAYER_MODE == 'db_sub':
        src_I = dyspec_subtracted
    else:
        lin = 10 ** (df_int / 10)
        src_I = lin / np.nanmedian(lin.to_numpy(), axis=0)

    LAYER, (kt_I, kf_I) = decimate(window(run, src_I))
    POL, (kt_P, kf_P) = decimate(window(run, df_pol))
    run.set(POL=POL)
    LAYER_T = LAYER.index
    run.set(LAYER_T=LAYER_T)
    LAYER_F = np.asarray(LAYER.columns, float)
    run.set(LAYER_F=LAYER_F)
    LAYER_D = LAYER.to_numpy().T                       # (nfreq, ntime)
    run.set(LAYER_D=LAYER_D)

    print(f'Stokes I layer: {LAYER.shape[0]} time x {LAYER.shape[1]} freq '
          f'(decimated {kt_I}x in time, {kf_I}x in frequency)')
    print(f'  {LAYER_T[0].time()}-{LAYER_T[-1].time()} UT, {LAYER_F.min():.1f}-{LAYER_F.max():.1f} MHz,'
          f' cadence {np.nanmedian(np.diff(LAYER_T) / np.timedelta64(1, "ms")):.0f} ms,'
          f' channel {np.nanmedian(np.diff(LAYER_F)) * 1e3:.0f} kHz')
    print(f'Stokes V/I layer: {POL.shape[0]} time x {POL.shape[1]} freq '
          f'(decimated {kt_P}x in time, {kf_P}x in frequency)')

    # --- convert the target Bezier jitter from MHz into channels of this layer -------------------
    # Jittering all four control points of a cubic independently by sigma does NOT move the curve by
    # sigma: the curve is a weighted average of them, so the displacement is attenuated. For a cubic
    # the rms attenuation over the curve is sqrt(int[(1-t)^6 + 9(1-t)^4 t^2 + 9(1-t)^2 t^4 + t^6] dt)
    # = sqrt(0.4571) = 0.676. Divide it out so the achieved spread lands on the target.
    CHAN_MHZ = float(np.nanmedian(np.diff(LAYER_F)))
    run.set(CHAN_MHZ=CHAN_MHZ)
    BEZIER_CURVE_ATTEN = 0.676
    if BEZIER_JITTER is None:
        BEZIER_JITTER = round(BEZIER_JITTER_MHZ / (CHAN_MHZ * BEZIER_CURVE_ATTEN), 1)
        print(f'\nBEZIER_JITTER = {BEZIER_JITTER:.1f} channels, derived from '
              f'BEZIER_JITTER_MHZ = {BEZIER_JITTER_MHZ} MHz')
        print(f'  ({CHAN_MHZ * 1e3:.0f} kHz per channel, divided by the {BEZIER_CURVE_ATTEN} cubic '
              'attenuation). The traced-lanes figure reports the spread actually achieved.')
    run.set(BEZIER_JITTER=BEZIER_JITTER)


def plot_stretch_comparison(run):
    """Compare display stretches side by side."""
    LAST_STRETCH = {}




    CBAR_LABEL = 'dB above background' if LAYER_MODE == 'db_sub' else 'ratio to background'
    run.set(CBAR_LABEL=CBAR_LABEL)

    # --- compare a few stretches: pick one and set LAYER_PLO / LAYER_PHI / LAYER_GAMMA above ---
    TRIALS = [(60, 99.5, 1, 'plo=60, gamma=1  (harsh: faint detail clipped)'),
              (2, 98, 1, 'plo=2, gamma=1'),
              (2, 98, 0.6, 'plo=2, gamma=0.6  (default)'),
              (1, 99, 0.4, 'plo=1, gamma=0.4  (softest)')]
    fig = plt.figure(figsize=[15, 12])
    for i, (plo, phi, gam, ttl) in enumerate(TRIALS, start=1):
        ax = fig.add_subplot(len(TRIALS), 1, i)
        pm = draw_layer(run, ax, plo=plo, phi=phi, gamma=gam)
        fig.colorbar(pm, ax=ax, pad=0.01, label=CBAR_LABEL)
        ax.set_title(ttl, fontsize=10)
        if i < len(TRIALS):
            ax.set_xlabel('')
    fig.suptitle('Choosing the display stretch', y=1.005, fontsize=13)
    fig.tight_layout()
    save_fig(fig, 'display_stretch_comparison')
    plt.show()


def plot_window(run):
    """The tracing layer over the type II window."""
    # from the run
    CBAR_LABEL = run.CBAR_LABEL

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    pm = draw_layer(run, ax)
    fig.colorbar(pm, ax=ax, pad=0.01, label=CBAR_LABEL)
    ax.set_title(f'NenuFAR type II tracing layer  (plo={LAST_STRETCH["plo"]}, '
                 f'phi={LAST_STRETCH["phi"]}, gamma={LAST_STRETCH["gamma"]})')
    fig.tight_layout()
    save_fig(fig, 'nenufar_typeii_window')
    plt.show()


def build_model_grid(run):
    """Assemble the density models at every fold."""
    BASE_MODELS = {'Newkirk': newkirk, 'Saito': saito, 'Leblanc': leblanc,
                   'Baumbach-Allen': baumbach_allen, 'Mann 2023': mann2023}
    run.set(BASE_MODELS=BASE_MODELS)

    MODEL_GRID = {f'{name} x{fold}': (lambda r, _f=fun, _k=fold: _f(r, fold=_k))
                  for name, fun in BASE_MODELS.items() for fold in FOLDS}
    run.set(MODEL_GRID=MODEL_GRID)
    print(f'{len(BASE_MODELS)} models x {len(FOLDS)} folds = {len(MODEL_GRID)} combinations')


def plot_density_models(run):
    """Density models and the emission frequency they imply."""
    # from the run
    BASE_MODELS = run.BASE_MODELS

    rr = np.linspace(1, 3, 400)
    fig = plt.figure(figsize=[14, 5])

    ax = fig.add_subplot(121)
    for name, fun in BASE_MODELS.items():
        ax.plot(rr, fun(rr), lw=1.8, label=name)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\,/\,R_\odot$')
    ax.set_ylabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_title('(a) Electron-density models (fold 1)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    ax = fig.add_subplot(122)
    for b in BANDS:
        for name, fun in BASE_MODELS.items():
            ax.plot(rr, HARM[b] * PLASMA_CONST * np.sqrt(fun(rr)) / 1e6, lw=1.6,
                    ls=('-' if b == 'F' else '--'), label=(name if b == 'F' else None))
    ax.axhspan(TYPEII_FLIM[0], TYPEII_FLIM[1], color='0.8', alpha=0.5, zorder=0)
    ax.text(2.85, np.sqrt(TYPEII_FLIM[0] * TYPEII_FLIM[1]), 'NenuFAR band', ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(8, 400)
    ax.set_xlabel(r'$r\,/\,R_\odot$')
    ax.set_ylabel(r'$s\,f_{pe}$ [MHz]')
    ax.set_title(r'(b) Emission frequency vs height: $s=1$ solid, $s=2$ dashed')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    save_fig(fig, 'density_models')
    plt.show()


def collect_traces(run):
    """Collect the traced lanes and report what was recorded."""
    # from the run
    tracer, BEZIER_JITTER = run.tracer, run.BEZIER_JITTER

    # ---- collect the traces -------------------------------------------------------------------
    passes = tracer.passes
    run.set(passes=passes)
    if not passes:
        raise RuntimeError('nothing traced yet - trace the lanes in the figure above, press '
                           '"End trace" for each, then re-run this cell')

    TRACED = [lab for lab in tracer.labels if any(p.get(lab, {}).get('f') for p in passes)]
    run.set(TRACED=TRACED)
    LANE_COL = {lab: tracer.colour(lab) for lab in TRACED}
    LANE_COL[JOINT] = 'black'
    run.set(LANE_COL=LANE_COL)
    LANE_BAND = {lab: lab.split()[0] for lab in TRACED}
    run.set(LANE_BAND=LANE_BAND)
    BANDS_TRACED = [b for b in BANDS if b in set(LANE_BAND.values())]
    run.set(BANDS_TRACED=BANDS_TRACED)

    # say plainly what came out of the store, and complain if it does not look like a traced event
    n_by_band = {b: sum(1 for l in TRACED if l.startswith(b + ' ')) for b in BANDS}
    print(f'{len(passes)} pass(es) over {len(TRACED)} lane(s): '
          + ', '.join(f'{n_by_band[b]} x {BAND_NAME[b]}' for b in BANDS))
    for b in BANDS:
        if n_by_band[b] == 0:
            print(f'  WARNING: no {BAND_NAME[b]} lane recorded - the {b} band diagnostics will be '
                  'skipped. Switch the Band toggle and trace it, then re-run this cell.')
        elif n_by_band[b] == 1:
            print(f'  NOTE: only one {BAND_NAME[b]} lane recorded, so that band has no split, '
                  'hence no X or M_A. Trace its other split branch if the spectrum shows one.')
    incomplete = [l for l in TRACED if len(tracer.traces[l]) < tracer.n_reps]
    if incomplete:
        print(f'  NOTE: fewer than {tracer.n_reps} repeats on: ' + ', '.join(incomplete))
    AUTO_REPEAT = [l for l in TRACED if TRACE_KIND.get(l) == 'bezier auto-repeats']
    run.set(AUTO_REPEAT=AUTO_REPEAT)

    # Are the repeats independent measurements? Only if every lane was re-traced by hand. This drives
    # whether the aggregation is allowed to divide the spread by sqrt(N_pass) in A.4: averaging N
    # jittered copies of one curve returns the curve you drew, so it buys no precision and the
    # sqrt(N) is unearned.
    # A lane whose repeats are byte-identical is not N measurements either. Pressing "End trace"
    # repeatedly in Bezier mode without touching the sliders records the same curve every time: the
    # spread is exactly zero, and calling that independent would divide an already-zero error by
    # sqrt(N) and then propagate it as though the lane had been measured N times.
    DUPLICATE_REPS = []
    run.set(DUPLICATE_REPS=DUPLICATE_REPS)
    for _l in TRACED:
        _r = tracer.traces.get(_l, [])
        if len(_r) > 1:
            _n = min(len(x['f']) for x in _r)
            _st = np.vstack([np.asarray(x['f'], float)[:_n] for x in _r])
            if np.allclose(_st, _st[0], rtol=0, atol=1e-9):
                DUPLICATE_REPS.append(_l)
    if DUPLICATE_REPS:
        print(f'\n  WARNING: {", ".join(DUPLICATE_REPS)} have IDENTICAL repeats - the same curve was '
              'recorded more\n        than once, so their spread is exactly zero. That is not a '
              'reproducibility measurement.\n        Move the curve between presses, or tick '
              'auto-repeats.')

    REPEATS_INDEPENDENT = (len(AUTO_REPEAT) == 0 and len(DUPLICATE_REPS) == 0)
    run.set(REPEATS_INDEPENDENT=REPEATS_INDEPENDENT)
    if AUTO_REPEAT:
        print(f'\n  NOTE: {", ".join(AUTO_REPEAT)} used Bezier AUTO-REPEATS. Those repeats are the '
              f'same curve with its\n        control points jittered by BEZIER_JITTER = '
              f'{BEZIER_JITTER:.1f} channels, so their spread measures that jitter,\n        NOT how '
              'reproducibly you can place a curve on the lane. The error bars for those lanes are\n'
              '        therefore fit-error-dominated and optimistic. For a genuine reproducibility '
              'error, untick\n        "auto-repeats" and place the curve independently '
              f'{tracer.n_reps} times.')
        print('        Because of this the aggregation in A.4 will NOT divide the spread by '
              'sqrt(N_pass).')

    tracer.summary()


def plot_traced_lanes(run):
    """Traced lanes over the spectrum, with the repeat spread magnified, and the tracing QC."""
    # from the run
    (TRACED, CBAR_LABEL, OUTDIR, passes, LAYER_D, LAYER_F, LANE_COL, LAYER_T) = (
        run.TRACED, run.CBAR_LABEL, run.OUTDIR, run.passes, run.LAYER_D, run.LAYER_F,
        run.LANE_COL, run.LAYER_T
    )

    # The repeat spread is a few tenths of a MHz. Over a 60 MHz frequency axis that is thinner than
    # the lines themselves, so on the full spectrum the repeats cannot separate no matter how they are
    # styled. The bottom row therefore re-plots each lane over a short window at a frequency scale
    # where the spread is wider than the linewidth: same curves, same data, readable magnification.
    _nz = max(len(TRACED), 1)
    fig = plt.figure(figsize=[15, 9])
    gs = fig.add_gridspec(2, _nz, height_ratios=[2.6, 1], hspace=0.28, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    pm = draw_layer(run, ax)
    fig.colorbar(pm, ax=ax, pad=0.01, label=CBAR_LABEL)

    # all repeats at one linewidth, plus the envelope they span: no repeat is privileged, and the
    # shaded band is the spread itself rather than a claim about it
    _spread, _zoom = {}, {}
    for lab in TRACED:
        reps = [p[lab] for p in passes if p.get(lab) and p[lab]['f']]
        if not reps:
            continue
        for k, L in enumerate(reps):
            ax.plot(pd.to_datetime(L['t']), L['f'], color=LANE_COL[lab], lw=1.3, alpha=0.9,
                    label=(lab if k == 0 else None))
        tref = mdates.date2num(pd.to_datetime(reps[0]['t']))
        stack = np.vstack([np.interp(tref, mdates.date2num(pd.to_datetime(L['t'])),
                                    np.asarray(L['f'], float)) for L in reps])
        if len(reps) > 1:
            ax.fill_between(pd.to_datetime(reps[0]['t']), stack.min(0), stack.max(0),
                            color=LANE_COL[lab], alpha=0.35, lw=0)
            _spread[lab] = float(np.nanmean(np.nanstd(stack, axis=0)))
        _zoom[lab] = (reps, tref, stack)

    _rms = float(np.nanmean(list(_spread.values()))) if _spread else np.nan
    ax.set_title('Traced lanes, all repeats overplotted'
                 + (f'  -  repeat spread {_rms:.3f} MHz rms (magnified below)'
                    if np.isfinite(_rms) else ''))
    ax.legend(fontsize=8, ncol=4, loc='lower left')

    # --- one magnified window per lane, centred where the lane is ------------------------------
    for j, lab in enumerate(TRACED):
        az = fig.add_subplot(gs[1, j])
        if lab not in _zoom:
            az.set_axis_off()
            continue
        reps, tref, stack = _zoom[lab]
        mid = len(tref) // 2
        half_s = max(30, 0.12 * (tref[-1] - tref[0]) * 86400 / 2)
        t_c = mdates.num2date(tref[mid]).replace(tzinfo=None)
        sel = np.abs((tref - tref[mid]) * 86400) <= half_s
        f_c = float(np.nanmean(stack[:, sel]))
        # frequency half-range: whichever is larger, 4x the spread or a 0.3 MHz floor, so the band
        # fills a useful part of the panel instead of collapsing to a line
        half_f = max(4 * _spread.get(lab, 0.1), 0.3)
        pmz = draw_layer(run, az, tstep=1)
        for L in reps:
            az.plot(pd.to_datetime(L['t']), L['f'], color=LANE_COL[lab], lw=1.3, alpha=0.95)
        az.fill_between(pd.to_datetime(reps[0]['t']), stack.min(0), stack.max(0),
                        color=LANE_COL[lab], alpha=0.3, lw=0)
        az.set_xlim(t_c - pd.Timedelta(seconds=half_s), t_c + pd.Timedelta(seconds=half_s))
        az.set_ylim(f_c + half_f, f_c - half_f) if INVERT_FREQ else az.set_ylim(f_c - half_f,
                                                                               f_c + half_f)
        az.set_title(f'{lab}: {len(reps)} repeats, '
                     + (f'{_spread[lab]:.3f} MHz rms' if lab in _spread else 'single trace'),
                     fontsize=9)
        # three ticks at most: a 60-90 s window at %H:%M:%S overruns the panel width otherwise, and
        # these labels are not to be rotated
        az.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))
        az.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        az.tick_params(labelsize=7.5)
        az.set_xlabel('')
        az.set_ylabel('MHz' if j == 0 else '', fontsize=8)
        az.grid(alpha=0.25)
    if not _spread:
        ax.text(0.5, 0.03, 'one repeat per lane - no spread to show',
                transform=ax.transAxes, ha='center', fontsize=9, color='0.3')
    save_fig(fig, 'traced_lanes_preview')
    plt.show()

    # --- tracing quality: how far is each traced point from the local intensity ridge? ---
    # worth checking in Bezier mode especially: a Bezier's interior control points are NOT on its
    # curve, so a curve can look plausible while sitting systematically off the lane
    qc = []
    for lab in TRACED:
        L = passes[0][lab]
        ti = np.searchsorted(LAYER_T, pd.to_datetime(L['t']).to_numpy()).clip(0, len(LAYER_T) - 1)
        fi = np.abs(LAYER_F[:, None] - np.asarray(L['f'], float)[None, :]).argmin(axis=0)
        half = max(3, len(LAYER_F) // 60)
        off = []
        for it, jf in zip(ti, fi):
            lo, hi = max(0, jf - half), min(len(LAYER_F), jf + half + 1)
            col = LAYER_D[lo:hi, it]
            if np.isfinite(col).any():
                off.append(int(np.nanargmax(col)) + lo - jf)
        off = np.asarray(off, float)
        # how bright is the layer ON the trace? A lane drawn across empty spectrum has a small
        # offset-from-ridge by accident (the brightest noise pixel in the window is near zero
        # offset), so the offset check alone cannot tell it from a good trace. The intensity can.
        amp = np.array([LAYER_D[jf, it] for it, jf in zip(ti, fi)], float)
        qc.append({'lane': lab, 'median_offset_chan': np.nanmedian(off),
                   'rms_offset_chan': np.sqrt(np.nanmean(off ** 2)),
                   'offset_MHz': np.nanmedian(off) * np.nanmedian(np.diff(LAYER_F)),
                   'median_on_trace': np.nanmedian(amp), 'min_on_trace': np.nanmin(amp)})
    qc = pd.DataFrame(qc)

    # the rms offset from the ridge is the honest 1-sigma frequency uncertainty of a traced point:
    # it is how far the curve actually sits from the emission it is meant to follow
    _chan = float(np.nanmedian(np.diff(LAYER_F)))
    if LANE_SIGMA_MHZ == 'auto':
        LANE_SIGMA = {r['lane']: max(abs(r['rms_offset_chan']) * _chan, LANE_SIGMA_FLOOR)
                      for _, r in qc.iterrows()}
    elif LANE_SIGMA_MHZ is None:
        LANE_SIGMA = {}
    else:
        LANE_SIGMA = {lab: float(LANE_SIGMA_MHZ) for lab in TRACED}
    run.set(LANE_SIGMA=LANE_SIGMA)
    qc['sigma_f_MHz'] = [LANE_SIGMA.get(l, np.nan) for l in qc['lane']]
    run.set(qc=qc)

    _unit = 'dB above background' if LAYER_MODE == 'db_sub' else 'x background'
    print(f'offset of each traced point from the local intensity maximum '
          f'(search window +/-{half} channels, channel width {_chan * 1e3:.0f} kHz).')
    print('more than a channel or two of median offset means the trace is sitting off the lane.')
    print(f'median_on_trace / min_on_trace are the layer value ON the trace, in {_unit}.')
    print('sigma_f_MHz is the per-point frequency uncertainty the lane fits will use.')
    _faint = qc[qc['median_on_trace'] < QC_MIN_SNR_DB]
    if len(_faint):
        print()
        for _, r in _faint.iterrows():
            print(f'  *** WARNING: {r["lane"]} sits at only {r["median_on_trace"]:.1f} '
                  f'{_unit} along its length (threshold {QC_MIN_SNR_DB}). Part of it is very likely '
                  'drawn across empty spectrum - check it against the traced-lanes figure before '
                  'using it. A low offset-from-ridge does NOT vouch for a trace over empty sky.')




    # Two questions about the number above, because sigma_f sets every statistical error bar in this
    # notebook and it is easy to mistake it for something the data determined on its own.
    #
    #   (1) Does it depend on the search window? A brightest-pixel search has no intrinsic scale: if
    #       the window is widened and the rms grows with it, the "ridge" is not pinning the trace
    #       down and sigma_f is really half/sqrt(3) channels. A uniform distribution over the window
    #       gives exactly that, so it is printed as the null.
    #   (2) Is there a ridge at all? The same measurement is repeated with every lane displaced
    #       QC_CONTROL_SHIFT channels in frequency, onto blank spectrum. If the on-lane rms is not
    #       clearly below the displaced one, the traces are not following emission.
    _halves = sorted({3, 5, half, 2 * half, 3 * half})
    print(f'\nsigma_f depends on the search window. rms offset [channels] vs half-width, against the '
          f'{len(_halves) and ""}uniform-in-window null (half/sqrt3):')
    print(f'  {"half":>5} {"null":>7} ' + ' '.join(f'{l:>11}' for l in TRACED) + '   on/off-lane')
    _ratio_at_half = np.nan
    for _h in _halves:
        _on = [ridge_rms(run, l, _h) for l in TRACED]
        _off = [ridge_rms(run, l, _h, shift_ch=QC_CONTROL_SHIFT) for l in TRACED]
        _rat = np.nanmean(_on) / np.nanmean(_off) if np.nanmean(_off) > 0 else np.nan
        if _h == half:
            _ratio_at_half = _rat
        print(f'  {_h:>5} {_h / np.sqrt(3):>7.2f} ' + ' '.join(f'{v:>11.2f}' for v in _on)
              + f'   {_rat:>6.2f}' + ('   <- used' if _h == half else ''))
    print(f'  The rms grows with the window, so sigma_f is set mainly by the +/-{half}-channel search '
          f'and not by')
    print(f'  a measured trace-to-ridge distance. Every statistical error bar below scales linearly '
          f'with it.')
    if np.isfinite(_ratio_at_half) and _ratio_at_half < 0.95:
        print(f'  But the ridge is real: displacing the lanes {QC_CONTROL_SHIFT} channels onto blank '
              f'spectrum raises the rms by')
        print(f'  {100 * (1 / _ratio_at_half - 1):.0f}%, so the traced curves are following emission '
              f'rather than noise.')
    else:
        print(f'  *** WARNING: displacing the lanes {QC_CONTROL_SHIFT} channels off the emission does '
              f'NOT raise the rms offset')
        print('      (ratio {:.2f}). The ridge finder is returning noise and these traces are not '
              'demonstrably'.format(_ratio_at_half))
        print('      on the lane. Re-check them against the traced-lanes figure before going further.')
    qc.to_csv(os.path.join(OUTDIR, 'tracing_quality.csv'), index=False)
    qc.round(3)


def analyse_lanes(run):
    """Order the branches, fit every lane, and derive the per-band and per-lane quantities."""
    # from the run
    (BANDS_TRACED, TRACED, passes, MODEL_GRID, OUTDIR, t0, LANE_BAND, LANE_SIGMA) = (
        run.BANDS_TRACED, run.TRACED, run.passes, run.MODEL_GRID, run.OUTDIR, run.t0,
        run.LANE_BAND, run.LANE_SIGMA
    )

    # --- which lane of each band is the upstream branch, decided over their overlap -------------
    LANE_ORDER, LANE_FREQ, LANE_ROLE = {}, {}, {}
    run.set(LANE_FREQ=LANE_FREQ, LANE_ORDER=LANE_ORDER)
    for b in BANDS_TRACED:
        labs = [l for l in TRACED if LANE_BAND[l] == b]
        LANE_ORDER[b], key, note = order_lanes(run, labs)
        LANE_FREQ.update(key)
        for i, lab in enumerate(LANE_ORDER[b]):
            LANE_ROLE[lab] = ('upstream (lower)' if i == 0 else
                              'downstream (upper)' if i == 1 else 'extra')
        print(f'{BAND_NAME[b]} (s={HARM[b]}), {note}:')
        for lab in LANE_ORDER[b]:
            print(f'    {lab:12s} {LANE_FREQ[lab]:6.1f} MHz  ->  {LANE_ROLE[lab]}')
        if len(labs) < 2:
            print('    only one lane on this band, so it has no band split, hence no X or M_A')
    run.set(LANE_ROLE=LANE_ROLE)
    SPLIT_PAIR = {b: (v[0], v[1] if len(v) > 1 else None) for b, v in LANE_ORDER.items()}
    run.set(SPLIT_PAIR=SPLIT_PAIR)
    ALL_TRACKS = list(TRACED) + ([JOINT] if MAKE_JOINT else [])
    run.set(ALL_TRACKS=ALL_TRACKS)
    print(f'\nanalysing {len(TRACED)} lane(s) separately'
          + ('; plus the combined F+H track (MAKE_JOINT = True)' if MAKE_JOINT
             else '. Set MAKE_JOINT = True to add the combined F+H track as well.'))

    tg = build_grid(run, passes)
    run.set(tg=tg)
    ALref = aggregate_lanes(run, passes, tg, MODEL_GRID[REF_MODEL_NAME])
    run.set(ALref=ALref)
    scalars = scalar_summary(run, passes)
    run.set(scalars=scalars)
    TRACKS = [k for k in ALL_TRACKS if k in ALref]
    run.set(TRACKS=TRACKS)

    # --- does the polynomial actually describe each traced lane? --------------------------------
    _f0 = pass_fits(run, passes[0])
    LANE_FIT_QC = pd.DataFrame([{'lane': lab, 'sigma_f_MHz': LANE_SIGMA.get(lab, np.nan),
                                 **lane_fit_quality(run, passes[0].get(lab), _f0.get(lab))}
                                for lab in TRACED])
    LANE_FIT_QC['verdict'] = np.where(
        LANE_FIT_QC['resid_over_sigma'] > LANE_FIT_MAX_RESID, 'FLAG: fit misses the trace',
        np.where(LANE_FIT_QC['turnover_s'].notna(), 'FLAG: fit turns over', 'ok'))
    run.set(LANE_FIT_QC=LANE_FIT_QC)
    LANE_FIT_QC.to_csv(os.path.join(OUTDIR, 'lane_fit_quality.csv'), index=False)
    _fitvar = 'log10 f' if FIT_IN_LOGF else 'f'
    print(f'\nlane fit quality (does a degree-{LANE_DEG} polynomial in {_fitvar} '
          'follow the traced curve?)')
    for _, r in LANE_FIT_QC.iterrows():
        print(f'  {r["lane"]:12s} rms residual {r["resid_MHz"]:6.3f} MHz '
              f'= {r["resid_over_sigma"]:5.1f} x sigma_f   {r["verdict"]}')
    for _, r in LANE_FIT_QC[LANE_FIT_QC.verdict.str.startswith('FLAG')].iterrows():
        if np.isfinite(r['turnover_s']):
            print(f'  *** {r["lane"]}: the fitted drift changes sign at '
                  f'{(t0 + pd.Timedelta(seconds=r["turnover_s"])).strftime("%H:%M:%S")} UT, inside '
                  'the traced span. A type II lane drifts one way, so the speed and acceleration '
                  'near the end of this lane are describing the polynomial, not the shock.')
        else:
            print(f'  *** {r["lane"]}: the fit sits {r["resid_MHz"]:.2f} MHz rms off the traced '
                  f'points ({r["resid_over_sigma"]:.0f} x the assumed frequency error). Raise '
                  'LANE_DEG or re-trace this lane; every quantity derived from it is suspect.')

    print(f'\nReference density model: {REF_MODEL_NAME}\n')
    print('model-independent scalars, measured separately from each band')
    for b in BANDS_TRACED:
        s = scalars[b]
        print(f'  {BAND_NAME[b]} (s={HARM[b]}):  '
              f'drift = {s["drift_MHz_s"][0]:+.4f} +/- {s["drift_MHz_s"][1]:.4f} MHz/s,'
              f'  rel. drift = {s["rel_drift_s"][0]:+.5f} +/- {s["rel_drift_s"][1]:.5f} 1/s,'
              f'  X = {s["X"][0]:.3f} +/- {s["X"][1]:.3f},'
              f'  M_A = {s["M_A"][0]:.3f} +/- {s["M_A"][1]:.3f},'
              f'  BDW = {s["rel_bandwidth"][0]:.3f} +/- {s["rel_bandwidth"][1]:.3f}')
        if np.isfinite(s['X_range'][0]):
            print(f'      X runs {s["X_range"][0]:.3f} to {s["X_range"][1]:.3f} along the overlap; '
                  f'the pass-to-pass scatter alone is only +/-{s["X_pass_se"]:.4f}. The quoted error '
                  'is the two combined,')
            print('      because X is being reported as one number for a burst over which it '
                  'demonstrably varies.')
        _bad, _n4 = s.get('X_frac_invalid', np.nan), s.get('X_frac_near4', np.nan)
        if np.isfinite(_bad) and _bad > 0:
            print(f'      *** WARNING: X falls outside 1 <= X < 4 over {100 * _bad:.0f}% of the '
                  'overlap, so M_A, v_A and B are')
            print('      NaN there. Every mean below is a nanmean, so those samples are dropped '
                  'silently and the')
            print('      quoted values describe only the part of the overlap that still works. '
                  'Check the branch ordering.')
        if np.isfinite(_n4) and _n4 > 0:
            print(f'      *** WARNING: X > 3.5 over {100 * _n4:.0f}% of the overlap. M_A diverges as '
                  'X -> 4 (X = 3.99 gives')
            print('      M_A = 42), so v_A = v_sh/M_A collapses and B with it. Values from that '
                  'region are not usable.')

    for b in BANDS_TRACED:
        x = scalars[b]['X'][0]
        if np.isfinite(x) and x < 1:
            print(f'  ERROR: X < 1 for the {BAND_NAME[b]} band. The split branches are the wrong way '
                  'round, so M_A, v_A and B will all be NaN. Check the ordering printed above '
                  'against the traced-lanes figure.')

    KIN_DEG_USED, KIN_BASELINE = {}, {}
    for lab in TRACED:
        d = ALref.get(lab)
        if d is None:
            continue
        sp = np.isfinite(d['f_mean'])
        # Measure the baseline from the lane's OWN traced span, not from the grid points that fall
        # inside it. The shared grid has N_GRID points across the whole burst (~33 s apart here), so
        # a lane is always short by up to two grid steps - enough to push a genuinely 624 s lane
        # under KIN_MIN_BASELINE_S = 600 s and silently drop its acceleration.
        _ff = _f0.get(lab)
        KIN_BASELINE[lab] = (float(_ff['tmax'] - _ff['tmin']) if _ff is not None
                             else (float(np.nanmax(tg[sp]) - np.nanmin(tg[sp])) if sp.any() else 0))
        KIN_DEG_USED[lab] = kin_degree(KIN_N_DENSE if _ff is not None else int(sp.sum()))
    run.set(KIN_BASELINE=KIN_BASELINE, KIN_DEG_USED=KIN_DEG_USED)

    # Every lane is fitted at KIN_DEG; whether its curvature is a measurement is decided by the error
    # bar, not by the traced duration.
    _INV_REF = invert_grid(MODEL_GRID[REF_MODEL_NAME])
    run.set(_INV_REF=_INV_REF)
    A_BIAS = {lab: accel_bias_floor(run, _f0.get(lab), LANE_BAND[lab], _INV_REF) for lab in TRACED}
    run.set(A_BIAS=A_BIAS)
    A_MEASURED = {}
    for lab in TRACED:
        if lab in ALref:
            _a, _ea = grid_scalar(ALref[lab], 'a')
            A_MEASURED[lab] = accel_is_measured(_a, _ea, bias=A_BIAS.get(lab, 0))
    run.set(A_MEASURED=A_MEASURED)

    print(f'\nmodel-dependent quantities on {REF_MODEL_NAME}')
    for lab in TRACED:
        if lab in A_MEASURED and not A_MEASURED[lab]:
            _a, _ea = grid_scalar(ALref[lab], 'a')
            print(f'  NOTE: {lab} gives a = {_a:+.1f} +/- {_ea:.1f} m/s^2 over a {KIN_BASELINE[lab]:.0f} s '
                  f'baseline - under {KIN_A_SIGMA} sigma, so its curvature is NOT SIGNIFICANT.')
            print('        The value and its error are still reported; the lane is not silently '
                  'dropped, and a short')
            print('        baseline is not by itself a reason to refuse the fit.')
    print('  Each lane gets two lines, because the quantities below do not all live on the same')
    print('  interval and averaging them as though they did makes a row contradict itself.')
    print('    full span         r and v_sh over everything the lane traced. A lane covering the')
    print('                      earlier part of a decelerating burst returns a higher mean speed,')
    print('                      so speeds are only comparable between lanes with the same coverage.')
    print('    band-split window the sub-interval where BOTH branches of that band exist. X and M_A')
    print('                      are defined only here, so v_A and B are too; r, v_sh and n_e are')
    print('                      repeated so the row closes v_A = v_sh/M_A on its own numbers.')
    for key in TRACKS:
        d = ALref[key]
        if np.isfinite(d['r_mean']).sum() <= 2:
            print(f'  {key:12s}:  no height solution for this model')
            continue
        w = lane_windows(d, key, tg, t0, roles=LANE_ROLE)
        print(f'  {key:12s} full span         :  r = {w["r_span"]:.3f} +/- {w["r_span_e"]:.3f} Rsun,'
              f'  v_sh = {w["v_span"]:.0f} +/- {w["v_span_e"]:.0f} km/s'
              f'   [{w["span_window"]}, {w["n_span"]} grid pts]')
        if not w['has_split']:
            print(f'  {"":12s} band-split window :  the two branches of this band never overlap in '
                  f'time, so X, M_A, v_A and B are undefined for it')
            continue
        tail = (f'  v_A = {w["vA_split"]:.0f} +/- {w["vA_split_e"]:.0f} km/s,'
                f'  B = {w["B_split"]:.3f} +/- {w["B_split_e"]:.3f} G'
                if w['upstream'] else
                '  v_A, B: not defined for a downstream branch (v_A from Rankine-Hugoniot is the '
                'UPSTREAM Alfven speed)')
        print(f'  {"":12s} band-split window :  r = {w["r_split"]:.3f} +/- {w["r_split_e"]:.3f} Rsun,'
              f'  v_sh = {w["v_split"]:.0f} +/- {w["v_split_e"]:.0f} km/s,{tail}')
        chk = (f',  check v_sh/M_A = {w["v_split"] / w["MA_split"]:.0f} km/s'
               if w['upstream'] and np.isfinite(w['MA_split']) and w['MA_split'] > 0 else '')
        print(f'  {"":12s}                      [{w["split_window"]}, {w["n_split"]} of '
              f'{w["n_span"]} grid pts]  n_e = {w["ne_split"]:.3e} cm^-3, '
              f'M_A = {w["MA_split"]:.3f}{chk}')


def polarisation(run):
    """Stokes V/I along each lane, and whether the bands differ."""
    # from the run
    (TRACED, POL, OUTDIR, t0, passes, LANE_BAND, LANE_ROLE) = (
        run.TRACED, run.POL, run.OUTDIR, run.t0, run.passes, run.LANE_BAND, run.LANE_ROLE
    )

    POL_T = POL.index.to_numpy()
    run.set(POL_T=POL_T)
    POL_F = np.asarray(POL.columns, float)
    run.set(POL_F=POL_F)
    POL_V = POL.to_numpy()
    run.set(POL_V=POL_V)






    pol_rows = []
    POL_TRACK = {}
    for lab in TRACED:
        ref = next(p[lab] for p in passes if p.get(lab, {}).get('f'))
        # resample the lane fit on a dense grid so the polarisation is not read at the click points only
        fit = lane_fit(run, ref)
        ts = np.linspace(fit['tmin'], fit['tmax'], 120)
        tt = [t0 + pd.Timedelta(seconds=float(s)) for s in ts]
        ff = lane_deriv(fit, ts)[0]
        p = sample_polarisation(run, tt, ff)
        POL_TRACK[lab] = dict(t=pd.to_datetime(tt), f=ff, p=p)
        n = int(np.isfinite(p).sum())
        pol_rows.append({'lane': lab, 'band': LANE_BAND[lab], 'role': LANE_ROLE[lab],
                         'V_over_I_mean': np.nanmean(p), 'V_over_I_sd': np.nanstd(p),
                         'V_over_I_sem': np.nanstd(p) / np.sqrt(max(n, 1)),
                         'abs_V_over_I_mean': np.nanmean(np.abs(p)),
                         'abs_V_over_I_max': np.nanmax(np.abs(p)), 'n_samples': n})
    run.set(POL_TRACK=POL_TRACK)

    pol_table = pd.DataFrame(pol_rows)
    run.set(pol_table=pol_table)
    pol_table.to_csv(os.path.join(OUTDIR, 'typeii_polarisation.csv'), index=False)

    # The SIGNED mean is the physical quantity. Taking |V/I| first and then averaging measures the
    # noise, not the polarisation: for a signal buried in noise, mean|x| -> 0.8 sigma whatever the
    # true mean is, so two bands with opposite signed polarisation report the same |V/I|.
    pF = pol_table[pol_table['band'] == 'F']['V_over_I_mean'].mean()
    run.set(pF=pF)
    pH = pol_table[pol_table['band'] == 'H']['V_over_I_mean'].mean()
    run.set(pH=pH)
    noise = pol_table['V_over_I_sd'].median()
    print(f'signed mean V/I:  fundamental {pF:+.4f},  harmonic {pH:+.4f}   '
          f'(point-to-point scatter along a lane {noise:.4f})')
    # The bar for calling the contrast real. |pF - pH| is measured against the point-to-point scatter
    # along a lane, undivided: dividing by sqrt(N) would assume independent samples, and neither the
    # points along a Bezier curve nor the overlapping V/I boxes of neighbouring points are independent.
    # Even undivided it is an UNDERSTATEMENT, because it carries no instrumental-leakage term - which is
    # what polarisation_caveats() tests for, and why a caveat firing withdraws the claim outright.
    _dpol = abs(pF - pH)
    _nsig = _dpol / noise if noise > 0 else np.nan
    _pol_caveats = polarisation_caveats(pol_table)
    print(f'  the two bands differ by {_dpol:.4f}, which is {_nsig:.1f}x that scatter')
    if not np.isfinite(_nsig) or _nsig < 2 or _pol_caveats:
        print('  that is NOT a significant contrast. Fundamental plasma emission escapes in the o-mode '
              'and is expected to be the')
        print('  more strongly polarised of the two, but these numbers neither confirm nor contradict '
              'that expectation, and')
        print('  the band identification below does not rest on them.')
    elif abs(pF) > abs(pH):
        print('  the fundamental is the more strongly polarised, which is the sense expected for '
              'o-mode fundamental emission')
    else:
        print('  the harmonic is the more strongly polarised - unusual, worth a second look at the '
              'tracing and at the RFI stripes in panel (a)')
    print(f'mean |V/I| is also tabulated but do NOT use it to compare the bands: for |V/I| ~ the '
          f'noise it just returns the noise level ({0.8 * noise:.4f} here).')
    for _n in _pol_caveats:
        print(f'  *** WARNING: {_n}')
    print('V/I here is the mean of an ALREADY-DIVIDED ratio with no instrumental-leakage floor '
          'removed. Where Stokes I is near')
    print('the background that ratio is noise-dominated and averaging it pulls the result toward the '
          'noise mean, diluting the')
    print('band-to-band contrast. At the ~1% level measured here, treat these as an upper bound on a '
          'polarisation contrast,')
    print('not as a detection.')
    pol_table.round(4)


def fh_tests(run):
    """The four fundamental / harmonic consistency tests."""
    # from the run
    (passes, MODEL_GRID, SPLIT_PAIR, pF, pH, OUTDIR, scalars, t0) = (
        run.passes, run.MODEL_GRID, run.SPLIT_PAIR, run.pF, run.pH, run.OUTDIR, run.scalars,
        run.t0
    )

    # ---- tests 1-3: ratio, relative drift and height agreement over the overlap ----
    _fits = pass_fits(run, passes[0])
    run.set(_fits=_fits)
    _rr, _ne = invert_grid(MODEL_GRID[REF_MODEL_NAME])
    fh_rows = []






    fF = _fits.get(SPLIT_PAIR['F'][0]) if 'F' in SPLIT_PAIR else None
    fH = _fits.get(SPLIT_PAIR['H'][0]) if 'H' in SPLIT_PAIR else None
    OVERLAP = None
    if fF is None or fH is None:
        print('both bands need at least one traced lane for the F/H consistency tests - skipped')
    else:
        lo, hi = max(fF['tmin'], fH['tmin']), min(fF['tmax'], fH['tmax'])
        OVERLAP = (lo, hi) if hi > lo else None
        if OVERLAP is None:
            print(f'the two bands do not overlap in time ({lo - hi:.0f} s apart); the ratio below is '
                  'EXTRAPOLATED and is much weaker evidence. Re-trace so the bands share some time '
                  'if the spectrum allows it.')
            lo, hi = min(fF['tmin'], fH['tmin']), max(fF['tmax'], fH['tmax'])
        ts = np.linspace(lo, hi, 60)
        f_F, r_F = band_height(run, fF, 'F', ts)
        f_H, r_H = band_height(run, fH, 'H', ts)
        ratio = f_H / f_F
        dres = r_H - r_F
        # the relative drift must be compared ON THE OVERLAP: it changes through the burst, and the
        # band-averaged values in `scalars` are averages over two different stretches of it
        rdF_fit = np.nanmean(lane_deriv(fF, ts)[2])
        rdH_fit = np.nanmean(lane_deriv(fH, ts)[2])
        rdF, rdF_e, rdF_n, rdF_sp = rel_drift_on_points(run, SPLIT_PAIR['F'][0], lo, hi)
        rdH, rdH_e, rdH_n, rdH_sp = rel_drift_on_points(run, SPLIT_PAIR['H'][0], lo, hi)
        if not (np.isfinite(rdF) and np.isfinite(rdH)):      # too few points in the window
            rdF, rdF_e = rdF_fit, np.nan
            rdH, rdH_e = rdH_fit, np.nan

        print(f'{SPLIT_PAIR["F"][0]} vs {SPLIT_PAIR["H"][0]}, '
              f'{(t0 + pd.Timedelta(seconds=lo)).strftime("%H:%M:%S")}'
              f'-{(t0 + pd.Timedelta(seconds=hi)).strftime("%H:%M:%S")} UT  ({hi - lo:.0f} s)\n')
        # If the ratio misses 2, say by how much the two bands would have to be offset IN TIME to
        # meet it. "1.94 instead of 2.00" is a number nobody can act on; "the two bands agree if the
        # harmonic runs 21 s ahead" is a statement about the source that can be checked against the
        # spectrum, and it separates a genuine harmonic-number error (which no shift can repair)
        # from two lanes tracking one shock at slightly different places on the front.
        _shifts = np.arange(-180, 181, 1)
        _obj = [abs(np.nanmean(band_height(run, fH, 'H', ts)[0]
                               / lane_deriv(fF, ts + d)[0]) - 2) for d in _shifts]
        _best = float(_shifts[int(np.nanargmin(_obj))])
        _resid = float(np.nanmin(_obj))
        _n_ok = max(int(np.isfinite(ratio).sum()), 1)
        _sem = float(np.nanstd(ratio) / np.sqrt(_n_ok))
        print(f'1. frequency ratio   f_H / f_F = {np.nanmean(ratio):.3f} +/- {_sem:.3f} (s.e.m.)'
              '        (2.000 expected)')
        print(f'   the ratio RAMPS from {ratio[0]:.3f} to {ratio[-1]:.3f} across the overlap instead '
              'of scattering about a constant,')
        print('   so its spread is not an uncertainty. That ramp is exactly test 2: '
              'd ln(f_H/f_F)/dt = rel drift H - rel drift F.')
        print('   Tests 1 and 2 are one statement seen twice - do not count them as two independent '
              'checks.')
        # Both f_F and f_H here are the two cubics, so the ramp inherits their edge behaviour. Quote
        # what the ramp would be from the drifts measured on the traced points, for comparison.
        if np.isfinite(rdF) and np.isfinite(rdH):
            _ramp = float(np.nanmean(ratio)) * np.exp((rdH - rdF) * np.array([-.5, .5]) * (hi - lo))
            print(f'   measured on the traced points instead of the fitted cubics, the same ramp is '
                  f'{_ramp[0]:.3f} to {_ramp[1]:.3f}')
        if abs(np.nanmean(ratio) - 2) > 2 * _sem:
            # convert the shift into a distance using the speed measured on this very track
            _vF = float(np.nanmean(np.gradient(r_F, ts))) * R_SUN_M / 1e3          # km/s
            _dr = abs(_best) * _vF * 1e3 / R_SUN_M
            print(f'   the ratio reaches 2.000 (to {_resid:.4f}) if the fundamental is evaluated '
                  f'{_best:+.0f} s later than the harmonic.')
            print(f'   At the {_vF:.0f} km/s measured on this track that is {_dr:.3f} Rsun of travel: '
                  'small enough for two source')
            print('   points on one shock front, too large to be rounding. A wrong harmonic number '
                  'would NOT be repaired')
            print('   by any shift, so a small best-fit offset supports the F/H identification '
                  'rather than undermining it.')
        _dsig = np.hypot(rdF_e, rdH_e)
        print(f'2. relative drift    over the overlap, measured on the TRACED POINTS in that window '
              f'({rdF_n} F, {rdH_n} H samples per pass)')
        print(f'                     F: {rdF:+.6f} +/- {rdF_e:.6f} 1/s,   '
              f'H: {rdH:+.6f} +/- {rdH_e:.6f} 1/s')
        print(f'                     differ by {100 * abs(rdH - rdF) / abs(rdF):.1f}% = '
              f'{abs(rdH - rdF) / _dsig:.1f} sigma; f^-1 df/dt is free of the harmonic number, so '
              f'these must agree')
        print(f'                     the fitted cubics give {rdF_fit:+.6f} and {rdH_fit:+.6f} '
              f'({100 * abs(rdH_fit - rdF_fit) / abs(rdF_fit):.1f}% apart) - NOT the number to quote.')
        print(f'                     The overlap is the last '
              f'{100 * (fF["tmax"] - lo) / (fF["tmax"] - fF["tmin"]):.0f}% of the fundamental\'s fit '
              f'and the first {100 * (hi - fH["tmin"]) / (fH["tmax"] - fH["tmin"]):.0f}% of the '
              f'harmonic\'s,')
        print('                     so those two derivatives are the least constrained values either '
              'cubic produces.')
        print(f'3. height residual   r_H - r_F = {np.nanmean(dres):+.4f} +/- {np.nanstd(dres):.4f} Rsun'
              f'   ({100 * np.nanmean(np.abs(dres)) / np.nanmean(r_F):.2f}% of the height)')
        print(f'4. band splits       X_F = {scalars["F"]["X"][0]:.3f} +/- {scalars["F"]["X"][1]:.3f},   '
              f'X_H = {scalars["H"]["X"][0]:.3f} +/- {scalars["H"]["X"][1]:.3f}   '
              '(same shock -> same density jump)')

        # Each row is self-contained. `kind` says whether it is a raw MEASUREMENT or a
        # CONSISTENCY TEST; only tests have an `expected` value and an n_sigma, and a blank expected
        # means the row is a measurement with nothing to compare against, not a missing number.
        def _row(kind, test, value, error, expected=np.nan, note=''):
            n = (abs(value - expected) / error) if (np.isfinite(expected) and np.isfinite(error)
                                                    and error > 0) else np.nan
            return {'kind': kind, 'test': test, 'value': value, 'error': error,
                    'expected': expected, 'n_sigma': n,
                    'verdict': ('' if not np.isfinite(n) else
                                ('FLAG' if n > CONSIST_SIGMA else 'ok')), 'note': note}

        _eX = np.hypot(scalars['F']['X'][1], scalars['H']['X'][1])
        fh_rows = [
            # The uncertainty is the STANDARD ERROR of the mean, not the scatter along the overlap.
            # The ratio does not scatter about a constant, it RAMPS, because the two bands have
            # different relative drifts - and feeding that ramp in as an error made this test pass
            # automatically whenever the drift test failed. The ramp is reported as a range instead.
            _row('consistency test', 'f_H / f_F over the overlap', float(np.nanmean(ratio)),
                 float(np.nanstd(ratio) / np.sqrt(max(int(np.isfinite(ratio).sum()), 1))), 2,
                 'must be 2 for a fundamental and its harmonic. NOT independent of the relative-drift '
                 'row: the ratio ramps at exactly (rel drift H - rel drift F), so read them together'),
            _row('measurement', 'f_H / f_F at the start of the overlap', float(ratio[0]), np.nan,
                 np.nan, 'the ratio ramps rather than scattering; start and end bracket it'),
            _row('measurement', 'f_H / f_F at the end of the overlap', float(ratio[-1]), np.nan,
                 np.nan, 'the ratio ramps rather than scattering; start and end bracket it'),
            # Measured on the traced samples inside the overlap, not on the two cubics. Comparing the
            # cubics puts the fundamental's last 32% against the harmonic's first 13%, where a cubic's
            # derivative is at its least constrained, and that alone produced a 15% gap at 5.6 sigma.
            # The error is now a real one from the fits to those points, in place of a nominal 2%.
            _row('consistency test', 'relative drift, H minus F [1/s]', rdH - rdF,
                 float(np.hypot(rdF_e, rdH_e)), 0,
                 'f^-1 df/dt is a property of the shock, so both bands must give the same value. '
                 'Measured on the traced points inside the overlap; the fitted cubics give '
                 f'{rdH_fit - rdF_fit:+.6f} there, which is edge behaviour of the fits, not the burst'),
            _row('measurement', 'relative drift, H minus F, from the fitted cubics [1/s]',
                 rdH_fit - rdF_fit, np.nan, np.nan,
                 'diagnostic only - shows how far two cubics disagree when each is evaluated near the '
                 'end of its own span; the row above is the measurement'),
            _row('consistency test', 'height residual r_H - r_F [Rsun]', float(np.nanmean(dres)),
                 float(np.nanstd(dres)), 0,
                 'converting each band with its own harmonic number must put them at the same height'),
            _row('consistency test', 'density jump, X_H minus X_F',
                 scalars['H']['X'][0] - scalars['F']['X'][0], _eX, 0,
                 'X is model-independent and both bands measure the same shock'),
            _row('measurement', 'relative drift F, overlap [1/s]', rdF, rdF_e, np.nan,
                 f'log-linear fit to the {rdF_n} traced samples per pass inside the overlap'),
            _row('measurement', 'relative drift H, overlap [1/s]', rdH, rdH_e, np.nan,
                 f'log-linear fit to the {rdH_n} traced samples per pass inside the overlap'),
            _row('measurement', 'density jump X (F)', *scalars['F']['X'], np.nan,
                 'from the F band split alone'),
            _row('measurement', 'density jump X (H)', *scalars['H']['X'], np.nan,
                 'from the H band split alone'),
            _row('measurement', 'signed mean V/I (F)', pF, np.nan, np.nan,
                 'scatter along the lane is in typeii_polarisation.csv'),
            _row('measurement', 'signed mean V/I (H)', pH, np.nan, np.nan,
                 'scatter along the lane is in typeii_polarisation.csv'),
        ]
    run.OVERLAP = OVERLAP
    run.rdF = rdF
    run.rdF_e = rdF_e
    run.rdF_fit = rdF_fit
    run.rdF_n = rdF_n
    run.rdH = rdH
    run.rdH_e = rdH_e
    run.rdH_fit = rdH_fit
    run.rdH_n = rdH_n

    fh_test = pd.DataFrame(fh_rows)
    run.set(fh_test=fh_test)
    if len(fh_test):
        fh_test.to_csv(os.path.join(OUTDIR, 'fundamental_harmonic_tests.csv'), index=False)
        print('\nkind = "consistency test" rows have an expected value and an n_sigma; '
              '"measurement" rows do not.')
        print('a blank expected/n_sigma means there is nothing to compare that number against, '
              'not a missing result.')
    fh_test.round(4)


def audit_consistency(run):
    """Quantities that must agree if the physics is right, and the acceleration report."""
    # from the run
    BANDS_TRACED = run.BANDS_TRACED
    TRACED = run.TRACED
    OVERLAP = run.OVERLAP
    rdF = run.rdF
    rdF_e = run.rdF_e
    rdH = run.rdH
    rdH_e = run.rdH_e
    SPLIT_PAIR = run.SPLIT_PAIR
    ALref = run.ALref
    OUTDIR = run.OUTDIR
    KIN_BASELINE = run.KIN_BASELINE
    A_BIAS = run.A_BIAS
    scalars = run.scalars
    tg = run.tg
    A_MEASURED = run.A_MEASURED

    # ---- consistency audit: quantities that MUST agree if the physics is right ----------------
    # Everything below compares two measurements of the same thing. A disagreement beyond
    # CONSIST_SIGMA is not a rounding issue, it means a lane is mis-traced or a branch is misassigned,
    # and it has to be resolved before any of these numbers are used.


    audit = []
    WIDENING_EXPLAINS = {}       # band -> (separation rate km/s, n_sigma) when it accounts for a flag


    def _check(name, a, ea, b, eb, what, tol=CONSIST_SIGMA):
        d, ed, n = compare(a, ea, b, eb)
        audit.append({'check': name, 'value_1': a, 'value_2': b, 'difference': d,
                      'combined_error': ed, 'n_sigma': n, 'verdict': 'FLAG' if n > tol else 'ok',
                      'why_it_matters': what})


    # 1. the two bands are the same shock, so the density jump and the relative drift must match
    if len(BANDS_TRACED) > 1 and all(np.isfinite(scalars[b]['X'][0]) for b in BANDS_TRACED):
        _check('density jump X: F vs H', *scalars['F']['X'], *scalars['H']['X'],
               'X is model-independent and both bands measure the same shock')
    if OVERLAP is not None:
        # rdF and rdH are now measured on the traced points inside the overlap and carry real errors,
        # so the nominal 2% that used to stand in for an uncertainty here is gone.
        _check('relative drift over the overlap: F vs H', rdF, rdF_e, rdH, rdH_e,
               'f^-1 df/dt is a property of the shock, free of the harmonic number')

    # 2. within a band the two split branches straddle one shock front, so their kinematics and B
    #    must be close; a large disagreement means one branch is not where you think it is.
    #
    #    Both branches are averaged over the IDENTICAL window here. On their own spans they cover
    #    different stretches of a curving track, and a speed difference would then be mostly a
    #    coverage difference.
    #
    #    A branch-to-branch speed difference that survives that is not automatically an error: the two
    #    branches separating in height IS the band split widening, which is the same fact as X rising
    #    along the overlap. So the difference is also tested against d(r_lower - r_upper)/dt. When
    #    those agree, the speed and acceleration flags and the "X varies along the overlap" note are
    #    one measurement reported three times, and should be discussed as one.
    for b in BANDS_TRACED:
        lo_, up_ = SPLIT_PAIR[b]
        if up_ is None or lo_ not in ALref or up_ not in ALref:
            continue
        _dlo, _dup = ALref[lo_], ALref[up_]
        _cmb = common_mask(_dlo)
        _cmu = common_mask(_dup)
        if _cmb is None or _cmu is None:
            continue
        _cmb = _cmb & _cmu
        if _cmb.sum() < 3:
            continue
        _sep = _dlo['r_mean'][_cmb] - _dup['r_mean'][_cmb]
        _dsep = float(np.polyfit(tg[_cmb], _sep, 1)[0]) * R_SUN_M / 1e3          # km/s
        # B is deliberately absent: it is only defined for the upstream branch (see A.10), so a
        # branch-to-branch comparison of it is not a consistency test.
        for key, lab in [('v', 'shock speed'), ('a', 'acceleration')]:
            if key == 'a' and not (A_MEASURED.get(lo_, False) and A_MEASURED.get(up_, False)):
                continue          # at least one branch's curvature does not clear its own error bar,
                                  # so comparing the two would be comparing noise
            m1, e1 = grid_scalar(_dlo, key, mask=_cmb)
            m2, e2 = grid_scalar(_dup, key, mask=_cmb)
            if np.isfinite(m1) and np.isfinite(m2):
                _check(f'{lab}: {lo_} vs {up_}', m1, e1, m2, e2,
                       'the two split branches of one band bracket the same shock front '
                       '(both averaged over the same window)')
            if key == 'v' and np.isfinite(m1) and np.isfinite(m2):
                _check(f'band split widening explains the {b} speed difference',
                       m1 - m2, np.hypot(e1, e2), _dsep, np.nan,
                       f'the {b} branches separate at {_dsep:+.0f} km/s, i.e. the split is widening '
                       f'(X runs {scalars[b]["X_range"][0]:.3f} to {scalars[b]["X_range"][1]:.3f} '
                       f'along the overlap). If this row passes, the speed and acceleration rows '
                       f'above are that same widening and not independent inconsistencies')
                if audit[-1]['verdict'] == 'ok':
                    WIDENING_EXPLAINS[b] = (_dsep, float(audit[-1]['n_sigma']))
    run.set(WIDENING_EXPLAINS=WIDENING_EXPLAINS)

    audit = pd.DataFrame(audit)
    run.set(audit=audit)
    if len(audit):
        audit.to_csv(os.path.join(OUTDIR, 'consistency_audit.csv'), index=False)
        for _, r in audit[audit.verdict == 'FLAG'].iterrows():
            # a flag on a split pair whose separation rate accounts for the difference is not a fourth
            # independent problem; it is the widening, already reported, seen through another quantity
            _b = next((k for k in WIDENING_EXPLAINS
                       if k in BANDS_TRACED and SPLIT_PAIR.get(k, (None,))[0] in str(r['check'])),
                      None)
            print(f'  *** FLAG: {r["check"]} differ by {r["n_sigma"]:.1f} sigma '
                  f'({r["value_1"]:.3g} vs {r["value_2"]:.3g}). {r["why_it_matters"]}.')
            if _b is not None:
                _d, _n = WIDENING_EXPLAINS[_b]
                print(f'            Accounted for: the {_b} branches are separating at {_d:+.0f} km/s '
                      f'(the split widening), which matches this difference to {_n:.1f} sigma.')
                print(f'            Report it as the band split widening through the event, not as '
                      f'{_b} lane 1 and {_b} lane 2 disagreeing about one shock.')
        if not (audit.verdict == 'FLAG').any():
            print('  all consistency checks pass')

    # 3. is the acceleration meaningful? Two separate questions, asked separately: does it clear its
    #    own error bar, and is its size physically plausible. Every lane is fitted at KIN_DEG, so
    #    every lane gets an answer rather than being excluded in advance by its traced duration.
    print()
    print(f'acceleration: significance (|a| > {KIN_A_SIGMA} sigma), method bias floor, plausibility')
    print('the bias floor is what this chain returns for a shock at EXACTLY constant speed over the')
    print('same span - a property of the method that no error bar can reveal')
    for lab in TRACED:
        d = ALref.get(lab)
        if d is None:
            continue
        a_, ea_ = grid_scalar(d, 'a')
        if not np.isfinite(a_):
            print(f'  {lab:10s} no acceleration: the density model does not resolve this lane')
            continue
        base = KIN_BASELINE.get(lab, np.nan)
        sig = abs(a_) / ea_ if (np.isfinite(ea_) and ea_ > 0) else np.nan
        bias = A_BIAS.get(lab, np.nan)
        tag = []
        if not accel_is_measured(a_, ea_, bias=bias):
            if np.isfinite(ea_) and ea_ > 0 and abs(a_) < KIN_A_SIGMA * ea_:
                tag.append(f'NOT SIGNIFICANT at {sig:.1f} sigma')
            else:
                tag.append(f'BELOW THE METHOD BIAS FLOOR of {abs(bias):.0f} m/s^2')
        if abs(a_) > A_PLAUSIBLE_MS2:
            tag.append(f'|a| = {abs(a_):.0f} m/s^2 is {abs(a_) / 50:.0f}x a typical coronal value')
        print(f'  {lab:10s} a = {a_:+8.1f} +/- {ea_:5.1f} m/s^2 over {base:5.0f} s '
              f'({sig:4.1f} sigma, bias floor {bias:+6.1f})'
              + ('   <-- ' + '; '.join(tag) if tag else '   (ok)'))
    print('a is the second derivative of a height track whose curvature is set largely by the assumed')
    print('density profile, so it carries a systematic far larger than the error bars above: see the')
    print('model x fold range in A.8 before quoting it.')


def plot_fh_checks(run):
    """Polarisation, V/I along the lanes, and the two height tracks."""
    # from the run
    TRACED = run.TRACED
    BANDS_TRACED = run.BANDS_TRACED
    OVERLAP = run.OVERLAP
    POL_TRACK = run.POL_TRACK
    ALref = run.ALref
    tg = run.tg
    POL = run.POL
    EVENT_DATE = run.EVENT_DATE
    pF = run.pF
    pH = run.pH
    _fits = run._fits
    LANE_COL = run.LANE_COL
    SPLIT_PAIR = run.SPLIT_PAIR
    t0 = run.t0

    fig = plt.figure(figsize=[15, 11])

    # (a) the V/I spectrogram with every traced lane on top
    ax = fig.add_subplot(311)
    step = max(1, POL.shape[0] // PREVIEW_MAX_TCOLS)
    pv = np.nanpercentile(np.abs(POL.to_numpy()), 99)
    pm = ax.pcolormesh(POL.index[::step], np.asarray(POL.columns, float), POL.to_numpy().T[:, ::step],
                       vmin=-pv, vmax=pv, cmap='seismic', rasterized=True)
    fig.colorbar(pm, ax=ax, pad=0.01, label='Stokes V/I')
    for lab in TRACED:
        ax.plot(POL_TRACK[lab]['t'], POL_TRACK[lab]['f'], color='k', lw=1.8)
        ax.plot(POL_TRACK[lab]['t'], POL_TRACK[lab]['f'], color=LANE_COL[lab], lw=1.1, label=lab)
    if INVERT_FREQ:
        ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel(f'Time (UT) on {EVENT_DATE}')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('(a) Circular polarisation with the traced lanes overlaid')
    ax.legend(fontsize=8, ncol=4, loc='lower left')

    # (b) V/I along each lane, F against H
    ax = fig.add_subplot(312)
    for lab in TRACED:
        d = POL_TRACK[lab]
        ax.plot(d['t'], d['p'], 'o', ms=3, color=LANE_COL[lab], alpha=0.45)
        ax.plot(d['t'], sg_smooth(d['p'], window=11), '-', lw=2, color=LANE_COL[lab],
                label=f'{lab}: {np.nanmean(d["p"]):+.3f} $\\pm$ {np.nanstd(d["p"]):.3f}')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel(f'Time (UT) on {EVENT_DATE}')
    ax.set_ylabel('Stokes V/I along the lane')
    ax.set_title('(b) Degree of circular polarisation: signed mean $V/I$ = '
                 f'{pF:+.3f} (F) vs {pH:+.3f} (H)')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    # (c) the two height tracks, which must be one trajectory
    ax = fig.add_subplot(313)
    for b, col in zip(BANDS_TRACED, ['navy', 'firebrick']):
        fit = _fits.get(SPLIT_PAIR[b][0])
        if fit is None:
            continue
        ts = np.linspace(fit['tmin'], fit['tmax'], 80)
        _, r = band_height(run, fit, b, ts)
        ax.plot(ts, r, '-', lw=2.4, color=col, label=f'{BAND_NAME[b]} ($s$ = {HARM[b]})')
    d = ALref.get(JOINT)
    if d is not None:
        ax.errorbar(tg, d['r_mean'], yerr=d['r_se'], fmt='o', ms=3.5, color='0.35', capsize=2,
                    alpha=0.75, label=JOINT, zorder=0)
    if OVERLAP is not None:
        ax.axvspan(*OVERLAP, color='0.85', alpha=0.6, zorder=-1)
        ax.text(np.mean(OVERLAP), ax.get_ylim()[0], 'overlap', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel(f'time since {t0.strftime("%H:%M")} UT [s]')
    ax.set_ylabel(r'$r\,/\,R_\odot$')
    ax.set_title(f'(c) Both bands on one height-time trajectory ({REF_MODEL_NAME})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fundamental_harmonic_checks')
    plt.show()


def plot_kinematics(run):
    """Height, speed, acceleration and magnetic field per lane."""
    # from the run
    TRACKS = run.TRACKS
    BANDS_TRACED = run.BANDS_TRACED
    REPEATS_INDEPENDENT = run.REPEATS_INDEPENDENT
    EVENT_DATE = run.EVENT_DATE
    LANE_ROLE = run.LANE_ROLE
    scalars = run.scalars
    TRACED = run.TRACED
    LANE_SIGMA = run.LANE_SIGMA

    fig = plt.figure(figsize=[14, 9])

    ax = fig.add_subplot(221)
    track_panel(run, ax, TRACKS, 'r', r'$r\,/\,R_\odot$', r'$R_\odot$', '{:.3f}')
    ax.set_title(f'(a) Height-time  ({REF_MODEL_NAME})')

    ax = fig.add_subplot(222)
    track_panel(run, ax, TRACKS, 'v', r'$v_{\rm sh}$ [km s$^{-1}$]', 'km/s', '{:.0f}')
    ax.set_title('(b) Shock speed')

    ax = fig.add_subplot(223)
    track_panel(run, ax, TRACKS, 'a', r'$a$ [m s$^{-2}$]', r'm s$^{-2}$', '{:.1f}')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_title('(c) Acceleration')

    ax = fig.add_subplot(224)
    # UPSTREAM branches only, matching the table in A.10. v_A = v_sh/M_A is the upstream Alfven speed,
    # so pairing it with a downstream branch's density gives B_1 sqrt(X) - neither the upstream field
    # nor the downstream one. Plotting all four lanes here invited exactly the branch-to-branch
    # comparison that is not meaningful.
    _B_TRACKS = [k for k in TRACKS if LANE_ROLE.get(k, '').startswith('upstream') or k == JOINT]
    track_panel(run, ax, _B_TRACKS, 'B', r'$B$ [G]', 'G', '{:.3f}')
    ax.set_title('(d) Coronal magnetic field  (upstream branches only)')
    if not any(has_track(run, k, 'B') for k in TRACKS):
        # B needs M_A, which needs 1 <= X < 4. Say so in the panel instead of leaving it blank.
        bad = [f'{BAND_NAME[b]}: X = {scalars[b]["X"][0]:.3f}' for b in BANDS_TRACED
               if not np.isfinite(scalars[b]['M_A'][0])]
        ax.text(0.5, 0.5, 'no B: it needs $M_A$, which is only defined for $1 \\leq X < 4$\n'
                + ('\n'.join(bad) if bad else 'no band split traced'),
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='firebrick')

    # Spell out what the bars are, rather than calling them "statistical". They propagate exactly one
    # thing - the assumed frequency uncertainty of the traced lane - through the fit. They do not
    # contain the density-model choice (a factor ~2, A.8), and with jittered repeats they do not
    # contain a tracing-reproducibility term either.
    _errsrc = ('bars propagate the traced-lane frequency error (LANE_SIGMA ~ '
               f'{np.nanmean([LANE_SIGMA.get(l, np.nan) for l in TRACED]):.2f} MHz) through the fit'
               + ('' if REPEATS_INDEPENDENT else ', with NO independent re-tracing term')
               + '; the density-model systematic (~2x, A.8) is NOT included')
    fig.suptitle(f'Type II kinematics and coronal magnetic field, NenuFAR {EVENT_DATE}\n' + _errsrc,
                 y=1.03, fontsize=11)
    fig.tight_layout()
    save_fig(fig, 'typeii_kinematics_Bfield')
    plt.show()


def height_time_fits(run):
    """Fit the reference track with each height-time method."""
    # from the run
    (ALref, tg, TRACED, BANDS_TRACED, SPLIT_PAIR) = (
        run.ALref, run.tg, run.TRACED, run.BANDS_TRACED, run.SPLIT_PAIR
    )

    RS_KM = R_SUN_M / 1e3
    run.set(RS_KM=RS_KM)

    # which track the height-time fits and the model sweep use
    if REF_LANE is not None:
        REF_TRACK = REF_LANE
    elif MAKE_JOINT and JOINT in ALref and np.isfinite(ALref[JOINT]['r_mean']).sum() > 3:
        REF_TRACK = JOINT
    else:
        _cand = [SPLIT_PAIR[b][0] for b in BANDS_TRACED] + list(TRACED)
        REF_TRACK = next((k for k in _cand
                          if k in ALref and np.isfinite(ALref[k]['r_mean']).sum() > 3), None)
    run.set(REF_TRACK=REF_TRACK)
    if REF_TRACK is None:
        raise RuntimeError('no usable height-time track on ' + REF_MODEL_NAME)
    print(f'height-time fits and the model sweep use: {REF_TRACK}  ({REF_MODEL_NAME})')


    # Each fitter returns h, v and a as callables with ANALYTIC derivatives. Nothing is finite
    # differenced: differencing an interpolant returns the slope of whichever segment the step lands
    # in, which produces staircase speeds and spikes of order 1e6 m/s^2 in the acceleration.










    FIT_METHODS = {'Polynomial': fit_polynomial, 'Gallagher (2003)': fit_gallagher,
                   'Byrne (2013)': fit_byrne}
    run.set(FIT_METHODS=FIT_METHODS)
    FIT_COLOR = {'Polynomial': 'tab:blue', 'Gallagher (2003)': 'tab:green', 'Byrne (2013)': 'tab:red'}
    run.set(FIT_COLOR=FIT_COLOR)

    _d = ALref[REF_TRACK]
    _good = np.isfinite(_d['r_mean'])
    t_fit = tg[_good].astype(float)
    run.set(t_fit=t_fit)
    r_fit = _d['r_mean'][_good].astype(float)
    run.set(r_fit=r_fit)
    se_r = _d['r_se'][_good].astype(float)
    _pos = se_r[np.isfinite(se_r) & (se_r > 0)]
    se_r = np.where(np.isfinite(se_r) & (se_r > 0), se_r, np.nanmedian(_pos) if _pos.size else 1e-3)
    run.set(se_r=se_r)
    h_fit = r_fit * RS_KM
    sig_h = se_r * RS_KM
    t_dense = np.linspace(t_fit.min(), t_fit.max(), 400)
    run.set(t_dense=t_dense)

    FIT_OUT = {}
    rng = np.random.default_rng(1)
    for name, fitter in FIT_METHODS.items():
        try:
            f0 = fitter(t_fit, h_fit, sig_h)
        except (RuntimeError, ValueError, TypeError) as ex:
            print(f'{name}: fit failed ({ex})')
            continue
        h0, v0, a0 = hva(f0, t_dense)
        boot = {'h': [], 'v': [], 'a': []}
        for _ in tqdm(range(N_BOOT), desc=name, leave=False):
            try:
                fb = fitter(t_fit, h_fit + rng.normal(0, sig_h), sig_h)
                hh, vv, aa = hva(fb, t_dense)
                boot['h'].append(hh)
                boot['v'].append(vv)
                boot['a'].append(aa)
            except (RuntimeError, ValueError):
                continue
        resid = f0['h'](t_fit) - h_fit
        # Free parameters differ by method: the polynomial has POLY_DEG+1, Gallagher has 6
        # (a_r, a_d, tau_r, tau_d, h0, v0), and Savitzky-Golay is a local smoother whose effective
        # count is about (points / window) x (POLY_DEG + 1). Using POLY_DEG+1 for all three, as this
        # did, understates Gallagher's dof and overstates Byrne's.
        _npar = {'Polynomial': POLY_DEG + 1, 'Gallagher (2003)': 6,
                 'Byrne (2013)': max(int(np.ceil(len(t_fit) / 11)) * (POLY_DEG + 1), POLY_DEG + 1)}
        dof = max(len(t_fit) - _npar.get(name, POLY_DEG + 1), 1)
        chi2_red = float(np.nansum((resid / sig_h) ** 2) / dof)
        # chi2_red >> 1 means the fit misses the points by more than the quoted errors allow, so the
        # bootstrap band built from those errors is too narrow by roughly sqrt(chi2_red)
        scale = np.sqrt(max(chi2_red, 1)) if INFLATE_BY_CHI2 else 1
        FIT_OUT[name] = dict(
            h=h0 / RS_KM, v=v0, a=a0,
            h_sd=(np.nanstd(boot['h'], axis=0) / RS_KM if boot['h'] else np.zeros_like(t_dense)) * scale,
            v_sd=(np.nanstd(boot['v'], axis=0) if boot['v'] else np.zeros_like(t_dense)) * scale,
            a_sd=(np.nanstd(boot['a'], axis=0) if boot['a'] else np.zeros_like(t_dense)) * scale,
            chi2_red=chi2_red, scale=scale,
            rse_Rsun=np.sqrt(np.nansum(resid ** 2) / dof) / RS_KM)
        # Even with the right dof this is not a goodness-of-fit statistic here: the points being
        # fitted are an analytic lane polynomial pushed through the density inversion, not independent
        # measurements, so the residuals are small by construction and chi2_red comes out far below 1.
        # rse_Rsun is the number to compare between methods.
        print(f'{name:18s} chi2_red = {chi2_red:8.3f} (not a goodness of fit - see note), '
              f'rms residual = {np.sqrt(np.nansum(resid ** 2) / dof) / RS_KM:.4f} Rsun')
    run.set(FIT_OUT=FIT_OUT)
    print(f'{len(FIT_OUT)} of {len(FIT_METHODS)} methods converged')

    # The spread BETWEEN methods is the number worth quoting from this section, so it is computed
    # rather than read off the figure. Also state how these compare with A.6, because the same lane
    # gets an acceleration in both places and they are not identical.
    _vm = [o['v'].mean() for o in FIT_OUT.values() if np.isfinite(o['v']).any()]
    _am = [o['a'].mean() for o in FIT_OUT.values() if np.isfinite(o['a']).any()]
    if len(_vm) > 1:
        print(f'\nspread between methods: mean speed {100 * (max(_vm) / min(_vm) - 1):.1f}% '
              f'({min(_vm):.0f}-{max(_vm):.0f} km/s), '
              f'mean acceleration {100 * (max(_am) / min(_am) - 1):.1f}% '
              f'({min(_am):.0f}-{max(_am):.0f} m/s^2)')
    _a6, _a6e = grid_scalar(ALref[REF_TRACK], 'a')
    _v6, _v6e = grid_scalar(ALref[REF_TRACK], 'v')
    print(f'against A.6 for the same lane: v_sh {_v6:.0f} +/- {_v6e:.0f} km/s, '
          f'a {_a6:+.0f} +/- {_a6e:.0f} m/s^2')
    print('  The speeds agree; the accelerations need not, and the difference is not an error in')
    print('  either. A.6 fits the lane over KIN_N_DENSE samples of its OWN traced span and averages')
    print('  the result over the Monte-Carlo draws; this section fits the shared-grid points that')
    print('  fall inside the lane and bootstraps them. r(t) is not exactly a quadratic, so a fitted')
    print('  curvature depends on where the samples sit - which is precisely why the spread between')
    print('  the three methods above is quoted as the uncertainty on a, not any single method\'s')
    print('  error bar. Quote A.6 for per-lane numbers; quote this section for method sensitivity.')

    # --- can each method actually describe this track? ------------------------------------------
    # Converging is not the same as being applicable. A model whose functional form cannot represent
    # the data will still return parameters; it just fits badly, and quoting its kinematics would be
    # wrong. Both checks below are about the shape of the model, not the quality of the data.
    APPLICABLE = {name: True for name in FIT_OUT}
    if FIT_OUT:
        _best = min(d['chi2_red'] for d in FIT_OUT.values())
        for name, d in FIT_OUT.items():
            if d['chi2_red'] > max(10 * _best, 10):
                APPLICABLE[name] = False
                print(f'\n  *** WARNING: {name} fits this track far worse than the best method '
                      f'(chi2_red {d["chi2_red"]:.1f} vs {_best:.2f}, rms residual '
                      f'{d["rse_Rsun"]:.4f} Rsun). Do not quote its kinematics.')
        _g, _p = FIT_OUT.get('Gallagher (2003)'), FIT_OUT.get('Polynomial')
        if _g is not None and np.nanmin(_g['a']) >= 0 and _p is not None and np.nanmin(_p['a']) < 0:
            APPLICABLE['Gallagher (2003)'] = False
            print('\n  *** WARNING: the Gallagher et al. (2003) reciprocal-sum profile is '
                  'POSITIVE-DEFINITE by construction -')
            print('      a(t) = [ (a_r e^(t/tau_r))^-1 + (a_d e^(-t/tau_d))^-1 ]^-1 with a_r, a_d > 0 '
                  'is strictly > 0.')
            print('      It describes the impulsive ACCELERATION phase of an eruption and cannot '
                  'represent a decelerating')
            print('      shock. The polynomial wants deceleration on this track, so the Gallagher '
                  'curve is inapplicable here.')
    run.set(APPLICABLE=APPLICABLE)


def plot_fit_comparison(run):
    """The three height-time methods side by side."""
    # from the run
    (t_fit, r_fit, FIT_OUT, FIT_COLOR, t_dense, se_r, APPLICABLE, REF_TRACK, t0) = (
        run.t_fit, run.r_fit, run.FIT_OUT, run.FIT_COLOR, run.t_dense, run.se_r,
        run.APPLICABLE, run.REF_TRACK, run.t0
    )

    fig = plt.figure(figsize=[16, 4.8])
    panels = [('h', r'$r\,/\,R_\odot$', '(a) Height'),
              ('v', r'$v_{\rm sh}$ [km s$^{-1}$]', '(b) Speed'),
              ('a', r'$a$ [m s$^{-2}$]', '(c) Acceleration')]
    for i, (key, ylab, ttl) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        if key == 'h':
            # quote what the traced points actually are, not the word "SEM"
            ax.errorbar(t_fit, r_fit, yerr=se_r, fmt='o', ms=4, color='0.4', capsize=2, alpha=0.8,
                        zorder=1, label=(rf'traced: {len(t_fit)} pts, '
                                         rf'{r_fit.min():.3f}$-${r_fit.max():.3f} $R_\odot$, '
                                         rf'$\sigma$ = {np.mean(se_r):.4f} $R_\odot$'))
        for name, d in FIT_OUT.items():
            c = FIT_COLOR[name]
            ok = APPLICABLE.get(name, True)
            y, sd = d[key], d[key + '_sd']
            if key == 'h':
                lbl = (rf'{name}: rms {d["rse_Rsun"]:.4f} $R_\odot$, '
                       rf'$\chi^2_\nu$ = {d["chi2_red"]:.2f}')
            elif key == 'v':
                lbl = (f'{name}: {np.nanmin(y):.0f}$-${np.nanmax(y):.0f}, '
                       f'mean {np.nanmean(y):.0f} km s$^{{-1}}$')
            else:
                lbl = (f'{name}: {np.nanmin(y):+.0f} to {np.nanmax(y):+.0f}, '
                       f'mean {np.nanmean(y):+.0f} m s$^{{-2}}$')
            if not ok:
                lbl += '  [INAPPLICABLE]'
            ax.plot(t_dense, y, ls=('-' if ok else '--'), color=c, lw=1.8, label=lbl)
            ax.fill_between(t_dense, y - sd, y + sd, color=c, alpha=0.18)
        if key == 'a':
            ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xlabel(f'time since {t0.strftime("%H:%M")} UT [s]')
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    _scaled = 'bands widened by sqrt(chi2) ' if INFLATE_BY_CHI2 else 'bands are the raw bootstrap 1$\sigma$ '
    fig.suptitle(f'Height-time fit comparison, {REF_TRACK} track ({REF_MODEL_NAME})\n'
                 + _scaled + f'over {N_BOOT} refits; dashed = the model cannot describe this track',
                 y=1.06, fontsize=12)
    fig.tight_layout()
    save_fig(fig, 'typeii_kinematics_fit_comparison')
    plt.show()


def fit_comparison_table(run):
    """Height-time fit comparison as a table."""
    # from the run
    OUTDIR, APPLICABLE, FIT_OUT = run.OUTDIR, run.APPLICABLE, run.FIT_OUT

    # goodness of fit and the grid-averaged kinematics each method implies. chi2_red is the primary
    # metric (about 1 means the fit is consistent with the height errors); rse is the residual
    # standard error in Rsun, i.e. the typical deviation of the fit from the traced heights
    fit_compare = pd.DataFrame([{'method': name, 'applicable': APPLICABLE.get(name, True),
                                 'chi2_red': d['chi2_red'],
                                 'band_scale': d['scale'], 'rse_Rsun': d['rse_Rsun'],
                                 'v_mean_kms': np.nanmean(d['v']), 'v_min_kms': np.nanmin(d['v']),
                                 'v_max_kms': np.nanmax(d['v']), 'a_mean_ms2': np.nanmean(d['a'])}
                                for name, d in FIT_OUT.items()])
    run.set(fit_compare=fit_compare)
    fit_compare.to_csv(os.path.join(OUTDIR, 'height_time_fit_comparison.csv'), index=False)
    fit_compare.round(3)


def model_sweep(run):
    """Recompute the chain for every density model and fold."""
    # from the run
    (passes, tg, REF_TRACK, OUTDIR, MODEL_GRID) = (
        run.passes, run.tg, run.REF_TRACK, run.OUTDIR, run.MODEL_GRID
    )

    sweep_rows = []
    for name, model in tqdm(MODEL_GRID.items(), desc='model x fold'):
        agg = aggregate_lanes(run, passes, tg, model)
        if REF_TRACK not in agg:
            continue
        row = {'model': name}
        for key, lab in [('r', 'r_Rsun'), ('v', 'v_kms'), ('a', 'a_ms2'), ('vA', 'vA_kms'),
                         ('B', 'B_G'), ('ne', 'ne_cm3')]:
            # No common mask here: this table reports the RANGE of each quantity across the model
            # grid, not a set of numbers that have to satisfy an identity with each other. r and v
            # should therefore span the whole traced lane, while v_A and B are confined to the
            # band-split window by their own definition.
            row[lab], row[lab + '_se'] = grid_scalar(agg[REF_TRACK], key)
        # The height on the band-split window, carried alongside, because r_Rsun above is NOT the
        # height at which B_G was measured and the two must not be plotted as a pair. B exists only
        # where both branches of the band do, and on this event that sub-interval sits 0.18 Rsun
        # higher than the lane average. Putting B at the lane-average height moved it along a
        # (r-1)^-1.5 reference curve far enough to change the quoted offset from Dulk & McLean from
        # 2.1x to 1.5x - an error in the one comparison that figure exists to make.
        _cms = common_mask(agg[REF_TRACK])
        row['r_at_B_Rsun'], row['r_at_B_Rsun_se'] = grid_scalar(agg[REF_TRACK], 'r', mask=_cms)
        sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    sweep['base'] = [m.rsplit(' x', 1)[0] for m in sweep['model']]
    sweep['fold'] = [int(m.rsplit(' x', 1)[1]) for m in sweep['model']]
    run.set(sweep=sweep)
    sweep.to_csv(os.path.join(OUTDIR, 'model_grid_sweep.csv'), index=False)
    print(f'the full chain was recomputed on the {REF_TRACK} track for all {len(MODEL_GRID)} '
          'density profiles (5 models x 4 folds); only n_e(r) changed, the tracing did not')
    print('the spread down each column IS the systematic error; compare it with the statistical '
          'errors in A.6 before quoting anything')
    sweep[['model', 'r_Rsun', 'v_kms', 'a_ms2', 'vA_kms', 'B_G']].round(3)


def plot_model_sweep(run):
    """Shock characteristics against density model and fold."""
    # from the run
    BASE_MODELS, REF_TRACK, sweep = run.BASE_MODELS, run.REF_TRACK, run.sweep

    bases = list(BASE_MODELS.keys())
    run.set(bases=bases)
    xpos = np.arange(len(bases))
    fold_off = {f: (f - 2.5) * 0.16 for f in FOLDS}
    fold_col = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red'}
    run.set(fold_col=fold_col)

    # The window is part of the axis label. r and v are lane averages over everything traced; v_A and
    # B exist only where both branches of the band do, which here is a 214 s sub-interval of a 596 s
    # lane. A reader comparing panel (a) with panel (d) is otherwise comparing two different heights.
    panels = [('r_Rsun', 'height $r$ [$R_\\odot$]\n(full traced span)'),
              ('v_kms', 'shock speed $v_{\\rm sh}$ [km s$^{-1}$]\n(full traced span)'),
              ('vA_kms', 'Alfv$\\acute{\\rm e}$n speed $v_A$ [km s$^{-1}$]\n(band-split window)'),
              ('B_G', '$B$ [G]\n(band-split window)')]

    fig = plt.figure(figsize=[14, 9])
    for i, (col, ylab) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i)
        for f in FOLDS:
            sub = sweep[sweep['fold'] == f].set_index('base').reindex(bases)
            ax.errorbar(xpos + fold_off[f], sub[col], yerr=sub[col + '_se'], fmt='o',
                        color=fold_col[f], ms=6, capsize=3, mec='k', mew=0.4,
                        label=(f'fold {f}' if i == 1 else None))
        ax.set_xticks(xpos)
        ax.set_xticklabels(bases, fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, axis='y')
        if i == 1:
            ax.legend(title='fold', fontsize=8, ncol=2)

    fig.suptitle(f'Shock characteristics vs density model x fold ({REF_TRACK} track)',
                 y=1.01, fontsize=13)
    fig.tight_layout()
    save_fig(fig, 'characteristics_vs_model_fold')
    plt.show()


def plot_bfield(run):
    """Band-split magnetic field against the published radial profiles."""
    # from the run
    sweep, bases, fold_col = run.sweep, run.bases, run.fold_col

    rr = np.linspace(1.05, 3, 300)
    fig = plt.figure(figsize=[9, 6.5])
    ax = fig.add_subplot(111)
    ax.plot(rr, B_dulk_mclean(rr), 'k-',
            label=r'Dulk & McLean (1978), $0.5\,(r-1)^{-1.5}$')
    ax.plot(rr, B_gopalswamy_yashiro(rr), 'k--',
            label=r'Gopalswamy & Yashiro (2011), $0.409\,r^{-1.30}$ (standoff distance,'
                  '\n' r'    calibrated 6$-$23 $R_\odot$, extrapolated here)')
    ax.plot(rr, B_mann2023(rr), 'k:',
            label=r'Mann et al. (2023) Eq. 8, $6r^{-3}+1.18r^{-2}$')

    mk = {'Newkirk': 'o', 'Saito': 's', 'Leblanc': '^', 'Baumbach-Allen': 'D', 'Mann 2023': 'v'}
    # x is r_at_B_Rsun, the height on the band-split window, NOT the lane-average r_Rsun of the A.8
    # table. B only exists on that window, and the reference curves are steep functions of r, so the
    # pair has to share one interval or the comparison is against the wrong part of the curve.
    for _, row in sweep.iterrows():
        if np.isfinite(row['r_at_B_Rsun']) and np.isfinite(row['B_G']):
            ax.errorbar(row['r_at_B_Rsun'], row['B_G'], yerr=row['B_G_se'],
                        xerr=row['r_at_B_Rsun_se'],
                        fmt=mk[row['base']], color=fold_col[row['fold']], ms=8, capsize=2,
                        mec='k', mew=0.5, alpha=0.9)
    if not np.isfinite(sweep['B_G']).any():
        ax.text(0.5, 0.5, 'no band-split estimate to plot: $B$ is NaN for every model.\n'
                '$B = v_A\\sqrt{\\mu_0\\rho}$ needs $M_A$, which is only defined for '
                '$1 \\leq X < 4$.\nCheck the upstream/downstream ordering printed in A.4.',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='firebrick')
    handles = [plt.Line2D([], [], marker=mk[b], color='0.4', ls='', mec='k', label=b) for b in bases]
    handles += [plt.Line2D([], [], marker='o', color=fold_col[f], ls='', label=f'fold {f}') for f in FOLDS]
    # Both legends stack in the top-right corner. The lower-left placement sat on top of the
    # Gopalswamy & Yashiro curve, which is the one the reader most needs to see is extrapolated.
    leg1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.4, fontsize=9)
    ax.add_artist(leg1)
    _h1 = leg1.get_window_extent().transformed(ax.transAxes.inverted()).height
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1 - _h1 - 0.03),
              borderaxespad=0.4, fontsize=8, ncol=2, title='band-split estimate')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\,/\,R_\odot$  (on the band-split window, where $B$ is defined)')
    ax.set_ylabel(r'$B$ [G]')
    ax.set_title('Coronal magnetic field: band-split estimate vs empirical laws')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, 'Bfield_comparison')
    plt.show()


def characteristics_table(run):
    """Assemble the burst characteristics table."""
    # from the run
    BANDS_TRACED = run.BANDS_TRACED
    TRACKS = run.TRACKS
    passes = run.passes
    pol_table = run.pol_table
    fh_test = run.fh_test
    LANE_ORDER = run.LANE_ORDER
    LANE_ROLE = run.LANE_ROLE
    OUTDIR = run.OUTDIR
    scalars = run.scalars
    ALref = run.ALref
    tracer = run.tracer
    TYPEII_WINDOW = run.TYPEII_WINDOW
    REF_TRACK = run.REF_TRACK
    sweep = run.sweep
    EVENT_DATE = run.EVENT_DATE
    RUN_START = run.RUN_START
    t0 = run.t0
    tg = run.tg

    rows = []
    for b in BANDS_TRACED:
        s = scalars[b]
        tag = f'{BAND_NAME[b].lower()}, s={HARM[b]}'
        for key, lab, unit in [('drift_MHz_s', 'drift rate', 'MHz/s'),
                               ('rel_drift_s', 'relative drift (1/f)(df/dt)', '1/s'),
                               ('X', 'density jump X', '-'),
                               ('M_A', 'Alfven Mach number M_A', '-'),
                               ('rel_bandwidth', 'relative band split (f_U-f_L)/f_L', '-')]:
            rows.append({'track': b, 'quantity': f'{lab} ({tag})', 'value': s[key][0],
                         'error': s[key][1], 'unit': unit, 'note': 'model-independent'})

    for key in TRACKS:
        d = ALref[key]
        if np.isfinite(d['r_mean']).sum() <= 2:
            continue
        tag = key
        fm = d['f_mean'][np.isfinite(d['f_mean'])]
        if fm.size:
            rows.append({'track': tag, 'quantity': f'frequency range ({tag})', 'value': np.nanmin(fm),
                         'error': np.nanmax(fm), 'unit': 'MHz (min, max)', 'note': 'observed'})
        # Two different averaging windows are unavoidable here, so both are stated rather than one
        # being quietly imposed. r, v, a and n_e exist over the lane's WHOLE traced span and that is
        # the measurement worth quoting. X, M_A, v_A and B need both branches of the band split, so
        # they exist only where the branches overlap in time. Averaging the first group over the whole
        # span and the second over the overlap - and then printing them in one row without saying so -
        # is what makes a row fail its own identity: v_A comes out unequal to v_sh/M_A and B unequal to
        # v_A sqrt(mu0 rho) at the n_e beside it. Forcing everything onto the overlap instead would fix
        # the identity but throw away most of the traced lane, so the speed and height would no longer
        # describe the burst. Both windows are therefore reported, each labelled, plus the speed on the
        # split window so the identity can be checked directly.
        _cm = common_mask(d)
        _fmt_win = lambda msk: (
            f'{(t0 + pd.Timedelta(seconds=float(tg[msk].min()))).strftime("%H:%M:%S")}-'
            f'{(t0 + pd.Timedelta(seconds=float(tg[msk].max()))).strftime("%H:%M:%S")} UT')
        _win_split = _fmt_win(_cm) if (_cm is not None and _cm.any()) else ''
        for k, lab, unit in [('r', 'height r', 'Rsun'), ('v', 'shock speed', 'km/s'),
                             ('a', 'acceleration', 'm/s^2'),
                             ('ne', 'upstream density n_e', 'cm^-3')]:
            m, se = grid_scalar(d, k)
            _msk = np.isfinite(d[k + '_mean'])
            rows.append({'track': tag, 'quantity': f'{lab} ({tag})', 'value': m, 'error': se,
                         'unit': unit, 'note': f'{REF_MODEL_NAME}, full traced span '
                                               f'{_fmt_win(_msk) if _msk.any() else ""}'})
        # Height and density on the split window as well as the full span. Only these are comparable
        # BETWEEN the two branches of a band: on their own spans the branches cover different stretches
        # of a drifting burst, so the full-span n_e of the upper branch comes out BELOW the lower one
        # and the heights come out the wrong way round, contradicting X > 1. On the common window the
        # ordering is the physical one.
        for k, lab, unit in [('r', 'height r on the band-split window', 'Rsun'),
                             ('v', 'shock speed on the band-split window', 'km/s'),
                             ('ne', 'n_e on the band-split window', 'cm^-3')]:
            m, se = grid_scalar(d, k, mask=_cm)
            rows.append({'track': tag, 'quantity': f'{lab} ({tag})', 'value': m, 'error': se,
                         'unit': unit,
                         'note': f'{REF_MODEL_NAME}, band-split window {_win_split}; only these are '
                                 'comparable between the two branches of a band'})
        # v_A and B belong to the UPSTREAM branch only. v_A = v_sh/M_A from Rankine-Hugoniot is by
        # construction the upstream Alfven speed, so pairing it with the downstream density gives
        # B_1 sqrt(X) - neither the upstream field B_1 nor the downstream field X B_1. Emitting it for
        # the downstream lane produced a number that is the field of nothing, and an audit row that
        # compared the two branches' "B" and passed only because two errors partly cancelled.
        if LANE_ROLE.get(tag, '').startswith('upstream'):
            for k, lab, unit in [('vA', 'Alfven speed', 'km/s'), ('B', 'magnetic field B', 'G')]:
                m, se = grid_scalar(d, k, mask=_cm)
                rows.append({'track': tag, 'quantity': f'{lab} ({tag})', 'value': m, 'error': se,
                             'unit': unit,
                             'note': f'{REF_MODEL_NAME}, band-split window {_win_split}, UPSTREAM '
                                     'branch; satisfies v_A = v_sh/M_A and B = v_A sqrt(mu0 rho) '
                                     'with the two rows above'})
        else:
            rows.append({'track': tag, 'quantity': f'magnetic field B ({tag})', 'value': np.nan,
                         'error': np.nan, 'unit': 'G',
                         'note': 'not defined for a downstream branch: v_A from Rankine-Hugoniot is '
                                 'the UPSTREAM Alfven speed, so pairing it with this branch\'s '
                                 'density gives B_1 sqrt(X), which is neither B_1 nor B_2'})
        if REPORT_EXCITER_ENERGY:
            v_mean, _ = grid_scalar(d, 'v')
            rows.append({'track': tag, 'quantity': f'bulk exciter energy ({tag})',
                         'value': electron_energy_from_speed(v_mean) * 1e3, 'error': np.nan,
                         'unit': 'eV', 'note': f'{REF_MODEL_NAME}, E = (gamma-1) m_e c^2 at v_sh; '
                                               'NOT the energy of the emitting electrons'})

    # the model x fold spread, i.e. the systematic error on every height-dependent quantity
    for col, lab, unit in [('r_Rsun', 'height r', 'Rsun'), ('v_kms', 'shock speed', 'km/s'),
                           ('a_ms2', 'acceleration', 'm/s^2'),
                           ('vA_kms', 'Alfven speed', 'km/s'), ('B_G', 'magnetic field B', 'G')]:
        vals = sweep[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            rows.append({'track': REF_TRACK,
                         'quantity': f'{lab} (model x fold range)', 'value': np.nanmin(vals),
                         'error': np.nanmax(vals), 'unit': f'{unit} (min, max)',
                         'note': 'systematic across the 20-model grid'})

    char_table = pd.DataFrame(rows)
    char_table.to_csv(os.path.join(OUTDIR, 'typeii_characteristics.csv'), index=False)




    with open(os.path.join(OUTDIR, 'typeii_characteristics.tex'), 'w') as fh:
        fh.write(df_to_latex(char_table,
                             f'Characteristics of the type II radio burst observed by NenuFAR on '
                             f'{EVENT_DATE}, from the band-split fundamental and harmonic lanes. '
                             f'Model-independent quantities come from the band splits alone; the rest '
                             f'assume the {REF_MODEL_NAME} density model, with the range across the '
                             f'full model grid given separately.',
                             'tab:typeii_nenufar'))

    picks = {'traces': tracer.traces, 'passes': passes, 'polarisation': pol_table,
             'fh_tests': fh_test, 'lane_order': LANE_ORDER, 'lane_role': LANE_ROLE,
             'config': {'N_REPS': N_REPS, 'TYPEII_WINDOW': TYPEII_WINDOW,
                        'TYPEII_FLIM': TYPEII_FLIM, 'HARM': HARM, 'POLY_DEG': POLY_DEG,
                        'FIT_IN_LOGF': FIT_IN_LOGF, 'REF_MODEL_NAME': REF_MODEL_NAME,
                        'REF_TRACK': REF_TRACK}}
    with open(os.path.join(OUTDIR, 'typeii_picks.pkl'), 'wb') as fh:
        pickle.dump(picks, fh)

    # anything in OUTDIR older than this run is left over from a previous version of the analysis.
    # Mixing the two is exactly the kind of mistake that is impossible to spot later.
    stale = sorted(f for f in os.listdir(OUTDIR)
                   if not f.startswith('.')
                   and os.path.getmtime(os.path.join(OUTDIR, f)) < RUN_START)
    print('written to', OUTDIR)
    if stale:
        print(f'\n  *** WARNING: {len(stale)} file(s) in {OUTDIR} were NOT written by this run and '
              'are left over from an earlier one:')
        for f in stale:
            print(f'        {f}')
        print('      delete them, or move this run to a fresh OUTDIR, before using anything from '
              'that folder.')
    char_table.round(4)


def results_text(run):
    """Write the measured values and the summary paragraph."""
    # from the run
    BANDS_TRACED = run.BANDS_TRACED
    ALref = run.ALref
    REF_TRACK = run.REF_TRACK
    tg = run.tg
    t0 = run.t0
    LANE_BAND = run.LANE_BAND
    fh_test = run.fh_test
    scalars = run.scalars
    _fits = run._fits
    LANE_ORDER = run.LANE_ORDER
    SPLIT_PAIR = run.SPLIT_PAIR
    OUTDIR = run.OUTDIR
    EVENT_DATE = run.EVENT_DATE
    pol_table = run.pol_table
    passes = run.passes
    sweep = run.sweep
    LANE_ROLE = run.LANE_ROLE

    # ---- summary text, filled with the numbers actually measured above ----










    d = ALref[REF_TRACK]
    # The paragraph quotes two groups of numbers and has to keep them apart. r, v_sh and a describe
    # the burst and belong on the whole traced lane. v_A and B exist only where both branches of the
    # band are present, so they come with that window's own r and n_e - which for this lane is its
    # last 214 s, where an accelerating shock is 143 km/s faster than its lane average. X and M_A come
    # from scalars[], the same object the table above prints, so the paragraph cannot quote a density
    # jump that disagrees with its own table.
    _w0 = lane_windows(d, REF_TRACK, tg, t0, roles=LANE_ROLE)
    r0, r0e = _w0['r_span'], _w0['r_span_e']
    v0, v0e = _w0['v_span'], _w0['v_span_e']
    a0, a0e = _w0['a_span'], _w0['a_span_e']
    rS, rSe = _w0['r_split'], _w0['r_split_e']
    vA0, vA0e = _w0['vA_split'], _w0['vA_split_e']
    B0, B0e = _w0['B_split'], _w0['B_split_e']
    ne0 = _w0['ne_split']
    _bref = LANE_BAND[REF_TRACK]
    X0, X0e = scalars[_bref]['X']
    MA0, MA0e = scalars[_bref]['M_A']
    ta = (t0 + pd.Timedelta(seconds=float(np.nanmin(tg)))).strftime('%H:%M')
    tb = (t0 + pd.Timedelta(seconds=float(np.nanmax(tg)))).strftime('%H:%M')

    lines = [f'Type II radio burst, {EVENT_DATE}, NenuFAR', '=' * 78, '',
             f'observed {ta}-{tb} UT; one shock seen in both harmonics',
             f'every lane is analysed separately'
             + (f'; {JOINT} is the combined track' if MAKE_JOINT else ''), '']
    for b in BANDS_TRACED:
        s_ = scalars[b]
        lines += [f'{BAND_NAME[b]} band (s = {HARM[b]})', '-' * 78,
                  f'  drift rate                   {fmt_value(*s_["drift_MHz_s"], min_dp=4)} MHz/s',
                  f'  relative drift (1/f)(df/dt)  {fmt_value(*s_["rel_drift_s"], min_dp=5)} 1/s',
                  f'  relative band split          {fmt_value(*s_["rel_bandwidth"], min_dp=3)}',
                  f'  density jump X               {fmt_value(*s_["X"], min_dp=2)}   (model-independent)',
                  f'  Alfven Mach number M_A       {fmt_value(*s_["M_A"], min_dp=2)}   (model-independent)']
        for lab in LANE_ORDER[b]:
            d = ALref.get(lab)
            if d is None:
                continue
            ff = _fits[lab]
            wa = (t0 + pd.Timedelta(seconds=float(ff['tmin']))).strftime('%H:%M:%S')
            wb_ = (t0 + pd.Timedelta(seconds=float(ff['tmax']))).strftime('%H:%M:%S')
            # The frequency extent has to come from the TRACED POINTS, not from d['f_mean'], which is
            # the fit sampled on the shared N_GRID grid. That grid spans the whole burst, so its
            # spacing (~30 s here) is coarse next to a steeply drifting lane: the first grid point
            # inside the lane already sits several MHz below where the trace actually starts. It cost
            # H lane 1 6.7 MHz off the top of its reported range and H lane 2 9.1 MHz.
            _pts = passes[0].get(lab)
            fm = (np.asarray(_pts['f'], float) if _pts and _pts['f']
                  else d['f_mean'][np.isfinite(d['f_mean'])])
            pol = pol_table[pol_table['lane'] == lab]
            # Same helper as A.4 and the A.9 audit. Every row states the interval it was averaged over,
            # because r and v_sh on the full traced span and v_A and B on the band-split window are
            # different measurements: for F lane 1 the split window is the last 214 s of a 596 s lane
            # that is accelerating, and the mean speed there is 771 km/s against 628 over the whole
            # lane. Printing one of those under a heading naming the other interval is how a 23%
            # discrepancy gets into a manuscript.
            w = lane_windows(d, lab, tg, t0, roles=LANE_ROLE)
            lines += ['', f'  {lab}  [{LANE_ROLE[lab]}]',
                      f'    traced                     {wa}-{wb_} UT over '
                      f'{np.nanmin(fm):.1f}-{np.nanmax(fm):.1f} MHz',
                      f'    -- over the full traced span ({w["n_span"]} grid points) --',
                      f'    height r                   {fmt_value(w["r_span"], w["r_span_e"], 2)} Rsun',
                      f'    shock speed v_sh           {fmt_value(w["v_span"], w["v_span_e"], 0)} km/s',
                      f'    acceleration a             {fmt_value(w["a_span"], w["a_span_e"], 1)} m/s^2',
                      f'    signed mean V/I            '
                      f'{pol["V_over_I_mean"].iloc[0]:+.4f} +/- {pol["V_over_I_sd"].iloc[0]:.4f} (sd)'
                      if len(pol) else '']
            if not w['has_split']:
                lines += ['    -- band-split window: the two branches of this band never overlap, '
                          'so X, M_A, v_A and B are undefined --']
                continue
            lines += [f'    -- on the band-split window {w["split_window"]} '
                      f'({w["n_split"]} of {w["n_span"]} grid points) --',
                      f'    height r                   {fmt_value(w["r_split"], w["r_split_e"], 2)} Rsun',
                      f'    shock speed v_sh           {fmt_value(w["v_split"], w["v_split_e"], 0)} km/s',
                      f'    electron density n_e       {w["ne_split"]:.3e} cm^-3']
            if w['upstream']:
                lines += [f'    Alfven speed v_A           '
                          f'{fmt_value(w["vA_split"], w["vA_split_e"], 0)} km/s',
                          f'    magnetic field B           {fmt_value(w["B_split"], w["B_split_e"], 2)} G']
            else:
                lines += ['    Alfven speed v_A           not defined for a downstream branch',
                          '    magnetic field B           not defined for a downstream branch: v_A '
                          'from Rankine-Hugoniot is the',
                          '                               UPSTREAM Alfven speed, so pairing it with '
                          'this branch\'s density']
                lines += ['                               gives B_1 sqrt(X), neither B_1 nor B_2']
        lines += ['']

    lines += [f'{REF_TRACK} on the model x fold grid ({REF_MODEL_NAME} is the reference)', '-' * 78,
              f'  height r      {np.nanmin(sweep["r_Rsun"]):.2f}-{np.nanmax(sweep["r_Rsun"]):.2f} Rsun',
              f'  shock speed   {np.nanmin(sweep["v_kms"]):.0f}-{np.nanmax(sweep["v_kms"]):.0f} km/s',
              f'  Alfven speed  {np.nanmin(sweep["vA_kms"]):.0f}-{np.nanmax(sweep["vA_kms"]):.0f} km/s',
              f'  magnetic field {np.nanmin(sweep["B_G"]):.3f}-{np.nanmax(sweep["B_G"]):.3f} G', '']

    if len(fh_test):
        lines += ['Fundamental / harmonic consistency', '-' * 66]
        for _, row in fh_test.iterrows():
            nd = 5 if abs(row['value']) < 0.01 else 3        # the drifts are ~1e-3 per second
            exp = '' if not np.isfinite(row['expected']) else f'   (expected {row["expected"]:.{nd}f})'
            lines.append(f'  {row["test"]:34s} {fmt_value(row["value"], row["error"], nd)}{exp}')

    summary = '\n'.join(lines)
    run.set(summary=summary)
    print(summary)

    sF = scalars['F']
    sH = scalars.get('H', sF)
    _wF = _fits[SPLIT_PAIR['F'][0]]
    _wH = _fits[SPLIT_PAIR['H'][0]] if 'H' in SPLIT_PAIR else _wF
    para = (
        f"A type II radio burst was observed by NenuFAR on "
        f"{pd.Timestamp(EVENT_DATE).strftime('%d %B %Y')} between {ta} and {tb}~UT, in both the "
        f"fundamental and the harmonic of plasma emission. The fundamental drifted through the "
        f"observing band from "
        f"{(t0 + pd.Timedelta(seconds=float(_wF['tmin']))).strftime('%H:%M')}~UT and the harmonic, at "
        f"twice its frequency, entered the top of the band from "
        f"{(t0 + pd.Timedelta(seconds=float(_wH['tmin']))).strftime('%H:%M')}~UT and followed the same "
        f"shock to the end of the event; over the interval in which both were visible their frequency "
        f"ratio is "
        f"{fmt_tex(float(fh_test.iloc[0]['value']), float(fh_test.iloc[0]['error']), 2) if len(fh_test) else 'n/a'}, "
        f"confirming the identification. The relative drift rate is "
        f"{fmt_tex(*sF['rel_drift_s'], min_dp=4)}~s$^{{-1}}$. Both bands are separately band-split, "
        f"giving relative bandwidths of {fmt_tex(*sF['rel_bandwidth'], min_dp=2)} (F) and "
        f"{fmt_tex(*sH['rel_bandwidth'], min_dp=2)} (H). Interpreting the split as emission from the "
        f"upstream and downstream sides of the shock front gives, for the "
        f"{BAND_NAME[_bref].lower()} band, a density jump of "
        f"$X = {fmt_tex(X0, X0e, 2)[1:-1]}$ and, through the Rankine-Hugoniot relation for a perpendicular "
        f"shock at $\\gamma=5/3$ and $\\beta=0$, an Alfv\\'en Mach number of "
        f"$M_A = {fmt_tex(MA0, MA0e, 2)[1:-1]}$; neither depends on the assumed coronal density model. "
        f"Each band was then converted with its own harmonic number and each lane analysed separately"
        + (f", and the two were also combined into the single track {JOINT}" if MAKE_JOINT else
           "; the two bands were not merged into a joint track") + f". Adopting a "
        f"{REF_MODEL_NAME.replace(' x', ', fold-')} electron-density profile places {REF_TRACK} at "
        f"$r = {fmt_tex(r0, r0e, 2)[1:-1]}\\,R_\\odot$ with a shock speed of "
        f"$v_{{\\rm sh}} = {fmt_tex(v0, v0e, 0)[1:-1]}$~km~s$^{{-1}}$, both averaged over its full traced "
        f"span. Repeating the analysis over a grid "
        f"of five density models at folds 1--4 shifts the height to "
        f"{np.nanmin(sweep['r_Rsun']):.2f}--{np.nanmax(sweep['r_Rsun']):.2f}~$R_\\odot$ and the speed "
        f"to {np.nanmin(sweep['v_kms']):.0f}--{np.nanmax(sweep['v_kms']):.0f}~km~s$^{{-1}}$, which we "
        f"adopt as the systematic uncertainty. Over the {_w0['split_window']} interval in which both "
        f"branches of the band are present, at $r = {fmt_tex(rS, rSe, 2)[1:-1]}\\,R_\\odot$ and an "
        f"upstream density of {fmt_sci(ne0)}~cm$^{{-3}}$, the implied upstream Alfv\\'en speed is "
        f"$v_A = v_{{\\rm sh}}/M_A = {fmt_tex(vA0, vA0e, 0)[1:-1]}$~km~s$^{{-1}}$ and the coronal magnetic "
        f"field is $B = {fmt_tex(B0, B0e, 2)[1:-1]}$~G "
        f"({np.nanmin(sweep['B_G']):.2f}--{np.nanmax(sweep['B_G']):.2f}~G across the model grid). "
        + field_context(rS, B0)
    )

    with open(os.path.join(OUTDIR, 'typeii_results_text.md'), 'w') as fh:
        fh.write(f'# Type II results, NenuFAR {EVENT_DATE}\n\n'
                 '## Measured values\n\n```\n' + summary + '\n```\n\n'
                 '## Summary paragraph\n\n' + para + '\n')

    print('\n' + '=' * 78 + '\nSUMMARY PARAGRAPH\n' + '=' * 78)
    print(textwrap.fill(para, width=95))
    print('\nsaved', os.path.join(OUTDIR, 'typeii_results_text.md'))


def export_tracks(run):
    """Write the height-time tracks for every density model."""
    # from the run
    BANDS_TRACED = run.BANDS_TRACED
    TRACED = run.TRACED
    passes = run.passes
    OUTDIR = run.OUTDIR
    scalars = run.scalars
    SPLIT_PAIR = run.SPLIT_PAIR
    tg = run.tg
    MODEL_GRID = run.MODEL_GRID
    LANE_BAND = run.LANE_BAND
    t0 = run.t0
    OVERLAP = run.OVERLAP
    rdF = run.rdF
    rdH = run.rdH
    LANE_ROLE = run.LANE_ROLE

    # ---- r(t) for every lane through every model x fold ----------------------------------------
    tg_exp = np.arange(tg.min(), tg.max() + EXPORT_DT_S, EXPORT_DT_S)
    rows, summary = [], []
    run.set(summary=summary)

    for mf_name, model in tqdm(MODEL_GRID.items(), desc='exporting model x fold'):
        base, fold = mf_name.rsplit(' x', 1)
        agg = aggregate_lanes(run, passes, tg_exp, model, n_mc=EXPORT_N_MC)
        for lab in TRACED:
            d = agg.get(lab)
            if d is None:
                continue
            # keep only grid points where EVERY realisation contributed. At a partly covered endpoint
            # the mean is taken over a subset of the passes and jumps by more than its own error bar,
            # which shows up as the height track briefly running backwards.
            ok = np.isfinite(d['r_mean'])
            if 'r_n' in d:
                ok &= (d['r_n'] == d['r_n'].max())
            if not ok.any():
                continue
            band = LANE_BAND[lab]
            t_utc = [t0 + pd.Timedelta(seconds=float(x)) for x in tg_exp[ok]]
            rows.append(pd.DataFrame({
                'model': base, 'fold': int(fold), 'model_fold': mf_name,
                'band': band, 'lane': lab, 'role': LANE_ROLE.get(lab, ''),
                'harmonic_s': HARM[band],
                # keep the fractional second: the export grid is anchored on the first traced
                # sample, not on a whole second, so truncating drops a constant ~0.6 s from every row
                'time_UT': [x.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] for x in t_utc],
                't_sec_from_window_start': tg_exp[ok],
                'f_MHz': d['f_mean'][ok],
                'r_heliocentric_Rsun': d['r_mean'][ok],
                'r_err_Rsun': d['r_sd'][ok],
                'h_above_limb_Rsun': d['r_mean'][ok] - R_LIMB_RSUN,
                'v_kms': d['v_mean'][ok],
                'B_G': d['B_mean'][ok]}))
            v_, ev_ = grid_scalar(d, 'v')
            a_, ea_ = grid_scalar(d, 'a')
            B_, eB_ = grid_scalar(d, 'B')
            summary.append({
                'model': base, 'fold': int(fold), 'model_fold': mf_name, 'band': band, 'lane': lab,
                'role': LANE_ROLE.get(lab, ''),
                't_start_UT': t_utc[0].strftime('%H:%M:%S'),
                't_end_UT': t_utc[-1].strftime('%H:%M:%S'),
                'r_start_Rsun': d['r_mean'][ok][0], 'r_end_Rsun': d['r_mean'][ok][-1],
                'h_start_above_limb_Rsun': d['r_mean'][ok][0] - R_LIMB_RSUN,
                'h_end_above_limb_Rsun': d['r_mean'][ok][-1] - R_LIMB_RSUN,
                'v_mean_kms': v_, 'v_err_kms': ev_,
                'v_chord_kms': ((d['r_mean'][ok][-1] - d['r_mean'][ok][0]) * R_SUN_M / 1e3
                                / (tg_exp[ok][-1] - tg_exp[ok][0])),
                'a_mean_ms2': a_, 'a_err_ms2': ea_, 'B_mean_G': B_, 'B_err_G': eB_})

    track_export = pd.concat(rows, ignore_index=True)
    run.set(track_export=track_export)
    model_summary = pd.DataFrame(summary)
    run.set(model_summary=model_summary)
    track_export.to_csv(os.path.join(OUTDIR, 'height_time_all_models.csv'), index=False)
    model_summary.to_csv(os.path.join(OUTDIR, 'height_time_model_summary.csv'), index=False)

    # the two quantities no density model touches, written on their own so they cannot be confused
    # with anything that does depend on one
    mi = []
    for b in BANDS_TRACED:
        s = scalars[b]
        lo, up = SPLIT_PAIR[b]
        mi.append({'band': b, 'band_name': BAND_NAME[b], 'harmonic_s': HARM[b],
                   'upstream_lane': lo, 'downstream_lane': up,
                   'X': s['X'][0], 'X_err': s['X'][1],
                   'X_min_along_overlap': s['X_range'][0], 'X_max_along_overlap': s['X_range'][1],
                   'M_A': s['M_A'][0], 'M_A_err': s['M_A'][1],
                   'relative_bandwidth': s['rel_bandwidth'][0],
                   'relative_bandwidth_err': s['rel_bandwidth'][1],
                   # Both drift columns are averaged over THIS BAND'S OWN traced span, which is a
                   # different stretch of the burst for F and for H. Do not compare them across bands:
                   # the textbook "harmonic drifts twice as fast" applies at a common instant, and the
                   # ratio of these two numbers is not 2 because they cover different intervals. The
                   # cross-band test belongs on the overlap, and that value is the last column.
                   'drift_own_span_MHz_s': s['drift_MHz_s'][0],
                   'drift_own_span_err_MHz_s': s['drift_MHz_s'][1],
                   'rel_drift_own_span_per_s': s['rel_drift_s'][0],
                   'rel_drift_own_span_err_per_s': s['rel_drift_s'][1],
                   'rel_drift_over_FH_overlap_per_s': (rdF if b == 'F' else rdH)
                   if OVERLAP is not None else np.nan})
    model_independent = pd.DataFrame(mi)
    run.set(model_independent=model_independent)
    model_independent.to_csv(os.path.join(OUTDIR, 'model_independent_scalars.csv'), index=False)

    print(f'{len(track_export)} rows over {track_export.model_fold.nunique()} model x fold '
          f'combinations x {track_export.lane.nunique()} lanes, at {EXPORT_DT_S} s cadence')
    print('  height_time_all_models.csv      r(t) per lane per model, BOTH height conventions')
    print('  height_time_model_summary.csv   one row per model x fold x lane')
    print('  model_independent_scalars.csv   X and M_A, no density model involved')
    # quote the upstream branches, which is what a height-time comparison should use, and say so -
    # taking min/max over the whole table mixes in the downstream branches and widens the speed range
    # from 281-903 to 240-903 km/s under a label that says "upstream"
    _ups = model_summary[model_summary.role.str.startswith('upstream')]
    _lanes = ', '.join(sorted(_ups.lane.unique()))
    print(f'\nacross the grid the UPSTREAM lanes ({_lanes}) span '
          f'{_ups.h_start_above_limb_Rsun.min():.2f} to '
          f'{_ups.h_end_above_limb_Rsun.max():.2f} Rsun above the limb, '
          f'mean speed {_ups.v_mean_kms.min():.0f}-{_ups.v_mean_kms.max():.0f} km/s')
    print(f'  all four lanes together span {model_summary.h_start_above_limb_Rsun.min():.2f} to '
          f'{model_summary.h_end_above_limb_Rsun.max():.2f} Rsun and '
          f'{model_summary.v_mean_kms.min():.0f}-{model_summary.v_mean_kms.max():.0f} km/s')
    print('  these are means over each lane\'s FULL traced span, which is the right window for a '
          'height-time comparison')
    model_summary.head(8).round(3)


def plot_export_tracks(run):
    """Every exported height-time track, above the limb."""
    # from the run
    (BANDS_TRACED, track_export, BASE_MODELS, SPLIT_PAIR, EVENT_DATE) = (
        run.BANDS_TRACED, run.track_export, run.BASE_MODELS, run.SPLIT_PAIR, run.EVENT_DATE
    )

    # one panel per band: every model x fold track for that band's UPSTREAM lane, which is the shock
    # front proper. Plotted as height ABOVE THE LIMB, the convention coronagraph tracks usually use.
    fig, axes = plt.subplots(1, 2, figsize=[14, 5.6], sharey=True)
    _styles = {1: ':', 2: '-', 3: '--', 4: '-.'}
    _cols = {m: c for m, c in zip(BASE_MODELS, plt.cm.tab10.colors)}

    for ax, b in zip(axes, BANDS_TRACED):
        lo = SPLIT_PAIR[b][0]
        sub = track_export[track_export.lane == lo]
        for mf, g in sub.groupby('model_fold'):
            base, fold = mf.rsplit(' x', 1)
            ax.plot(pd.to_datetime(g.time_UT), g.h_above_limb_Rsun, lw=1.3,
                    color=_cols[base], ls=_styles[int(fold)], alpha=0.85,
                    label=(base if fold == '1' else None))
        ref = sub[sub.model_fold == REF_MODEL_NAME]
        if len(ref):
            ax.plot(pd.to_datetime(ref.time_UT), ref.h_above_limb_Rsun, lw=3.2, color='k',
                    alpha=0.85, label=f'{REF_MODEL_NAME} (reference)')
        ax.set_title(f'{BAND_NAME[b]} (s = {HARM[b]}), {lo}')
        ax.set_xlabel(f'Time (UT) on {EVENT_DATE}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'height above the limb  $r - R_\odot$  [$R_\odot$]')
    axes[0].legend(fontsize=8, ncol=2, title='line style = fold 1-4')
    fig.suptitle('Type II height-time track under every density model x fold\n'
                 'heights are ABOVE THE LIMB; add 1 Rsun for heliocentric r', y=1.02, fontsize=11)
    fig.tight_layout()
    save_fig(fig, 'height_time_all_models')
    plt.show()
