"""Interactive lane tracing and its quality control.

Lanes are placed by hand, either by clicking points or by fitting a cubic Bezier
with sliders. METHODS.md section 2 covers the repeat convention and what the
per-point frequency uncertainty does and does not measure.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

try:   # only the interactive tracer needs these; a headless re-run from saved picks
    from ipywidgets import (BoundedIntText, HBox, IntSlider, Label, Layout, VBox,
                            jslink)
except ImportError:    # pragma: no cover - notebook-only dependency
    BoundedIntText = HBox = IntSlider = Label = Layout = VBox = jslink = None

from .config import (BAND_NAME, BANDS, BEZIER_ANCHORS, BEZIER_NUM_POINTS, BEZIER_SEED,
                     N_REPS, TRACE_METHOD)
from .spectra import draw_layer


def draw_bezier(x1=0, y1=0, x2=0, y2=0, controls=[[0, 0]], n=2, num_points=30):
    """Sample a quadratic or cubic Bezier through the given control points."""
    P0 = np.array([x1, y1], dtype=float)
    P3 = np.array([x2, y2], dtype=float)
    t = np.linspace(0, 1, num_points)
    if n == 1:
        curve = (1 - t)[:, None] * P0 + t[:, None] * P3
    elif n == 2:
        P1 = np.array(controls[0], dtype=float)
        curve = ((1 - t)[:, None] ** 2 * P0
                 + 2 * (1 - t)[:, None] * t[:, None] * P1
                 + t[:, None] ** 2 * P3)
    elif n == 3:
        P1 = np.array(controls[0], dtype=float)
        P2 = np.array(controls[1], dtype=float)
        curve = ((1 - t)[:, None] ** 3 * P0
                 + 3 * (1 - t)[:, None] ** 2 * t[:, None] * P1
                 + 3 * (1 - t)[:, None] * t[:, None] ** 2 * P2
                 + t[:, None] ** 3 * P3)
    else:
        raise ValueError('n must be 1 (linear), 2 (quadratic) or 3 (cubic).')
    return curve

def bezier_indices(x1, y1, x2, y2, controls, n, num_points=BEZIER_NUM_POINTS):
    """Bezier samples as integer (time, frequency) indices into the layer."""
    curve = draw_bezier(x1, y1, x2, y2, controls, n, num_points)
    xi = np.clip(np.round(curve[:, 0]).astype(int), 0, len(LAYER_T) - 1)
    yi = np.clip(np.round(curve[:, 1]).astype(int), 0, len(LAYER_F) - 1)
    return xi, yi

def bezier_freq_lane(x_idx, y_idx):
    """Map integer curve samples onto a {t, f} lane in MHz.

    Sorted in time with duplicate times averaged. Where the curve is steep
    several samples fall in one time column and their frequencies differ by
    several MHz, so keeping the first would bias the lane.
    """
    t_sel = pd.to_datetime([LAYER_T[i] for i in x_idx])
    f_val = np.asarray(LAYER_F[y_idx], float)
    x_num = mdates.date2num(t_sel)
    uniq_x, inv = np.unique(x_num, return_inverse=True)
    f_mean = np.bincount(inv, weights=f_val) / np.bincount(inv)
    t_uniq = pd.to_datetime(mdates.num2date(uniq_x)).tz_localize(None)
    return {'t': list(t_uniq), 'f': [float(v) for v in f_mean]}

class LaneTracer:
    """Interactive tracer: one figure, both methods, named lanes.

    Click points along a lane or place a Bezier with sliders, switching at any
    time. Lanes are recorded as "F lane 1", "H lane 1" and so on; which branch of
    a split is upstream is decided later, in physics.order_lanes, by frequency.
    Traces persist in TRACE_STORE across re-runs of the cell; pass reset=True to
    discard them.
    """

    SLIDERS = [('x_start', 'start x (time)', 'x'), ('y_start', 'start y (freq)', 'y'),
               ('x_end', 'end x (time)', 'x'), ('y_end', 'end y (freq)', 'y'),
               ('cx1', 'anchor 1 x', 'x'), ('cy1', 'anchor 1 y', 'y'),
               ('cx2', 'anchor 2 x', 'x'), ('cy2', 'anchor 2 y', 'y')]

    def __init__(self, n_reps=N_REPS, bands=None, method=TRACE_METHOD, anchors=BEZIER_ANCHORS,
                 reset=False):
        import ipywidgets as widgets
        from IPython.display import display
        if reset:
            TRACE_STORE.clear()
            TRACE_HISTORY.clear()
            TRACE_KIND.clear()
        self.n_reps = n_reps
        self.bands = list(bands) if bands else list(BANDS)
        self.band = self.bands[0]
        self.method = method
        self.n_anchors = anchors
        self.traces = TRACE_STORE            # bound, not copied: survives re-running the cell
        self.history = TRACE_HISTORY
        self.lane_no = {b: 1 for b in self.bands}
        self.current = {'t': [], 'f': []}
        self.rng = np.random.default_rng(BEZIER_SEED)
        for b in self.bands:
            self._sync_label(b)
        if self.traces:
            print(f'restored {len(self.traces)} lane(s) already recorded: '
                  + ', '.join(f'{l} ({len(v)} rep)' for l, v in sorted(self.traces.items())))
            print('pass reset=True to LaneTracer if you want to start over')
        self._build(widgets, display)

    # ---------------------------------------------------------------- labels and state
    @property
    def label(self):
        return f'{self.band} lane {self.lane_no[self.band]}'

    def _used(self, band):
        return sorted(int(l.split()[-1]) for l in self.traces if l.startswith(band + ' lane '))

    def _sync_label(self, band=None):
        """Point the current label at a lane that is still being filled, otherwise at the next
        free number. Derived from `self.traces`, never from a free-running counter."""
        b = band or self.band
        used = self._used(b)
        n_cur = len(self.traces.get(f'{b} lane {self.lane_no[b]}', []))
        if 0 < n_cur < self.n_reps:
            return
        self.lane_no[b] = (max(used) + 1) if used else 1

    @property
    def labels(self):
        """Every recorded label, ordered by band then lane number."""
        return sorted(self.traces, key=lambda l: (self.bands.index(l.split()[0]),
                                                  int(l.split()[-1])))

    @property
    def order(self):
        return self.labels

    def colour(self, label):
        b = label.split()[0]
        i = int(label.split()[-1]) - 1
        cmap = plt.cm.winter if b == 'F' else plt.cm.autumn
        return cmap(min(i, 3) / 3.5)

    @property
    def passes(self):
        """The k-th repeat of every label assembled into pass k, so the lanes of a pass are
        mutually consistent. Labels with fewer repeats cycle through the ones they have."""
        if not self.traces:
            return []
        n = max(len(v) for v in self.traces.values())
        out = []
        for k in range(n):
            pas = {}
            for lab in self.labels:
                reps = self.traces[lab]
                if reps:
                    r = reps[k % len(reps)]
                    pas[lab] = {'t': list(r['t']), 'f': list(r['f'])}
            out.append(pas)
        return out

    def summary(self):
        """What is currently recorded. Run `tracer.summary()` at any point to check."""
        rows = []
        for lab in self.labels:
            reps = self.traces[lab]
            f = np.concatenate([np.asarray(r['f'], float) for r in reps])
            t = pd.to_datetime(np.concatenate([np.asarray(r['t']) for r in reps]))
            rows.append({'lane': lab, 'band': lab.split()[0], 'repeats': len(reps),
                         'repeat_kind': TRACE_KIND.get(lab, 'independent traces'),
                         'points_per_repeat': int(np.median([len(r['f']) for r in reps])),
                         'start_UT': t.min().strftime('%H:%M:%S'),
                         'end_UT': t.max().strftime('%H:%M:%S'),
                         'f_min_MHz': round(float(f.min()), 2),
                         'f_max_MHz': round(float(f.max()), 2)})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- figure and widgets
    def _build(self, widgets, display):
        from ipywidgets import IntSlider, BoundedIntText, HBox, VBox, Label, Layout, jslink
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=[13, 6])
        plt.ion()
        try:
            self.fig.canvas.header_visible = False
        except AttributeError:
            pass
        pm = draw_layer(RUN, self.ax)
        self.fig.colorbar(pm, ax=self.ax, pad=0.01, label=CBAR_LABEL)

        (self.live,) = self.ax.plot([], [], 'o-', color='k', ms=6, lw=1.4, mfc='white',
                                    mew=1.4, animated=True)
        (self.bez_line,) = self.ax.plot([], [], '-', color='k', lw=2.2, animated=True)
        (self.bez_ends,) = self.ax.plot([], [], 'o', mfc='white', mec='black', mew=1.5, ms=9,
                                        ls='none', animated=True)
        (self.bez_anch,) = self.ax.plot([], [], '^', mfc='white', mec='black', mew=1.3, ms=10,
                                        ls='none', animated=True)
        self.done_lines = {}
        self._bg = None
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self.sel = widgets.ToggleButtons(
            options=[(f'{BAND_NAME[b]} ({b})', b) for b in self.bands], value=self.band,
            description='Band:', style={'description_width': 'initial'})
        self.meth = widgets.ToggleButtons(
            options=[('Click points', 'click'), ('Bezier curve', 'bezier')], value=self.method,
            description='Method:', style={'description_width': 'initial'})
        self.deg = widgets.Dropdown(options=[('quadratic', 1), ('cubic', 2)], value=self.n_anchors,
                                    description='Bezier:', layout=Layout(width='170px'),
                                    style={'description_width': 'initial'})
        self.jit = widgets.Checkbox(value=True, description=f'auto-repeats ({self.n_reps})',
                                    indent=False, layout=Layout(width='170px'))
        self.btn_end = widgets.Button(description='End trace', icon='check',
                                      button_style='success',
                                      tooltip='record the trace in progress')
        self.btn_undo = widgets.Button(description='Undo point', icon='rotate-left')
        self.btn_new = widgets.Button(description='New lane', icon='plus',
                                      tooltip='start the next lane of this band')
        self.btn_del = widgets.Button(description='Delete last', icon='trash',
                                      button_style='danger',
                                      tooltip='drop the most recently recorded trace')
        self.status = widgets.HTML()

        d = self._defaults()
        self._sliders, rows = {}, []
        for name, lab, axis in self.SLIDERS:
            hi = (len(LAYER_T) if axis == 'x' else len(LAYER_F)) - 1
            sld = IntSlider(value=d[name], min=0, max=hi, step=1, readout=False,
                            continuous_update=True, layout=Layout(width='520px'))
            box = BoundedIntText(value=d[name], min=0, max=hi, step=1, layout=Layout(width='95px'))
            jslink((sld, 'value'), (box, 'value'))
            sld.observe(self._on_slider, names='value')
            self._sliders[name] = sld
            rows.append(HBox([Label(lab, layout=Layout(width='120px')), sld, box]))
        self._rows = rows
        self.bez_box = VBox(rows)

        self.sel.observe(self._on_band, names='value')
        self.meth.observe(self._on_method, names='value')
        self.deg.observe(self._on_deg, names='value')
        self.btn_end.on_click(lambda _: self.end_trace())
        self.btn_undo.on_click(lambda _: self.undo())
        self.btn_new.on_click(lambda _: self.new_lane())
        self.btn_del.on_click(lambda _: self.delete_last())
        display(VBox([HBox([self.sel, self.meth]),
                      HBox([self.btn_end, self.btn_undo, self.btn_new, self.btn_del,
                            self.deg, self.jit]),
                      self.bez_box, self.status, self.fig.canvas]))
        for lab in self.labels:
            self._redraw_label(lab)
        self._apply_method()
        self._title()
        self._refresh()

    def _defaults(self):
        """Seed the Bezier on the brightest channel near the start, middle and end of the window,
        so the curve already lies near an emission lane before you touch it."""
        nt, nf = len(LAYER_T), len(LAYER_F)
        xs = [int(f * (nt - 1)) for f in (0.10, 0.35, 0.65, 0.90)]
        w = max(1, nt // 40)
        ys = []
        for x in xs:
            col = np.nanmean(LAYER_D[:, max(0, x - w):min(nt, x + w) + 1], axis=1)
            ys.append(int(np.nanargmax(col)) if np.isfinite(col).any() else nf // 2)
        return dict(x_start=xs[0], y_start=ys[0], x_end=xs[3], y_end=ys[3],
                    cx1=xs[1], cy1=ys[1], cx2=xs[2], cy2=ys[2])

    def _title(self):
        n = len(self.traces.get(self.label, []))
        how = 'click points' if self.method == 'click' else 'place the Bezier with the sliders'
        self.ax.set_title(f'tracing {self.label}  -  repeat {min(n + 1, self.n_reps)} of '
                          f'{self.n_reps}   |   {how}, then End trace')

    # ---------------------------------------------------------------- blitting
    def _on_draw(self, event=None):
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self._blit()

    def _blit(self):
        for ln in self.done_lines.values():
            self.ax.draw_artist(ln)
        for art in (self.live, self.bez_line, self.bez_ends, self.bez_anch):
            self.ax.draw_artist(art)

    def _refresh(self):
        if self.method == 'click':
            if self.current['f']:
                self.live.set_data(mdates.date2num(pd.to_datetime(self.current['t'])),
                                   self.current['f'])
            else:
                self.live.set_data([], [])
        else:
            v = {k: s.value for k, s in self._sliders.items()}
            anch = self._anchors(v)
            curve = draw_bezier(v['x_start'], v['y_start'], v['x_end'], v['y_end'],
                                anch, self.n_anchors + 1, BEZIER_NUM_POINTS)
            xi = np.clip(np.round(curve[:, 0]).astype(int), 0, len(LAYER_T) - 1)
            yi = np.clip(np.round(curve[:, 1]).astype(int), 0, len(LAYER_F) - 1)
            self.bez_line.set_data(mdates.date2num(pd.to_datetime(LAYER_T[xi])), LAYER_F[yi])
            self.bez_ends.set_data(mdates.date2num(pd.to_datetime(
                LAYER_T[[v['x_start'], v['x_end']]])), LAYER_F[[v['y_start'], v['y_end']]])
            self.bez_anch.set_data(mdates.date2num(pd.to_datetime(
                LAYER_T[[int(a[0]) for a in anch]])), LAYER_F[[int(a[1]) for a in anch]])
        if self._bg is not None:
            self.fig.canvas.restore_region(self._bg)
            self._blit()
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw_idle()
        self._update_status()

    def _line_for(self, label):
        if label not in self.done_lines:
            (self.done_lines[label],) = self.ax.plot([], [], '-', lw=2, alpha=0.9,
                                                     color=self.colour(label), animated=True,
                                                     label=label)
        return self.done_lines[label]

    def _redraw_label(self, label):
        xs, ys = [], []
        for rep in self.traces.get(label, []):
            xs += list(mdates.date2num(pd.to_datetime(rep['t']))) + [np.nan]
            ys += list(rep['f']) + [np.nan]
        self._line_for(label).set_data(xs, ys)

    # ---------------------------------------------------------------- Bezier helpers
    def _anchors(self, v):
        if self.n_anchors == 1:
            return [(v['cx1'], v['cy1'])]
        return [(v['cx1'], v['cy1']), (v['cx2'], v['cy2'])]

    def _bezier_rep(self, jitter=0):
        v = {k: s.value for k, s in self._sliders.items()}
        base = np.array([[v['x_start'], v['y_start']], [v['x_end'], v['y_end']],
                         *self._anchors(v)], float)
        if jitter > 0:
            base = base + self.rng.normal(0, jitter, base.shape)
        (sx, sy), (ex, ey) = base[0], base[1]
        xi, yi = bezier_indices(sx, sy, ex, ey, [list(p) for p in base[2:]], self.n_anchors + 1)
        return bezier_freq_lane(xi, yi)

    def _apply_method(self):
        show = '' if self.method == 'bezier' else 'none'
        self.bez_box.layout.display = show
        self.deg.layout.display = show
        self.jit.layout.display = show
        self.btn_undo.disabled = self.method != 'click'
        for i, row in enumerate(self._rows):
            row.layout.display = 'none' if (self.method != 'bezier' or
                                            (i >= 6 and self.n_anchors == 1)) else ''

    # ---------------------------------------------------------------- interaction
    def _on_band(self, change):
        self.band = change['new']
        self.current = {'t': [], 'f': []}
        self._sync_label()
        self._title()
        self.fig.canvas.draw_idle()
        self._refresh()

    def _on_method(self, change):
        self.method = change['new']
        self.current = {'t': [], 'f': []}
        self.live.set_data([], [])
        for art in (self.bez_line, self.bez_ends, self.bez_anch):
            art.set_data([], [])
        self._apply_method()
        self._title()
        self.fig.canvas.draw_idle()
        self._refresh()

    def _on_deg(self, change):
        self.n_anchors = change['new']
        self._apply_method()
        self._refresh()

    def _on_slider(self, change=None):
        self._refresh()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        mode = getattr(getattr(self.fig.canvas, 'toolbar', None), 'mode', '')
        if mode:                                  # zoom or pan is active, so do not add points
            return
        if self.method != 'click':
            # in Bezier mode the curve is placed with the sliders, so a stray click on the plot
            # must not record anything: End trace is the only way to commit it
            return
        if event.button == 3:
            self.end_trace()
            return
        self.current['t'].append(pd.Timestamp(mdates.num2date(event.xdata).replace(tzinfo=None)))
        self.current['f'].append(float(event.ydata))
        self._refresh()

    def undo(self):
        if self.method == 'click' and self.current['t']:
            self.current['t'].pop()
            self.current['f'].pop()
        self._refresh()

    def end_trace(self):
        lab = self.label
        if self.method == 'click':
            if len(self.current['f']) < 2:
                self._update_status('a trace needs at least two points', warn=True)
                return
            new_reps = [{'t': list(self.current['t']), 'f': list(self.current['f'])}]
            self.current = {'t': [], 'f': []}
        else:
            done = len(self.traces.get(lab, []))
            if done >= self.n_reps:
                self._update_status(f'{lab} already has its {self.n_reps} repeats - press '
                                    '"New lane" to move on', warn=True)
                return
            if self.jit.value:
                # repeat 0 is the curve you placed; the rest jitter the control points, which is
                # the Bezier equivalent of re-tracing a lane by hand
                new_reps = [self._bezier_rep(0 if (done + k) == 0 else BEZIER_JITTER)
                            for k in range(self.n_reps - done)]
            else:
                new_reps = [self._bezier_rep(0)]
            # keep the numbers the curve was placed with, so the lane can be rebuilt without the
            # widget and without the sampled trace
            _v = {k: s.value for k, s in self._sliders.items()}
            TRACE_CONTROLS[lab] = {'x_start': _v['x_start'], 'y_start': _v['y_start'],
                                   'x_end': _v['x_end'], 'y_end': _v['y_end'],
                                   'cx1': _v['cx1'], 'cy1': _v['cy1'],
                                   **({'cx2': _v['cx2'], 'cy2': _v['cy2']}
                                      if self.n_anchors == 2 else {}),
                                   'n_anchors': self.n_anchors,
                                   'num_points': BEZIER_NUM_POINTS, 'n_reps': self.n_reps,
                                   'jitter_seed': BEZIER_SEED}
        self.traces.setdefault(lab, []).extend(new_reps)
        self.history.extend([lab] * len(new_reps))
        # Any jittered repeat makes the lane's spread a jitter measurement, however many were
        # added. Keying this on len(new_reps) > 1 mislabels a lane that already had repeats and
        # is being topped up one at a time, and that label decides whether A.4 is allowed to
        # divide the spread by sqrt(N_pass).
        jittered = (self.method == 'bezier' and self.jit.value
                    and any((len(self.traces[lab]) - len(new_reps) + k) > 0
                            for k in range(len(new_reps))))
        if jittered:
            TRACE_KIND[lab] = 'bezier auto-repeats'
        else:
            TRACE_KIND.setdefault(lab, 'independent traces')
        self._redraw_label(lab)
        msg = f'recorded {lab}, {len(self.traces[lab])}/{self.n_reps} repeat(s)'
        self._sync_label()
        if self.label != lab:
            msg += f' - moving on to {self.label}'
        self._title()
        self.fig.canvas.draw_idle()
        self._refresh()
        self._update_status(msg)

    def new_lane(self):
        used = self._used(self.band)
        self.lane_no[self.band] = (max(used) + 1) if used else 1
        self.current = {'t': [], 'f': []}
        self._title()
        self.fig.canvas.draw_idle()
        self._refresh()
        self._update_status(f'started {self.label}')

    def delete_last(self):
        if not self.history:
            self._update_status('nothing recorded yet', warn=True)
            return
        lab = self.history.pop()
        if self.traces.get(lab):
            self.traces[lab].pop()
        if lab in self.traces and not self.traces[lab]:
            del self.traces[lab]
            # drop the kind with the lane. Leaving it behind means a lane later re-traced by hand
            # keeps the deleted lane's 'bezier auto-repeats' label, or worse, a lane later filled
            # with jittered repeats keeps 'independent traces' and gets its spread divided by
            # sqrt(N_pass) in A.4 as though it had been re-traced.
            TRACE_KIND.pop(lab, None)
        self._redraw_label(lab)
        self._sync_label(lab.split()[0])
        self._sync_label()
        self._title()
        self.fig.canvas.draw_idle()
        self._refresh()
        self._update_status(f'deleted one repeat of {lab}, '
                            f'{len(self.traces.get(lab, []))} left on it')

    def _update_status(self, msg='', warn=False):
        if self.traces:
            bits = []
            for lab in self.labels:
                reps = len(self.traces[lab])
                col = mpl.colors.to_hex(self.colour(lab))
                mark = '' if reps >= self.n_reps else ' <i>(incomplete)</i>'
                bits.append(f'<span style="color:{col}"><b>{lab}</b> {reps}/{self.n_reps}'
                            f'{mark}</span>')
            rec = ' &nbsp;|&nbsp; '.join(bits)
            tot = (f'{len(self.traces)} lane(s), '
                   f'{sum(len(v) for v in self.traces.values())} trace(s)')
        else:
            rec = '<i>nothing recorded yet</i>'
            tot = '0 lanes'
        extra = (f'&nbsp; <b>points in this trace:</b> {len(self.current["f"])}'
                 if self.method == 'click' else '&nbsp; <b>method:</b> Bezier')
        colour = '#a00' if warn else '#060'
        self.status.value = (f'<b>now tracing:</b> {self.label} {extra}<br>'
                             f'<b>recorded ({tot}):</b> {rec}<br>'
                             f'<span style="color:{colour}">{msg}</span>')

def ridge_rms(run, lab, half_ch, shift_ch=0):
    """rms offset [channels] between a lane and the brightest pixel within half_ch.

    shift_ch displaces the lane in frequency first, which puts the same curve over
    blank spectrum as a control.
    """
    # state this step works on
    LAYER_T, LAYER_F, LAYER_D = run.LAYER_T, run.LAYER_F, run.LAYER_D
    passes = run.passes

    L = passes[0][lab]
    ti = np.searchsorted(LAYER_T, pd.to_datetime(L['t']).to_numpy()).clip(0, len(LAYER_T) - 1)
    fi = (np.abs(LAYER_F[:, None] - np.asarray(L['f'], float)[None, :]).argmin(axis=0)
          + shift_ch).clip(0, len(LAYER_F) - 1)
    o = []
    for it, jf in zip(ti, fi):
        lo_, hi_ = max(0, jf - half_ch), min(len(LAYER_F), jf + half_ch + 1)
        col = LAYER_D[lo_:hi_, it]
        if np.isfinite(col).any():
            o.append(int(np.nanargmax(col)) + lo_ - jf)
    return np.sqrt(np.nanmean(np.asarray(o, float) ** 2)) if o else np.nan



# Recorded traces live here rather than on the tracer object, so re-running the cell that opens
# the tracer never discards them. LaneTracer(reset=True) clears them deliberately.
TRACE_STORE = {}     # label -> list of repeats, each {'t': [...], 'f': [...]}
TRACE_HISTORY = []   # labels in the order they were recorded, for Delete last
TRACE_KIND = {}      # label -> how the repeats were produced. Jittered copies of one Bezier are
                     # not independent re-tracings and their spread is not a reproducibility error
TRACE_CONTROLS = {}  # label -> the Bezier control points it was placed with, in layer index units.
                     # A trace is 80 sampled frequencies; these six or eight numbers are what the
                     # person actually chose, and they are what makes the lane reproducible.

# The tracer is a widget bound to one displayed layer, so the layer it draws on is held here and
# set by bind_layer once the layer exists. Everything else in the package takes its state as an
# argument; this is the exception an interactive figure forces.
LAYER_T = LAYER_F = LAYER_D = None
CBAR_LABEL = ''
BEZIER_JITTER = None
RUN = None


def bind_layer(run):
    """Point the tracer at this run's layer. Called once, by pipeline.build_layer."""
    global LAYER_T, LAYER_F, LAYER_D, CBAR_LABEL, BEZIER_JITTER, RUN
    LAYER_T, LAYER_F, LAYER_D = run.LAYER_T, run.LAYER_F, run.LAYER_D
    CBAR_LABEL = run.CBAR_LABEL
    BEZIER_JITTER = run.BEZIER_JITTER
    RUN = run


class ReplayTracer:
    """A tracer that replays recorded control points instead of opening a widget.

    Presents the same surface pipeline.collect_traces uses - traces, passes, labels, colour,
    n_reps - so the rest of the analysis cannot tell the difference between a lane placed by
    hand and one rebuilt from its six numbers.
    """
    labels = LaneTracer.labels
    order = LaneTracer.order
    colour = LaneTracer.colour
    passes = LaneTracer.passes
    summary = LaneTracer.summary

    def __init__(self, controls, bands=None, jitter=None):
        self.n_anchors = int(controls[next(iter(controls))]['n_anchors'])
        self.n_reps = int(controls[next(iter(controls))]['n_reps'])
        self.bands = list(bands) if bands else list(BANDS)
        self.traces = TRACE_STORE
        self.history = TRACE_HISTORY
        self.lane_no = {b: 1 for b in self.bands}
        self.method = 'bezier'
        self.controls = controls
        self._replay(BEZIER_JITTER if jitter is None else jitter)

    def _replay(self, jitter):
        """Rebuild every lane, in the recorded order, off one seeded stream.

        The rng is shared across lanes exactly as it is in a live session, so the draws a lane
        receives depend on how many lanes preceded it. Replaying them in a different order
        returns different repeats - the same curve, a different jitter realisation.
        """
        TRACE_STORE.clear(); TRACE_HISTORY.clear(); TRACE_KIND.clear()
        TRACE_CONTROLS.clear(); TRACE_CONTROLS.update(self.controls)
        rng = np.random.default_rng(int(list(self.controls.values())[0]['jitter_seed']))
        for lab, c in self.controls.items():
            pts = [[c['x_start'], c['y_start']], [c['x_end'], c['y_end']], [c['cx1'], c['cy1']]]
            if int(c['n_anchors']) == 2:
                pts.append([c['cx2'], c['cy2']])
            base0 = np.array(pts, float)
            reps = []
            for k in range(int(c['n_reps'])):
                base = base0 if k == 0 else base0 + rng.normal(0, jitter, base0.shape)
                (sx, sy), (ex, ey) = base[0], base[1]
                xi, yi = bezier_indices(sx, sy, ex, ey, [list(p) for p in base[2:]],
                                        int(c['n_anchors']) + 1, int(c['num_points']))
                reps.append(bezier_freq_lane(xi, yi))
            TRACE_STORE[lab] = reps
            TRACE_HISTORY.extend([lab] * len(reps))
            TRACE_KIND[lab] = ('bezier auto-repeats' if int(c['n_reps']) > 1
                               else 'independent traces')
        for b in self.bands:
            used = [int(l.split()[-1]) for l in TRACE_STORE if l.startswith(b + ' lane ')]
            self.lane_no[b] = (max(used) + 1) if used else 1


def read_controls(path):
    """Read a Bezier control-point table written by pipeline.export_controls.

    Row order is significant: it is the order the lanes were traced in, which is the order the
    shared jitter stream is consumed in.
    """
    df = pd.read_csv(path)
    need = {'lane', 'x_start', 'y_start', 'x_end', 'y_end', 'cx1', 'cy1', 'n_anchors',
            'num_points', 'n_reps', 'jitter_seed'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'{path} is missing column(s): {sorted(missing)}')
    return {r['lane']: {k: v for k, v in r.items() if k != 'lane'}
            for _, r in df.iterrows()}
