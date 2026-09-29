# -*- coding: utf-8 -*-
"""Constants and tunables for the type II analysis.

Physical constants are taken from astropy. Everything below the first section is a choice; the
reasons for each are in METHODS.md, and the entries most worth revisiting for a different event
are the display stretch, the tracing repeats and the fit degrees.
"""
import numpy as np
from astropy.constants import R_sun, c, e, eps0, m_e, m_p, mu0

# --- physical constants, SI ---------------------------------------------------------------
R_SUN_M    = R_sun.to('m').value
C_MS       = c.to('m/s').value
E_CHARGE_J = e.si.value
M_E        = m_e.value
M_P        = m_p.value
EPS0       = eps0.value
MU0        = mu0.value

# electron plasma frequency: f_p [Hz] = PLASMA_CONST * sqrt(n_e [cm^-3])
PLASMA_CONST = (1 / (2 * np.pi)) * np.sqrt(1e6 * E_CHARGE_J ** 2 / (EPS0 * M_E))

MU = 1.27                        # mass per electron in units of m_p, for rho = MU m_p n_e.
                                 # The common choice in the type II literature; the fully ionised
                                 # 10% He value of 1.167 lowers every B by 4.2%.

# --- bands ---------------------------------------------------------------------------------
BANDS     = ['F', 'H']           # offered by the tracer, in this order
HARM      = {'F': 1, 'H': 2}     # harmonic number s per band
BAND_NAME = {'F': 'Fundamental', 'H': 'Harmonic'}
JOINT     = 'F+H joint'          # key of the combined fundamental + harmonic track

# --- event window --------------------------------------------------------------------------
TYPEII_START  = '09:18:00'       # UT, start of the window the burst is analysed over
TYPEII_END    = '09:52:00'
TYPEII_FLIM   = [25, 85]         # MHz, the part of the band the burst occupies

# --- tracing ---------------------------------------------------------------------------------
N_REPS            = 3            # repeats recorded per lane
TRACE_METHOD      = 'click'      # method the tracer opens with; switchable live
BEZIER_ANCHORS    = 1            # 1 -> quadratic, 2 -> cubic; the 26 Mar 2025 lanes are quadratic
BEZIER_NUM_POINTS = 80           # samples taken along the curve when it is recorded
BEZIER_JITTER_MHZ = 0.4          # target 1-sigma frequency displacement of the auto-repeats
BEZIER_SEED       = 0
BEZIER_CURVE_ATTEN = 0.676       # a cubic control-point displacement moves the curve by this much

# --- display ----------------------------------------------------------------------------------
LAYER_MAX_T  = 4000              # time samples kept in the tracing layer
LAYER_MAX_F  = 1200              # frequency channels kept; decimation is in time only
LAYER_MODE   = 'db_sub'          # 'db_sub' -> dB above background, 'ratio' -> linear ratio
LAYER_CMAP   = 'Spectral_r'
LAYER_PLO    = 2                 # lower percentile of the colour scale
LAYER_PHI    = 98                # upper percentile
LAYER_GAMMA  = 0.6               # <1 brightens faint features, 1 = linear
INVERT_FREQ  = True              # low frequency at the top
PREVIEW_MAX_TCOLS = 1600         # further time down-sampling, previews only

# --- density models ---------------------------------------------------------------------------
FOLDS          = [1, 2, 3, 4]
REF_MODEL_NAME = 'Newkirk x2'    # model x fold used for the reference figures and tables
R_BOUNDS       = (1, 5)          # heliocentric range over which n_e(r) is inverted

# --- lane fits --------------------------------------------------------------------------------
FIT_IN_LOGF     = True           # fit log10 f(t); over an octave a polynomial in f is too stiff
LANE_DEG        = 3              # degree of the lane fit. Degree 2 leaves a misfit that reappears
                                 # as spurious curvature in r(t): a constant-speed shock comes back
                                 # at -5 to -54 m/s^2 instead of zero.
FIT_MIN_DT_S    = 10             # s, minimum spacing of the points a lane fit uses
LANE_SIGMA_MHZ  = 'auto'         # 'auto' -> per-lane rms ridge offset; a number -> that value in
                                 # MHz; None -> unweighted. See METHODS.md section 2: 'auto' is an
                                 # assumed scale set by the search window, and every statistical
                                 # error bar is proportional to it.
LANE_SIGMA_FLOOR = 0.2           # MHz, floor on that uncertainty
FIT_CLIP_TO_TRACED_F = True      # blank fitted samples outside the traced frequency range
FIT_F_TOL_MHZ   = 0.05           # tolerance on that clip
LANE_FIT_MAX_RESID = 3           # flag a lane whose fit misses the points by this many sigma_f

# --- kinematics --------------------------------------------------------------------------------
KIN_DEG      = 2                 # degree of the r(t) fit speed and acceleration come from
KIN_MIN_PTS  = 6                 # minimum finite height points before v and a are computed
KIN_MIN_FRAC = 0.5               # of a lane's own traced span the density model must resolve
KIN_N_DENSE  = 200               # samples along a lane's own span used to fit r(t)
KIN_A_SIGMA  = 2                 # |a| must exceed this many times its own uncertainty to count
A_PLAUSIBLE_MS2 = 200            # |a| above this is worth questioning for a coronal shock

# --- height-time fits --------------------------------------------------------------------------
POLY_DEG  = 2                    # degree of the height-time fits. A cubic in r(t) bends into an S.
N_BOOT    = 300                  # bootstrap refits for the error bands
INFLATE_BY_CHI2 = False          # widening a band until it covers the data hides the comparison

# --- aggregation and grids ----------------------------------------------------------------------
N_MC     = 100                   # MC draws per pass for the fit error
N_GRID   = 60                    # points on the shared time grid
AGG_KEYS = ('r', 'v', 'a', 'vA', 'B', 'X', 'MA', 'ne', 'f')
MAKE_JOINT = False               # combine the two bands into one track as well
REF_LANE   = None                # lane the height-time fits and model sweep use; None -> the
                                 # longest upstream lane

# --- polarisation --------------------------------------------------------------------------------
POL_DT_S   = 2                   # +/- seconds of the box the V/I median is taken over
POL_DF_MHZ = 0.5                 # +/- MHz of that box

# --- quality control and audit ---------------------------------------------------------------------
QC_MIN_SNR_DB    = 3             # a trace sitting below this above background is not on the burst
QC_CONTROL_SHIFT = 40            # channels the control displaces each lane onto blank spectrum
CONSIST_SIGMA    = 3             # flag a disagreement beyond this many combined sigma

# --- export ------------------------------------------------------------------------------------
EXPORT_DT_S  = 10                # cadence of the exported height-time tracks [s]
# Which published B(r) laws to draw on the A.9 comparison. Only laws calibrated over a range that
# overlaps this burst (about 1.3-4 Rsun) are on by default. Gopalswamy & Yashiro (2011) is
# available but off: it was calibrated on CME-shock standoff distances over 6-23 Rsun, so drawing
# it here extrapolates it 1.5 to 4.7 times below its lower bound and it constrains nothing.
# Override per call: pl.plot_bfield(run, refs=('dulk_mclean', 'mann2023', 'gopalswamy_yashiro')).
B_REF_CURVES = ('dulk_mclean', 'mann2023')

CONTROLS_FILE = 'typeii_bezier_controls.csv'   # the Bezier control points a run was traced
                                               # from; load_controls replays them
EXPORT_N_MC  = 40                # MC draws per pass for the exported error column
R_LIMB_RSUN  = 1                 # heliocentric radius of the limb, for the above-limb convention
REPORT_EXCITER_ENERGY = False    # (gamma-1) m_e c^2 at the shock speed is ~1 eV and misleads

# --- set at runtime ------------------------------------------------------------------------------
# Filled in by the notebook once the data are loaded; listed here so every name the package uses
# has one home.
OUTDIR = None                    # output directory, set by setup_outputs()
