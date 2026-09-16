"""Coronal density models, the frequency-height inversion, and the shock relations.

Every density model returns n_e in cm^-3 for r in solar radii. Sources are given
per function; METHODS.md section 3 discusses the spread between them.
"""
import numpy as np
from scipy.optimize import brentq

from .config import C_MS, E_CHARGE_J, M_E, PLASMA_CONST, R_BOUNDS


def newkirk(r, fold=1):
    """Newkirk (1961) exponential model, n_e [cm^-3] at r [Rsun]."""
    r = np.asarray(r, float)
    return fold * 4.2e4 * 10 ** (4.32 / r)

def saito(r, fold=1):
    """Saito (1970) equatorial two-term model, n_e [cm^-3] at r [Rsun]."""
    r = np.asarray(r, float)
    return fold * (1.36e6 * r ** -2.14 + 1.68e8 * r ** -6.13)

def leblanc(r, fold=1):
    """Leblanc, Dulk & Bougeret (1998), n_e [cm^-3] at r [Rsun].

    Normalised to 7.2 cm^-3 at 1 AU and published for r > 1.8 Rsun. Below that it is
    an extrapolation; at fold 1 it cannot place a fundamental above ~82 MHz without
    pushing the source onto the photosphere.
    """
    r = np.asarray(r, float)
    return fold * (3.3e5 * r ** -2 + 4.1e6 * r ** -4 + 8e7 * r ** -6)

def baumbach_allen(r, fold=1):
    """Baumbach (1937) with the Allen (1947) correction, n_e [cm^-3] at r [Rsun]."""
    r = np.asarray(r, float)
    return fold * 1e8 * (0.036 * r ** -1.5 + 1.55 * r ** -6 + 2.99 * r ** -16)

def mann2023(r, fold=1):
    """Mann et al. (2023), A&A 679, A64, Eq. 11, n_e [cm^-3] at r [Rsun].

    Eq. 11 is printed symbolically, so the exponent comes from their Table 5. Point M
    gives r_c = 3877 Mm, hence 2 r_c / Rsun = 11.146, which reproduces their quoted
    n_e(3 Rsun) = 4.267e5 cm^-3. Point D gives 11.34 with a different base density;
    the two sets are not interchangeable. Valid to about 3 Rsun.
    """
    r = np.asarray(r, float)
    return fold * 7.17e8 * np.exp(11.14 * (1 / r - 1))

def freq_to_density(f_hz, harmonic=1):
    """Electron density [cm^-3] from an emission frequency [Hz] at harmonic s."""
    return (np.asarray(f_hz, float) / harmonic / PLASMA_CONST) ** 2

def freq_to_radius(f_hz, model, harmonic=1, r_bounds=R_BOUNDS, nres=6000):
    """Heliocentric radius [Rsun] where model(r) matches the density implied by f.

    Returns NaN outside the model range rather than extrapolating.
    """
    rr = np.linspace(r_bounds[0], r_bounds[1], nres)
    ne = model(rr)
    ne_t = freq_to_density(np.atleast_1d(f_hz), harmonic=harmonic)
    r = np.interp(ne_t, ne[::-1], rr[::-1])
    r[(ne_t > np.nanmax(ne)) | (ne_t < np.nanmin(ne))] = np.nan
    return r if r.size > 1 else float(r[0])

def invert_grid(model, r_bounds=R_BOUNDS, nres=6000):
    """Lookup table (r, n_e) for inverting a density model.

    Returns two arrays over R_BOUNDS; n_e is strictly decreasing, so callers
    interpolate on the reversed pair.
    """
    rr = np.linspace(r_bounds[0], r_bounds[1], nres)
    return rr, model(rr)

def alfven_mach_from_X(X, gamma=5/3, beta=0):
    """Alfven Mach number from the density jump X = (f_U/f_L)^2.

    Perpendicular shock, gamma = 5/3, beta = 0:
        M_A = sqrt(X(X + 5) / (2(4 - X))).
    NaN outside 1 <= X < 4. Other gamma or beta raise, rather than being
    silently ignored.
    """
    if abs(gamma - 5 / 3) > 1e-12:
        raise ValueError(f'gamma = {gamma} is not supported: the closed form here is the '
                         'gamma = 5/3 case. Re-derive the coefficients for another gamma.')
    if beta != 0:
        raise ValueError(f'beta = {beta} is not supported: this is the cold-upstream form. '
                         'Finite beta lowers M_A; see the caveats in A.12.')
    X = np.asarray(X, float)
    out = np.full(X.shape, np.nan)
    ok = (X >= 1) & (X < 4)
    out[ok] = np.sqrt(X[ok] * (X[ok] + 5) / (2 * (4 - X[ok])))
    return out

def electron_energy_from_speed(v_kms):
    """Kinetic energy [keV] of an electron moving at v [km/s].

    The energy of an electron travelling with the shock, not that
    of the electrons producing the emission.
    """
    v = np.asarray(v_kms, float) * 1e3
    beta = np.clip(v / C_MS, 0, 0.999999)
    gamma = 1 / np.sqrt(1 - beta ** 2)
    return (gamma - 1) * M_E * C_MS ** 2 / E_CHARGE_J / 1e3

def B_dulk_mclean(r):
    """Dulk & McLean (1978), B = 0.5 (r - 1)^-1.5 [G]. Valid 1.02 <= r <= 10 Rsun."""
    r = np.asarray(r, float)
    return 0.5 * (r - 1) ** -1.5

def B_gopalswamy_yashiro(r):
    """Gopalswamy & Yashiro (2011), ApJL 736, L17: B = 0.409 r^-1.30 [G].

    From the standoff distance between a CME-driven shock and its flux rope in white-light
    coronagraph images over 6-23 Rsun, so an independent technique from band splitting. They
    also quote 0.377 r^-1.25 from the same data reduced with the Saito rather than the
    Leblanc density model. Extrapolated well below its range at 1-3 Rsun.
    """
    r = np.asarray(r, float)
    return 0.409 * r ** -1.30

def B_mann2023(r):
    """Mann et al. (2023) Eq. 8, B_r = 6 r^-3 + 1.18 r^-2 [G]."""
    r = np.asarray(r, float)
    return 6 * r ** -3 + 1.18 * r ** -2
