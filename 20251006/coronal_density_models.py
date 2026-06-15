#!/usr/bin/env python
# coding: utf-8

# # Electron-density models for the solar corona and heliosphere
# 
# This notebook collects the electron-density models that are commonly used when
# interpreting solar radio bursts observed from metric to kilometric wavelengths.
# Each model is implemented as a single Python function that takes a heliocentric
# radial distance $r/R_\odot$ and returns the electron number density $N_e$ in
# $\mathrm{cm^{-3}}$. A companion routine inverts a model to convert an observed
# emission frequency to a height, which is what one actually does when reading a
# radio dynamic spectrum.
# 
# The notebook then works two examples:
# 
# * a **type II burst** with band-splitting, from which we estimate the shock
#   speed, the density jump across the shock, and the Alfvén Mach number;
# * a **type III burst**, from which we estimate the exciter (electron-beam)
#   speed as a fraction of $c$.
# 
# Both examples are run for the fundamental ($f_p$) and the harmonic ($2 f_p$)
# plasma-emission assumptions.
# 
# The relation between the local electron density and the plasma frequency is
# 
# $$
# f_p \;[\mathrm{Hz}] \;=\; \frac{1}{2\pi}\sqrt{\frac{N_e e^{2}}{\varepsilon_0 m_e}}
# \;\approx\; 8.98\times10^{3}\,\sqrt{N_e\,[\mathrm{cm^{-3}}]},
# $$
# 
# so $N_e\,[\mathrm{cm^{-3}}] \approx (f_p\,[\mathrm{Hz}]/8978.7)^{2}$. For
# fundamental emission $f_\text{obs}=f_p$; for harmonic emission
# $f_\text{obs}=2 f_p$, so the inferred density is a factor of four smaller for
# the same observed frequency.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.constants import e, m_e, epsilon_0, m_p, mu_0, c

R_SUN_M = 6.957e8          # solar radius in metres
AU_RS   = 215.032          # 1 AU in solar radii
PLASMA_CONST = 8.9787e3    # f_p [Hz] = PLASMA_CONST * sqrt(N_e [cm^-3])

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ## Plasma-frequency conversions
# 
# `freq_to_density` converts an observed radio frequency to the local electron
# density for either the fundamental or the harmonic plasma-emission assumption.
# `density_to_freq` is the inverse.

# In[ ]:


def freq_to_density(f_obs_hz, harmonic=1):
    """Convert observed radio frequency to local electron density.

    Parameters
    ----------
    f_obs_hz : float or array
        Observed emission frequency in Hz.
    harmonic : {1, 2}
        1 for fundamental plasma emission (f_obs = f_p),
        2 for harmonic plasma emission   (f_obs = 2 f_p).

    Returns
    -------
    N_e : float or array
        Electron number density in cm^-3.
    """
    f_obs = np.asarray(f_obs_hz, dtype=float)
    f_p = f_obs / harmonic
    return (f_p / PLASMA_CONST) ** 2


def density_to_freq(ne_cm3, harmonic=1):
    """Convert electron density (cm^-3) to observed emission frequency (Hz)."""
    ne = np.asarray(ne_cm3, dtype=float)
    f_p = PLASMA_CONST * np.sqrt(ne)
    return harmonic * f_p


# ## Electron-density models
# 
# Each function below takes the heliocentric distance `r` in units of $R_\odot$
# (so $r=1$ is the photosphere) and returns $N_e$ in $\mathrm{cm^{-3}}$. The
# docstring records the original reference and the coronal conditions for which
# the model was derived. Where the original paper uses $R = r/R_\odot$ I keep
# that convention.
# 
# The models cover several regimes:
# 
# | Model | Region of validity | Conditions |
# |---|---|---|
# | Baumbach–Allen (1947) | $1 \lesssim r/R_\odot \lesssim 3$ | quiet equatorial corona |
# | Newkirk (1961) | $1 \lesssim r/R_\odot \lesssim 4$ | quiet ($\times1$), streamer ($\times2$), active region ($\times4$) |
# | Saito (1977) | $1.03 \lesssim r/R_\odot \lesssim 5$ | equatorial / polar |
# | Saito, Poland & Munro (1977), full | $r/R_\odot \lesssim 5$ | streamer belt |
# | Sittler & Guhathakurta (1999) | $1.05 \lesssim r/R_\odot \lesssim 5$ | helmet streamer / coronal hole |
# | Leblanc, Dulk & Bougeret (1998) | $r/R_\odot \lesssim 215$ | corona out to 1 AU |
# | Parker (isothermal) | $r/R_\odot \lesssim 215$ | analytic isothermal wind |
# | Mann et al. (1999) | $r/R_\odot \lesssim 250$ | Parker-equation special solution |
# | Mann et al. (2023) | $r/R_\odot \lesssim 250$ | updated with PSP, quiet equatorial |
# 

# In[ ]:


def newkirk(r, fold=1):
    """Newkirk (1961) coronal electron density model.

    N_e(r) = fold * 4.2e4 * 10^(4.32 / R)   [cm^-3],   R = r / R_sun.

    Parameters
    ----------
    r : float or array
        Heliocentric distance in solar radii.
    fold : {1, 2, 4}
        Density multiplier: 1 = quiet corona, 2 = streamer, 4 = active region.

    Reference
    ---------
    Newkirk Jr., G. 1961, ApJ, 133, 983.
    Validity: 1 <= r/R_sun <= ~4.
    """
    r = np.asarray(r, dtype=float)
    return fold * 4.2e4 * 10.0 ** (4.32 / r)


def baumbach_allen(r):
    """Baumbach-Allen quiet equatorial corona.

    N_e(r) = 1e8 * (2.99 / R^16 + 1.55 / R^6 + 0.036 / R^1.5)   [cm^-3].

    Reference
    ---------
    Allen, C. W. 1947, MNRAS, 107, 426 (formulation of Baumbach 1937).
    Validity: 1 <= r/R_sun <= ~3, quiet equatorial corona.
    """
    r = np.asarray(r, dtype=float)
    return 1.0e8 * (2.99 / r ** 16 + 1.55 / r ** 6 + 0.036 / r ** 1.5)


def saito(r, region="equatorial"):
    """Saito (1970) two-term radial density model.

    N_e(r) = a1 * R^-b1 + a2 * R^-b2   [cm^-3].

    Parameters
    ----------
    region : {"equatorial", "polar"}
        Equatorial uses (a1, b1, a2, b2) = (1.36e6, 2.14, 1.68e8, 6.13).
        Polar uses (a1, b1, a2, b2) = (2.5e5, 2.5, 1.0e8, 6.0), which is the
        widely used high-latitude fit (e.g. cited in Cairns et al. 2009 and
        used by Vrsnak et al. 2004 for coronal-hole conditions).

    Reference
    ---------
    Saito, K. 1970, Ann. Tokyo Astron. Obs., 12, 53.
    Validity: 1.03 <= r/R_sun <= ~5.
    """
    r = np.asarray(r, dtype=float)
    if region == "equatorial":
        a1, b1, a2, b2 = 1.36e6, 2.14, 1.68e8, 6.13
    elif region == "polar":
        a1, b1, a2, b2 = 2.5e5, 2.5, 1.0e8, 6.0
    else:
        raise ValueError("region must be 'equatorial' or 'polar'")
    return a1 * r ** (-b1) + a2 * r ** (-b2)


def saito_poland_munro(r):
    """Saito, Poland & Munro (1977) streamer model.

    N_e(r) = 3.09e8 / R^16 + 1.58e8 / R^6 + 0.0251e8 / R^2.5   [cm^-3].

    Reference
    ---------
    Saito, K., Poland, A. I., Munro, R. H. 1977, Sol. Phys., 55, 121.
    Validity: streamer belt, 1 < r/R_sun < ~5.
    """
    r = np.asarray(r, dtype=float)
    return 3.09e8 / r ** 16 + 1.58e8 / r ** 6 + 0.0251e8 / r ** 2.5


def sittler_guhathakurta(r):
    """Sittler & Guhathakurta (1999) coronal helmet-streamer density model.

    Semi-empirical streamer model constrained by Skylab/SOHO white-light data:

        N_e(r) = N0 * R^-2 * exp(a/R) * (1 + b/R + c/R^2 + d/R^3),

    with N0 = 1.36e6 cm^-3, a = 1.09, b = 8.45, c = -28.59, d = 41.96
    (their published fit to the helmet-streamer belt).

    Reference
    ---------
    Sittler, E. C., Guhathakurta, M. 1999, ApJ, 523, 812.
    Validity: 1.05 <= r/R_sun <= ~5, helmet streamer / equatorial belt.
    """
    r = np.asarray(r, dtype=float)
    N0, a, b, c, d = 1.36e6, 1.09, 8.45, -28.59, 41.96
    R = r
    return N0 / R ** 2 * np.exp(a / R) * (1.0 + b / R + c / R ** 2 + d / R ** 3)


def leblanc(r):
    """Leblanc, Dulk & Bougeret (1998) corona-to-1-AU density model.

    N_e(r) = 3.3e5 / R^2 + 4.1e6 / R^4 + 8.0e7 / R^6   [cm^-3].

    Calibrated against Wind/WAVES type III bursts and gives ~7.2 cm^-3 at 1 AU.

    Reference
    ---------
    Leblanc, Y., Dulk, G. A., Bougeret, J.-L. 1998, Sol. Phys., 183, 165.
    Validity: r/R_sun up to 215 (= 1 AU).
    """
    r = np.asarray(r, dtype=float)
    return 3.3e5 / r ** 2 + 4.1e6 / r ** 4 + 8.0e7 / r ** 6


def parker_isothermal(r, T=1.4e6, ne_base=1.0e8):
    """Isothermal Parker wind density (analytic near-Sun limit).

    From Parker's wind equation with an isothermal corona at temperature T,
    in the subsonic near-Sun limit one has

        N_e(r) = N_e(R_sun) * exp( (2 r_c / R_sun) * (R_sun/r - 1) )

    with the critical radius r_c = G M_sun / (2 v_c^2),
    v_c = sqrt(k_B T / (mu m_p)), mu = 0.6.

    Parameters
    ----------
    T : float
        Coronal temperature in K (default 1.4 MK).
    ne_base : float
        Electron density at r = R_sun in cm^-3 (default 1e8).

    Reference
    ---------
    Parker, E. N. 1958, ApJ, 128, 664; analytic limit as in Mann et al. 1999.
    Validity: subsonic region, r/R_sun <= ~10 with the analytic form;
              breaks down near and beyond the critical point.
    """
    from scipy.constants import G, k as kB
    M_SUN = 1.989e30
    mu_mean = 0.6
    v_c = np.sqrt(kB * T / (mu_mean * m_p))
    r_c_m = G * M_SUN / (2.0 * v_c ** 2)
    r_c_Rs = r_c_m / R_SUN_M
    r = np.asarray(r, dtype=float)
    return ne_base * np.exp(2.0 * r_c_Rs * (1.0 / r - 1.0))


def mann1999(r, T=1.3e6, ne_base=5.14e8):
    """Mann et al. (1999) full Parker-wind special solution.

    Solves Parker's equation
        v'^2 - ln(v'^2) = 4 ln(r') + 4/r' - 3
    and applies continuity r^2 N v = const, yielding N_e(r) in cm^-3.
    The branch is chosen automatically: subsonic for r < r_c, supersonic
    for r > r_c.

    Parameters
    ----------
    T : float
        Isothermal coronal temperature in K. Default 1.3 MK reproduces
        ~5 cm^-3 at 1 AU as in Mann et al. (1999).
    ne_base : float
        Electron density at r = R_sun in cm^-3 (default 5.14e8 to match
        the calibration used in the original paper).

    Reference
    ---------
    Mann, G., Jansen, F., MacDowall, R. J., Kaiser, M. L., Stone, R. G.
    1999, A&A, 348, 614.
    Validity: 1 <= r/R_sun <= ~250 (out to ~5 AU in the original paper).
    """
    from scipy.constants import G, k as kB
    M_SUN = 1.989e30
    mu_mean = 0.6
    v_c = np.sqrt(kB * T / (mu_mean * m_p))
    r_c_m = G * M_SUN / (2.0 * v_c ** 2)
    r_c_Rs = r_c_m / R_SUN_M

    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    ne_out = np.empty_like(r_arr)

    # The Parker equation has two branches separated by the critical point.
    # We solve for v' on the physical (accelerating) branch.
    def parker_residual(v_prime, r_prime):
        # avoid log of zero
        v2 = max(v_prime ** 2, 1e-30)
        return v2 - np.log(v2) - 4.0 * np.log(r_prime) - 4.0 / r_prime + 3.0

    for i, rr in enumerate(r_arr):
        r_prime = rr / r_c_Rs
        if r_prime < 1.0:
            # subsonic branch: v' in (0, 1)
            v_prime = brentq(parker_residual, 1e-8, 1.0 - 1e-8, args=(r_prime,))
        elif r_prime > 1.0:
            # supersonic branch: v' in (1, large)
            v_prime = brentq(parker_residual, 1.0 + 1e-8, 50.0, args=(r_prime,))
        else:
            v_prime = 1.0
        # Continuity: at r = R_sun the velocity is v_R; using continuity
        # r^2 N v = R_sun^2 N_base v_base. We get v_base from the same solver.
        ne_out[i] = v_prime  # store v' temporarily; convert below

    # compute v' at the base
    r_prime_base = 1.0 / r_c_Rs  # since r = R_sun
    v_prime_base = brentq(parker_residual, 1e-8, 1.0 - 1e-8,
                          args=(r_prime_base,))
    # N(r) = N_base * (R_sun/r)^2 * v_base/v(r)
    ne_out = ne_base * (1.0 / r_arr) ** 2 * v_prime_base / ne_out

    return ne_out if ne_out.size > 1 else float(ne_out[0])


def mann2023(r):
    """Mann, Warmuth, Vocks & Rouillard (2023) heliospheric model (Table 8).

    Tabulated electron-density values from the published Table 8 (point M
    of the Ce-T diagram), interpolated log-linearly in r. The underlying
    physical model is Parker's wind equation calibrated against PSP, OMNI,
    HELIOS and coronal data; see the paper for details.

    Reference
    ---------
    Mann, G., Warmuth, A., Vocks, C., Rouillard, A. P. 2023, A&A, 679, A64.
    Validity: 1 <= r/R_sun <= 250, quiet equatorial conditions,
    long-term average global model.
    """
    # Mann et al. 2023, Table 8 (selected rows, r in R_sun, N_e in cm^-3)
    table_r = np.array([
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
        6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0,
        60.0, 70.0, 80.0, 100.0, 120.0, 160.0, 200.0, 215.0, 250.0,
    ])
    table_ne = np.array([
        7.17e8, 2.60e8, 1.12e8, 5.46e7, 2.96e7, 1.74e7, 1.09e7, 5.02e6,
        2.69e6, 8.58e5, 3.89e5, 1.33e5, 6.42e4, 3.70e4, 1.65e4, 9.10e3,
        5.70e3, 3.88e3, 2.80e3, 1.63e3, 9.66e2, 6.32e2, 3.28e2, 1.98e2,
        1.32e2, 9.37e1, 6.98e1, 4.28e1, 2.83e1, 1.54e1, 9.53, 8.16, 5.91,
    ])
    r = np.asarray(r, dtype=float)
    log_r = np.log10(np.clip(r, table_r.min(), table_r.max()))
    return 10.0 ** np.interp(log_r, np.log10(table_r), np.log10(table_ne))


# ## Inverting a density model: frequency $\to$ height
# 
# Given an observed frequency and a choice of density model, we want the radial
# distance $r$ at which the local plasma frequency matches. Most of the models
# above are monotonically decreasing in $r$, so a simple bracketed root search
# works. The helper returns `np.nan` if the requested density is outside the
# model's range.

# In[ ]:


def freq_to_radius(f_obs_hz, model, harmonic=1,
                   r_bounds=(1.001, 250.0), **model_kwargs):
    """Invert a density model to find r given an observed frequency.

    Parameters
    ----------
    f_obs_hz : float or array
        Observed emission frequency in Hz.
    model : callable
        Density model, signature model(r, **kwargs) -> N_e [cm^-3].
    harmonic : {1, 2}
        Plasma-emission harmonic (1 = fundamental, 2 = harmonic).
    r_bounds : (float, float)
        Search bracket in solar radii.
    model_kwargs : dict
        Extra arguments passed to the model.

    Returns
    -------
    r : float or array
        Heliocentric distance in solar radii. NaN if the density is
        outside the model's bracket.
    """
    f_obs = np.atleast_1d(np.asarray(f_obs_hz, dtype=float))
    ne_target = freq_to_density(f_obs, harmonic=harmonic)
    r_lo, r_hi = r_bounds

    def f_root(r, ne_t):
        return model(r, **model_kwargs) - ne_t

    r_out = np.empty_like(f_obs)
    for i, ne_t in enumerate(ne_target):
        ne_lo = model(r_lo, **model_kwargs)
        ne_hi = model(r_hi, **model_kwargs)
        if not (min(ne_lo, ne_hi) <= ne_t <= max(ne_lo, ne_hi)):
            r_out[i] = np.nan
            continue
        try:
            r_out[i] = brentq(f_root, r_lo, r_hi, args=(ne_t,))
        except ValueError:
            r_out[i] = np.nan
    return r_out if r_out.size > 1 else float(r_out[0])


# ## Sanity check
# 
# A quick verification at standard reference points:
# 
# * Newkirk quiet at $r = 1\,R_\odot$ should give $4.2\times10^{4}\times10^{4.32}\approx 8.8\times10^{8}\,\mathrm{cm^{-3}}$.
# * Leblanc at $r = 215\,R_\odot$ should give $\sim 7.2\,\mathrm{cm^{-3}}$ (1 AU value quoted in Leblanc et al. 1998).
# * Mann et al. (2023) at $r = 215\,R_\odot$ should give $\sim 8.2\,\mathrm{cm^{-3}}$ from their Table 8.
# * The plasma-frequency conversion should be self-consistent: $f \to N_e \to f$.
# 

# In[ ]:


# Reference-point check
print(f"Newkirk(1) quiet:           {newkirk(1.0, fold=1):.3e} cm^-3 "
      f"(expected ~8.8e8)")
print(f"Newkirk(1) streamer x2:     {newkirk(1.0, fold=2):.3e} cm^-3")
print(f"Newkirk(1) active region:   {newkirk(1.0, fold=4):.3e} cm^-3")
print(f"Leblanc at 1 AU (215 R_s):  {leblanc(215.0):.3f} cm^-3 "
      f"(expected ~7.2)")
print(f"Mann 2023 at 215 R_s:       {mann2023(215.0):.3f} cm^-3 "
      f"(expected ~8.2)")
print(f"Saito equatorial at 2 R_s:  {saito(2.0):.3e} cm^-3")
print(f"Baumbach-Allen at 1.5 R_s:  {baumbach_allen(1.5):.3e} cm^-3")
print()

# Round-trip check on the plasma-frequency relation
f_test = np.array([1e6, 30e6, 100e6, 300e6])   # 1, 30, 100, 300 MHz
ne_fund = freq_to_density(f_test, harmonic=1)
ne_harm = freq_to_density(f_test, harmonic=2)
print("Frequency  N_e (F)         N_e (H)         round-trip f (F)")
for f, nf, nh in zip(f_test, ne_fund, ne_harm):
    f_back = density_to_freq(nf, harmonic=1)
    print(f"{f/1e6:6.1f} MHz  {nf:.3e}  {nh:.3e}  {f_back/1e6:8.3f} MHz")


# ## Comparison plot: $N_e(r)$ for all models
# 
# The left panel covers the near-Sun corona ($1$–$5\,R_\odot$). The right panel
# extends out to $1\,\mathrm{AU} = 215\,R_\odot$, where only the wide-range
# models are physically meaningful. A secondary horizontal axis on the right
# panel shows the corresponding fundamental plasma frequency.

# In[ ]:


r_near = np.linspace(1.01, 5.0, 400)
r_far  = np.logspace(np.log10(1.5), np.log10(250.0), 600)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- near-Sun panel ---
ax = axes[0]
ax.semilogy(r_near, newkirk(r_near, 1), label="Newkirk quiet (x1)")
ax.semilogy(r_near, newkirk(r_near, 2), label="Newkirk streamer (x2)", ls="--")
ax.semilogy(r_near, newkirk(r_near, 4), label="Newkirk active (x4)", ls=":")
ax.semilogy(r_near, baumbach_allen(r_near), label="Baumbach-Allen")
ax.semilogy(r_near, saito(r_near, "equatorial"), label="Saito equatorial")
ax.semilogy(r_near, saito(r_near, "polar"), label="Saito polar", ls="--")
ax.semilogy(r_near, saito_poland_munro(r_near), label="Saito-Poland-Munro")
ax.semilogy(r_near, sittler_guhathakurta(r_near),
            label="Sittler-Guhathakurta streamer")
ax.semilogy(r_near, leblanc(r_near), label="Leblanc et al. 1998", color="k")
ax.semilogy(r_near, mann2023(r_near), label="Mann et al. 2023",
            color="firebrick", lw=2)
ax.set_xlabel(r"$r / R_\odot$")
ax.set_ylabel(r"$N_e\;\;[\mathrm{cm^{-3}}]$")
ax.set_title("Corona (1 - 5 $R_\\odot$)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(1.0, 5.0)

# --- wide panel ---
ax = axes[1]
ax.loglog(r_far, leblanc(r_far), label="Leblanc 1998", color="k")
ax.loglog(r_far, mann2023(r_far), label="Mann 2023", color="firebrick", lw=2)
ax.loglog(r_far, mann1999(r_far), label="Mann 1999",
          color="steelblue", ls="--")
ax.loglog(r_far, parker_isothermal(r_far), label="Parker isothermal",
          color="seagreen", ls=":")
ax.loglog(r_far, newkirk(r_far, 1), label="Newkirk (extrapolated)",
          color="gray", alpha=0.6, ls=":")
ax.set_xlabel(r"$r / R_\odot$")
ax.set_ylabel(r"$N_e\;\;[\mathrm{cm^{-3}}]$")
ax.set_title("Corona to 1 AU")
ax.axvline(AU_RS, color="gray", lw=0.7, alpha=0.6)
ax.text(AU_RS * 1.05, ax.get_ylim()[1] * 0.4, "1 AU",
        fontsize=9, color="gray")
ax.legend(fontsize=9, loc="upper right")

# Top axis: corresponding fundamental plasma frequency
def ne_to_fp(ne):
    return PLASMA_CONST * np.sqrt(np.clip(ne, 1e-3, None)) / 1e6  # MHz
def fp_to_ne(fp):
    return (fp * 1e6 / PLASMA_CONST) ** 2

secax = ax.secondary_yaxis("right", functions=(ne_to_fp, fp_to_ne))
secax.set_ylabel(r"$f_p\;\;[\mathrm{MHz}]$ (fundamental)")

plt.tight_layout()
plt.show()


# ## Example 1 — Type II radio burst with band-splitting
# 
# Type II bursts are produced by electrons accelerated at the front of a coronal
# or interplanetary shock. They drift slowly in frequency because the shock
# moves outward through a decreasing density gradient. They commonly show
# **band-splitting** into an upper-frequency (UF) and a lower-frequency (LF)
# lane; the standard interpretation is that the two lanes come from the
# downstream (compressed) and upstream regions of the shock, so
# 
# $$
# X \;\equiv\; \frac{N_e^{\rm down}}{N_e^{\rm up}}
# \;=\; \left(\frac{f_U}{f_L}\right)^{2}
# $$
# 
# is the density-jump (compression ratio) across the shock
# (Vrsnak et al. 2002; Mann et al. 1995). From the Rankine–Hugoniot relation
# for a perpendicular shock,
# 
# $$
# M_A^{2} \;=\; \frac{X(X+5)}{2(4-X)}\quad (\text{for } \gamma=5/3),
# $$
# 
# so the Alfvén Mach number follows directly from $X$ without needing any
# geometry assumption. The Alfvén speed $v_A$ at the inferred height comes from
# the Mann et al. (2023) Table 8 values.
# 
# ### Synthetic data
# 
# Two drifting lanes between roughly $80$ and $30\,\mathrm{MHz}$, observed over
# about three minutes — broadly consistent with a metric type II in the low
# corona.

# In[ ]:


# Fabricated (time, f_upper, f_lower) pairs for a type II burst, mid-corona.
# Times are seconds from the start of the burst; frequencies in MHz.
# The lanes drift from ~50 down to ~15 MHz in 5 minutes, placing the source
# in the 2 - 5 R_sun range where the Alfven speed from Mann 2023 is defined.
t_sec   = np.array([   0.0,  30.0,  60.0,  90.0, 120.0, 150.0, 180.0, 240.0, 300.0])
f_upper = np.array([  50.0,  43.0,  37.0,  32.0,  28.0,  25.0,  22.5,  18.5,  15.5])  # UF lane
f_lower = np.array([  43.0,  37.0,  32.0,  28.0,  24.5,  21.5,  19.5,  16.0,  13.5])  # LF lane

# Convert MHz to Hz
f_U = f_upper * 1e6
f_L = f_lower * 1e6

# Density jump across the shock from band-splitting (Vrsnak et al. 2002)
X = (f_U / f_L) ** 2

print(f"Band-splitting density jump X = N_down / N_up:")
for ti, fi_u, fi_l, xi in zip(t_sec, f_upper, f_lower, X):
    print(f"  t = {ti:6.1f} s   f_U = {fi_u:5.1f} MHz  "
          f"f_L = {fi_l:5.1f} MHz   X = {xi:5.3f}")


# In[ ]:


def shock_kinematics(t_sec, f_obs_hz, model, harmonic=1, poly_deg=2,
                     **model_kwargs):
    """Estimate r(t) and shock speed from a drifting type II lane.

    Fits a degree-`poly_deg` polynomial to r(t) and differentiates it
    analytically to get the instantaneous radial speed. This is more robust
    against endpoint distortion than a raw finite difference.

    Parameters
    ----------
    t_sec : array
        Times in seconds.
    f_obs_hz : array
        Observed frequencies in Hz (one frequency per time point).
    model : callable
        Density model.
    harmonic : {1, 2}
        Plasma-emission harmonic for the lane.
    poly_deg : int
        Polynomial degree for the r(t) fit (default 2).

    Returns
    -------
    r_Rs : array
        Heliocentric distance at each time, in solar radii.
    v_shock_kms : array
        Radial shock speed in km/s.
    """
    r = freq_to_radius(f_obs_hz, model, harmonic=harmonic, **model_kwargs)
    valid = np.isfinite(r)
    if valid.sum() <= poly_deg:
        # not enough points for the requested fit; fall back to gradient
        dr_m = np.gradient(r * R_SUN_M)
        dt_s = np.gradient(t_sec)
        return r, dr_m / dt_s / 1e3
    coeffs = np.polyfit(t_sec[valid], r[valid], poly_deg)
    dcoeffs = np.polyder(coeffs)
    v_Rs_per_s = np.polyval(dcoeffs, t_sec)
    v_kms = v_Rs_per_s * R_SUN_M / 1e3
    return r, v_kms


def alfven_mach_from_X(X, gamma=5.0 / 3.0):
    """Perpendicular-shock Alfven Mach number from density jump X.

    From the Rankine-Hugoniot jump conditions for a perpendicular MHD shock,
    eliminating the downstream variables gives, for gamma = 5/3:

        M_A^2 = X (X + 5) / (2 (4 - X)),

    valid for 1 <= X < 4 (X = 4 is the strong-shock limit for gamma = 5/3).
    """
    return np.sqrt(X * (X + 5.0) / (2.0 * (4.0 - X)))


def vA_mann2023(r_Rs):
    """Alfven speed (km/s) from Mann et al. 2023 Table 8 (mean / point M).

    Returns NaN for r outside the tabulated range [2, 250] R_sun, because
    the Mann (2023) model does not define an Alfven speed in the low corona
    (mixed open/closed field) or beyond 250 R_sun.
    """
    table_r = np.array([
        2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0,
        25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 160.0,
        200.0, 215.0, 250.0,
    ])
    table_vA = np.array([
        122.0, 169.0, 198.0, 219.0, 216.0, 206.0, 181.0, 159.0, 141.0, 127.0,
        115.0, 96.9, 81.2, 70.0, 55.3, 46.0, 39.6, 35.0, 31.4, 26.4, 23.0,
        19.0, 16.6, 15.9, 14.7,
    ])
    r_arr = np.atleast_1d(np.asarray(r_Rs, dtype=float))
    log_r = np.log10(np.where((r_arr >= table_r.min()) & (r_arr <= table_r.max()),
                              r_arr, np.nan))
    out = np.interp(log_r, np.log10(table_r), table_vA, left=np.nan, right=np.nan)
    return out if out.size > 1 else float(out[0])


# In[ ]:


# --- analyse the type II burst with several models and both harmonics ---
models_to_try = [
    ("Newkirk x2 (streamer)", lambda r: newkirk(r, fold=2)),
    ("Saito equatorial",      lambda r: saito(r, "equatorial")),
    ("Leblanc 1998",          leblanc),
    ("Mann 2023",             mann2023),
]

print(f"{'Model':<24}{'harm.':<7}{'<r> [R_s]':<12}"
      f"{'<v_sh> [km/s]':<16}{'<X>':<8}{'<M_A>':<8}{'<v_A> [km/s]':<14}")
print("-" * 90)

results = {}
for name, model in models_to_try:
    for harm in (1, 2):
        r_U, v_U = shock_kinematics(t_sec, f_U, model, harmonic=harm)
        r_L, v_L = shock_kinematics(t_sec, f_L, model, harmonic=harm)
        # use the upper-frequency (downstream) lane as the shock location
        r_mean = np.nanmean(r_U)
        v_mean = np.nanmean(v_U)
        X_mean = np.nanmean(X)
        MA_mean = alfven_mach_from_X(X_mean)
        vA_mean = vA_mann2023(r_mean)
        harm_label = "F" if harm == 1 else "H"
        print(f"{name:<24}{harm_label:<7}{r_mean:<12.2f}"
              f"{v_mean:<16.1f}{X_mean:<8.3f}{MA_mean:<8.2f}{vA_mean:<14.1f}")
        results[(name, harm)] = dict(r_U=r_U, r_L=r_L, v_U=v_U, v_L=v_L,
                                      X=X, MA=alfven_mach_from_X(X),
                                      vA=vA_mann2023(r_U))


# In[ ]:


# --- visualise the type II analysis (using Mann 2023, fundamental) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# (a) the dynamic-spectrum-like view
ax = axes[0, 0]
ax.plot(t_sec, f_upper, "o-", color="firebrick", label="upper lane (UF)")
ax.plot(t_sec, f_lower, "s-", color="steelblue",  label="lower lane (LF)")
ax.set_xlabel("time [s]")
ax.set_ylabel("frequency [MHz]")
ax.set_title("Synthetic type II band-split lanes")
ax.invert_yaxis()
ax.legend()

# (b) height-time using several models, fundamental emission
ax = axes[0, 1]
for name, model in models_to_try:
    r = freq_to_radius(f_U, model, harmonic=1)
    ax.plot(t_sec, r, "o-", label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$r / R_\odot$ (from UF lane, fundamental)")
ax.set_title("Shock height vs time")
ax.legend(fontsize=8)

# (c) shock speed
ax = axes[1, 0]
for name, model in models_to_try:
    _, v = shock_kinematics(t_sec, f_U, model, harmonic=1)
    ax.plot(t_sec, v, "o-", label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$v_{\rm shock}$ [km/s]")
ax.set_title("Radial shock speed (fundamental)")
ax.legend(fontsize=8)

# (d) Alfven Mach number vs height (using Mann 2023)
ax = axes[1, 1]
r_for_plot = freq_to_radius(f_U, mann2023, harmonic=1)
MA = alfven_mach_from_X(X)
ax.plot(r_for_plot, MA, "o-", color="purple")
ax.set_xlabel(r"$r / R_\odot$")
ax.set_ylabel(r"$M_A$")
ax.set_title("Alfven Mach number from band-splitting")
ax.axhline(1.0, color="gray", lw=0.7, ls="--")

plt.tight_layout()
plt.show()


# ## Example 2 — Type III radio burst, electron-beam speed
# 
# Type III bursts are produced by sub-relativistic electron beams streaming
# outward along open magnetic field lines. The frequency drift is fast and is
# set by the radial speed of the beam through the ambient density gradient:
# 
# $$
# v_\text{beam} \;=\; \frac{\mathrm{d}r}{\mathrm{d}t}
# \;=\; \frac{\mathrm{d}r}{\mathrm{d}f}\,\frac{\mathrm{d}f}{\mathrm{d}t}.
# $$
# 
# There is no shock, so we report the beam speed as a fraction of $c$ instead
# of computing Mach numbers.

# In[ ]:


# Fabricated (time, frequency) pairs for a type III burst:
# fast drift from ~250 MHz down to ~30 MHz in a few seconds.
t3 = np.array([0.0, 0.4, 0.8, 1.2, 1.8, 2.5, 3.4, 4.6, 6.0])
f3_MHz = np.array([250.0, 200.0, 160.0, 130.0, 100.0, 75.0, 55.0, 40.0, 30.0])
f3 = f3_MHz * 1e6


# In[ ]:


def beam_kinematics(t_sec, f_obs_hz, model, harmonic=1, poly_deg=2,
                    **model_kwargs):
    """Estimate r(t) and beam speed (in km/s and units of c) for a type III.

    Fits r(t) with a polynomial of degree `poly_deg` and differentiates it.
    """
    r = freq_to_radius(f_obs_hz, model, harmonic=harmonic, **model_kwargs)
    valid = np.isfinite(r)
    if valid.sum() <= poly_deg:
        dr_m = np.gradient(r * R_SUN_M)
        dt_s = np.gradient(t_sec)
        v_kms = dr_m / dt_s / 1e3
    else:
        coeffs = np.polyfit(t_sec[valid], r[valid], poly_deg)
        dcoeffs = np.polyder(coeffs)
        v_kms = np.polyval(dcoeffs, t_sec) * R_SUN_M / 1e3
    v_over_c = v_kms * 1e3 / c
    return r, v_kms, v_over_c


print(f"{'Model':<24}{'harm.':<7}{'<v_beam> [km/s]':<18}"
      f"{'<v_beam/c>':<14}{'r range [R_s]':<18}")
print("-" * 80)
type3_results = {}
for name, model in models_to_try:
    for harm in (1, 2):
        r3, v3_kms, beta3 = beam_kinematics(t3, f3, model, harmonic=harm)
        harm_label = "F" if harm == 1 else "H"
        print(f"{name:<24}{harm_label:<7}{np.nanmean(v3_kms):<18.0f}"
              f"{np.nanmean(beta3):<14.3f}"
              f"{np.nanmin(r3):.2f} - {np.nanmax(r3):.2f}")
        type3_results[(name, harm)] = dict(r=r3, v_kms=v3_kms, beta=beta3)


# In[ ]:


# --- visualise the type III analysis ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) dynamic spectrum
ax = axes[0]
ax.plot(t3, f3_MHz, "o-", color="darkorange")
ax.set_xlabel("time [s]")
ax.set_ylabel("frequency [MHz]")
ax.set_title("Synthetic type III drift")
ax.set_yscale("log")
ax.invert_yaxis()

# (b) height-time, several models, fundamental
ax = axes[1]
for name, model in models_to_try:
    r, _, _ = beam_kinematics(t3, f3, model, harmonic=1)
    ax.plot(t3, r, "o-", label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$r / R_\odot$ (fundamental)")
ax.set_title("Beam height vs time")
ax.legend(fontsize=8)

# (c) beam speed as v/c
ax = axes[2]
for name, model in models_to_try:
    _, _, beta = beam_kinematics(t3, f3, model, harmonic=1)
    ax.plot(t3, beta, "o-", label=name)
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$v_{\rm beam} / c$")
ax.set_title("Electron-beam speed (fundamental)")
ax.legend(fontsize=8)
ax.axhline(0.3, color="gray", lw=0.7, ls="--")
ax.text(t3[-1] * 0.6, 0.31, "typical type III: 0.1 - 0.5 c",
        color="gray", fontsize=8)

plt.tight_layout()
plt.show()


# ## Notes on usage
# 
# * The density-model functions are intentionally written so they can each take a
#   scalar or a NumPy array. The inversion `freq_to_radius` works element-wise.
# * The Newkirk model strictly applies to the low corona; the extrapolation to
#   large $r$ in the wide-range comparison plot is shown only to illustrate that
#   it grossly underestimates the density beyond a few $R_\odot$. Use Leblanc,
#   Mann 1999 or Mann 2023 once you are above the inner corona.
# * Band-splitting interpretation as a density jump assumes the upper and lower
#   lanes really come from the downstream and upstream sides of the shock;
#   alternative interpretations exist (e.g. emission from different parts of an
#   inhomogeneous shock front). Cross-check with imaging when possible.
# * The Mann (2023) Alfvén-speed values used here are the mean ("point M")
#   case from their Table 8. The minimum / maximum cases bracket the heliosphere
#   to within roughly a factor of two; you can replicate that by interpolating
#   the corresponding columns from the CDS table.
# 
# ### References
# 
# * Allen, C. W. 1947, MNRAS, 107, 426.
# * Leblanc, Y., Dulk, G. A., Bougeret, J.-L. 1998, Sol. Phys., 183, 165.
# * Mann, G., Jansen, F., MacDowall, R. J., Kaiser, M. L., Stone, R. G. 1999, A&A, 348, 614.
# * Mann, G., Warmuth, A., Vocks, C., Rouillard, A. P. 2023, A&A, 679, A64.
# * Newkirk, G., Jr. 1961, ApJ, 133, 983.
# * Parker, E. N. 1958, ApJ, 128, 664.
# * Saito, K. 1970, Ann. Tokyo Astron. Obs., 12, 53.
# * Saito, K., Poland, A. I., Munro, R. H. 1977, Sol. Phys., 55, 121.
# * Sittler, E. C., Guhathakurta, M. 1999, ApJ, 523, 812.
# * Vrsnak, B., Magdalenic, J., Aurass, H., Mann, G. 2002, A&A, 396, 673.
# 

# In[ ]:




