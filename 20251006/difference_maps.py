import os
import numpy as np
import sunpy
from tqdm import tqdm
from matplotlib import colors

# =============================================================================
# The prepare functions (base_difference, running_difference, running_ratio)
# do NOT set any intensity normalisation. They return the maps unscaled so the
# vmin/vmax can be chosen externally. Use apply_norm() afterwards to fix one
# shared scaling across the sequence (this is what avoids movie flickering).
# =============================================================================


def _export_sequence(seq, save_dir, prefix):
    """
    Write each map of a sequence to a FITS file in a dedicated folder.

    Parameters
    ----------
    seq : iterable of sunpy.map.GenericMap
        Maps to save.
    save_dir : str
        Target directory. Created if it does not exist.
    prefix : str
        Filename prefix, e.g. ``'basediff'``. Files are named
        ``<prefix>_<index>_<obstime>.fits``.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, m in enumerate(seq):
        try:
            stamp = m.date.to_datetime().strftime('%Y%m%dT%H%M%S')
        except Exception:
            # fall back to the running index if the date cannot be parsed
            stamp = f'{i:04d}'
        fname = os.path.join(save_dir, f'{prefix}_{i:04d}_{stamp}.fits')
        m.save(fname, overwrite=True)


def apply_norm(seq, vmin=None, vmax=None, norm=None, cmap='Greys_r'):
    """
    Apply one shared display normalisation to every frame of a sequence.

    Applying the *same* norm to all frames is what keeps the colour scaling
    fixed and stops a movie from flickering. The map data is not modified;
    only ``plot_settings`` are updated.

    Parameters
    ----------
    seq : iterable of sunpy.map.GenericMap (e.g. a MapSequence)
        Maps to scale.
    vmin, vmax : float, optional
        Limits for a linear ``matplotlib.colors.Normalize``. Ignored if
        ``norm`` is given.
    norm : matplotlib.colors.Normalize or subclass, optional
        An explicit norm instance to apply to every frame, e.g.
        ``colors.SymLogNorm(...)`` or ``colors.TwoSlopeNorm(vcenter=0, ...)``.
        Takes precedence over ``vmin``/``vmax``.
    cmap : str or None, optional
        Colour map to apply. Default ``'Greys_r'``. Pass ``None`` to leave the
        existing colour map untouched.

    Returns
    -------
    The same sequence, with ``plot_settings`` updated in place.
    """
    if norm is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for m in seq:
        m.plot_settings['norm'] = norm
        if cmap is not None:
            m.plot_settings['cmap'] = cmap
    return seq


def base_difference(maps, ref_index=0, cmap='Greys_r', save_dir=None):
    """
    Compute base-difference images for a list of SunPy maps.

    Each output frame is ``maps[i] - maps[ref_index]``, i.e. every image
    differenced against a single fixed reference frame. Arithmetic is
    unit-aware (the reference is subtracted as a ``Quantity``), so the result
    keeps the physical units and inherits the metadata of ``maps[i]``.

    No intensity normalisation is applied; set it afterwards with
    ``apply_norm`` so you control ``vmin``/``vmax``.

    Parameters
    ----------
    maps : list of sunpy.map.GenericMap
        Input image sequence, assumed time-ordered.
    ref_index : int, optional
        Index of the reference (base) frame. Default 0.
    cmap : str, optional
        Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
    save_dir : str or None, optional
        If given, every output map is written as a FITS file into this folder.

    Returns
    -------
    sunpy.map.MapSequence
        Base-difference maps, one per input frame (the reference frame
        differences to zero).
    """
    ref = maps[ref_index]
    diffs = []
    with tqdm(total=len(maps), desc='Computing base difference ...') as pbar:
        for m in maps:
            d = m - ref.quantity
            d.data[np.isnan(d.data)] = 0
            d.plot_settings['cmap'] = cmap
            diffs.append(d)
            pbar.update(1)

    if save_dir is not None:
        _export_sequence(diffs, save_dir, prefix='basediff')

    return sunpy.map.Map(diffs, sequence=True)


def running_difference(maps, step=1, cmap='Greys_r', save_dir=None):
    """
    Compute running-difference images for a list of SunPy maps.

    Each output frame is ``maps[i] - maps[i - step]``. With ``step=1`` this is
    the consecutive-frame difference; larger ``step`` differences against an
    earlier frame, useful for slow features or sparse cadence. Arithmetic is
    unit-aware and the result inherits the metadata of ``maps[i]``.

    No intensity normalisation is applied; set it afterwards with
    ``apply_norm`` so you control ``vmin``/``vmax``.

    Parameters
    ----------
    maps : list of sunpy.map.GenericMap
        Input image sequence, assumed time-ordered.
    step : int, optional
        Frame separation between the current and reference image. Default 1.
    cmap : str, optional
        Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
    save_dir : str or None, optional
        If given, every output map is written as a FITS file into this folder.

    Returns
    -------
    sunpy.map.MapSequence
        Running-difference maps (length ``len(maps) - step``).
    """
    diffs = []
    with tqdm(total=len(maps) - step, desc='Computing running difference ...') as pbar:
        for i in range(step, len(maps)):
            d = maps[i] - maps[i - step].quantity
            d.data[np.isnan(d.data)] = 0
            d.plot_settings['cmap'] = cmap
            diffs.append(d)
            pbar.update(1)

    if save_dir is not None:
        _export_sequence(diffs, save_dir, prefix='rundiff')

    return sunpy.map.Map(diffs, sequence=True)


def running_ratio(maps, step=1, cmap='Greys_r', save_dir=None):
    """
    Compute running-ratio images for a list of SunPy maps.

    Each output frame is ``maps[i] / maps[i - step]``. Pixels with no change
    sit at 1; brightenings exceed 1 and dimmings fall below it. Division by a
    ``Quantity`` yields a dimensionless map that inherits the metadata of
    ``maps[i]``. Non-finite pixels (from zeros in the reference frame) are set
    to 1, i.e. treated as "no change".

    No intensity normalisation is applied; set it afterwards with
    ``apply_norm`` so you control ``vmin``/``vmax``.

    Parameters
    ----------
    maps : list of sunpy.map.GenericMap
        Input image sequence, assumed time-ordered.
    step : int, optional
        Frame separation between the current and reference image. Default 1.
    cmap : str, optional
        Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
    save_dir : str or None, optional
        If given, every output map is written as a FITS file into this folder.

    Returns
    -------
    sunpy.map.MapSequence
        Running-ratio maps (length ``len(maps) - step``).
    """
    ratios = []
    with tqdm(total=len(maps) - step, desc='Computing running ratio ...') as pbar:
        for i in range(step, len(maps)):
            r = maps[i] / maps[i - step].quantity
            r.data[~np.isfinite(r.data)] = 1
            r.plot_settings['cmap'] = cmap
            ratios.append(r)
            pbar.update(1)

    if save_dir is not None:
        _export_sequence(ratios, save_dir, prefix='runratio')

    return sunpy.map.Map(ratios, sequence=True)







# # def _global_powernorm(seq, gamma=0.2, lower_percentile=1, upper_percentile=99):
# #     """
# #     Build a single shared ``PowerNorm`` for an entire map sequence.

# #     Using one fixed normalisation for every frame is what prevents the
# #     brightness flickering you get when each frame is auto-scaled
# #     independently. ``vmin``/``vmax`` are taken from percentiles computed over
# #     the pooled data of the whole stack, so the colour mapping is identical
# #     across frames.

# #     Parameters
# #     ----------
# #     seq : list of sunpy.map.GenericMap
# #         The output maps to be scaled (differences or ratios).
# #     gamma : float, optional
# #         Power-law exponent for ``matplotlib.colors.PowerNorm``. Default 0.2.
# #     lower_percentile, upper_percentile : float, optional
# #         Percentiles of the pooled stack used for ``vmin``/``vmax``.

# #     Returns
# #     -------
# #     matplotlib.colors.PowerNorm
# #         A norm instance with globally fixed limits.

# #     Notes
# #     -----
# #     ``PowerNorm`` clips values below ``vmin`` and applies a one-sided power
# #     law. For signed difference images this means negative excursions
# #     (dimming) are pushed to the bottom of the colour bar. The underlying map
# #     data is left untouched, so a diverging norm can be substituted afterwards
# #     via ``m.plot_settings['norm']`` if both signs need to be shown.
# #     """
# #     pooled = np.concatenate([np.asarray(m.data, dtype=float).ravel() for m in seq])
# #     pooled = pooled[np.isfinite(pooled)]
# #     vmin = np.percentile(pooled, lower_percentile)
# #     vmax = np.percentile(pooled, upper_percentile)
# #     return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


# def _global_diffnorm(seq, clip='mad', k=3.0, percentile=99.5,
#                      scaling='linear', linthresh_frac=0.05):
#     """
#     Single shared, zero-centred symmetric norm for a difference sequence.

#     The half-range V is set from the feature amplitude in the tails (robustly),
#     not from percentiles of the near-zero bulk. One shared norm across frames
#     keeps the scaling fixed (no flickering).

#     clip       : 'mad' uses V = k * 1.4826 * MAD; 'percentile' uses
#                  V = percentile of |data|.
#     scaling    : 'linear' -> Normalize(-V, V); 'symlog' -> SymLogNorm.
#     linthresh_frac : symlog only; linear core half-width as a fraction of V.
#     """
#     pooled = np.concatenate([np.asarray(m.data, dtype=float).ravel() for m in seq])
#     pooled = pooled[np.isfinite(pooled)]

#     if clip == 'mad':
#         sigma = 1.4826 * np.median(np.abs(pooled - np.median(pooled)))
#         V = k * sigma
#     elif clip == 'percentile':
#         V = np.percentile(np.abs(pooled), percentile)
#     else:
#         raise ValueError("clip must be 'mad' or 'percentile'")

#     if scaling == 'linear':
#         return colors.Normalize(vmin=-V, vmax=V)
#     elif scaling == 'symlog':
#         return colors.SymLogNorm(linthresh=linthresh_frac * V, vmin=-V, vmax=V, base=10)
#     else:
#         raise ValueError("scaling must be 'linear' or 'symlog'")


# def _export_sequence(seq, save_dir, prefix):
#     """
#     Write each map of a sequence to a FITS file in a dedicated folder.

#     Parameters
#     ----------
#     seq : iterable of sunpy.map.GenericMap
#         Maps to save.
#     save_dir : str
#         Target directory. Created if it does not exist.
#     prefix : str
#         Filename prefix, e.g. ``'basediff'``. Files are named
#         ``<prefix>_<index>_<obstime>.fits``.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     for i, m in enumerate(seq):
#         try:
#             stamp = m.date.to_datetime().strftime('%Y%m%dT%H%M%S')
#         except Exception:
#             # fall back to the running index if the date cannot be parsed
#             stamp = f'{i:04d}'
#         fname = os.path.join(save_dir, f'{prefix}_{i:04d}_{stamp}.fits')
#         m.save(fname, overwrite=True)


# def base_difference(maps, ref_index=0, gamma=0.2, cmap='Greys_r', save_dir=None):
#     """
#     Compute base-difference images for a list of SunPy maps.

#     Each output frame is ``maps[i] - maps[ref_index]``, i.e. every image
#     differenced against a single fixed reference frame. Arithmetic is
#     unit-aware (the reference is subtracted as a ``Quantity``), so the result
#     keeps the physical units and inherits the metadata of ``maps[i]``.

#     Parameters
#     ----------
#     maps : list of sunpy.map.GenericMap
#         Input image sequence, assumed time-ordered.
#     ref_index : int, optional
#         Index of the reference (base) frame. Default 0.
#     gamma : float, optional
#         Exponent for the shared ``PowerNorm``. Default 0.2.
#     cmap : str, optional
#         Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
#     save_dir : str or None, optional
#         If given, every output map is written as a FITS file into this folder.

#     Returns
#     -------
#     sunpy.map.MapSequence
#         Base-difference maps, one per input frame (the reference frame
#         differences to zero), sharing one grey-scale norm to avoid flickering.
#     """
#     ref = maps[ref_index]
#     diffs = []
#     with tqdm(total=len(maps), desc='Computing base difference ...') as pbar:
#         for m in maps:
#             d = m - ref.quantity
#             d.data[np.isnan(d.data)] = 0
#             diffs.append(d)
#             pbar.update(1)

#     # norm = _global_powernorm(diffs, gamma=gamma)
#     norm = _global_diffnorm(diffs, clip=clip, k=k, scaling=scaling)
#     for d in diffs:
#         d.plot_settings['norm'] = norm
#         d.plot_settings['cmap'] = cmap

#     if save_dir is not None:
#         _export_sequence(diffs, save_dir, prefix='basediff')

#     return sunpy.map.Map(diffs, sequence=True)


# def running_difference(maps, step=1, gamma=0.2, cmap='Greys_r', save_dir=None):
#     """
#     Compute running-difference images for a list of SunPy maps.

#     Each output frame is ``maps[i] - maps[i - step]``. With ``step=1`` this is
#     the consecutive-frame difference; larger ``step`` differences against an
#     earlier frame, useful for slow features or sparse cadence. Arithmetic is
#     unit-aware and the result inherits the metadata of ``maps[i]``.

#     Parameters
#     ----------
#     maps : list of sunpy.map.GenericMap
#         Input image sequence, assumed time-ordered.
#     step : int, optional
#         Frame separation between the current and reference image. Default 1.
#     gamma : float, optional
#         Exponent for the shared ``PowerNorm``. Default 0.2.
#     cmap : str, optional
#         Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
#     save_dir : str or None, optional
#         If given, every output map is written as a FITS file into this folder.

#     Returns
#     -------
#     sunpy.map.MapSequence
#         Running-difference maps (length ``len(maps) - step``), sharing one
#         grey-scale norm to avoid flickering.
#     """
#     diffs = []
#     with tqdm(total=len(maps) - step, desc='Computing running difference ...') as pbar:
#         for i in range(step, len(maps)):
#             d = maps[i] - maps[i - step].quantity
#             d.data[np.isnan(d.data)] = 0
#             diffs.append(d)
#             pbar.update(1)

#     # norm = _global_powernorm(diffs, gamma=gamma)
#     norm = _global_diffnorm(diffs, clip=clip, k=k, scaling=scaling)
#     for d in diffs:
#         d.plot_settings['norm'] = norm
#         d.plot_settings['cmap'] = cmap

#     if save_dir is not None:
#         _export_sequence(diffs, save_dir, prefix='rundiff')

#     return sunpy.map.Map(diffs, sequence=True)


# def running_ratio(maps, step=1, gamma=0.2, cmap='Greys_r', save_dir=None):
#     """
#     Compute running-ratio images for a list of SunPy maps.

#     Each output frame is ``maps[i] / maps[i - step]``. Pixels with no change
#     sit at 1; brightenings exceed 1 and dimmings fall below it. Division by a
#     ``Quantity`` yields a dimensionless map that inherits the metadata of
#     ``maps[i]``. Non-finite pixels (from zeros in the reference frame) are set
#     to 1, i.e. treated as "no change".

#     Parameters
#     ----------
#     maps : list of sunpy.map.GenericMap
#         Input image sequence, assumed time-ordered.
#     step : int, optional
#         Frame separation between the current and reference image. Default 1.
#     gamma : float, optional
#         Exponent for the shared ``PowerNorm``. Default 0.2. A power-law norm
#         suits ratios well since the data is strictly positive.
#     cmap : str, optional
#         Colour map applied to every frame. Default ``'Greys_r'`` (grey scale).
#     save_dir : str or None, optional
#         If given, every output map is written as a FITS file into this folder.

#     Returns
#     -------
#     sunpy.map.MapSequence
#         Running-ratio maps (length ``len(maps) - step``), sharing one
#         grey-scale norm to avoid flickering.
#     """
#     ratios = []
#     with tqdm(total=len(maps) - step, desc='Computing running ratio ...') as pbar:
#         for i in range(step, len(maps)):
#             r = maps[i] / maps[i - step].quantity
#             r.data[~np.isfinite(r.data)] = 1
#             ratios.append(r)
#             pbar.update(1)

#     # norm = _global_powernorm(ratios, gamma=gamma)
#     norm = _global_diffnorm(diffs, clip=clip, k=k, scaling=scaling)
#     for r in ratios:
#         r.plot_settings['norm'] = norm
#         r.plot_settings['cmap'] = cmap

#     if save_dir is not None:
#         _export_sequence(ratios, save_dir, prefix='runratio')

#     return sunpy.map.Map(ratios, sequence=True)
