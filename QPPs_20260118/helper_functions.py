import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.constants import c
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.map import Map, make_fitswcs_header
from sunpy.time import TimeRange
from sunpy.net import Fido, attrs as a
from xrayvision.clean import vis_clean
from xrayvision.imaging import vis_to_image, vis_to_map
from xrayvision.mem import mem, resistant_mean
import os

from stixpy.calibration.visibility import calibrate_visibility, create_meta_pixels, create_visibility
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord
from stixpy.imaging.em import em
from stixpy.map.stix import STIXMap  # noqa
from stixpy.product import Product
from stixpy.product.sources.science import ScienceData

import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

__all__ = [
    "create_maps",
    "ltc_calc"
]

def ltc_calc(time):

    time = Time(time)

    solo = get_horizons_coord("Solar Orbiter", time)
    earth = get_body_heliographic_stonyhurst("earth", time)

    # Distance to Sun (radial distance from Sun-centered frame)
    dist_sun = solo.radius.to(u.m)

    ltt_sun = (dist_sun / c).to(u.s)

    ltt_diff = ((earth.radius.to(u.m)) / c).to(u.s) - ltt_sun

    return ltt_diff


def create_maps(sci_data, 
                bkg_data, 
                start_time, 
                end_time, 
                energy_range,
                plot_location=True,
                plot_maps=True,
                save_maps=True,
                save_directory=None
                ):
    

    if not isinstance(sci_data,ScienceData):
        cpd_sci = Product(sci_data)
    else:
        cpd_sci = sci_data

    if not isinstance(bkg_data,ScienceData):
        cpd_bkg = Product(bkg_data)
    else:
        cpd_bkg = bkg_data
        
    if isinstance(start_time,Time):
        time_range_sci = [start_time, 
                      end_time]
    else:
        time_range_sci = [pd.to_datetime(start_time), 
                      pd.to_datetime(end_time)]        

    time_range_bkg = [
        cpd_bkg.time_range.start-timedelta(minutes=1),
        cpd_bkg.time_range.end+timedelta(minutes=1),
    ] 


    meta_pixels_sci = create_meta_pixels(
        cpd_sci, time_range=time_range_sci, energy_range=energy_range, flare_location=[0, 0] * u.arcsec, no_shadowing=True
    )

    meta_pixels_bkg = create_meta_pixels(
        cpd_bkg, time_range=time_range_bkg, energy_range=energy_range, flare_location=[0, 0] * u.arcsec, no_shadowing=True
    )

    meta_pixels_bkg_subtracted = {
        **meta_pixels_sci,
        "abcd_rate_kev_cm": meta_pixels_sci["abcd_rate_kev_cm"] - meta_pixels_bkg["abcd_rate_kev_cm"],
        "abcd_rate_error_kev_cm": np.sqrt(
            meta_pixels_sci["abcd_rate_error_kev_cm"] ** 2 + meta_pixels_bkg["abcd_rate_error_kev_cm"] ** 2
        )
    }   

    tot = (meta_pixels_bkg_subtracted['abcd_rate_kev_cm'] * meta_pixels_bkg_subtracted['areas']) * np.diff(meta_pixels_bkg_subtracted['energy_range']) * meta_pixels_bkg_subtracted['time_range'].seconds  

    counts = np.sum(tot)

    vis = create_visibility(meta_pixels_bkg_subtracted)

    vis_tr = TimeRange(vis.meta["time_range"])
    roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
    center_hpc = SkyCoord(0 * u.deg, 0 * u.deg, frame=Helioprojective(obstime=vis_tr.center, observer=solo))

    cal_vis = calibrate_visibility(vis, flare_location=center_hpc)

    # order by sub-collimator e.g. 10a, 10b, 10c, 9a, 9b, 9c ....
    isc_10_7 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]
    idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_7)).ravel()    

    vis10_7 = cal_vis[idx]

    imsize = [512, 512] * u.pixel  # number of pixels of the map to reconstruct
    pixel = [10, 10] * u.arcsec / u.pixel  # pixel size in arcsec

    bp_image = vis_to_image(vis10_7, imsize, pixel_size=pixel)

    vis_tr = TimeRange(vis.meta["time_range"])
    roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
    solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")
    coord_stix = center_hpc.transform_to(STIXImaging(obstime=vis_tr.start, obstime_end=vis_tr.end, observer=solo))
    header = make_fitswcs_header(
        bp_image, coord_stix, telescope="STIX", observatory="Solar Orbiter", scale=[10, 10] * u.arcsec / u.pix
    )
    fd_bp_map = Map((bp_image, header))

    header_hp = make_fitswcs_header(
        bp_image, center_hpc, scale=[10, 10] * u.arcsec / u.pix, rotation_angle=90 * u.deg + roll
    )
    hp_map = Map((bp_image, header_hp))
    hp_map_rotated = hp_map.rotate()

    if plot_location:
        fig = plt.figure(layout="constrained", figsize=(12, 6))
        ax = fig.subplot_mosaic(
            [["stix", "hpc"]], per_subplot_kw={"stix": {"projection": fd_bp_map}, "hpc": {"projection": hp_map_rotated}}
        )
        fd_bp_map.plot(axes=ax["stix"])
        fd_bp_map.draw_limb()
        fd_bp_map.draw_grid()

        hp_map_rotated.plot(axes=ax["hpc"])
        hp_map_rotated.draw_limb()
        hp_map_rotated.draw_grid()

        max_pixel = np.argwhere(fd_bp_map.data == fd_bp_map.data.max()).ravel() * u.pixel
        # because WCS axes and array are reversed
        max_stix = fd_bp_map.pixel_to_world(max_pixel[1], max_pixel[0])

        ax["stix"].plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
        ax["hpc"].plot_coord(max_stix, marker=".", markersize=50, fillstyle="none", color="r", markeredgewidth=2)
        fig.tight_layout()
    
    meta_pixels_sci = create_meta_pixels(
        cpd_sci, time_range=time_range_sci, energy_range=energy_range, flare_location=max_stix, no_shadowing=True
    )

    meta_pixels_bkg_subtracted = {
        **meta_pixels_sci,
        "abcd_rate_kev_cm": meta_pixels_sci["abcd_rate_kev_cm"] - meta_pixels_bkg["abcd_rate_kev_cm"],
        "abcd_rate_error_kev_cm": np.sqrt(
            meta_pixels_sci["abcd_rate_error_kev_cm"] ** 2 + meta_pixels_bkg["abcd_rate_error_kev_cm"] ** 2
        ),
    }

    vis = create_visibility(meta_pixels_bkg_subtracted)
    cal_vis = calibrate_visibility(vis, flare_location=max_stix)

    isc_10_3 = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1]
    idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_3)).ravel()

    cal_vis.meta["offset"] = max_stix
    vis10_3 = cal_vis[idx]   

    imsize = [129, 129] * u.pixel  # number of pixels of the map to reconstruct
    pixel = [2, 2] * u.arcsec / u.pixel  # pixel size in arcsec

    bp_nat = vis_to_image(vis10_3, imsize, pixel_size=pixel)

    bp_uni = vis_to_image(vis10_3, imsize, pixel_size=pixel, scheme="uniform")      

    bp_map = vis_to_map(vis10_3, imsize, pixel_size=pixel)  

    niter = 200  # number of iterations
    gain = 0.1  # gain used in each clean iteration
    beam_width = 20.0 * u.arcsec
    clean_map, model_map, resid_map = vis_clean(
        vis10_3, imsize, pixel_size=pixel, gain=gain, niter=niter, clean_beam_width=20 * u.arcsec
    )    

    header = make_fitswcs_header(
        clean_map.data,
        max_stix.transform_to(Helioprojective(obstime=vis_tr.center, observer=solo)),
        telescope="STIX",
        observatory="Solar Orbiter",
        scale=pixel,
        rotation_angle=90 * u.deg + roll,
        
    )    

    snr_value, _ = resistant_mean((np.abs(vis10_3.visibilities) / vis10_3.amplitude_uncertainty).flatten(), 3)
    percent_lambda = 2 / (snr_value**2 + 90)
    mem_map = mem(vis10_3, shape=imsize, pixel_size=pixel, percent_lambda=percent_lambda)

    em_map = em(
        meta_pixels_bkg_subtracted["abcd_rate_kev_cm"],
        cal_vis,
        shape=imsize,
        pixel_size=pixel,
        flare_location=max_stix,
        idx=idx,
    )


    clean_map = Map((clean_map.data, header)).rotate()
    bp_map = Map((bp_nat, header)).rotate()
    mem_map = Map((mem_map.data, header)).rotate()
    em_map = Map((em_map, header)).rotate()

    vmax = max([clean_map.data.max(), mem_map.data.max(), em_map.data.max()])

    st = time_range_sci[0]
    et = time_range_sci[1]

    if save_maps:

        if save_directory is not None:
            mem_name = f'{save_directory}stix_{int(energy_range[0].value)}-{int(energy_range[1].value)}_{st.strftime('%Y-%m-%dT%H:%M:%S')}-{et.strftime('%Y-%m-%dT%H:%M:%S')}_MEM.fits'
            em_name = f'{save_directory}stix_{int(energy_range[0].value)}-{int(energy_range[1].value)}_{st.strftime('%Y-%m-%dT%H:%M:%S')}-{et.strftime('%Y-%m-%dT%H:%M:%S')}_MEM.fits'

            mem_map.save(mem_name,overwrite=True)
            em_map.save(em_name,overwrite=True)

        else:
            mem_name = f'stix_{int(energy_range[0].value)}-{int(energy_range[1].value)}_{st.strftime('%Y-%m-%dT%H:%M:%S')}-{et.strftime('%Y-%m-%dT%H:%M:%S')}_MEM.fits'
            em_name = f'stix_{int(energy_range[0].value)}-{int(energy_range[1].value)}_{st.strftime('%Y-%m-%dT%H:%M:%S')}-{et.strftime('%Y-%m-%dT%H:%M:%S')}_MEM.fits'

            mem_map.save(mem_name,overwrite=True)
            em_map.save(em_name,overwrite=True)

    if plot_maps:

        fig = plt.figure(figsize=(12, 12))
        ax = fig.subplot_mosaic(
            [
                ["bp", "clean"],
                ["mem", "em"],
            ],
            subplot_kw={"projection": clean_map},
        )
        a = bp_map.plot(axes=ax["bp"])
        ax["bp"].set_title("Back Projection")
        b = clean_map.plot(axes=ax["clean"])
        ax["clean"].set_title("Clean")
        c = mem_map.plot(axes=ax["mem"])
        ax["mem"].set_title("MEM GE")
        d = em_map.plot(axes=ax["em"])
        ax["em"].set_title("EM")
        fig.colorbar(a, ax=ax.values())
        plt.show()

    dict = {'time_range':time_range_sci,
             'energy_range':energy_range,
              'counts':counts,
              'mem_map':mem_name,
               'em_map':em_name }

    return dict