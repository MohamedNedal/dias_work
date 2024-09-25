#!/usr/bin/env python
# coding: utf-8

import astropy.units as u
from sunkit_pyvista import SunpyPlotter
from sunkit_pyvista.sample import LOW_RES_AIA_193

# Start by creating a plotter
plotter = SunpyPlotter()

# Plot a map
plotter.plot_map(LOW_RES_AIA_193, clip_interval=[1, 99] * u.percent, assume_spherical_screen=False)
# Add an arrow to show the solar rotation axis
plotter.plot_solar_axis()
plotter.show()


