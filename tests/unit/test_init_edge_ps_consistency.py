"""The initializer's barometric correction, and the probe that measures it.

The ERA5 initializer corrects the CELL surface pressure after it smooths the
terrain onto the mesh, and does not correct the EDGE surface pressure that
places the wind levels.  The probe
``scripts/validate/amip_bias/init_edge_ps_consistency.py`` measures how large
that missing term is; this pins the arithmetic it relies on, on a synthetic
terrain whose answer is known, so a number quoted from the probe means
something.
"""
from __future__ import annotations

import numpy as np

from legoesm import constants


def _barometric(p_s, dphis, T):
    """The correction the initializer applies at cells (era5_to_state.py)."""
    return p_s * np.exp(dphis / (constants.R_d * T))


def test_lowering_terrain_raises_surface_pressure_by_the_hydrostatic_amount():
    """Smoothing a mountain DOWN must raise p_s, by rho*g*dz to first order."""
    p_s, T = 1.0e5, 270.0
    dz = 120.0                       # metres of terrain removed by smoothing
    dphis = dz * constants.g         # phis_raw - phis_smooth > 0
    out = _barometric(p_s, dphis, T)
    assert out > p_s
    rho = p_s / (constants.R_d * T)
    np.testing.assert_allclose(out - p_s, rho * constants.g * dz,
                               rtol=0.02)   # first order in dphis/(R_d T)
    # the scale the probe reports over coastal cells: ~120 m is ~15 hPa
    assert 12.0 < (out - p_s) / 100.0 < 18.0


def test_untouched_terrain_gets_no_correction():
    """A cell the smoother does not move keeps its surface pressure exactly."""
    p_s = np.array([1.0e5, 9.0e4, 1.01e5])
    out = _barometric(p_s, np.zeros(3), np.full(3, 270.0))
    np.testing.assert_allclose(out, p_s, rtol=0.0, atol=0.0)


def test_the_correction_is_signed_the_right_way_when_terrain_is_raised():
    """Smoothing a valley UP must lower p_s, the mirror of the mountain case."""
    p_s, T = 1.0e5, 270.0
    out = _barometric(p_s, -120.0 * constants.g, T)
    assert out < p_s
    assert 12.0 < (p_s - out) / 100.0 < 18.0
