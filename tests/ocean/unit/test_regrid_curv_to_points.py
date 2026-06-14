"""Point-target curvilinear IDW regrid used to put NEMO bathy on cube cells
(`run_omip_core2._regrid_curv_to_points`).

Mirrors the scorer's `regrid_curv_to_latlon` but onto ARBITRARY target points
(any shape, e.g. cube (6,n,n)) — the scorer version meshgrids 1-D axes so it
can't target cube cell centres. Tests: a smooth field is reproduced at target
points; the output keeps the target's shape; the ocean flag marks far-from-source
targets as 0.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


def _fn():
    from scripts.run.run_omip_core2 import _regrid_curv_to_points
    return _regrid_curv_to_points


def test_reproduces_smooth_field_and_shape():
    """A smooth field f=lat (deg) on a dense source is recovered at scattered
    target points to a few tenths of a degree, with the target's shape."""
    fn = _fn()
    # dense lat-lon source covering the globe
    slat = np.linspace(-89.0, 89.0, 180)
    slon = np.linspace(0.0, 358.0, 180)
    SLON, SLAT = np.meshgrid(slon, slat)
    field = SLAT.copy()                       # value == latitude
    ocean = np.ones_like(SLAT)                # all ocean
    # arbitrary-shaped target (mimics cube (6,n,n) with a 3-D array)
    tlat = np.array([[[-40.0, 10.0], [55.0, -5.0]],
                     [[0.0, 33.0], [-70.0, 80.0]]])
    tlon = np.array([[[10.0, 200.0], [120.0, 300.0]],
                     [[45.0, 270.0], [180.0, 5.0]]])
    out, oflag = fn(field, SLAT, SLON, ocean, tlat, tlon, max_deg=3.0)
    assert out.shape == tlat.shape
    assert oflag.shape == tlat.shape
    # recovered value ~ target latitude (smooth field, dense source)
    assert np.allclose(out, tlat, atol=0.6), np.abs(out - tlat).max()
    assert np.all(oflag == 1.0)               # every target near a source


def test_ocean_flag_marks_far_targets():
    """A target far from any SOURCE OCEAN cell gets ocean_flag=0."""
    fn = _fn()
    # source ocean only in a small equatorial patch
    slat = np.linspace(-5.0, 5.0, 40)
    slon = np.linspace(0.0, 10.0, 40)
    SLON, SLAT = np.meshgrid(slon, slat)
    field = SLAT.copy()
    ocean = np.ones_like(SLAT)
    tlat = np.array([0.0, 80.0])              # one inside patch, one far (N pole)
    tlon = np.array([5.0, 200.0])
    out, oflag = fn(field, SLAT, SLON, ocean, tlat, tlon, max_deg=3.0)
    assert oflag[0] == 1.0                     # near the source patch
    assert oflag[1] == 0.0                     # far -> flagged not-ocean


def test_ignores_land_source_cells():
    """Land source cells (ocean_mask=0) are excluded from the interpolation."""
    fn = _fn()
    slat = np.linspace(-10.0, 10.0, 20)
    slon = np.linspace(0.0, 20.0, 20)
    SLON, SLAT = np.meshgrid(slon, slat)
    field = np.where(SLAT > 0.0, 100.0, 1.0)   # north half = sentinel 100
    ocean = (SLAT <= 0.0).astype(float)        # only south half is ocean
    # target in the north: nearest CELLS are land(100) but only ocean(1) counts
    out, oflag = fn(field, SLAT, SLON, ocean, np.array([3.0]), np.array([10.0]),
                    max_deg=30.0)
    assert out[0] == pytest.approx(1.0, abs=1e-9)   # pulled from ocean, not land
