"""Tests for the vertical-dipole discriminator.

The probe's whole job is to tell a REDISTRIBUTION of freshwater apart from a
DIFFERENT AMOUNT of it. So the central test plants one of each, with the same
surface signal, and asserts the cancellation ratio separates them. A probe
that only looked at the surface passes neither.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROBE = (Path(__file__).resolve().parents[2]
          / "scripts" / "validate" / "ocean_fidelity" / "runoff_vertical_dipole.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_dipole", _PROBE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_dipole"] = m
    spec.loader.exec_module(m)
    return m


def test_dz_from_centers_tiles_the_column():
    m = _mod()
    z = np.array([1.0, 3.0, 7.0, 15.0])
    dz = m.dz_from_centers(z)
    assert dz.size == z.size
    # First layer runs from the surface to the midpoint below it.
    assert dz[0] == pytest.approx(2.0)
    # No gaps and no overlaps: the faces must tile.
    assert dz.sum() == pytest.approx(15.0 + 0.5 * 8.0)
    assert (dz > 0).all()


def test_pure_redistribution_cancels_and_a_net_offset_does_not():
    """The discriminator itself. Same surface value, opposite verdicts."""
    m = _mod()
    z = np.arange(5.0, 155.0, 10.0)      # 15 levels inside 150 m
    dz = m.dz_from_centers(z)

    # (b) vertical placement: fresh at the top, salty below, nothing added.
    redist = np.zeros_like(z)
    redist[0] = -1.0
    redist[1:] = -redist[0] * dz[0] / dz[1:].sum()
    r = m.dipole_stats(redist, dz, z, 150.0)
    assert r["surface"] < 0 < r["subsurface"]
    assert r["cancel"] == pytest.approx(0.0, abs=1e-9)

    # (a) a different amount of freshwater: fresh at the top, nothing to
    # compensate it. The surface value is IDENTICAL to the case above.
    offset = np.zeros_like(z)
    offset[0] = -1.0
    o = m.dipole_stats(offset, dz, z, 150.0)
    assert o["surface"] == pytest.approx(r["surface"])
    assert o["cancel"] == pytest.approx(1.0, abs=1e-9)


def test_uniform_offset_through_the_column_is_fully_uncancelled():
    m = _mod()
    z = np.arange(5.0, 155.0, 10.0)
    dz = m.dz_from_centers(z)
    s = m.dipole_stats(np.full_like(z, -0.3), dz, z, 150.0)
    assert s["cancel"] == pytest.approx(1.0, abs=1e-9)


def test_band_ignores_levels_below_the_spreading_depth():
    """Deep structure must not leak into the verdict."""
    m = _mod()
    z = np.array([5.0, 25.0, 75.0, 125.0, 500.0, 1000.0])
    dz = m.dz_from_centers(z)
    prof = np.array([-1.0, 0.2, 0.2, 0.2, 99.0, -99.0])
    s = m.dipole_stats(prof, dz, z, 150.0)
    # The two huge deep values would dominate any statistic that saw them.
    assert abs(s["gross"]) < 1e3
    assert np.isfinite(s["cancel"])


def test_node_to_cell_mean_averages_into_the_right_cell():
    m = _mod()
    cell_lat = np.array([[0.0, 0.0], [10.0, 10.0]])
    cell_lon = np.array([[0.0, 40.0], [0.0, 40.0]])
    wet = np.ones((2, 2))
    # Two nodes on the first cell, one on the last; nothing near cell (0,1).
    node_lat = np.array([0.1, -0.1, 10.0])
    node_lon = np.array([0.1, -0.1, 40.0])
    node_val = np.array([[1.0], [3.0], [7.0]])
    out = m.node_to_cell_mean(node_val, node_lat, node_lon,
                              cell_lat, cell_lon, wet)
    assert out.shape == (2, 2, 1)
    assert out[0, 0, 0] == pytest.approx(2.0)     # mean of 1 and 3
    assert out[1, 1, 0] == pytest.approx(7.0)
    # A cell no node landed in must stay NaN rather than borrow a distant one.
    assert np.isnan(out[0, 1, 0])


def test_dry_geometry_raises_rather_than_returning_empty():
    m = _mod()
    with pytest.raises(SystemExit):
        m.node_to_cell_mean(np.zeros((3, 1)), np.zeros(3), np.zeros(3),
                            np.zeros((2, 2)), np.zeros((2, 2)),
                            np.zeros((2, 2)))
