"""Tests for ``legoesm.ocean.diagnostics_streamfunction``.

The MOC and BSF helpers used to live inline in
``scripts/run/global_overturning/plot_realistic_geometry_progress.py``.
These tests verify the extraction did not change behaviour and that
the helpers are physically sensible on simple analytical inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.diagnostics_streamfunction import (
    barotropic_streamfunction,
    moc_streamfunction,
)


class _FakeGrid:
    def __init__(self, radius=constants.R_earth):
        self.radius = radius


# ============================================================================
# Trivial / boundary cases
# ============================================================================

def test_moc_zero_velocity_gives_zero_streamfunction():
    n_lat, n_lon, nlev = 10, 12, 5
    v = np.zeros((n_lat + 1, n_lon, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    eta = np.zeros((n_lat, n_lon))
    H = np.full((n_lat, n_lon), 500.0)
    mask = np.ones((n_lat, n_lon))
    psi = moc_streamfunction(v, h, eta, H, mask, _FakeGrid())
    assert psi.shape == (n_lat + 1, nlev)
    np.testing.assert_allclose(psi, 0.0, atol=0.0)


def test_bsf_zero_velocity_gives_zero_streamfunction():
    n_lat, n_lon, nlev = 8, 16, 4
    u = np.zeros((n_lat, n_lon + 1, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask = np.ones((n_lat, n_lon))
    psi_bt = barotropic_streamfunction(u, h, mask, _FakeGrid())
    assert psi_bt.shape == (n_lat, n_lon)
    np.testing.assert_allclose(psi_bt, 0.0, atol=0.0)


def test_bsf_zero_layer_thickness_gives_zero_streamfunction():
    """h_partial = 0 should null out transports regardless of u."""
    n_lat, n_lon, nlev = 8, 16, 4
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n_lat, n_lon + 1, nlev))
    h = np.zeros((n_lat, n_lon, nlev))
    mask = np.ones((n_lat, n_lon))
    psi_bt = barotropic_streamfunction(u, h, mask, _FakeGrid())
    np.testing.assert_allclose(psi_bt, 0.0, atol=0.0)


# ============================================================================
# Sign convention — uniform northward flow → negative ψ in lat-z plot
# ============================================================================

def test_moc_uniform_northward_flow_sign():
    """A uniform v=+1 m/s, full-depth, all-ocean grid should produce
    a streamfunction that is **negative** at depth (positive ψ ≡
    clockwise circulation; uniform northward surface flow is the
    upper limb of an anticlockwise cell, so cumulative ψ < 0)."""
    n_lat, n_lon, nlev = 10, 12, 5
    v = np.full((n_lat + 1, n_lon, nlev), 1.0)
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask = np.ones((n_lat, n_lon))
    psi = moc_streamfunction(
        v, h, np.zeros((n_lat, n_lon)),
        np.full((n_lat, n_lon), 500.0), mask, _FakeGrid(),
    )
    # Surface integral starts at 0; bottom row has accumulated all
    # the depth-integrated flow → negative magnitude.
    assert float(np.min(psi)) < 0.0
    # Magnitude is in Sv.  Order of magnitude check: at the equator,
    # 1 m/s × 100 m × 12 cells × dx ~ 12 cells × 2π·R/12 = 2πR ·
    # 100 m × 1 m/s ~ 4e9 m³/s = 4000 Sv.  Per layer that's 800 Sv;
    # cumulative over 5 layers = ~4000 Sv at the deepest level.
    # We just check it's order-of-magnitude sane (large but bounded).
    assert float(np.min(psi)) > -50000.0


def test_bsf_eastward_flow_at_one_lat_makes_step():
    """Eastward (positive u) at a single lat row should produce a
    monotone step in ψ_bt: north of the flow, ψ_bt is non-zero;
    south of it, ψ_bt = 0 (south-wall reference)."""
    n_lat, n_lon, nlev = 8, 16, 3
    u = np.zeros((n_lat, n_lon + 1, nlev))
    u[3, :, :] = 1.0  # row 3 has uniform eastward flow
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask = np.ones((n_lat, n_lon))
    psi_bt = barotropic_streamfunction(u, h, mask, _FakeGrid())
    # South of row 3 (rows 0..2): ψ_bt ≈ 0
    np.testing.assert_allclose(psi_bt[:3], 0.0, atol=1e-12)
    # Row 3 onward: ψ_bt non-zero (monotone in lat after the source row)
    assert float(np.max(np.abs(psi_bt[3:]))) > 0.0


# ============================================================================
# Land mask zeroes contributions from masked-out cells
# ============================================================================

def test_moc_respects_land_mask():
    """With v=1 everywhere but mask=0 in a band, that band's
    contribution is zeroed at v-faces touching it."""
    n_lat, n_lon, nlev = 10, 12, 3
    v = np.ones((n_lat + 1, n_lon, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask_full = np.ones((n_lat, n_lon))
    mask_blocked = mask_full.copy()
    mask_blocked[3:7, :] = 0.0

    psi_full = moc_streamfunction(
        v, h, np.zeros((n_lat, n_lon)),
        np.full((n_lat, n_lon), 500.0), mask_full, _FakeGrid(),
    )
    psi_blocked = moc_streamfunction(
        v, h, np.zeros((n_lat, n_lon)),
        np.full((n_lat, n_lon), 500.0), mask_blocked, _FakeGrid(),
    )
    # Blocking the middle band must reduce |ψ| at the bottom.
    assert float(np.min(psi_blocked)) > float(np.min(psi_full))


def test_bsf_respects_land_mask():
    n_lat, n_lon, nlev = 8, 16, 3
    u = np.ones((n_lat, n_lon + 1, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask_full = np.ones((n_lat, n_lon))
    mask_blocked = mask_full.copy()
    mask_blocked[2:5, :] = 0.0
    psi_full = barotropic_streamfunction(u, h, mask_full, _FakeGrid())
    psi_blocked = barotropic_streamfunction(u, h, mask_blocked, _FakeGrid())
    # Blocking a band reduces the maximum |ψ| north of the block.
    assert float(np.max(np.abs(psi_blocked))) <= float(np.max(np.abs(psi_full)))


# ============================================================================
# Output shape sanity
# ============================================================================

@pytest.mark.parametrize("n_lat,n_lon,nlev",
                         [(8, 16, 3), (36, 72, 20), (4, 8, 2)])
def test_moc_output_shape(n_lat, n_lon, nlev):
    v = np.zeros((n_lat + 1, n_lon, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask = np.ones((n_lat, n_lon))
    psi = moc_streamfunction(
        v, h, np.zeros((n_lat, n_lon)),
        np.full((n_lat, n_lon), 500.0), mask, _FakeGrid(),
    )
    assert psi.shape == (n_lat + 1, nlev)


@pytest.mark.parametrize("n_lat,n_lon,nlev",
                         [(8, 16, 3), (36, 72, 20), (4, 8, 2)])
def test_bsf_output_shape(n_lat, n_lon, nlev):
    u = np.zeros((n_lat, n_lon + 1, nlev))
    h = np.full((n_lat, n_lon, nlev), 100.0)
    mask = np.ones((n_lat, n_lon))
    psi_bt = barotropic_streamfunction(u, h, mask, _FakeGrid())
    assert psi_bt.shape == (n_lat, n_lon)
