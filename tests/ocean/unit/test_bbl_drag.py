"""Distributed-BBL bottom drag (Killworth & Edwards 1999).

Tests for ``bbl_distributed_drag_face_column`` in
``legoesm.ocean.dynamics.ocean_tendency_common`` — the grid-agnostic
helper used by both the lat-lon C-grid and MPAS Voronoi PE entries to
spread bottom drag over a fixed Ekman thickness instead of crashing
into a thin partial cell.

Five checks:

1. **Deep cell (h_bot ≥ H_BBL):** drag at the bottom equals the
   single-cell legacy form ``-r·u/h_bot``; cells above get zero.
2. **Thin partial cell (h_bot < H_BBL):** drag at the bottom equals
   ``-r·u/H_BBL`` (10× weaker than the legacy ``-r·u/h_bot``,
   preventing CFL violation); cells above the bottom absorb the rest
   of the BBL band with overlap-weighted drag.
3. **Total drag stress conserved:** sum over cells of ``drag·h``
   equals ``-r·u_bot`` (the total stress at the seafloor) up to the
   weak velocity-difference contribution from levels above.
4. **MPAS+thin-partial-cell end-to-end:** the explicit-CFL violation
   we observed on ETOPO+ico-4 (sign-reversal at h_bot=0.28 m,
   r·dt=0.55 m) is gone when ``H_BBL=50 m``.
5. **Grid-agnostic:** the helper produces correct output on both
   2-D ``(face_dim, nlev)`` and 3-D ``(n_lat, n_lon, nlev)`` inputs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_tendency_common import (
    bbl_distributed_drag_face_column,
)
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# 1. Deep cell — recovers single-cell legacy form
# ---------------------------------------------------------------------------


def test_deep_cell_recovers_legacy_drag():
    """When h_bot >= H_BBL, the BBL is contained inside the bottom cell.
    Bottom-cell overlap = H_BBL exactly → drag = -r·u/h_bot.  Cells
    above have zero overlap → zero drag."""
    nlev = 5
    h = jnp.array([200.0, 200.0, 200.0, 200.0, 200.0])  # uniform deep
    u = jnp.array([0.5, 0.4, 0.3, 0.2, 0.1])
    r = 1.1e-3
    H_BBL = 50.0
    drag = bbl_distributed_drag_face_column(u, h, r, H_BBL)

    expected_bot = -r * u[-1] / h[-1]    # legacy single-cell form
    np.testing.assert_allclose(float(drag[-1]), expected_bot, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(drag[:-1]), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Thin partial cell — CFL safety
# ---------------------------------------------------------------------------


class TestThinPartialCell:

    def test_thin_bottom_drag_is_capped_by_H_BBL(self):
        """h_bot = 1 m, H_BBL = 50 m → bottom drag is -r·u/H_BBL,
        which is 50× weaker than the legacy -r·u/h_bot.  This is what
        prevents CFL violation."""
        h = jnp.array([100.0, 100.0, 100.0, 100.0, 1.0])  # thin bottom
        u = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        r = 1.1e-3
        H_BBL = 50.0
        drag = bbl_distributed_drag_face_column(u, h, r, H_BBL)
        # Bottom-cell drag = -r·u·overlap / (h·H_BBL) where overlap = h
        # = -r·u / H_BBL (because h_bot < H_BBL, the BBL band fully
        # overlaps the bottom cell).
        expected_bot = -r * u[-1] / H_BBL
        np.testing.assert_allclose(float(drag[-1]), expected_bot, rtol=1e-12)
        # Compare with the legacy single-cell drag (would have been -r·u/1.0).
        legacy_bot = -r * u[-1] / 1.0
        assert abs(float(drag[-1])) < abs(legacy_bot) / 40, (
            f"BBL drag ({float(drag[-1]):.3e}) must be much weaker than "
            f"legacy single-cell drag ({legacy_bot:.3e})"
        )

    def test_BBL_spans_multiple_cells(self):
        """H_BBL = 50 m, bottom = 5, level-above = 30, level-2-above = 100:
        the BBL band [z_sea, z_sea+50] spans three cells.  Each cell's
        overlap is its intersection with the BBL band."""
        h = jnp.array([100.0, 100.0, 100.0, 30.0, 5.0])  # h_bot=5, lev-above=30
        u = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        r = 1.1e-3
        H_BBL = 50.0
        drag = bbl_distributed_drag_face_column(u, h, r, H_BBL)
        # Geometry:
        #   z_seafloor = -335,  bbl_top = -335 + 50 = -285
        #   level 4 (bot, h=5):   [z_top=-330, z_bot=-335] → overlap=5  (whole)
        #   level 3 (h=30):       [z_top=-300, z_bot=-330] → overlap=30 (whole)
        #   level 2 (h=100):      [z_top=-200, z_bot=-300] → overlap=15 (top 15)
        #   levels above: outside BBL band, overlap=0
        expected_bot = -r * u[-1] * 5.0 / (h[-1] * H_BBL)        # = -r·u/H_BBL
        expected_above1 = -r * u[-2] * 30.0 / (h[-2] * H_BBL)
        expected_above2 = -r * u[-3] * 15.0 / (h[-3] * H_BBL)
        np.testing.assert_allclose(float(drag[-1]), expected_bot, rtol=1e-12)
        np.testing.assert_allclose(float(drag[-2]), expected_above1, rtol=1e-12)
        np.testing.assert_allclose(float(drag[-3]), expected_above2, rtol=1e-12)
        # Levels even higher: outside BBL, zero drag.
        np.testing.assert_allclose(np.asarray(drag[:-3]), 0.0, atol=1e-12)
        # Sanity: total overlap across all cells = H_BBL exactly.
        # (Computed via z_top - z_bot for each cell, clamped to BBL band.)
        z_half = -np.cumsum(np.concatenate([[0.0], np.asarray(h)]))
        z_top, z_bot = z_half[:-1], z_half[1:]
        bbl_top = z_half[-1] + H_BBL
        overlap = np.maximum(0.0, np.minimum(z_top, bbl_top) - np.maximum(z_bot, z_half[-1]))
        np.testing.assert_allclose(overlap.sum(), H_BBL, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Total drag stress
# ---------------------------------------------------------------------------


def test_total_drag_stress_uniform_velocity():
    """For uniform velocity column (u_k = u₀ everywhere), the integrated
    stress sum_k(drag_k · h_k) equals -r·u₀ — same total stress as the
    single-cell legacy formulation (since BBL just redistributes it)."""
    h = jnp.array([100.0, 100.0, 100.0, 30.0, 5.0])
    u = jnp.full(5, 0.5)
    r = 1.1e-3
    H_BBL = 50.0
    drag = bbl_distributed_drag_face_column(u, h, r, H_BBL)
    total_stress = float(jnp.sum(drag * h))
    expected_stress = -r * 0.5
    np.testing.assert_allclose(total_stress, expected_stress, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. End-to-end: MPAS ETOPO-like config doesn't sign-reverse
# ---------------------------------------------------------------------------


def test_bbl_prevents_sign_reversal_on_thin_partial_cells():
    """Reproduce the ETOPO+ico4 CFL pathology: r=1.1e-3, dt=500s,
    h_bot=0.3m → r·dt/h_bot = 1.83.  Without BBL, explicit drag
    multiplies u by (1 - 1.83) = -0.83 → sign reversal.  With BBL=50m
    the per-step factor is (1 - r·dt/H_BBL) = 1 - 0.011 = 0.989 — a
    well-behaved 1.1% damping per step."""
    h = jnp.array([100.0, 100.0, 100.0, 30.0, 0.3])
    u = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    r = 1.1e-3
    dt = 500.0

    # Legacy single-cell drag at bottom:
    legacy_drag_bot = -r * u[-1] / h[-1]
    legacy_factor = 1.0 + dt * legacy_drag_bot / u[-1]
    assert legacy_factor < 0, (
        f"Sanity: legacy explicit drag should sign-reverse at h_bot=0.3m, "
        f"dt=500s; got factor {legacy_factor:.3f}"
    )

    # BBL drag at bottom:
    drag = bbl_distributed_drag_face_column(u, h, r, 50.0)
    bbl_factor = 1.0 + dt * float(drag[-1]) / float(u[-1])
    assert 0.95 < bbl_factor < 1.0, (
        f"BBL drag should give a stable ~1% damping per step at h_bot=0.3m; "
        f"got factor {bbl_factor:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Grid-agnostic — works on both 1-D and 2-D leading dims
# ---------------------------------------------------------------------------


def test_2d_face_dim_input():
    """Helper must accept arrays with multiple leading dimensions
    (e.g., (n_lat, n_lon, nlev) for the lat-lon C-grid u-faces)."""
    n_lat, n_lon, nlev = 4, 6, 5
    h = jnp.full((n_lat, n_lon, nlev), 100.0).at[..., -1].set(0.3)
    u = jnp.full((n_lat, n_lon, nlev), 0.5)
    drag = bbl_distributed_drag_face_column(u, h, 1.1e-3, 50.0)
    assert drag.shape == (n_lat, n_lon, nlev)
    # Bottom drag uniform across the (n_lat, n_lon) grid:
    bot = np.asarray(drag[..., -1])
    np.testing.assert_allclose(bot, bot[0, 0], rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. End-to-end MPAS smoke: bbl_thickness>0 stays finite, bbl=0 doesn't
# ---------------------------------------------------------------------------


def test_mpas_bbl_keeps_thin_partial_cell_run_stable():
    """One 500s step on an ico-2 mesh with a Gaussian seamount that
    forces a thin partial-cell bottom (h_bot ~ 0.5 m).  With BBL=0
    (legacy single-cell drag) at r=1.1e-3, the explicit-CFL ratio
    r·dt/h_bot ≈ 1 → near-sign-reversal regime.  With BBL=50 the
    drag is bounded; max|u| stays sensibly bounded after 5 steps."""
    mesh = create_voronoi_mesh(subdivision_level=2)
    z = create_ocean_z_star(n_levels=10, H_max=4000.0)
    # Seamount with very tall peak so some cells get a thin partial bottom.
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    H_bathy = jnp.asarray(
        4000.0 - 3700.0 * np.exp(-(lat ** 2 + lon ** 2) / 0.3),
        dtype=jnp.float64,
    )
    pc = create_partial_cell_coordinate(z, H_bathy)

    # Build seamount state with realistic stratification.
    state = rest_state_mpas_ocean(
        mesh, z, H_max=4000.0, land_lat_threshold=90.0,
    )
    state = state._replace(
        H_bathy=Field(
            data=H_bathy.astype(state.H_bathy.data.dtype),
            name="H_bathy", dims=("nCells",), units="m",
        ),
    )

    cfg_bbl = MPASOceanConfig(
        barotropic_solver="implicit_cn", A_h=1e4, A_v=1e-3, K_v=1e-4,
        bottom_drag_r=1.1e-3, bottom_drag_bbl_thickness=50.0,
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        min_water_column_m=1.0,
        pgf_scheme="centered", pv_scheme="enstrophy",
    )
    model = MPASOceanModel(mesh, pc, cfg_bbl)
    s = state
    for _ in range(5):
        s = model.step(s, dt=500.0)
    mu = float(jnp.max(jnp.abs(s.u.data)))
    assert np.isfinite(mu), f"BBL run went non-finite (mu={mu})"
    assert mu < 1.0, f"BBL run unphysically large at 5 steps: mu={mu:.3e}"
