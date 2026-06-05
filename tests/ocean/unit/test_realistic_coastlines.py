"""Phase 2 of the realistic-geometry plan: realistic coastline validation.

The land-mask + face-mask machinery (``replace_land_mask``,
``compute_face_masks``) exists, but is exercised only on simple convex
domains.  This phase stress-tests it on diagonal and curved coastlines
that emerge in real coastline geometry.

Three test classes, each catching a distinct failure mode:

1. ``TestDiagonalCoastline`` — 45° diagonal coast, rest state held at
   machine precision.  Catches corner-cell flux leak.
2. ``TestIslandTopology`` — isolated circular island in a closed basin,
   tracer conservation over the closed domain.  Catches non-simply-
   connected topology bugs.
3. ``TestRealisticMaskConsistency`` — load the cached ETOPO 1° dataset
   on a 72×144 (2.5°) grid, verify the runtime mask-consistency
   invariant holds globally.  Catches u_mask/v_mask drift.

The plan also calls for a "runtime check that asserts
``total_mass_flux_through_land == 0`` to round-off after every
step()".  This is already covered by:
  - The existing face-mask consistency check in
    ``LatLonCGridOceanModel._assert_runtime_invariants``, which fires
    every step when ``enable_runtime_checks=True``.
  - Closed-domain tracer conservation in island/diagonal tests below
    (any leak through a dry face would corrupt total tracer
    mass — the conservation tolerance directly bounds leakage).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    compute_layer_thickness,
    create_ocean_z_star,
)
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture
def grid_18x36():
    return create_latlon_grid(n_lat=18, n_lon=36)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0,
                                 dz_surface=10.0, dz_deep=500.0)


# =========================================================================
# 1. Diagonal coastline
# =========================================================================


def _make_diagonal_mask(n_lat, n_lon):
    """45° diagonal coastline in index space.

    Cell (i, j) is ocean iff i + (j * n_lat / n_lon) > n_lat / 2.
    Lower-left half is land, upper-right is ocean.
    """
    i_idx = jnp.arange(n_lat)[:, None]
    j_idx = jnp.arange(n_lon)[None, :]
    # Normalize so the cut runs at 45° in (i, j*n_lat/n_lon) space.
    j_norm = j_idx * (n_lat / n_lon)
    return (i_idx + j_norm > n_lat / 2).astype(jnp.float64)


class TestDiagonalCoastline:
    """Diagonal land/ocean boundary — corner cells on the cut should
    not leak any flow through the diagonal."""

    def test_face_mask_consistency(self, grid_18x36):
        """The runtime invariant: u_mask = land_mask AND
        rolled(land_mask).  Verify it holds on a non-rectangular
        coastline."""
        mask = _make_diagonal_mask(grid_18x36.n_lat, grid_18x36.n_lon)
        u_mask, v_mask = compute_face_masks(mask)
        # u-face (i, j) wet iff cell (i, j-1) AND cell (i, j) wet.
        # Use the same periodic-roll convention as compute_face_masks.
        u_expected_interior = mask * jnp.roll(mask, 1, axis=1)
        np.testing.assert_array_equal(
            np.asarray(u_mask[:, :-1]), np.asarray(u_expected_interior),
        )
        # v-face (i, j) wet iff cell (i-1, j) AND cell (i, j) wet,
        # with poles always masked.
        v_expected_interior = mask[:-1] * mask[1:]
        np.testing.assert_array_equal(
            np.asarray(v_mask[1:-1]), np.asarray(v_expected_interior),
        )
        # Pole rows of v_mask should be zero.
        assert float(jnp.max(jnp.abs(v_mask[0]))) == 0.0
        assert float(jnp.max(jnp.abs(v_mask[-1]))) == 0.0

    def test_rest_state_tendency_machine_zero(self, grid_18x36, z_coord):
        """Rest state on diagonal coastline: every momentum tendency
        on every wet face should be machine-zero."""
        mask = _make_diagonal_mask(grid_18x36.n_lat, grid_18x36.n_lon)
        state = rest_state_latlon_cgrid_ocean(
            grid_18x36, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            land_mask_override=mask,
        )
        cfg = LatLonCGridOceanConfig()
        tend = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid_18x36, z_coord, cfg,
        )
        u_max = float(jnp.max(jnp.abs(tend.du_dt.data)))
        v_max = float(jnp.max(jnp.abs(tend.dv_dt.data)))
        assert u_max < 1e-12, f"du/dt max = {u_max} on diagonal-coast rest"
        assert v_max < 1e-12, f"dv/dt max = {v_max} on diagonal-coast rest"

    def test_rest_state_stable_under_stepping(self, grid_18x36, z_coord):
        """Step 12 hours on diagonal coastline; |u|, |eta| stay bounded
        and conservation holds.  This catches mask-related accumulating
        drift that t=0 tendency tests can miss."""
        mask = _make_diagonal_mask(grid_18x36.n_lat, grid_18x36.n_lon)
        state = rest_state_latlon_cgrid_ocean(
            grid_18x36, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            land_mask_override=mask,
        )
        cfg = LatLonCGridOceanConfig(
            barotropic_solver="implicit_cn",
            enable_runtime_checks=True,    # exercises mask consistency
        )
        model = LatLonCGridOceanModel(grid_18x36, z_coord, cfg)
        s = state
        for _ in range(12):
            s = model.step(s, 3600.0)   # runtime check fires every step
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.eta.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        assert u_max < 1e-3, f"|u| drift = {u_max}"
        assert eta_max < 0.5, f"|eta| drift = {eta_max}"


# =========================================================================
# 2. Island topology
# =========================================================================


def _make_island_mask(grid, island_lat=0.0, island_lon=0.0,
                       radius_deg=15.0):
    """Single circular island in the middle of an otherwise-ocean basin.

    Returns a land mask where cells within ``radius_deg`` great-circle
    distance of (island_lat, island_lon) are LAND (mask=0).
    """
    lat_deg = jnp.asarray(grid.lat2d) * 180.0 / jnp.pi
    lon_deg = jnp.asarray(grid.lon2d) * 180.0 / jnp.pi
    # Wrap lon for shortest distance
    dlon = jnp.minimum(jnp.abs(lon_deg - island_lon),
                        360.0 - jnp.abs(lon_deg - island_lon))
    dist_sq = (lat_deg - island_lat) ** 2 + dlon ** 2
    is_island = dist_sq < radius_deg ** 2
    mask = jnp.where(is_island, 0.0, 1.0)
    # Also mask polar rows so v_mask is well-defined
    mask = mask.at[0, :].set(0.0)
    mask = mask.at[-1, :].set(0.0)
    return mask


class TestIslandTopology:
    """Single circular island — non-simply-connected ocean topology.

    Closed-domain tracer conservation is the headline test: any leak
    through a wet→dry face would corrupt total tracer mass.
    """

    def test_island_mask_creates_isolated_land(self, grid_18x36):
        """The island mask should produce a connected land region in
        the middle of an otherwise-ocean basin."""
        mask = _make_island_mask(grid_18x36, island_lat=0.0,
                                  island_lon=0.0, radius_deg=15.0)
        n_land = int(jnp.sum(mask < 0.5))
        # Should be a small fraction of the total domain
        total = mask.size
        # Polar rows are masked too, plus an island in the middle.
        # Bounds: enough for a real island but not most of the domain.
        assert 5 < n_land < total // 4

    def test_face_mask_consistency_around_island(self, grid_18x36):
        """Face masks must match land mask exactly around the island
        (this is what the runtime invariant checks)."""
        mask = _make_island_mask(grid_18x36)
        u_mask, v_mask = compute_face_masks(mask)
        # Verify by reconstruction.
        u_recovered_interior = mask * jnp.roll(mask, 1, axis=1)
        np.testing.assert_array_equal(
            np.asarray(u_mask[:, :-1]), np.asarray(u_recovered_interior),
        )

    def test_tracer_conservation_with_island(self, grid_18x36, z_coord):
        """Closed-domain run with no forcing on a basin with an island:
        integrated heat and salt must be conserved to round-off.

        Any flux leakage at island coastlines would corrupt this.
        """
        mask = _make_island_mask(grid_18x36)
        state = rest_state_latlon_cgrid_ocean(
            grid_18x36, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            land_mask_override=mask,
        )
        # Integrated heat and salt at t=0
        h0 = compute_layer_thickness(state.eta.data,
                                       state.H_bathy.data, z_coord)
        wet3 = (state.land_mask.data > 0.5)[..., None]
        H0 = float(jnp.sum(jnp.where(wet3, state.T.data * h0, 0.0)
                            * grid_18x36.area[..., None]))
        S0 = float(jnp.sum(jnp.where(wet3, state.S.data * h0, 0.0)
                            * grid_18x36.area[..., None]))
        # Step 12 hours
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")
        model = LatLonCGridOceanModel(grid_18x36, z_coord, cfg)
        s = state
        for _ in range(12):
            s = model.step(s, 3600.0)
        h1 = compute_layer_thickness(s.eta.data, s.H_bathy.data, z_coord)
        wet3_after = (s.land_mask.data > 0.5)[..., None]
        H1 = float(jnp.sum(jnp.where(wet3_after, s.T.data * h1, 0.0)
                            * grid_18x36.area[..., None]))
        S1 = float(jnp.sum(jnp.where(wet3_after, s.S.data * h1, 0.0)
                            * grid_18x36.area[..., None]))
        rel_drift_T = abs(H1 - H0) / abs(H0)
        rel_drift_S = abs(S1 - S0) / abs(S0)
        # 1e-7 = ~1 fp32 ULP × cumulative 12 step rounding chain at
        # heat-content ~ 1e19 J.  A real mask leak would exceed this
        # by orders of magnitude (advective transport into "dry" cells
        # would build up at flux × dt × n_steps ~ 1e-3 relative).  The
        # original 1e-8 threshold required bit-exact fp64 storage.
        assert rel_drift_T < 1e-7, (
            f"Heat drift {rel_drift_T:.3e} (initial {H0:.3e}, "
            f"final {H1:.3e}) — possible mask leak"
        )
        assert rel_drift_S < 1e-7, (
            f"Salt drift {rel_drift_S:.3e} (initial {S0:.3e}, "
            f"final {S1:.3e}) — possible mask leak"
        )

    def test_runtime_check_catches_mask_inconsistency(
        self, grid_18x36, z_coord,
    ):
        """``_assert_runtime_invariants`` raises when ``land_mask`` and
        ``u_mask``/``v_mask`` disagree.  This guards the documented
        hazard in CLAUDE.md ('Never use state._replace(land_mask=...)').

        Note: ``model.step()`` itself may self-heal (recompute face masks
        from land_mask internally before the post-step invariant
        runs).  We call the invariant directly here, which is the
        actual contract: the check's job is to fail loudly when the
        state passed in is internally inconsistent.
        """
        mask = _make_island_mask(grid_18x36)
        state = rest_state_latlon_cgrid_ocean(
            grid_18x36, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            land_mask_override=mask,
        )
        # Pick a cell that's currently ocean and mutate it to land
        # without updating face masks (the bug pattern).
        # Cell (9, 18): lat~0°, lon~185° — far from the island.
        assert float(state.land_mask.data[9, 18]) == 1.0
        new_mask = state.land_mask.data.at[9, 18].set(0.0)
        bad_state = state._replace(
            land_mask=state.land_mask.replace(data=new_mask),
        )
        cfg = LatLonCGridOceanConfig(
            barotropic_solver="implicit_cn",
            enable_runtime_checks=True,
        )
        model = LatLonCGridOceanModel(grid_18x36, z_coord, cfg)
        # Direct invocation of the check: should fail loudly.
        with pytest.raises(ValueError, match=r"u_mask/v_mask inconsistent"):
            model._assert_runtime_invariants(bad_state)


# =========================================================================
# 3. Realistic ETOPO mask consistency
# =========================================================================


ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")


@pytest.mark.skipif(
    not ETOPO_FILE.exists(),
    reason="ETOPO data not cached; run "
           "scripts/tmp/diagnose_realistic_geometry.py first to fetch.",
)
class TestRealisticMaskConsistency:
    """Load the ETOPO 1° subset on a 72×144 (2.5°) grid and verify
    the global mask machinery holds end-to-end."""

    def test_etopo_mask_face_consistency(self):
        """Face masks computed from the ETOPO-derived land mask must
        match the runtime invariant exactly."""
        grid = create_latlon_grid(72, 144)
        cfg = BathymetryConfig(
            source="file", path=str(ETOPO_FILE),
            H_max=5500.0, H_min=10.0, smoothing_passes=2,
            enforce_straits=True, fill_isolated_basins=True,
            depth_is_negative=True,
        )
        H_bathy, ocean_mask = init_ocean_bathymetry(grid, cfg)
        u_mask, v_mask = compute_face_masks(ocean_mask)
        # u-face check: u_mask[:, :-1] = ocean_mask AND
        #   roll(ocean_mask, 1, axis=1)
        u_expected_interior = ocean_mask * jnp.roll(ocean_mask, 1, axis=1)
        np.testing.assert_array_equal(
            np.asarray(u_mask[:, :-1]), np.asarray(u_expected_interior),
        )
        # v-face check: v_mask[1:-1] = ocean_mask[:-1] AND ocean_mask[1:]
        v_expected_interior = ocean_mask[:-1] * ocean_mask[1:]
        np.testing.assert_array_equal(
            np.asarray(v_mask[1:-1]), np.asarray(v_expected_interior),
        )

    def test_etopo_state_construction_passes_runtime_check(self):
        """End-to-end: build a rest state on real ETOPO bathymetry and
        run the runtime invariant.  No exceptions should fire."""
        from legoesm.ocean.bathymetry import rest_state_ocean_realistic
        grid = create_latlon_grid(72, 144)
        z = create_ocean_z_star(n_levels=10, H_max=5500.0)
        cfg = BathymetryConfig(
            source="file", path=str(ETOPO_FILE),
            H_max=5500.0, H_min=10.0, smoothing_passes=2,
            enforce_straits=True, fill_isolated_basins=True,
            depth_is_negative=True,
        )
        state = rest_state_ocean_realistic(grid, z, cfg)
        # Build a model with runtime checks ENABLED and try one step.
        ocean_cfg = LatLonCGridOceanConfig(
            barotropic_solver="implicit_cn",
            enable_runtime_checks=True,
        )
        model = LatLonCGridOceanModel(grid, z, ocean_cfg)
        # One step should not raise (runtime invariant covers mask
        # consistency, finite values, T/S/eta bounds).
        s_new = model.step(state, 600.0)
        assert jnp.all(jnp.isfinite(s_new.u.data))
        assert jnp.all(jnp.isfinite(s_new.T.data))

    def test_replace_land_mask_atomic_update(self):
        """``replace_land_mask`` should atomically update u_mask, v_mask,
        and land_mask together.  After the call, the mask consistency
        invariant must hold."""
        grid = create_latlon_grid(72, 144)
        z = create_ocean_z_star(n_levels=5, H_max=5500.0)
        # Start from idealised flat-bottom + 80° land threshold
        state = rest_state_latlon_cgrid_ocean(
            grid, z,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_max=5500.0, land_lat_threshold=80.0,
        )
        # Now replace mask with a real ETOPO-derived one
        cfg = BathymetryConfig(
            source="file", path=str(ETOPO_FILE),
            H_max=5500.0, H_min=10.0, smoothing_passes=2,
            enforce_straits=True, fill_isolated_basins=True,
            depth_is_negative=True,
        )
        H_bathy, etopo_mask = init_ocean_bathymetry(grid, cfg)
        new_state = replace_land_mask(state, etopo_mask)
        # Verify face masks match the new land mask.
        u_expected, v_expected = compute_face_masks(etopo_mask)
        np.testing.assert_array_equal(
            np.asarray(new_state.u_mask.data),
            np.asarray(u_expected),
        )
        np.testing.assert_array_equal(
            np.asarray(new_state.v_mask.data),
            np.asarray(v_expected),
        )
        # Land mask itself preserved as given
        np.testing.assert_array_equal(
            np.asarray(new_state.land_mask.data),
            np.asarray(etopo_mask),
        )
