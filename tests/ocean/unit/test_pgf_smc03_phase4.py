"""Phase 4 of the density-Jacobian PGF (SMC03).

Wires ``pgf_scheme="smc03"`` into the lat-lon C-grid PE pipeline.
This phase has no new operators — it only adds a config dispatch
between the existing Adcroft path and the SMC03 path implemented in
Phase 3.

The four gates are:

1. **Default unchanged**: with ``pgf_scheme="adcroft"`` (the default)
   the pipeline is bit-exact identical to its pre-SMC03 behaviour
   for any coordinate.  This is the backward-compat contract from
   plan §5.
2. **Pure-z\\* unaffected**: setting ``pgf_scheme="smc03"`` on a
   ``OceanZStarCoordinate`` is a no-op (the SMC03 branch only fires
   for ``OceanPartialCellCoordinate``).  Bit-exact w.r.t. the
   Adcroft default.
3. **SMC03 active on partial cells**: the SMC03 path produces
   non-trivially different PGF from the Adcroft path on a stepped
   bathymetry (sanity check that the dispatch is wired in).
4. **Hallberg-Adcroft column-sum identity preserved**: the H&A 2009
   barotropic-baroclinic consistency invariant is preserved when
   running with ``pgf_scheme="smc03"`` — SMC03 changes only the PGF,
   not the barotropic transport / tracer mass-flux discretization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    barotropic_implicit_latlon_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
    compute_layer_thickness,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy


@pytest.fixture(autouse=True)
def _enable_x64_fp64():
    orig_x64 = jax.config.jax_enable_x64
    orig_policy = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_policy)
    jax.config.update("jax_enable_x64", orig_x64)


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=18, n_lon=36)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(
        n_levels=10, H_max=4000.0, dz_surface=10.0, dz_deep=500.0,
    )


def _step_bathy(grid, H_deep=4000.0, H_shallow=800.0):
    H = jnp.full((grid.n_lat, grid.n_lon), H_deep)
    H = H.at[: grid.n_lat // 2, :].set(H_shallow)
    return H


def _stratified_partial_state(grid, z_coord, H_bathy, partial_coord):
    """Centroid-aware exponential thermocline on partial cells."""
    centroid = compute_centroid_depth(
        jnp.zeros_like(H_bathy), H_bathy, partial_coord,
    )
    T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
    T_per_cell = jnp.where(partial_coord.is_active, T_per_cell, 2.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy,
    )
    return state._replace(T=state.T.replace(data=T_per_cell))


# ---------------------------------------------------------------------------
# 1. Default unchanged: adcroft is bit-exact pre-SMC03
# ---------------------------------------------------------------------------


class TestDefaultUnchanged:
    """``LatLonCGridOceanConfig()`` defaults to ``pgf_scheme="adcroft"``;
    explicit ``pgf_scheme="adcroft"`` matches the implicit default
    bit-exactly.  A trivial smoke test — the contract is that any
    user not setting the new field sees no behaviour change."""

    def test_default_is_adcroft(self):
        cfg = LatLonCGridOceanConfig()
        assert cfg.pgf_scheme == "adcroft"

    def test_explicit_adcroft_matches_default(self, grid, z_coord):
        H_bathy = _step_bathy(grid)
        partial = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_partial_state(grid, z_coord, H_bathy, partial)

        cfg_default = LatLonCGridOceanConfig()
        cfg_explicit = LatLonCGridOceanConfig(pgf_scheme="adcroft")

        tend_default = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial, cfg_default,
        )
        tend_explicit = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial, cfg_explicit,
        )
        np.testing.assert_array_equal(
            np.asarray(tend_default.du_dt.data),
            np.asarray(tend_explicit.du_dt.data),
        )
        np.testing.assert_array_equal(
            np.asarray(tend_default.dv_dt.data),
            np.asarray(tend_explicit.dv_dt.data),
        )


# ---------------------------------------------------------------------------
# 2. Pure-z* path unaffected by SMC03 setting
# ---------------------------------------------------------------------------


class TestZStarUnaffected:
    """SMC03 only fires when ``isinstance(z_coord, OceanPartialCellCoordinate)``.
    On a pure-z\\* coordinate the dispatch falls through and the
    Adcroft branch is also skipped (centroids align).  Setting
    ``pgf_scheme="smc03"`` on z\\* therefore has zero effect."""

    def test_zstar_smc03_equals_adcroft(self, grid, z_coord):
        H_bathy = jnp.full((grid.n_lat, grid.n_lon), z_coord.H_max)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )

        tend_a = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, LatLonCGridOceanConfig(pgf_scheme="adcroft"),
        )
        tend_s = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, LatLonCGridOceanConfig(pgf_scheme="smc03"),
        )
        np.testing.assert_array_equal(
            np.asarray(tend_a.du_dt.data), np.asarray(tend_s.du_dt.data),
        )
        np.testing.assert_array_equal(
            np.asarray(tend_a.dv_dt.data), np.asarray(tend_s.dv_dt.data),
        )


# ---------------------------------------------------------------------------
# 3. SMC03 produces different PGF on partial cells
# ---------------------------------------------------------------------------


class TestSMC03ActiveOnPartialCells:
    """Sanity check that SMC03 is actually wired in: on a stepped
    bathymetry with partial cells, the tendencies under
    ``pgf_scheme="smc03"`` differ non-trivially from the Adcroft
    baseline.  We don't claim either is "right" here — that's the
    Phase 5 BH stress test.  Just that the dispatch is active."""

    def test_smc03_differs_from_adcroft(self, grid, z_coord):
        H_bathy = _step_bathy(grid)
        partial = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_partial_state(grid, z_coord, H_bathy, partial)

        tend_a = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial, LatLonCGridOceanConfig(pgf_scheme="adcroft"),
        )
        tend_s = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, partial, LatLonCGridOceanConfig(pgf_scheme="smc03"),
        )
        # The bathymetry varies in latitude only (step at the equator),
        # so x-faces see identical conditions on either side and the
        # x-PGF is zero in both schemes.  The y-direction picks up the
        # step — that is where SMC03 vs Adcroft must disagree.
        diff_v = float(jnp.max(jnp.abs(
            tend_a.dv_dt.data - tend_s.dv_dt.data
        )))
        assert diff_v > 0.0, "SMC03 dispatch produced identical dv/dt to Adcroft"
        assert jnp.isfinite(jnp.asarray(diff_v))
        # SMC03 should also produce a *smaller* rest-state PGF residual
        # than Adcroft on a centroid-aware stratified state.  This is
        # the qualitative gain we're after — Phase 5 quantifies it on
        # the BH seamount.
        smc_dv_max = float(jnp.max(jnp.abs(tend_s.dv_dt.data)))
        adc_dv_max = float(jnp.max(jnp.abs(tend_a.dv_dt.data)))
        assert smc_dv_max < adc_dv_max, (
            f"SMC03 dv/dt residual ({smc_dv_max:.3e}) not smaller than "
            f"Adcroft ({adc_dv_max:.3e})."
        )


# ---------------------------------------------------------------------------
# 4. Hallberg-Adcroft column-sum identity under SMC03
# ---------------------------------------------------------------------------


class TestHallbergAdcroftUnderSMC03:
    """The H&A 2009 column-sum identity ties the implicit-CN
    barotropic transport (``Hu_avg``) to the per-layer 3D velocity
    (``sum_k h_u·u·u_mask``).  SMC03 only changes the PGF — the
    barotropic-baroclinic split, the slow-forcing depth-average, and
    the tracer mass-flux face thickness are all unchanged.  So the
    invariant must hold under either ``pgf_scheme``.
    """

    def _step_through_correction(self, grid, z_coord, state, dt, cfg):
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            _forward_backward_coriolis_3d,
        )
        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        mask_3d = state.land_mask.data[..., jnp.newaxis]

        tend = latlon_cgrid_ocean_baroclinic_tendencies(state, grid, z_coord, cfg)
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data
        du_dt = tend.du_dt.data
        dv_dt = tend.dv_dt.data

        h_k_pre = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        h_u_pre = min_cell_to_uface(h_k_pre)
        h_v_pre = min_cell_to_vface(h_k_pre)
        H_u_pre = jnp.maximum(jnp.sum(h_u_pre, axis=-1), 1e-10)
        H_v_pre = jnp.maximum(jnp.sum(h_v_pre, axis=-1), 1e-10)
        F_slow_u = jnp.sum(du_dt * h_u_pre, axis=-1) / H_u_pre * state.u_mask.data
        F_slow_v = jnp.sum(dv_dt * h_v_pre, axis=-1) / H_v_pre * state.v_mask.data
        du_dt_pert = du_dt - F_slow_u[..., jnp.newaxis]
        dv_dt_pert = dv_dt - F_slow_v[..., jnp.newaxis]
        u_star = state.u.data + dt * du_dt_pert
        v_star = state.v.data + dt * dv_dt_pert
        u_star, v_star = _forward_backward_coriolis_3d(
            u_star, v_star, dt, grid, z_coord, cfg,
            state.u_mask.data, state.v_mask.data, state.land_mask.data,
            state.eta.data, state.H_bathy.data,
        )
        u_star = u_star.at[:, -1].set(u_star[:, 0])
        state_mid = state._replace(
            u=state.u.replace(data=u_star * u_mask_3d),
            v=state.v.replace(data=v_star * v_mask_3d),
            T=state.T.replace(data=T_new * mask_3d),
            S=state.S.replace(data=S_new * mask_3d),
        )
        h_k_old = compute_layer_thickness(
            state_mid.eta.data, state_mid.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        state_new, (Hu_avg, Hv_avg) = barotropic_implicit_latlon_cgrid(
            state_mid, dt, grid, z_coord, cfg,
            F_slow_eta=None, F_slow_u=F_slow_u, F_slow_v=F_slow_v,
        )
        h_u_old = min_cell_to_uface(h_k_old)
        h_v_old = min_cell_to_vface(h_k_old)
        H_u_old = jnp.sum(h_u_old, axis=-1)
        H_v_old = jnp.sum(h_v_old, axis=-1)

        u_mask_3d_tracer, v_mask_3d_tracer = compute_face_masks_3d(z_coord.is_active)
        u_mask_3d_tracer = u_mask_3d_tracer.astype(h_u_old.dtype)
        v_mask_3d_tracer = v_mask_3d_tracer.astype(h_v_old.dtype)

        u_3d = state_new.u.data
        v_3d = state_new.v.data
        Hu_3d = jnp.sum(u_3d * h_u_old, axis=-1)
        Hv_3d = jnp.sum(v_3d * h_v_old, axis=-1)
        delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
        delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
        u_corrected = u_3d + delta_U[..., jnp.newaxis]
        v_corrected = v_3d + delta_V[..., jnp.newaxis]

        return (
            h_u_old, h_v_old, u_corrected, v_corrected,
            u_mask_3d_tracer, v_mask_3d_tracer, Hu_avg, Hv_avg,
        )

    def test_column_sum_under_smc03(self, grid, z_coord):
        H_bathy = _step_bathy(grid)
        partial = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_partial_state(grid, z_coord, H_bathy, partial)
        cfg = LatLonCGridOceanConfig(
            barotropic_solver="implicit_cn", pgf_scheme="smc03",
        )

        (h_u_old, h_v_old, u_corr, v_corr, u_mask_3d, v_mask_3d,
         Hu_avg, Hv_avg) = self._step_through_correction(
            grid, partial, state, 600.0, cfg,
        )
        Hu_from_corrected = jnp.sum(h_u_old * u_corr * u_mask_3d, axis=-1)
        Hv_from_corrected = jnp.sum(h_v_old * v_corr * v_mask_3d, axis=-1)
        np.testing.assert_allclose(
            np.asarray(Hu_from_corrected), np.asarray(Hu_avg),
            rtol=0, atol=1e-12,
            err_msg="H&A column-sum violated under pgf_scheme=smc03 (u)",
        )
        np.testing.assert_allclose(
            np.asarray(Hv_from_corrected), np.asarray(Hv_avg),
            rtol=0, atol=1e-12,
            err_msg="H&A column-sum violated under pgf_scheme=smc03 (v)",
        )
