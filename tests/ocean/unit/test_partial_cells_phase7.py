"""Phase 7 invariant tests: face-thickness consistency, column-sum
identity, and tracer mass conservation on partial cells.

These are the gates the audits demanded but Phase 6 lacked:

1. Hallberg & Adcroft 2009 column-sum identity:
   ``sum_k(h_u_old * u_corrected_k * u_mask_3d) == Hu_avg`` after the
   model.step's barotropic correction.  This is the consistency
   condition that links the implicit-CN barotropic transport with the
   per-layer mass flux divergence used for tracers.  Mismatched face-
   thickness conventions across the step would break it.

2. Tracer mass conservation on stepped bathymetry: with no surface
   forcing or sponging, ``sum(h*T*area)`` and ``sum(h*S*area)`` over
   the wet domain should be bit-conserved across one step.  Catches
   silent leaks at topographic-step partial faces (the kind of bug
   Phase 7 fixed but had no regression gate for).

3. Diagnosed vertical velocity at the partial seafloor: ``w_baro`` at
   the bottom interface of every column must be machine-zero.  If
   the per-level mass flux divergence is consistent with the
   barotropic continuity equation, the cumsum from the surface
   telescopes to zero at the seafloor automatically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface,
    min_cell_to_vface,
    compute_face_masks_3d,
    divergence_cgrid,
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
    diagnose_w_from_flux_div,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy


@pytest.fixture(autouse=True)
def _enable_x64_fp64():
    """These tests check column-sum and conservation identities at
    machine precision.  The default fp32 compute precision masks the
    invariants behind 1e-7 relative round-off.  Force fp64 throughout."""
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


def _step_bathy(grid):
    H = jnp.full((grid.n_lat, grid.n_lon), 4000.0)
    H = H.at[: grid.n_lat // 2, :].set(800.0)
    return H


def _stratified_state_partial(grid, z_coord, H_bathy, partial_coord):
    """Centroid-aware stratified rest state on partial cells."""
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
# 1. Hallberg-Adcroft 2009 column-sum identity
# ---------------------------------------------------------------------------


class TestHallbergAdcroftColumnSumIdentity:
    """``sum_k(h_u_old * u_corrected_k * u_mask_3d_tracer) == Hu_avg``
    after the barotropic correction.  This is the canonical
    Hallberg-Adcroft 2009 invariant linking the implicit-CN barotropic
    transport to the per-layer tracer mass flux."""

    def _replicate_step_through_correction(
        self, grid, z_coord, state, dt, cfg,
    ):
        """Replicate model.step up through the post-barotropic delta_U
        correction.  Returns ``(h_u_old, h_v_old, u_corrected,
        v_corrected, u_mask_3d_tracer, v_mask_3d_tracer, Hu_avg, Hv_avg)``.

        Mirrors ``LatLonCGridOceanModel.step`` lines 357-545 but
        without tracer transport, so we can directly assert the
        column-sum identity.
        """
        from legoesm.ocean.vertical import OceanPartialCellCoordinate
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
        H_u_pre = jnp.maximum(jnp.sum(h_u_pre, axis=-1), 1e-10)
        h_v_pre = min_cell_to_vface(h_k_pre)
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

        if isinstance(z_coord, OceanPartialCellCoordinate):
            u_mask_3d_tracer, v_mask_3d_tracer = compute_face_masks_3d(
                z_coord.is_active,
            )
            u_mask_3d_tracer = u_mask_3d_tracer.astype(h_u_old.dtype)
            v_mask_3d_tracer = v_mask_3d_tracer.astype(h_v_old.dtype)
        else:
            u_mask_3d_tracer = state.u_mask.data[..., jnp.newaxis]
            v_mask_3d_tracer = state.v_mask.data[..., jnp.newaxis]

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

    def test_partial_cells_step_bathymetry(self, grid, z_coord):
        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_state_partial(grid, z_coord, H_bathy, partial_coord)
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")

        (h_u_old, h_v_old, u_corr, v_corr, u_mask_3d, v_mask_3d,
         Hu_avg, Hv_avg) = self._replicate_step_through_correction(
            grid, partial_coord, state, 600.0, cfg,
        )

        # Per-face: column sum of (h * u_corr * face_mask) must equal Hu_avg.
        Hu_from_corrected = jnp.sum(h_u_old * u_corr * u_mask_3d, axis=-1)
        Hv_from_corrected = jnp.sum(h_v_old * v_corr * v_mask_3d, axis=-1)
        np.testing.assert_allclose(
            np.asarray(Hu_from_corrected), np.asarray(Hu_avg),
            rtol=0, atol=1e-12,
            err_msg="H&A 2009 column-sum identity violated for u",
        )
        np.testing.assert_allclose(
            np.asarray(Hv_from_corrected), np.asarray(Hv_avg),
            rtol=0, atol=1e-12,
            err_msg="H&A 2009 column-sum identity violated for v",
        )

    def test_zstar_flat_bottom(self, grid, z_coord):
        H_bathy = jnp.full((grid.n_lat, grid.n_lon), z_coord.H_max)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")

        (h_u_old, h_v_old, u_corr, v_corr, u_mask_3d, v_mask_3d,
         Hu_avg, Hv_avg) = self._replicate_step_through_correction(
            grid, z_coord, state, 600.0, cfg,
        )

        Hu_from_corrected = jnp.sum(h_u_old * u_corr * u_mask_3d, axis=-1)
        Hv_from_corrected = jnp.sum(h_v_old * v_corr * v_mask_3d, axis=-1)
        np.testing.assert_allclose(
            np.asarray(Hu_from_corrected), np.asarray(Hu_avg),
            rtol=0, atol=1e-12,
            err_msg="Column-sum identity violated for u (z*, flat)",
        )
        np.testing.assert_allclose(
            np.asarray(Hv_from_corrected), np.asarray(Hv_avg),
            rtol=0, atol=1e-12,
            err_msg="Column-sum identity violated for v (z*, flat)",
        )


# ---------------------------------------------------------------------------
# 2. Tracer mass conservation across a step on stepped bathymetry
# ---------------------------------------------------------------------------


def _column_integrated_tracer_mass(state, z_coord, cfg, grid):
    """``sum(h * tracer * area)`` over the wet 3D domain."""
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord,
        min_water_column_m=cfg.min_water_column_m,
    )
    area = grid.area  # (n_lat, n_lon)
    mass_T = jnp.sum(h_k * state.T.data * area[..., jnp.newaxis])
    mass_S = jnp.sum(h_k * state.S.data * area[..., jnp.newaxis])
    return mass_T, mass_S


class TestTracerMassConservation:
    """``sum(h*T*area)`` and ``sum(h*S*area)`` are conserved across a
    step on stepped bathymetry under no surface forcing.  Bit-exact
    in z\\*; expect float-precision drift on partial cells (the
    flux-form update reorders summations across topographic steps,
    so cancellation can leave O(1e-13 relative) residuals)."""

    def _no_diffusion_cfg(self):
        """Config with all non-flux-form tendencies disabled: pure
        flux-form transport is the only thing that can change the
        column-integrated tracer mass.  Conservation should then hold
        to machine precision."""
        return LatLonCGridOceanConfig(
            barotropic_solver="implicit_cn",
            K_h=0.0, K_bih=0.0, K_v=0.0,
            A_h=0.0, B_h=0.0, A_v=0.0,
            C_smag=0.0,
            bottom_drag_r=0.0,
            physics=None,
        )

    def test_partial_cells_step_bathymetry(self, grid, z_coord):
        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_state_partial(grid, z_coord, H_bathy, partial_coord)
        cfg = self._no_diffusion_cfg()
        model = LatLonCGridOceanModel(grid, partial_coord, cfg)

        m_T_0, m_S_0 = _column_integrated_tracer_mass(state, partial_coord, cfg, grid)
        s = model.step(state, 600.0)
        m_T_1, m_S_1 = _column_integrated_tracer_mass(s, partial_coord, cfg, grid)

        rel_drift_T = float(abs(m_T_1 - m_T_0) / max(abs(m_T_0), 1.0))
        rel_drift_S = float(abs(m_S_1 - m_S_0) / max(abs(m_S_0), 1.0))
        assert rel_drift_T < 1e-12, f"T mass drift = {rel_drift_T:.2e}"
        assert rel_drift_S < 1e-12, f"S mass drift = {rel_drift_S:.2e}"

    def test_zstar_step_bathymetry(self, grid, z_coord):
        """Same conservation property on legacy z* — the Phase 7
        face-thickness change should not regress this."""
        H_bathy = _step_bathy(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        cfg = self._no_diffusion_cfg()
        model = LatLonCGridOceanModel(grid, z_coord, cfg)

        m_T_0, m_S_0 = _column_integrated_tracer_mass(state, z_coord, cfg, grid)
        s = model.step(state, 600.0)
        m_T_1, m_S_1 = _column_integrated_tracer_mass(s, z_coord, cfg, grid)

        rel_drift_T = float(abs(m_T_1 - m_T_0) / max(abs(m_T_0), 1.0))
        rel_drift_S = float(abs(m_S_1 - m_S_0) / max(abs(m_S_0), 1.0))
        assert rel_drift_T < 1e-12, f"T mass drift = {rel_drift_T:.2e}"
        assert rel_drift_S < 1e-12, f"S mass drift = {rel_drift_S:.2e}"


# ---------------------------------------------------------------------------
# 3. Diagnosed w at the partial seafloor is exactly zero
# ---------------------------------------------------------------------------


class TestVerticalVelocityAtPartialSeafloor:
    """The cumsum from the surface of a consistent per-level mass flux
    divergence telescopes to zero at the seafloor.  Verifies the
    partial-cell mass flux convention through the full pipeline."""

    def test_w_zero_at_partial_seafloor(self, grid, z_coord):
        """Test that diagnostically computed w from the mass-consistent
        velocity (u_corrected) is zero at the seafloor.  IMPORTANT:
        ``state.u`` post-step is the pre-correction velocity; we must
        replicate the model.step's ``delta_U`` correction to get the
        velocity that satisfies mass continuity with eta_new."""
        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_state_partial(grid, z_coord, H_bathy, partial_coord)
        cfg = LatLonCGridOceanConfig(barotropic_solver="implicit_cn")

        # Re-use the column-sum identity helper to get u_corrected,
        # which is the mass-consistent velocity used internally by the
        # tracer step.
        helper = TestHallbergAdcroftColumnSumIdentity()
        (h_u_old, h_v_old, u_corr, v_corr, u_mask_3d_tracer,
         v_mask_3d_tracer, Hu_avg, Hv_avg) = (
            helper._replicate_step_through_correction(
                grid, partial_coord, state, 600.0, cfg,
            )
        )

        mass_flux_u = h_u_old * u_corr * u_mask_3d_tracer
        mass_flux_v = h_v_old * v_corr * v_mask_3d_tracer
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
        w_baro = diagnose_w_from_flux_div(
            flux_div_k, partial_coord, thickness_weighted=True,
        )

        # w at the partial seafloor: half-level just below the
        # bottom_level cell.  w_baro shape: (n_lat, n_lon, nlev+1).
        n_lev = z_coord.n_levels
        bot = partial_coord.bottom_level   # (n_lat, n_lon)
        seafloor_idx = bot + 1             # (n_lat, n_lon), in [0, nlev]
        # Gather along the level axis using fancy indexing.
        ii = jnp.arange(grid.n_lat)[:, None]
        jj = jnp.arange(grid.n_lon)[None, :]
        w_at_seafloor = w_baro[ii, jj, seafloor_idx]
        max_w = float(jnp.max(jnp.abs(w_at_seafloor)))
        assert max_w < 1e-10, (
            f"w at partial seafloor not zero: max|w| = {max_w:.3e} m/s"
        )
