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
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
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
        cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn")

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
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
            H_bathy_override=H_bathy,
        )
        cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn")

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
        return LatLonCGridOceanConfig.from_flat(
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

    def _mixing_cfg(self):
        """K_v>0 + implicit vertical mixing — exercises the implicit tracer
        solve. Its per-cell layer-thickness weighting must conserve the PHYSICAL
        column heat sum(h_partial*T*area), not the dz_ref-weighted sum. All
        other tendencies disabled + conservation fixer off, so the only thing
        that can break sum(h*T) is the mixing's own thickness weighting."""
        return LatLonCGridOceanConfig.from_flat(
            barotropic_solver="implicit_cn",
            K_h=0.0, K_bih=0.0, A_h=0.0, B_h=0.0, C_smag=0.0,
            bottom_drag_r=0.0,
            K_v=1.0e-3, A_v=0.0,
            implicit_vertical_mixing=True,
            use_conservation_fixer=False,
            physics=None,
        )

    def test_partial_cells_implicit_mixing_conserves_heat(self, grid, z_coord):
        """Implicit vertical diffusion (K_v>0) over partial cells must conserve
        the physical column heat sum(h_partial*T*area). The backward-Euler solve
        with zero-flux BCs conserves sum(dz_cell*T); if dz_cell is the reference
        thickness dz_ref instead of the partial-cell h_partial at the thin bottom
        cell, it conserves the WRONG integral and leaks heat at the topography.
        At rest (u=0) the flux-form advection is conservative, so any sum(h*T)
        drift is the implicit mixing's thickness-weighting error."""
        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = _stratified_state_partial(grid, z_coord, H_bathy, partial_coord)
        cfg = self._mixing_cfg()
        model = LatLonCGridOceanModel(grid, partial_coord, cfg)

        m_T_0, _ = _column_integrated_tracer_mass(state, partial_coord, cfg, grid)
        s = model.step(state, 600.0)
        m_T_1, _ = _column_integrated_tracer_mass(s, partial_coord, cfg, grid)

        rel_drift_T = float(abs(m_T_1 - m_T_0) / max(abs(m_T_0), 1.0))
        assert rel_drift_T < 1e-12, (
            f"implicit-mixing heat drift = {rel_drift_T:.2e} "
            f"(partial-cell vertical mixing must weight by h_partial, not dz_ref)"
        )

    def test_zstar_step_bathymetry(self, grid, z_coord):
        """Same conservation property on legacy z* — the Phase 7
        face-thickness change should not regress this."""
        H_bathy = _step_bathy(grid)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
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


class TestNemoSshAvgFaceDepthGate:
    """#1226 zero-deviation item 2 machine gate: the DINO kamm barotropic
    composition (explicit_substep + nemo_boxcar_centred + een_metric +
    alpha=0, dynspg_ts.F90 zhU construction) must conserve volume and
    tracer mass to machine precision under BOTH face-depth modes.

    Why this closes: the substep loop builds the eta-update divergence and
    the ``Hu_avg`` accumulation from the SAME per-substep flux
    ``H_u_flux*U_mid`` (barotropic_latlon_cgrid.py substep_body), exactly
    as NEMO builds ssh and un_adv from the same zhU (dynspg_ts.F90:604-643);
    the tracer step then consumes ``Hu_avg`` directly through the
    Hallberg-Adcroft ``delta_U`` correction, which enforces
    ``sum_k(h_u_old*u_corrected) == Hu_avg`` identically for ANY face-depth
    weights (ocean_model_latlon_cgrid.py section 7) — the lego analogue of
    NEMO's barotropic-component replacement (dynspg_ts.F90:984-988).  So
    the face-depth mode changes WHAT Hu_avg is, never the closure.  This
    gate pins that machine-checkably (tolerances mirror
    TestTracerMassConservation / TestHallbergAdcroftColumnSumIdentity).

    Protocol mirrors the existing min-rule conservation tests:
    ``fix_eta_drift`` stays at its production default (True — the kamm
    cards run it), so total VOLUME is fixer-enforced; total TRACER mass is
    the structural claim (``use_conservation_fixer=False`` — nothing
    patches hT closure).

    Measured 2026-07-23 (this setup, 18x36, 3x600s steps): the kamm
    composition carries a PRE-EXISTING, face-depth-INDEPENDENT tracer-mass
    residual ~4.2e-10 — root cause is ``divergence_cgrid``'s global
    area-weighted sum not being identically zero (exact-spherical-cap cell
    ``area`` vs midpoint ``dx*dy`` face metrics, the mismatch documented in
    barotropic_latlon_cgrid.py; ~9e-12 relative for random flux), which the
    een_metric barotropic-Coriolis flux structure amplifies to ~6e-10/step
    pre-fixer (avg-Coriolis: ~1.6e-11); the eta-drift fixer then feeds it
    into hT via the h_k_new(eta_fixed)-vs-div(Hu_avg) bookkeeping.  It is
    NOT a face-depth effect: min_rule 4.236991e-10 vs nemo_ssh_avg
    4.236085e-10 (diff 9.1e-14).  Hence the two-part conservation gate:
    a per-mode absolute bound at the composition's measured level, plus
    MODE-INERTNESS at machine precision (the zero-deviation claim: flipping
    the face rule adds nothing to the conservation residual)."""

    # Conservation-relevant barotropic composition of the nemo_dino_kamm
    # (FE-frame) card — see DINO_RECIPES["nemo_dino_kamm"] in dino.py.
    _KAMM_BARO = dict(
        barotropic_solver="explicit_substep",
        barotropic_time_filter="nemo_boxcar_centred",
        barotropic_coriolis="een_metric",
        barotropic_diffusion_alpha=0.0,
        n_barotropic_substeps=23,
    )

    def _cfg(self, face_depth):
        """Pure flux-form transport only.  fix_eta_drift keeps its
        production default (True); use_conservation_fixer=False so tracer
        mass closure stays structural (see class docstring)."""
        return LatLonCGridOceanConfig.from_flat(
            K_h=0.0, K_bih=0.0, K_v=0.0,
            A_h=0.0, B_h=0.0, A_v=0.0,
            C_smag=0.0,
            bottom_drag_r=0.0,
            physics=None,
            use_conservation_fixer=False,
            barotropic_face_depth=face_depth,
            **self._KAMM_BARO,
        )

    def _perturbed_state(self, grid, partial_coord, H_bathy, z_coord, seed=7):
        """Stratified state + nonuniform eta + noise u/v (nonuniform eta is
        what makes min_rule and nemo_ssh_avg face depths actually differ)."""
        state = _stratified_state_partial(grid, z_coord, H_bathy, partial_coord)
        eta = (
            0.4 * jnp.sin(2.0 * grid.lon2d) * jnp.cos(3.0 * grid.lat2d)
        ) * state.land_mask.data
        key = jax.random.PRNGKey(seed)
        ku, kv = jax.random.split(key)
        u0 = 0.05 * jax.random.normal(ku, state.u.data.shape)
        v0 = 0.05 * jax.random.normal(kv, state.v.data.shape)
        return state._replace(
            eta=state.eta.replace(data=eta),
            u=state.u.replace(data=u0 * state.u_mask.data[..., jnp.newaxis]),
            v=state.v.replace(data=v0 * state.v_mask.data[..., jnp.newaxis]),
        )

    def test_volume_and_tracer_conservation_mode_inert(self, grid, z_coord):
        """Closed basin, no forcing, nonuniform eta+tracer, 3 model.step's
        under the kamm barotropic composition, BOTH face-depth modes:
        volume conserved to machine precision (fixer-enforced — sanity that
        the fixer holds under nemo_ssh_avg), tracer mass within the
        composition's measured pre-existing bound, and the min_rule vs
        nemo_ssh_avg residual DIFFERENCE at machine precision."""
        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = self._perturbed_state(grid, partial_coord, H_bathy, z_coord)

        area = grid.area
        maskd = state.land_mask.data
        vol_ref = float(jnp.sum(state.H_bathy.data * area * maskd))
        vol_0 = float(jnp.sum(state.eta.data * area * maskd))

        drifts = {}
        for face_depth in ("min_rule", "nemo_ssh_avg"):
            cfg = self._cfg(face_depth)
            model = LatLonCGridOceanModel(grid, partial_coord, cfg)
            m_T_0, m_S_0 = _column_integrated_tracer_mass(
                state, partial_coord, cfg, grid)
            s = state
            for _ in range(3):
                s = model.step(s, 600.0)
            vol_1 = float(jnp.sum(s.eta.data * area * maskd))
            m_T_1, m_S_1 = _column_integrated_tracer_mass(
                s, partial_coord, cfg, grid)
            rel_vol = abs(vol_1 - vol_0) / vol_ref
            rel_T = float(abs(m_T_1 - m_T_0) / max(abs(m_T_0), 1.0))
            rel_S = float(abs(m_S_1 - m_S_0) / max(abs(m_S_0), 1.0))
            assert rel_vol < 1e-12, (
                f"[{face_depth}] volume drift = {rel_vol:.2e}")
            # Pre-existing een_metric/divergence-metric residual (see class
            # docstring): measured 4.2e-10; 10x headroom.
            assert rel_T < 5e-9, f"[{face_depth}] T mass drift = {rel_T:.2e}"
            assert rel_S < 5e-9, f"[{face_depth}] S mass drift = {rel_S:.2e}"
            drifts[face_depth] = (rel_T, rel_S)

        # The zero-deviation claim: the face-depth mode contributes NOTHING
        # to the conservation residual (measured diff 9.1e-14; 50x headroom).
        dT = abs(drifts["nemo_ssh_avg"][0] - drifts["min_rule"][0])
        dS = abs(drifts["nemo_ssh_avg"][1] - drifts["min_rule"][1])
        # NB this bound rides on the pre-existing metric residual (class
        # docstring): if it trips, check the per-mode absolute drifts FIRST —
        # a grown divergence_cgrid residual looks like a face-depth regression.
        assert dT < 5e-12, (
            f"face-depth mode NOT conservation-inert: dT={dT:.2e} "
            f"(per-mode drifts: {drifts})")
        assert dS < 5e-12, (
            f"face-depth mode NOT conservation-inert: dS={dS:.2e} "
            f"(per-mode drifts: {drifts})")

    @pytest.mark.parametrize("time_filter", [
        "nemo_boxcar_centred",   # kamm FE card
        "nemo_boxcar_ab3",       # kamm MLF card (AB3 eta_mid flux-depth branch;
                                 # direct solver call — the MLF-frame guard
                                 # lives in the model wrapper, not the solver)
    ])
    @pytest.mark.parametrize("face_depth", ["min_rule", "nemo_ssh_avg"])
    def test_column_sum_identity_and_seafloor_w(
            self, grid, z_coord, face_depth, time_filter):
        """H&A column-sum identity + seafloor-w telescoping under both
        face-depth modes through the explicit substep solver: replicate the
        section-7 consumption (min-rule h_u_old + delta_U correction) on the
        solver's Hu_avg and assert the invariant + w(seafloor)=0."""
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_substeps_latlon_cgrid,
        )

        H_bathy = _step_bathy(grid)
        partial_coord = create_partial_cell_coordinate(z_coord, H_bathy)
        state = self._perturbed_state(grid, partial_coord, H_bathy, z_coord)
        cfg = self._cfg(face_depth)
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_time_filter=time_filter))

        h_k_old = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, partial_coord,
            min_water_column_m=cfg.min_water_column_m,
        )
        n = 23
        state_new, (Hu_avg, Hv_avg) = barotropic_substeps_latlon_cgrid(
            state, 600.0 / n, n, grid, partial_coord, cfg,
            add_barotropic_coriolis=True,
        )

        # Section-7 consumption (ocean_model_latlon_cgrid.py): min-rule
        # per-layer face weights + uniform barotropic correction.
        h_u_old = min_cell_to_uface(h_k_old)
        h_v_old = min_cell_to_vface(h_k_old)
        # grid passed to match production section 7 exactly (fold/seam-wall
        # handling — identical on this regular grid, load-bearing on tripolar).
        u_mask_3d, v_mask_3d = compute_face_masks_3d(
            partial_coord.is_active, grid)
        u_mask_3d = u_mask_3d.astype(h_u_old.dtype)
        v_mask_3d = v_mask_3d.astype(h_v_old.dtype)

        u_3d = state_new.u.data
        v_3d = state_new.v.data
        H_u_old = jnp.sum(h_u_old, axis=-1)
        H_v_old = jnp.sum(h_v_old, axis=-1)
        Hu_3d = jnp.sum(u_3d * h_u_old, axis=-1)
        Hv_3d = jnp.sum(v_3d * h_v_old, axis=-1)
        delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
        delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
        u_corr = u_3d + delta_U[..., jnp.newaxis]
        v_corr = v_3d + delta_V[..., jnp.newaxis]

        Hu_from_corr = jnp.sum(h_u_old * u_corr * u_mask_3d, axis=-1)
        Hv_from_corr = jnp.sum(h_v_old * v_corr * v_mask_3d, axis=-1)
        np.testing.assert_allclose(
            np.asarray(Hu_from_corr), np.asarray(Hu_avg), rtol=0, atol=1e-12,
            err_msg=f"[{face_depth}] H&A column-sum identity violated for u",
        )
        np.testing.assert_allclose(
            np.asarray(Hv_from_corr), np.asarray(Hv_avg), rtol=0, atol=1e-12,
            err_msg=f"[{face_depth}] H&A column-sum identity violated for v",
        )

        mass_flux_u = h_u_old * u_corr * u_mask_3d
        mass_flux_v = h_v_old * v_corr * v_mask_3d
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
        w_baro = diagnose_w_from_flux_div(
            flux_div_k, partial_coord, thickness_weighted=True,
        )
        bot = partial_coord.bottom_level
        ii = jnp.arange(grid.n_lat)[:, None]
        jj = jnp.arange(grid.n_lon)[None, :]
        max_w = float(jnp.max(jnp.abs(w_baro[ii, jj, bot + 1])))
        assert max_w < 1e-10, (
            f"[{face_depth}] w at partial seafloor not zero: {max_w:.3e} m/s"
        )


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
        cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn")

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
