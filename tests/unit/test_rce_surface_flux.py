"""RCE surface flux composer tests — layout-agnostic across NH states."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.rce_surface_flux import (
    apply_rce_surface_fluxes,
    compose_rce_surface_scalar_tendencies,
    wind_speed_at_lowest_level_cs,
    wind_speed_at_lowest_level_plane,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    MPASNonHydrostaticState, NonHydrostaticState,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

NLEV = 6
NX, NY = 4, 4
NCUBE = 4   # cubed-sphere n per face
NCELLS = 12  # MPAS cells (synthetic)
NEDGES = 30  # MPAS edges (synthetic)


def _plane_state():
    grid = create_plane_grid(
        nx=NX, ny=NY, nlev=NLEV, dx=2_000.0, dy=2_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=10_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((NY, NX, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def _cs_state(hc):
    """Synthetic cubed-sphere NH state — shape (6, NCUBE, NCUBE, NLEV)."""
    shp_3d = (6, NCUBE, NCUBE, NLEV)
    shp_w = (6, NCUBE, NCUBE, NLEV + 1)
    shp_2d = (6, NCUBE, NCUBE)
    shp_tr = (6, NCUBE, NCUBE, NLEV, 3)
    zero = lambda s: jnp.zeros(s, dtype=jnp.float64)
    tracers_data = zero(shp_tr).at[..., 0].set(0.01)
    return NonHydrostaticState(
        u=Field(zero(shp_3d), name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(zero(shp_3d), name="v", dims=("face", "x", "y", "level"), units="m/s"),
        w=Field(zero(shp_w), name="w", dims=("face", "x", "y", "ilevel"), units="m/s"),
        theta_prime=Field(zero(shp_3d), name="theta_prime", dims=("face", "x", "y", "level"), units="K"),
        rho_prime=Field(zero(shp_3d), name="rho_prime", dims=("face", "x", "y", "level"), units="kg/m^3"),
        phis=Field(zero(shp_2d), name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
        tracers=Field(tracers_data, name="tracers", dims=("face", "x", "y", "level", "tracer"), units="kg/kg"),
    )


def _mpas_state(hc):
    """Synthetic MPAS NH state — shapes (nCells, NLEV) for cell-centred."""
    shp_cell_3d = (NCELLS, NLEV)
    shp_cell_w = (NCELLS, NLEV + 1)
    shp_cell_2d = (NCELLS,)
    shp_edge_3d = (NEDGES, NLEV)
    shp_tr = (NCELLS, NLEV, 3)
    zero = lambda s: jnp.zeros(s, dtype=jnp.float64)
    tracers_data = zero(shp_tr).at[..., 0].set(0.01)
    return MPASNonHydrostaticState(
        u=Field(zero(shp_edge_3d), name="u", dims=("edge", "level"), units="m/s"),
        w=Field(zero(shp_cell_w), name="w", dims=("cell", "ilevel"), units="m/s"),
        theta_prime=Field(zero(shp_cell_3d), name="theta_prime", dims=("cell", "level"), units="K"),
        rho_prime=Field(zero(shp_cell_3d), name="rho_prime", dims=("cell", "level"), units="kg/m^3"),
        phis=Field(zero(shp_cell_2d), name="phis", dims=("cell",), units="m^2/s^2"),
        tracers=Field(tracers_data, name="tracers", dims=("cell", "level", "tracer"), units="kg/kg"),
    )


def _T_sfc_q_sfc_wspd_plane():
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float64)
    q_sfc = jnp.full((NY, NX), 0.02, dtype=jnp.float64)
    wspd = jnp.full((NY, NX), 5.0, dtype=jnp.float64)
    return T_sfc, q_sfc, wspd


def _T_sfc_q_sfc_wspd_cs():
    T_sfc = jnp.full((6, NCUBE, NCUBE), 300.0, dtype=jnp.float64)
    q_sfc = jnp.full((6, NCUBE, NCUBE), 0.02, dtype=jnp.float64)
    wspd = jnp.full((6, NCUBE, NCUBE), 5.0, dtype=jnp.float64)
    return T_sfc, q_sfc, wspd


def _T_sfc_q_sfc_wspd_mpas():
    T_sfc = jnp.full((NCELLS,), 300.0, dtype=jnp.float64)
    q_sfc = jnp.full((NCELLS,), 0.02, dtype=jnp.float64)
    wspd = jnp.full((NCELLS,), 5.0, dtype=jnp.float64)
    return T_sfc, q_sfc, wspd


# Layout-agnostic correctness
def test_plane_dtheta_positive_when_T_sfc_above_T_atm():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    dth, _ = compose_rce_surface_scalar_tendencies(state, hc, T_sfc, q_sfc, wspd)
    assert dth.shape == (NY, NX)
    assert bool(jnp.all(dth > 0.0))


def test_plane_dqv_positive_when_q_sfc_above_q_atm():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    _, dqv = compose_rce_surface_scalar_tendencies(state, hc, T_sfc, q_sfc, wspd)
    assert dqv.shape == (NY, NX)
    assert bool(jnp.all(dqv > 0.0))


def test_cs_layout_works():
    _, _, hc, _ = _plane_state()
    state = _cs_state(hc)
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_cs()
    dth, dqv = compose_rce_surface_scalar_tendencies(state, hc, T_sfc, q_sfc, wspd)
    assert dth.shape == (6, NCUBE, NCUBE)
    assert dqv.shape == (6, NCUBE, NCUBE)
    assert bool(jnp.all(dth > 0.0))
    assert bool(jnp.all(dqv > 0.0))


def test_mpas_layout_works():
    _, _, hc, _ = _plane_state()
    state = _mpas_state(hc)
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_mpas()
    dth, dqv = compose_rce_surface_scalar_tendencies(state, hc, T_sfc, q_sfc, wspd)
    assert dth.shape == (NCELLS,)
    assert dqv.shape == (NCELLS,)
    assert bool(jnp.all(dth > 0.0))
    assert bool(jnp.all(dqv > 0.0))


def test_no_flux_when_T_sfc_equals_T_atm_and_q_sfc_equals_q_atm():
    """Zero ΔT + Δq → zero tendency."""
    state, _, hc, _ = _plane_state()
    # T_atm at k_sfc = theta_ref * exner_ref + theta_prime (= 0) → use that
    T_atm_sfc = float(hc.theta_ref[-1] * hc.exner_ref[-1])
    T_sfc = jnp.full((NY, NX), T_atm_sfc, dtype=jnp.float64)
    q_sfc = jnp.full((NY, NX), 0.01, dtype=jnp.float64)
    wspd = jnp.full((NY, NX), 5.0, dtype=jnp.float64)
    dth, dqv = compose_rce_surface_scalar_tendencies(state, hc, T_sfc, q_sfc, wspd)
    np.testing.assert_allclose(np.asarray(dth), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(dqv), 0.0, atol=1.0e-14)


def test_tendency_scales_with_wind_speed():
    """Linear scaling holds when gustiness floor disabled."""
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd_1 = _T_sfc_q_sfc_wspd_plane()
    wspd_2 = 2.0 * wspd_1
    dth_1, dqv_1 = compose_rce_surface_scalar_tendencies(
        state, hc, T_sfc, q_sfc, wspd_1, gustiness_floor=0.0,
    )
    dth_2, dqv_2 = compose_rce_surface_scalar_tendencies(
        state, hc, T_sfc, q_sfc, wspd_2, gustiness_floor=0.0,
    )
    np.testing.assert_allclose(np.asarray(dth_2 / dth_1), 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(dqv_2 / dqv_1), 2.0, rtol=1.0e-12)


def test_apply_only_touches_sfc_level_theta_and_qv():
    """All other prognostic fields + interior levels unchanged."""
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    new_state = apply_rce_surface_fluxes(state, hc, 60.0, T_sfc, q_sfc, wspd)
    # u, v, w, rho', phis, q_c, q_r untouched.
    np.testing.assert_array_equal(np.asarray(new_state.u.data), np.asarray(state.u.data))
    np.testing.assert_array_equal(np.asarray(new_state.v.data), np.asarray(state.v.data))
    np.testing.assert_array_equal(np.asarray(new_state.w.data), np.asarray(state.w.data))
    np.testing.assert_array_equal(np.asarray(new_state.rho_prime.data), np.asarray(state.rho_prime.data))
    np.testing.assert_array_equal(np.asarray(new_state.phis.data), np.asarray(state.phis.data))
    np.testing.assert_array_equal(
        np.asarray(new_state.tracers.data[..., 1]),
        np.asarray(state.tracers.data[..., 1]),
    )
    # Interior theta' levels unchanged.
    np.testing.assert_array_equal(
        np.asarray(new_state.theta_prime.data[..., :-1]),
        np.asarray(state.theta_prime.data[..., :-1]),
    )
    # Bottom level changed.
    assert bool(jnp.all(
        new_state.theta_prime.data[..., -1] != state.theta_prime.data[..., -1]
    ))
    # qv interior unchanged.
    np.testing.assert_array_equal(
        np.asarray(new_state.tracers.data[..., :-1, 0]),
        np.asarray(state.tracers.data[..., :-1, 0]),
    )


def test_apply_jit_compilable():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    fn = jax.jit(
        lambda s, T, q, w: apply_rce_surface_fluxes(s, hc, 60.0, T, q, w),
    )
    out = fn(state, T_sfc, q_sfc, wspd)
    assert bool(jnp.all(jnp.isfinite(out.theta_prime.data)))


def test_apply_supports_jax_grad():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()

    def loss_fn(T_sfc):
        new_state = apply_rce_surface_fluxes(state, hc, 60.0, T_sfc, q_sfc, wspd)
        return jnp.sum(new_state.theta_prime.data ** 2)

    g = jax.grad(loss_fn)(T_sfc)
    assert g.shape == T_sfc.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# Wind speed helpers — RAW (no gustiness floor, Codex iter-2)
def test_plane_wind_speed_zero_at_rest():
    state, _, hc, _ = _plane_state()
    wspd = wind_speed_at_lowest_level_plane(state)
    np.testing.assert_allclose(np.asarray(wspd), 0.0, atol=1.0e-14)


def test_plane_wind_speed_combines_u_v():
    state, _, hc, _ = _plane_state()
    u = jnp.zeros_like(state.u.data).at[..., -1].set(3.0)
    v = jnp.zeros_like(state.v.data).at[..., -1].set(4.0)
    state = state._replace(
        u=state.u.replace(data=u),
        v=state.v.replace(data=v),
    )
    wspd = wind_speed_at_lowest_level_plane(state)
    # sqrt(9 + 16) = 5
    np.testing.assert_allclose(np.asarray(wspd), 5.0, rtol=1.0e-12)


def test_composer_applies_gustiness_floor_to_zero_wind():
    """Codex iter-2: composer applies gustiness, so calm cell still
    fluxes heat / moisture."""
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, _ = _T_sfc_q_sfc_wspd_plane()
    wspd_zero = jnp.zeros((NY, NX), dtype=jnp.float64)
    dth, _ = compose_rce_surface_scalar_tendencies(
        state, hc, T_sfc, q_sfc, wspd_zero,
        gustiness_floor=5.0,
    )
    assert bool(jnp.all(dth > 0.0))


def test_composer_no_flux_when_gust_zero_and_wind_zero():
    """gustiness_floor=0 + raw wind=0 → no flux (degenerate-by-design)."""
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, _ = _T_sfc_q_sfc_wspd_plane()
    wspd_zero = jnp.zeros((NY, NX), dtype=jnp.float64)
    dth, dqv = compose_rce_surface_scalar_tendencies(
        state, hc, T_sfc, q_sfc, wspd_zero,
        gustiness_floor=0.0,
    )
    np.testing.assert_allclose(np.asarray(dth), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(dqv), 0.0, atol=1.0e-14)


def test_composer_rejects_negative_gustiness():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="non-negative"):
        compose_rce_surface_scalar_tendencies(
            state, hc, T_sfc, q_sfc, wspd,
            gustiness_floor=-1.0,
        )


def test_cs_wind_speed_rejects_plane_layout():
    state, _, hc, _ = _plane_state()
    with pytest.raises(ValueError, match="cubed-sphere"):
        wind_speed_at_lowest_level_cs(state)


def test_plane_wind_speed_rejects_cs_layout():
    _, _, hc, _ = _plane_state()
    state = _cs_state(hc)
    with pytest.raises(ValueError, match="plane"):
        wind_speed_at_lowest_level_plane(state)


# Validation
def test_compose_rejects_qv_slot_negative():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="out of range"):
        compose_rce_surface_scalar_tendencies(
            state, hc, T_sfc, q_sfc, wspd, qv_slot=-1,
        )


def test_compose_rejects_qv_slot_oob():
    state, _, hc, _ = _plane_state()
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="out of range"):
        compose_rce_surface_scalar_tendencies(
            state, hc, T_sfc, q_sfc, wspd, qv_slot=99,
        )


def test_compose_rejects_mismatched_rho_theta_shape():
    """Codex iter-2: full-shape mismatch (not just nlev) must raise."""
    state, _, hc, _ = _plane_state()
    bad_rho_data = state.rho_prime.data[..., :NLEV - 1]
    state_bad = state._replace(
        rho_prime=state.rho_prime.replace(data=bad_rho_data),
    )
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="rho_prime shape"):
        compose_rce_surface_scalar_tendencies(
            state_bad, hc, T_sfc, q_sfc, wspd,
        )


def test_compose_rejects_mismatched_horizontal_rho_shape():
    """Codex iter-2: rho with wrong HORIZONTAL layout must raise."""
    state, _, hc, _ = _plane_state()
    # rho_prime with (NY+1, NX, NLEV) — broadcasts silently w/o full check
    bad_rho_data = jnp.zeros((NY + 1, NX, NLEV), dtype=jnp.float64)
    state_bad = state._replace(
        rho_prime=state.rho_prime.replace(data=bad_rho_data),
    )
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="rho_prime shape"):
        compose_rce_surface_scalar_tendencies(
            state_bad, hc, T_sfc, q_sfc, wspd,
        )


def test_compose_rejects_mismatched_tracers_shape():
    """Codex iter-2: tracers with horizontal mismatch must raise."""
    state, _, hc, _ = _plane_state()
    bad_tr_data = jnp.zeros((NY + 1, NX, NLEV, 3), dtype=jnp.float64)
    state_bad = state._replace(
        tracers=state.tracers.replace(data=bad_tr_data),
    )
    T_sfc, q_sfc, wspd = _T_sfc_q_sfc_wspd_plane()
    with pytest.raises(ValueError, match="tracers shape"):
        compose_rce_surface_scalar_tendencies(
            state_bad, hc, T_sfc, q_sfc, wspd,
        )
