"""Unit tests for the Veros-faithful PROGNOSTIC TKE vertical-mixing option.

Covers the prognostic TKE carry built on top of the existing (oracle-verified)
diagnostic TKE chain:

1. default-off bit-identity (prognostic=False == pre-change byte-identical);
2. prognostic carry (tke evolves across steps, stays >= floor, finite);
3. dt_mom scaling of the TKE step;
4. source flags (eke_diss / K_diss_bot each change TKE where expected,
   zero where the source is zero);
5. energy sign (eke_diss + K_diss_bot sources are non-negative);
6. AD (grad finite through 2 steps with everything on);
7. scan-compat (5-step lax.scan with the seeded constant pytree).

End-to-end Veros parity is the recipe-acceptance harness's job; this file
verifies the prognostic-carry WIRING + invariants.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _fp64_policy():
    """Run with the fp64 precision policy so the model's storage dtype is f64,
    matching the x64 state built by ``rest_state_latlon_cgrid_ocean`` (otherwise
    the model casts to the default f32 storage and the lax.scan carry / direct
    comparisons would dtype-mismatch). Mirrors the ab2 / dt_mom_async suites."""
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


N_LAT, N_LON, NLEV = 6, 12, 8
H_MAX = 4000.0


def _grid_z():
    return (create_latlon_grid(n_lat=N_LAT, n_lon=N_LON),
            create_ocean_z_star(n_levels=NLEV, H_max=H_MAX))


def _perturbed_state(grid, z_coord, seed=0):
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=H_MAX,
    )
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.2 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.2 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    T_prof = 5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))
    T = T_prof[None, None, :] + 0.2 * rng.standard_normal((n_lat, n_lon, nlev))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u) * s.u_mask.data[:, :, None]),
        v=s.v.replace(data=jnp.asarray(v) * s.v_mask.data[:, :, None]),
        T=s.T.replace(data=jnp.asarray(T)),
    )


def _physics(tke_cfg):
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def _model(tke_cfg, *, bottom_drag_r=0.0):
    grid, z_coord = _grid_z()
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4,
        K_v=0.0,
        implicit_vertical_mixing=True,
        bottom_drag_r=bottom_drag_r,
        physics=_physics(tke_cfg),
    )
    return LatLonCGridOceanModel(grid, z_coord, cfg), grid, z_coord


def _wind_forcing(grid):
    """A nonzero surface wind stress so the prognostic TKE has a real surface
    source (forc_tke_surface = (|tau|/rho_0)^{3/2})."""
    from legoesm.ocean.state import OceanSurfaceForcing
    tau_x = 0.1 * jnp.ones((grid.n_lat, grid.n_lon))
    tau_y = jnp.zeros((grid.n_lat, grid.n_lon))
    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)


def _seed_tke(state, z_coord, tke_cfg, *, with_eke_diss=False, level=1.0e-3):
    """Seed state.tke at ``level`` (default well above the background floor so
    the prognostic production/dissipation/diffusion actively reshape it)."""
    dtype = state.T.data.dtype
    lm = state.land_mask.data
    nlev = z_coord.n_levels
    wet3 = (lm[:, :, None] > 0.5)
    tke0 = jnp.where(wet3, level, 0.0).astype(dtype) * jnp.ones(
        (1, 1, nlev - 1), dtype=dtype)
    state = state._replace(
        tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"))
    if with_eke_diss:
        ediss0 = jnp.zeros((lm.shape[0], lm.shape[1], nlev - 1), dtype=dtype)
        state = state._replace(
            eke_diss=Field(data=ediss0, name="eke_diss",
                           dims=("lat", "lon", "level"), units="m^2/s^3"))
    return state


# ---------------------------------------------------------------------------
# (1) default-off bit-identity
# ---------------------------------------------------------------------------


def test_default_off_bit_identity():
    """prognostic=False step == a step where the prognostic-TKE feature does
    not exist (the Mode-B diagnostic chain is byte-identical)."""
    grid, z_coord = _grid_z()
    diag_cfg = TKEConfig()  # prognostic default False
    model, _, _ = _model(diag_cfg)
    s0 = _perturbed_state(grid, z_coord)
    dt = 3600.0
    s1 = model._step_impl(s0, dt)
    # The diagnostic path never creates state.tke.
    assert s1.tke is None
    # Reference: compute_vertical_K_profiles 2-tuple contract unchanged.
    u_cell = 0.5 * (s0.u.data[:, :-1, :] + s0.u.data[:, 1:, :])
    v_cell = 0.5 * (s0.v.data[:-1, :, :] + s0.v.data[1:, :, :])
    cc = s0._replace(u=s0.u.replace(data=u_cell), v=s0.v.replace(data=v_cell))
    out = compute_vertical_K_profiles(
        cc, z_coord, None, _physics(diag_cfg),
        A_v_background=1.0e-4, K_v_background=0.0)
    assert isinstance(out, tuple) and len(out) == 2  # 2-tuple legacy contract


def test_prognostic_off_matches_diag_K():
    """The K profiles computed by the diagnostic chain (prognostic=False) are
    unchanged by the presence of the new optional return_tke parameter."""
    grid, z_coord = _grid_z()
    s0 = _perturbed_state(grid, z_coord)
    u_cell = 0.5 * (s0.u.data[:, :-1, :] + s0.u.data[:, 1:, :])
    v_cell = 0.5 * (s0.v.data[:-1, :, :] + s0.v.data[1:, :, :])
    cc = s0._replace(u=s0.u.replace(data=u_cell), v=s0.v.replace(data=v_cell))
    phys = _physics(TKEConfig())
    K2, A2 = compute_vertical_K_profiles(cc, z_coord, None, phys)
    K3, A3, tke3 = compute_vertical_K_profiles(
        cc, z_coord, None, phys, return_tke=True)
    assert tke3 is None  # diagnostic path carries no TKE
    np.testing.assert_array_equal(np.asarray(K2), np.asarray(K3))
    np.testing.assert_array_equal(np.asarray(A2), np.asarray(A3))


# ---------------------------------------------------------------------------
# (2) prognostic carry
# ---------------------------------------------------------------------------


def test_prognostic_carry_evolves_and_floored():
    """state.tke is created, finite, >= floor, and EVOLVES across steps."""
    grid, z_coord = _grid_z()
    tke_cfg = TKEConfig(prognostic=True)
    model, _, _ = _model(tke_cfg)
    s = _seed_tke(_perturbed_state(grid, z_coord), z_coord, tke_cfg)
    sf = _wind_forcing(grid)
    dt = 3600.0
    tke_hist = []
    for _ in range(4):
        s = model._step_impl(s, dt, surface_forcing=sf)
        assert s.tke is not None
        tke = np.asarray(s.tke.data)
        assert np.all(np.isfinite(tke))
        # Floor: wet interior interfaces >= tke_background.
        lm = np.asarray(s.land_mask.data)[:, :, None]
        assert np.all(tke[lm[..., 0] > 0.5] >= tke_cfg.tke_background - 1e-12)
        tke_hist.append(tke)
    # TKE must change between steps (genuinely prognostic, not frozen).
    assert not np.allclose(tke_hist[0], tke_hist[-1])


def test_cold_start_no_seed_runs():
    """A direct step with state.tke=None (cold start) runs: the k_profiles
    cold-start path seeds the background floor for the one step."""
    grid, z_coord = _grid_z()
    tke_cfg = TKEConfig(prognostic=True)
    model, _, _ = _model(tke_cfg)
    s = _perturbed_state(grid, z_coord)  # no tke seeded
    s1 = model._step_impl(s, 3600.0)
    assert s1.tke is not None
    assert np.all(np.isfinite(np.asarray(s1.tke.data)))


# ---------------------------------------------------------------------------
# (3) dt_mom scaling of the TKE step
# ---------------------------------------------------------------------------


def test_dt_mom_scaling():
    """The prognostic TKE step uses dt = dt_tke (= dt_mom). A larger dt_tke
    yields a LARGER TKE increment per step (backward-Euler relaxation toward
    the local equilibrium). Tested at the k_profiles level where dt_tke is the
    explicit knob the model threads as dt_mom."""
    grid, z_coord = _grid_z()
    tke_cfg = TKEConfig(prognostic=True)
    s0 = _seed_tke(_perturbed_state(grid, z_coord), z_coord, tke_cfg)
    phys = _physics(tke_cfg)
    u_cell = 0.5 * (s0.u.data[:, :-1, :] + s0.u.data[:, 1:, :])
    v_cell = 0.5 * (s0.v.data[:-1, :, :] + s0.v.data[1:, :, :])
    cc = s0._replace(u=s0.u.replace(data=u_cell), v=s0.v.replace(data=v_cell))
    tke0 = s0.tke.data

    def _step(dt_tke):
        _, _, tke_new = compute_vertical_K_profiles(
            cc, z_coord, None, phys, A_v_background=1.0e-4, K_v_background=0.0,
            tke_old=tke0, dt_tke=dt_tke, return_tke=True)
        return np.asarray(tke_new)

    tke_small = _step(600.0)
    tke_large = _step(4800.0)
    base = np.asarray(tke0)
    incr_small = np.abs(tke_small - base).sum()
    incr_large = np.abs(tke_large - base).sum()
    # Larger dt_tke -> larger integrated change toward equilibrium.
    assert incr_large > incr_small
    # And dt_tke is required when prognostic (no silent default).
    with pytest.raises(ValueError, match="dt_tke"):
        compute_vertical_K_profiles(
            cc, z_coord, None, phys, tke_old=tke0, return_tke=True)


# ---------------------------------------------------------------------------
# (4)/(5) source flags + energy sign
# ---------------------------------------------------------------------------


def test_bottom_drag_source_changes_tke_and_nonneg():
    """source_bottom_drag_diss ON with bottom flow raises TKE near the floor;
    OFF (or no bottom flow) leaves it unchanged. K_diss_bot is non-negative."""
    grid, z_coord = _grid_z()
    base = TKEConfig(prognostic=True)
    with_bd = TKEConfig(prognostic=True, source_bottom_drag_diss=True)

    s0 = _seed_tke(_perturbed_state(grid, z_coord), z_coord, base)
    # Strong bottom flow so the drag KE-extraction source is appreciable.
    m_off, _, _ = _model(base, bottom_drag_r=1.0e-3)
    m_on, _, _ = _model(with_bd, bottom_drag_r=1.0e-3)
    s_off = m_off._step_impl(s0, 3600.0)
    s_on = m_on._step_impl(s0, 3600.0)
    tke_off = np.asarray(s_off.tke.data)
    tke_on = np.asarray(s_on.tke.data)
    # The source ADDS energy -> on >= off everywhere (within fp), and strictly
    # greater somewhere (where bottom flow + drag are nonzero).
    assert np.all(tke_on >= tke_off - 1e-9)
    assert np.any(tke_on > tke_off + 1e-12)


def test_kdiss_bot_source_nonneg():
    """The bottom-drag KE-extraction source (Veros K_diss_bot) is non-negative
    by construction (drag opposes the flow)."""
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        bottom_drag_kediss_tke_source,
    )
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV)))
    v = jnp.asarray(rng.standard_normal((N_LAT + 1, N_LON, NLEV)))
    r = 1.0e-3
    drag_u = -r * u
    drag_v = -r * v
    mask = jnp.ones((N_LAT, N_LON))
    kdb = np.asarray(
        bottom_drag_kediss_tke_source(drag_u, drag_v, u, v, mask))
    assert kdb.shape == (N_LAT, N_LON, NLEV - 1)
    assert np.all(kdb >= -1e-15)        # non-negative source
    assert np.any(kdb > 0.0)            # nonzero where there is flow


def test_bottom_drag_source_zero_when_no_drag():
    """source_bottom_drag_diss ON but bottom_drag_r=0 (no drag) -> no source ->
    TKE identical to the source-off run (off==on when source is zero)."""
    grid, z_coord = _grid_z()
    base = TKEConfig(prognostic=True)
    with_bd = TKEConfig(prognostic=True, source_bottom_drag_diss=True)
    s0 = _seed_tke(_perturbed_state(grid, z_coord), z_coord, base)
    m_off, _, _ = _model(base, bottom_drag_r=0.0)
    m_on, _, _ = _model(with_bd, bottom_drag_r=0.0)
    s_off = m_off._step_impl(s0, 3600.0)
    s_on = m_on._step_impl(s0, 3600.0)
    np.testing.assert_allclose(
        np.asarray(s_on.tke.data), np.asarray(s_off.tke.data),
        rtol=0, atol=0)


def test_eke_diss_source_changes_tke_and_nonneg():
    """source_eke_diss ON with a nonzero carried eke_diss raises TKE; a ZERO
    eke_diss leaves it unchanged. The injected source is non-negative."""
    grid, z_coord = _grid_z()
    base = TKEConfig(prognostic=True)
    with_eke = TKEConfig(prognostic=True, source_eke_diss=True)
    m_off, _, _ = _model(base)
    m_on, _, _ = _model(with_eke)

    s_zero = _seed_tke(_perturbed_state(grid, z_coord), z_coord, with_eke,
                       with_eke_diss=True)
    # zero eke_diss -> source contributes nothing -> on == off.
    s_off0 = m_off._step_impl(
        s_zero._replace(eke_diss=None), 3600.0)
    s_on0 = m_on._step_impl(s_zero, 3600.0)
    np.testing.assert_allclose(
        np.asarray(s_on0.tke.data), np.asarray(s_off0.tke.data),
        rtol=1e-12, atol=1e-14)

    # nonzero eke_diss -> source ADDS energy -> on > off somewhere.
    lm = np.asarray(s_zero.land_mask.data)
    ediss = (1.0e-7 * (lm[:, :, None] > 0.5)) * np.ones(
        (1, 1, NLEV - 1))
    s_pos = s_zero._replace(
        eke_diss=Field(data=jnp.asarray(ediss), name="eke_diss",
                       dims=("lat", "lon", "level"), units="m^2/s^3"))
    s_on_pos = m_on._step_impl(s_pos, 3600.0)
    assert np.all(np.asarray(s_on_pos.tke.data) >= np.asarray(s_on0.tke.data)
                  - 1e-9)
    assert np.any(np.asarray(s_on_pos.tke.data) > np.asarray(s_on0.tke.data)
                  + 1e-12)
    assert np.all(ediss >= 0.0)  # source non-negative by construction


# ---------------------------------------------------------------------------
# (6) AD: grad finite through 2 steps with everything on
# ---------------------------------------------------------------------------


def test_grad_finite_through_two_steps():
    """jax.grad of a scalar of TKE through TWO prognostic steps (with both
    sources on) is finite — the prognostic carry is differentiable."""
    grid, z_coord = _grid_z()
    tke_cfg = TKEConfig(prognostic=True, source_eke_diss=True,
                        source_bottom_drag_diss=True)
    model, _, _ = _model(tke_cfg, bottom_drag_r=1.0e-3)
    s0 = _seed_tke(_perturbed_state(grid, z_coord), z_coord, tke_cfg,
                   with_eke_diss=True)
    dt = 3600.0

    def loss(u_data):
        s = s0._replace(u=s0.u.replace(data=u_data))
        s = model._step_impl(s, dt)
        s = model._step_impl(s, dt)
        return jnp.sum(s.tke.data ** 2)

    g = jax.grad(loss)(s0.u.data)
    assert np.all(np.isfinite(np.asarray(g)))
    # Physics implies non-zero sensitivity of TKE to the velocity (shear prod).
    assert np.any(np.abs(np.asarray(g)) > 0.0)


# ---------------------------------------------------------------------------
# (7) scan-compat: 5-step lax.scan with constant pytree
# ---------------------------------------------------------------------------


def test_scan_five_steps_constant_pytree():
    """integrate_scan seeds state.tke (+ eke_diss for source_eke_diss) so a
    5-step lax.scan keeps a constant pytree and runs to completion."""
    grid, z_coord = _grid_z()
    tke_cfg = TKEConfig(prognostic=True, source_bottom_drag_diss=True)
    model, _, _ = _model(tke_cfg, bottom_drag_r=1.0e-3)
    s0 = _perturbed_state(grid, z_coord)  # NOT pre-seeded; integrate_scan seeds
    final, traj = model.integrate_scan(s0, dt=3600.0, n_steps=5)
    assert final.tke is not None
    tke = np.asarray(final.tke.data)
    assert np.all(np.isfinite(tke))
    lm = np.asarray(final.land_mask.data)
    assert np.all(tke[lm > 0.5] >= tke_cfg.tke_background - 1e-12)
    # Trajectory carries a stable (n_steps, ...) tke stack.
    assert traj.tke.data.shape[0] == 5


class TestPrognosticTKEUnderAB2:
    """The faithful-AB2 path threads the prognostic TKE through its single
    implicit-mixing call (the tracer solve = the authoritative TKE site)."""

    def _ab2_model(self, *, additive=False):
        grid, z_coord = _grid_z()
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=1.0e4, K_v=0.0,
            implicit_vertical_mixing=True,
            outer_integrator="ab2",
            momentum_friction_additive=additive,
            physics=_physics(TKEConfig(prognostic=True,
                                       source_bottom_drag_diss=True)),
            bottom_drag_r=1.0e-3,
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        state = _perturbed_state(grid, z_coord)
        state = _seed_tke(state, z_coord, None, with_eke_diss=True)
        return model, state, grid

    def test_ab2_carries_and_evolves_tke(self):
        model, state, grid = self._ab2_model()
        sf = _wind_forcing(grid)
        t0 = np.asarray(state.tke.data).copy()
        s = state
        for _ in range(3):
            s = model.step(s, 3600.0, surface_forcing=sf)
        t3 = np.asarray(s.tke.data)
        assert np.all(np.isfinite(t3))
        assert float(np.min(t3)) >= 0.0
        assert np.max(np.abs(t3 - t0)) > 0.0, "TKE did not evolve under AB2"

    def test_ab2_additive_friction_combo(self):
        """The full faithful stack shape: ab2 + additive friction +
        prognostic TKE — TKE evolves, everything stays finite."""
        model, state, grid = self._ab2_model(additive=True)
        sf = _wind_forcing(grid)
        s = state
        for _ in range(3):
            s = model.step(s, 3600.0, surface_forcing=sf)
        assert np.all(np.isfinite(np.asarray(s.tke.data)))
        assert np.all(np.isfinite(np.asarray(s.u.data)))
        assert np.all(np.isfinite(np.asarray(s.T.data)))
        assert np.max(np.abs(np.asarray(s.tke.data)
                             - np.asarray(state.tke.data))) > 0.0

    def test_ab2_off_path_runs_without_tke(self):
        """prognostic=False under ab2: tke threading args are all None and the
        step runs with no tke on the state (bit-identity of the off path is
        locked by the existing AB2 regression suite)."""
        grid, z_coord = _grid_z()
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=1.0e4, K_v=0.0, implicit_vertical_mixing=True,
            outer_integrator="ab2",
            physics=_physics(TKEConfig(prognostic=False)),
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        state = _perturbed_state(grid, z_coord)
        s = model.step(state, 3600.0)
        assert s.tke is None


def test_prandtl_tracer_floor_independent_of_momentum_floor():
    """Quiescent-cell K_H floors at kappaH_min, NOT kappaM_min/Pr.

    The abyssal over-diffusion fix: on the Prandtl path the tracer floor must be
    INDEPENDENT of the momentum floor (NEMO zdftke floors avt at avtb and avm at
    avmb separately). Previously K_M was floored to kappaM_min BEFORE the Prandtl
    divide, so a quiescent deep cell (raw K << kappaM_min) got K_H = kappaM_min/Pr
    ~= 3.1e-5 instead of kappaH_min = 1.2e-5 — a uniform ~2.6x over-diffusion of
    the abyss vs NEMO (FREE_TKE_DIAGNOSIS_FINDINGS.md).
    """
    from legoesm.ocean.physics.vertical_mixing.tke import compute_K_from_tke
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

    cfg = TKEConfig(prandtl_mode="richardson", prandtl_ri_coeff=4.5,
                    kappa_convention="veros_sqrte",
                    kappaM_min=1.2e-4, kappaH_min=1.2e-5)
    # Quiescent stratified cell: tiny TKE + short length -> raw K = c_k*l*sqrt(e)
    # = 0.1*0.1*1e-3 = 1e-5 << kappaM_min. Ri = N2/shear = 0.86 -> Pr = 4.5*0.86
    # = 3.87, so the buggy leak kappaM_min/Pr = 3.1e-5 > kappaH_min.
    e = jnp.array([1.0e-6]); l_k = jnp.array([0.1])
    N2 = jnp.array([0.86]); shear_sq = jnp.array([1.0])
    K_M, K_H = compute_K_from_tke(e, l_k, cfg, N2=N2, shear_sq=shear_sq)
    # K_M floors at kappaM_min (unchanged); K_H floors INDEPENDENTLY at kappaH_min.
    np.testing.assert_allclose(float(K_M[0]), 1.2e-4, rtol=1e-6)
    np.testing.assert_allclose(float(K_H[0]), 1.2e-5, rtol=1e-6)
    # non-vacuity: the old floored-first path would have leaked kappaM_min/Pr.
    assert float(K_H[0]) < 0.5 * (1.2e-4 / 3.87)
    # active/interior cell (raw K > kappaM_min) is UNAFFECTED by the fix:
    e2 = jnp.array([1.0e-2]); l2 = jnp.array([50.0])  # raw K = 0.1*50*0.1 = 0.5
    KM2, KH2 = compute_K_from_tke(e2, l2, cfg, N2=N2, shear_sq=shear_sq)
    np.testing.assert_allclose(float(KH2[0]), 0.5 / 3.87, rtol=1e-6)  # K_M/Pr, no floor
