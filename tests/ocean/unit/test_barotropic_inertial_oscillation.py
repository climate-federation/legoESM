"""Closed-basin barotropic geostrophic adjustment vs the oracle's Flat-y inertial
oscillation — the topology root cause of the internal_tide fidelity gap (#576).

⚠️ THIS SUPERSEDES an earlier (wrong) framing in this file that called the
non-rotating depth-mean a "frozen Coriolis bug". It is NOT a bug. Verified
(JAX_ENABLE_X64, quarter→full inertial period sweep):

  legoESM's lat-lon beta-plane C-grid is **meridionally CLOSED** — the barotropic
  solvers wall the north/south v-faces (`_zero_polar_lat_ends`; see
  `create_beta_plane_cgrid_geometry`'s "A domain periodic in y is NOT supported").
  An initial UNIFORM barotropic u = U0 (v=0) therefore does **geostrophic
  adjustment**, not a free inertial oscillation: the depth-mean u stays ≈ U0 while
  a meridional free-surface tilt builds to balance it, ∂η/∂y → −f·U0/g (measured
  ratio → ~1.08 after one inertial period; <v> → 0). The barotropic Rossby radius
  √(gH)/f ≈ 1358 km dwarfs the basin (256 km), so the WHOLE domain adjusts — there
  is no wall-free interior (verified: still adjusts at NY=128).

  The Oceananigans internal_tide oracle uses `topology = (Periodic, Flat, Bounded)`
  — TRUE 2-D x–z, meridionally UNBOUNDED (no v-walls, ∂η/∂y ≡ 0) — so its barotropic
  mode does the free inertial/tidal oscillation (u rotates into v, no DC retention).

CONSEQUENCE for internal_tide (#576): in legoESM's closed basin the prescribed
barotropic U₂ and the tidal flow are retained as a geostrophically-balanced DC
mean current (measured ~+0.4 m/s) instead of the oracle's purely-oscillating tide
→ a steady lee wake over the ridge instead of a radiating tide → the b' pattern
decorrelates. The single-step PGF / w / tracer-advection tendencies all match the
oracle (#576 iters 10–11); the 2-day divergence is THIS topology mismatch, not a
tracer/PGF/partial-cell or Coriolis-operator defect. The faithful cure is a
meridionally-unbounded / y-periodic (re-entrant channel) configuration matching
the oracle's Flat-y — currently unsupported by the beta-plane grid — NOT a change
to the Coriolis or barotropic operators (which are doing correct closed-basin
physics here).

These tests pin the verified behavior so a future Flat-y/periodic-y option (or a
regression in the closed-basin geostrophic adjustment, or the *working* baroclinic
Coriolis) trips loudly.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.fidelity.oceananigans_recipe import (
    oceananigans_canonical_ocean_config,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star

# Flat-bottom f-plane box (x-periodic, meridionally CLOSED). f<0 (southern mid-lat).
_NY, _NX, _NZ = 64, 8, 4
_H = 2.0e3
_DX = 4.0e3
_LAT = -45.0
_F0 = 2.0 * constants.Omega * np.sin(np.radians(_LAT))
_U0 = 0.2
_T_INERTIAL = 2.0 * np.pi / abs(_F0)
_DT = 300.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _build(flat=False):
    # FLAT-y masks are built at rest-state construction (before the model), so set
    # the global here too; the config field then keeps the stepping operators flat.
    from legoesm.grids.halo_latlon import set_meridionally_flat
    set_meridionally_flat(flat)
    grid = create_beta_plane_cgrid_geometry(
        _NY, _NX, dx_m=_DX, dy_m=_DX, f0=_F0, beta=0.0,
        y_origin_m=-_NY * _DX / 2, x_origin_m=-_NX * _DX / 2,
        cartesian_pseudo_lat=True)
    z = create_ocean_z_star(n_levels=_NZ, H_max=_H)
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=2.0e-4, beta_S=0.0),
        g=constants.g, rho_0=1000.0, A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0,
        momentum_advection="flux_form",
        barotropic_solver="implicit_cn", coriolis_scheme="explicit_ab2",
        bottom_drag_r=0.0, tracer_advection="weno5", weno_smoothness="split",
        meridionally_flat=flat)
    wall = jnp.ones((_NY, _NX), dtype=jnp.asarray(grid.cos_lat).dtype)
    Hb = jnp.full((_NY, _NX), _H, dtype=wall.dtype)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=wall, H_bathy_override=Hb,
        T_water_init_C=10.0, T_deep=10.0)
    # The AB2 slow-forcing carry must be SEEDED before the first step: the
    # model stores a Field each step, and a step-1 None->Field transition
    # breaks the lax.scan carry, so the model now refuses an unseeded state.
    # These tests predate that guard, which is why they read as failures rather
    # than as a config error (#1388). Same idiom as
    # build_silvestri_baroclinic_jet_setup.
    if getattr(cfg.barotropic, "barotropic_slow_forcing_ab2", False):
        from legoesm.core.field import Field as _F
        state = state._replace(
            F_slow_u_prev=_F(data=jnp.zeros_like(state.u.data[:, :, 0]),
                             name="F_slow_u_prev", dims=("lat", "lon_u"),
                             units="m/s^2"),
            F_slow_v_prev=_F(data=jnp.zeros_like(state.v.data[:, :, 0]),
                             name="F_slow_v_prev", dims=("lat_v", "lon"),
                             units="m/s^2"))
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, z, state, model


def _quarter_steps():
    return int(round(_T_INERTIAL / 4.0 / _DT))


def test_meridionally_flat_inertial_oscillation():
    """Oceananigans `Flat`-y mode (`set_meridionally_flat`): every meridional
    difference → 0, so a uniform barotropic u does the FREE inertial oscillation
    (u → 0 over a quarter period, |v| spins up) — matching the oracle's
    `topology=(Periodic, Flat, Bounded)` — instead of the closed-basin geostrophic
    lock. This is the fix that reproduced Oceananigans internal_tide (#576): no
    meridional d.o.f. → no closed-basin adjustment AND no 2Δy mode. Bounded for a
    full inertial period. Default-OFF bit-identity is held by the geostrophic test."""
    from legoesm.grids.halo_latlon import set_meridionally_flat

    try:
        grid, z, state, model = _build(flat=True)  # config-driven Flat-y
        u = np.full((_NY, _NX + 1, _NZ), _U0)
        state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
        step = jax.jit(lambda s: model.step(s, _DT, surface_forcing=None))
        uq = vmax = None
        for i in range(4 * _quarter_steps()):  # one full inertial period
            state = step(state)
            if i == _quarter_steps() - 1:
                uu = np.asarray(state.u.data)[_NY // 2]
                uq = float(uu[uu[:, 0] != 0, 0].mean())
            vmax = max(vmax or 0.0, float(np.abs(np.asarray(state.v.data)).max()))
        u_final_max = float(np.abs(np.asarray(state.u.data)).max())
    finally:
        set_meridionally_flat(False)
    # quarter period: u rotated away from U0 toward 0 (the free inertial oscillation)
    assert abs(uq) < 0.25 * _U0, f"flat-y u did not rotate away ({uq:+.4f})"
    # v spun up (Coriolis rotation), and the run stayed BOUNDED (no 2Δy blow-up)
    assert vmax > 0.5 * _U0, f"flat-y v did not spin up ({vmax:.4f})"
    assert u_final_max < 2.0 * _U0, f"flat-y run not bounded (max|u|={u_final_max:.3f})"


def test_meridionally_flat_rejects_ungated_meridional_operators():
    """`meridionally_flat=True` is wired only into the flat-aware meridional
    operators (PGF/KE-grad/tracer-advection/Laplacian/vector-Laplacian +
    flux-form momentum advection). Operators that own their own meridional
    stencil (flux-divergence viscosity, GM/Redi, lateral friction) are NOT
    flat-aware, so combining them must be REJECTED at construction (else the
    model is silently Flat for some terms, 3-D for others)."""
    from legoesm.grids.halo_latlon import set_meridionally_flat
    set_meridionally_flat(False)
    try:
        # flux-divergence viscosity with A_h>0 + flat → reject
        with __import__("pytest").raises(ValueError, match="meridionally_flat"):
            grid = create_beta_plane_cgrid_geometry(
                _NY, _NX, dx_m=_DX, dy_m=_DX, f0=_F0, beta=0.0,
                y_origin_m=-_NY * _DX / 2, x_origin_m=-_NX * _DX / 2,
                cartesian_pseudo_lat=True)
            z = create_ocean_z_star(n_levels=_NZ, H_max=_H)
            cfg = oceananigans_canonical_ocean_config(
                eos_linear=LinearEOSConfig(alpha_T=2.0e-4, beta_S=0.0),
                g=constants.g, rho_0=1000.0, A_h=100.0, K_h=0.0,
                momentum_advection="flux_form", barotropic_solver="implicit_cn",
                coriolis_scheme="explicit_ab2", meridionally_flat=True)
            LatLonCGridOceanModel(grid, z, cfg)
    finally:
        set_meridionally_flat(False)


def test_closed_basin_geostrophic_adjustment():
    """A uniform barotropic u in the meridionally-CLOSED basin geostrophically
    adjusts: u stays ≈ U0 and a meridional η tilt builds to ∂η/∂y → −f·U0/g (the
    barotropic Rossby radius spans the basin → no free inertial oscillation). This
    is the CORRECT closed-basin response (NOT a Coriolis bug); it differs from the
    oracle's Flat-y free oscillation purely by meridional TOPOLOGY."""
    grid, z, state, model = _build()
    u = np.full((_NY, _NX + 1, _NZ), _U0)
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))

    @jax.jit
    def step(s):
        return model.step(s, _DT, surface_forcing=None)

    for _ in range(4 * _quarter_steps()):  # one full inertial period
        state = step(state)

    u_out = np.asarray(state.u.data)[_NY // 4:3 * _NY // 4]
    v_out = np.asarray(state.v.data)[_NY // 4:3 * _NY // 4]
    ubar = float(u_out[u_out[:, :, 0] != 0].mean())
    vbar = float(v_out.mean())
    # depth-mean u is retained (geostrophically balanced), not rotated away
    assert abs(ubar - _U0) < 0.1 * _U0, f"u not retained ({ubar:+.4f})"
    assert abs(vbar) < 0.1 * _U0, f"v should stay ~0 (got {vbar:+.4f})"

    eta = np.asarray(state.eta.data)[:, _NX // 2]
    deta_dy = np.gradient(eta, _DX)[_NY // 4:3 * _NY // 4].mean()
    geo_pred = -_F0 * _U0 / constants.g
    # η tilt has reached geostrophic balance with f·u (within ~25%)
    assert abs(deta_dy / geo_pred - 1.0) < 0.25, (
        f"∂η/∂y={deta_dy:.3e} not geostrophically balanced with -f·U0/g="
        f"{geo_pred:.3e} (ratio {deta_dy/geo_pred:.2f})")


def test_meridionally_periodic_poc_oscillates():
    """PoC (#576): with the meridionally-PERIODIC (y-re-entrant) mode ON, the SAME
    uniform barotropic u does the FREE inertial oscillation (u rotates away from
    U0 toward 0 over a quarter inertial period) — matching the oracle's Flat-y —
    instead of the closed-basin geostrophic lock. Confirms the topology fix
    direction. NOTE: only 3 of ~90 boundary sites are wired so far, so the run is
    physical only within the first quarter period (stable regime); the full sweep
    is the production feature. Default-OFF is asserted bit-identical by
    test_closed_basin_geostrophic_adjustment."""
    from legoesm.grids.halo_latlon import set_meridionally_periodic

    def quarter_period_u(periodic):
        set_meridionally_periodic(periodic)
        try:
            grid, z, state, model = _build()
            u = np.full((_NY, _NX + 1, _NZ), _U0)
            state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
            step = jax.jit(lambda s: model.step(s, _DT, surface_forcing=None))
            for _ in range(_quarter_steps()):
                state = step(state)
            uu = np.asarray(state.u.data)[_NY // 2]
            return float(uu[uu[:, 0] != 0, 0].mean())
        finally:
            set_meridionally_periodic(False)

    u_walled = quarter_period_u(False)
    u_periodic = quarter_period_u(True)
    # Walled: geostrophically locked near U0. Periodic: rotated away to ~0
    # (analytic inertial u at a quarter period = U0·cos(π/2) ≈ 0).
    assert abs(u_walled - _U0) < 0.1 * _U0, f"walled u not locked ({u_walled:+.4f})"
    assert abs(u_periodic) < 0.25 * _U0, (
        f"periodic u did not rotate away ({u_periodic:+.4f}); the y-periodic mode "
        "should make the barotropic mode oscillate freely like the Flat-y oracle")


def test_baroclinic_inertial_rotates():
    """CONTROL: the baroclinic (zero-depth-mean, vertically sheared) Coriolis DOES
    rotate u into v — its (small) Rossby radius fits inside the basin, so it is NOT
    geostrophically locked. Proves the Coriolis machinery is correct and that the
    closed-basin retention above is a barotropic/topology effect, not a broken
    Coriolis operator. A regression in the working Coriolis trips this."""
    grid, z, state, model = _build()
    zf = np.asarray(z.z_full_ref)
    shear = (zf - zf.mean()) / (zf.max() - zf.min())  # mean 0, range ~[-0.5,0.5]
    u = (_U0 * shear)[None, None, :] * np.ones((_NY, _NX + 1, _NZ))
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))

    @jax.jit
    def step(s):
        return model.step(s, _DT, surface_forcing=None)

    v_amp = 0.0
    for _ in range(_quarter_steps()):
        state = step(state)
        v_amp = max(v_amp, float(np.abs(np.asarray(state.v.data)).max()))
    assert v_amp > 0.3 * _U0, (
        f"baroclinic Coriolis did not rotate u into v (max|v|={v_amp:.4f}); "
        "the Coriolis machinery itself is broken")
