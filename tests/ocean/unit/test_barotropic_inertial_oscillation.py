"""The barotropic (depth-mean) inertial oscillation must actually ROTATE.

Root cause of the Oceananigans ``internal_tide`` fidelity failure (issue #576,
scoreboard row 2d): legoESM's BAROCLINIC Coriolis is correct (a vertically
sheared, zero-depth-mean velocity traces a clean inertial circle — verified in
``test_baroclinic_inertial_rotates`` below), but the BAROTROPIC (depth-mean)
Coriolis for the **x-uniform k=0 mode** is FROZEN: an initial uniform u = U0,
v = 0 on an f-plane stays u = U0, v = 0 forever instead of rotating
u → 0, v → −U0/f-sign over a quarter inertial period.

Physics (flat bottom, x- and y-uniform, free surface, no forcing):
    du/dt = +f v,  dv/dt = −f u,  η ≡ 0  (since ∂_x u = 0 ⇒ ∂_t η = 0)
  ⇒ a pure inertial oscillation  u = U0 cos(ft), v = −U0 sin(ft).

Why it matters for ``internal_tide``: the case sets u = U₂ and a barotropic M2
tide. With the k=0 inertial mode frozen, the prescribed U₂ never rotates away —
it lingers as a spurious DC mean current (~+0.4 m/s, measured) that advects a
steady lee wake over the ridge instead of letting the OSCILLATING tide radiate.
The single-step PGF / w / tracer-advection tendencies all match the oracle
(issue #576 iter 10–11); the 2-day divergence is THIS frozen depth-mean mode.

The barotropic Coriolis is routed (``coriolis_scheme``) either through the
in-substep solver term (``matsuno_split``) or through the AB2 slow forcing
F_slow (``explicit_ab2``); the k=0 depth-mean rotation is lost in BOTH, and
across both free-surface solvers (``implicit_cn``, ``explicit_substep``) — see
``test_barotropic_inertial_frozen_all_configs``. Fixing it is a barotropic-solver
change (the depth-mean Coriolis must rotate the k=0 mode without re-admitting the
2Δx C-grid rotational null mode that ``test_barotropic_coriolis_null_mode`` pins).

These tests are the mechanical tripwire for that fix: ``xfail(strict=True)`` on the
barotropic rotation (it currently does NOT rotate) so the fix flips it to xpass
LOUDLY, and a passing baroclinic-rotation test so a regression that breaks the
*working* Coriolis also trips.
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

# Small flat-bottom f-plane box (x-periodic, thin y). f<0 (southern mid-lat).
_NY, _NX, _NZ = 8, 16, 8
_H = 2.0e3
_DX = 4.0e3
_LAT = -45.0
_F0 = 2.0 * constants.Omega * np.sin(np.radians(_LAT))
_U0 = 0.2
_T_INERTIAL = 2.0 * np.pi / abs(_F0)


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _build(barotropic_solver="implicit_cn", coriolis_scheme="explicit_ab2"):
    grid = create_beta_plane_cgrid_geometry(
        _NY, _NX, dx_m=_DX, dy_m=_DX, f0=_F0, beta=0.0,
        y_origin_m=-_NY * _DX / 2, x_origin_m=-_NX * _DX / 2,
        cartesian_pseudo_lat=True)
    z = create_ocean_z_star(n_levels=_NZ, H_max=_H)
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=2.0e-4, beta_S=0.0),
        g=constants.g, rho_0=1000.0, A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0,
        momentum_advection="flux_form",
        barotropic_solver=barotropic_solver, coriolis_scheme=coriolis_scheme,
        bottom_drag_r=0.0, tracer_advection="weno5", weno_smoothness="split")
    if coriolis_scheme != "explicit_ab2":
        cfg = cfg._replace(outer_integrator="forward_euler")
    wall = jnp.ones((_NY, _NX), dtype=jnp.asarray(grid.cos_lat).dtype)
    Hb = jnp.full((_NY, _NX), _H, dtype=wall.dtype)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=wall, H_bathy_override=Hb,
        T_water_init_C=10.0, T_deep=10.0)
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, z, state, model


def _set_uniform_u(state, u0):
    u = np.full((_NY, _NX + 1, _NZ), u0)
    return state._replace(u=state.u.replace(data=jnp.asarray(u)))


def _depth_mean_uv(state):
    u = np.asarray(state.u.data)[_NY // 2]
    v = np.asarray(state.v.data)[_NY // 2]
    ubar = float(u[u[:, 0] != 0, :].mean()) if np.any(u[:, 0] != 0) else 0.0
    vbar = float(v[1:-1].mean())
    return ubar, vbar


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN BUG (issue #576): the barotropic x-uniform k=0 inertial mode is "
    "frozen — depth-mean u stays at U0 instead of rotating to ~0 over a quarter "
    "inertial period. Flips to xpass when the barotropic-Coriolis fix lands."))
def test_barotropic_inertial_rotates():
    """A uniform barotropic u must rotate into v over a quarter inertial period."""
    grid, z, state, model = _build()
    state = _set_uniform_u(state, _U0)

    @jax.jit
    def step(s):
        return model.step(s, 300.0, surface_forcing=None)

    nq = int(round(_T_INERTIAL / 4.0 / 300.0))
    for _ in range(nq):
        state = step(state)
    ubar, vbar = _depth_mean_uv(state)
    # After a quarter inertial period: u→0, |v|→U0 (sign from f<0).
    assert abs(ubar) < 0.25 * _U0, f"depth-mean u did not rotate away: {ubar:+.4f}"
    assert abs(vbar) > 0.5 * _U0, f"depth-mean v did not spin up: {vbar:+.4f}"


def test_baroclinic_inertial_rotates():
    """CONTROL: the baroclinic (zero-depth-mean, vertically sheared) Coriolis
    DOES rotate — proves the Coriolis machinery is correct and isolates the bug
    to the barotropic depth-mean mode. A regression that breaks the working
    baroclinic Coriolis trips this (it passes today)."""
    grid, z, state, model = _build()
    zf = np.asarray(z.z_full_ref)
    shear = (zf - zf.mean()) / (zf.max() - zf.min())  # mean 0, range ~[-0.5,0.5]
    u = (_U0 * shear)[None, None, :] * np.ones((_NY, _NX + 1, _NZ))
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))

    @jax.jit
    def step(s):
        return model.step(s, 300.0, surface_forcing=None)

    v0_amp = 0.0
    nq = int(round(_T_INERTIAL / 4.0 / 300.0))
    for _ in range(nq):
        state = step(state)
        v0_amp = max(v0_amp, float(np.abs(np.asarray(state.v.data)).max()))
    # The sheared u must spin up a comparable sheared v (inertial rotation).
    assert v0_amp > 0.3 * _U0, (
        f"baroclinic Coriolis did not rotate u into v (max|v|={v0_amp:.4f}); "
        "the Coriolis machinery itself is broken")


@pytest.mark.parametrize("bsolver", ["implicit_cn", "explicit_substep"])
@pytest.mark.parametrize("cscheme", ["explicit_ab2", "matsuno_split"])
def test_barotropic_inertial_frozen_all_configs(bsolver, cscheme):
    """Document that the frozen k=0 mode is NOT solver/scheme specific: the
    depth-mean u stays pinned at its initial value across both free-surface
    solvers and both Coriolis routings. (Asserts the CURRENT buggy behavior so a
    fix that rotates ANY config trips this and forces this test to be updated
    together with the xfail above.)"""
    grid, z, state, model = _build(barotropic_solver=bsolver,
                                   coriolis_scheme=cscheme)
    state = _set_uniform_u(state, _U0)

    @jax.jit
    def step(s):
        return model.step(s, 300.0, surface_forcing=None)

    nq = int(round(_T_INERTIAL / 4.0 / 300.0))
    for _ in range(nq):
        state = step(state)
    ubar, _ = _depth_mean_uv(state)
    assert abs(ubar - _U0) < 0.1 * _U0, (
        f"{bsolver}/{cscheme}: depth-mean u={ubar:+.4f} is no longer frozen at "
        f"{_U0} — the barotropic-Coriolis fix may have landed; update this test "
        "and flip test_barotropic_inertial_rotates to a real (non-xfail) assert.")
