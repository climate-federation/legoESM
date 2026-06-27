"""Additive momentum vertical-friction placement (config.momentum_friction_additive).

Veros computes the implicit vertical-friction increment du_mix from the PRE-STEP
velocity u^n (core/friction.py, backward-Euler on u^n) and adds it ADDITIVELY to
the AB2-extrapolated explicit tendency (core/external/solve_stream.py)::

    u^{n+1} = AB2(u) + (BE(u^n) - u^n)

legoESM's default placement is SEQUENTIAL — backward-Euler on the AB2 state::

    u^{n+1} = BE(AB2(u))

Both are implicit/unconditionally stable; the O(dt²·A_v) placement delta
dominates the realized momentum increment near equilibrium (where
AB2(du) ≈ −du_mix).  TRACERS keep the sequential placement in BOTH modes (that
IS Veros's tracer placement, core/thermodynamics.py).

These tests use a constant background A_v/K_v (physics=None ⇒ the K-profile
fallback returns state-independent constants), so the placement identities hold
exactly (to fp64 round-off) and the two modes are algebraically relatable.

Run in the fp64 precision policy.
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

_DT = 1800.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _channel(n_lat=8, n_lon=16, **cfg_kw):
    """Closed channel with a thermal front and a vertically-sheared jet.

    The depth-varying u seed makes the vertical-friction increment nonzero
    (friction on a depth-uniform or zero flow is a no-op).
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    # Vertically-sheared zonal jet (surface-intensified), masked + lon-wrapped.
    nlev = z_coord.n_levels
    shear = np.linspace(1.0, 0.1, nlev)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("outer_integrator", "ab2")
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


# --------------------------------------------------------------- validation

def test_rejects_without_ab2():
    with pytest.raises(ValueError, match="outer_integrator"):
        _channel(momentum_friction_additive=True,
                 outer_integrator="forward_euler")


def test_rejects_without_implicit_vmix():
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _channel(momentum_friction_additive=True,
                 implicit_vertical_mixing=False)


# ----------------------------------------------------- placement semantics

def _step_both(n_steps=1):
    """Run sequential and additive models from the SAME initial state."""
    state, model_seq = _channel(momentum_friction_additive=False)
    _, model_add = _channel(momentum_friction_additive=True)
    s_seq, s_add = state, state
    for _ in range(n_steps):
        s_seq = model_seq.step(s_seq, _DT)
        s_add = model_add.step(s_add, _DT)
    return state, model_seq, model_add, s_seq, s_add


def test_tracers_bit_identical_across_modes():
    """The flag relocates MOMENTUM friction only; the tracer solve acts on the
    same AB2 state with the same K profiles in both modes ⇒ T, S bit-identical."""
    _, _, _, s_seq, s_add = _step_both(n_steps=1)
    np.testing.assert_array_equal(np.asarray(s_seq.T.data),
                                  np.asarray(s_add.T.data))
    np.testing.assert_array_equal(np.asarray(s_seq.S.data),
                                  np.asarray(s_add.S.data))
    np.testing.assert_array_equal(np.asarray(s_seq.eta.data),
                                  np.asarray(s_add.eta.data))


def test_additive_equals_sequential_reconstruction():
    """Placement identity (constant A_v ⇒ exact):

        u_add  = u_ab2 + (BE(u^n) − u^n)      [the new path]
        u_seq  = BE(u_ab2)                     [the default path]

    Recover u_ab2 from the additive run by subtracting the independently
    computed friction increment, apply BE to it, and require the sequential
    answer.  Validates both that the additive path adds EXACTLY the
    u^n-evaluated increment and that the default path is untouched."""
    state, model_seq, model_add, s_seq, s_add = _step_both(n_steps=1)

    fric = model_add._apply_implicit_vertical_mixing(
        state, _DT, None, do_tracers=False)
    du = np.asarray(fric.u.data) - np.asarray(state.u.data)
    dv = np.asarray(fric.v.data) - np.asarray(state.v.data)
    assert np.max(np.abs(du)) > 0.0, "friction increment unexpectedly zero"

    u_ab2 = jnp.asarray(np.asarray(s_add.u.data) - du)
    v_ab2 = jnp.asarray(np.asarray(s_add.v.data) - dv)
    st_ab2 = s_add._replace(u=s_add.u.replace(data=u_ab2),
                            v=s_add.v.replace(data=v_ab2))
    seq_recon = model_seq._apply_implicit_vertical_mixing(
        st_ab2, _DT, None, do_tracers=False)

    np.testing.assert_allclose(np.asarray(seq_recon.u.data),
                               np.asarray(s_seq.u.data), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(seq_recon.v.data),
                               np.asarray(s_seq.v.data), rtol=0, atol=1e-13)


def test_modes_differ_in_momentum():
    """The placement is a real O(dt²·A_v) difference — the two modes must NOT
    coincide on a sheared flow (guards against a silently-dead flag)."""
    _, _, _, s_seq, s_add = _step_both(n_steps=2)
    assert np.max(np.abs(np.asarray(s_seq.u.data)
                         - np.asarray(s_add.u.data))) > 0.0


def test_split_solve_composition_identity():
    """do_tracers/do_momentum split: the combined solve equals the composition
    of the two partial solves (the solves are independent per field) — the
    refactor cannot have introduced cross-field coupling."""
    state, model_seq, _, _, _ = _step_both(n_steps=0)
    both = model_seq._apply_implicit_vertical_mixing(state, _DT, None)
    trac = model_seq._apply_implicit_vertical_mixing(
        state, _DT, None, do_momentum=False)
    mom = model_seq._apply_implicit_vertical_mixing(
        state, _DT, None, do_tracers=False)
    np.testing.assert_array_equal(np.asarray(both.T.data),
                                  np.asarray(trac.T.data))
    np.testing.assert_array_equal(np.asarray(both.S.data),
                                  np.asarray(trac.S.data))
    np.testing.assert_array_equal(np.asarray(both.u.data),
                                  np.asarray(mom.u.data))
    np.testing.assert_array_equal(np.asarray(both.v.data),
                                  np.asarray(mom.v.data))
    # Partial calls leave the untouched fields bit-identical to the input.
    np.testing.assert_array_equal(np.asarray(mom.T.data),
                                  np.asarray(state.T.data))
    np.testing.assert_array_equal(np.asarray(trac.u.data),
                                  np.asarray(state.u.data))


def test_barotropic_mode_untouched():
    """Zero-flux BCs ⇒ the friction increment's thickness-weighted depth-mean
    is ~0, so adding it cannot perturb the un-AB2'd barotropic mode."""
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface,
    )
    from legoesm.ocean.physics.vertical_mixing import build_dz_half  # noqa: F401
    state, _, model_add, _, _ = _step_both(n_steps=0)
    fric = model_add._apply_implicit_vertical_mixing(
        state, _DT, None, do_tracers=False)
    du = np.asarray(fric.u.data) - np.asarray(state.u.data)
    from legoesm.ocean.vertical import compute_ocean_jacobian
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data,
                               model_add.z_coord)
    dz_cell = model_add.z_coord.dz_ref * np.asarray(J)[..., None]
    dz_u = np.asarray(interp_cell_to_uface(jnp.asarray(dz_cell)))
    umask = np.asarray(state.u_mask.data)[..., None]
    depth_int = np.abs(np.sum(du * dz_u * umask, axis=-1))
    scale = np.max(np.sum(np.abs(du) * dz_u * umask, axis=-1))
    assert scale > 0.0
    assert np.max(depth_int) <= 1e-10 * scale


# ------------------------------------------------------------ robustness

def test_stability_100_steps():
    state, model = _channel(momentum_friction_additive=True)
    for _ in range(100):
        state = model.step(state, _DT)
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(T))
    assert np.max(np.abs(u)) < 5.0
    assert -5.0 < T.min() and T.max() < 40.0


def test_differentiable():
    state, model = _channel(n_lat=6, n_lon=8,
                            momentum_friction_additive=True)

    def loss(scale):
        st = state._replace(u=state.u.replace(data=state.u.data * scale))
        s1 = model.step(st, _DT)
        s2 = model.step(s1, _DT)
        return jnp.sum(s2.u.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0
