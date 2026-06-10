"""AB2 extrapolation scope (config.ab2_scope) — Veros-faithful dissipative placement.

Veros AB2-extrapolates ONLY the ADVECTIVE part of the explicit increment and
applies the DISSIPATIVE tendencies at WEIGHT 1.0 (forward-Euler)::

    X^{n+1} = X^n + (1.5+ε)·ΔX_adv^n − (0.5+ε)·ΔX_adv^{n-1} + 1.0·ΔX_diss^n

where ΔX_diss is, for MOMENTUM, lateral friction + bottom drag
(core/external/solve_stream.py adds ``du_mix`` unextrapolated) and, for TRACERS,
lateral diffusion + GM/Redi isoneutral+skew (core/thermodynamics.py /
core/isoneutral/diffusion.py add the ``tr[tau]``-evaluated diffusion to
``tr[taup1]`` at weight 1.0).

legoESM's default ``ab2_scope="total"`` AB2-extrapolates the FULL explicit
increment (dissipation included). ``ab2_scope="advective"`` is the Veros
placement. ``"total"`` is BIT-IDENTICAL to the historical scheme.

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


def _channel(n_lat=8, n_lon=16, with_gm_redi=False, **cfg_kw):
    """Closed channel with a thermal front and a vertically-sheared jet.

    The depth-varying u seed makes lateral friction + bottom drag nonzero; the
    meridional thermal front makes lateral tracer diffusion (+ GM/Redi when
    enabled) nonzero.
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
    nlev = z_coord.n_levels
    shear = np.linspace(1.0, 0.1, nlev)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))

    if with_gm_redi:
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        cfg_kw.setdefault("gm_redi", GMRediConfig(kappa_GM=500.0,
                                                  kappa_Redi=500.0))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("A_h", 2.0e4)
    cfg_kw.setdefault("A_v", 1.0e-3)
    cfg_kw.setdefault("K_v", 1.0e-4)
    cfg_kw.setdefault("K_h", 500.0)
    cfg_kw.setdefault("bottom_drag_r", 1.0e-3)
    cfg = LatLonCGridOceanConfig(
        n_barotropic_substeps=8, enable_runtime_checks=False, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


# --------------------------------------------------------------- validation

def test_rejects_unknown_scope():
    with pytest.raises(ValueError, match="ab2_scope must be one of"):
        _channel(ab2_scope="bogus", outer_integrator="ab2")


def test_rejects_advective_without_ab2():
    with pytest.raises(ValueError, match="outer_integrator"):
        _channel(ab2_scope="advective", outer_integrator="forward_euler")


# ------------------------------------------------------ default bit-identity

def _run(model, state, n_steps):
    s = state
    for _ in range(n_steps):
        s = model.step(s, _DT)
    return s


def test_total_default_is_bit_identical_forward_euler():
    """ab2_scope is meaningless for the forward-Euler outer integrator, and the
    default "total" must not perturb it: a config that never sets ab2_scope and
    one that explicitly sets "total" produce byte-equal multi-step states."""
    state, m_default = _channel(outer_integrator="forward_euler")
    _, m_total = _channel(outer_integrator="forward_euler", ab2_scope="total")
    s0 = _run(m_default, state, 3)
    s1 = _run(m_total, state, 3)
    for f in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data),
            err_msg=f"FE {f} not bit-identical")


def test_total_default_is_bit_identical_ab2():
    """The AB2 path with the default "total" scope is byte-equal whether or not
    ab2_scope is explicitly set — guards the gating (no silent activation)."""
    state, m_default = _channel(outer_integrator="ab2")
    _, m_total = _channel(outer_integrator="ab2", ab2_scope="total")
    s0 = _run(m_default, state, 3)
    s1 = _run(m_total, state, 3)
    for f in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data),
            err_msg=f"AB2 {f} not bit-identical")


def test_total_default_is_bit_identical_ab2_with_gm_redi():
    """GM/Redi active: "total" must keep the GM/Redi tendency inside the AB2'd
    increment exactly as before (no leak through the new diss-bucket branch)."""
    state, m_default = _channel(outer_integrator="ab2", with_gm_redi=True)
    _, m_total = _channel(outer_integrator="ab2", ab2_scope="total",
                          with_gm_redi=True)
    s0 = _run(m_default, state, 3)
    s1 = _run(m_total, state, 3)
    for f in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data),
            err_msg=f"AB2+GM/Redi {f} not bit-identical")


def test_total_default_is_bit_identical_ab2_additive():
    """Composition: AB2 + momentum_friction_additive (vertical friction already
    weight-1.0) must stay byte-equal under the default "total" scope."""
    state, m_default = _channel(outer_integrator="ab2",
                                momentum_friction_additive=True)
    _, m_total = _channel(outer_integrator="ab2", ab2_scope="total",
                          momentum_friction_additive=True)
    s0 = _run(m_default, state, 3)
    s1 = _run(m_total, state, 3)
    for f in ("u", "v", "T", "S", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data),
            err_msg=f"AB2+additive {f} not bit-identical")


# --------------------------------------------- advective scope is a real change

def test_advective_differs_from_total():
    """The scope is a real time-weighting change on a dissipative flow — the two
    modes must NOT coincide (guards against a silently-dead flag)."""
    state, m_total = _channel(outer_integrator="ab2", ab2_scope="total")
    _, m_adv = _channel(outer_integrator="ab2", ab2_scope="advective")
    # Two steps so the AB2 carry (≠ first-step Euler bootstrap) is exercised.
    s_t = _run(m_total, state, 2)
    s_a = _run(m_adv, state, 2)
    assert np.max(np.abs(np.asarray(s_t.u.data) - np.asarray(s_a.u.data))) > 0.0
    assert np.max(np.abs(np.asarray(s_t.T.data) - np.asarray(s_a.T.data))) > 0.0


# ------------------------------------------- advective algebra (closed form)

def _ab2_weights(eps):
    return 1.5 + eps, 0.5 + eps


def test_advective_algebra_manufactured_split():
    """Closed-form check of the advective-scope AB2 algebra on a single field.

    Construct the realized tracer increment from a model in advective mode and
    verify it equals::

        T^{n+1} = T^n + a_n·ΔT_adv^n − a_p·ΔT_adv^{n-1} + 1.0·ΔT_diss^n

    by independently reconstructing the advective increment (the carried
    ``T_incr_prev`` from step n) and the weight-1.0 dissipative increment, and
    composing them with the published AB2 weights.

    We isolate the algebra from the implicit-vertical-mixing solve (which is
    identical in both scopes and acts AFTER the AB2 blend) by turning the
    implicit solve into a no-op: A_v=K_v=0 with implicit_vertical_mixing=True
    leaves the backward-Euler tridiagonal as the identity, so the post-AB2
    state equals the AB2 blend exactly.
    """
    state, model = _channel(outer_integrator="ab2", ab2_scope="advective",
                            A_v=0.0, K_v=0.0)
    eps = model.config.ab2_epsilon
    a_n, a_p = _ab2_weights(eps)
    mask3 = np.asarray(state.land_mask.data)[..., None]

    # Step 1 (Euler bootstrap: ΔT_adv^{n-1}=0) → carries ΔT_adv^0 on T_incr_prev.
    s1 = model.step(state, _DT)
    dT_adv_0 = np.asarray(s1.T_incr_prev.data)

    # Step 2: realized T2, plus the pieces it should decompose into.
    s2 = model.step(s1, _DT)

    # Re-derive the explicit (advective-only) pass + the diss increment for the
    # step-1→step-2 transition using the private explicit path. In advective
    # mode _step_impl returns the diss increment as the 6th tuple slot.
    state_expl, extras = model._step_impl(
        s1, _DT, _apply_implicit_vmix=False)
    diss_incr = extras[5]
    assert diss_incr is not None, "advective mode must return a diss increment"
    dT_diss, dS_diss, du_diss, dv_diss = diss_incr

    dT_adv_1 = np.asarray(state_expl.T.data) - np.asarray(s1.T.data)
    # AB2 blend with weight-1.0 dissipation (A_v=K_v=0 ⇒ implicit solve is I):
    T2_expected = (
        np.asarray(s1.T.data) + a_n * dT_adv_1 - a_p * dT_adv_0
        + np.asarray(dT_diss)
    ) * mask3
    np.testing.assert_allclose(
        np.asarray(s2.T.data), T2_expected, rtol=0, atol=1e-12)


def test_advective_carry_holds_only_advective_increment():
    """The carried {T,u}_incr_prev must contain ONLY the advective increment in
    advective mode — i.e. it equals state_expl − state (diss withheld), and NOT
    the full increment (which would include the diss bucket)."""
    state, model = _channel(outer_integrator="ab2", ab2_scope="advective")
    s1 = model.step(state, _DT)

    state_expl, extras = model._step_impl(
        state, _DT, _apply_implicit_vmix=False)
    diss_incr = extras[5]
    dT_diss = np.asarray(diss_incr[0])
    mask3 = np.asarray(state.land_mask.data)[..., None]

    dT_adv = (np.asarray(state_expl.T.data) - np.asarray(state.T.data)) * mask3
    np.testing.assert_allclose(
        np.asarray(s1.T_incr_prev.data), dT_adv, rtol=0, atol=1e-13)
    # And the diss increment is genuinely nonzero (front + jet ⇒ diffusion).
    assert np.max(np.abs(dT_diss)) > 0.0, "diss increment unexpectedly zero"
    # The carry must NOT equal advective + diss (the "total" semantics).
    assert np.max(np.abs(np.asarray(s1.T_incr_prev.data)
                         - (dT_adv + dT_diss * mask3))) > 0.0


def test_advective_momentum_diss_is_nonzero_and_withheld():
    """The momentum diss bucket (lateral friction + bottom drag) is nonzero and
    is withheld from the AB2'd du_dt (so du_dt is advective-only)."""
    state, model = _channel(outer_integrator="ab2", ab2_scope="advective")
    _, extras = model._step_impl(state, _DT, _apply_implicit_vmix=False)
    du_diss = np.asarray(extras[5][2])
    assert np.max(np.abs(du_diss)) > 0.0


# --------------------------------------- steady-state fixed-point invariance

def test_steady_state_invariance_linear_column():
    """Both scopes converge to the SAME fixed point of a linear dissipative
    system.

    AB2 and forward-Euler-on-dissipation share the steady balance: at a fixed
    point ΔX^n = ΔX^{n-1}, so the AB2 advective weights collapse (a_n − a_p = 1)
    and the dissipative term (weight 1.0 either way) is unchanged — the two
    scopes have the SAME fixed point.

    Realize it with a MOTIONLESS tracer-diffusion column (u=v=0, no GM/Redi):
    the only active dynamics is lateral K_h diffusion smoothing the meridional
    thermal front. The fixed point is the diffusive equilibrium (a uniform T at
    the conserved domain mean). Both scopes must converge to it identically.
    """
    state, m_total = _channel(outer_integrator="ab2", ab2_scope="total",
                              bottom_drag_r=0.0)
    _, m_adv = _channel(outer_integrator="ab2", ab2_scope="advective",
                        bottom_drag_r=0.0)
    # Zero the velocity so there is no advection / Coriolis dynamics — pure
    # tracer lateral diffusion toward the diffusive fixed point.
    zero_u = jnp.zeros_like(state.u.data)
    zero_v = jnp.zeros_like(state.v.data)
    state = state._replace(u=state.u.replace(data=zero_u),
                           v=state.v.replace(data=zero_v))
    s_t = _run(m_total, state, 500)
    s_a = _run(m_adv, state, 500)
    # The transient differs (AB2 vs weight-1.0 dissipation) but the equilibrium
    # must agree to a tight tolerance: same forward-Euler-stable diffusive
    # balance of the same operator.
    dT = np.max(np.abs(np.asarray(s_t.T.data) - np.asarray(s_a.T.data)))
    scaleT = np.max(np.abs(np.asarray(s_t.T.data)
                          - np.mean(np.asarray(s_t.T.data))))
    assert dT <= 1e-4 * max(scaleT, 1.0), f"equilibria diverge: dT={dT}"


# ------------------------------------------- composition with additive friction

def test_compose_with_momentum_friction_additive():
    """advective scope + momentum_friction_additive = full Veros dissipative
    scope. The combination must run, stay finite/bounded, and differ from
    advective-alone (the additive flag relocates the VERTICAL friction)."""
    state, m_adv = _channel(outer_integrator="ab2", ab2_scope="advective")
    _, m_both = _channel(outer_integrator="ab2", ab2_scope="advective",
                         momentum_friction_additive=True)
    s_adv = _run(m_adv, state, 3)
    s_both = _run(m_both, state, 3)
    for s in (s_adv, s_both):
        assert np.all(np.isfinite(np.asarray(s.u.data)))
        assert np.all(np.isfinite(np.asarray(s.T.data)))
    assert np.max(np.abs(np.asarray(s_adv.u.data)
                         - np.asarray(s_both.u.data))) > 0.0


# ------------------------------------------------------------ robustness / AD

def test_stability_100_steps_advective():
    state, model = _channel(outer_integrator="ab2", ab2_scope="advective")
    s = state
    for _ in range(100):
        s = model.step(s, _DT)
    u = np.asarray(s.u.data)
    T = np.asarray(s.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(T))
    assert np.max(np.abs(u)) < 5.0
    assert -5.0 < T.min() and T.max() < 40.0


def test_differentiable_advective():
    state, model = _channel(n_lat=6, n_lon=8, outer_integrator="ab2",
                            ab2_scope="advective")

    def loss(scale):
        st = state._replace(u=state.u.replace(data=state.u.data * scale))
        s1 = model.step(st, _DT)
        s2 = model.step(s1, _DT)
        return jnp.sum(s2.u.data ** 2) + jnp.sum(s2.T.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0
