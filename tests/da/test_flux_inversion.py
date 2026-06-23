"""Boundary / surface-flux inversion: ForcingControlSpec + build_flux_cost_fn.

A controllable twin model (h_{t+1} = h_t + flux_t * dt) lets us check the full
contract: the time-varying forcing control round-trips, the cost gradient is
correct (vs finite differences), all checkpoint schedules agree, and a real
optimisation RECOVERS the true forcing from state observations.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.da.background_error import DiagonalB  # noqa: E402
from legoesm.da.control_vector import (  # noqa: E402
    apply_forcing_slice,
    build_forcing_control_spec,
    control_to_forcing_series,
    forcing_series_to_control,
)
from legoesm.da.cost_function import (  # noqa: E402
    build_flux_cost_and_grad_fn,
    build_flux_cost_fn,
)
from legoesm.da.observation import DirectObsOperator, Observation  # noqa: E402


class _FState(NamedTuple):
    h: jax.Array


class _FForcing(NamedTuple):
    flux: jax.Array  # invertible surface flux
    bias: jax.Array  # uncontrolled leaf (held at template value)


def _step(state, forcing, dt):
    return _FState(h=state.h + forcing.flux * dt)


def _template(ncell=3):
    return _FForcing(flux=jnp.zeros(ncell), bias=jnp.ones(ncell) * 0.5)


# --------------------------------------------------------------------------
# ForcingControlSpec
# --------------------------------------------------------------------------

def test_forcing_spec_default_selects_all_array_leaves():
    spec = build_forcing_control_spec(_template(3), n_window=4)
    assert [e.field_name for e in spec.entries] == ["flux", "bias"]
    assert spec.total_size == 4 * 3 + 4 * 3
    assert spec.n_window == 4


def test_forcing_spec_subset_fields():
    spec = build_forcing_control_spec(_template(3), 4, fields=("flux",))
    assert [e.field_name for e in spec.entries] == ["flux"]
    assert spec.total_size == 12


def test_forcing_roundtrip_identity():
    spec = build_forcing_control_spec(_template(3), 4, fields=("flux",))
    series = {"flux": jax.random.normal(jax.random.PRNGKey(0), (4, 3))}
    x = forcing_series_to_control(series, spec)
    back = control_to_forcing_series(x, spec)
    assert jnp.allclose(back["flux"], series["flux"])


def test_forcing_roundtrip_log_transform():
    spec = build_forcing_control_spec(
        _template(2), 3, fields=("flux",), transforms={"flux": "log"})
    series = {"flux": jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (3, 2))) + 0.1}
    x = forcing_series_to_control(series, spec)
    back = control_to_forcing_series(x, spec)
    assert jnp.allclose(back["flux"], series["flux"], rtol=1e-6)


def test_apply_forcing_slice_preserves_uncontrolled_leaf():
    tmpl = _template(3)
    spec = build_forcing_control_spec(tmpl, 4, fields=("flux",))
    series = {"flux": jnp.arange(12.0).reshape(4, 3)}
    f2 = apply_forcing_slice(tmpl, series, 2)
    assert jnp.allclose(f2.flux, series["flux"][2])
    assert jnp.allclose(f2.bias, tmpl.bias)  # uncontrolled leaf inherited


def test_build_forcing_spec_rejects_bad_n_window():
    with pytest.raises(ValueError, match="n_window"):
        build_forcing_control_spec(_template(2), 0)


def test_forcing_series_to_control_rejects_wrong_shape():
    spec = build_forcing_control_spec(_template(3), 4, fields=("flux",))
    # same size (4*3 == 3*4 == 12) but transposed axes => must be rejected
    with pytest.raises(ValueError, match="expected"):
        forcing_series_to_control({"flux": jnp.zeros((3, 4))}, spec)


def test_forcing_series_to_control_rejects_missing_field():
    spec = build_forcing_control_spec(_template(3), 4, fields=("flux",))
    with pytest.raises(ValueError, match="missing field"):
        forcing_series_to_control({"bias": jnp.zeros((4, 3))}, spec)


def test_control_to_forcing_series_rejects_wrong_size():
    spec = build_forcing_control_spec(_template(3), 4, fields=("flux",))
    with pytest.raises(ValueError, match="expected"):
        control_to_forcing_series(jnp.zeros(spec.total_size + 1), spec)
    with pytest.raises(ValueError, match="expected"):
        control_to_forcing_series(jnp.zeros((spec.total_size, 1)), spec)


# --------------------------------------------------------------------------
# build_flux_cost_fn
# --------------------------------------------------------------------------

def _make_problem(ncell=2, n_steps=4, seed=0):
    tmpl = _FForcing(flux=jnp.zeros(ncell), bias=jnp.zeros(ncell))
    spec = build_forcing_control_spec(tmpl, n_window=n_steps, fields=("flux",))
    init = _FState(h=jnp.zeros(ncell))
    dt = 1.0
    true_flux = jax.random.normal(jax.random.PRNGKey(seed), (n_steps, ncell))
    true_x = forcing_series_to_control({"flux": true_flux}, spec)
    # synthetic noiseless obs of h at EVERY step (constrains each flux slice)
    series = control_to_forcing_series(true_x, spec)
    op = DirectObsOperator("h", (jnp.arange(ncell),))
    state = init
    obs_list = []
    for i in range(n_steps):
        state = _step(state, apply_forcing_slice(tmpl, series, i), dt)
        obs_list.append(Observation(
            values=op(state), errors=jnp.ones(ncell) * 0.01,
            time_index=i, operator=op,
        ))
    B = DiagonalB(sigma=jnp.ones(spec.total_size) * 10.0)  # weak prior
    return dict(step=_step, init=init, tmpl=tmpl, spec=spec,
                observations=tuple(obs_list), B=B, dt=dt, n_steps=n_steps,
                true_x=true_x)


def _cg(p, bg, schedule="uniform"):
    return build_flux_cost_and_grad_fn(
        p["step"], p["init"], bg, p["observations"], p["B"],
        p["spec"], p["tmpl"], p["dt"], p["n_steps"],
        checkpoint_schedule=schedule,
    )


def test_flux_cost_gradient_finite_nonzero():
    p = _make_problem()
    bg = jnp.zeros_like(p["true_x"])
    J, g = _cg(p, bg)(bg)
    assert jnp.isfinite(J)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(jnp.abs(g) > 0.0)


def test_flux_cost_zero_at_truth():
    p = _make_problem()
    cost = build_flux_cost_fn(
        p["step"], p["init"], p["true_x"], p["observations"], p["B"],
        p["spec"], p["tmpl"], p["dt"], p["n_steps"])
    # background == truth (J_b=0) and noiseless obs from truth (J_o=0)
    assert float(cost(p["true_x"])) < 1e-8


@pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
def test_flux_cost_schedules_agree(schedule):
    p = _make_problem()
    bg = jnp.zeros_like(p["true_x"])
    J_u, g_u = _cg(p, bg, "uniform")(bg)
    J, g = _cg(p, bg, schedule)(bg)
    assert jnp.allclose(J, J_u, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(g, g_u, rtol=1e-7, atol=1e-9)


def test_flux_cost_grad_matches_finite_difference():
    p = _make_problem()
    bg = jnp.zeros_like(p["true_x"])
    cost = build_flux_cost_fn(
        p["step"], p["init"], bg, p["observations"], p["B"],
        p["spec"], p["tmpl"], p["dt"], p["n_steps"])
    g = jax.grad(cost)(bg)
    eps = 1e-5
    for k in range(min(6, bg.shape[0])):
        e = jnp.zeros_like(bg).at[k].set(1.0)
        fd = (cost(bg + eps * e) - cost(bg - eps * e)) / (2 * eps)
        assert jnp.allclose(g[k], fd, rtol=1e-4, atol=1e-6)


def test_flux_inversion_recovers_true_forcing():
    p = _make_problem(ncell=2, n_steps=4)
    bg = jnp.zeros_like(p["true_x"])
    cost_and_grad = _cg(p, bg, "binomial")
    opt = optax.adam(3e-2)
    opt_state = opt.init(bg)

    @jax.jit
    def gd_step(x, opt_state):
        _, g = cost_and_grad(x)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(x, upd), opt_state

    x = bg
    for _ in range(3000):
        x, opt_state = gd_step(x, opt_state)

    err0 = float(jnp.linalg.norm(bg - p["true_x"]))
    err = float(jnp.linalg.norm(x - p["true_x"]))
    assert err < 0.05 * err0, f"recovered forcing error {err:.3e} vs initial {err0:.3e}"


def test_flux_n_window_must_equal_n_steps():
    p = _make_problem(n_steps=4)
    with pytest.raises(ValueError, match="n_window"):
        build_flux_cost_fn(
            p["step"], p["init"], jnp.zeros(p["spec"].total_size),
            p["observations"], p["B"], p["spec"], p["tmpl"], p["dt"], n_steps=5)


def test_flux_cost_rejects_malformed_background():
    """A non-1D background must be rejected at build time so it cannot broadcast
    into the prior term J_b (codex flux round-2)."""
    p = _make_problem(n_steps=4)
    bad_bg = jnp.zeros((p["spec"].total_size, 1))
    with pytest.raises(ValueError, match="expected a 1-D"):
        build_flux_cost_fn(
            p["step"], p["init"], bad_bg, p["observations"], p["B"],
            p["spec"], p["tmpl"], p["dt"], p["n_steps"])
