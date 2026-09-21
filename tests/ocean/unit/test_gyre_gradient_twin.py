"""Direct controls for the GYRE identical-twin adjoint experiment."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
import numpy as np
import pytest
from legoesm.ml.loss import volume_weighted_mse

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts/experiment/gyre_gradient_twin.py"
_SPEC = importlib.util.spec_from_file_location("gyre_gradient_twin_tested", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
G = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = G
_SPEC.loader.exec_module(G)


def test_volume_weighted_mse_reuses_weighted_reducer():
    pred = jnp.asarray([1.0, 3.0, 99.0], dtype=jnp.float64)
    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    volume = jnp.asarray([1.0, 3.0, 0.0], dtype=jnp.float64)
    # (1 * 1^2 + 3 * 2^2) / 4
    assert float(volume_weighted_mse(pred, target, volume)) == pytest.approx(3.25)
    mean = jnp.sum(target * volume) / jnp.sum(volume)
    assert float(volume_weighted_mse(target, mean, volume)) == pytest.approx(0.1875)


def test_no_inert_and_fd_controls_fire():
    """The production gates reject a dynamically severed parameter path."""
    selected = ("A_h", "c_k")
    start = G.start_log_parameters(selected)

    def controlled_loss(parameters, activity):
        scores = []
        for name in selected:
            delta = activity[name] * (
                jnp.exp(parameters[name]) - jnp.asarray(G.TRUTH[name]))
            scores.append(delta * delta / jnp.asarray(G.TRUTH[name]) ** 2)
        return jnp.mean(jnp.stack(scores))

    live = {name: jnp.asarray(1.0) for name in selected}
    live_grad = jax.grad(controlled_loss)(start, live)
    G.assert_no_inert(live_grad, selected)
    live_rows = G.finite_difference_rows(
        lambda parameters: controlled_loss(parameters, live),
        start,
        live_grad,
        selected,
    )
    assert all(row["status"] == "CONFIRMED" for row in live_rows), live_rows

    for severed in selected:
        broken = dict(live)
        broken[severed] = jnp.asarray(0.0)
        broken_grad = jax.grad(controlled_loss)(start, broken)
        with pytest.raises(G.InertParameterError, match=severed):
            G.assert_no_inert(broken_grad, (severed,))
        broken_rows = G.finite_difference_rows(
            lambda parameters: controlled_loss(parameters, broken),
            start,
            broken_grad,
            (severed,),
        )
        assert all(row["status"] == "REFUTED" for row in broken_rows)


def test_nemo_literal_dissl_guard_primal_bits_and_zero_vjp():
    """The production post-AVN guard preserves finite primals and zero VJP."""
    values = jnp.asarray(
        [4.0, 0.0, np.nextafter(0.0, 1.0), -1.0, np.nan],
        dtype=jnp.float64,
    )
    l_eps = jnp.asarray([2.0, 3.0, 5.0, 7.0, 11.0], dtype=jnp.float64)

    def reverse_safe(value, divisor):
        positive = value > 0.0
        safe = jnp.where(positive, value, 1.0)
        safe_divisor = jnp.where(positive, divisor, 1.0)
        return jnp.where(
            positive, jnp.sqrt(safe) / safe_divisor, 0.0)

    old_primal = np.asarray(jnp.sqrt(values) / l_eps)
    new_primal = np.asarray(reverse_safe(values, l_eps))
    old_finite = np.isfinite(old_primal)
    np.testing.assert_array_equal(
        new_primal[old_finite].view(np.uint64),
        old_primal[old_finite].view(np.uint64),
    )

    _, pullback = jax.vjp(reverse_safe, values, l_eps)
    tke_vjp, divisor_vjp = map(
        np.asarray, pullback(jnp.ones_like(values)))
    assert np.isfinite(tke_vjp[1])
    assert tke_vjp[1] == 0.0
    assert np.isfinite(divisor_vjp[1])
    assert divisor_vjp[1] == 0.0

    # Production masks can pair zero TKE with a zero mixing length.  Both
    # cotangents must remain zero; guarding only sqrt leaves 0/0 in the
    # inactive quotient's divisor transpose.
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    _, zero_pullback = jax.vjp(reverse_safe, zero, zero)
    zero_tke_vjp, zero_divisor_vjp = zero_pullback(
        jnp.asarray(1.0, dtype=jnp.float64))
    assert float(zero_tke_vjp) == 0.0
    assert float(zero_divisor_vjp) == 0.0

    # Pin the exact running symbol, not a wrapper. Reverting the production
    # guard to ``jnp.sqrt(tke_curr)`` must make this direct regression red.
    source = inspect.getsource(tke_mod.tke_vertical_mixing)
    assert "_dissl_sqrt_input = jnp.where(" in source
    assert "_dissl_divisor = jnp.where(" in source
    assert "jnp.sqrt(_dissl_sqrt_input)" in source


def test_two_step_loss_overrides_no_inert_and_adjoint_fd():
    """Require both real paths to pass FD and leave a complete failure log."""
    G.configure_runtime()
    context = G.build_context()
    base = context.card.recipe.model_config
    assert "lax.scan" not in inspect.getsource(G.rollout_observations)
    assert "jax.jit(jax.checkpoint" in inspect.getsource(G.build_context)
    assert context.initial_state.bt_hist is None

    # Direct reachability/control for the two nested NamedTuple leaves.
    overridden = G.config_with_parameters(
        base,
        {"A_h": jnp.asarray(2.5e5), "c_k": jnp.asarray(0.23)},
    )
    assert float(overridden.lateral_viscosity.A_h) == 2.5e5
    assert float(overridden.physics.vertical_mixing.tke.c_k) == 0.23
    config_grad = jax.grad(
        lambda x: (
            G.config_with_parameters(
                base, {"A_h": jnp.exp(x[0]), "c_k": jnp.exp(x[1])})
            .lateral_viscosity.A_h
            + G.config_with_parameters(
                base, {"A_h": jnp.exp(x[0]), "c_k": jnp.exp(x[1])})
            .physics.vertical_mixing.tke.c_k
        )
    )(jnp.log(jnp.asarray([2.0e5, 0.2], dtype=jnp.float64)))
    np.testing.assert_allclose(np.asarray(config_grad), np.asarray([2.0e5, 0.2]))

    print("[gyre-gradient-test] compiling truth rollout", flush=True)
    truth = G.rollout_observations(
        context,
        base,
        n_steps=2,
        obs_steps=(2,),
        initial_state=context.initial_state,
    )
    truth.T.block_until_ready()

    # Decision 49 control: the parameterized loop must reproduce two literal
    # calls through the campaign's public ``model.step`` program bit for bit.
    campaign_state = context.initial_state
    for kt in range(1, 3):
        freshwater, surface = G.gyre_surface_forcings(
            context.card, campaign_state, kt)
        campaign_state = context.model.step(
            campaign_state,
            dt=context.card.dt_s,
            freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=None,
        )
    for name, expected in (
        ("T", campaign_state.T.data),
        ("S", campaign_state.S.data),
        ("u", campaign_state.u.data),
        ("v", campaign_state.v.data),
        ("eta", campaign_state.eta.data),
    ):
        actual = np.asarray(getattr(truth, name)[-1])
        expected_array = np.asarray(expected)
        np.testing.assert_array_equal(
            actual.view(np.uint64), expected_array.view(np.uint64))

    reference = G.build_reference(context, truth)
    selected = ("A_h", "c_k")
    loss = G.make_loss(
        context, reference, selected, n_steps=2, obs_steps=(2,))
    start = G.start_log_parameters(selected)
    activity_live = {name: jnp.asarray(1.0) for name in selected}
    print("[gyre-gradient-test] evaluating live loss", flush=True)
    compiled = float(loss(start, activity_live))
    assert compiled > 0.0

    print("[gyre-gradient-test] compiling reverse mode", flush=True)
    value_and_grad = jax.value_and_grad(loss, argnums=0)
    started = time.perf_counter()
    _, gradients = value_and_grad(start, activity_live)
    jax.block_until_ready(gradients)
    gradient_wall_seconds = time.perf_counter() - started
    rows = G.finite_difference_rows(
        lambda parameters: loss(parameters, activity_live),
        start,
        gradients,
        selected,
        steps=(1.0e-3, 1.0e-4),
    )
    print(
        "[gyre-gradient-test] "
        + json.dumps(
            {
                "gradient_wall_seconds": gradient_wall_seconds,
                "gradients": {
                    name: str(float(np.asarray(gradients[name])))
                    for name in selected
                },
                "fd_rows": rows,
                "peak_rss_mib": G._rss_mib(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # This assertion is deliberately after the FD measurements: a poisoned AD
    # path must leave a complete REFUTED comparison in the test log.
    G.assert_no_inert(gradients, selected)
    assert len(rows) == 4
    assert all(row["status"] == "CONFIRMED" for row in rows), rows

    # Planted violations: remove one selected leaf from the forward config.
    # The same production gate must fail, and the FD diagnostic must not be
    # able to manufacture a passing ratio from 0/0.
    for severed in selected:
        activity_broken = dict(activity_live)
        activity_broken[severed] = jnp.asarray(0.0)
        _, broken_gradients = value_and_grad(start, activity_broken)
        with pytest.raises(G.InertParameterError, match=severed):
            G.assert_no_inert(broken_gradients, (severed,))
        broken_rows = G.finite_difference_rows(
            lambda parameters: loss(parameters, activity_broken),
            start,
            broken_gradients,
            (severed,),
            steps=(1.0e-3, 1.0e-4),
        )
        assert all(row["status"] == "REFUTED" for row in broken_rows)
