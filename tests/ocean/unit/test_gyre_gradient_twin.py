"""Direct controls for the GYRE identical-twin adjoint experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
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


def test_nemo_literal_dissl_zero_vjp_localizes_poison():
    """The literal post-AVN sqrt has a NaN VJP at masked zero TKE."""
    tke = jnp.asarray([0.0, 4.0], dtype=jnp.float64)
    l_eps = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    cotangent = jnp.asarray([0.0, 1.0], dtype=jnp.float64)

    # This is the exact operation at vertical_mixing/tke.py:3434.  Its primal
    # is finite, but sqrt's transpose evaluates 0 / sqrt(0) on a masked entry.
    _, pullback = jax.vjp(lambda value: jnp.sqrt(value) / l_eps, tke)
    literal_vjp = np.asarray(pullback(cotangent)[0])
    assert np.isnan(literal_vjp[0])
    assert literal_vjp[1] == 0.125

    # A double-where guard is primal-identical and demonstrates the
    # discriminating repair, but the production operation is intentionally not
    # changed in this stopped/refuted experiment.
    def reverse_safe(value):
        positive = value > 0.0
        return jnp.where(
            positive,
            jnp.sqrt(jnp.where(positive, value, 1.0)) / l_eps,
            0.0,
        )

    _, safe_pullback = jax.vjp(reverse_safe, tke)
    safe_vjp = np.asarray(safe_pullback(cotangent)[0])
    np.testing.assert_array_equal(safe_vjp, np.asarray([0.0, 0.125]))


def test_two_step_loss_overrides_no_inert_and_adjoint_fd():
    """Require both real paths to pass FD and leave a complete failure log."""
    G.configure_runtime()
    context = G.build_context()
    base = context.card.recipe.model_config

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
    truth = jax.jit(
        lambda initial: G.rollout_observations(
            context,
            base,
            n_steps=2,
            obs_steps=(2,),
            initial_state=initial,
        ))(context.initial_state)
    truth.T.block_until_ready()
    reference = G.build_reference(context, truth)
    selected = ("A_h", "c_k")
    loss = G.make_loss(
        context, reference, selected, n_steps=2, obs_steps=(2,))
    start = G.start_log_parameters(selected)
    activity_live = {name: jnp.asarray(1.0) for name in selected}
    loss_jit = jax.jit(loss)
    print("[gyre-gradient-test] compiling live loss", flush=True)
    compiled = float(loss_jit(start, activity_live))
    assert compiled > 0.0

    print("[gyre-gradient-test] compiling reverse mode", flush=True)
    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
    started = time.perf_counter()
    _, gradients = value_and_grad(start, activity_live)
    gradient_wall_seconds = time.perf_counter() - started
    rows = G.finite_difference_rows(
        lambda parameters: loss_jit(parameters, activity_live),
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
            lambda parameters: loss_jit(parameters, activity_broken),
            start,
            broken_gradients,
            (severed,),
            steps=(1.0e-3, 1.0e-4),
        )
        assert all(row["status"] == "REFUTED" for row in broken_rows)
