"""Direct controls for the certified GYRE gradient-gate instrument."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
import numpy as np
import pytest
from jax import test_util as jtu

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradient_gate.py"
)
_GPU_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradgate_gpu_rows.sh"
)
_ROUND9_RUN_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradgate_round9_run.sh"
)
_ROUND10_RUN_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradgate_round10_run.sh"
)
_SPEC = importlib.util.spec_from_file_location("gyre_gradient_gate_tested", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
G = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = G
_SPEC.loader.exec_module(G)


def test_gpu_runner_continues_every_classified_outcome():
    result = subprocess.run(
        [str(_GPU_SCRIPT), "--self-test"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == (
        "GYRE_GRADGATE_GPU_RUNNER_SELF_TEST statuses="
        "REFUTED UNMEASURED VACUOUS RETRACTED CONFIRMED"
    )


def test_round9_operator_entrypoint_reaches_fail_closed_runner():
    result = subprocess.run(
        [str(_ROUND9_RUN_SCRIPT), "--self-test"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().startswith(
        "GYRE_GRADGATE_GPU_RUNNER_SELF_TEST statuses=")


def test_round10_operator_entrypoint_reaches_fail_closed_runner():
    result = subprocess.run(
        [str(_ROUND10_RUN_SCRIPT), "--self-test"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().startswith(
        "GYRE_GRADGATE_GPU_RUNNER_SELF_TEST statuses=")


def test_gpu_runner_creates_evidence_directory_before_acquisition():
    source = _GPU_SCRIPT.read_text(encoding="utf-8")
    create = source.index('mkdir -p -- "$EVIDENCE_DIR"')
    existence_check = source.index(
        '[[ -d "$EVIDENCE_DIR" ]] || refuse "could not create evidence directory')
    first_row = source.index("run_row()")
    assert create < existence_check < first_row


def test_broken_vjp_plant_flips_passing_row():
    base = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    direction = jnp.asarray([3.0, -1.0], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    def loss(value):
        return jnp.sum(value * value)

    control = G.directional_derivatives(loss, base, direction)
    planted = G.directional_derivatives(loss, base, direction, plant=True)
    assert control["status"] == "CONFIRMED"
    assert planted["status"] == "REFUTED"
    assert planted["reverse"] == pytest.approx(2.0 * planted["forward"])


def test_nonfinite_agreement_row_is_not_labeled_nonzero():
    row = G.classify_agreement(np.nan, np.nan, np.nan)
    assert row["finite"] is False
    assert row["nonzero"] is False
    assert row["status"] == "REFUTED"


def test_absent_derivatives_are_unmeasured_not_refuted():
    row = G.unmeasured_agreement("planted pre-derivative refusal")
    assert row["status"] == "UNMEASURED"
    assert row["error"] == "planted pre-derivative refusal"
    for field in (
        "reverse", "forward", "finite_difference",
        "rev_fwd_relative_error", "rev_fd_relative_error",
    ):
        assert row[field] is None


def test_scalar_line_reverse_equals_full_vjp_dot_direction():
    base = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)
    direction = jnp.asarray([0.5, -0.1, 0.3], dtype=jnp.float64)

    def loss(value):
        return jnp.sum(jnp.sin(value) + value**3)

    full_gradient = jnp.asarray(jax.grad(loss)(base))
    expected = float(jnp.vdot(full_gradient, direction))
    row = G.directional_derivatives(loss, base, direction)
    assert row["reverse"] == pytest.approx(expected, rel=1.0e-13)
    assert row["gradient_dot_abs_sum"] == pytest.approx(
        float(jnp.sum(jnp.abs(full_gradient * direction))))
    assert row["gradient_direction_norm_product"] == pytest.approx(
        float(jnp.linalg.norm(full_gradient) * jnp.linalg.norm(direction)))
    assert row["rev_fwd_scale_aware_error"] == pytest.approx(0.0, abs=1.0e-15)


def test_explicit_stepwise_vjp_matches_whole_window_gradient(monkeypatch):
    monkeypatch.setattr(
        G,
        "state_bitwise_report",
        lambda _expected, _actual: {
            "bitwise_equal": True,
            "mismatch_count": 0,
            "mismatches": [],
        },
    )

    class Field(NamedTuple):
        data: jax.Array

        def replace(self, **changes):
            return self._replace(**changes)

    class State(NamedTuple):
        T: Field

    class Context(NamedTuple):
        initial_state: State
        card: object
        model: object
        surface_forcings: object
        observed_ad_step: object

    class Card(NamedTuple):
        dt_s: float

    class Model:
        def _step_jitted(self, state, _dt_s, freshwater, surface):
            return State(Field(
                state.T.data * (1.0 + freshwater) + 0.01 * surface))

    def forcing(_card, state, kt):
        return 0.1 * kt, state.T.data**2

    initial = State(Field(jnp.asarray([0.2, -0.4], dtype=jnp.float64)))
    card = Card(1.0)
    model = Model()

    def raw_step(state, kt):
        freshwater, surface = forcing(card, state, kt)
        return model._step_jitted(state, card.dt_s, freshwater, surface)

    context = Context(
        initial, card, model, forcing,
        jax.jit(G.build_observed_primal_preserving_step(raw_step, raw_step)),
    )
    monkeypatch.setattr(
        G,
        "loss_at_state",
        lambda _reference, state: jnp.sum(jnp.sin(state.T.data)),
    )
    base = jnp.asarray([0.01, -0.02], dtype=jnp.float64)
    direction = jnp.asarray([0.6, -0.8], dtype=jnp.float64)
    row = G.explicit_stepwise_directional_derivatives(
        context, None, base, direction, 3)
    planted = G.explicit_stepwise_directional_derivatives(
        context, None, base, direction, 3, plant=True)

    def whole_window(value):
        state = G._apply_initial_perturbation(context, value)
        for kt in range(1, 4):
            state = G._production_step(context, state, kt)
        return G.loss_at_state(None, state)

    expected_gradient = jax.grad(whole_window)(base)
    assert row["reverse"] == pytest.approx(
        float(jnp.vdot(expected_gradient, direction)), rel=1e-13)
    assert row["gradient_dot_abs_sum"] == pytest.approx(
        float(jnp.sum(jnp.abs(expected_gradient * direction))), rel=1e-13)
    assert set(row["step_vjp_boundary_reports"]) == {"1", "2", "3"}
    assert row["step_vjp_boundary_outputs_bitwise"] is True
    assert set(row["step_vjp_replay_reports"]) == {"1", "2", "3"}
    assert row["step_vjp_replay_used_as_primal"] is False
    assert set(row["step_jvp_boundary_reports"]) == {"1", "2", "3"}
    assert row["step_jvp_boundary_outputs_bitwise"] is True
    assert set(row["step_jvp_replay_reports"]) == {"1", "2", "3"}
    assert row["step_jvp_replay_used_as_primal"] is False
    assert row["step_transform_compile_footprint"] == {
        "whole_window_transform_jitted": False,
        "distinct_state_pytree_specializations": 2,
        "cold_program_invocations": 1,
        "warm_program_invocations": 2,
        "vjp_step_invocations": 3,
        "jvp_step_invocations": 3,
        "dynamic_model_step_argument": True,
        "production_step_traces_per_specialization": 2,
    }
    assert planted["status"] == "REFUTED"
    assert planted["reverse"] == pytest.approx(2.0 * planted["forward"])


def test_compiled_step_transforms_reuse_cold_and_warm_programs():
    class Field(NamedTuple):
        data: jax.Array

        def replace(self, **changes):
            return self._replace(**changes)

    class State(NamedTuple):
        T: Field
        tke: Field | None

    class Card(NamedTuple):
        dt_s: float

    class Context(NamedTuple):
        initial_state: State
        card: Card
        model: object
        surface_forcings: object
        observed_ad_step: object

    class Model:
        def __init__(self):
            self.traces = 0

        def _step_jitted(self, state, _dt_s, freshwater, surface):
            self.traces += 1
            next_t = state.T.data * (1.0 + freshwater) + surface
            next_tke = (
                Field(jnp.zeros_like(next_t))
                if state.tke is None else state.tke
            )
            return State(Field(next_t), next_tke)

    def forcing(_card, _state, kt):
        return 0.01 * kt, 0.001 * kt

    model = Model()
    cold = State(Field(jnp.asarray([0.2, -0.4], dtype=jnp.float64)), None)
    def raw_step(state, kt):
        freshwater, surface = forcing(Card(1.0), state, kt)
        return model._step_jitted(state, 1.0, freshwater, surface)

    observed = jax.jit(G.build_observed_primal_preserving_step(
        raw_step, raw_step))
    context = Context(cold, Card(1.0), model, forcing, observed)
    states = [cold]
    for kt in range(1, 4):
        states.append(G._production_step(context, states[-1], kt))
    model.traces = 0
    compiled_vjp, compiled_jvp = G.build_compiled_step_transforms(context)

    cotangent = jax.tree_util.tree_map(jnp.ones_like, states[-1])
    for kt in range(3, 0, -1):
        _selected, _replay, cotangent = compiled_vjp(
            states[kt - 1], jnp.asarray(kt, dtype=jnp.int32), states[kt],
            cotangent)
        jax.block_until_ready(cotangent)

    tangent = State(Field(jnp.ones_like(cold.T.data)), None)
    for kt in range(1, 4):
        _selected, _replay, tangent = compiled_jvp(
            states[kt - 1], jnp.asarray(kt, dtype=jnp.int32), states[kt],
            tangent)
        jax.block_until_ready(tangent)

    # VJP/JVP each trace the ordinary production and transformed replay arms
    # once for each cold/warm pytree specialization.
    assert model.traces == 8


def test_compiled_step_transform_uses_materialized_production_primal():
    class Field(NamedTuple):
        data: jax.Array

        def replace(self, **changes):
            return self._replace(**changes)

    class State(NamedTuple):
        T: Field

    class Context(NamedTuple):
        observed_ad_step: object

    @jax.custom_jvp
    def replay_source(value):
        return jnp.nextafter(value, jnp.full_like(value, jnp.inf))

    @replay_source.defjvp
    def replay_source_jvp(primals, tangents):
        (value,), (value_dot,) = primals, tangents
        return replay_source(value), value_dot

    def mismatching_observed_step(state, _kt):
        replay = State(Field(replay_source(state.T.data)))
        return replay, replay

    state = State(Field(jnp.asarray([-0.0, 1.0], dtype=jnp.float64)))
    production_after = State(Field(jnp.asarray([-0.0, 2.0], dtype=jnp.float64)))
    cotangent = State(Field(jnp.ones_like(state.T.data)))
    tangent = State(Field(jnp.ones_like(state.T.data)))
    context = Context(jax.jit(mismatching_observed_step))

    compiled_vjp, compiled_jvp = G.build_compiled_step_transforms(context)
    vjp_selected, vjp_replay, _ = compiled_vjp(
        state, jnp.asarray(1, dtype=jnp.int32), production_after, cotangent)
    jvp_selected, jvp_replay, _ = compiled_jvp(
        state, jnp.asarray(1, dtype=jnp.int32), production_after, tangent)
    planted_vjp, planted_jvp = G.build_compiled_step_transforms(
        context, plant_replay_primal=True)
    vjp_plant, _, _ = planted_vjp(
        state, jnp.asarray(1, dtype=jnp.int32), production_after, cotangent)
    jvp_plant, _, _ = planted_jvp(
        state, jnp.asarray(1, dtype=jnp.int32), production_after, tangent)

    for selected in (vjp_selected, jvp_selected):
        assert G.states_bitwise_equal(selected, production_after)
    for replay in (vjp_replay, jvp_replay, vjp_plant, jvp_plant):
        assert not G.states_bitwise_equal(replay, production_after)


def test_local_ad_boundary_preserves_primal_and_uses_replay_derivative():
    @jax.custom_jvp
    def production_source(value):
        return value

    @production_source.defjvp
    def production_source_jvp(primals, tangents):
        (value,), (value_dot,) = primals, tangents
        transformed = jnp.nextafter(value, jnp.full_like(value, jnp.inf))
        return transformed, 3.0 * value_dot

    @jax.custom_jvp
    def replay_source(value):
        return jnp.nextafter(value, jnp.full_like(value, jnp.inf))

    @replay_source.defjvp
    def replay_source_jvp(primals, tangents):
        (value,), (value_dot,) = primals, tangents
        replay = jnp.nextafter(value, jnp.full_like(value, jnp.inf))
        return replay, 7.0 * value_dot

    values = jnp.asarray([-0.0, 1.0], dtype=jnp.float64)
    direction = jnp.ones_like(values)
    bounded = G.build_primal_preserving_step(
        production_source, replay_source)
    planted = G.build_primal_preserving_step(
        production_source, replay_source, plant_replay_primal=True)

    ordinary = bounded(values)
    compiled = jax.jit(bounded)(values)
    jvp_primal, tangent = jax.jvp(
        bounded, (values,), (direction,))
    vjp_primal, pullback = jax.vjp(bounded, values)
    reverse = pullback(direction)[0]
    planted_primal, _ = jax.jvp(
        planted, (values,), (direction,))
    observed = G.build_observed_primal_preserving_step(
        production_source, replay_source)
    (observed_primal, raw_replay), _ = jax.jvp(
        observed, (values,), (direction,))

    for candidate in (
        ordinary, compiled, jvp_primal, vjp_primal, observed_primal,
    ):
        np.testing.assert_array_equal(
            np.asarray(candidate).view(np.uint64),
            np.asarray(values).view(np.uint64),
        )
    np.testing.assert_array_equal(np.asarray(tangent), np.full(2, 7.0))
    np.testing.assert_array_equal(np.asarray(reverse), np.full(2, 7.0))
    assert G.state_bitwise_report(values, raw_replay)["bitwise_equal"] is False
    assert G.state_bitwise_report(values, planted_primal)["bitwise_equal"] is False

    smooth = G.build_primal_preserving_step(
        lambda value: jnp.sin(value) + value**3,
        lambda value: jnp.sin(value) + value**3,
    )
    jtu.check_grads(
        smooth,
        (jnp.asarray([0.2, -0.4], dtype=jnp.float64),),
        order=2,
        modes=("fwd", "rev"),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_transform_boundary_byte_gate_refuses_a_planted_mismatch():
    measured = G.classify_agreement(1.0, 1.0, 1.0)
    measured.update({
        "gradient_dot_abs_sum": 2.0,
        "gradient_direction_norm_product": 3.0,
        "rev_fwd_absolute_discrepancy": 0.0,
        "rev_fwd_scale_aware_error": 0.0,
    })
    good = {"1": {"bitwise_equal": True, "mismatch_count": 0}}
    planted = {"1": {"bitwise_equal": False, "mismatch_count": 1}}

    accepted = G.enforce_step_replay_byte_gate(
        measured.copy(), good, good, planted, planted)
    refused = G.enforce_step_replay_byte_gate(
        measured.copy(), planted, good, good, good)

    assert accepted["status"] == "CONFIRMED"
    assert refused["status"] == "UNMEASURED"
    assert refused["reverse"] is None
    assert refused["forward"] is None
    assert refused["finite_difference"] is None
    assert refused["error"] == (
        "production-step local AD boundary changed the production primal")
    assert refused["refused_derivative_evidence"]["status"] == "CONFIRMED"
    assert refused["refused_derivative_evidence"]["reverse"] == 1.0
    assert refused["step_vjp_boundary_outputs_bitwise"] is False
    assert refused["step_jvp_boundary_outputs_bitwise"] is True
    assert refused["step_vjp_replays_bitwise"] is True


def test_gyre_resource_schedule_starts_after_cold_step(monkeypatch):
    context = G.build_context()
    assert context.card.case == G.CASE
    assert context.initial_state.tke is None
    assert context.initial_state.tke_avm is not None
    calls = []

    def fake_ad_production_step(_context, state, kt, *, remat=False):
        assert remat is False
        calls.append(int(np.asarray(kt)))
        return state._replace(tke=context.initial_state.tke_avm)

    def fake_checkpointed_loop(body, init, n_steps, *, schedule):
        assert schedule == "uniform"
        assert init[1].tke is not None, "cold GYRE state entered checkpoint loop"
        carry = init
        for _ in range(n_steps):
            carry = body(carry)
        return carry

    monkeypatch.setattr(G, "_ad_production_step", fake_ad_production_step)
    monkeypatch.setattr(G, "checkpointed_loop", fake_checkpointed_loop)
    zero = jnp.zeros_like(context.initial_state.T.data)
    result = G.scheduled_rollout(context, zero, 6, "uniform")

    assert result.tke is not None
    assert calls == [1, 2, 3, 4, 5, 6]


def test_reverse_only_schedule_compares_against_forward_reference():
    base = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)
    direction = jnp.asarray([0.5, -0.1, 0.3], dtype=jnp.float64)

    def forward_loss(value):
        return jnp.sum(jnp.sin(value) + value**3)

    def reverse_loss(value):
        return G.checkpointed_loop(
            lambda carry: carry,
            forward_loss(value),
            2,
            schedule="binomial",
        )

    row = G.directional_derivatives(
        forward_loss, base, direction, reverse_loss=reverse_loss)
    assert row["status"] == "CONFIRMED"


def _healthy_resource_row():
    return {
        "resource_schema": G.RESOURCE_SCHEMA,
        "reverse_transform": G.REVERSE_TRANSFORM,
        "warm_reverse_timer_span": G.WARM_TIMER_SPAN,
        "case": G.CASE,
        "schedule": "uniform",
        "backend": "cpu",
        "steps": 6,
        "status": "CONFIRMED",
        "resource_status": "CONFIRMED",
        "gradient_finite": True,
        "gradient_nonzero": True,
        "schedule_primal_bitwise_production": True,
        "loss": 1.0,
        "cold_compile_and_reverse_wall_seconds": 2.0,
        "warm_reverse_wall_seconds": 1.0,
        "memory_baseline_mib": 100.0,
        "memory_peak_mib": 110.0,
        "memory_increment_mib": 10.0,
        "memory_samples": 3,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("gradient_finite", False),
        ("gradient_nonzero", False),
        ("schedule_primal_bitwise_production", False),
        ("loss", np.nan),
        ("warm_reverse_wall_seconds", 0.0),
        ("memory_peak_mib", np.nan),
        ("memory_samples", 1),
    ],
)
def test_resource_rows_fail_closed(field, bad_value):
    row = _healthy_resource_row()
    row[field] = bad_value
    row["status"] = G.classify_resource_row(row)
    assert G.resource_row_passes(row) is False


def test_resource_cli_exit_binds_final_gpu_vacuity_status():
    row = _healthy_resource_row()
    row["status"] = "VACUOUS"
    row["resource_vacuity_reason"] = "planted unresolved GPU peak"
    assert G.classify_resource_row(row) == "CONFIRMED"
    assert G.resource_row_passes(row) is False


def test_healthy_resource_row_passes():
    assert G.resource_row_passes(_healthy_resource_row()) is True


def test_resource_row_refuses_finite_but_wrong_scheduled_gradient():
    row = _healthy_resource_row()
    agreement = G.classify_agreement(2.0, 1.0, 1.0)
    G.bind_resource_classification(row, agreement)
    assert row["resource_status"] == "CONFIRMED"
    assert row["gradient_agreement_status"] == "REFUTED"
    assert row["status"] == "REFUTED"
    assert G.resource_row_passes(row) is False


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("none", 700.0),
        ("uniform", 110.0),
        ("binomial", 100.0 + 10.0 * 19.0 / 3.0),
    ],
)
def test_registered_360_step_memory_extrapolation(schedule, expected):
    assert G.extrapolate_360(schedule, 100.0, 110.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("none", 200.0),
        ("uniform", 110.0),
        ("binomial", 100.0 + 10.0 * 8.0 / 3.0),
    ],
)
def test_registered_60_step_memory_extrapolation(schedule, expected):
    assert G.extrapolate_steps(schedule, 100.0, 110.0, 60) == pytest.approx(
        expected)


def test_schedule_primal_contract_records_bits_and_refuses_large_drift():
    base = {"x": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    last_bit = {"x": jnp.asarray(
        [np.nextafter(1.0, 2.0), 2.0], dtype=jnp.float64)}
    bad = {"x": jnp.asarray([1.0, 2.1], dtype=jnp.float64)}
    close = G.compare_scheduled_primal(base, last_bit)
    assert close["within_contract"] is True
    assert close["bitwise"] is False
    assert G.compare_scheduled_primal(base, bad)["within_contract"] is False


def test_volume_weighted_mse_ignores_zero_volume_cells():
    pred = jnp.asarray([1.0, 3.0, 99.0], dtype=jnp.float64)
    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    volume = jnp.asarray([1.0, 3.0, 0.0], dtype=jnp.float64)
    assert float(G.volume_weighted_mse(pred, target, volume)) == pytest.approx(3.25)


def test_state_bitwise_report_detects_last_bit_and_signed_zero():
    left = {
        "x": jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        "i": jnp.asarray([1], dtype=jnp.int32),
    }
    right = {
        "x": jnp.asarray(
            [np.nextafter(1.0, 2.0), -0.0], dtype=jnp.float64),
        "i": jnp.asarray([1], dtype=jnp.int32),
    }
    report = G.state_bitwise_report(left, right)
    assert report["bitwise_equal"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["path"] == "['x']"
    assert report["mismatches"][0]["max_abs"] == pytest.approx(
        np.nextafter(1.0, 2.0) - 1.0)
    assert G.states_bitwise_equal(left, right) is False
    assert G.states_bitwise_equal(left, left) is True


def test_nemo_literal_dissl_guard_preserves_primal_bits_and_zero_vjp():
    """The approved production guard changes only the zero-TKE transpose."""
    values = jnp.asarray(
        [4.0, 0.0, np.nextafter(0.0, 1.0), -1.0, np.nan],
        dtype=jnp.float64,
    )
    divisors = jnp.asarray([2.0, 3.0, 5.0, 7.0, 11.0], dtype=jnp.float64)

    def reverse_safe(value, divisor):
        positive = value > 0.0
        safe = jnp.where(positive, value, 1.0)
        safe_divisor = jnp.where(positive, divisor, 1.0)
        return jnp.where(positive, jnp.sqrt(safe) / safe_divisor, 0.0)

    old_primal = np.asarray(jnp.sqrt(values) / divisors)
    new_primal = np.asarray(reverse_safe(values, divisors))
    old_finite = np.isfinite(old_primal)
    np.testing.assert_array_equal(
        new_primal[old_finite].view(np.uint64),
        old_primal[old_finite].view(np.uint64),
    )

    _, pullback = jax.vjp(reverse_safe, values, divisors)
    tke_vjp, divisor_vjp = map(
        np.asarray, pullback(jnp.ones_like(values)))
    assert np.isfinite(tke_vjp[1])
    assert tke_vjp[1] == 0.0
    assert np.isfinite(divisor_vjp[1])
    assert divisor_vjp[1] == 0.0

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    _, zero_pullback = jax.vjp(reverse_safe, zero, zero)
    zero_tke_vjp, zero_divisor_vjp = zero_pullback(
        jnp.asarray(1.0, dtype=jnp.float64))
    assert float(zero_tke_vjp) == 0.0
    assert float(zero_divisor_vjp) == 0.0

    source = inspect.getsource(tke_mod.tke_vertical_mixing)
    assert "_dissl_sqrt_input = jnp.where(" in source
    assert "_dissl_divisor = jnp.where(" in source
    assert "jnp.sqrt(_dissl_sqrt_input)" in source


@pytest.mark.parametrize("mode", ["agreement", "resources"])
def test_cli_requires_a_schedule(mode):
    with pytest.raises(SystemExit):
        G.parse_args([
            mode, "--steps", "1", "--expect-commit", "deadbeef",
            "--output", "/tmp/unused-gyre-gradient-row.json",
        ])


def test_gpu_cli_can_select_native_transcendentals_explicitly():
    args = G.parse_args([
        "agreement", "--steps", "1", "--transcendentals", "native",
        "--schedule", "uniform",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.transcendentals == "native"


def test_per_step_composition_is_agreement_only():
    args = G.parse_args([
        "agreement", "--steps", "6", "--schedule", "per_step",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.schedule == "per_step"
    with pytest.raises(SystemExit):
        G.parse_args([
            "resources", "--steps", "6", "--schedule", "per_step",
            "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
        ])


def test_long_windows_are_registered_for_agreement_and_resources():
    args = G.parse_args([
        "agreement", "--steps", "360", "--schedule", "per_step",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.steps == 360
    resources = G.parse_args([
        "resources", "--steps", "60", "--schedule", "uniform",
        "--agreement-row", "/tmp/reference.json",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert resources.steps == 60


def test_peak_sampler_propagates_worker_failure(monkeypatch):
    sampler = G.PeakMemorySampler("cpu", interval_s=0.001)

    def fail():
        raise OSError("planted sampler read failure")

    sampler._read = fail
    sampler.start()
    sampler._thread.join(timeout=1.0)
    with pytest.raises(RuntimeError, match="peak-memory sampler failed"):
        sampler.stop()


def test_measurement_provenance_is_rechecked_after_row(monkeypatch):
    answers = iter(["changed-commit"])
    monkeypatch.setattr(G, "_git_output", lambda *_args: next(answers))
    with pytest.raises(RuntimeError, match="measurement commit"):
        G.require_unchanged_measurement_stamp(
            {"commit": "opening-commit", "dirty": False})


def test_cli_refuses_existing_output_before_measurement(tmp_path):
    output = tmp_path / "retained.json"
    output.write_text("retained\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite evidence"):
        G.main([
            "agreement", "--steps", "1", "--schedule", "uniform",
            "--expect-commit", "deadbeef", "--output", str(output),
        ])
    assert output.read_text(encoding="utf-8") == "retained\n"


def test_measurement_policy_labels_refuse_cross_backend_combinations():
    assert G.measurement_policy_label("cpu", "libm") == (
        "CPU, certified fp64/libm bit-exact policy")
    assert G.measurement_policy_label("gpu", "native") == (
        "GPU, XLA transcendentals — not the certified bit-exact policy")
    with pytest.raises(ValueError, match="unsupported GYRE gradient"):
        G.measurement_policy_label("gpu", "libm")
    with pytest.raises(ValueError, match="unsupported GYRE gradient"):
        G.measurement_policy_label("cpu", "native")


def test_historical_uncompiled_resource_timing_is_executable_retraction():
    old = _healthy_resource_row()
    old.pop("resource_schema")
    old.pop("reverse_transform")
    old.pop("warm_reverse_timer_span")
    assert G.classify_resource_row(old) == "RETRACTED"
    assert G.resource_row_passes(old) is False


def test_time_scaling_classifier_and_constant_time_plant():
    six = _healthy_resource_row()
    sixty = {**six, "steps": 60, "warm_reverse_wall_seconds": 8.0}
    control = G.classify_time_scaling([six, sixty])
    planted = G.classify_time_scaling([six, sixty], plant=True)
    assert control["status"] == "CONFIRMED"
    assert control["slope_seconds_per_step"] > 0.0
    assert planted["status"] == "REFUTED"
    assert planted["slope_seconds_per_step"] == pytest.approx(0.0, abs=1e-15)


def _gpu_resource_row(schedule, jax_increment, smi_increment):
    return {
        **_healthy_resource_row(),
        "backend": "gpu",
        "schedule": schedule,
        "steps": 360,
        "jax_device_memory_increment_mib": jax_increment,
        "nvidia_smi_memory_increment_mib": smi_increment,
    }


def test_gpu_resource_ordering_plant_flips():
    rows = [
        _gpu_resource_row("none", 900.0, 1000.0),
        _gpu_resource_row("uniform", 200.0, 250.0),
        _gpu_resource_row("binomial", 400.0, 500.0),
    ]
    assert G.classify_gpu_resource_suite(rows)["status"] == "CONFIRMED"
    assert G.classify_gpu_resource_suite(rows, plant=True)["status"] == "REFUTED"


def test_gpu_memory_scaling_classifier_and_plant():
    rows = []
    for schedule, scale in (("none", 10.0), ("uniform", 1.0),
                            ("binomial", 3.0)):
        for steps, factor in ((6, 1.0), (60, 2.0), (360, 4.0)):
            row = _gpu_resource_row(
                schedule, scale * factor, scale * factor * 1.1)
            row["steps"] = steps
            rows.append(row)
    assert G.classify_memory_scaling(rows)["status"] == "CONFIRMED"
    assert G.classify_memory_scaling(rows, plant=True)["status"] == "REFUTED"


def test_device_memory_stats_requires_allocator_peak_and_live_bytes():
    class Device:
        def memory_stats(self):
            return {
                "bytes_in_use": 2**20,
                "peak_bytes_in_use": 3 * 2**20,
                "num_allocs": np.int64(7),
            }

    assert G.device_memory_stats(Device())["num_allocs"] == 7

    class MissingPeak:
        def memory_stats(self):
            return {"bytes_in_use": 2**20}

    with pytest.raises(RuntimeError, match="peak_bytes_in_use"):
        G.device_memory_stats(MissingPeak())


def test_gpu_allocator_contract_refuses_platform(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    G.require_gpu_allocator_instrument()
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    with pytest.raises(RuntimeError, match="refuse.*platform"):
        G.require_gpu_allocator_instrument()


def test_resource_two_pass_control_uses_fresh_sampler(monkeypatch):
    instances = []

    class SingleUseSampler:
        def __init__(self, _backend):
            self.started = 0
            self.stopped = 0
            self.baseline_mib = 1.0
            self.peak_mib = 2.0
            self.samples = 2
            instances.append(self)

        def start(self):
            assert self.started == 0
            self.started += 1

        def stop(self):
            assert self.started == 1 and self.stopped == 0
            self.stopped += 1

    monkeypatch.setattr(G, "PeakMemorySampler", SingleUseSampler)
    reverse = jax.jit(jax.value_and_grad(lambda x: x**2))
    cold, warm, _ = G.measure_cold_and_warm_reverse(
        reverse, jnp.asarray(3.0), "cpu")
    assert cold.sampler is instances[0]
    assert warm.sampler is instances[1]
    assert all(item.started == item.stopped == 1 for item in instances)


def test_compiled_reverse_warm_call_does_not_retrace():
    traces = 0

    def loss(x):
        nonlocal traces
        traces += 1
        return x**2

    reverse = G.build_compiled_reverse(loss)
    reverse(jnp.asarray(2.0))
    reverse(jnp.asarray(3.0))
    assert traces == 1
