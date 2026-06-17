"""Unit tests for :mod:`legoesm.training.correction_loop`.

The capstone: one diagnose→correct→verify iteration of the LES-informed
correction loop, with the heavy AMIP/LES steps mocked.  Demonstrates the loop
closes — a correction that lowers the worst-column scores yields
``bias.improved == True`` and a positive worst-column change — and that a
worsening correction is correctly reported as NOT improved.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.training.correction_loop import (
    CompareResult,
    run_correction_iteration,
)


class _Eddy(NamedTuple):
    K: jax.Array
    valid: jax.Array


class _Env(NamedTuple):
    cape_J_kg: float


class _Rec(NamedTuple):
    flat_index: int
    lat_deg: float
    environment: _Env


_WORST = [
    _Rec(flat_index=0, lat_deg=10.0, environment=_Env(200.0)),
    _Rec(flat_index=3, lat_deg=-20.0, environment=_Env(2500.0)),
]
_AREA_W = jnp.ones((2, 2))


def _is_corrected(config) -> bool:
    """The config is 'corrected' once tau_equator is a per-column array."""
    return jnp.ndim(jnp.asarray(config.tau_equator)) > 0


def _diagnose(record, model_ctx):
    # A valid eddy-diffusivity diagnosis (value doesn't affect the mock physics).
    return _Eddy(K=jnp.array([10.0, 20.0]), valid=jnp.array([True, True]))


def _make_compare_fn(*, baseline_score, corrected_score):
    def compare_fn(config) -> CompareResult:
        score = corrected_score if _is_corrected(config) else baseline_score
        return CompareResult(
            combined_score=jnp.asarray(score),
            manifest=_WORST,
            area_weights=_AREA_W,
            model_ctx=None,
        )

    return compare_fn


def test_loop_detects_improvement():
    # baseline: worst columns (flat 0, 3) have high score; corrected: lowered.
    baseline = [[2.0, 1.0], [1.0, 2.0]]
    corrected = [[1.0, 1.0], [1.0, 1.0]]
    compare_fn = _make_compare_fn(baseline_score=baseline, corrected_score=corrected)

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2),
        background=7.2,
    )
    assert bool(result.bias.improved)
    assert float(result.bias.baseline_bias) == pytest.approx(1.5)
    assert float(result.bias.updated_bias) == pytest.approx(1.0)
    assert float(result.worst_column_change) == pytest.approx(1.0)  # mean(1,1)
    assert result.n_corrected == 2
    # The updated config carries a per-column tau_equator (the feedback reached it).
    assert jnp.ndim(jnp.asarray(result.updated_config.tau_equator)) == 1


def test_loop_reports_worsening():
    baseline = [[1.0, 1.0], [1.0, 1.0]]
    corrected = [[2.0, 1.0], [1.0, 2.0]]  # got worse
    compare_fn = _make_compare_fn(baseline_score=baseline, corrected_score=corrected)
    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
    )
    assert not bool(result.bias.improved)
    assert float(result.worst_column_change) < 0.0


def test_loop_feedback_field_scattered_at_worst_columns():
    compare_fn = _make_compare_fn(
        baseline_score=[[2.0, 1.0], [1.0, 2.0]],
        corrected_score=[[1.0, 1.0], [1.0, 1.0]],
    )
    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
    )
    f = np.asarray(result.feedback_field).reshape(-1)
    # worst columns 0 and 3 carry the diagnosed value (mean(10,20)=15);
    # the others keep the background 7.2.
    assert f[0] == pytest.approx(15.0)
    assert f[3] == pytest.approx(15.0)
    assert f[1] == pytest.approx(7.2)
    assert f[2] == pytest.approx(7.2)


def test_loop_empty_manifest_is_noop():
    """No worst columns → one compare call, no diagnosis, baseline config kept."""
    compare_calls = {"n": 0}
    diagnose_calls = {"n": 0}

    def compare_fn(config):
        compare_calls["n"] += 1
        return CompareResult(
            combined_score=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            manifest=[],  # nothing flagged
            area_weights=_AREA_W, model_ctx=None,
        )

    def diagnose(record, ctx):
        diagnose_calls["n"] += 1
        return _diagnose(record, ctx)

    cfg = GrayRadiationConfig()
    result = run_correction_iteration(
        cfg, compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
    )
    assert compare_calls["n"] == 1   # NO second AMIP run
    assert diagnose_calls["n"] == 0
    assert result.n_corrected == 0
    assert not bool(result.bias.improved)  # baseline vs baseline
    assert float(result.worst_column_change) == 0.0
    # config unchanged (scalar tau_equator).
    assert result.updated_config is cfg
    assert jnp.ndim(jnp.asarray(result.updated_config.tau_equator)) == 0
    # feedback field is uniform background.
    np.testing.assert_allclose(np.asarray(result.feedback_field), 7.2)


def test_loop_multi_iteration_threads_updated_config():
    """A two-step campaign: the 2nd iteration starts from the 1st's updated config."""
    compare_fn = _make_compare_fn(
        baseline_score=[[2.0, 1.0], [1.0, 2.0]],
        corrected_score=[[1.0, 1.0], [1.0, 1.0]],
    )
    cfg0 = GrayRadiationConfig()
    r1 = run_correction_iteration(
        cfg0, compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
    )
    # Feed the updated config back in (it is already 'corrected').
    r2 = run_correction_iteration(
        r1.updated_config, compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
    )
    # Starting from a corrected config, both compare calls see the corrected
    # scores ⇒ no further improvement (already at the floor), but no crash.
    assert jnp.ndim(jnp.asarray(r2.updated_config.tau_equator)) == 1
    assert r2.n_corrected == 2


def test_loop_unknown_promotion_key_raises():
    compare_fn = _make_compare_fn(
        baseline_score=[[1.0, 1.0], [1.0, 1.0]],
        corrected_score=[[1.0, 1.0], [1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="Unknown promotable coefficient"):
        run_correction_iteration(
            GrayRadiationConfig(),
            compare_fn=compare_fn, diagnose_fn=_diagnose,
            promotion_key="bogus", grid_shape=(2, 2), background=7.2,
        )


def test_loop_expected_ncol_guard():
    compare_fn = _make_compare_fn(
        baseline_score=[[1.0, 1.0], [1.0, 1.0]],
        corrected_score=[[1.0, 1.0], [1.0, 1.0]],
    )
    with pytest.raises(ValueError, match="expected_ncol"):
        run_correction_iteration(
            GrayRadiationConfig(),
            compare_fn=compare_fn, diagnose_fn=_diagnose,
            promotion_key="gray_tau_equator", grid_shape=(2, 2),
            background=7.2, expected_ncol=99,  # != 4
        )
