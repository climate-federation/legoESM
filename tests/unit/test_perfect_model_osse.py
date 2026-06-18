"""Perfect-model (identical-twin) OSSE harness for the correction loop.

Demonstrates clause 5 ("updating these parameters IMPROVE the biases") in a
CONTROLLED twin where the truth is known: when the LES diagnosis is accurate the
loop both LOWERS the bias against the pseudo-truth AND recovers the known
parameter; when the diagnosis is useless the monotonic gate keeps the bias flat
(never worse) and the verdict reports the no-recovery honestly.  The real-ERA5
outcome is the separate, HPC-only empirical question — here the physics is
synthetic and fully controlled so the harness logic is non-vacuously testable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
from legoesm.training.correction_loop import CompareResult
from legoesm.training.perfect_model_osse import (
    OSSEResult,
    osse_verdict,
    run_perfect_model_osse,
)

NLAT, NLON = 2, 2
TRUE_CK = 0.9
BIASED_CK = float(CLUBBLiteConfig().C_K)  # 0.4 — the biased start


class _Eddy(NamedTuple):
    K: jax.Array
    valid: jax.Array


class _Env(NamedTuple):
    cape_j_kg: float


class _Rec(NamedTuple):
    flat_index: int
    lat_deg: float
    environment: _Env


def _run_fn(config):
    """Stand-in 'model time mean': the per-column C_K broadcast to the grid."""
    ck = jnp.asarray(config.C_K)
    if ck.ndim == 0:
        return jnp.full((NLAT, NLON), float(ck))
    return ck.reshape(NLAT, NLON)


def _build_compare_fn(reference):
    """compare_fn(config): bias = |run_fn(config) - pseudo-truth|, worst-2 manifest."""
    area_w = jnp.ones((NLAT, NLON))

    def compare_fn(config):
        bias = jnp.abs(_run_fn(config) - reference)
        flat = np.asarray(bias).reshape(-1)
        worst = [i for i in np.argsort(-flat)[:2] if flat[i] > 1e-9]
        manifest = [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                    for i in worst]
        return CompareResult(combined_score=bias, manifest=manifest,
                             area_weights=area_w, model_ctx=None)

    return compare_fn


def _diagnose_true(record, ctx):
    return _Eddy(K=jnp.array([TRUE_CK]), valid=jnp.array([True]))


def _diagnose_useless(record, ctx):
    # Returns the biased start → corrected columns don't change the bias.
    return _Eddy(K=jnp.array([BIASED_CK]), valid=jnp.array([True]))


def _run(diagnose_fn, n_iterations=3):
    return run_perfect_model_osse(
        true_config=CLUBBLiteConfig(C_K=TRUE_CK),
        biased_config=CLUBBLiteConfig(),
        coefficient_field="C_K",
        run_fn=_run_fn, build_compare_fn=_build_compare_fn,
        diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(NLAT, NLON),
        n_iterations=n_iterations)


# --------------------------------------------------------------------------- #
# Accurate diagnosis → the loop recovers the parameter and lowers the bias.
# --------------------------------------------------------------------------- #
def test_accurate_diagnosis_recovers_and_improves():
    res = _run(_diagnose_true)
    assert res.true_value == TRUE_CK
    assert res.initial_value == BIASED_CK
    assert res.bias_reduced
    assert res.param_error_reduced
    assert res.final_bias < res.initial_bias
    # All 4 columns corrected (2 per round) → field ≈ the true value.
    np.testing.assert_allclose(res.recovered_value, TRUE_CK, atol=1e-6)
    assert res.final_param_error < 1e-6
    verdict = osse_verdict(res)
    assert verdict.status == "recovered" and verdict.ok


def test_reference_is_generated_from_true_config():
    seen = []

    def spy_run_fn(config):
        seen.append(float(jnp.mean(jnp.asarray(config.C_K))))
        return _run_fn(config)

    run_perfect_model_osse(
        true_config=CLUBBLiteConfig(C_K=TRUE_CK),
        biased_config=CLUBBLiteConfig(),
        coefficient_field="C_K",
        run_fn=spy_run_fn, build_compare_fn=_build_compare_fn,
        diagnose_fn=_diagnose_true,
        promotion_key="clubb_lite_C_K", grid_shape=(NLAT, NLON), n_iterations=1)
    # The FIRST run_fn call builds the pseudo-truth from true_config (C_K=0.9).
    assert seen[0] == TRUE_CK


# --------------------------------------------------------------------------- #
# Useless diagnosis → the gate keeps the bias flat (never worse); honest verdict.
# --------------------------------------------------------------------------- #
def test_useless_diagnosis_does_not_worsen_and_reports_no_change():
    res = _run(_diagnose_useless)
    assert res.final_bias <= res.initial_bias + 1e-12   # monotonic gate
    assert not res.bias_reduced
    assert res.n_accepted == 0
    assert osse_verdict(res).status == "no_change"
    assert not osse_verdict(res).ok


# --------------------------------------------------------------------------- #
# Verdict classifier branches (hand-built results; summary unused by the verdict).
# --------------------------------------------------------------------------- #
def _result(**over):
    base = dict(
        initial_bias=1.0, final_bias=0.5, bias_reduction=0.5, bias_reduced=True,
        true_value=0.9, initial_value=0.4, recovered_value=0.9,
        initial_param_error=0.5, final_param_error=0.1, param_error_reduced=True,
        n_rounds=2, n_accepted=2, summary=None)
    base.update(over)
    return OSSEResult(**base)


def test_verdict_bias_only_when_param_not_recovered():
    r = _result(param_error_reduced=False, final_param_error=0.5)
    assert osse_verdict(r).status == "bias_only"


def test_verdict_worsened_is_flagged_loudly():
    r = _result(initial_bias=0.5, final_bias=1.0, bias_reduced=False,
                bias_reduction=-0.5)
    v = osse_verdict(r)
    assert v.status == "worsened" and not v.ok


def test_verdict_recovered_is_ok():
    assert osse_verdict(_result()).ok


# --------------------------------------------------------------------------- #
# Input guards (non-vacuousness + self-consistency).
# --------------------------------------------------------------------------- #
def test_zero_iterations_rejected():
    import pytest
    with pytest.raises(ValueError, match="n_iterations >= 1"):
        run_perfect_model_osse(
            true_config=CLUBBLiteConfig(C_K=TRUE_CK),
            biased_config=CLUBBLiteConfig(), coefficient_field="C_K",
            run_fn=_run_fn, build_compare_fn=_build_compare_fn,
            diagnose_fn=_diagnose_true, promotion_key="clubb_lite_C_K",
            grid_shape=(NLAT, NLON), n_iterations=0)


def test_coefficient_field_must_match_promotion_key():
    import pytest
    # promotion_key clubb_lite_C_K corrects C_K, NOT Pr_t — a mismatch is rejected
    # so the recovery metric can never describe a different field than the loop moves.
    with pytest.raises(ValueError, match="does not match"):
        run_perfect_model_osse(
            true_config=CLUBBLiteConfig(C_K=TRUE_CK),
            biased_config=CLUBBLiteConfig(), coefficient_field="Pr_t",
            run_fn=_run_fn, build_compare_fn=_build_compare_fn,
            diagnose_fn=_diagnose_true, promotion_key="clubb_lite_C_K",
            grid_shape=(NLAT, NLON), n_iterations=1)
