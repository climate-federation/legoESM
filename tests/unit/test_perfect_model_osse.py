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
from legoesm.training.correction_loop import CompareResult, CorrectionSpec
from legoesm.training.perfect_model_osse import (
    CoefRecovery,
    MultiOSSEResult,
    OSSEResult,
    multi_osse_verdict,
    osse_verdict,
    run_multi_perfect_model_osse,
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


# --------------------------------------------------------------------------- #
# SIMULTANEOUS multi-coefficient OSSE: recover C_K + Pr_t + C_eps at once.
# --------------------------------------------------------------------------- #
_TRUE = {"C_K": 0.9, "Pr_t": 0.5, "C_eps": 0.3}
_BIAS = {"C_K": 0.4, "Pr_t": 0.33, "C_eps": 0.1}
_SPECS = [
    CorrectionSpec(promotion_key="clubb_lite_C_K",
                   diagnosis_method="clubb_coefficient", background=_BIAS["C_K"]),
    CorrectionSpec(promotion_key="clubb_lite_Pr_t",
                   diagnosis_method="prandtl_number", background=_BIAS["Pr_t"]),
    CorrectionSpec(promotion_key="clubb_lite_C_eps",
                   diagnosis_method="c_eps", background=_BIAS["C_eps"]),
]


class _MultiDiag(NamedTuple):  # noqa: N815 - mirrors the real CLUBB symbol names
    C_K: jax.Array
    Pr_t: jax.Array
    C_eps: jax.Array
    valid: jax.Array


def _grid_of(x):
    x = jnp.asarray(x)
    return jnp.full((NLAT, NLON), float(x)) if x.ndim == 0 else x.reshape(NLAT, NLON)


def _multi_run_fn(config):
    return jnp.stack([_grid_of(config.C_K), _grid_of(config.Pr_t),
                      _grid_of(config.C_eps)])  # (3, NLAT, NLON)


def _multi_build_compare(reference):
    area_w = jnp.ones((NLAT, NLON))

    def compare_fn(config):
        bias = jnp.mean(jnp.abs(_multi_run_fn(config) - reference), axis=0)
        flat = np.asarray(bias).reshape(-1)
        worst = [i for i in np.argsort(-flat)[:2] if flat[i] > 1e-9]
        manifest = [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                    for i in worst]
        return CompareResult(combined_score=bias, manifest=manifest,
                             area_weights=area_w, model_ctx=None)

    return compare_fn


def _multi_diagnose(values):
    d = _MultiDiag(C_K=jnp.array([values["C_K"]]), Pr_t=jnp.array([values["Pr_t"]]),
                   C_eps=jnp.array([values["C_eps"]]), valid=jnp.array([True]))

    def diagnose(record, ctx):
        return {"clubb_coefficient": d, "prandtl_number": d, "c_eps": d}

    return diagnose


def _run_multi(diag_values, **kw):
    return run_multi_perfect_model_osse(
        true_config=CLUBBLiteConfig(**_TRUE), biased_config=CLUBBLiteConfig(**_BIAS),
        specs=_SPECS, run_fn=_multi_run_fn, build_compare_fn=_multi_build_compare,
        diagnose_fn=_multi_diagnose(diag_values), grid_shape=(NLAT, NLON), **kw)


def test_multi_accurate_diagnosis_recovers_all_three():
    res = _run_multi(_TRUE, n_iterations=3)
    assert isinstance(res, MultiOSSEResult)
    assert res.bias_reduced and res.all_recovered
    assert set(res.per_coefficient) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    for key, field in [("clubb_lite_C_K", "C_K"), ("clubb_lite_Pr_t", "Pr_t"),
                       ("clubb_lite_C_eps", "C_eps")]:
        c = res.per_coefficient[key]
        np.testing.assert_allclose(c.recovered_value, _TRUE[field], atol=1e-6)
        assert c.param_error_reduced
    v = multi_osse_verdict(res)
    assert v.status == "recovered" and v.ok


def test_multi_sequential_mode_recovers():
    # Block-coordinate (staged) mode also recovers the coupled coefficients.
    res = _run_multi(_TRUE, n_iterations=3, sequential=True)
    assert res.all_recovered
    assert multi_osse_verdict(res).status == "recovered"


def test_multi_useless_diagnosis_no_change():
    res = _run_multi(_BIAS, n_iterations=2)  # diagnoses the biased start → no help
    assert res.final_bias <= res.initial_bias + 1e-12
    assert not res.all_recovered
    assert multi_osse_verdict(res).status == "no_change"


def test_multi_clip_to_bounds_clamps_unphysical_diagnosis():
    # An LES diagnosis past the registered bounds must be CLIPPED (production parity),
    # so the OSSE cannot declare recovery using a value the real campaign would clip.
    over = {"C_K": 5.0, "Pr_t": 0.5, "C_eps": 0.3}  # C_K=5.0 >> the (0.1,1.2) bound
    res = _run_multi(over, n_iterations=2, clip_to_bounds=True)
    ck = res.per_coefficient["clubb_lite_C_K"]
    assert ck.recovered_value <= 1.2 + 1e-9   # clamped into bounds, not 5.0


def test_multi_rejects_duplicate_promotion_key():
    import pytest
    dup = [_SPECS[0], _SPECS[0]]
    with pytest.raises(ValueError, match="duplicate promotion_key"):
        run_multi_perfect_model_osse(
            true_config=CLUBBLiteConfig(**_TRUE),
            biased_config=CLUBBLiteConfig(**_BIAS), specs=dup,
            run_fn=_multi_run_fn, build_compare_fn=_multi_build_compare,
            diagnose_fn=_multi_diagnose(_TRUE), grid_shape=(NLAT, NLON))


def test_multi_rejects_empty_specs_and_bad_key():
    import pytest
    with pytest.raises(ValueError, match="at least one CorrectionSpec"):
        run_multi_perfect_model_osse(
            true_config=CLUBBLiteConfig(**_TRUE),
            biased_config=CLUBBLiteConfig(**_BIAS), specs=[],
            run_fn=_multi_run_fn, build_compare_fn=_multi_build_compare,
            diagnose_fn=_multi_diagnose(_TRUE), grid_shape=(NLAT, NLON))
    bad = [CorrectionSpec(promotion_key="not_a_field",
                          diagnosis_method="clubb_coefficient", background=0.4)]
    with pytest.raises(ValueError, match="unknown promotion_key"):
        run_multi_perfect_model_osse(
            true_config=CLUBBLiteConfig(**_TRUE),
            biased_config=CLUBBLiteConfig(**_BIAS), specs=bad,
            run_fn=_multi_run_fn, build_compare_fn=_multi_build_compare,
            diagnose_fn=_multi_diagnose(_TRUE), grid_shape=(NLAT, NLON))


def _coef(reduced):
    return CoefRecovery(
        field="C_K", true_value=0.9, initial_value=0.4, recovered_value=0.7,
        initial_param_error=0.5, final_param_error=0.2 if reduced else 0.5,
        param_error_reduced=reduced)


def test_multi_verdict_bias_only_when_not_all_recover():
    res = MultiOSSEResult(
        initial_bias=1.0, final_bias=0.5, bias_reduction=0.5, bias_reduced=True,
        per_coefficient={"a": _coef(True), "b": _coef(False)}, all_recovered=False,
        n_rounds=2, n_accepted=2, summary=None)
    v = multi_osse_verdict(res)
    assert v.status == "bias_only" and "1/2" in v.message
