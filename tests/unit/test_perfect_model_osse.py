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
from legoesm.training.column_manifest import ColumnEnvironment, ColumnRecord
from legoesm.training.correction_loop import CompareResult, CorrectionSpec
from legoesm.training.perfect_model_osse import (
    CoefRecovery,
    CrossResOSSEResult,
    MultiOSSEResult,
    OSSEResult,
    cross_res_osse_verdict,
    multi_osse_verdict,
    osse_verdict,
    run_cross_resolution_osse,
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
    import pytest
    res = _run(_diagnose_true)
    # fp32-safe (unit tier): the config C_K round-trips through float32, so
    # compare with tolerance not exact equality (the prior == passed only via
    # the session-wide x64 leak). rel=1e-6 still catches a wrong true value.
    assert res.true_value == pytest.approx(TRUE_CK, rel=1e-6)
    assert res.initial_value == pytest.approx(BIASED_CK, rel=1e-6)
    assert res.bias_reduced
    assert res.param_error_reduced
    assert res.final_bias < res.initial_bias
    # All 4 columns corrected (2 per round) → field ≈ the true value.
    np.testing.assert_allclose(res.recovered_value, TRUE_CK, atol=1e-6)
    assert res.final_param_error < 1e-6
    verdict = osse_verdict(res)
    assert verdict.status == "recovered" and verdict.ok


def test_reference_is_generated_from_true_config():
    import pytest
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
    assert seen[0] == pytest.approx(TRUE_CK, rel=1e-6)  # fp32-safe (see above)


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


def test_verdict_diverged_on_non_finite_bias():
    """A non-finite OSSE bias (a twin model blow-up from an unstable coefficient) is
    flagged 'diverged' (not ok), NOT silently mislabelled no_change/bias_only — the
    go/no-go must not green-light a diverged twin. (NaN > x and NaN < x are both False,
    so without the guard it would fall through.)"""
    v = osse_verdict(_result(final_bias=float("nan")))
    assert v.status == "diverged" and not v.ok
    assert "DIVERGED" in v.message
    assert osse_verdict(_result(initial_bias=float("inf"))).status == "diverged"


def test_multi_verdict_diverged_on_non_finite_bias():
    """multi_osse_verdict flags a non-finite multi-coefficient bias 'diverged' too
    (same shared guard as the single verdict)."""
    res = _run_multi(_TRUE, n_iterations=1)._replace(final_bias=float("nan"))
    assert multi_osse_verdict(res).status == "diverged"


def test_cross_res_verdict_diverged_on_non_finite_bias():
    """cross_res_osse_verdict flags a non-finite fine-grid bias 'diverged' BEFORE the
    coverage/out_of_hull check — a twin blow-up is a blow-up regardless of coverage."""
    res = _cross_res(_FINE_ENV, (2, 4))._replace(fine_bias_corrected=float("nan"))
    assert cross_res_osse_verdict(res).status == "diverged"


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
    import pytest
    over = {"C_K": 5.0, "Pr_t": 0.5, "C_eps": 0.3}  # C_K=5.0 >> the (0.1,1.2) bound
    res = _run_multi(over, n_iterations=2, clip_to_bounds=True)
    ck = res.per_coefficient["clubb_lite_C_K"]
    # The clipped diagnosis (5.0 → upper bound 1.2) was APPLIED: recovered lands AT
    # 1.2 — a two-sided check (Codex). It fails if the clamp broke (→ raw 5.0), OR if
    # the clamped step was rejected and stayed at the biased 0.4 (the one-sided <=1.2
    # would have passed in that regression). rel=1e-5 is fp32-safe (float32(1.2)).
    assert ck.recovered_value == pytest.approx(1.2, rel=1e-5)   # clamped, applied
    assert float(ck.initial_value) == pytest.approx(BIASED_CK, rel=1e-6)  # moved up from 0.4


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


# --------------------------------------------------------------------------- #
# Cross-resolution OSSE: learn the env-dependent correction on a COARSE grid,
# deploy via the env-kernel on a FINE grid (different ncol), measure the bias.
# --------------------------------------------------------------------------- #
class _CKDiag(NamedTuple):
    C_K: jax.Array
    valid: jax.Array


def _c_true(env):
    """Env-OPTIMAL C_K rises with CAPE (predictor 1): 0.4..0.7 over [0, 3000]."""
    cape = np.asarray(env)[:, 1]
    return 0.4 + 1.0e-4 * cape


def _crec(flat, score, sst, cape, shear):
    return ColumnRecord(
        flat_index=int(flat), grid_index=(int(flat),), lat_deg=0.0, lon_deg=0.0,
        time_index=0, combined_score=float(score), T_rmse_K=0.0, qv_rmse_kg_kg=0.0,
        wind_rmse_m_s=0.0, precip_err_mm_day=0.0,
        environment=ColumnEnvironment(sst_K=float(sst), cape_J_kg=float(cape),
                                      bulk_shear_m_s=float(shear)))


def _obs(config, grid_shape):
    """'Model time mean' = the per-column C_K broadcast to the grid (flat)."""
    ck = jnp.asarray(config.C_K)
    n = int(np.prod(grid_shape))
    flat = jnp.full((n,), float(ck)) if ck.ndim == 0 else ck.reshape(-1)
    return np.asarray(flat)


def _make_run_fn(grid_shape):
    def run_fn(config):
        return jnp.asarray(_obs(config, grid_shape)).reshape(grid_shape)
    return run_fn


def _make_build_compare(grid_shape, env):
    # The pseudo-truth observable is the env-OPTIMAL C_K (env-dependent); bias =
    # |C_K - c_true(env)|.  ALL columns with non-zero bias are flagged worst so the
    # kernel samples span the env range (the reference arg is the constant run, not
    # used — the analytic env truth is the pseudo-reanalysis).
    truth = _c_true(env)
    area_w = jnp.ones(grid_shape)

    def build(_reference):
        def compare(config):
            bias = np.abs(_obs(config, grid_shape) - truth)
            manifest = [_crec(i, bias[i], *np.asarray(env)[i])
                        for i in range(bias.shape[0]) if bias[i] > 1e-9]
            return CompareResult(
                combined_score=jnp.asarray(bias.reshape(grid_shape)),
                manifest=manifest, area_weights=area_w, model_ctx=None)
        return compare
    return build


def _make_env_grid_fn(env):
    grid_env = jnp.asarray(env, dtype=float)
    length_scales = jnp.std(grid_env, axis=0)

    def env_grid_fn(_model_ctx):
        return grid_env, length_scales
    return env_grid_fn


def _diagnose_c_true(record, ctx):
    env = np.asarray([[record.environment.sst_K, record.environment.cape_J_kg,
                       record.environment.bulk_shear_m_s]])
    return _CKDiag(C_K=jnp.array([float(_c_true(env)[0])]), valid=jnp.array([True]))


# Coarse: 6 columns spanning CAPE [200, 3000]; SST/shear co-vary.
_COARSE_ENV = np.array([
    [296.0, 200.0, 4.0], [298.0, 800.0, 7.0], [300.0, 1400.0, 11.0],
    [301.0, 2000.0, 15.0], [302.0, 2600.0, 19.0], [303.0, 3000.0, 22.0],
])
# Fine: 8 columns, CAPE WITHIN the coarse hull (covered) — different ncol + envs.
_FINE_ENV = np.array([
    [297.0, 500.0, 5.0], [299.0, 1000.0, 9.0], [300.5, 1700.0, 13.0],
    [301.5, 2300.0, 17.0], [296.5, 350.0, 4.5], [298.5, 900.0, 8.0],
    [302.5, 2800.0, 20.0], [300.0, 1500.0, 12.0],
])
# Far-climate fine grid (CAPE ~10000, way above the coarse hull) — out of hull.
_FAR_ENV = np.array([[310.0, 9000.0, 60.0], [312.0, 11000.0, 70.0]] * 2)


def _cross_res(fine_env, fine_shape, coverage_threshold=0.8):
    return run_cross_resolution_osse(
        true_config=CLUBBLiteConfig(C_K=0.55),     # value irrelevant (env truth)
        biased_config=CLUBBLiteConfig(),           # C_K = 0.4 constant
        coefficient_field="C_K", promotion_key="clubb_lite_C_K",
        diagnosis_method="clubb_coefficient",
        coarse_grid_shape=(2, 3),
        coarse_run_fn=_make_run_fn((2, 3)),
        coarse_build_compare_fn=_make_build_compare((2, 3), _COARSE_ENV),
        coarse_diagnose_fn=_diagnose_c_true,
        coarse_env_grid_fn=_make_env_grid_fn(_COARSE_ENV),
        fine_grid_shape=fine_shape,
        fine_run_fn=_make_run_fn(fine_shape),
        fine_build_compare_fn=_make_build_compare(fine_shape, fine_env),
        fine_env_grid_fn=_make_env_grid_fn(fine_env),
        coverage_threshold=coverage_threshold, n_iterations=2)


def test_cross_resolution_kernel_transfers_and_lowers_fine_bias():
    """The coarse-learned env-kernel deploys on a DIFFERENT-ncol fine grid and
    LOWERS its bias — the cross-resolution deploy generalizes across resolution."""
    res = _cross_res(_FINE_ENV, (2, 4))
    assert isinstance(res, CrossResOSSEResult)
    assert res.n_fine_columns == 8                  # not the coarse 6
    assert res.kernel_field == "C_K"
    assert res.coarse_bias_reduced                  # the coarse leg learned
    assert res.fine_bias_corrected < res.fine_bias_uncorrected   # cross-res improved
    assert res.fine_bias_reduction > 0.0
    assert res.fraction_in_hull >= 0.8              # fine climate inside coarse hull
    v = cross_res_osse_verdict(res)
    assert v.status == "transferred" and v.ok


def test_cross_resolution_out_of_hull_is_untrusted():
    """A fine grid whose climate lies OUTSIDE the coarse-sampled env → low coverage
    → 'out_of_hull' verdict (the kernel extrapolates; not trustworthy)."""
    res = _cross_res(_FAR_ENV, (2, 2))
    assert res.fraction_in_hull < 0.8
    v = cross_res_osse_verdict(res)
    assert v.status == "out_of_hull" and not v.ok


def test_cross_res_out_of_hull_takes_precedence_over_a_bias_drop():
    """The coverage gate OUTRANKS a bias reduction (iter 286): a deploy that LOWERED the
    fine-grid bias but lies OUTSIDE the coarse env hull is 'out_of_hull' (untrustworthy),
    NOT 'transferred'.  A regression checking ``fine_bias_reduced`` before the coverage
    gate would falsely green-light an extrapolating deploy.  The existing _FAR_ENV test
    can't pin this — its out-of-hull bias typically does NOT fall (the kernel no-ops), so
    it can't distinguish the two orderings; this constructs bias-fell + out-of-hull."""
    base = _cross_res(_FINE_ENV, (2, 4))
    res = base._replace(                                       # bias FELL 5 -> 2 ...
        fine_bias_uncorrected=5.0, fine_bias_corrected=2.0, fine_bias_reduced=True,
        fraction_in_hull=0.1, fraction_covered=0.1)           # ... but only 10% in-hull
    assert res.fraction_in_hull < res.coverage_threshold      # genuinely out of hull
    v = cross_res_osse_verdict(res)
    assert v.status == "out_of_hull" and not v.ok             # NOT 'transferred'
    assert "untrustworthy" in v.message


def test_cross_resolution_rejects_mismatched_field():
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        run_cross_resolution_osse(
            true_config=CLUBBLiteConfig(), biased_config=CLUBBLiteConfig(),
            coefficient_field="Pr_t", promotion_key="clubb_lite_C_K",
            diagnosis_method="clubb_coefficient",
            coarse_grid_shape=(2, 3), coarse_run_fn=_make_run_fn((2, 3)),
            coarse_build_compare_fn=_make_build_compare((2, 3), _COARSE_ENV),
            coarse_diagnose_fn=_diagnose_c_true,
            coarse_env_grid_fn=_make_env_grid_fn(_COARSE_ENV),
            fine_grid_shape=(2, 4), fine_run_fn=_make_run_fn((2, 4)),
            fine_build_compare_fn=_make_build_compare((2, 4), _FINE_ENV),
            fine_env_grid_fn=_make_env_grid_fn(_FINE_ENV))


def test_last_accepted_env_kernel_skips_rejected_final_round():
    """The helper returns the last ACCEPTED round's kernel, not a later rejected
    round's (the rejected kernel is inconsistent with the accepted final state)."""
    from legoesm.training.correction_loop import (
        CampaignResult,
        CorrectionResult,
        last_accepted_env_kernel,
    )

    def _cres(kernel):
        return CorrectionResult(
            updated_config=None, bias=None, worst_column_change=jnp.asarray(0.0),
            feedback_field=jnp.zeros((2, 2)), n_corrected=1, env_kernel=kernel)

    r = CampaignResult(
        final_config=None,
        iterations=(_cres("K0"), _cres("K1"), _cres("K2")),
        final_field=jnp.zeros((2, 2)), accepted=(True, True, False))
    assert last_accepted_env_kernel(r) == "K1"      # K2's round was rejected
    # Empty accepted ⇒ all kept ⇒ the last kernel.
    r2 = r._replace(accepted=())
    assert last_accepted_env_kernel(r2) == "K2"
    # No kernel anywhere ⇒ None.
    r3 = r._replace(iterations=(_cres(None),), accepted=(True,))
    assert last_accepted_env_kernel(r3) is None
