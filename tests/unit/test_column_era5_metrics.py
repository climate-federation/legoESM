"""Unit tests for :mod:`legoesm.training.column_era5_metrics`.

Pins the per-column model-vs-ERA5 comparison metrics used to rank worst
columns for LES spin-off (``docs/COMPARE_REANALYSIS.md`` stage 2 / gap #1):
mass-weighted profile RMSE, vector-wind RMSE, NaN masking, the combined
dimensionless score, and worst-column ranking.  Also guards differentiability
(the field must flow through ``jax.grad``) and reuse of the shared
``scm_rce_metrics`` arithmetic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.training.column_era5_metrics import (
    ColumnErrorConfig,
    normalized_mass_weights,
    per_column_vector_wind_rmse,
    per_column_weighted_rmse,
    rank_worst_columns,
    score_columns,
)
from legoesm.training.scm_rce_metrics import weighted_rmse


def test_normalized_mass_weights_sum_to_one():
    dsigma = jnp.array([0.4, 0.3, 0.2, 0.1])
    w = normalized_mass_weights(dsigma)
    assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-12)
    # Proportional to dsigma.
    assert float(w[0] / w[3]) == pytest.approx(4.0, abs=1e-10)


def test_rmse_zero_when_model_equals_ref():
    nlev = 5
    w = normalized_mass_weights(jnp.ones((nlev,)))
    field = jnp.linspace(280.0, 220.0, nlev)[None, None, :] * jnp.ones((3, 4, 1))
    rmse = per_column_weighted_rmse(field, field, w)
    assert rmse.shape == (3, 4)
    assert float(jnp.max(jnp.abs(rmse))) == pytest.approx(0.0, abs=1e-10)


def test_constant_offset_recovers_offset():
    nlev = 6
    w = normalized_mass_weights(jnp.ones((nlev,)))
    ref = jnp.zeros((2, 2, nlev))
    model = ref + 2.5
    rmse = per_column_weighted_rmse(model, ref, w)
    # abs=1e-5 is fp32-safe (the constant-offset RMSE recovers 2.5 to ~2e-7 at
    # float32, the unit tier's default precision) yet still catches a real
    # weighting/RMSE bug; the prior 1e-10 passed ONLY via the session-wide x64 leak.
    assert float(jnp.max(jnp.abs(rmse - 2.5))) == pytest.approx(0.0, abs=1e-5)


def test_per_column_matches_scm_rce_weighted_rmse():
    """A single column must reproduce the shared ``weighted_rmse`` exactly."""
    nlev = 7
    dsigma = jnp.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1])
    w = normalized_mass_weights(dsigma)
    model = jnp.array([1.0, 2.0, 3.0, 2.5, 1.5, 0.5, 0.0])
    ref = jnp.array([0.5, 2.5, 2.0, 3.0, 1.0, 1.0, 0.5])
    direct = float(weighted_rmse(model - ref, w))
    via = float(per_column_weighted_rmse(model[None, :], ref[None, :], w)[0])
    assert via == pytest.approx(direct, abs=1e-12)


def test_nan_levels_are_masked_and_renormalized():
    """A NaN ERA5 level should be dropped, not poison the column."""
    nlev = 4
    w = normalized_mass_weights(jnp.ones((nlev,)))
    ref = jnp.array([[0.0, 0.0, 0.0, 0.0]])
    model = jnp.array([[1.0, 1.0, 1.0, 1.0]])
    full = float(per_column_weighted_rmse(model, ref, w)[0])
    assert full == pytest.approx(1.0, abs=1e-10)
    # Inserting a NaN at one level (both fields agree elsewhere) keeps RMSE=1.
    ref_nan = ref.at[0, 2].set(jnp.nan)
    masked = float(per_column_weighted_rmse(model, ref_nan, w)[0])
    assert masked == pytest.approx(1.0, abs=1e-10)


def test_vector_wind_rmse_combines_components():
    nlev = 3
    w = normalized_mass_weights(jnp.ones((nlev,)))
    u_m = jnp.zeros((1, nlev))
    v_m = jnp.zeros((1, nlev))
    u_r = jnp.full((1, nlev), 3.0)
    v_r = jnp.full((1, nlev), 4.0)
    rmse = float(per_column_vector_wind_rmse(u_m, v_m, u_r, v_r, w)[0])
    # sqrt(mean(3^2 + 4^2)) = 5 for a uniform error.
    assert rmse == pytest.approx(5.0, abs=1e-10)


def test_score_columns_zero_when_perfect():
    nlev = 5
    w = normalized_mass_weights(jnp.ones((nlev,)))
    shape = (2, 3, nlev)
    T = jnp.full(shape, 250.0)
    qv = jnp.full(shape, 1e-3)
    u = jnp.full(shape, 5.0)
    v = jnp.full(shape, -2.0)
    fields = score_columns(
        T_model=T, qv_model=qv, u_model=u, v_model=v,
        T_ref=T, qv_ref=qv, u_ref=u, v_ref=v,
        mass_weights=w,
        precip_model_mm_day=jnp.full((2, 3), 3.0),
        precip_ref_mm_day=jnp.full((2, 3), 3.0),
    )
    assert fields.combined_score.shape == (2, 3)
    assert float(jnp.max(fields.combined_score)) == pytest.approx(0.0, abs=1e-10)


def test_score_columns_precip_optional_drops_term():
    nlev = 4
    w = normalized_mass_weights(jnp.ones((nlev,)))
    shape = (2, 2, nlev)
    T = jnp.full(shape, 250.0)
    qv = jnp.full(shape, 1e-3)
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)
    fields = score_columns(
        T_model=T + 3.0, qv_model=qv, u_model=u, v_model=v,
        T_ref=T, qv_ref=qv, u_ref=u, v_ref=v,
        mass_weights=w,
    )
    # T error == one normalization scale; only T contributes (weights 1 each,
    # precip dropped): combined = sqrt((1*1^2)/(3)) since qv,wind zero.
    expected = (1.0 / 3.0) ** 0.5
    assert float(jnp.max(jnp.abs(fields.combined_score - expected))) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(jnp.max(jnp.abs(fields.precip_err_mm_day))) == 0.0


def test_combined_score_commensurable_across_variables():
    """The per-variable normalization makes T/qv/wind COMMENSURABLE: an absolute
    PHYSICAL error of one normalization scale in any single variable yields the SAME
    combined score.

    NON-VACUOUS against a mis-set *_norm: the errors below are HARD-CODED absolute
    magnitudes (not ``cfg.*_norm``, which would cancel the divisor), and the locked
    default scales are asserted first — so a drift in any default makes the assert
    (or the resulting term≠1) fail, which is exactly the regression that would make
    a variable invisible to worst-column ranking."""
    cfg = ColumnErrorConfig()
    # Default normalization scales this test is calibrated to (drift fails loudly).
    t_scale, qv_scale, wind_scale = 3.0, 1.5e-3, 5.0
    assert cfg.T_norm_K == pytest.approx(t_scale)
    assert cfg.qv_norm_kg_kg == pytest.approx(qv_scale)
    assert cfg.wind_norm_m_s == pytest.approx(wind_scale)
    w = normalized_mass_weights(jnp.ones((4,)))
    zero = jnp.zeros((3, 4))
    # absolute one-scale error per variable (hard-coded, so the ratio does NOT cancel)
    fields = score_columns(
        T_model=zero.at[0].set(t_scale),
        qv_model=zero.at[1].set(qv_scale),
        u_model=zero.at[2].set(wind_scale),
        v_model=zero, T_ref=zero, qv_ref=zero, u_ref=zero, v_ref=zero,
        mass_weights=w, config=cfg,
    )
    cs = fields.combined_score
    # each is exactly one normalized unit ⇒ combined = sqrt((1·1²)/3) for all three
    expected = (1.0 / 3.0) ** 0.5
    for c in range(3):
        assert float(cs[c]) == pytest.approx(expected, abs=1e-6)
    # and they are EQUAL to each other (the commensurability invariant)
    assert float(jnp.max(cs) - jnp.min(cs)) == pytest.approx(0.0, abs=1e-6)


def test_qv_and_wind_errors_drive_ranking():
    """A qv-only and a wind-only error column each outrank a column with a tiny
    T error — proving qv and wind actually PARTICIPATE in worst-column selection
    (the RANKING here catches a FORMULA regression that drops a variable from the
    combine — e.g. ``w_qv`` forced to 0 → column 1 scores ~0 → not in the worst-two).

    Constant-drift in a *_norm is caught by the explicit value-lock assert below
    (the ranking alone only flips for a large >~50× drift); the two guards are
    complementary. Errors are hard-coded absolute magnitudes (not cfg-relative,
    which would cancel the divisor)."""
    cfg = ColumnErrorConfig()
    assert (cfg.T_norm_K, cfg.qv_norm_kg_kg, cfg.wind_norm_m_s) == pytest.approx(
        (3.0, 1.5e-3, 5.0))
    w = normalized_mass_weights(jnp.ones((4,)))
    zero = jnp.zeros((3, 4))
    # col 0: tiny T error (0.1·3 K); col 1: large qv error (5·1.5e-3); col 2: large wind (5·5)
    fields = score_columns(
        T_model=zero.at[0].set(0.3),
        qv_model=zero.at[1].set(7.5e-3),
        u_model=zero.at[2].set(25.0),
        v_model=zero, T_ref=zero, qv_ref=zero, u_ref=zero, v_ref=zero,
        mass_weights=w, config=cfg,
    )
    idx, vals, valid = rank_worst_columns(fields.combined_score, 2)
    worst_two = set(int(i) for i in idx)
    assert worst_two == {1, 2}, f"qv/wind columns must rank worst, got {worst_two}"
    # the tiny-T column (0) is the LEAST bad
    assert int(jnp.argmin(fields.combined_score)) == 0


def test_rank_worst_columns_selects_highest():
    score = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    idx, vals, valid = rank_worst_columns(score, 2)
    assert idx.shape == (2,)
    assert float(vals[0]) == pytest.approx(0.9)
    assert float(vals[1]) == pytest.approx(0.5)
    # Flat index 1 == (0,1) is the worst.
    assert int(idx[0]) == 1
    assert bool(jnp.all(valid))


def test_rank_worst_columns_respects_valid_mask():
    score = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    mask = jnp.array([[True, False], [True, True]])
    idx, vals, valid = rank_worst_columns(score, 1, valid_mask=mask)
    # 0.9 masked out → worst valid is 0.5 at flat index 2.
    assert int(idx[0]) == 2
    assert float(vals[0]) == pytest.approx(0.5)
    assert bool(valid[0])


def test_rank_caps_at_n_columns():
    score = jnp.array([0.1, 0.2, 0.3])
    idx, vals, valid = rank_worst_columns(score, 10)
    assert idx.shape == (3,)


def test_rank_flags_padded_slots_when_n_exceeds_valid_count():
    """n > number of valid columns → trailing slots flagged invalid."""
    score = jnp.array([[0.1, 0.9], [0.5, 0.2]])
    mask = jnp.array([[True, False], [False, False]])  # only one valid column
    idx, vals, valid = rank_worst_columns(score, 3, valid_mask=mask)
    assert idx.shape == (3,)
    # Exactly one valid slot; the rest fell back to -inf.
    assert bool(valid[0])
    assert int(jnp.sum(valid)) == 1
    assert int(idx[0]) == 0  # flat index of (0,0), score 0.1, the only valid


def test_score_columns_raises_on_partial_precip():
    nlev = 3
    w = normalized_mass_weights(jnp.ones((nlev,)))
    shape = (1, 1, nlev)
    z = jnp.zeros(shape)
    with pytest.raises(ValueError):
        score_columns(
            T_model=z, qv_model=z, u_model=z, v_model=z,
            T_ref=z, qv_ref=z, u_ref=z, v_ref=z,
            mass_weights=w,
            precip_model_mm_day=jnp.zeros((1, 1)),
            precip_ref_mm_day=None,
        )


def test_all_nan_column_is_finite_zero_with_finite_grad():
    """A fully-masked column must yield finite zero RMSE and finite gradient."""
    nlev = 4
    w = normalized_mass_weights(jnp.ones((nlev,)))
    ref = jnp.full((1, nlev), jnp.nan)  # entire column missing in ERA5

    def rmse_of(scale):
        model = jnp.full((1, nlev), 1.0) * scale
        return per_column_weighted_rmse(model, ref, w)[0]

    val = float(rmse_of(1.0))
    assert val == pytest.approx(0.0, abs=1e-12)
    g = jax.grad(rmse_of)(1.0)
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(0.0, abs=1e-12)


def test_exact_match_rmse_grad_is_finite():
    """sqrt(0) AD hazard: grad of a zero-error column must be finite (zero)."""
    nlev = 5
    w = normalized_mass_weights(jnp.ones((nlev,)))
    ref = jnp.linspace(280.0, 220.0, nlev)[None, :]

    def rmse_of(scale):
        model = ref * scale
        return jnp.sum(per_column_weighted_rmse(model, ref, w))

    g = jax.grad(rmse_of)(1.0)  # at scale=1 model==ref, RMSE==0
    assert jnp.isfinite(g)


def test_score_columns_grad_finite_at_zero_error():
    nlev = 4
    w = normalized_mass_weights(jnp.ones((nlev,)))
    shape = (2, 2, nlev)
    T_ref = jnp.full(shape, 250.0)
    qv = jnp.full(shape, 1e-3)
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)

    def loss(scale):
        fields = score_columns(
            T_model=T_ref * scale, qv_model=qv, u_model=u, v_model=v,
            T_ref=T_ref, qv_ref=qv, u_ref=u, v_ref=v,
            mass_weights=w,
            config=ColumnErrorConfig(precip_weight=0.0),
        )
        return jnp.sum(fields.combined_score)

    g = jax.grad(loss)(1.0)  # zero error everywhere
    assert jnp.isfinite(g)


def test_vector_wind_asymmetric_nan_drops_level():
    """NaN in only one component drops the whole level (valid = u & v)."""
    nlev = 3
    w = normalized_mass_weights(jnp.ones((nlev,)))
    u_m = jnp.zeros((1, nlev))
    v_m = jnp.zeros((1, nlev))
    u_r = jnp.full((1, nlev), 3.0)
    v_r = jnp.full((1, nlev), 4.0)
    # Make v NaN at level 1 only; that level is dropped for both components.
    v_r_nan = v_r.at[0, 1].set(jnp.nan)
    rmse = float(per_column_vector_wind_rmse(u_m, v_m, u_r, v_r_nan, w)[0])
    # Remaining levels still have (3,4) error → vector RMS stays 5.
    assert rmse == pytest.approx(5.0, abs=1e-10)


def test_combined_score_differentiable():
    """The score must flow through jax.grad (autodiff is first-class)."""
    nlev = 4
    w = normalized_mass_weights(jnp.ones((nlev,)))
    shape = (2, 2, nlev)
    T_ref = jnp.full(shape, 250.0)
    qv = jnp.full(shape, 1e-3)
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)

    def loss(scale):
        fields = score_columns(
            T_model=T_ref + scale, qv_model=qv, u_model=u, v_model=v,
            T_ref=T_ref, qv_ref=qv, u_ref=u, v_ref=v,
            mass_weights=w,
            config=ColumnErrorConfig(precip_weight=0.0),
        )
        return jnp.sum(fields.combined_score)

    g = jax.grad(loss)(2.0)
    assert jnp.isfinite(g)
    assert float(g) > 0.0  # larger T error → larger score
