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
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
from legoesm.training.column_manifest import ColumnEnvironment, ColumnRecord
from legoesm.training.compare_reanalysis import ColumnState
from legoesm.training.correction_loop import (
    CompareResult,
    make_compare_fn,
    run_correction_campaign,
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


def test_make_compare_fn_uses_real_comparison_in_loop():
    """End-to-end loop with the REAL compare_state_to_reference (not a mocked
    score field): a config-driven model that moves CLOSER to the ERA5 reference
    yields a measurable bias reduction."""
    nlat, nlon, nlev = 4, 4, 5
    shape = (nlat, nlon, nlev)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    lat = jnp.linspace(-60.0, 60.0, nlat)
    lon = jnp.linspace(0.0, 270.0, nlon)
    area_w = jnp.cos(jnp.deg2rad(lat))[:, None] * jnp.ones((nlat, nlon))

    # ERA5 reference column state.
    reference = ColumnState(
        T=jnp.full(shape, 250.0), q_v=jnp.full(shape, 1e-3),
        u=jnp.zeros(shape), v=jnp.zeros(shape), p_s=jnp.full((nlat, nlon), 1e5),
    )

    # SPATIALLY VARYING bias so the worst-column ranking is exercised: a big
    # +10 K bias at flat columns {0,5,10,15}, a small +1 K elsewhere.
    bias_field = np.full((nlat, nlon), 1.0)
    for flat in (0, 5, 10, 15):
        bias_field[np.unravel_index(flat, (nlat, nlon))] = 10.0
    bias_field = jnp.asarray(bias_field)

    def run_amip_fn(config):
        # Corrected (per-column tau_equator): bias removed; else the bias field.
        corrected = jnp.ndim(jnp.asarray(config.tau_equator)) > 0
        b = jnp.zeros((nlat, nlon)) if corrected else bias_field
        return reference._replace(T=reference.T + b[:, :, None])

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, area_weights=area_w, n_worst=4,
        run_amip_fn=run_amip_fn,
    )

    def diagnose(record, model_ctx):
        return _Eddy(K=jnp.array([8.0]), valid=jnp.array([True]))

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="gray_tau_equator", grid_shape=(nlat, nlon),
        background=7.2,
    )
    # The REAL manifest must have flagged exactly the 4 high-bias columns.
    flagged = {int(r.flat_index) for r in compare_fn(GrayRadiationConfig()).manifest}
    assert flagged == {0, 5, 10, 15}
    # The corrected model matches ERA5 ⇒ zero bias ⇒ improvement detected.
    assert bool(result.bias.improved)
    assert float(result.bias.updated_bias) == pytest.approx(0.0, abs=1e-9)
    assert float(result.bias.baseline_bias) > 0.0
    assert result.n_corrected == 4


def test_make_compare_fn_real_compare_not_improved():
    """Real-compare path: a correction that does NOT remove the bias reports
    improved == False."""
    nlat, nlon, nlev = 4, 4, 4
    shape = (nlat, nlon, nlev)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    lat = jnp.linspace(-30.0, 30.0, nlat)
    lon = jnp.linspace(0.0, 270.0, nlon)
    area_w = jnp.ones((nlat, nlon))
    reference = ColumnState(
        T=jnp.full(shape, 250.0), q_v=jnp.full(shape, 1e-3),
        u=jnp.zeros(shape), v=jnp.zeros(shape), p_s=jnp.full((nlat, nlon), 1e5),
    )

    def run_amip_fn(config):
        # The "correction" makes it WORSE (bias grows from 3 to 5 K).
        b = 5.0 if jnp.ndim(jnp.asarray(config.tau_equator)) > 0 else 3.0
        return reference._replace(T=reference.T + b)

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, area_weights=area_w, n_worst=4,
        run_amip_fn=run_amip_fn,
    )
    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn,
        diagnose_fn=lambda r, c: _Eddy(K=jnp.array([8.0]), valid=jnp.array([True])),
        promotion_key="gray_tau_equator", grid_shape=(nlat, nlon), background=7.2,
    )
    assert not bool(result.bias.improved)
    assert float(result.bias.updated_bias) > float(result.bias.baseline_bias)


def test_make_compare_fn_rejects_reserved_kwarg():
    ref = ColumnState(
        T=jnp.zeros((2, 2, 3)), q_v=jnp.zeros((2, 2, 3)),
        u=jnp.zeros((2, 2, 3)), v=jnp.zeros((2, 2, 3)), p_s=jnp.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="may not override"):
        make_compare_fn(
            reference=ref, sigma_full=jnp.zeros(3), sigma_half=jnp.zeros(4),
            lat_deg=jnp.zeros(2), lon_deg=jnp.zeros(2), area_weights=jnp.ones((2, 2)),
            n_worst=1, run_amip_fn=lambda c: ref,
            model=ref,  # reserved (set internally) -> reaches compare_kwargs -> raises
        )


def test_campaign_accumulates_corrections_across_rounds():
    """The iterative loop: each round corrects the worst columns and ACCUMULATES
    (earlier corrections persist), driving the bias to zero over rounds."""
    nlat, nlon = 2, 2  # 4 columns
    area_w = jnp.ones((nlat, nlon))
    default_ck = float(CLUBBLiteConfig().C_K)

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        if ck.ndim == 0:  # round 0: nothing corrected yet
            bias = jnp.full((nlat, nlon), 8.0)
        else:
            ck2d = ck.reshape((nlat, nlon))
            # A column is "corrected" once its C_K differs from the default.
            bias = jnp.where(jnp.isclose(ck2d, default_ck), 8.0, 0.0)
        # combined_score field = the T bias proxy (uniform per column).
        return CompareResult(
            combined_score=bias, manifest=_manifest_from_bias(bias),
            area_weights=area_w, model_ctx=None,
        )

    def _manifest_from_bias(bias):
        flat = np.asarray(bias).reshape(-1)
        # the high-bias (uncorrected) columns are the worst.
        worst = [i for i in np.argsort(-flat)[:2] if flat[i] > 0.0]
        return [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                for i in worst]

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))  # != default

    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=2,
        compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(nlat, nlon),
        background=default_ck,
    )
    assert len(campaign.iterations) == 2
    # Round 0 corrected 2 columns, round 1 the other 2 (different worst columns).
    assert campaign.iterations[0].n_corrected == 2
    assert campaign.iterations[1].n_corrected == 2
    # ACCUMULATION: the final field corrected ALL 4 columns (round-0 corrections
    # were NOT overwritten by round 1).
    final_ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    assert np.all(np.isclose(final_ck, 0.9))
    # The bias fell monotonically to zero across rounds.
    assert float(campaign.iterations[0].bias.baseline_bias) == pytest.approx(8.0)
    assert float(campaign.iterations[1].bias.updated_bias) == pytest.approx(0.0, abs=1e-9)


def test_campaign_zero_iterations_is_noop():
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=0,
        compare_fn=lambda c: CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2))),
        diagnose_fn=lambda r, c: None,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=0.4,
    )
    assert campaign.iterations == ()
    assert jnp.ndim(jnp.asarray(campaign.final_config.C_K)) == 0  # unchanged scalar
    # final_field is the (scalar) background broadcast to the grid.
    assert campaign.final_field.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(campaign.final_field), 0.4)


def test_campaign_negative_iterations_raises():
    with pytest.raises(ValueError, match="n_iterations must be"):
        run_correction_campaign(
            CLUBBLiteConfig(), n_iterations=-1,
            compare_fn=lambda c: CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2))),
            diagnose_fn=lambda r, c: None,
            promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        )


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


# --- LES-cost reduction via environment clustering (iter 34) ----------------

def _crec(flat, score, sst, cape, shear):
    return ColumnRecord(
        flat_index=flat, grid_index=(flat,), lat_deg=0.0, lon_deg=0.0,
        time_index=0, combined_score=score, T_rmse_K=0.0, qv_rmse_kg_kg=0.0,
        wind_rmse_m_s=0.0, precip_err_mm_day=0.0,
        environment=ColumnEnvironment(sst_K=sst, cape_J_kg=cape, bulk_shear_m_s=shear),
    )


# 4 worst columns on a (2,3)=6 grid: cols 0,1 (cold/dry group A) + 4,5 (warm/moist B).
_CLUSTER_WORST = [
    _crec(0, 2.0, 280.0, 100.0, 2.0),
    _crec(1, 1.5, 281.0, 120.0, 2.5),
    _crec(4, 1.8, 302.0, 3000.0, 25.0),
    _crec(5, 1.4, 303.0, 2900.0, 26.0),
]


def _cluster_compare_fn():
    baseline = jnp.asarray([[2.0, 2.0, 1.0], [1.0, 2.0, 2.0]])
    corrected = jnp.ones((2, 3))

    def compare_fn(config):
        score = corrected if _is_corrected(config) else baseline
        return CompareResult(
            combined_score=score, manifest=_CLUSTER_WORST,
            area_weights=jnp.ones((2, 3)), model_ctx=None,
        )

    return compare_fn


def _diagnose_by_cape(record, model_ctx):
    """K depends on CAPE, so the two cluster representatives give DISTINCT
    coefficients (10 for the low-CAPE group, 20 for the high-CAPE group)."""
    k = 10.0 if record.environment.cape_J_kg < 1000.0 else 20.0
    return _Eddy(K=jnp.array([k, k]), valid=jnp.array([True, True]))


def test_les_budget_clusters_and_reduces_diagnoses():
    """les_budget=2 ⇒ diagnose only the 2 environment representatives, then map
    each representative's coefficient to its cluster members (all 4 corrected)."""
    n_calls = {"n": 0}

    def diag(rec, ctx):
        n_calls["n"] += 1
        return _diagnose_by_cape(rec, ctx)

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=_cluster_compare_fn(), diagnose_fn=diag,
        promotion_key="gray_tau_equator", grid_shape=(2, 3),
        background=7.2, les_budget=2,
    )
    assert result.n_corrected == 4          # all worst columns corrected
    assert result.n_diagnosed == 2          # but only 2 LES diagnoses run
    assert n_calls["n"] == 2                 # diagnose_fn called exactly twice
    f = np.asarray(result.feedback_field).reshape(-1)
    # group A (cols 0,1) → low-CAPE rep's K=10; group B (cols 4,5) → 20.
    assert f[0] == pytest.approx(10.0) and f[1] == pytest.approx(10.0)
    assert f[4] == pytest.approx(20.0) and f[5] == pytest.approx(20.0)
    # non-worst columns keep the background.
    assert f[2] == pytest.approx(7.2) and f[3] == pytest.approx(7.2)


def test_les_budget_none_diagnoses_every_worst_column():
    n_calls = {"n": 0}

    def diag(rec, ctx):
        n_calls["n"] += 1
        return _diagnose_by_cape(rec, ctx)

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=_cluster_compare_fn(), diagnose_fn=diag,
        promotion_key="gray_tau_equator", grid_shape=(2, 3), background=7.2,
    )
    assert result.n_corrected == 4
    assert result.n_diagnosed == 4   # no budget → diagnose all
    assert n_calls["n"] == 4


def test_les_budget_geq_records_diagnoses_all():
    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=_cluster_compare_fn(), diagnose_fn=_diagnose_by_cape,
        promotion_key="gray_tau_equator", grid_shape=(2, 3), background=7.2,
        les_budget=10,  # >= 4 worst columns → diagnose all
    )
    assert result.n_diagnosed == 4


def test_les_budget_nonpositive_raises():
    with pytest.raises(ValueError, match="les_budget must be > 0"):
        run_correction_iteration(
            GrayRadiationConfig(),
            compare_fn=_cluster_compare_fn(), diagnose_fn=_diagnose_by_cape,
            promotion_key="gray_tau_equator", grid_shape=(2, 3), background=7.2,
            les_budget=0,
        )
