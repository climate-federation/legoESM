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

# The correction loop is a scientific (not float32/Metal) path — run x64 so the
# registered-bound clamp + bias reductions are at the production precision (a float32
# field clamps to float32(hi), e.g. float32(1.2)=1.2000000476 > the float64 bound; the
# campaign itself runs x64). Matches tests/run/test_run_correction_campaign.py.
jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig  # noqa: E402
from legoesm.training.column_manifest import (  # noqa: E402
    ColumnEnvironment,
    ColumnRecord,
)
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.correction_loop import (  # noqa: E402
    CompareResult,
    CorrectionSpec,
    MultiCorrectionResult,
    make_compare_fn,
    run_correction_campaign,
    run_correction_iteration,
    run_multi_correction_campaign,
    run_multi_correction_iteration,
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


def test_loop_iteration_is_deterministic_end_to_end():
    """The whole correction iteration (rank → diagnose → assemble feedback → apply →
    re-compare) must be BIT-reproducible: identical inputs ⇒ identical corrected
    field + bias.

    Composes the per-stage determinism locks (ranking tie-break iter 212, clustering
    tie-break iter 224, feedback assembly) into an END-TO-END guarantee.  A
    composition-level non-determinism — a ``set``/``dict`` iteration order or a
    Python hash-order leak threaded through the manifest→feedback chain — would slip
    past the per-stage tests yet make a resumed/re-run campaign drift; running the
    SAME iteration twice and demanding bit-identical output catches it.
    """
    baseline = [[2.0, 1.0], [1.0, 2.0]]
    corrected = [[1.0, 1.0], [1.0, 1.0]]

    def _run():
        return run_correction_iteration(
            GrayRadiationConfig(),
            compare_fn=_make_compare_fn(
                baseline_score=baseline, corrected_score=corrected),
            diagnose_fn=_diagnose,
            promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
        )

    r1, r2 = _run(), _run()
    np.testing.assert_array_equal(
        np.asarray(r1.updated_config.tau_equator),
        np.asarray(r2.updated_config.tau_equator))
    assert float(r1.bias.baseline_bias) == float(r2.bias.baseline_bias)
    assert float(r1.bias.updated_bias) == float(r2.bias.updated_bias)
    assert float(r1.worst_column_change) == float(r2.worst_column_change)
    assert r1.n_corrected == r2.n_corrected == 2


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


def test_loop_empty_local_manifest_proceeds_when_global_count_nonzero():
    """DISTRIBUTED gate (iter 88): a rank with an EMPTY local manifest must still
    RE-RUN ``compare_fn`` (the model, an MPI collective) when ANOTHER rank owns a
    flagged column — else it skips the collective the peers execute and deadlocks.
    A ``global_reduce`` that reports the global count as nonzero forces the
    collective proceed path even with no local records (compare called TWICE)."""
    compare_calls = {"n": 0}

    def compare_fn(config):
        compare_calls["n"] += 1
        return CompareResult(
            combined_score=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            manifest=[],  # THIS rank owns nothing flagged...
            area_weights=_AREA_W, model_ctx=None,
        )

    # ...but a peer rank does: the global count is len(local)=0 + 1 = 1 > 0.
    def global_reduce(x):
        return jnp.asarray(x) + jnp.asarray(1, dtype=jnp.asarray(x).dtype)

    result = run_correction_iteration(
        GrayRadiationConfig(), compare_fn=compare_fn, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2,
        global_reduce=global_reduce,
    )
    assert compare_calls["n"] == 2   # baseline + the lockstep re-run (NOT a no-op)
    assert result.n_corrected == 0   # this rank still corrected none of its own
    # The single-process default (no reducer) IS a no-op for the same empty manifest.
    solo = {"n": 0}

    def compare_solo(config):
        solo["n"] += 1
        return CompareResult(jnp.array([[1.0, 1.0], [1.0, 1.0]]), [], _AREA_W)

    run_correction_iteration(
        GrayRadiationConfig(), compare_fn=compare_solo, diagnose_fn=_diagnose,
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=7.2)
    assert solo["n"] == 1            # no peer ⇒ no-op, single compare


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
    # END-TO-END per-variable wiring (iter 131): the REAL make_compare_fn ⇒
    # compose_compare_fn must populate CompareResult.error_fields + have_precip so the
    # loop computes result.per_variable_bias — a regression that dropped error_fields
    # (mocks supply them directly, so the unit tests would NOT catch it) makes the
    # per-variable trajectory silently None in real campaigns.  The corrected model
    # matches ERA5, so the T-RMSE falls from a real bias to 0 ⇒ T_improved.
    pv = result.per_variable_bias
    assert pv is not None
    assert float(pv.baseline.global_T_rmse_K) > 0.0
    assert float(pv.updated.global_T_rmse_K) == pytest.approx(0.0, abs=1e-9)
    assert bool(pv.T_improved)
    # have_precip THREADED through the real path: no precip in the states ⇒ precip is
    # NaN (not a spurious 0) — locks the iter-131 have_precip plumbing end-to-end.
    assert bool(jnp.isnan(pv.baseline.global_precip_err_mm_day))


def test_make_compare_fn_model_ctx_is_the_model_run_not_the_reference():
    """Glue contract (iter 407): ``compare_fn`` returns ``model_ctx`` = the
    ``run_amip_fn`` MODEL state — the state every LES diagnosis extracts its column
    FORCING from (``make_les_diagnose_fn`` reads ``model_ctx.T/q_v/u/v/p_s/sst_K``),
    NOT the ERA5 reference.  A regression returning the REFERENCE as ``model_ctx`` would
    pass the scoring tests (the RMSE is symmetric in which state is the 'model') yet make
    every diagnosis force its LES from the REANALYSIS instead of the model — silently
    diagnosing the wrong column state.  Locked with a model state DISTINCT from the
    reference (T+7, u+3)."""
    nlat, nlon, nlev = 2, 2, 3
    shape = (nlat, nlon, nlev)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    reference = ColumnState(
        T=jnp.full(shape, 250.0), q_v=jnp.full(shape, 1e-3),
        u=jnp.zeros(shape), v=jnp.zeros(shape), p_s=jnp.full((nlat, nlon), 1e5))
    model_T = reference.T + 7.0          # noqa: N806 — T is the temperature symbol
    model_u = reference.u + 3.0

    def run_amip_fn(config):                                  # noqa: ARG001
        return reference._replace(T=model_T, u=model_u)

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=jnp.linspace(-30.0, 30.0, nlat), lon_deg=jnp.linspace(0.0, 180.0, nlon),
        area_weights=jnp.ones((nlat, nlon)), n_worst=1, run_amip_fn=run_amip_fn)
    res = compare_fn(object())
    # model_ctx carries the MODEL state (the diagnosis source), NOT the reference.
    np.testing.assert_array_equal(np.asarray(res.model_ctx.T), np.asarray(model_T))
    np.testing.assert_array_equal(np.asarray(res.model_ctx.u), np.asarray(model_u))
    assert not np.allclose(np.asarray(res.model_ctx.T), np.asarray(reference.T))


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


def test_campaign_resume_is_bit_exactly_equivalent_to_uninterrupted():
    """HPC reproducibility / desync guard: a checkpoint+resume across a ROUND boundary
    reproduces the uninterrupted trajectory BIT-EXACTLY.  The loop offsets only the
    checkpoint LABEL by ``start_round`` (correction_loop.py:328) and carries every
    accepted state in ``initial_field`` — so resuming from the round-0 checkpoint MUST
    yield the same round-1 field as never interrupting.  A past 'checkpoint resume
    desync' bug motivates locking this end-to-end: the per-piece roundtrip/count tests
    (test_run_correction_campaign) never run the ACTUAL loop across the boundary.

    RESUME CONTRACT (the executable spec): ``initial_config`` must ALREADY carry
    ``initial_field`` — i.e. resume with BOTH the accumulated field AND the config
    promoted FROM it, exactly as the CLI does (``_load_single_resume`` rebuilds
    ``CLUBBLiteConfig(C_K=field.reshape(-1))``).  The first resumed round's
    ``compare_fn`` reads ``config`` (NOT ``base``), so a SCALAR ``initial_config`` +
    array ``initial_field`` would make round-1 re-rank against the un-promoted uniform
    bias and silently mis-select columns.  With deterministic mocks, any hidden
    round-index dependence — or a mis-modeled resume — diverges the two fields."""
    nlat, nlon = 2, 2
    area_w = jnp.ones((nlat, nlon))
    default_ck = float(CLUBBLiteConfig().C_K)

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        if ck.ndim == 0:                                     # round 0: nothing corrected
            bias = jnp.full((nlat, nlon), 8.0)
        else:
            bias = jnp.where(
                jnp.isclose(ck.reshape((nlat, nlon)), default_ck), 8.0, 0.0)
        flat = np.asarray(bias).reshape(-1)
        manifest = [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                    for i in np.argsort(-flat)[:2] if flat[i] > 0.0]
        return CompareResult(combined_score=bias, manifest=manifest,
                             area_weights=area_w, model_ctx=None)

    def diagnose(record, ctx):                               # noqa: ARG001
        return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))   # != default

    kw = dict(compare_fn=compare_fn, diagnose_fn=diagnose,
              promotion_key="clubb_lite_C_K", grid_shape=(nlat, nlon),
              background=default_ck)

    # Uninterrupted: capture the ACCEPTED accumulated field after EACH round.
    fields = []
    run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=2,
        checkpoint_callback=lambda r, res, field: fields.append(np.asarray(field)),
        **kw)
    assert len(fields) == 2                                  # field after round 0, round 1

    # Resume the SECOND round from the round-0 checkpoint, reconstructing the config
    # FROM the field exactly as the CLI's _load_single_resume does (start_round=1
    # offsets only the label).
    resume_field = jnp.asarray(fields[0])
    resume_config = CLUBBLiteConfig(C_K=resume_field.reshape(-1))
    resumed = run_correction_campaign(
        resume_config, n_iterations=1,
        initial_field=resume_field, start_round=1, **kw)
    f_resumed = np.asarray(resumed.final_config.C_K).reshape(-1)

    # BIT-EXACT: resuming reproduces the uninterrupted round-1 field exactly.
    np.testing.assert_array_equal(f_resumed, fields[1].reshape(-1))
    # NON-VACUOUS boundary: round 0 left 2 columns uncorrected; the resumed round
    # corrects them (2 → 4), so the equality is not trivially true of a no-op.
    assert int(np.sum(np.isclose(fields[0].reshape(-1), 0.9))) == 2
    assert int(np.sum(np.isclose(f_resumed, 0.9))) == 4


def test_campaign_resume_rejects_config_not_carrying_field():
    """Fail-loud on the resume foot-gun the equivalence test documents: a SCALAR
    initial_config paired with an ARRAY initial_field and ≥1 round is a silent desync
    (the first round's compare would re-rank against the un-promoted uniform field), so
    it RAISES the resume contract.  n_iterations=0 is EXEMPT (no round reads the config
    — see test_campaign_resume_zero_iterations_returns_initial_field).  The CORRECT
    resume (config promoted FROM the field) is accepted."""
    field = jnp.full((2, 2), 0.7)
    noop_compare = lambda c: CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)))  # noqa: E731,ARG005
    kw = dict(compare_fn=noop_compare, diagnose_fn=lambda r, c: None,
              promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=0.4,
              initial_field=field)
    with pytest.raises(ValueError, match="resume contract violated"):
        run_correction_campaign(CLUBBLiteConfig(), n_iterations=1, **kw)   # SCALAR config
    # Config promoted FROM the field ⇒ accepted (noop round keeps the resumed field).
    ok = run_correction_campaign(
        CLUBBLiteConfig(C_K=field.reshape(-1)), n_iterations=1, **kw)
    np.testing.assert_allclose(np.asarray(ok.final_field), 0.7)


def test_campaign_resume_rejects_scalar_config_even_at_single_column():
    """codex iter 401 (Finding 1.1): at ncol=1 a SCALAR config leaf + a (1,) field share
    the same reshape(-1) shape AND value, so the shape + array_equal checks ALONE would
    FALSELY pass — yet a scalar config still triggers the first round's round-0
    uniform-bias branch (the compare reads ``config.<field>``'s NDIM).  The explicit
    scalar-vs-array ndim check rejects it; the (1,)-array-carrying config is accepted."""
    field = jnp.full((1, 1), 0.7)                     # a single-column accumulated field
    noop = lambda c: CompareResult(jnp.zeros((1, 1)), [], jnp.ones((1, 1)))  # noqa: E731,ARG005
    kw = dict(compare_fn=noop, diagnose_fn=lambda r, c: None,
              promotion_key="clubb_lite_C_K", grid_shape=(1, 1), background=0.4,
              initial_field=field)
    with pytest.raises(ValueError, match=r"scalar \(un-promoted\)"):
        run_correction_campaign(CLUBBLiteConfig(C_K=0.7), n_iterations=1, **kw)  # scalar == value
    ok = run_correction_campaign(
        CLUBBLiteConfig(C_K=field.reshape(-1)), n_iterations=1, **kw)   # (1,) array ⇒ OK
    np.testing.assert_allclose(np.asarray(ok.final_field), 0.7)


def test_multi_campaign_resume_rejects_config_not_carrying_field():
    """The SAME resume contract on the multi-coefficient path: a SCALAR initial_config
    + an array ``initial_fields`` entry (≥1 round) RAISES rather than silently
    desyncing the first round's worst-column ranking."""
    field = jnp.full((2, 2), 0.7)
    with pytest.raises(ValueError, match="resume contract violated"):
        run_multi_correction_campaign(
            CLUBBLiteConfig(), 1, _SPECS,
            compare_fn=lambda c: CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2))),
            diagnose_fn=lambda r, c: None, grid_shape=(2, 2),
            initial_fields={"clubb_lite_C_K": field})


def _always_worst_compare_fn(nlat=2, nlon=2):
    """compare_fn that ALWAYS reports a high uniform bias ⇒ the 2 worst columns are
    flagged every round (regardless of the config) — to exercise the dry-LES abort."""
    area_w = jnp.ones((nlat, nlon))

    def compare_fn(config):                              # noqa: ARG001
        bias = jnp.full((nlat, nlon), 8.0)
        flat = np.asarray(bias).reshape(-1)
        manifest = [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                    for i in np.argsort(-flat)[:2]]
        return CompareResult(combined_score=bias, manifest=manifest,
                             area_weights=area_w, model_ctx=None)
    return compare_fn


def _diag_invalid(record, ctx):                          # noqa: ARG001
    return _Eddy(K=jnp.array([0.9]), valid=jnp.array([False]))  # REJECTED by realism


def test_campaign_aborts_on_dry_les_diagnoses():
    """`patience` consecutive rounds that flag columns but produce ZERO valid LES
    diagnoses ⇒ the campaign STOPS early with stop_reason='no_valid_diagnoses' (the
    spin-off LES develops no turbulence — don't waste multi-day rounds; iter 101)."""
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=5,
        compare_fn=_always_worst_compare_fn(), diagnose_fn=_diag_invalid,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2)
    assert campaign.stop_reason == "no_valid_diagnoses"
    assert len(campaign.iterations) == 2                 # stopped after 2 dry rounds
    assert all(it.n_corrected > 0 for it in campaign.iterations)
    assert all(it.n_diagnoses_valid == 0 for it in campaign.iterations)


def test_campaign_no_dry_abort_when_diagnoses_valid():
    """Valid diagnoses every round ⇒ NO dry-abort; the campaign runs all rounds."""
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=3,
        compare_fn=_always_worst_compare_fn(),
        diagnose_fn=lambda r, c: _Eddy(K=jnp.array([0.9]), valid=jnp.array([True])),
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2)
    assert campaign.stop_reason == "max_iterations"
    assert len(campaign.iterations) == 3


def test_campaign_dry_abort_can_be_disabled():
    """stop_on_no_valid_diagnoses=False keeps the old behaviour (run all rounds)."""
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=3,
        compare_fn=_always_worst_compare_fn(), diagnose_fn=_diag_invalid,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2,
        stop_on_no_valid_diagnoses=False)
    assert campaign.stop_reason == "max_iterations"
    assert len(campaign.iterations) == 3


def test_campaign_noop_round_is_not_dry():
    """A NO-OP round (no columns flagged — bias already zero) is NOT 'dry' (there was
    nothing to diagnose), so it must NOT trigger the dry-abort (iter 101 edge case)."""
    def compare_zero(config):                            # noqa: ARG001
        z = jnp.zeros((2, 2))
        return CompareResult(combined_score=z, manifest=[],
                             area_weights=jnp.ones((2, 2)), model_ctx=None)

    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=3,
        compare_fn=compare_zero, diagnose_fn=_diag_invalid,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2)
    assert campaign.stop_reason == "max_iterations"     # NOT no_valid_diagnoses
    assert len(campaign.iterations) == 3


def test_campaign_dry_abort_uses_global_not_local_valid_count():
    """The dry test reads the GLOBAL valid count (via global_reduce), NOT the rank-
    local one — proven by a reducer that ADDS a peer rank's valid diagnosis: this
    rank is LOCALLY dry (n_diagnoses_valid==0 every round) but GLOBALLY there IS a
    valid diagnosis, so the campaign must NOT abort. A rank-local implementation would
    (wrongly) abort and deadlock its peers under real MPI (iter 101 collective-safety)."""
    # +1 to every count == a peer rank that owns a flagged column with a VALID diagnosis.
    peer_has_valid = lambda x: x + 1                     # noqa: E731

    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=3,
        compare_fn=_always_worst_compare_fn(), diagnose_fn=_diag_invalid,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2,
        global_reduce=peer_has_valid)
    # globally non-dry (peer's valid count > 0) ⇒ NO abort, runs all rounds.
    assert all(it.n_diagnoses_valid == 0 for it in campaign.iterations)   # LOCALLY dry
    assert campaign.stop_reason == "max_iterations"
    assert len(campaign.iterations) == 3


def test_campaign_dry_abort_fires_when_globally_dry():
    """The companion: identity reduce (np=1, no peer) ⇒ the campaign IS globally dry
    and aborts. Together these pin the abort to the GLOBAL valid count (iter 101)."""
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=5,
        compare_fn=_always_worst_compare_fn(), diagnose_fn=_diag_invalid,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=float(CLUBBLiteConfig().C_K), patience=2,
        global_reduce=lambda x: x)                       # single-rank SUM == identity
    assert campaign.stop_reason == "no_valid_diagnoses"
    assert len(campaign.iterations) == 2


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
    # validity is counted over the 2 LES RUNS (representatives), NOT the 4 expanded
    # columns — so n_diagnoses_valid never exceeds n_diagnosed (iter 100 Codex fix).
    assert result.n_diagnoses_valid == 2
    assert n_calls["n"] == 2                 # diagnose_fn called exactly twice
    f = np.asarray(result.feedback_field).reshape(-1)
    # group A (cols 0,1) → low-CAPE rep's K=10; group B (cols 4,5) → 20.
    assert f[0] == pytest.approx(10.0) and f[1] == pytest.approx(10.0)
    assert f[4] == pytest.approx(20.0) and f[5] == pytest.approx(20.0)
    # non-worst columns keep the background.
    assert f[2] == pytest.approx(7.2) and f[3] == pytest.approx(7.2)


def test_les_budget_counts_valid_over_runs_not_columns():
    """One of the 2 cluster representatives is INVALID ⇒ n_diagnoses_valid == 1 even
    though each rep maps to 2 columns — the SHARP proof that validity is counted over
    LES RUNS (representatives), not the expanded columns (iter 100 Codex)."""
    def diag(rec, ctx):                                  # noqa: ARG001
        if rec.environment.cape_J_kg < 1000.0:
            return _Eddy(K=jnp.array([10.0, 10.0]), valid=jnp.array([True, True]))
        return _Eddy(K=jnp.array([20.0, 20.0]),         # high-CAPE rep: REJECTED
                     valid=jnp.array([False, False]))

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=_cluster_compare_fn(), diagnose_fn=diag,
        promotion_key="gray_tau_equator", grid_shape=(2, 3),
        background=7.2, les_budget=2,
    )
    assert result.n_corrected == 4          # all 4 worst columns flagged
    assert result.n_diagnosed == 2          # 2 LES runs
    assert result.n_diagnoses_valid == 1    # but only ONE run was valid (≤ n_diagnosed)
    f = np.asarray(result.feedback_field).reshape(-1)
    assert f[0] == pytest.approx(10.0) and f[1] == pytest.approx(10.0)  # valid rep
    assert f[4] == pytest.approx(7.2) and f[5] == pytest.approx(7.2)    # invalid → bg


def test_les_budget_exceeding_distinct_environments_early_stops():
    """les_budget > the number of DISTINCT environments ⇒ clustering early-stops to the
    actual distinct count (2 here, not the requested 3) rather than emitting duplicate
    representatives, and the consumer (_diagnose_columns) still expands EVERY worst column
    from the K'<budget reps WITHOUT an index error (rep[label] with labels in 0..K'-1).
    Locks the early-stop <-> expansion integration boundary the budget==distinct tests miss:
    a regression producing labels that assume the REQUESTED budget would IndexError here."""
    # 4 worst columns but only 2 DISTINCT environments (exact env-duplicate pairs), so a
    # budget of 3 cannot find a 3rd distinct representative → clustering stops early at 2.
    worst = [
        _crec(0, 2.0, 280.0, 100.0, 2.0),
        _crec(1, 1.5, 280.0, 100.0, 2.0),    # exact env-duplicate of col 0 (low CAPE)
        _crec(4, 1.8, 302.0, 3000.0, 25.0),
        _crec(5, 1.4, 302.0, 3000.0, 25.0),  # exact env-duplicate of col 4 (high CAPE)
    ]
    baseline = jnp.asarray([[2.0, 2.0, 1.0], [1.0, 2.0, 2.0]])

    def compare_fn(config):
        score = jnp.ones((2, 3)) if _is_corrected(config) else baseline
        return CompareResult(combined_score=score, manifest=worst,
                             area_weights=jnp.ones((2, 3)), model_ctx=None)

    n_calls = {"n": 0}

    def diag(rec, ctx):
        n_calls["n"] += 1
        return _diagnose_by_cape(rec, ctx)

    result = run_correction_iteration(
        GrayRadiationConfig(),
        compare_fn=compare_fn, diagnose_fn=diag,
        promotion_key="gray_tau_equator", grid_shape=(2, 3),
        background=7.2, les_budget=3,        # > the 2 distinct environments present
    )
    assert result.n_corrected == 4           # all worst columns still corrected
    assert result.n_diagnosed == 2           # EARLY STOP: 2 distinct envs, not the asked 3
    assert n_calls["n"] == 2                  # no duplicate-representative LES run
    f = np.asarray(result.feedback_field).reshape(-1)
    assert f[0] == pytest.approx(10.0) and f[1] == pytest.approx(10.0)   # low-CAPE group A
    assert f[4] == pytest.approx(20.0) and f[5] == pytest.approx(20.0)   # high-CAPE group B
    assert f[2] == pytest.approx(7.2) and f[3] == pytest.approx(7.2)     # non-worst → bg


def test_baseline_diverged_run_fails_loud():
    """run_correction_iteration FAILS LOUD when the BASELINE (current-config) model run is
    non-finite — a diverged/blown-up run — instead of flowing a garbage NaN-masked bias into
    the gate and wasting a multi-day HPC round (iter 301). A diverged CANDIDATE/line-search
    run is gate-rejected (NaN bias never < baseline), but a diverged BASELINE is fatal: there
    is nothing finite to correct against. The guard only fires for a real ColumnState
    model_ctx, so the mock tests above (model_ctx=None) are unaffected."""
    def _nan_model_compare(*_a, **_k):
        nan_state = ColumnState(
            T=jnp.full((2, 3, 4), jnp.nan), q_v=jnp.zeros((2, 3, 4)),
            u=jnp.zeros((2, 3, 4)), v=jnp.zeros((2, 3, 4)), p_s=jnp.full((2, 3), 1.0e5))
        return CompareResult(
            combined_score=jnp.ones((2, 3)), manifest=[],
            area_weights=jnp.ones((2, 3)), model_ctx=nan_state)

    with pytest.raises(ValueError, match=r"baseline model run.*DIVERGED"):
        run_correction_iteration(
            GrayRadiationConfig(), compare_fn=_nan_model_compare,
            diagnose_fn=lambda *_a: None, promotion_key="gray_tau_equator",
            grid_shape=(2, 3))


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


def _ef_compare_fn(shape=(2, 3)):
    """A compare_fn whose CompareResult carries per-variable error_fields: the
    CORRECTED config lowers the combined score AND T_rmse but RAISES wind_rmse (a
    trade-off the combined score hides)."""
    from legoesm.training.column_era5_metrics import ColumnErrorFields

    def _ef(t_rmse, wind_rmse):
        return ColumnErrorFields(
            T_rmse_K=jnp.full(shape, t_rmse), qv_rmse_kg_kg=jnp.full(shape, 1.0e-3),
            wind_rmse_m_s=jnp.full(shape, wind_rmse), precip_err_mm_day=jnp.zeros(shape),
            combined_score=jnp.full(shape, t_rmse))

    def compare_fn(config):
        corrected = _is_corrected(config)
        ef = _ef(1.0, 5.0) if corrected else _ef(4.0, 2.0)   # T 4->1 down, wind 2->5 UP
        return CompareResult(
            combined_score=jnp.full(shape, 1.0 if corrected else 2.0),
            manifest=_CLUSTER_WORST, area_weights=jnp.ones(shape),
            model_ctx=None, error_fields=ef, have_precip=False)

    return compare_fn


def test_run_correction_iteration_computes_per_variable_bias():
    """When compare_fn carries per-variable error_fields, the round result exposes a
    PerVariableBiasImprovement: T improved (4->1, global RMSE) but wind WORSENED (2->5)
    — the trade-off the combined score hides; precip is NaN (no precip compared)."""
    from legoesm.training.bias_metrics import PerVariableBiasImprovement

    res = run_correction_iteration(
        GrayRadiationConfig(), compare_fn=_ef_compare_fn(),
        diagnose_fn=_diagnose_by_cape, promotion_key="gray_tau_equator",
        grid_shape=(2, 3), background=7.2)
    pvb = res.per_variable_bias
    assert isinstance(pvb, PerVariableBiasImprovement)
    assert float(pvb.baseline.global_T_rmse_K) == pytest.approx(4.0)   # uniform ⇒ RMSE=value
    assert float(pvb.updated.global_T_rmse_K) == pytest.approx(1.0)
    assert bool(pvb.T_improved) and not bool(pvb.wind_improved)        # trade-off exposed
    assert bool(jnp.isnan(pvb.baseline.global_precip_err_mm_day))      # precip not compared


def test_run_correction_iteration_per_variable_bias_none_without_error_fields():
    """A mock compare_fn that omits error_fields ⇒ per_variable_bias is None (graceful;
    backward-compat for the loop's mock-driven tests)."""
    res = run_correction_iteration(
        GrayRadiationConfig(), compare_fn=_cluster_compare_fn(),
        diagnose_fn=_diagnose_by_cape, promotion_key="gray_tau_equator",
        grid_shape=(2, 3), background=7.2)
    assert res.per_variable_bias is None


# --- restartable campaign: checkpoint + resume (iter 41) --------------------

def _accum_campaign_fns(nlat=2, nlon=2):
    """compare_fn/diagnose for the accumulation scenario (a column is 'corrected'
    once its C_K differs from the default; the 2 worst uncorrected columns flag)."""
    area_w = jnp.ones((nlat, nlon))
    default_ck = float(CLUBBLiteConfig().C_K)

    def _manifest_from_bias(bias):
        flat = np.asarray(bias).reshape(-1)
        worst = [i for i in np.argsort(-flat)[:2] if flat[i] > 0.0]
        return [_Rec(flat_index=int(i), lat_deg=0.0, environment=_Env(0.0))
                for i in worst]

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        if ck.ndim == 0:
            bias = jnp.full((nlat, nlon), 8.0)
        else:
            ck2d = ck.reshape((nlat, nlon))
            bias = jnp.where(jnp.isclose(ck2d, default_ck), 8.0, 0.0)
        return CompareResult(
            combined_score=bias, manifest=_manifest_from_bias(bias),
            area_weights=area_w, model_ctx=None)

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))

    return compare_fn, diagnose, default_ck


def test_campaign_checkpoint_callback_per_round():
    """checkpoint_callback fires once per round with (start_round+i, result,
    accumulated_field); the field grows as corrections accumulate."""
    compare_fn, diagnose, default_ck = _accum_campaign_fns()
    calls = []

    def ckpt(round_idx, result, field):
        calls.append((round_idx, result.n_corrected, np.asarray(field).copy()))

    run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=2, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=default_ck, start_round=5, checkpoint_callback=ckpt)

    assert [c[0] for c in calls] == [5, 6]          # start_round offset
    assert all(c[1] == 2 for c in calls)            # 2 corrected per round
    # the checkpointed field accumulates (round 6 ≥ round 5 corrected columns).
    n5 = int(np.count_nonzero(np.isclose(calls[0][2].reshape(-1), 0.9)))
    n6 = int(np.count_nonzero(np.isclose(calls[1][2].reshape(-1), 0.9)))
    assert n6 >= n5 and n6 == 4


def test_campaign_resume_equals_uninterrupted():
    """Splitting a 2-round campaign into round-0 + a RESUMED round-1 (from the
    saved config + accumulated field) gives the SAME final config/field as the
    uninterrupted 2-round run — restart is seamless."""
    compare_fn, diagnose, default_ck = _accum_campaign_fns()
    kw = dict(compare_fn=compare_fn, diagnose_fn=diagnose,
              promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
              background=default_ck)

    full = run_correction_campaign(CLUBBLiteConfig(), n_iterations=2, **kw)
    part1 = run_correction_campaign(CLUBBLiteConfig(), n_iterations=1, **kw)
    resumed = run_correction_campaign(
        part1.final_config, n_iterations=1,
        initial_field=part1.final_field, start_round=1, **kw)

    np.testing.assert_allclose(
        np.asarray(resumed.final_config.C_K), np.asarray(full.final_config.C_K))
    np.testing.assert_allclose(
        np.asarray(resumed.final_field), np.asarray(full.final_field))
    # The resumed first round reproduces the full campaign's SECOND round.
    np.testing.assert_allclose(
        np.asarray(resumed.iterations[0].updated_config.C_K),
        np.asarray(full.iterations[1].updated_config.C_K))


def test_campaign_resume_zero_iterations_returns_initial_field():
    """n_iterations=0 on resume returns the supplied initial_field (the resume
    base) as the final field — NOT the scalar background."""
    field = jnp.full((2, 2), 0.7)
    camp = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=0,
        compare_fn=lambda c: CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2))),
        diagnose_fn=lambda r, c: None,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=0.4,
        initial_field=field)
    assert camp.iterations == ()
    np.testing.assert_allclose(np.asarray(camp.final_field), 0.7)  # not 0.4


# --- environment-kernel feedback strategy (iter 42) -------------------------

def test_loop_environment_strategy_generalizes_to_similar_columns():
    """feedback_strategy='environment' spreads the diagnosed coefficient to ALL
    env-similar columns (incl. non-worst), via env_grid_fn(model_ctx)."""
    grid_env = jnp.asarray([
        [280.0, 100.0, 2.0], [281.0, 110.0, 2.2],      # env A (cols 0,1)
        [302.0, 3000.0, 25.0], [301.0, 2950.0, 24.0],  # env B (cols 2,3)
    ])
    length_scales = jnp.std(grid_env, axis=0)

    def env_grid_fn(model_ctx):
        return grid_env, length_scales

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        if ck.ndim == 0:                       # baseline: col 0 (env A) is worst
            score = jnp.zeros((2, 2)).at[0, 0].set(8.0)
            manifest = [_crec(0, 8.0, 280.0, 100.0, 2.0)]
        else:
            score = jnp.zeros((2, 2))
            manifest = []
        return CompareResult(combined_score=score, manifest=manifest,
                             area_weights=jnp.ones((2, 2)), model_ctx=None)

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))

    result = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=0.4,
        feedback_strategy="environment", env_grid_fn=env_grid_fn)

    ck = np.asarray(result.updated_config.C_K).reshape(-1)
    # env A columns (0 AND the non-worst 1) generalized to ~0.9; env B kept 0.4.
    assert ck[0] == pytest.approx(0.9, abs=0.1)
    assert ck[1] == pytest.approx(0.9, abs=0.1)   # NON-worst, generalized
    assert ck[2] == pytest.approx(0.4, abs=0.1)
    assert ck[3] == pytest.approx(0.4, abs=0.1)


def test_loop_environment_strategy_requires_env_grid_fn():
    def compare_fn(config):
        return CompareResult(
            jnp.zeros((2, 2)).at[0, 0].set(8.0),
            [_crec(0, 8.0, 280.0, 100.0, 2.0)], jnp.ones((2, 2)))

    with pytest.raises(ValueError, match="requires env_grid_fn"):
        run_correction_iteration(
            CLUBBLiteConfig(), compare_fn=compare_fn,
            diagnose_fn=lambda r, c: _Eddy(K=jnp.array([0.9]), valid=jnp.array([True])),
            promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=0.4,
            feedback_strategy="environment", env_grid_fn=None)


# --- env-kernel PRODUCER wiring (iter 70: closes deploy Finding 6) -----------

_KERNEL_GRID_ENV = jnp.asarray([
    [280.0, 100.0, 2.0], [281.0, 110.0, 2.2],      # env A (cols 0,1)
    [302.0, 3000.0, 25.0], [301.0, 2950.0, 24.0],  # env B (cols 2,3)
])


def _kernel_env_grid_fn(model_ctx):
    return _KERNEL_GRID_ENV, jnp.std(_KERNEL_GRID_ENV, axis=0)


def _kernel_compare_fn(config):
    ck = jnp.asarray(config.C_K)
    if ck.ndim == 0:
        return CompareResult(
            jnp.zeros((2, 2)).at[0, 0].set(8.0),
            [_crec(0, 8.0, 280.0, 100.0, 2.0)], jnp.ones((2, 2)))
    return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)))


def _kernel_diagnose(record, ctx):
    return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))


def _kernel_diagnose_invalid(record, ctx):
    return _Eddy(K=jnp.array([0.9]), valid=jnp.array([False]))   # all-invalid


def test_env_strategy_all_invalid_diagnoses_skips_kernel_no_crash():
    """An "environment"-strategy round whose LES diagnoses are ALL invalid yields
    no transferable kernel — the OPTIONAL export is skipped (env_kernel=None), the
    campaign does NOT crash on build_env_kernel's strict 'no VALID samples' raise.
    The round itself still completes (the feedback keeps the background)."""
    result = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
        diagnose_fn=_kernel_diagnose_invalid, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=0.4, clip_to_bounds=True,
        feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)
    assert result.env_kernel is None          # gracefully skipped, not raised
    assert result.n_corrected == 1            # the round still ran


def test_env_strategy_iteration_exports_raw_kernel_matching_feedback_field():
    """The exported env_kernel reproduces THIS round's injected field on the SAME
    grid in the controlled case (scalar background, full step, no clip) — proving
    the kernel IS the raw env→coefficient regression the campaign applied, so a
    cross-resolution deploy lands the same values the campaign would have."""
    from legoesm.training.deploy_correction import apply_env_kernel_override

    result = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
        diagnose_fn=_kernel_diagnose, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=0.4,
        feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)

    assert result.env_kernel is not None
    assert result.env_kernel.field == "C_K"
    assert float(result.env_kernel.background) == pytest.approx(0.4)
    override, coverage = apply_env_kernel_override(
        result.env_kernel, _KERNEL_GRID_ENV)
    # frac=1, no clip, scalar bg ⇒ feedback_field == the raw env regression.
    np.testing.assert_allclose(
        np.asarray(override.clubb_lite.C_K).reshape(-1),
        np.asarray(result.feedback_field).reshape(-1), rtol=1e-5)
    # Only env A (cols 0,1) is near the single diagnosed sample; env B (cols 2,3)
    # falls back to background — so half the campaign's OWN grid is covered.
    assert coverage["fraction_covered"] == pytest.approx(0.5)


def test_env_kernel_deploys_on_different_resolution_grid():
    """The producer→serialize→deserialize→deploy-on-a-DIFFERENT-ncol chain: the
    saved kernel evaluates on a 6-column grid (vs the campaign's 4) + the deployed
    override passes validate_strict (the end-to-end non-dead gate)."""
    import json

    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.training.deploy_correction import (
        apply_env_kernel_override,
        env_kernel_from_dict,
        env_kernel_to_dict,
    )

    result = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
        diagnose_fn=_kernel_diagnose, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=0.4,
        feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)
    k2 = env_kernel_from_dict(json.loads(json.dumps(env_kernel_to_dict(result.env_kernel))))
    new_env = jnp.asarray([[280.5, 105.0, 2.1]] * 3 + [[301.5, 2975.0, 24.5]] * 3)
    override, coverage = apply_env_kernel_override(k2, new_env)
    assert np.asarray(override.clubb_lite.C_K).shape == (6,)   # not the campaign's 4
    assert coverage["n_columns"] == 6
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite", turbulence_override=override)
    cfg.validate_strict()


def test_static_or_nonclubb_strategy_exports_no_kernel():
    """env_kernel is None for static strategy AND for a non-CLUBB promotion key
    (the env kernel is a CLUBB-turbulence deploy artifact)."""
    static = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
        diagnose_fn=_kernel_diagnose, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=0.4, feedback_strategy="static")
    assert static.env_kernel is None

    def gray_compare(config):
        ck = jnp.asarray(config.tau_equator)
        if ck.ndim == 0:
            return CompareResult(jnp.zeros((2, 2)).at[0, 0].set(8.0),
                                 [_crec(0, 8.0, 280.0, 100.0, 2.0)], jnp.ones((2, 2)))
        return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)))

    gray = run_correction_iteration(
        GrayRadiationConfig(), compare_fn=gray_compare,
        diagnose_fn=lambda r, c: _Eddy(K=jnp.array([0.9]), valid=jnp.array([True])),
        promotion_key="gray_tau_equator", grid_shape=(2, 2), background=0.5,
        feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)
    assert gray.env_kernel is None   # non-CLUBB key → no kernel


def test_env_kernel_array_background_requires_scalar_fallback():
    """Condition: an accumulated per-column array background with NO
    kernel_background RAISES (no silent arbitrary out-of-hull fallback)."""
    with pytest.raises(ValueError, match="SCALAR out-of-hull fallback"):
        run_correction_iteration(
            CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
            diagnose_fn=_kernel_diagnose, promotion_key="clubb_lite_C_K",
            grid_shape=(2, 2), background=jnp.full((2, 2), 0.4),
            feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)
    # ...but supplying the scalar kernel_background succeeds (campaign forwards it).
    ok = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=_kernel_compare_fn,
        diagnose_fn=_kernel_diagnose, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=jnp.full((2, 2), 0.4), kernel_background=0.4,
        feedback_strategy="environment", env_grid_fn=_kernel_env_grid_fn)
    assert ok.env_kernel is not None
    assert float(ok.env_kernel.background) == pytest.approx(0.4)


def test_campaign_environment_strategy_accumulates_across_rounds():
    """2-round campaign with feedback_strategy='environment': round 0 corrects
    env A, round 1 corrects env B, and round 0's env-generalized values PERSIST
    (the round-k field is round-(k+1) background — array-background path)."""
    grid_env = jnp.asarray([
        [280.0, 100.0, 2.0], [281.0, 110.0, 2.2],      # env A (cols 0,1)
        [302.0, 3000.0, 25.0], [301.0, 2950.0, 24.0],  # env B (cols 2,3)
    ])
    length_scales = jnp.std(grid_env, axis=0)

    def env_grid_fn(model_ctx):
        return grid_env, length_scales

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        if ck.ndim == 0:                       # round 0: col 0 (env A) worst
            worst, env = 0, (280.0, 100.0, 2.0)
        else:
            ck4 = np.asarray(ck).reshape(-1)
            if not np.isclose(ck4[2], 0.4):    # env B already corrected → done
                return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)))
            worst, env = 2, (302.0, 3000.0, 25.0)
        score = jnp.zeros((2, 2)).reshape(-1).at[worst].set(8.0).reshape(2, 2)
        return CompareResult(score, [_crec(worst, 8.0, *env)], jnp.ones((2, 2)))

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([0.9]), valid=jnp.array([True]))

    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=2, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=0.4, feedback_strategy="environment", env_grid_fn=env_grid_fn)

    ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    # BOTH env clusters ended up corrected (~0.9): round-0 env-A correction
    # persisted into round 1 (array-background accumulation), and round 1 added env B.
    np.testing.assert_allclose(ck, 0.9, atol=0.1)
    assert campaign.iterations[0].n_corrected == 1   # env A worst column
    assert campaign.iterations[1].n_corrected == 1   # env B worst column


def _abs_target_setup(diag_for):
    """compare_fn + diagnose where combined_score[col] = |C_K[col] - 1.0| on a
    2x2 grid (uniform area weights). The single worst column is flagged each
    round; ``diag_for(flat_index) -> C_K`` sets the diagnosed coefficient, so a
    correction toward 1.0 lowers the global bias (improves) and one away raises
    it (worsens) — a controllable improve/worsen knob for the acceptance gate."""
    nlat, nlon = 2, 2
    area_w = jnp.ones((nlat, nlon))
    default_ck = float(CLUBBLiteConfig().C_K)
    target = 1.0

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        ck2d = (jnp.broadcast_to(ck, (nlat, nlon)) if ck.ndim == 0
                else ck.reshape((nlat, nlon)))
        score = jnp.abs(ck2d - target)
        flat = np.asarray(score).reshape(-1)
        worst = [int(i) for i in np.argsort(-flat)[:1] if flat[i] > 1e-9]
        manifest = [_Rec(flat_index=i, lat_deg=0.0, environment=_Env(0.0))
                    for i in worst]
        return CompareResult(combined_score=score, manifest=manifest,
                             area_weights=area_w, model_ctx=None)

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([diag_for(record.flat_index)]),
                     valid=jnp.array([True]))

    return compare_fn, diagnose, default_ck


def test_campaign_gate_rejects_worsening_round():
    # A round whose correction RAISES the global bias is rejected: the config +
    # field are discarded, leaving the unchanged scalar default.
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 5.0)  # away from 1.0
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True)
    assert campaign.accepted == (False,)
    assert not bool(campaign.iterations[0].bias.improved)
    assert jnp.ndim(jnp.asarray(campaign.final_config.C_K)) == 0   # unchanged
    np.testing.assert_allclose(np.asarray(campaign.final_field), dck)


def test_campaign_gate_keeps_improving_round():
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 1.0)  # onto the target
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True)
    assert campaign.accepted == (True,)
    assert bool(campaign.iterations[0].bias.improved)
    final_ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    assert final_ck[0] == pytest.approx(1.0)            # worst col corrected


def test_campaign_gate_default_off_keeps_worsening_round():
    # Backward compat: default accept_only_if_improved=False accumulates the
    # worsening correction unconditionally (the prior loop semantics).
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 5.0)
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck)
    assert campaign.accepted == (True,)                 # accepted unconditionally
    final_ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    assert final_ck[0] == pytest.approx(5.0)            # worsening correction KEPT


def test_campaign_gate_noop_round_vacuously_accepted():
    # A round with no flagged columns changes nothing → accepted even under the gate.
    def compare_fn(config):
        return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)), model_ctx=None)
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=lambda r, c: None, promotion_key="clubb_lite_C_K",
        grid_shape=(2, 2), background=0.4, accept_only_if_improved=True)
    assert campaign.accepted == (True,)
    assert campaign.iterations[0].n_corrected == 0


def test_campaign_gate_monotonic_two_rounds_no_regression():
    # Round 0 (worst col 0, diag 1.0) IMPROVES → kept; round 1 (now worst col 1,
    # diag 5.0) WORSENS → rejected. The accumulated field never regresses: the
    # round-0 correction survives and the round-1 worsening is discarded.
    compare_fn, diagnose, dck = _abs_target_setup(
        lambda i: 1.0 if i == 0 else 5.0)
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=2, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True)
    assert campaign.accepted == (True, False)
    final_ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    assert final_ck[0] == pytest.approx(1.0)            # round-0 kept
    assert final_ck[1] == pytest.approx(dck)            # round-1 rejected
    # final field == the round-0 accepted field (best-so-far, no regression).
    np.testing.assert_allclose(
        np.asarray(campaign.final_field),
        np.asarray(campaign.iterations[0].feedback_field))


def test_campaign_checkpoint_field_is_accepted_not_rejected_config():
    # Locks the HIGH checkpoint-desync fix: on a REJECTED round the
    # checkpoint_callback receives the PRIOR accepted base (`field`), which
    # DIVERGES from result.updated_config (the discarded worsening update). A
    # caller MUST persist `field`, not result.updated_config — else resume loads a
    # rejected C_K against an accepted field. This test would catch a revert.
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 5.0)  # worsens → reject
    seen = []

    def ckpt(round_idx, result, field):  # noqa: ARG001
        seen.append((np.asarray(field).copy(),
                     np.asarray(result.updated_config.C_K).copy(),
                     bool(result.bias.improved)))

    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True, checkpoint_callback=ckpt)

    assert campaign.accepted == (False,)
    field, rejected_ck, improved = seen[0]
    assert not improved
    # The DISCARDED updated_config carries the worsening 5.0 correction ...
    assert np.any(np.isclose(rejected_ck, 5.0))
    # ... but the checkpointed `field` is the ACCEPTED base (uniform default) —
    # it must NOT contain the rejected 5.0, or resume would desync.
    assert not np.any(np.isclose(field.reshape(-1), 5.0))
    np.testing.assert_allclose(field, dck)


def _line_search_setup(diag_value, target):
    """2x2 grid; combined_score[col] = |C_K[col] - target|; the single worst
    column is flagged and diagnosed `diag_value`. Choosing target/diag relative to
    the (0.4) background controls whether the full step overshoots while a smaller
    step still lands closer to target — the line-search knob."""
    nlat, nlon = 2, 2
    area_w = jnp.ones((nlat, nlon))

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        ck2d = (jnp.broadcast_to(ck, (nlat, nlon)) if ck.ndim == 0
                else ck.reshape((nlat, nlon)))
        score = jnp.abs(ck2d - target)
        flat = np.asarray(score).reshape(-1)
        worst = [int(i) for i in np.argsort(-flat)[:1] if flat[i] > 1e-9]
        manifest = [_Rec(flat_index=i, lat_deg=0.0, environment=_Env(0.0))
                    for i in worst]
        return CompareResult(combined_score=score, manifest=manifest,
                             area_weights=area_w, model_ctx=None)

    def diagnose(record, ctx):
        return _Eddy(K=jnp.array([diag_value]), valid=jnp.array([True]))

    return compare_fn, diagnose


def test_validate_step_fractions():
    from legoesm.training.correction_loop import _validate_step_fractions
    assert _validate_step_fractions(None) == (1.0,)
    # deduplicated + sorted DESCENDING (backtrack largest-first).
    assert _validate_step_fractions([0.25, 1.0, 0.5, 0.5]) == (1.0, 0.5, 0.25)
    with pytest.raises(ValueError, match="non-empty"):
        _validate_step_fractions([])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _validate_step_fractions([1.5])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _validate_step_fractions([0.0])


def test_iteration_line_search_picks_largest_improving_step():
    # Full step (C_K->2.0) overshoots target 1.0 and WORSENS; half step (->1.2)
    # improves. The search tries 1.0 first (rejected) then 0.5 (kept).
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=2.0, target=1.0)
    seen_ck = []

    def logged_compare(config):                   # record the worst-column C_K per call
        ck = np.asarray(config.C_K)
        seen_ck.append(float(ck.reshape(-1)[0]) if ck.ndim else float(ck))
        return compare_fn(config)

    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=logged_compare, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        step_fractions=(1.0, 0.5))
    assert res.step_fraction == pytest.approx(0.5)
    assert bool(res.bias.improved)
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    assert ck[0] == pytest.approx(1.2)            # 0.4 + 0.5*(2.0-0.4)
    # Non-vacuity: the FULL step (2.0) was actually tried + rejected BEFORE the
    # half step (1.2); exactly baseline + 2 candidate re-runs (no extra/skipped).
    assert seen_ck == pytest.approx([dck, 2.0, 1.2])


def test_iteration_line_search_all_worsen_falls_back_to_largest():
    """When NO step improves (the diagnosed C_K=0.1 moves the worst column AWAY from
    target 1.0 at every fraction: 1.0→0.1 bias 0.9, 0.5→0.25 bias 0.75, both > the 0.6
    baseline), the line search returns the LARGEST step (the k==0 fallback — never None,
    which would crash on unpack) reported as NOT improved, so the monotonic gate rejects
    the round. diag=0.1 is the C_K lower bound, so clipping cannot rescue it."""
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=0.1, target=1.0)
    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        step_fractions=(1.0, 0.5))
    assert res.step_fraction == pytest.approx(1.0)   # largest fallback (not 0.5, not None)
    assert not bool(res.bias.improved)               # → the monotonic gate rejects


def test_iteration_line_search_takes_full_step_when_it_improves():
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=1.0, target=1.0)  # full lands on target
    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        step_fractions=(1.0, 0.5, 0.25))
    assert res.step_fraction == pytest.approx(1.0)   # largest improving step
    assert bool(res.bias.improved)


def test_iteration_line_search_none_improve_reports_full_step():
    # Diagnosis moves AWAY from target at every fraction → no improvement; the
    # largest configured step is reported with improved=False (gate then rejects).
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=5.0, target=0.0)
    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        step_fractions=(1.0, 0.5, 0.25))
    assert res.step_fraction == pytest.approx(1.0)
    assert not bool(res.bias.improved)
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    assert ck[0] == pytest.approx(5.0)               # full-step value reported


def test_campaign_line_search_rescues_round_the_gate_would_reject():
    # Composition: with the gate ON, the full-step round WORSENS and is rejected;
    # adding step_fractions lets the line search find the improving half step, so
    # the round is ACCEPTED at fraction 0.5 — strictly more progress than the gate alone.
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=2.0, target=1.0)
    rescued = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True, step_fractions=(1.0, 0.5))
    assert rescued.accepted == (True,)
    assert rescued.iterations[0].step_fraction == pytest.approx(0.5)
    assert np.asarray(rescued.final_config.C_K).reshape(-1)[0] == pytest.approx(1.2)

    # Same round, full step only (step_fractions=None) → rejected, no progress.
    bare = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True)
    assert bare.accepted == (False,)
    assert jnp.ndim(jnp.asarray(bare.final_config.C_K)) == 0   # unchanged default


def test_iteration_clip_to_bounds_clamps_unphysical_diagnosis():
    # An out-of-range LES diagnosis (5.0 >> C_K's hi 1.2) is clamped to 1.2 before
    # injection when clip_to_bounds=True; the injected config + stored field agree.
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=5.0, target=1.2)
    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        clip_to_bounds=True)
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    assert ck[0] == pytest.approx(1.2)                       # clamped from 5.0
    assert np.asarray(res.feedback_field).reshape(-1)[0] == pytest.approx(1.2)
    assert bool(res.bias.improved)                           # 1.2 lands on target

    # Without the clamp (default), the raw unphysical 5.0 is injected.
    res2 = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck)
    assert np.asarray(res2.updated_config.C_K).reshape(-1)[0] == pytest.approx(5.0)


def test_iteration_clip_to_bounds_keeps_line_search_in_range():
    # clip clamps the raw diagnosis BEFORE the line search, so every blended
    # sub-step stays within (0.1, 1.2): a 5.0 diagnosis clamped to 1.2, half step
    # lands at 0.4 + 0.5*(1.2-0.4) = 0.8 (in range), not 0.4 + 0.5*(5.0-0.4).
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=5.0, target=0.8)
    res = run_correction_iteration(
        CLUBBLiteConfig(), compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2), background=dck,
        clip_to_bounds=True, step_fractions=(1.0, 0.5))
    assert res.step_fraction == pytest.approx(0.5)
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    assert ck[0] == pytest.approx(0.8)                       # 0.4 + 0.5*(1.2-0.4)
    assert 0.1 <= ck[0] <= 1.2


def test_iteration_clip_to_bounds_clamps_out_of_range_background():
    # The post-blend clamp is the invariant: even when the BASE (background) is
    # itself out of range (e.g. the guard toggled on mid-campaign, or an
    # out-of-range initial_field), the injected field can never leave the bounds.
    compare_fn, diagnose = _line_search_setup(diag_value=1.0, target=1.0)
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=jnp.full((4,), 5.0)),       # out-of-range base config
        compare_fn=compare_fn, diagnose_fn=diagnose,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=jnp.full((2, 2), 5.0),               # out-of-range accumulated base
        clip_to_bounds=True, step_fractions=(0.5,))
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    assert float(ck.max()) <= 1.2                        # blend clamped despite base=5.0
    assert float(ck.min()) >= 0.1


def test_campaign_clip_to_bounds_never_injects_unphysical():
    # Across rounds, the accumulated field stays within the registered C_K bounds.
    dck = float(CLUBBLiteConfig().C_K)
    compare_fn, diagnose = _line_search_setup(diag_value=9.0, target=1.2)
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=1, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, clip_to_bounds=True, accept_only_if_improved=True)
    ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    assert float(ck.min()) >= 0.1
    assert float(ck.max()) <= 1.2


def test_iteration_noop_round_clamps_field_under_clip_to_bounds():
    # Even a no-op round (no flagged columns) keeps the bounds invariant when
    # clip_to_bounds=True, so an out-of-range carried background is not propagated
    # unclamped into the campaign's next-round base.
    def compare_fn(config):
        return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)), model_ctx=None)
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=jnp.full((4,), 5.0)),
        compare_fn=compare_fn, diagnose_fn=lambda r, c: None,
        promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=jnp.full((2, 2), 5.0), clip_to_bounds=True)
    assert res.n_corrected == 0
    ck = np.asarray(res.feedback_field).reshape(-1)
    assert float(ck.max()) <= 1.2 and float(ck.min()) >= 0.1   # clamped no-op field


def test_run_line_search_helper_picks_largest_improving():
    # Direct test of the factored line-search helper (shared by single + future
    # multi-coefficient iterations): largest-improving step, k==0 fallback.
    from legoesm.training.correction_loop import _run_line_search

    class _Base:
        combined_score = jnp.full((2, 2), 1.0)
        area_weights = jnp.ones((2, 2))
        valid_mask = None

    # make_candidate(frac) -> (config, field); compare_fn(config) lowers the score
    # only for frac <= 0.5 (the full step 1.0 "overshoots" and does not improve).
    def make_candidate(frac):
        return {"frac": frac}, jnp.asarray(frac)

    def compare_fn(cfg):
        improved = cfg["frac"] <= 0.5
        score = jnp.full((2, 2), 0.5 if improved else 1.0)
        return CompareResult(score, [], jnp.ones((2, 2)), model_ctx=None)

    frac, field, cfg, upd, imp = _run_line_search(
        compare_fn, _Base(), (1.0, 0.5, 0.25), make_candidate)
    assert frac == pytest.approx(0.5)            # first improving (largest) step
    assert bool(imp.improved)
    assert float(field) == pytest.approx(0.5)

    # None improve → the k==0 (largest) fallback is returned, improved=False.
    def compare_none(cfg):
        return CompareResult(jnp.full((2, 2), 2.0), [], jnp.ones((2, 2)), model_ctx=None)

    frac2, _, _, _, imp2 = _run_line_search(
        compare_none, _Base(), (1.0, 0.5), make_candidate)
    assert frac2 == pytest.approx(1.0) and not bool(imp2.improved)


# --- simultaneous multi-coefficient correction (C_K + Pr_t) ---------------
class _CkDiag(NamedTuple):
    C_K: jax.Array
    valid: jax.Array


class _PrtDiag(NamedTuple):
    Pr_t: jax.Array
    valid: jax.Array


def _multi_setup(diag_ck, diag_prt, target_ck, target_prt):
    """2x2 grid; combined_score = |C_K - tC| + |Pr_t - tP| per column. One worst
    column flagged; the multi diagnose_fn returns {method: diagnosis} per column."""
    nlat, nlon = 2, 2
    area_w = jnp.ones((nlat, nlon))

    def compare_fn(config):
        ck = jnp.asarray(config.C_K)
        prt = jnp.asarray(config.Pr_t)
        ck2d = jnp.broadcast_to(ck, (nlat, nlon)) if ck.ndim == 0 else ck.reshape((nlat, nlon))
        prt2d = jnp.broadcast_to(prt, (nlat, nlon)) if prt.ndim == 0 else prt.reshape((nlat, nlon))
        score = jnp.abs(ck2d - target_ck) + jnp.abs(prt2d - target_prt)
        flat = np.asarray(score).reshape(-1)
        worst = [int(i) for i in np.argsort(-flat)[:1] if flat[i] > 1e-9]
        manifest = [_Rec(flat_index=i, lat_deg=0.0, environment=_Env(0.0)) for i in worst]
        return CompareResult(score, manifest, area_w, model_ctx=None)

    def diagnose_fn(record, ctx):
        return {
            "clubb_coefficient": _CkDiag(jnp.array([diag_ck]), jnp.array([True])),
            "prandtl_number": _PrtDiag(jnp.array([diag_prt]), jnp.array([True])),
        }

    return compare_fn, diagnose_fn


_SPECS = (
    CorrectionSpec("clubb_lite_C_K", "clubb_coefficient", float(CLUBBLiteConfig().C_K)),
    CorrectionSpec("clubb_lite_Pr_t", "prandtl_number", float(CLUBBLiteConfig().Pr_t)),
)


def test_multi_iteration_corrects_both_coefficients():
    # diag == targets ⇒ the full step corrects BOTH C_K and Pr_t at the worst column.
    compare_fn, diagnose_fn = _multi_setup(
        diag_ck=1.0, diag_prt=0.8, target_ck=1.0, target_prt=0.8)
    res = run_multi_correction_iteration(
        CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2))
    assert isinstance(res, MultiCorrectionResult)
    assert set(res.feedback_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}
    assert bool(res.bias.improved)
    ck = np.asarray(res.updated_config.C_K).reshape(-1)
    prt = np.asarray(res.updated_config.Pr_t).reshape(-1)
    assert ck[0] == pytest.approx(1.0)               # worst col C_K corrected
    assert prt[0] == pytest.approx(0.8)              # worst col Pr_t corrected
    assert res.n_corrected == 1


def test_multi_iteration_one_les_run_per_column():
    # The (multi) diagnose_fn is called ONCE per worst column (it returns BOTH
    # coefficients from one LES run); the C_K + Pr_t share that single call.
    compare_fn, _ = _multi_setup(1.0, 0.8, 1.0, 0.8)
    calls = {"n": 0}

    def diagnose_fn(record, ctx):
        calls["n"] += 1
        return {
            "clubb_coefficient": _CkDiag(jnp.array([1.0]), jnp.array([True])),
            "prandtl_number": _PrtDiag(jnp.array([0.8]), jnp.array([True])),
        }

    run_multi_correction_iteration(
        CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2))
    assert calls["n"] == 1                            # one LES run, both diagnoses


def test_multi_iteration_validates_specs():
    compare_fn, diagnose_fn = _multi_setup(1.0, 0.8, 1.0, 0.8)
    with pytest.raises(ValueError, match="non-empty"):
        run_multi_correction_iteration(
            CLUBBLiteConfig(), (), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
            grid_shape=(2, 2))
    dup = (_SPECS[0], _SPECS[0])
    with pytest.raises(ValueError, match="distinct promotion_keys"):
        run_multi_correction_iteration(
            CLUBBLiteConfig(), dup, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
            grid_shape=(2, 2))


def test_multi_iteration_no_records_is_noop():
    def compare_fn(config):
        return CompareResult(jnp.zeros((2, 2)), [], jnp.ones((2, 2)), model_ctx=None)
    res = run_multi_correction_iteration(
        CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn,
        diagnose_fn=lambda r, c: None, grid_shape=(2, 2))
    assert res.n_corrected == 0 and res.step_fraction == 0.0
    assert set(res.feedback_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}


def test_multi_campaign_accumulates_and_gate_reverts_atomically():
    # Round 0: worst col 0 (both coeffs off target) -> corrected to target -> improves.
    # Round 1: worst col is now elsewhere; diag stays on-target so it keeps improving.
    compare_fn, diagnose_fn = _multi_setup(
        diag_ck=1.0, diag_prt=0.8, target_ck=1.0, target_prt=0.8)
    campaign = run_multi_correction_campaign(
        CLUBBLiteConfig(), 2, _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2), accept_only_if_improved=True)
    assert all(campaign.accepted)
    ck = np.asarray(campaign.final_config.C_K).reshape(-1)
    prt = np.asarray(campaign.final_config.Pr_t).reshape(-1)
    # both coefficients accumulated across rounds (>=2 columns moved toward target).
    assert np.sum(np.isclose(ck, 1.0)) >= 2
    assert np.sum(np.isclose(prt, 0.8)) >= 2
    assert set(campaign.final_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}


def test_multi_campaign_gate_rejects_worsening_round_both_revert():
    # diag moves AWAY from target ⇒ the combined bias worsens ⇒ the WHOLE round is
    # rejected and BOTH coefficients revert (no partial accumulation).
    compare_fn, diagnose_fn = _multi_setup(
        diag_ck=5.0, diag_prt=5.0, target_ck=1.0, target_prt=0.8)
    campaign = run_multi_correction_campaign(
        CLUBBLiteConfig(), 1, _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2), accept_only_if_improved=True)
    assert campaign.accepted == (False,)
    # both config fields stay the scalar defaults (atomic revert).
    assert jnp.ndim(jnp.asarray(campaign.final_config.C_K)) == 0
    assert jnp.ndim(jnp.asarray(campaign.final_config.Pr_t)) == 0


def test_multi_iteration_requires_dict_diagnoses():
    # A multi diagnose_fn that returns a single diagnosis (not a {method: ...} dict)
    # — or one missing a spec's method — fails clearly, not with a bare KeyError.
    compare_fn, _ = _multi_setup(1.0, 0.8, 1.0, 0.8)
    with pytest.raises(TypeError, match="dict per column"):
        run_multi_correction_iteration(
            CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn,
            diagnose_fn=lambda r, c: _CkDiag(jnp.array([1.0]), jnp.array([True])),
            grid_shape=(2, 2))

    def missing_method(record, ctx):
        return {"clubb_coefficient": _CkDiag(jnp.array([1.0]), jnp.array([True]))}

    with pytest.raises(ValueError, match="missing the method"):
        run_multi_correction_iteration(
            CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn,
            diagnose_fn=missing_method, grid_shape=(2, 2))


def _help_hurt_setup():
    """compare_fn where applying C_K LOWERS the bias (helps) but applying Pr_t
    RAISES it more (hurts). Combined mode rejects the whole round; sequential
    keeps C_K and rejects Pr_t."""
    area_w = jnp.ones((2, 2))

    def compare_fn(config):
        ck_applied = jnp.asarray(config.C_K).ndim > 0
        prt_applied = jnp.asarray(config.Pr_t).ndim > 0
        bias = 10.0 - (3.0 if ck_applied else 0.0) + (5.0 if prt_applied else 0.0)
        return CompareResult(jnp.full((2, 2), bias),
                             [_Rec(flat_index=0, lat_deg=0.0, environment=_Env(0.0))],
                             area_w, model_ctx=None)

    def diagnose_fn(record, ctx):
        return {"clubb_coefficient": _CkDiag(jnp.array([0.9]), jnp.array([True])),
                "prandtl_number": _PrtDiag(jnp.array([0.9]), jnp.array([True]))}

    return compare_fn, diagnose_fn


def test_multi_iteration_combined_rejects_help_plus_hurt():
    compare_fn, diagnose_fn = _help_hurt_setup()
    res = run_multi_correction_iteration(
        CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2))                                   # combined (default)
    assert not bool(res.bias.improved)                      # whole round not improving
    assert res.step_fractions_by_key is None


def test_multi_iteration_sequential_keeps_helping_rejects_hurting():
    compare_fn, diagnose_fn = _help_hurt_setup()
    res = run_multi_correction_iteration(
        CLUBBLiteConfig(), _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2), sequential=True)
    assert bool(res.bias.improved)                          # C_K's help is KEPT
    assert jnp.ndim(jnp.asarray(res.updated_config.C_K)) == 1    # C_K corrected
    assert jnp.ndim(jnp.asarray(res.updated_config.Pr_t)) == 0   # Pr_t rejected (scalar)
    assert res.step_fractions_by_key["clubb_lite_C_K"] > 0.0
    assert res.step_fractions_by_key["clubb_lite_Pr_t"] == 0.0
    # the rejected coefficient's stored field is its (scalar-broadcast) base.
    assert res.feedback_fields["clubb_lite_Pr_t"].shape == (2, 2)


def test_multi_campaign_sequential_accumulates():
    # A sequential campaign where both coefficients help → both accumulate.
    compare_fn, diagnose_fn = _multi_setup(
        diag_ck=1.0, diag_prt=0.8, target_ck=1.0, target_prt=0.8)
    campaign = run_multi_correction_campaign(
        CLUBBLiteConfig(), 1, _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2), accept_only_if_improved=True, sequential=True)
    assert campaign.accepted == (True,)
    it = campaign.iterations[0]
    assert it.step_fractions_by_key is not None
    assert set(campaign.final_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}


def test_campaign_early_stops_on_convergence():
    # A round that can never improve (mock worsens) is rejected ⇒ no progress;
    # after `patience` such rounds the campaign STOPS early instead of running all
    # n_iterations ("converged" once the bias has plateaued).
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 5.0)   # worsens → reject
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=10, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True, bias_tol=1e-9, patience=2)
    assert campaign.stop_reason == "converged"
    assert len(campaign.iterations) == 2          # 2 no-progress rounds → stop early


def test_campaign_no_bias_tol_runs_all_rounds():
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 5.0)
    campaign = run_correction_campaign(
        CLUBBLiteConfig(), n_iterations=3, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
        background=dck, accept_only_if_improved=True)   # bias_tol=None
    assert campaign.stop_reason == "max_iterations"
    assert len(campaign.iterations) == 3


def test_multi_campaign_early_stops_on_convergence():
    compare_fn, diagnose_fn = _help_hurt_setup()        # combined can't improve → reject
    campaign = run_multi_correction_campaign(
        CLUBBLiteConfig(), 8, _SPECS, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(2, 2), accept_only_if_improved=True, bias_tol=1e-9, patience=2)
    assert campaign.stop_reason == "converged"
    assert len(campaign.iterations) == 2


def test_multi_campaign_aborts_on_dry_les_diagnoses():
    """The dry-LES early-abort works for the SIMULTANEOUS multi-coefficient campaign
    too: off-target so a column is always flagged, but BOTH coefficient diagnoses are
    invalid every round ⇒ stop_reason='no_valid_diagnoses' after `patience` (iter 101)."""
    compare_fn, _ = _multi_setup(
        diag_ck=1.0, diag_prt=0.8, target_ck=5.0, target_prt=2.0)  # off-target → flagged

    def diag_invalid(record, ctx):                       # noqa: ARG001
        return {"clubb_coefficient": _CkDiag(jnp.array([1.0]), jnp.array([False])),
                "prandtl_number": _PrtDiag(jnp.array([0.8]), jnp.array([False]))}

    campaign = run_multi_correction_campaign(
        CLUBBLiteConfig(), 5, _SPECS, compare_fn=compare_fn, diagnose_fn=diag_invalid,
        grid_shape=(2, 2), patience=2)
    assert campaign.stop_reason == "no_valid_diagnoses"
    assert len(campaign.iterations) == 2
    assert all(it.n_diagnoses_valid == 0 for it in campaign.iterations)


def test_campaign_real_compare_multi_round_converges_to_zero_bias():
    """PERFECT-MODEL OSSE for the LOOP mechanics: a multi-round campaign over the
    REAL compare_state_to_reference drives the model's per-column coefficient so the
    bias DECREASES MONOTONICALLY to ~0 and the campaign converges + early-stops.
    Only the LES is replaced by a perfect diagnose_fn (returns the fixing value);
    every other stage (real compare, ranking, apply, gate, accumulation) is real —
    the strongest in-environment demonstration that updating the parameters lowers
    the bias (the LES↔GCM transfer is the separate, HPC-only empirical question)."""
    nlat, nlon, nlev = 4, 4, 5
    shape = (nlat, nlon, nlev)
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[1:] + sigma_half[:-1])
    lat = jnp.linspace(-60.0, 60.0, nlat)
    lon = jnp.linspace(0.0, 270.0, nlon)
    area_w = jnp.cos(jnp.deg2rad(lat))[:, None] * jnp.ones((nlat, nlon))
    reference = ColumnState(
        T=jnp.full(shape, 250.0), q_v=jnp.full(shape, 1e-3),
        u=jnp.zeros(shape), v=jnp.zeros(shape), p_s=jnp.full((nlat, nlon), 1e5))

    # +8 K bias at 8 columns; a column is "fixed" (zero bias) once its per-column
    # tau_equator differs from the background — so accumulating corrections over
    # rounds removes the bias column-by-column.
    biased = {0, 2, 5, 7, 8, 10, 13, 15}
    bias_vec = np.array([8.0 if c in biased else 0.0 for c in range(16)])
    bg_tau = float(GrayRadiationConfig().tau_equator)

    def run_amip_fn(config):
        tau = np.asarray(config.tau_equator)
        if tau.ndim == 0:
            b = bias_vec                                 # round 0: nothing corrected
        else:
            fixed = ~np.isclose(tau, bg_tau)             # corrected columns → bias 0
            b = np.where(fixed, 0.0, bias_vec)
        return reference._replace(
            T=reference.T + jnp.asarray(b).reshape(nlat, nlon)[:, :, None])

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat, lon_deg=lon, area_weights=area_w, n_worst=4,
        run_amip_fn=run_amip_fn)

    def diagnose(record, model_ctx):                     # perfect: fix the column
        return _Eddy(K=jnp.array([2.0 * bg_tau]), valid=jnp.array([True]))

    campaign = run_correction_campaign(
        GrayRadiationConfig(), n_iterations=12, compare_fn=compare_fn,
        diagnose_fn=diagnose, promotion_key="gray_tau_equator", grid_shape=(nlat, nlon),
        background=bg_tau, accept_only_if_improved=True, bias_tol=1e-6, patience=2)

    # Monotonic decrease across the ACCEPTED rounds, converging to ~0.
    accepted_biases = [float(it.bias.updated_bias)
                       for it, ok in zip(campaign.iterations, campaign.accepted) if ok]
    assert accepted_biases == sorted(accepted_biases, reverse=True)  # monotone ↓
    assert float(campaign.iterations[-1].bias.updated_bias) == pytest.approx(0.0, abs=1e-9)
    # It CONVERGED + stopped early (well before the 12-round budget).
    assert campaign.stop_reason == "converged"
    assert len(campaign.iterations) < 12


def test_campaign_patience_must_be_positive_with_bias_tol():
    compare_fn, diagnose, dck = _abs_target_setup(lambda _i: 1.0)
    with pytest.raises(ValueError, match="patience must be >= 1"):
        run_correction_campaign(
            CLUBBLiteConfig(), n_iterations=3, compare_fn=compare_fn,
            diagnose_fn=diagnose, promotion_key="clubb_lite_C_K", grid_shape=(2, 2),
            background=dck, bias_tol=1e-9, patience=0)


def test_clubb_field_for_promotion_key_resolves_clubb_only():
    """_clubb_field_for_promotion_key resolves a clubb_lite_* key to its field but returns
    None for a REGISTERED NON-CLUBB key (gray radiation) AND an unregistered key -- the
    cross-resolution env-kernel is a CLUBB deploy artifact, so a non-CLUBB campaign must
    NOT fabricate one. Resolves via the PROMOTABLE_FIELDS registry, not string-stripping
    (which would wrongly fabricate 'tau_equator' for the gray key)."""
    from legoesm.training.correction_loop import _clubb_field_for_promotion_key

    assert _clubb_field_for_promotion_key("clubb_lite_C_K") == "C_K"
    assert _clubb_field_for_promotion_key("clubb_lite_C_eps") == "C_eps"
    assert _clubb_field_for_promotion_key("gray_tau_equator") is None    # registered, non-CLUBB
    assert _clubb_field_for_promotion_key("nonexistent_key") is None     # unregistered


def test_make_compare_fn_coordinate_activates_hybrid_mass_weighting():
    """make_compare_fn(coordinate=<hybrid>) weights the bias by the TRUE hybrid layer mass
    (computed per-run from model.p_s), vs the pure-sigma default — so for a terrain column
    (p_s != p_ref) the combined score DIFFERS (iter 338: the campaign-side activation of the
    iter-337 hybrid-coordinate fix).  coordinate=None reproduces the legacy pure-sigma
    weighting; the difference here is the bug that was silently corrupting elevated columns."""
    from legoesm.grids.vertical import make_hybrid_levels

    nlev = 6
    hc = make_hybrid_levels(nlev, p_top_Pa=100.0)
    shape = (1, 1)

    def _state():
        return ColumnState(
            T=jnp.full(shape + (nlev,), 250.0), q_v=jnp.full(shape + (nlev,), 1e-3),
            u=jnp.full(shape + (nlev,), 5.0), v=jnp.zeros(shape + (nlev,)),
            p_s=jnp.full(shape, 70000.0))                # 700 hPa terrain (p_s != p_ref ~1e5)

    reference = _state()
    model = reference._replace(T=reference.T.at[0, 0, 0].set(260.0))   # 10 K bias, TOP layer

    def _run_fn(_config):
        return model

    common = dict(
        reference=reference, sigma_full=jnp.asarray(hc.sigma_full),
        sigma_half=jnp.asarray(hc.sigma_half), lat_deg=jnp.array([0.0]),
        lon_deg=jnp.array([0.0]), area_weights=jnp.ones(shape), n_worst=1, run_amip_fn=_run_fn)

    score_sigma = float(jnp.max(
        make_compare_fn(**common, coordinate=None)(None).combined_score))
    score_hybrid = float(jnp.max(
        make_compare_fn(**common, coordinate=hc)(None).combined_score))
    assert abs(score_sigma - score_hybrid) > 1e-6     # hybrid re-weights the terrain column
