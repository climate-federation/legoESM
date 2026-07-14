"""Smoke test for the LES-informed correction CAMPAIGN driver composition.

Exercises the genuinely-new orchestration (``make_clubb_build_driver`` +
``make_les_diagnose_fn`` + ``build_correction_campaign``) WITHOUT a real model
run: a mock driver supplies the column state and a mock LES supplies the plane
state, so the REAL process_column forcing-extract + diagnose path runs (the real
model run + real LES are covered by iter 35/37 and iter 20 respectively).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402

from scripts.run.run_correction_campaign import (  # noqa: E402
    CampaignDryRun,
    _area_weights,
    _assert_output_path_writable,
    _atomic_write_json,
    _build_arg_parser,
    _campaign_knobs_from_args,
    _capture_initial_record,
    _distributed_campaign_kwargs,
    _dry_run_report,
    _env_kernel_export_note,
    _format_per_variable_bias,
    _json_finite,
    _per_variable_bias_dict,
    _per_variable_bias_from_dict,
    _per_variable_to_json,
    _realism_campaign_summary_line,
    _realism_summary_dict,
    _RealismCapture,
    _resolve_era5_n_times,
    _summary_to_json,
    build_campaign_output_dict,
    build_correction_campaign,
    build_distributed_multi_correction_campaign,
    build_multi_correction_campaign,
    compose_compare_fn,
    fast_validation_les_regime,
    grid_latlon_deg,
    load_base_config_and_grid,
    make_base_driver_builder,
    make_clubb_build_driver,
    make_les_diagnose_fn,
    maybe_env_grid_fn,
    print_realism_summary,
    realism_summary,
    reduce_realism_summary_mpi,
    refuse_unsupported_multirank,
    resolve_orographic_phis,
)
from tests._offline_era5_rda import write_full_archive  # noqa: E402


def test_env_kernel_export_note():
    """The env-kernel export note: WARN only when --feedback-strategy environment
    produced NO kernel; silent (None) when a kernel exists or for static/non-CLUBB."""
    # Kernel produced ⇒ no note (it gets written).
    assert _env_kernel_export_note("environment", has_kernel=True) is None
    # static campaign with no kernel ⇒ silent (correctly has nothing to transfer).
    assert _env_kernel_export_note("static", has_kernel=False) is None
    # environment-strategy with NO kernel ⇒ a WARNING explaining the absent artifact.
    note = _env_kernel_export_note("environment", has_kernel=False)
    assert note is not None
    assert "feedback-strategy environment" in note
    assert "no <out>.env_kernel.json" in note.lower() or "env_kernel.json" in note


def test_per_variable_bias_dict_roundtrip():
    """The raw-PerVariableBias (campaign-start baseline) checkpoint serializer round-trips
    losslessly, with NaN precip <-> JSON null (preserving precip-not-compared)."""
    import json

    from legoesm.training.bias_metrics import PerVariableBias

    assert _per_variable_bias_dict(None) is None
    assert _per_variable_bias_from_dict(None) is None
    pv = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                         jnp.asarray(float("nan")))
    d = _per_variable_bias_dict(pv)
    assert d["T_rmse_K"] == pytest.approx(4.0)
    assert d["precip_err_mm_day"] is None              # NaN -> null
    json.loads(json.dumps(d, allow_nan=False))         # valid strict JSON
    back = _per_variable_bias_from_dict(d)
    assert float(back.global_T_rmse_K) == pytest.approx(4.0)
    assert float(back.global_wind_rmse_m_s) == pytest.approx(2.0)
    assert bool(jnp.isnan(back.global_precip_err_mm_day))   # null -> NaN


def test_per_variable_bias_dict_sanitizes_inf_not_just_nan():
    """A blown-up / OVERFLOWED model can produce a ±inf RMSE (overflow precedes NaN), so
    the serializer must map ±inf to ``null`` too — a NaN-ONLY guard would leak a
    non-standard ``Infinity`` token into the output FILE, breaking the plotters / deploy
    reader's ``json.load`` (iter 246; same class as the iter-245 trajectory bug)."""
    import json

    from legoesm.training.bias_metrics import PerVariableBias

    pv = PerVariableBias(jnp.asarray(float("inf")), jnp.asarray(1e-3),
                         jnp.asarray(float("-inf")), jnp.asarray(5.0))
    d = _per_variable_bias_dict(pv)
    assert d["T_rmse_K"] is None and d["wind_rmse_m_s"] is None   # ±inf -> null
    assert d["qv_rmse_kg_kg"] == pytest.approx(1e-3)              # a finite RMSE survives
    json.loads(json.dumps(d), parse_constant=_reject_nonstandard)  # strict JSON, no Infinity token


def test_capture_initial_record():
    """_capture_initial_record: the FIRST fresh round captures the campaign-start
    baseline (combined + per-variable); later/resumed rounds PRESERVE it; a non-finite
    baseline is NOT stored (it must not poison every future resume's reported start)."""
    from types import SimpleNamespace

    from legoesm.training.bias_metrics import (
        BiasImprovement,
        PerVariableBias,
        compare_per_variable_bias,
    )

    def _res(base, t_base, n_diag=2, n_valid=1):
        bias = BiasImprovement(
            baseline_bias=jnp.asarray(base), updated_bias=jnp.asarray(base * 0.9),
            absolute_reduction=jnp.asarray(0.0), fractional_improvement=jnp.asarray(0.0),
            improved=jnp.asarray(True))
        pvb = compare_per_variable_bias(
            PerVariableBias(jnp.asarray(t_base), jnp.asarray(1e-3), jnp.asarray(2.0),
                            jnp.asarray(float("nan"))),
            PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                            jnp.asarray(float("nan"))))
        return SimpleNamespace(bias=bias, per_variable_bias=pvb,
                               n_diagnosed=n_diag, n_diagnoses_valid=n_valid)

    box = {"initial_bias": None, "initial_per_variable": None}
    _capture_initial_record(box, _res(5.0, 8.0))       # first fresh round captures
    assert box["initial_bias"] == pytest.approx(5.0)
    assert box["initial_per_variable"]["T_rmse_K"] == pytest.approx(8.0)
    assert box["n_diag_seg"] == 2 and box["n_valid_seg"] == 1   # counts accumulate
    _capture_initial_record(box, _res(0.6, 3.0, n_diag=3, n_valid=2))  # later round
    assert box["initial_bias"] == pytest.approx(5.0)            # original PRESERVED
    assert box["initial_per_variable"]["T_rmse_K"] == pytest.approx(8.0)
    assert box["n_diag_seg"] == 5 and box["n_valid_seg"] == 3   # SEGMENT running sum
    # A non-finite baseline is NOT stored (the next round retries) — never poison resumes.
    box2 = {"initial_bias": None, "initial_per_variable": None}
    _capture_initial_record(box2, _res(float("nan"), 8.0))
    assert box2["initial_bias"] is None


def test_resume_accumulates_diagnosis_counts_across_segments():
    """The cumulative-trajectory invariant across a SLURM-resume boundary (iters 137/138):
    _capture_initial_record accumulates each SEGMENT's diagnosis counts, the checkpoint
    persists prior+seg as the running total, and _load_single_resume restores it as the
    NEXT segment's prior — so the final reported count is the TRUE start→end sum, not just
    the last resumed segment. Locks the COMPOSITION (capture starts each segment fresh; the
    resume seed carries the prior) that the per-helper tests don't cover together."""
    from types import SimpleNamespace

    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.bias_metrics import BiasImprovement

    import scripts.run.run_correction_campaign as rcc

    def _res(n_diag, n_valid):
        bias = BiasImprovement(
            baseline_bias=jnp.asarray(1.0), updated_bias=jnp.asarray(0.9),
            absolute_reduction=jnp.asarray(0.1), fractional_improvement=jnp.asarray(0.1),
            improved=jnp.asarray(True))
        return SimpleNamespace(bias=bias, per_variable_bias=None,
                               n_diagnosed=n_diag, n_diagnoses_valid=n_valid)

    # Segment 1 (fresh): two rounds diagnose 2 then 3 (valid 1, 2).
    box = {"initial_bias": None, "initial_per_variable": None,
           "n_diagnosed_prior": 0, "n_diagnoses_valid_prior": 0}
    _capture_initial_record(box, _res(2, 1))
    _capture_initial_record(box, _res(3, 2))
    total1 = box["n_diagnosed_prior"] + box["n_diag_seg"]          # the checkpoint arithmetic
    valid1 = box["n_diagnoses_valid_prior"] + box["n_valid_seg"]
    assert total1 == 5 and valid1 == 3

    # Resume from a checkpoint carrying those running totals.
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    ckpt = {"round": 0, "grid": rcc._grid_provenance(_base_config(), grid),
            "corrected_field": "C_K",
            "field": np.asarray(jnp.full((8, 16), 0.4)).tolist(),
            "n_diagnosed_total": total1, "n_diagnoses_valid_total": valid1}
    _f, _c, _sr, seed = rcc._load_single_resume(ckpt, grid, "C_K")
    box2 = {"initial_bias": None, "initial_per_variable": None,
            "n_diagnosed_prior": 0, "n_diagnoses_valid_prior": 0}
    box2.update(seed)
    assert box2["n_diagnosed_prior"] == 5    # restored as the NEXT segment's prior

    # Segment 2: one round diagnoses 4 (valid 4). Cumulative = 2+3+4 = 9 (valid 1+2+4 = 7).
    _capture_initial_record(box2, _res(4, 4))
    total2 = box2["n_diagnosed_prior"] + box2["n_diag_seg"]
    valid2 = box2["n_diagnoses_valid_prior"] + box2["n_valid_seg"]
    assert total2 == 9 and valid2 == 7       # NOT reset to the last segment, NOT double-counted


def test_resume_field_roundtrips_bit_exactly_with_per_column_identity(tmp_path):
    """A NON-uniform corrected field must survive the checkpoint round-trip with
    EXACT values AND per-column identity preserved (the checkpoint analog of the
    iter-36 feedback↔physics column-ordering contract).

    The existing resume test uses a UNIFORM 0.4 field, which reshapes identically
    under any axis order and so cannot catch a flat↔2-D ordering bug.  Here every
    cell carries a UNIQUE value, so if the writer's ``np.asarray(field).tolist()``
    (C-order) and the loader's ``reshape(grid_shape_2d).reshape(-1)`` ever disagree
    on flatten order (or JSON drops precision), the restored per-column C_K would
    mismatch the originally-accepted field and a resumed campaign would silently
    seed the WRONG column corrections.  Driven through the REAL on-disk path
    (``_atomic_write_json`` → ``json.load`` → ``_load_single_resume``).
    """
    import json

    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    nlat, nlon = grid.grid_shape_2d
    # Unique per-cell value: cell (i,j) = 0.4 + (i*nlon + j)*1e-3 — a transpose or
    # F-order flatten anywhere in the chain changes the per-column mapping.
    field2d = 0.4 + np.arange(nlat * nlon, dtype=np.float64).reshape(nlat, nlon) * 1e-3
    flat = field2d.reshape(-1)

    ckpt_path = tmp_path / "ckpt.json"
    rcc._atomic_write_json(str(ckpt_path), {
        "round": 3,
        "grid": rcc._grid_provenance(_base_config(), grid),
        "corrected_field": "C_K",
        "C_K": flat.tolist(),
        "field": field2d.tolist(),
        "n_diagnosed_total": 0, "n_diagnoses_valid_total": 0,
    }, indent=2)

    with open(ckpt_path) as f:
        ckpt = json.load(f)

    init_field, init_clubb, start_round, _seed = rcc._load_single_resume(ckpt, grid, "C_K")

    # 1) start_round advances past the checkpointed round.
    assert start_round == 4
    # 2) 2-D field restored bit-exactly with the right shape.
    assert tuple(np.asarray(init_field).shape) == (nlat, nlon)
    np.testing.assert_array_equal(np.asarray(init_field), field2d)
    # 3) The CLUBB C_K (flat, fed to the physics) preserves per-column identity:
    #    column flat-index k holds EXACTLY the accepted field's column k.
    np.testing.assert_array_equal(np.asarray(init_clubb.C_K), flat)
    assert np.asarray(init_clubb.C_K).shape == (nlat * nlon,)
    # The persisted flat copy ("C_K" key, for inspection/method-guard) and the 2-D
    # "field" the loader actually trusts describe the SAME ordering.
    np.testing.assert_array_equal(np.asarray(ckpt["C_K"]), flat)


def test_json_finite_maps_nonfinite_to_none():
    """_json_finite: finite floats pass through; NaN AND ±inf map to None (valid JSON
    null), None passes through. Non-vacuous: covers the ±inf fractional_reduction case
    (initial_bias==0) that math.isnan alone would have missed."""
    assert _json_finite(1.5) == pytest.approx(1.5)
    assert _json_finite(0.0) == 0.0
    assert _json_finite(float("nan")) is None
    assert _json_finite(float("inf")) is None
    assert _json_finite(float("-inf")) is None
    assert _json_finite(None) is None


def test_summary_to_json_diverged_biases_serialize_as_null():
    """A DIVERGED run (the iter-161 non_finite_bias case) has a NaN final_bias and a
    ±inf fractional_reduction; _summary_to_json must serialize them as JSON null (not a
    non-standard NaN/Infinity token) so the output file is STANDARD JSON and consistent
    with the per-variable convention. The finite initial_bias survives unchanged."""
    import json
    from types import SimpleNamespace

    # A blown-up run can also leave NaN per-coefficient field stats — those go
    # through the coefficients block, so they must be sanitized too (else the dict
    # is still non-standard JSON via that branch).
    coeff = SimpleNamespace(
        promotion_key="clubb_lite_C_K", n_columns=4,
        field_min=float("nan"), field_max=float("nan"),
        field_mean=float("nan"), field_std=float("nan"),
        n_at_lower_bound=0, n_at_upper_bound=0, bounds=(0.0, 1.0))
    summary = SimpleNamespace(
        n_rounds=2, n_accepted=0, acceptance_rate=0.0, stop_reason="max_iterations",
        initial_bias=5.0, final_bias=float("nan"),
        absolute_reduction=float("nan"), fractional_reduction=float("inf"),
        n_diagnosed_total=4, n_diagnoses_valid_total=2, per_variable=None,
        coefficients=(coeff,))
    out = _summary_to_json(summary)
    assert out["initial_bias"] == pytest.approx(5.0)         # finite survives
    assert out["final_bias"] is None                          # NaN -> null
    assert out["absolute_reduction"] is None
    assert out["fractional_reduction"] is None                # +inf -> null
    c0 = out["coefficients"][0]
    assert c0["field_min"] is None and c0["field_std"] is None  # coeff stats -> null
    assert c0["bounds"] == [0.0, 1.0]                         # finite metadata intact
    # The WHOLE summary block stays STANDARD JSON (every non-finite field sanitized to
    # null — no NaN/Infinity token leaks through any branch).
    json.loads(json.dumps(out, allow_nan=False))
    # The WHOLE dict is STANDARD JSON (no NaN/Infinity tokens) under the strict parser.
    reparsed = json.loads(json.dumps(out), parse_constant=_reject_nonstandard)
    assert reparsed["final_bias"] is None
    assert reparsed["coefficients"][0]["field_mean"] is None


def _reject_nonstandard(token):  # pragma: no cover - only fires on a regression
    raise AssertionError(f"non-standard JSON token {token!r} in summary output")


def test_per_variable_to_json_nan_precip_is_null():
    """The persisted per-variable JSON: None->None; a NaN precip (not compared)
    serializes as JSON null (valid JSON, round-trips), NEVER a misleading 0/NaN."""
    import json

    from legoesm.training.bias_metrics import (
        PerVariableBias,
        compare_per_variable_bias,
    )

    assert _per_variable_to_json(None) is None
    b = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                        jnp.asarray(float("nan")))
    u = PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(5.0),
                        jnp.asarray(float("nan")))
    d = _per_variable_to_json(compare_per_variable_bias(b, u))
    assert d["baseline"]["T_rmse_K"] == pytest.approx(4.0)
    assert d["final"]["wind_rmse_m_s"] == pytest.approx(5.0)
    assert d["baseline"]["precip_err_mm_day"] is None     # NaN -> null, not 0
    assert d["improved"] == {"T": True, "qv": False, "wind": False, "precip": False}
    json.loads(json.dumps(d, allow_nan=False))            # valid strict JSON (no NaN token)


def test_format_per_variable_bias():
    """The per-round per-variable line: None -> '' (mock round); a real
    PerVariableBiasImprovement -> baseline->updated per variable + which improved."""
    from legoesm.training.bias_metrics import (
        PerVariableBias,
        PerVariableBiasImprovement,
    )

    assert _format_per_variable_bias(None) == ""
    base = PerVariableBias(jnp.asarray(4.0), jnp.asarray(1e-3), jnp.asarray(2.0),
                           jnp.asarray(float("nan")))
    upd = PerVariableBias(jnp.asarray(1.0), jnp.asarray(1e-3), jnp.asarray(5.0),
                          jnp.asarray(float("nan")))
    pvb = PerVariableBiasImprovement(
        baseline=base, updated=upd,
        T_improved=jnp.asarray(True), qv_improved=jnp.asarray(False),
        wind_improved=jnp.asarray(False), precip_improved=jnp.asarray(False))
    line = _format_per_variable_bias(pvb)
    assert "per-var RMSE" in line
    assert "T 4->1K" in line and "wind 2->5m/s" in line
    assert "improved: T" in line and "wind" not in line.split("improved:")[1]


def test_format_resume_line_surfaces_checkpointed_progress():
    """On resume the console shows the invested work (cumulative LES count + start bias)
    from the checkpoint seed, so an operator restarting after a SLURM timeout sees real
    progress, not a cold start at round N (iter 274). A null start bias → 'unknown'."""
    from scripts.run.run_correction_campaign import _format_resume_line

    seed = {"initial_bias": 5.0, "n_diagnosed_prior": 42}
    line = _format_resume_line("ckpt.json", 7, seed)
    assert "resuming from ckpt.json" in line and "at round 7" in line
    assert "start bias 5" in line and "42 LES diagnoses" in line
    # multi flag + a null start bias (a diverged/uncaptured initial) → 'unknown', count 0.
    multi = _format_resume_line("c.json", 3, {"initial_bias": None}, multi=True)
    assert "resuming multi from c.json" in multi
    assert "start bias unknown" in multi and "0 LES diagnoses" in multi


def test_format_round_line_surfaces_les_validity():
    """The per-round console line distinguishes a no-turbulence LES round (0/N valid —
    a forcing/setup issue) from a correction-didn't-help round (N/N valid but no
    improvement — a science result); both would otherwise read as 'no improvement;
    REJECTED' (iter 266)."""
    from scripts.run.run_correction_campaign import _format_round_line

    no_turb = _format_round_line(3, 0.5, 0.5, False, False, 1.0, 0, 4)
    assert "round 3" in no_turb and "0/4 LES valid" in no_turb
    assert "no improvement" in no_turb and "REJECTED" in no_turb
    good = _format_round_line(2, 1.0, 0.4, True, True, 0.5, 3, 4)
    assert "3/4 LES valid" in good and "IMPROVED" in good and "kept" in good


def test_resolve_orographic_phis_off_skips_provider():
    """'off' returns None WITHOUT calling the provider (so no driver/probe is built
    — the legacy flat path constructs nothing)."""
    calls = []

    def provider():
        calls.append(1)
        return jnp.full((4, 8), 5000.0)

    assert resolve_orographic_phis("off", provider) is None
    assert calls == []                                   # provider NOT called for 'off'


def test_resolve_orographic_phis_auto_terrain_and_flat():
    """'auto' returns the model's terrain when present, and None (safe) when the
    model is flat (identically-zero phis) — passing 'auto' is always safe."""
    terrain = jnp.zeros((4, 8)).at[1, 1].set(3000.0)
    out = resolve_orographic_phis("auto", lambda: terrain)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(terrain))
    assert resolve_orographic_phis("auto", lambda: jnp.zeros((4, 8))) is None


def test_resolve_orographic_phis_on_requires_terrain():
    """'on' returns terrain when present but FAILS LOUD (SystemExit) on a flat model
    — an explicit terrain request must not silently run flat."""
    terrain = jnp.full((4, 8), 2000.0)
    np.testing.assert_array_equal(
        np.asarray(resolve_orographic_phis("on", lambda: terrain)),
        np.asarray(terrain))
    with pytest.raises(SystemExit, match="requires model topography"):
        resolve_orographic_phis("on", lambda: jnp.zeros((4, 8)))


def test_resolve_orographic_phis_unknown_mode_raises():
    """Dispatch hardening: an unknown mode raises ValueError (argparse choices=
    constrains the CLI, but the helper must not silently accept a typo)."""
    with pytest.raises(ValueError, match="unknown orographic_forcing mode"):
        resolve_orographic_phis("terrain", lambda: jnp.zeros((4, 8)))


def _dry_run_sigma():
    return SimpleNamespace(sigma_full=jnp.linspace(0.1, 0.9, 5),
                           sigma_half=jnp.linspace(0.0, 1.0, 6))


def test_build_correction_campaign_dry_run_constructs_without_running():
    """--dry-run pre-flight: build_correction_campaign(dry_run=True) ASSEMBLES the
    campaign (validating units/grid/scheme/method + the turbulence='clubb_lite' build
    driver) and returns a CampaignDryRun WITHOUT running the model/LES — every model-
    touching callback would raise if invoked, proving nothing ran. The launch check
    that catches a misconfig in ms instead of after a multi-day run."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

    def boom(*a, **k):
        raise AssertionError("dry_run must NOT run the model / LES")

    common = dict(
        build_base_driver=boom, extract_column_state=boom, run_les_fn=boom,
        reference=SimpleNamespace(T=SimpleNamespace(shape=(2, 3, 5))),
        sigma=_dry_run_sigma(), grid=object(), area_weights=jnp.ones((2, 3)),
        lat_deg=jnp.zeros((2, 3)), lon_deg=jnp.zeros((2, 3)),
        n_iterations=5, n_worst=4, validate_reference=False, dry_run=True)
    res = build_correction_campaign(
        base_atm_config=_base_config(),
        les_config=ColumnLESConfig(diagnosis_method="clubb_coefficient"), **common)
    assert isinstance(res, CampaignDryRun)
    assert tuple(res.grid_shape) == (2, 3) and res.n_worst == 4
    assert res.coefficients == ("C_K",)             # clubb_coefficient → clubb_lite C_K
    assert res.n_iterations == 5                     # compute estimate: rounds…
    assert res.les_per_round == 4                    # …× LES/round (no les_budget → n_worst)
    assert res.surface_flux is False                 # default LES config ⇒ surface-flux-free
    # The dry-run PROPAGATES the material surface-flux setting (iter 403) so the launch
    # pre-flight report can surface it before the multi-day run.
    res_sf = build_correction_campaign(
        base_atm_config=_base_config(),
        les_config=ColumnLESConfig(diagnosis_method="clubb_coefficient",
                                   surface_flux=True), **common)
    assert res_sf.surface_flux is True
    # dry_run STILL validates: a non-clubb base config fails LOUD (the pre-flight's point).
    with pytest.raises(ValueError, match="clubb_lite"):
        build_correction_campaign(
            base_atm_config=SimpleNamespace(turbulence="bulk"),
            les_config=ColumnLESConfig(diagnosis_method="clubb_coefficient"), **common)


def test_dry_run_rejects_per_column_fields_not_matching_grid():
    """Pre-flight (iter 351): the dry-run RAISES on a per-column field (``area_weights`` /
    ``valid_mask``) whose shape is not broadcastable to the model grid
    (``= reference.T.shape[:-1]``).  Such a mismatched-resolution field would otherwise
    crash only at the FIRST compare, after a (non-free) model run.  The guard uses the SAME
    ``np.broadcast_shapes`` rules as the runtime ``broadcast_to``, so a genuinely
    broadcastable shape is NEVER rejected (no false positive)."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

    def boom(*a, **k):
        raise AssertionError("dry_run must NOT run the model / LES")

    common = dict(
        build_base_driver=boom, extract_column_state=boom, run_les_fn=boom,
        reference=SimpleNamespace(T=SimpleNamespace(shape=(2, 3, 5))),   # grid_shape (2, 3)
        sigma=_dry_run_sigma(), grid=object(),
        lat_deg=jnp.zeros((2, 3)), lon_deg=jnp.zeros((2, 3)),
        n_iterations=5, n_worst=4, validate_reference=False, dry_run=True)
    les = ColumnLESConfig(diagnosis_method="clubb_coefficient")

    # (a) a clearly mismatched resolution ⇒ ValueError, BEFORE any model run (boom never fires).
    with pytest.raises(ValueError, match="area_weights.*not broadcastable to the model grid"):
        build_correction_campaign(base_atm_config=_base_config(), les_config=les,
                                  area_weights=jnp.ones((4, 5)), **common)
    # (b) a genuinely broadcastable shape ((2, 1) → (2, 3)) is NOT rejected (no false positive).
    res = build_correction_campaign(base_atm_config=_base_config(), les_config=les,
                                    area_weights=jnp.ones((2, 1)), **common)
    assert isinstance(res, CampaignDryRun)
    # (c) a mismatched valid_mask is also caught.
    with pytest.raises(ValueError, match="valid_mask.*not broadcastable to the model grid"):
        build_correction_campaign(base_atm_config=_base_config(), les_config=les,
                                  area_weights=jnp.ones((2, 3)),
                                  valid_mask=jnp.ones((7,), dtype=bool), **common)
    # (d) a mismatched lat/lon (1-D vectors with the WRONG lengths for the (2,3) grid) ⇒
    #     caught in the dry-run, before the manifest's lat_flat[flat_index] gather mis-indexes.
    common_d = {**common, "lat_deg": jnp.zeros((4,)), "lon_deg": jnp.zeros((5,))}
    with pytest.raises(ValueError, match="not broadcastable to the column grid"):
        build_correction_campaign(base_atm_config=_base_config(), les_config=les,
                                  area_weights=jnp.ones((2, 3)), **common_d)


def test_campaign_exit_code_reflects_health_verdict():
    """The campaign CLI exit code is the health verdict (iter 287, mirrors the OSSE
    go/no-go): 0 ONLY when the run IMPROVED, non-zero for every other status — so an HPC
    workflow gating `run_campaign && deploy` does NOT deploy a no-op (no_change /
    no_valid_diagnoses / stalled) or diverged (non_finite_bias) correction."""
    from types import SimpleNamespace

    from scripts.run.run_correction_campaign import _campaign_exit_code

    assert _campaign_exit_code(SimpleNamespace(ok=True, status="improved")) == 0
    for status in ("stalled", "no_change", "no_valid_diagnoses", "non_finite_bias"):
        assert _campaign_exit_code(SimpleNamespace(ok=False, status=status)) == 1


def test_checkpoint_common_keys_and_sanitization(monkeypatch):
    """_checkpoint_common (iter 293) builds the round/grid/initial-bias/diagnosis-count
    keys SHARED by the single + multi checkpoint callbacks: the cumulative counts thread
    the resume seed (137/138), and a diverged-initial NaN initial_bias is sanitised to null
    (a recorded metric, not an unparseable token — 245/271)."""
    import json

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(rcc, "_grid_provenance", lambda bc, g: {"grid_type": "stub"})
    init_box = {"initial_bias": float("nan"), "initial_per_variable": {"T_rmse_K": 1.0},
                "n_diagnosed_prior": 3, "n_diag_seg": 2,
                "n_diagnoses_valid_prior": 1, "n_valid_seg": 1}
    d = rcc._checkpoint_common(7, object(), object(), init_box)
    assert d["round"] == 7 and d["grid"] == {"grid_type": "stub"}
    assert d["initial_bias"] is None                     # NaN → null (sanitised)
    assert d["initial_per_variable"] == {"T_rmse_K": 1.0}
    assert d["n_diagnosed_total"] == 5                    # 3 + 2 (cumulative)
    assert d["n_diagnoses_valid_total"] == 2              # 1 + 1
    json.loads(json.dumps(d))                             # strict JSON: no NaN token


def test_checkpoint_resume_count_roundtrip(monkeypatch):
    """_checkpoint_common (write, 293) + _resume_seed (read, 294) round-trip the cumulative
    diagnosis counts: the write's ``n_*_total`` becomes the read's ``n_*_prior``, so a
    resumed campaign keeps a cumulative bias trajectory + LES-validity across a SLURM
    timeout (iter 137/138) — the write/read symmetry the no-duplication factoring locks."""
    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(rcc, "_grid_provenance", lambda bc, g: {"grid_type": "stub"})
    init_box = {"initial_bias": 4.0, "initial_per_variable": {"T_rmse_K": 2.0},
                "n_diagnosed_prior": 3, "n_diag_seg": 2,
                "n_diagnoses_valid_prior": 1, "n_valid_seg": 1}
    written = rcc._checkpoint_common(7, object(), object(), init_box)
    assert written["n_diagnosed_total"] == 5 and written["n_diagnoses_valid_total"] == 2
    seed = rcc._resume_seed(written)                 # read it back as the resume seed
    assert seed["n_diagnosed_prior"] == 5            # write total → read prior
    assert seed["n_diagnoses_valid_prior"] == 2
    assert seed["initial_bias"] == 4.0 and seed["initial_per_variable"] == {"T_rmse_K": 2.0}
    # an older checkpoint without the counts → 0 priors (no crash).
    bare = rcc._resume_seed({"initial_bias": None})
    assert bare["n_diagnosed_prior"] == 0 and bare["initial_bias"] is None


def test_resolve_coupled_preset_resolves_not_passes_raw_name():
    """_resolve_coupled_preset (iter 295) returns the RESOLVED CoupledConfig for --mode
    cmip — make_base_driver_builder reads ``preset.ocean_mode``, so the raw --coupled-preset
    NAME string crashes ('str' has no attribute ocean_mode, iter 240); the OSSE CLI had that
    latent bug (passed the raw name). AMIP → None; an unknown preset → a fail-loud SystemExit."""
    from types import SimpleNamespace

    from legoesm.driver.coupled_config import PRESETS

    from scripts.run.run_correction_campaign import _resolve_coupled_preset

    assert _resolve_coupled_preset(SimpleNamespace(mode="amip", coupled_preset=None)) is None
    name = sorted(PRESETS)[0]
    resolved = _resolve_coupled_preset(SimpleNamespace(mode="cmip", coupled_preset=name))
    assert not isinstance(resolved, str) and resolved is not None  # resolved object, NOT name
    assert hasattr(resolved, "ocean_mode")          # the attr make_base_driver_builder reads
    with pytest.raises(SystemExit, match="unknown --coupled-preset"):
        _resolve_coupled_preset(SimpleNamespace(mode="cmip", coupled_preset="bogus_preset"))


def test_build_campaign_harness_returns_the_shared_wiring():
    """_build_campaign_harness (iter 292) factors the compare/diagnose/env-grid wiring that
    was BYTE-IDENTICAL in the single + multi campaign builders: a callable compare_fn +
    diagnose_fn, and the env_grid_fn (None for 'static', a callable for 'environment') — so
    the harness lives in ONE place, not the copy-paste CLAUDE.md forbids."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

    from scripts.run.run_correction_campaign import _build_campaign_harness

    def boom(*a, **k):
        raise AssertionError("the harness builds fns; it must not run them")

    common = dict(
        base_atm_config=_base_config(), build_base_driver=boom, extract_column_state=boom,
        reference=object(), sigma=_dry_run_sigma(), grid=object(),
        area_weights=jnp.ones((2, 3)), n_worst=4, lat_deg=jnp.zeros((2, 3)),
        lon_deg=jnp.zeros((2, 3)), valid_mask=None, manifest_reducer=None,
        les_config=ColumnLESConfig(diagnosis_method="clubb_coefficient"),
        run_les_fn=boom, phis=None)
    compare_fn, diagnose_fn, env_grid_fn = _build_campaign_harness(
        **common, feedback_strategy="static")
    assert callable(compare_fn) and callable(diagnose_fn)
    assert env_grid_fn is None                       # static ⇒ no env grid
    # 'environment' strategy ⇒ a real env_grid_fn (the cross-resolution kernel predictors).
    _, _, env_fn = _build_campaign_harness(**common, feedback_strategy="environment")
    assert callable(env_fn)


def test_build_correction_campaign_rejects_unknown_diagnosis_method():
    """Dispatch hardening (iter 285): an unknown ``diagnosis_method`` RAISES (no silent
    default).  The method→coefficient resolution fires BEFORE any run (and before the
    --dry-run construction), so a typo'd ``--diagnosis-method`` is caught at launch, not
    after a multi-day run quietly correcting the wrong coefficient.  The dispatch-
    hardening RATCHET counts the guard exists; this locks that it FIRES."""
    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

    def boom(*a, **k):
        raise AssertionError("the method guard must fire before anything runs")

    common = dict(
        build_base_driver=boom, extract_column_state=boom, run_les_fn=boom,
        reference=SimpleNamespace(T=SimpleNamespace(shape=(2, 3, 5))),
        sigma=_dry_run_sigma(), grid=object(), area_weights=jnp.ones((2, 3)),
        lat_deg=jnp.zeros((2, 3)), lon_deg=jnp.zeros((2, 3)),
        n_iterations=5, n_worst=4, validate_reference=False, dry_run=True)
    with pytest.raises(ValueError, match=r"unknown diagnosis_method 'bogus_method'"):
        build_correction_campaign(
            base_atm_config=_base_config(),
            les_config=ColumnLESConfig(diagnosis_method="bogus_method"), **common)


def test_dry_run_les_per_round_caps_budget_at_n_worst():
    """The --dry-run LES/round estimate is min(les_budget, n_worst): clustering yields
    at most n_worst representatives, so a budget ABOVE n_worst must NOT over-state the
    multi-day compute (iter 260). Below n_worst the budget caps it; None → n_worst."""
    from scripts.run.run_correction_campaign import _les_per_round_estimate

    assert _les_per_round_estimate(50, 20) == 20     # budget > n_worst → capped at n_worst
    assert _les_per_round_estimate(8, 20) == 8       # budget < n_worst → the budget
    assert _les_per_round_estimate(20, 20) == 20     # equal
    assert _les_per_round_estimate(None, 20) == 20   # no clustering → one LES per column


def test_dry_run_report_formats():
    """_dry_run_report renders the assembled metadata + the would-write path."""
    rep = _dry_run_report(
        CampaignDryRun(grid_shape=(6, 8), n_worst=12, feedback_strategy="environment",
                       coefficients=("C_K", "Pr_t"), n_iterations=10, les_per_round=8),
        mode="cmip", out="/scratch/run/corrected.json")
    assert "DRY-RUN OK" in rep and "grid_shape=(6, 8)" in rep and "n_worst=12" in rep
    assert "('C_K', 'Pr_t')" in rep and "cmip" in rep
    assert "/scratch/run/corrected.json" in rep
    # compute estimate (10 rounds × 8 LES = 80 LES total) — the launch decision input.
    assert "10 rounds" in rep and "8 LES/round" in rep and "80 LES total" in rep
    # The LES spin-off is the DOMINANT runtime cost (iter 499/501 measured even a tiny
    # res-4 gray campaign as LES-dominated, >10 min CPU); the dry-run surfaces it so the
    # operator budgets HPC on the LES count, not the (small) model-run fraction.
    assert "COST" in rep and "LES spin-offs dominate" in rep
    # The spin-off LES surface-flux BC (iter 364) is a MATERIAL physics setting; the
    # pre-flight surfaces it so an operator confirms ON/off before the multi-day run
    # (iter 403). Default-constructed ⇒ off.
    assert "surface_flux=off" in rep
    rep_on = _dry_run_report(
        CampaignDryRun(grid_shape=(6, 8), n_worst=12, feedback_strategy="environment",
                       coefficients=("C_K",), n_iterations=10, les_per_round=8,
                       surface_flux=True),
        mode="amip", out="/scratch/run/c.json")
    assert "surface_flux=ON" in rep_on


def test_dry_run_report_surfaces_era5_window_and_warns_on_snapshot():
    """The pre-flight surfaces the ERA5 comparison window (iter 275) so a snapshot-vs-
    climatology misconfiguration is caught BEFORE the multi-day run: n>1 → a climatology
    line, n==1 → a weather-vs-climate WARNING; absent → no line (back-compat)."""
    from scripts.run.run_correction_campaign import _dry_run_era5_line

    dry = CampaignDryRun(grid_shape=(6, 8), n_worst=12, feedback_strategy="static",
                         coefficients=("C_K",), n_iterations=10, les_per_round=8)
    assert _dry_run_era5_line(None) == "" and _dry_run_era5_line({}) == ""
    clim = _dry_run_report(dry, mode="amip", out="o.json",
                           era5={"era5_n_times": 30, "era5_time_idx": 12})
    assert "30-time climatology @ idx 12" in clim and "WARNING" not in clim
    snap = _dry_run_report(dry, mode="amip", out="o.json",
                           era5={"era5_n_times": 1, "era5_time_idx": 5})
    assert "SINGLE snapshot @ idx 5" in snap and "WARNING" in snap
    assert "weather-vs-climate" in snap and "--era5-n-times" in snap


def test_dry_run_report_surfaces_amip_forcing_provenance():
    """The pre-flight surfaces the --amip-forcing-from-local-era5 build (iter 424): the
    forcing is BUILT before the dry-run short-circuit, so the report's AMIP-forcing line is
    the operator's confirmation that the offline boundary condition is in place; absent when
    the flag is off (back-compat)."""
    from scripts.run.run_correction_campaign import (
        _amip_forcing_provenance,
        _dry_run_amip_forcing_line,
    )

    # Provenance from args: None when off, the build dict when on.
    assert _amip_forcing_provenance(
        SimpleNamespace(amip_forcing_from_local_era5=False)) is None
    prov = _amip_forcing_provenance(SimpleNamespace(
        amip_forcing_from_local_era5=True, local_era5_dir="/rda/ERA5",
        local_era5_date="20170901", amip_forcing_out="forcing.nc",
        amip_forcing_hour_stride=24, amip_forcing_n_months=3))
    assert prov == {"source": "/rda/ERA5", "date": "20170901", "out": "forcing.nc",
                    "hour_stride": 24, "n_months": 3}

    # The line: empty when off/None, formatted when on.
    assert _dry_run_amip_forcing_line(None) == ""
    line = _dry_run_amip_forcing_line(prov)
    assert "/rda/ERA5" in line and "20170901" in line and "forcing.nc" in line
    assert "every 24h" in line and "SSTK/CI" in line

    # The report includes the forcing line only when provenance is given.
    dry = CampaignDryRun(grid_shape=(6, 8), n_worst=12, feedback_strategy="static",
                         coefficients=("C_K",), n_iterations=10, les_per_round=8)
    rep = _dry_run_report(dry, mode="amip", out="o.json", amip_forcing=prov)
    assert "AMIP forcing: built from /rda/ERA5" in rep
    rep_off = _dry_run_report(dry, mode="amip", out="o.json")
    assert "AMIP forcing" not in rep_off


def test_dry_run_flag_parsed():
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    assert p.parse_args(base).dry_run is False           # default off
    assert p.parse_args([*base, "--dry-run"]).dry_run is True


def test_main_dry_run_end_to_end_on_synthetic_era5(tmp_path):
    """The FIRST test to drive the REAL campaign main() end-to-end on real (synthetic)
    data with NO stubbing: generate a base config + a synthetic ERA5 zarr (the turnkey
    smoke-test workflow), then main([--config --era5-zarr --dry-run …]) exercises config
    load → ERA5 time-mean load → REGRID to the model grid → reference physical-validation
    → campaign construction → the dry-run report → exit 0. Covers main()'s heavy-I/O
    preamble (otherwise pragma:no-cover) for REAL, not stubbed."""
    from scripts.data.make_synthetic_era5 import main as make_era5
    from scripts.experiment.write_amip_clubb_lite_config import main as make_cfg
    from scripts.run.run_correction_campaign import main as campaign_main

    zp = str(tmp_path / "syn.zarr")
    cfg = str(tmp_path / "cfg.json")
    out = str(tmp_path / "out.json")
    assert make_era5([zp, "--nlat", "12", "--nlon", "24", "--ntime", "2"]) == 0
    assert make_cfg([cfg, "--resolution", "8", "--nlev", "8"]) == 0
    rc = campaign_main(["--config", cfg, "--era5-zarr", zp, "--mode", "amip",
                        "--n-worst", "4", "--iterations", "1", "--out", out, "--dry-run"])
    assert rc == 0                       # full construction validated on synthetic data


def test_main_dry_run_offline_amip_forcing_end_to_end(tmp_path, capsys):
    """The COMPLETE turnkey offline-AMIP path through the REAL main() (iter 425): one local
    NCAR-RDA archive supplies BOTH the SST/sea-ice FORCING (--amip-forcing-from-local-era5)
    AND the compare REFERENCE (--local-era5-dir), fully offline.  main() exercises config
    load → BUILD the AMIP forcing + inject → open the local archive → ERA5 time-mean → regrid
    → reference physical-validation → campaign construction → the dry-run report → exit 0.
    Neither main()'s --local-era5-dir end-to-end nor the --amip-forcing-from-local-era5 path
    was covered before (iter-411 was a routing spy; iter-421 tested the pieces)."""
    from scripts.experiment.write_amip_clubb_lite_config import main as make_cfg
    from scripts.run.run_correction_campaign import main as campaign_main

    write_full_archive(tmp_path)                # the shared full COMPARE+FORCING archive
    cfg = str(tmp_path / "cfg.json")
    out = str(tmp_path / "out.json")
    forcing = str(tmp_path / "forcing.nc")
    assert make_cfg([cfg, "--resolution", "8", "--nlev", "8"]) == 0

    rc = campaign_main([
        "--config", cfg, "--mode", "amip",
        "--local-era5-dir", str(tmp_path), "--local-era5-date", "20200101",
        "--amip-forcing-from-local-era5", "--amip-forcing-out", forcing,
        "--amip-forcing-hour-stride", "4",      # 8 synthetic sfc times / 4 = 2 (>= 2 guard)
        "--n-worst", "4", "--iterations", "1", "--out", out, "--dry-run"])

    assert rc == 0                              # full offline-AMIP construction validated
    # The forcing was actually BUILT during the dry-run — a VALID NetCDF with the combined
    # SST + sea-ice (not a zero-byte/partial artifact that os.path.exists alone would accept).
    import xarray as xr
    with xr.open_dataset(forcing) as fds:
        assert "SSTK" in fds.data_vars and "CI" in fds.data_vars
    report = capsys.readouterr().out
    assert "DRY-RUN OK" in report
    assert "AMIP forcing: built from" in report  # the iter-424 pre-flight line fired


def test_main_dry_run_offline_cmip_compare_end_to_end(tmp_path, capsys):
    """The offline COMPARE (--local-era5-dir) through the REAL main() in CMIP mode (iter 429).

    The iter-425 test was AMIP (with the offline FORCING); this validates the COUPLED
    construct (`CoupledESMDriver` via `--coupled-preset`) + the SAME offline compare
    reference — the path the iter-428 sbatch offline branch enables for CMIP (the ocean is
    interactive, so there is NO prescribed-SST forcing).  main() drives config load → open
    the local archive → ERA5 time-mean → regrid → reference physical-validation → the CMIP
    campaign construction → the dry-run report → exit 0."""
    from scripts.experiment.write_amip_clubb_lite_config import main as make_cfg
    from scripts.run.run_correction_campaign import main as campaign_main

    write_full_archive(tmp_path)                # the same full COMPARE archive (no forcing)
    cfg = str(tmp_path / "cfg.json")
    out = str(tmp_path / "out.json")
    assert make_cfg([cfg, "--resolution", "8", "--nlev", "8"]) == 0

    rc = campaign_main([
        "--config", cfg, "--mode", "cmip", "--coupled-preset", "aquaplanet",
        "--local-era5-dir", str(tmp_path), "--local-era5-date", "20200101",
        "--n-worst", "4", "--iterations", "1", "--out", out, "--dry-run"])

    # rc==0 + DRY-RUN OK are load-bearing: the offline archive load + the coupled-builder
    # construct both run BEFORE build_correction_campaign returns the dry-run, so a broken
    # preset resolution or archive load propagates to a non-zero exit (codex-review iter 429).
    assert rc == 0                              # full offline-CMIP construction validated
    report = capsys.readouterr().out
    assert "DRY-RUN OK" in report and "mode=cmip" in report
    assert "n_worst=4" in report                # the CampaignDryRun was built with the knob
    assert "AMIP forcing" not in report         # CMIP has no prescribed-SST forcing line


def test_surface_flux_flag_parsed():
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    assert p.parse_args(base).surface_flux is False          # default OFF
    assert p.parse_args([*base, "--surface-flux"]).surface_flux is True


def test_surface_flux_flag_wired_into_les_config(tmp_path, monkeypatch):
    """--surface-flux (iter 365) flows into the campaign's ``ColumnLESConfig.surface_flux``
    (default OFF) — spies on build_correction_campaign through the synthetic dry-run, so the
    flag→config WIRING is locked, not just the flag parse (a hardcoded value would pass a
    parse-only test).  Both the default-OFF and the opt-in are asserted."""
    import scripts.run.run_correction_campaign as rcc
    from scripts.data.make_synthetic_era5 import main as make_era5
    from scripts.experiment.write_amip_clubb_lite_config import main as make_cfg

    captured = {}
    real = rcc.build_correction_campaign

    def spy(**kw):
        captured["surface_flux"] = kw["les_config"].surface_flux
        return real(**kw)

    monkeypatch.setattr(rcc, "build_correction_campaign", spy)
    zp = str(tmp_path / "syn.zarr")
    cfg = str(tmp_path / "cfg.json")
    out = str(tmp_path / "o.json")
    assert make_era5([zp, "--nlat", "12", "--nlon", "24", "--ntime", "2"]) == 0
    assert make_cfg([cfg, "--resolution", "8", "--nlev", "8"]) == 0
    base = ["--config", cfg, "--era5-zarr", zp, "--mode", "amip", "--n-worst", "4",
            "--iterations", "1", "--out", out, "--dry-run"]
    assert rcc.main(base) == 0
    assert captured["surface_flux"] is False                 # default OFF
    assert rcc.main([*base, "--surface-flux"]) == 0
    assert captured["surface_flux"] is True                  # opt-in flows through


def test_warn_if_ignored_diagnosis_method():
    """--coefficients silently overrides --diagnosis-method (the multi path ignores it);
    a NON-default method alongside --coefficients warns so the user is not surprised. The
    default method (indistinguishable from unset) and the single-coefficient path (no
    --coefficients) do NOT warn."""
    import warnings

    import scripts.run.run_correction_campaign as rcc

    with pytest.warns(UserWarning, match="is IGNORED because --coefficients"):
        rcc._warn_if_ignored_diagnosis_method("C_K,Pr_t", "c_eps")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # any warning would fail
        rcc._warn_if_ignored_diagnosis_method("C_K,Pr_t", "clubb_coefficient")  # default→silent
        rcc._warn_if_ignored_diagnosis_method(None, "c_eps")                    # single→silent


def test_enable_line_buffered_stdout(monkeypatch):
    """The helper line-buffers stdout (real-time SLURM-log progress for a multi-day run),
    and is a SAFE no-op when stdout cannot be reconfigured (a captured/replaced stream)."""
    import sys
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    calls = {}
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(reconfigure=lambda **k: calls.update(k)))
    rcc._enable_line_buffered_stdout()
    assert calls == {"line_buffering": True}
    # No reconfigure attribute → no-op (must NOT raise).
    monkeypatch.setattr(sys, "stdout", object())
    rcc._enable_line_buffered_stdout()


def test_les_n_steps_guards_zero():
    """n_steps = int(les_hours·3600/les_dt); a too-short --les-hours (or too-large
    --les-dt) yields 0 LES steps — the forced LES would develop NO turbulence and
    diagnose nothing, silently correcting nothing over a multi-day run — so it FAILS
    LOUD at launch. A normal duration and a tiny-but-≥1 case pass through."""
    import scripts.run.run_correction_campaign as rcc

    assert rcc._les_n_steps(2.0, 0.5) == 14400           # default: 2 h / 0.5 s
    assert rcc._les_n_steps(0.0003, 0.5) == 2            # tiny but ≥ 1 step
    with pytest.raises(SystemExit, match="LES steps"):
        rcc._les_n_steps(0.0001, 0.5)                    # 0.72 → 0 steps → fail loud
    # A non-positive / NaN dt or duration is a CLEAN fail-loud, not a ZeroDivisionError
    # (les_dt==0) or a confusing negative-step message (iter 250).
    for dt in (0.0, -0.5, float("nan")):
        with pytest.raises(SystemExit, match=r"--les-dt .* must be > 0"):
            rcc._les_n_steps(2.0, dt)
    for hrs in (0.0, -2.0, float("nan")):
        with pytest.raises(SystemExit, match=r"--les-hours .* must be > 0"):
            rcc._les_n_steps(hrs, 0.5)


def test_atomic_write_json(tmp_path):
    """_atomic_write_json writes valid JSON, REPLACES an existing file, leaves no temp
    behind, and on a serialization FAILURE preserves the previous file (no corruption)
    + leaks no temp — the crash-safety a multi-day SLURM-resumable checkpoint relies on
    (a half-written checkpoint would otherwise abort the restart)."""
    import json
    import os

    p = tmp_path / "ckpt.json"
    _atomic_write_json(str(p), {"round": 0, "x": [1.0, 2.0]}, indent=2)
    assert json.loads(p.read_text()) == {"round": 0, "x": [1.0, 2.0]}
    _atomic_write_json(str(p), {"round": 1}, indent=2)          # per-round re-checkpoint
    assert json.loads(p.read_text()) == {"round": 1}            # atomically replaced
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_campaign_")]

    class _Bad:                                                 # not JSON-serializable
        pass

    with pytest.raises(TypeError):
        _atomic_write_json(str(p), {"bad": _Bad()})
    assert json.loads(p.read_text()) == {"round": 1}           # PREVIOUS content intact
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_campaign_")]  # no leak

    # Permissions match open(path,"w") — NOT mkstemp's owner-only 0o600 (which would
    # silently lock out a shared-HPC reader): a NEW file gets the umask default, and an
    # EXISTING file keeps its mode across the atomic replace.
    import stat
    new = tmp_path / "fresh.json"
    ref = tmp_path / "ref.json"
    ref.write_text("{}")                                       # open(w)-equivalent reference
    _atomic_write_json(str(new), {"k": 1})
    assert stat.S_IMODE(new.stat().st_mode) == stat.S_IMODE(ref.stat().st_mode)
    os.chmod(new, 0o640)                                       # existing file's mode…
    _atomic_write_json(str(new), {"k": 2})                     # …survives the re-checkpoint
    assert stat.S_IMODE(new.stat().st_mode) == 0o640


def test_fsync_dir_is_best_effort(tmp_path):
    """_fsync_dir makes the atomic-rename crash-DURABLE (the directory entry survives a
    crash, so the last checkpoint round is not lost), but is BEST-EFFORT: a filesystem
    that cannot fsync a directory must not break an otherwise-successful write.  A
    non-existent path (os.open raises OSError) is swallowed, not raised."""
    import scripts.run.run_correction_campaign as rcc

    rcc._fsync_dir(str(tmp_path))                              # a real dir: succeeds, no raise
    rcc._fsync_dir(str(tmp_path / "does_not_exist"))          # OSError swallowed (best-effort)


def test_assert_output_path_writable(tmp_path, monkeypatch):
    """The launch pre-flight fails LOUD (in ms) on an unwritable --out/--checkpoint so
    a multi-day run never crashes at the final json.dump: a writable existing dir is
    accepted; a MISSING parent dir raises 'does not exist'; a non-writable parent (here
    via a patched os.access) raises 'not writable'. The flag name is surfaced."""
    import os

    # Writable existing directory → no raise.
    _assert_output_path_writable(str(tmp_path / "out.json"), flag="out")
    # Missing parent directory → fail loud (more likely a typo than intent).
    with pytest.raises(SystemExit, match="does not exist"):
        _assert_output_path_writable(str(tmp_path / "nope" / "out.json"), flag="out")
    # Non-writable parent (the dir exists but W_OK is denied) → fail loud.
    monkeypatch.setattr(os, "access", lambda p, mode: False)
    with pytest.raises(SystemExit, match="not writable"):
        _assert_output_path_writable(str(tmp_path / "ckpt.json"), flag="checkpoint")


def test_assert_era5_zarr_readable(tmp_path):
    """The launch pre-flight fails FAST on a typo'd LOCAL --era5-zarr (opened only AFTER
    the model/grid build, so a bad path otherwise wastes that setup on a cryptic zarr
    error).  A valid local store passes; a missing one raises a clear, actionable
    SystemExit; and a REMOTE URI is NOT existence-checked — so a valid cloud store is
    never falsely rejected (a false positive would be worse than the late error)."""
    from scripts.run.run_correction_campaign import _assert_era5_zarr_readable

    # A valid LOCAL store (a real directory, like a Zarr) → no raise.
    _assert_era5_zarr_readable(str(tmp_path))
    # A missing LOCAL path → fail FAST with a clear, actionable message.
    with pytest.raises(SystemExit, match="no ERA5 reference store"):
        _assert_era5_zarr_readable(str(tmp_path / "typo.zarr"))
    # REMOTE URIs are left to the fsspec/zarr backend — never existence-checked here
    # (a nonexistent local path SPELLED as a URI must NOT raise → no false positive).
    for uri in ("gs://weatherbench2/datasets/era5.zarr",
                "s3://bucket/era5.zarr", "http://example.com/era5.zarr"):
        _assert_era5_zarr_readable(uri)


def test_warn_if_grid_exceeds_era5_lat_coverage():
    """The silent-failure guard: a REGIONAL --era5-zarr (whose latitude band does not
    span the global model grid) would extrapolate a GARBAGE reference outside its
    domain — surfaced as a WARNING (not a block) before the multi-day run.  A global /
    near-global ERA5 (whose pole rows sit just inside the model's, within the 2×
    spacing tolerance) is NEVER falsely flagged."""
    import warnings

    import numpy as np

    from scripts.run.run_correction_campaign import (
        _warn_if_grid_exceeds_era5_lat_coverage,
    )

    model_global = np.deg2rad(np.linspace(-88.0, 88.0, 32))      # ~global model grid
    # GLOBAL ERA5 (90..−90, 5°): covers the model ⇒ NO warning (any warning ⇒ error).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_grid_exceeds_era5_lat_coverage(
            np.deg2rad(np.linspace(90.0, -90.0, 37)), model_global)
    # REGIONAL ERA5 (30..−30): the global model reaches ~48° beyond ⇒ WARN.
    with pytest.warns(UserWarning, match="REGIONAL ERA5 store yields a GARBAGE"):
        _warn_if_grid_exceeds_era5_lat_coverage(
            np.deg2rad(np.linspace(30.0, -30.0, 13)), model_global)


def test_main_fails_fast_on_missing_era5_zarr(tmp_path):
    """main() WIRES the --era5-zarr pre-flight: a typo'd local store raises at LAUNCH
    (before the config/grid/driver build the reference load follows), not deep in
    load_era5_time_mean.  A valid --out is given so the (earlier) output guard passes
    and execution reaches the era5 guard."""
    import scripts.run.run_correction_campaign as rcc

    with pytest.raises(SystemExit, match="no ERA5 reference store"):
        rcc.main(["--config", "c.json",
                  "--era5-zarr", str(tmp_path / "missing.zarr"),
                  "--out", str(tmp_path / "out.json")])


def test_print_round_progress_emits_a_real_time_line(capsys):
    """The per-round progress line (iter 318) prints DURING the run (line-buffered) so a
    multi-day SLURM .out log shows LIVE progress instead of silence until the end — the
    operator can tell a running campaign from a hung one and watch the bias fall. It always
    fires (independent of --checkpoint, via the wrapper), off the bias / n_corrected /
    n_diagnoses_valid every per-round result (single + multi) carries."""
    from scripts.run.run_correction_campaign import _print_round_progress

    res = SimpleNamespace(
        bias=SimpleNamespace(baseline_bias=1.0, updated_bias=0.6, improved=True),
        n_corrected=5, n_diagnoses_valid=3, step_fraction=1.0)
    _print_round_progress(round_idx=2, res=res, total_rounds=10)
    out = capsys.readouterr().out
    assert "round 3/10" in out                              # 1-based display of round_idx 2
    assert "1 -> 0.6" in out and "kept" in out
    assert "5 cols corrected" in out and "3 valid LES" in out
    assert "step" not in out                                # full step ⇒ no step annotation
    # a non-improving round is labelled 'rejected' (the monotonic gate discarded it).
    res_rej = SimpleNamespace(
        bias=SimpleNamespace(baseline_bias=0.6, updated_bias=0.6, improved=False),
        n_corrected=0, n_diagnoses_valid=0, step_fraction=0.5)
    _print_round_progress(round_idx=3, res=res_rej, total_rounds=10)
    assert "rejected" in capsys.readouterr().out
    # a KEPT round at a BACKTRACKED step surfaces the line-search activity (iter 319).
    res_step = SimpleNamespace(
        bias=SimpleNamespace(baseline_bias=1.0, updated_bias=0.8, improved=True),
        n_corrected=5, n_diagnoses_valid=5, step_fraction=0.5)
    _print_round_progress(round_idx=0, res=res_step, total_rounds=10)
    assert "kept, step 0.5" in capsys.readouterr().out


def test_area_weights_prefers_true_cell_areas():
    """_area_weights returns the grid's TRUE cell areas: grid_area first, then area
    (the bias quadrature must use real areas — incl. Gaussian weights — when present,
    not a cos-lat proxy). Dispatch order: grid_area wins over area."""
    both = SimpleNamespace(grid_area=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                           area=jnp.zeros((2, 2)), grid_lat=jnp.zeros((2, 2)))
    np.testing.assert_array_equal(
        np.asarray(_area_weights(both)), [[1.0, 2.0], [3.0, 4.0]])  # grid_area preferred
    area_only = SimpleNamespace(area=jnp.array([[5.0, 6.0]]), grid_lat=jnp.zeros((1, 2)))
    np.testing.assert_array_equal(np.asarray(_area_weights(area_only)), [[5.0, 6.0]])


def test_area_weights_coslat_fallback_uses_radian_latitude():
    """When the grid exposes NEITHER grid_area NOR area, _area_weights warns and
    falls back to cos-latitude. grid_lat is stored in RADIANS across every grid
    family, so the weight is cos(lat_rad) DIRECTLY — non-vacuous: a deg2rad
    regression (treating radians as degrees) would shrink the angle ~57x and return
    ≈1 everywhere instead of the true cos-latitude profile."""
    lat_rad = jnp.array([[0.0, jnp.pi / 3.0]])  # equator + 60N, in radians
    g = SimpleNamespace(grid_lat=lat_rad)        # no grid_area / area attrs
    with pytest.warns(UserWarning, match="no cell-area weights"):
        w = _area_weights(g)
    np.testing.assert_allclose(np.asarray(w), [[1.0, 0.5]], atol=1e-6)  # cos(0)=1, cos(60°)=0.5
    # The buggy deg2rad path would have given cos(deg2rad(π/3)) ≈ 0.99985, not 0.5.
    assert abs(float(w[0, 1]) - 0.5) < 1e-6


def test_orographic_forcing_flag_parsed():
    """The --orographic-forcing flag parses to the expected choices (default auto)."""
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    assert p.parse_args(base).orographic_forcing == "auto"
    assert p.parse_args(base + ["--orographic-forcing", "on"]).orographic_forcing == "on"
    with pytest.raises(SystemExit):                       # argparse rejects bad choice
        p.parse_args(base + ["--orographic-forcing", "terrain"])


def test_era5_time_mean_flags_parsed():
    """The iter-140 time-mean reference flags parse to the expected defaults+values
    (wiring guard: a rename/removal of --era5-n-times / --era5-time-idx is caught)."""
    p = _build_arg_parser()
    base = ["--config", "c.json", "--era5-zarr", "z"]
    d = p.parse_args(base)
    assert d.era5_time_idx == 0           # default: first time
    assert d.era5_n_times == 1            # default: a single snapshot (old behaviour)
    v = p.parse_args(base + ["--era5-time-idx", "12", "--era5-n-times", "30"])
    assert v.era5_time_idx == 12
    assert v.era5_n_times == 30


def test_resolve_era5_n_times_validates_loudly():
    """--era5-n-times must be >= 1; a value < 1 fails LOUDLY (no silent max(1,...)
    clamp that would mask a fat-fingered 0 / negative) — CLAUDE.md fail-loud."""
    assert _resolve_era5_n_times(1) == 1          # the default single-snapshot window
    assert _resolve_era5_n_times(30) == 30        # a real climatology window
    assert _resolve_era5_n_times(2.0) == 2        # int-coerces a clean whole float
    for bad in (0, -1, -7):
        with pytest.raises(SystemExit, match=r"--era5-n-times must be >= 1"):
            _resolve_era5_n_times(bad)
    # a non-integral float is rejected too (no silent truncation 2.9 -> 2).
    with pytest.raises(SystemExit, match=r"must be a whole number"):
        _resolve_era5_n_times(2.9)


def test_refuse_unsupported_multirank_guards_cli():
    """The single-process CLI refuses an mpirun -np >1 launch LOUDLY (it wires none
    of the distributed hooks); a single rank (or no MPI) is allowed (iter 88)."""
    refuse_unsupported_multirank(comm_size=1)          # single rank: allowed
    with pytest.raises(SystemExit, match="SINGLE-PROCESS CLI"):
        refuse_unsupported_multirank(comm_size=2)      # multi-rank: refused

_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


def _full_grid_state(nlat=8, nlon=16, nlev=5):
    shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0), q_v=jnp.full(shp, 5e-3),
        u=jnp.full(shp, 5.0), v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5), sst_K=jnp.full(sfc, 290.0))


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        segment_callback(self, 0.0, 1.0)        # one diagnostic segment
        return "OK"


def _mock_run_les(setup):
    """Mock plane-LES result: a synthetic state shaped to the setup's grid/hc."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import make_rest_state

    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    return state._replace(
        w=state.w.replace(data=w),
        theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr))


def _base_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")


def test_grid_latlon_deg_defaults_to_grid_centre_degrees():
    """grid_latlon_deg defaults lat/lon to the grid's centre coords CONVERTED to
    DEGREES (radians·180/π); explicit values pass through UNCHANGED (no double
    conversion). A dropped rad2deg would tag the manifest with radian 'latitudes'."""
    from types import SimpleNamespace

    grid = SimpleNamespace(grid_lat=np.array([0.0, np.pi / 2]),
                           grid_lon=np.array([0.0, np.pi]))
    lat, lon = grid_latlon_deg(grid, None, None)
    np.testing.assert_allclose(np.asarray(lat), [0.0, 90.0])
    np.testing.assert_allclose(np.asarray(lon), [0.0, 180.0])
    # Explicit values pass through (caller already supplied degrees).
    lat2, lon2 = grid_latlon_deg(grid, jnp.array([12.0]), jnp.array([34.0]))
    np.testing.assert_array_equal(np.asarray(lat2), [12.0])
    np.testing.assert_array_equal(np.asarray(lon2), [34.0])


def test_load_base_config_and_grid_roundtrip(tmp_path):
    """load_base_config_and_grid loads the ExperimentConfig JSON + builds the SAME grid
    + vertical coord the run uses (latlon res 8, nlev 5), WITHOUT the heavy setup() —
    the config-loading entry shared by the campaign CLI AND the OSSE CLI, so a
    regression here breaks both launches."""
    import json

    from legoesm.driver.config import experiment_config_to_dict

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(experiment_config_to_dict(_base_config())))
    cfg, grid, sigma = load_base_config_and_grid(str(p))
    assert cfg.grid.grid_type == "latlon" and cfg.grid.resolution == 8
    assert cfg.turbulence == "clubb_lite"             # the campaign-required scheme survives
    assert grid is not None and hasattr(grid, "grid_lat")
    assert int(np.asarray(sigma.sigma_full).shape[0]) == 5   # nlev levels built


def test_maybe_env_grid_fn_dispatch():
    """maybe_env_grid_fn returns the env-grid producer ONLY for 'environment' (else
    None — the static-scatter feedback needs no env grid), with ``sigma`` bound."""
    from legoesm.training.feedback_assembly import column_environment_grid

    sigma = object()
    fn = maybe_env_grid_fn("environment", sigma)
    assert callable(fn)
    assert fn.func is column_environment_grid        # partial of the right producer
    assert fn.keywords.get("sigma") is sigma         # sigma bound into the partial
    assert maybe_env_grid_fn("static", sigma) is None
    assert maybe_env_grid_fn("anything_else", sigma) is None


def test_main_local_era5_dir_routes_to_the_archive_adapter(monkeypatch):
    """iter 410/411: --local-era5-dir routes the reference load through the offline
    NCAR-RDA adapter (``open_local_era5_dataset(dir, date)``) — NOT the Zarr/network
    path — and threads the opened dataset to ``load_era5_time_mean(ds=...)``.  Spies the
    adapter (so no real archive is needed) + --dry-run so nothing runs."""
    import scripts.data.load_local_era5 as lle
    import scripts.run.run_correction_campaign as rcc

    _stub_campaign_main_io(monkeypatch)
    seen = {}

    class _ReachedError(Exception):
        pass

    def _spy(data_dir, date):
        seen["call"] = (data_dir, date)     # record the routing, then stop main() here
        raise _ReachedError

    monkeypatch.setattr(lle, "open_local_era5_dataset", _spy)
    with pytest.raises(_ReachedError):           # main() reached the LOCAL adapter (not the zarr)
        rcc.main(["--config", "c.json", "--out", "o.json", "--local-era5-dir",
                  "/rda/ERA5", "--local-era5-date", "20170901"])
    assert seen["call"] == ("/rda/ERA5", "20170901")     # the adapter WAS the source


def test_main_requires_exactly_one_era5_source():
    """iter 410: the ERA5 reference is EITHER --era5-zarr OR --local-era5-dir (the
    offline NCAR-RDA archive) — never neither and never both — and --local-era5-dir
    needs --local-era5-date.  Fail LOUD at launch (right after parse), before any
    model/grid build, so a misconfigured source never wastes the (multi-day) setup."""
    import scripts.run.run_correction_campaign as rcc

    base = ["--config", "c.json", "--out", "o.json"]
    with pytest.raises(SystemExit, match="EXACTLY one ERA5 reference"):
        rcc.main(base)                                                   # neither source
    with pytest.raises(SystemExit, match="EXACTLY one ERA5 reference"):
        rcc.main(base + ["--era5-zarr", "z", "--local-era5-dir", "d"])   # both sources
    with pytest.raises(SystemExit, match="requires --local-era5-date"):
        rcc.main(base + ["--local-era5-dir", "d"])                       # local, no date


def test_main_amip_forcing_from_local_era5_validation():
    """iter 421: --amip-forcing-from-local-era5 builds the AMIP SST/sea-ice forcing from the
    SAME local archive — only valid for a prescribed-SST AMIP run from --local-era5-dir.
    Fail LOUD at launch on a CMIP request or a missing local source (before the model
    build)."""
    import scripts.run.run_correction_campaign as rcc

    base = ["--config", "c.json", "--out", "o.json", "--amip-forcing-from-local-era5"]
    # CMIP has an interactive ocean — a prescribed SST forcing is meaningless.
    with pytest.raises(SystemExit, match="requires --mode amip"):
        rcc.main(base + ["--mode", "cmip", "--era5-zarr", "z"])
    # AMIP but no local archive: the forcing has nowhere to come from.
    with pytest.raises(SystemExit, match="requires --local-era5-dir"):
        rcc.main(base + ["--mode", "amip", "--era5-zarr", "z"])
    # AMIP + local dir but no date: the existing date guard fires first (the build needs the
    # date) — pins the guard ordering so the flag can never reach the build date-less.
    with pytest.raises(SystemExit, match="requires --local-era5-date"):
        rcc.main(base + ["--mode", "amip", "--local-era5-dir", "D"])


def test_maybe_apply_local_era5_forcing_builds_and_injects_only_when_flagged(monkeypatch):
    """iter 421: the campaign helper is a NO-OP unless --amip-forcing-from-local-era5; when
    set it builds the forcing from the local archive (with the CLI's stride/out/date) and
    injects it into base_cfg.  The build is separately tested, so stub it here."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.forcing.amip import AMIPForcingConfig

    import scripts.run.run_correction_campaign as rcc

    base_cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation="gray", days=2,
        dataset="analytical")

    called = {}

    def _fake_build(data_dir, date, out_path, *, hour_stride, n_months=1):
        called.update(dir=data_dir, date=date, out=out_path, stride=hour_stride,
                      n_months=n_months)
        return AMIPForcingConfig(
            dataset="custom", path=out_path, sst_var="SSTK", sic_var="CI",
            sst_offset=0.0, sic_scale=1.0)

    monkeypatch.setattr(
        "scripts.data.load_local_era5.build_era5_amip_forcing", _fake_build)

    # OFF ⇒ identity (and the build is never called).
    args_off = SimpleNamespace(amip_forcing_from_local_era5=False)
    assert rcc._maybe_apply_local_era5_forcing(args_off, base_cfg) is base_cfg
    assert not called

    # ON ⇒ build with the CLI args (incl. n_months, iter 459) + inject the forcing fields.
    args_on = SimpleNamespace(
        amip_forcing_from_local_era5=True, local_era5_dir="D",
        local_era5_date="20200101", amip_forcing_out="F.nc",
        amip_forcing_hour_stride=12, amip_forcing_n_months=2)
    out = rcc._maybe_apply_local_era5_forcing(args_on, base_cfg)
    assert called == {"dir": "D", "date": "20200101", "out": "F.nc", "stride": 12,
                      "n_months": 2}
    assert out.dataset == "custom" and out.forcing_path == "F.nc"
    assert out.sst_var == "SSTK" and out.sic_var == "CI"
    assert out.radiation == "gray" and out.days == 2     # unrelated fields preserved


def test_main_cmip_rejects_unknown_coupled_preset(monkeypatch):
    """--mode cmip with an unknown --coupled-preset fails LOUD at launch (a typo'd preset)
    with the valid choices, NOT a cryptic KeyError deeper in the coupled-driver build.
    Fires before the reference loading, so only load_base_config_and_grid is stubbed."""
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(
        rcc, "load_base_config_and_grid",
        lambda path: (SimpleNamespace(
            grid=SimpleNamespace(grid_type="latlon", resolution=8, nlev=5)),
            object(), object()))
    # The era5-readability pre-flight (iter 389) runs before _build_run_setup (where the
    # coupled-preset guard fires), so bypass it for this dummy --era5-zarr path.
    monkeypatch.setattr(rcc, "_assert_era5_zarr_readable", lambda path: None)
    with pytest.raises(SystemExit, match="unknown --coupled-preset"):
        rcc.main(["--config", "c.json", "--era5-zarr", "z", "--mode", "cmip",
                  "--coupled-preset", "not_a_preset"])


def _stub_campaign_main_io(monkeypatch):
    """Stub the campaign main()'s heavy preamble — config/grid load, driver builder, ERA5
    reference loading (load_era5_time_mean → regrid → column_state_from_carry), orographic
    phis — on their SOURCE modules (function-scoped-import-safe), up to the resume/dispatch
    logic. Returns the fake grid. Shared by the main() launch-guard tests."""
    from types import SimpleNamespace

    import legoesm.training.compare_reanalysis as cr
    import legoesm.training.era5_to_state as e2s

    import scripts.run.run_correction_campaign as rcc
    import scripts.validate.compare_amip_era5 as cae

    fake_cfg = SimpleNamespace(
        grid=SimpleNamespace(grid_type="latlon", resolution=8, nlev=5))
    fake_grid = SimpleNamespace(grid_shape_2d=(2, 3), grid_lat=jnp.zeros(2))
    monkeypatch.setattr(rcc, "load_base_config_and_grid",
                        lambda path: (fake_cfg, fake_grid, object()))
    monkeypatch.setattr(
        rcc, "make_base_driver_builder",
        lambda mode, coupled_preset=None, ocean_grid=None: ((lambda c: None),
                                                            (lambda d, day, dt: None)))
    monkeypatch.setattr(e2s, "load_era5_time_mean",
                        lambda cfg, idx, **kw: SimpleNamespace(lat=jnp.zeros(2)))
    monkeypatch.setattr(cae, "select_era5_regrid", lambda canon: (lambda slc, g, s: object()))
    monkeypatch.setattr(cr, "column_state_from_carry", lambda carry: object())
    monkeypatch.setattr(rcc, "resolve_orographic_phis", lambda forcing, provider: None)
    # The launch pre-flights (iters 389-391) run in the preamble BEFORE the resume/
    # dispatch logic these tests target; the dummy --era5-zarr path + stubbed
    # (attribute-less) era5 slice would trip the readability + lat-coverage checks
    # first, so stub them here (each has its own direct test).
    monkeypatch.setattr(rcc, "_assert_era5_zarr_readable", lambda path: None)
    monkeypatch.setattr(rcc, "_warn_if_grid_exceeds_era5_lat_coverage", lambda a, b: None)
    # The effective-config sidecar write (iter 464) runs in the preamble BEFORE the resume/
    # dispatch guards these tests target; it calls config_to_dict(base_cfg), which the
    # SimpleNamespace fake_cfg above does not support — stub it (a preamble side-effect, not
    # what these tests assert) so the launch reaches the resume/dispatch logic.
    monkeypatch.setattr(rcc, "_write_effective_config", lambda base_cfg, out: "")
    return fake_grid


def test_main_resume_rejects_mismatched_corrected_field(tmp_path, monkeypatch):
    """The HOT resume path (multi-day SLURM restarts): resuming a checkpoint whose
    corrected coefficient (Pr_t) differs from --diagnosis-method's field (C_K) fails LOUD
    at launch, NOT silently loading the field into the WRONG config slot (a garbage
    correction). Drives main() with the heavy preamble stubbed up to the resume guard."""
    import json

    import scripts.run.run_correction_campaign as rcc

    _stub_campaign_main_io(monkeypatch)
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps({"corrected_field": "Pr_t", "field": [[0.4]], "round": 0}))
    with pytest.raises(SystemExit, match="checkpoint corrects"):
        rcc.main(["--config", "c.json", "--era5-zarr", "z", "--resume", str(ckpt),
                  "--diagnosis-method", "clubb_coefficient"])    # corrected_field "C_K" != "Pr_t"


def test_main_multi_resume_rejects_mismatched_coefficients(tmp_path, monkeypatch):
    """The MULTI-coefficient resume guard (parallel to the single-coefficient one): the
    simultaneous campaign also resumes after a SLURM timeout, and resuming a checkpoint
    whose coefficient SET differs from --coefficients fails LOUD, not silently loading the
    wrong per-coefficient fields. --coefficients dispatches main() into _run_multi_main."""
    import json

    import scripts.run.run_correction_campaign as rcc

    _stub_campaign_main_io(monkeypatch)
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps(
        {"coefficients": ["C_K", "Pr_t"], "fields": {}, "round": 0}))
    with pytest.raises(SystemExit, match="checkpoint coefficients"):
        rcc.main(["--config", "c.json", "--era5-zarr", "z", "--resume", str(ckpt),
                  "--coefficients", "C_K,C_eps"])     # set {C_K,C_eps} != ckpt {C_K,Pr_t}


def test_maybe_write_env_kernel_writes_when_kernel_present(tmp_path, monkeypatch, capsys):
    """``_maybe_write_env_kernel`` exports the RAW env-kernel JSON (the grid-AGNOSTIC
    cross-resolution deploy PRODUCER) when the last ACCEPTED round produced one — the
    write/skip glue was untested (its components are)."""
    import json
    from types import SimpleNamespace

    import legoesm.training.correction_loop as cl
    import legoesm.training.deploy_correction as dc

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(cl, "last_accepted_env_kernel", lambda result: "KERNEL")
    monkeypatch.setattr(dc, "env_kernel_to_dict", lambda k: {"field": "C_K", "k": k})
    out = str(tmp_path / "camp.json")
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="environment",
                        era5_time_idx=12, era5_n_times=30), object())
    with open(f"{out}.env_kernel.json") as f:
        written = json.load(f)
    # the kernel dict PLUS the iter-269 averaging-window provenance stamp (locks the
    # write-site stamp, not just the codec tolerance — the gap that let iter 269 break
    # this test undetected for several iters).
    assert written == {"field": "C_K", "k": "KERNEL",
                       "averaging": {"era5_time_idx": 12, "era5_n_times": 30}}
    assert "wrote RAW environment kernel" in capsys.readouterr().out


def test_maybe_write_env_kernel_skips_and_warns_when_no_kernel(tmp_path, monkeypatch, capsys):
    """No kernel (a rejected/no-op round) → NO file; an 'environment'-strategy campaign
    that produced none WARNS (the cross-grid artifact the user expected is absent), while
    'static' is SILENT (it has no env→coefficient regression to transfer)."""
    import os
    from types import SimpleNamespace

    import legoesm.training.correction_loop as cl

    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(cl, "last_accepted_env_kernel", lambda result: None)
    out = str(tmp_path / "camp.json")
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="environment"), object())
    assert not os.path.exists(f"{out}.env_kernel.json")            # nothing written
    assert "WARNING" in capsys.readouterr().out                    # env-strategy + no kernel ⇒ warn
    rcc._maybe_write_env_kernel(
        SimpleNamespace(out=out, feedback_strategy="static"), object())
    assert "WARNING" not in capsys.readouterr().out                # static ⇒ silent


def test_print_deploy_hint_verifies_output_deploys_on_own_grid(tmp_path, capsys):
    """``_print_deploy_hint`` round-trips the JUST-WRITTEN campaign output through the
    production deploy loader WITH its own grid, so a non-deployable / grid-inconsistent
    output fails LOUD at WRITE time — not silently when the HPC user deploys it days
    later. A valid output passes + prints the deploy hint; an output verified against a
    DIFFERENT grid raises (the deploy guard's rejection, propagated not swallowed)."""
    import json

    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    output = {"C_K": [0.4] * (8 * 16),
              "grid": rcc._grid_provenance(_base_config(), grid)}
    path = str(tmp_path / "out.json")
    with open(path, "w") as f:
        json.dump(output, f)
    rcc._print_deploy_hint(path, grid)                  # deploys onto its OWN grid → no raise
    assert "deploy into a production run" in capsys.readouterr().out
    # An output verified against a DIFFERENT grid fails LOUD at write-time.
    other = create_latlon_grid(4, 8, dtype=jnp.float64)
    with pytest.raises(ValueError):
        rcc._print_deploy_hint(path, other)


def test_grid_provenance_produces_deploy_compatible_fingerprint():
    """``_grid_provenance`` is the campaign-side deploy grid-identity fingerprint PRODUCER
    embedded in the output JSON; the deploy CONSUMER (``assert_deploy_compatible``) is
    tested but the producer was NOT (a private helper the def-vs-test sweep skips). Lock
    (a) the recorded metadata, (b) the producer→consumer ROUND-TRIP (a provenanced output
    deploys onto its OWN grid), and (c) non-vacuity (a DIFFERENT grid → DIFFERENT coord
    fingerprint) — so a producer regression cannot silently break the grid-safety guard
    that stops per-column coefficients landing on the wrong cells (clause-6 deploy)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.deploy_correction import (
        assert_deploy_compatible,
        corrected_turbulence_override,
    )

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    base_cfg = _base_config()                           # latlon, resolution 8, nlev 5
    prov = rcc._grid_provenance(base_cfg, grid)
    assert prov["grid_type"] == "latlon"
    assert prov["resolution"] == 8 and prov["nlev"] == 5
    assert "coord_sha256" in prov                       # the coordinate fingerprint

    output = {"C_K": [0.4] * (8 * 16), "grid": prov}
    assert_deploy_compatible(output, grid)              # PRODUCER → CONSUMER round-trip OK
    assert corrected_turbulence_override(output, grid=grid).scheme == "clubb_lite"

    # Non-vacuity: a DIFFERENT grid yields a DIFFERENT coord fingerprint (so the guard
    # would catch a cross-grid deploy — the consumer's tested job; here we prove the
    # PRODUCER distinguishes grids rather than emitting a constant).
    other = create_latlon_grid(4, 8, dtype=jnp.float64)
    assert rcc._grid_provenance(base_cfg, other)["coord_sha256"] != prov["coord_sha256"]


def test_assert_resume_grid_matches():
    """The RESUME wrong-cell guard (partner to the iter-185 atomic checkpoint): a
    checkpoint written on a DIFFERENT grid fails LOUD before its per-column field is
    reshaped onto the current grid. The dangerous case is a SAME-column-count but
    different-SHAPE grid — (8,16) vs (16,8), both 128 cells — which would reshape
    SUCCESSFULLY yet scramble the field onto the wrong cells (same hazard as the deploy
    guard). The SAME grid passes; an older checkpoint with no fingerprint only WARNS
    (the field-length reshape stays the fallback)."""
    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    prov = rcc._grid_provenance(_base_config(), grid)
    rcc._assert_resume_grid_matches({"grid": prov}, grid)          # same grid → no raise
    # Same ncol (128) but TRANSPOSED shape → reshape would scramble silently → fail loud.
    other = create_latlon_grid(16, 8, dtype=jnp.float64)
    with pytest.raises(SystemExit, match="resume grid mismatch"):
        rcc._assert_resume_grid_matches({"grid": prov}, other)
    # An older checkpoint without a fingerprint only WARNS (no hard fail).
    with pytest.warns(UserWarning, match="no grid fingerprint"):
        rcc._assert_resume_grid_matches({"round": 3}, grid)


def test_load_single_resume_roundtrip():
    """The HOT multi-day-restart path: a checkpoint written by the campaign (grid block +
    2-D 'field' + corrected_field + cumulative baseline/counts) reconstructs back to the
    SAME accumulated per-column field + CLUBB config (config == flattened field) + resume
    round (round+1). Locks the write→read round-trip — the reconstruction was inline in
    pragma:no-cover main() — plus the wrong-slot / wrong-cell / old-checkpoint paths."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    field2d = (jnp.arange(128.0) * 0.001 + 0.4).reshape(8, 16)   # plausible per-column C_K
    ckpt = {
        "round": 3, "grid": rcc._grid_provenance(_base_config(), grid),
        "corrected_field": "C_K",
        "C_K": np.asarray(field2d).reshape(-1).tolist(),
        "field": np.asarray(field2d).tolist(),
        "initial_bias": 5.0, "initial_per_variable": None,
        "n_diagnosed_total": 12, "n_diagnoses_valid_total": 9,
    }
    f, clubb, sr, seed = rcc._load_single_resume(ckpt, grid, "C_K")
    np.testing.assert_allclose(np.asarray(f), np.asarray(field2d))          # field rebuilt
    np.testing.assert_allclose(np.asarray(clubb.C_K).reshape(-1),
                               np.asarray(field2d).reshape(-1))             # config == field
    assert sr == 4                                                         # round + 1
    assert seed["initial_bias"] == 5.0 and seed["n_diagnosed_prior"] == 12
    assert seed["n_diagnoses_valid_prior"] == 9
    # Wrong-SLOT guard: a Pr_t checkpoint resumed as C_K fails loud.
    with pytest.raises(SystemExit, match="checkpoint corrects"):
        rcc._load_single_resume({**ckpt, "corrected_field": "Pr_t"}, grid, "C_K")
    # Wrong-MODE guard (iter 252): a MULTI checkpoint ('fields' dict, no single 'field')
    # resumed as single (forgot --coefficients) fails loud with a fix, not a bare KeyError —
    # symmetric with the multi path's coefficient-set guard.
    multi_like = {k: v for k, v in ckpt.items() if k != "field"}
    multi_like["fields"] = {"clubb_lite_C_K": list(range(128))}
    with pytest.raises(SystemExit, match=r"has no 'field'.*--coefficients"):
        rcc._load_single_resume(multi_like, grid, "C_K")
    # Wrong-CELL guard: a transposed-shape grid fails loud.
    with pytest.raises(SystemExit, match="resume grid mismatch"):
        rcc._load_single_resume(ckpt, create_latlon_grid(16, 8, dtype=jnp.float64), "C_K")
    # An OLD checkpoint (no grid block, no cumulative fields) warns + falls back to None/0.
    with pytest.warns(UserWarning, match="no grid fingerprint"):
        _f2, _c2, sr2, seed2 = rcc._load_single_resume(
            {"round": 0, "field": np.asarray(field2d).tolist(), "corrected_field": "C_K"},
            grid, "C_K")
    assert sr2 == 1 and seed2["initial_bias"] is None and seed2["n_diagnosed_prior"] == 0


def test_load_multi_resume_roundtrip():
    """The symmetric MULTI-coefficient resume reconstruction (parallel to
    _load_single_resume): a campaign-written multi checkpoint (grid block + per-coefficient
    flat 'fields' keyed by promotion_key + coefficient SET + cumulative seed) rebuilds the
    accumulated per-coefficient initial_fields (2-D, keyed by promotion_key) + the CLUBB
    config from those fields + round+1; a coefficient-SET mismatch and a transposed grid
    both fail LOUD."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    ck = (jnp.arange(128.0) * 0.001 + 0.4).reshape(8, 16)
    prt = (jnp.arange(128.0) * 0.002 + 0.6).reshape(8, 16)
    coefficients = ("C_K", "Pr_t")
    pk_ck = rcc.COEFFICIENT_SPEC_MAP["C_K"][0]              # promotion key, e.g. clubb_lite_C_K
    pk_prt = rcc.COEFFICIENT_SPEC_MAP["Pr_t"][0]
    promo_to_field = {pk_ck: "C_K", pk_prt: "Pr_t"}
    ckpt = {
        "round": 2, "grid": rcc._grid_provenance(_base_config(), grid),
        "coefficients": ["C_K", "Pr_t"],
        "fields": {pk_ck: np.asarray(ck).reshape(-1).tolist(),
                   pk_prt: np.asarray(prt).reshape(-1).tolist()},
        "initial_bias": 7.0, "initial_per_variable": None,
        "n_diagnosed_total": 8, "n_diagnoses_valid_total": 5,
    }
    gshape = grid.grid_shape_2d
    clubb, fields, sr, seed = rcc._load_multi_resume(
        ckpt, grid, coefficients, gshape, promo_to_field)
    np.testing.assert_allclose(np.asarray(clubb.C_K).reshape(-1), np.asarray(ck).reshape(-1))
    np.testing.assert_allclose(np.asarray(clubb.Pr_t).reshape(-1), np.asarray(prt).reshape(-1))
    np.testing.assert_allclose(np.asarray(fields[pk_ck]), np.asarray(ck))  # 2-D, keyed by promo key
    assert sr == 3 and seed["initial_bias"] == 7.0 and seed["n_diagnosed_prior"] == 8
    # Coefficient-SET mismatch → fail loud (the wrong per-coefficient fields would load).
    with pytest.raises(SystemExit, match="checkpoint coefficients"):
        rcc._load_multi_resume({**ckpt, "coefficients": ["C_K", "C_eps"]}, grid,
                               coefficients, gshape, promo_to_field)
    # Wrong-cell guard: a transposed-shape grid fails loud before the reshape.
    with pytest.raises(SystemExit, match="resume grid mismatch"):
        rcc._load_multi_resume(ckpt, create_latlon_grid(16, 8, dtype=jnp.float64),
                               coefficients, (16, 8), promo_to_field)


def test_compose_compare_fn_threads_valid_mask_and_manifest_reducer(monkeypatch):
    """compose_compare_fn must pass ``valid_mask`` + ``manifest_reducer`` THROUGH to
    make_compare_fn, so an ocean/land mask or the distributed owned-cell mask actually
    restricts the worst-column ranking. A dropped pass-through would silently rank +
    correct masked (e.g. non-owned halo / land) cells — otherwise caught ONLY by the
    MPI e2e tests (which do not run in the default fast suite)."""
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    captured = {}

    def fake_make_compare_fn(**kwargs):
        captured.update(kwargs)
        return lambda config: None

    def reducer(manifest):                           # the distributed top-k post-processor
        return manifest

    monkeypatch.setattr(rcc, "make_compare_fn", fake_make_compare_fn)
    vm = jnp.array([True, False, True])
    sigma = SimpleNamespace(sigma_full=jnp.zeros(5), sigma_half=jnp.zeros(6))
    compose_compare_fn(
        base_atm_config=_base_config(),          # turbulence="clubb_lite" (required)
        build_base_driver=(lambda cfg: object()),
        extract_column_state=(lambda d, day, dt: None),
        reference=object(), sigma=sigma, area_weights=jnp.ones(3), n_worst=2,
        lat_deg=jnp.zeros(3), lon_deg=jnp.zeros(3),
        valid_mask=vm, manifest_reducer=reducer,
    )
    assert captured["valid_mask"] is vm              # the mask is threaded, not dropped
    assert captured["manifest_reducer"] is reducer   # the distributed top-k reducer too
    assert captured["n_worst"] == 2                  # (sanity: other args also threaded)


def test_compose_compare_fn_threads_coordinate_for_hybrid_weights(monkeypatch):
    """compose_compare_fn must pass ``coordinate=sigma`` THROUGH to make_compare_fn so
    the per-run layer pressures (hence the bias MASS WEIGHTS) are hybrid-correct
    (iter 337/338).  A dropped pass-through would silently revert the campaign compare
    to PURE-SIGMA weights — the exact iter-337 over-terrain mis-weighting (~276% upper-
    level over-weighting at p_s≠p_ref) — and NO fast test would catch it: the
    ``make_compare_fn`` unit test supplies ``coordinate`` explicitly, so it cannot see
    a campaign-side drop.  This is the campaign-level regression guard for that fix."""
    from types import SimpleNamespace

    import scripts.run.run_correction_campaign as rcc

    captured = {}

    def fake_make_compare_fn(**kwargs):
        captured.update(kwargs)
        return lambda config: None

    monkeypatch.setattr(rcc, "make_compare_fn", fake_make_compare_fn)
    sigma = SimpleNamespace(sigma_full=jnp.zeros(5), sigma_half=jnp.zeros(6))
    compose_compare_fn(
        base_atm_config=_base_config(),          # turbulence="clubb_lite" (required)
        build_base_driver=(lambda cfg: object()),
        extract_column_state=(lambda d, day, dt: None),
        reference=object(), sigma=sigma, area_weights=jnp.ones(3), n_worst=2,
        lat_deg=jnp.zeros(3), lon_deg=jnp.zeros(3),
    )
    # The coordinate is threaded as the SAME object (⇒ make_compare_fn derives hybrid
    # pressures from it); a None here would be the silent pure-sigma fallback.
    assert captured["coordinate"] is sigma
    assert captured["coordinate"] is not None


def test_make_clubb_build_driver_injects_override():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    captured = {}

    def build_base(cfg):
        captured["cfg"] = cfg
        return _FakeDriver(_full_grid_state())

    build_driver = make_clubb_build_driver(_base_config(), build_base)
    build_driver(CLUBBLiteConfig(C_K=0.9))
    cfg = captured["cfg"]
    assert cfg.turbulence == "clubb_lite"
    assert cfg.turbulence_override is not None
    assert float(cfg.turbulence_override.clubb_lite.C_K) == 0.9


def test_make_les_diagnose_fn_runs_process_column():
    """diagnose_fn(record, model_ctx) drives the REAL process_column + a mock LES."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815  (mirrors ColumnEnvironment field)

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    out = diagnose_fn(_Rec(), _full_grid_state())
    assert out.K.shape == (_SMALL_RES.nlev - 1,)   # eddy-K profile
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_make_les_diagnose_fn_threads_phis_to_process_column(monkeypatch):
    """iter-118 wiring: make_les_diagnose_fn(phis=...) forwards the static model
    topography to process_column (→ the orographic geostrophic term); the default
    forwards None (flat). The forcing extraction inside process_column is stubbed so
    the test pins ONLY the phis threading."""
    import legoesm.atmosphere.dynamics.les.column_les as cl
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    captured = {}

    def fake_process_column(record, **kw):
        captured["phis"] = kw.get("phis", "MISSING")
        return "DIAG"

    # make_les_diagnose_fn imports process_column at call time (function-scope), so
    # patch the module attribute BEFORE building the diagnose_fn.
    monkeypatch.setattr(cl, "process_column", fake_process_column)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    phis_grid = jnp.zeros((8, 16))
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, phis=phis_grid)
    assert diagnose_fn(_Rec(), _full_grid_state()) == "DIAG"
    assert captured["phis"] is phis_grid          # forwarded to process_column

    diagnose_flat = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)                 # no phis → flat
    diagnose_flat(_Rec(), _full_grid_state())
    assert captured["phis"] is None


def test_make_les_diagnose_fn_threads_sst_to_process_column(monkeypatch):
    """iter-364 wiring: make_les_diagnose_fn forwards ``model_ctx.sst_K`` to
    process_column (→ the opt-in surface-flux BC, activated by
    ``ColumnLESConfig.surface_flux``).  process_column is stubbed so the test pins ONLY
    the SST threading."""
    import legoesm.atmosphere.dynamics.les.column_les as cl
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    captured = {}

    def fake_process_column(record, **kw):
        captured["sst_K"] = kw.get("sst_K", "MISSING")
        return "DIAG"

    monkeypatch.setattr(cl, "process_column", fake_process_column)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)
    state = _full_grid_state()                     # the comparison state carries sst_K (290 K)
    assert diagnose_fn(_Rec(), state) == "DIAG"
    assert captured["sst_K"] is state.sst_K        # the model-grid SST is forwarded


def test_make_les_diagnose_fn_mpas_routes_edge_velocity():
    """An MPAS worst column spins off its LES end-to-end (iter 76): the model_ctx
    carries the native u_edge, make_les_diagnose_fn routes it to the Voronoi
    forcing extractor (grid=mesh, v=None), and the mock LES yields a diagnosis."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    diagnose_fn = make_les_diagnose_fn(
        mesh, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    # An MPAS ColumnState: cell T/q_v/p_s + the NATIVE edge velocity in u_edge.
    u_edge = 6.0 * jnp.cos(jnp.asarray(mesh.angleEdge))[:, None] * jnp.ones((1, nlev))
    mpas_ctx = ColumnState(
        T=jnp.full((mesh.nCells, nlev), 285.0),
        q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)),     # cell winds present but unused (routed)
        v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        u_edge=u_edge)

    cell = 40

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (cell,)                  # arity-1 cell index for MPAS
        lat_deg = float(np.rad2deg(np.asarray(mesh.latCell)[cell]))
        environment = _Env()

    out = diagnose_fn(_Rec(), mpas_ctx)
    assert out.K.shape == (_SMALL_RES.nlev - 1,)
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_make_les_diagnose_fn_mpas_u_edge_on_wrong_grid_raises():
    """u_edge set but a non-Voronoi grid → loud grid/state mismatch (Codex)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    diagnose_fn = make_les_diagnose_fn(
        grid, create_sigma_coordinate(5),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME), run_les_fn=_mock_run_les)
    ctx = _full_grid_state()._replace(u_edge=jnp.zeros((10, 5)))

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0

    with pytest.raises(ValueError, match="not a VoronoiMesh"):
        diagnose_fn(_Rec(), ctx)


@pytest.mark.slow
def test_build_correction_campaign_wiring_one_round():
    """WIRING test: the whole campaign composes + runs one round with a mock
    driver + mock LES — a per-column clubb C_K is produced and the bias is a
    finite measurement.  Does NOT validate that the C_K injection changes the
    model output (the mock driver returns a fixed state) — that mechanism is
    covered by iter 35/37."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    # reference = model with a localized +6 K bias → one deterministic worst column.
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    def build_base(cfg):           # mock: fixed state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1,
        # Test the config→model wiring (a per-column C_K is applied); the mock
        # compare has no real improvement signal, so disable the monotonic gate
        # that would otherwise reject this round (gate tested in test_correction_loop).
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1
    # The campaign produced a per-column clubb C_K (the loop closed).
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 1
    assert result.final_config.C_K.shape == (8 * 16,)


def _mpas_base_config(nlev=5):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="mpas"),
        radiation="gray", turbulence="clubb_lite")


def _mpas_full_state(mesh, nlev=5, *, bias_cell=None, bias_dt=0.0):
    """An MPAS comparison ColumnState: cell T/q_v/p_s + the native u_edge."""
    temp = np.full((mesh.nCells, nlev), 285.0)
    if bias_cell is not None:
        temp[bias_cell] += bias_dt
    u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
    return ColumnState(
        T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        sst_K=jnp.full((mesh.nCells,), 290.0),
        u_edge=jnp.asarray(u_edge))


def test_make_les_diagnose_fn_mpas_surface_flux_works(monkeypatch):
    """The surface-flux BC now WORKS on MPAS (iter 371, was fail-loud in 369): the
    cell-centred wind is RECONSTRUCTED from ``u_edge`` via the canonical Perot
    ``reconstruct_cell_velocity`` (REUSE), so ``--surface-flux`` + MPAS composes end-to-end
    — the surface flux IS computed (``column_surface_kinematic_fluxes`` called once) and the
    diagnosis is finite, no crash on the old ``v=None`` gather."""
    from legoesm.atmosphere.dynamics.les import column_les
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    called = {"n": 0}
    real_flux = column_les.column_surface_kinematic_fluxes

    def spy(**kw):
        called["n"] += 1
        return real_flux(**kw)

    monkeypatch.setattr(column_les, "column_surface_kinematic_fluxes", spy)

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    state = _mpas_full_state(mesh, nlev)             # carries u_edge + sst_K (290 K)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (5,)
        lat_deg = 20.0
        environment = _Env()

    diagnose_fn = make_les_diagnose_fn(
        mesh, sigma, run_les_fn=_mock_run_les,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME, surface_flux=True))
    out = diagnose_fn(_Rec(), state)                 # composes (no crash)
    assert called["n"] == 1                          # the surface flux WAS computed for MPAS
    assert bool(jnp.all(jnp.isfinite(out.K)))        # a finite eddy-diffusivity diagnosis


@pytest.mark.parametrize("feedback_strategy", ["static", "environment"])
def test_build_correction_campaign_mpas_one_round(feedback_strategy):
    """CAPSTONE (iters 73-76): the FULL MPAS pipeline composes through the REAL
    build_correction_campaign — cell compare/rank → Voronoi forcing extract (via
    the native u_edge) → mock LES → per-CELL clubb C_K feedback → re-run → finite
    bias. NON-VACUOUS: a SHEARED mock LES gives a VALID clubb_coefficient
    diagnosis, so the diagnosed C_K (clamped to bounds) actually REPLACES the
    background at the worst cell (static) / spreads via the env regression +
    produces a deploy kernel (environment) — not a no-op. Mock driver + mock LES;
    the C_K-changes-MODEL-output mechanism is iter 35/37."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _mpas_full_state(mesh, nlev)
    # reference = model with one cold-biased cell → a deterministic worst cell.
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    def build_base(cfg):           # mock: fixed MPAS state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        feedback_strategy=feedback_strategy,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst CELL spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # A per-CELL clubb C_K reached the config (the loop closed on the 1-D cell axis).
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (mesh.nCells,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the correction actually changed C_K away from the background.
    assert not np.allclose(ck, bg), "the diagnosed C_K never reached the cells"
    if feedback_strategy == "static":
        # The diagnosed value landed on the worst CELL; the rest stay background.
        assert not np.isclose(float(ck[bias_cell]), bg)
        np.testing.assert_allclose(np.delete(ck, bias_cell), bg)
    else:
        # The env-strategy round produced a transferable deploy kernel (iter 70).
        assert it.env_kernel is not None and it.env_kernel.field == "C_K"


def test_build_correction_campaign_owned_cell_mask_excludes_halo():
    """WIRING (iter 86): the ``valid_mask`` param threads through
    build_correction_campaign → compose_compare_fn → make_compare_fn, so a
    DISTRIBUTED-MPAS owned-cell mask keeps a HALO cell out of the ranking. A halo
    cell carrying the globally-LARGEST bias is masked out; the campaign instead
    spins off + corrects the worst OWNED cell — proving the mask both threads
    through AND changes the outcome (non-vacuous)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.training.compare_reanalysis import owned_cell_valid_mask

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    n_owned = 120                          # owned = [0, 120); halo = [120, 162)
    halo_worst, owned_worst = 150, 50
    bg = float(CLUBBLiteConfig().C_K)
    model_state = _mpas_full_state(mesh, nlev)
    # reference: TWO cold-biased cells — the halo one colder (worst), the owned one
    # second.  Without the mask the halo cell is worst; with it, the owned cell is.
    ref = _mpas_full_state(mesh, nlev)
    rt = np.asarray(ref.T).copy()
    rt[halo_worst] -= 12.0
    rt[owned_worst] -= 8.0
    reference = ref._replace(T=jnp.asarray(rt))
    owned_mask = owned_cell_valid_mask(jnp.arange(mesh.nCells) < n_owned)

    def build_base(cfg):
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False, valid_mask=owned_mask)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K)
    # The masked HALO cell was NEVER diagnosed (stays background); the worst OWNED
    # cell got the correction.
    assert np.isclose(float(ck[halo_worst]), bg), "a halo cell was wrongly corrected"
    assert not np.isclose(float(ck[owned_worst]), bg), "the owned worst cell was skipped"


def test_build_correction_campaign_mpas_les_budget_clusters_cells():
    """The LES-cost reduction (env clustering) works on the MPAS cell layout: 4
    worst cells in 2 distinct-SST environments + les_budget=2 → only 2 LES run
    (the cluster representatives), but all 4 cells are corrected. Exercises
    cluster_columns_by_environment over the 1-D cell manifest."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # 4 worst cells split into 2 SST environments (2 cells each) → 2 clusters.
    cold_cells, warm_cells = [10, 20], [120, 130]
    worst = cold_cells + warm_cells

    def _state(*, biased):
        temp = np.full((mesh.nCells, nlev), 285.0)
        if biased:
            for c in worst:
                temp[c] -= 6.0
        sst = np.full((mesh.nCells,), 285.0)
        sst[warm_cells] = 300.0                    # the env tag that splits clusters
        u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
        return ColumnState(
            T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
            u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
            p_s=jnp.full((mesh.nCells,), 1.0e5), sst_K=jnp.asarray(sst),
            u_edge=jnp.asarray(u_edge))

    model_state = _state(biased=False)
    reference = _state(biased=True)               # the 4 biased cells are the worst

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=4, les_budget=2,
        accept_only_if_improved=False)

    it = result.iterations[0]
    assert it.n_corrected == 4                    # all 4 worst cells corrected
    assert it.n_diagnosed == 2                    # but only 2 LES (cluster reps)
    assert np.isfinite(float(it.bias.baseline_bias))
    assert np.isfinite(float(it.bias.updated_bias))


def _gaussian_full_state(grid, nlev=5, *, bias_col=None, bias_dt=0.0):
    """A Gaussian (spectral) comparison ColumnState on the (n_lat, n_lon) grid.

    The zonal wind is DIVERGENT (``u = U cos λ``, varying with longitude) so the SH
    continuity chain in the iter-83 Gaussian extractor produces a genuinely NONZERO
    ω — a solid-body (``u = U cos φ``) field is non-divergent (∇·v = 0) and would
    leave ω ≈ 0, making the spectral path vacuously exercised.  T/q_v/p_s are
    uniform with one optionally cold-biased column for a deterministic worst column;
    the winds are identical in model and reference, so the worst-column ranking is
    driven purely by the T bias.  ALL arrays are float64 (the Gaussian extractor
    hard-raises on a float32 ``T``)."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    temp = np.full((n_lat, n_lon, nlev), 285.0)
    if bias_col is not None:
        temp[bias_col] += bias_dt
    lon = np.asarray(grid.grid_lon)[:, :, None]    # (n_lat, n_lon, 1)
    u = 15.0 * np.cos(lon) * np.ones((n_lat, n_lon, nlev))   # divergent → ω ≠ 0
    return ColumnState(
        T=jnp.asarray(temp, dtype=jnp.float64),
        q_v=jnp.full((n_lat, n_lon, nlev), 6e-3, dtype=jnp.float64),
        u=jnp.asarray(u, dtype=jnp.float64),
        v=jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64),
        p_s=jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64),
        sst_K=jnp.full((n_lat, n_lon), 290.0, dtype=jnp.float64))


def test_build_correction_campaign_gaussian_one_round(monkeypatch):
    """CAPSTONE (iters 83-84): the FULL Gaussian/spectral pipeline composes through
    the REAL build_correction_campaign — 2-D (n_lat, n_lon) compare/rank → the
    iter-83 SH-divergence forcing extract (``extract_column_forcing_gaussian`` via
    the GaussianGrid dispatch branch, col_index=(i_lat, i_lon)) → sheared mock LES →
    per-COLUMN clubb C_K feedback → re-run → finite bias.

    NON-VACUOUS, each property MACHINE-CHECKED rather than inferred:
      1. A SPY wraps ``extract_column_forcing_gaussian`` and asserts the spectral
         branch fired EXACTLY ONCE, for the BIASED worst column, producing a FINITE,
         NONZERO ω (the SH continuity chain genuinely ran — the divergent wind makes
         ω ≠ 0).  The lat-lon FD extractor is monkeypatched to RAISE, so a silent
         FD fallback is impossible (Codex iter-85 issues 1+2).
      2. A SHEARED mock LES gives a VALID clubb_coefficient diagnosis, so the
         diagnosed C_K actually REPLACES the background at the worst column and the
         OTHER columns stay at the background (Codex iter-85 issue 4).
      3. The float64 ``T`` survives the campaign into the extractor's float64 guard
         (a float32 ``T`` would raise; the other fields are widened to T's dtype).
    Mock driver + mock LES; the C_K-changes-model-output mechanism is iter 35/37."""
    from legoesm.atmosphere.forcing import column_large_scale_extract as clse
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # mid-grid, off the poles AND off the u=15cos(λ) divergence nodes (λ=0,π at
    # i_lon 0,22) so the SH continuity chain produces a genuinely nonzero ω there.
    bias_col = (11, 10)
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _gaussian_full_state(grid, nlev)
    # reference = model with one cold-biased column → a deterministic worst column.
    reference = _gaussian_full_state(grid, nlev, bias_col=bias_col, bias_dt=-6.0)

    # ROUTING + FORCING probe (Codex iter-85): the dispatcher resolves both
    # extractors as module globals, so patch them on the module the dispatcher reads.
    real_gaussian = clse.extract_column_forcing_gaussian
    calls: dict = {"n": 0}

    def _spy_gaussian(*, col_index, **kw):
        out = real_gaussian(col_index=col_index, **kw)
        omega = np.asarray(out.omega)
        assert np.all(np.isfinite(omega)), "spectral extractor produced non-finite ω"
        assert float(np.max(np.abs(omega))) > 1e-4, "ω is vacuously ~0 (no divergence)"
        calls["n"] += 1
        calls["col_index"] = tuple(int(c) for c in col_index)
        return out

    def _no_latlon(**kw):  # noqa: ARG001
        raise AssertionError("lat-lon FD extractor called for a GaussianGrid")

    monkeypatch.setattr(clse, "extract_column_forcing_gaussian", _spy_gaussian)
    monkeypatch.setattr(clse, "extract_column_forcing_latlon", _no_latlon)

    def build_base(cfg):           # mock: fixed Gaussian state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.asarray(grid.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst COLUMN spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # ROUTING: the SPECTRAL branch fired exactly once, for the biased worst column
    # (the lat-lon FD extractor would have raised — proven not taken).
    assert calls["n"] == 1, "the Gaussian/spectral extractor was not the path taken"
    assert calls["col_index"] == bias_col, "ranked the wrong column as worst"
    # A per-column clubb C_K reached the config, flattened on the (n_lat*n_lon) axis.
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (grid.n_lat * grid.n_lon,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the diagnosed C_K landed on the worst column, others stay at bg.
    flat_worst = bias_col[0] * grid.n_lon + bias_col[1]
    assert not np.isclose(ck[flat_worst], bg), "diagnosed C_K never reached the column"
    np.testing.assert_allclose(np.delete(ck, flat_worst), bg)  # rest untouched


def _mock_run_les_sheared(setup):
    """Mock plane-LES with a mean-wind shear so the clubb_coefficient diagnosis
    yields a VALID dimensionless C_K (the rest-state mock has no shear)."""
    state = _mock_run_les(setup)
    z = jnp.asarray(setup.height_coord.z_full)
    ny, nx = setup.grid.ny, setup.grid.nx
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))   # constant shear
    return state._replace(u=state.u.replace(data=u))


def test_build_correction_campaign_clubb_coefficient_method():
    """The dimensionless clubb_coefficient diagnosis is wired end-to-end: the
    campaign auto-populates l_mix_max from the GCM config, the LES diagnoses a
    DIMENSIONLESS C_K, the loop reduces it (same method), and a per-column,
    in-bounds C_K is produced — proving the units-correct path runs."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)            # wiring test (no improvement signal)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    assert ck.shape == (8 * 16,)
    # clip_to_bounds default ON ⇒ the dimensionless C_K stays in (0.1, 1.2).
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2


def test_campaign_auto_populates_clubb_l_mix_max_from_tuned_config(monkeypatch):
    """The diagnosis ℓ MUST use the GCM's ACTUAL ``l_mix_max`` — ``C_K = K_m/(ℓ·√wp2)``
    is the exact inverse of clubb_lite's ``K_m = C_K·ℓ·√wp2`` ONLY if both use the
    same mixing length, so a hardcoded ``l_mix_max`` would BIAS the diagnosed C_K
    whenever the GCM's value is tuned away from the 100.0 default.  The sibling wiring
    test uses the DEFAULT 100.0, so it cannot distinguish 'auto-populated from the
    config' from a hardcoded 100.0.  Here a TUNED ``initial_clubb.l_mix_max = 175.0``
    must reach ``make_les_diagnose_fn``'s ``les_config.clubb_l_mix_max`` — proving the
    campaign threads the real GCM mixing length into the parameter estimation.  (Stub
    the harness's diagnose builder + the run so this is a fast wiring assertion.)"""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    tuned = 175.0                                    # ≠ the 100.0 default ⇒ non-vacuous
    assert tuned != float(CLUBBLiteConfig().l_mix_max)
    captured = {}

    def spy_make_les_diagnose_fn(*_a, les_config, **_kw):
        captured["l_mix_max"] = les_config.clubb_l_mix_max
        return lambda record, ctx: None             # dummy diagnose_fn (never called)

    monkeypatch.setattr(rcc, "make_les_diagnose_fn", spy_make_les_diagnose_fn)
    monkeypatch.setattr(rcc, "run_correction_campaign", lambda *_a, **_kw: "SENTINEL")

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    out = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,         # noqa: ARG005
        reference=model_state, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        initial_clubb=CLUBBLiteConfig(l_mix_max=tuned))

    assert out == "SENTINEL"                         # the stubbed run was reached
    # The TUNED GCM mixing length (not the 100.0 default) reached the diagnosis.
    assert captured["l_mix_max"] == pytest.approx(tuned)


def test_campaign_threads_phis_to_make_les_diagnose_fn(monkeypatch):
    """build_correction_campaign(phis=...) must thread the model's STATIC topography
    THROUGH _build_campaign_harness to make_les_diagnose_fn, so the orographic
    geostrophic LES forcing uses the real terrain.  ``resolve_orographic_phis`` and the
    ``make_les_diagnose_fn`` → ``process_column`` hop are tested separately, but the
    CAMPAIGN-level pass-through is not — a dropped ``phis`` here would SILENTLY flatten
    the orographic forcing over terrain (no error, just a wrong shear ⇒ biased
    shear-based C_K).  Non-vacuous: ``phis=None`` is exactly the dropped state, so
    passing a real (non-None) terrain and asserting identity catches a drop."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    captured = {}

    def spy_make_les_diagnose_fn(*_a, phis=None, **_kw):
        captured["phis"] = phis
        return lambda record, ctx: None

    monkeypatch.setattr(rcc, "make_les_diagnose_fn", spy_make_les_diagnose_fn)
    monkeypatch.setattr(rcc, "run_correction_campaign", lambda *_a, **_kw: "SENTINEL")

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    phis = jnp.linspace(0.0, 5.0e4, 8 * 16).reshape(8, 16)   # non-trivial g·z_s terrain
    out = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,         # noqa: ARG005
        reference=model_state, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, phis=phis)

    assert out == "SENTINEL"
    # The real terrain reached the diagnosis builder (a dropped pass-through ⇒ None).
    assert captured["phis"] is phis


def test_campaign_threads_env_scales_to_run(monkeypatch):
    """build_correction_campaign(env_scales=...) must thread the clustering env-feature
    weights THROUGH to run_correction_campaign (→ _diagnose_columns →
    cluster_columns_by_environment).  The clustering's USE of env_scales is unit-tested
    (`test_env_scales_changes_representative`), but the CAMPAIGN-level pass-through is
    not — a dropped hop would SILENTLY revert to default (equal-weight) clustering,
    ignoring an operator who weighted e.g. SST to group columns by their dominant
    regime.  Non-vacuous: a non-default tuple ≠ the None default.  (Completes the
    campaign pass-through review: coordinate / l_mix_max / phis / env_scales.)"""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    captured = {}

    def stub_run_correction_campaign(*_a, **kw):
        captured["env_scales"] = kw.get("env_scales", "MISSING")
        return "SENTINEL"

    monkeypatch.setattr(rcc, "run_correction_campaign", stub_run_correction_campaign)

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    scales = (1.0, 1000.0, 1.0)                       # ≠ None default ⇒ non-vacuous
    out = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,         # noqa: ARG005
        reference=model_state, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, env_scales=scales)

    assert out == "SENTINEL"
    # The operator's env-feature weights reached the run (a dropped hop ⇒ None).
    assert captured["env_scales"] == scales


def test_build_correction_campaign_rejects_multi_methods():
    """build_correction_campaign is single-coefficient: a diagnosis_methods config
    (which makes process_column return a dict) is rejected up front, not crashed
    downstream (the simultaneous multi-coefficient campaign is not wired here)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="diagnosis_methods"):
        build_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(
                regime=_SMALL_REGIME,
                diagnosis_methods=("clubb_coefficient", "prandtl_number"),
                clubb_l_mix_max=100.0),
            run_les_fn=_mock_run_les_sheared, n_worst=1)


def test_build_multi_correction_campaign_mpas_three_coefficients():
    """The SIMULTANEOUS multi-coefficient campaign composes on the MPAS cell
    layout: C_K + Pr_t + C_eps co-corrected from ONE Voronoi LES spin-off per
    worst CELL (via the native u_edge through the SAME make_les_diagnose_fn) →
    three in-bounds per-CELL (nCells,) fields. Exercises run_multi_correction_*
    + the {method: diagnosis} dict path for MPAS."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=37, bias_dt=-6.0)

    result = build_multi_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ck.shape == (mesh.nCells,) and prt.shape == (mesh.nCells,)
    assert ceps.shape == (mesh.nCells,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2       # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5     # Pr_t bounds
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6  # C_eps bounds


def test_multi_campaign_valid_mask_restricts_the_ranked_columns():
    """The multi-coefficient campaign USES the valid_mask (iter 466 forwards it; this confirms
    it RESTRICTS): masking out the biased cell removes it from the worst-column ranking AND the
    bias metric, so the campaign's baseline bias drops vs the unmasked run — proving the mask
    actually restricts the multi path, not just gets accepted."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=37, bias_dt=-6.0)

    def _run(valid_mask):
        return build_multi_correction_campaign(
            base_atm_config=_mpas_base_config(nlev),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=reference, sigma=sigma, grid=mesh,
            area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1,
            coefficients=("C_K",), accept_only_if_improved=False, valid_mask=valid_mask)

    b_full = float(_run(None).iterations[0].bias.baseline_bias)
    # exclude the single biased cell (37) — it should drop out of the bias metric + ranking
    mask = jnp.ones(mesh.nCells, dtype=bool).at[37].set(False)
    b_masked = float(_run(mask).iterations[0].bias.baseline_bias)
    assert b_full > b_masked                       # masking the biased cell lowers the bias
    assert b_masked < 0.5 * b_full                 # most of the bias was at the masked cell


def test_build_correction_campaign_prandtl_number_method():
    """The prandtl_number diagnosis is wired end-to-end: the campaign selects the
    clubb_lite_Pr_t promotion + Pr_t background by method, the LES diagnoses a
    DIMENSIONLESS Pr_t, and an in-bounds per-column Pr_t is produced."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="prandtl_number"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    # The corrected coefficient is Pr_t (not C_K); C_K stays the scalar default.
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert prt.shape == (8 * 16,)
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched
    assert result.final_config.C_K == CLUBBLiteConfig().C_K


def test_build_correction_campaign_bias_tol_early_stops():
    """build_correction_campaign forwards bias_tol/patience: a mock driver that
    never improves the bias is rejected every round, so the campaign stops early
    ('converged') instead of running all n_iterations. Uses diagnosis_method=
    'clubb_coefficient' (which the sheared mock LES diagnoses VALIDLY) so the rounds
    genuinely RUN + get rejected — not the dry-LES abort path (iter 101)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # fixed → never improves
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=10,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        bias_tol=1e-9, patience=2)            # default gate ON
    assert result.stop_reason == "converged"
    # sanity: the rounds genuinely produced VALID diagnoses (so this exercises the
    # convergence path, not the dry-LES abort).
    assert all(it.n_diagnoses_valid > 0 for it in result.iterations)
    assert len(result.iterations) == 2        # 2 rejected rounds → early stop


def test_build_correction_campaign_default_gate_rejects_non_improving():
    """build_correction_campaign defaults the monotonic gate ON: the mock driver
    returns a fixed state regardless of C_K, so the round does not lower the bias
    and is REJECTED — the config stays the uncorrected scalar default."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1)            # default gate ON

    assert result.accepted == (False,)                  # non-improving → rejected
    assert not bool(result.iterations[0].bias.improved)
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0  # unchanged default


def test_make_base_driver_builder_dispatch():
    """The run-mode dispatch returns the right SST extractor + raises on a bad
    mode / a CMIP call missing the coupled pieces (dispatch hardening, iter 40)."""
    from legoesm.training.run_to_column_mean import (
        amip_column_state,
        cmip_column_state,
    )

    _build_amip, extract_amip = make_base_driver_builder("amip")
    assert extract_amip is amip_column_state
    assert callable(_build_amip)

    # cmip needs ONLY coupled_preset; ocean_grid is optional (None ⇒ the coupled
    # driver uses its own atm grid, same-grid coupling).
    _build_cmip, extract_cmip = make_base_driver_builder(
        "cmip", coupled_preset=object())
    assert extract_cmip is cmip_column_state
    assert callable(_build_cmip)

    with pytest.raises(ValueError, match="unknown mode"):
        make_base_driver_builder("xyz")
    with pytest.raises(ValueError, match="requires coupled_preset"):
        make_base_driver_builder("cmip")  # missing coupled_preset


@pytest.mark.slow
def test_make_base_driver_builder_amip_builds_real_driver():
    """The AMIP builder constructs + sets up a real ModelDriver (the campaign's
    build_base_driver path that main() uses)."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.model_driver import ModelDriver

    build_amip, _ = make_base_driver_builder("amip")
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")
    driver = build_amip(cfg)
    assert isinstance(driver, ModelDriver)
    assert driver.grid is not None and driver.sigma is not None


@pytest.mark.slow
def test_make_base_driver_builder_cmip_builds_real_driver():
    """The CMIP builder constructs + sets up a real CoupledESMDriver with
    ocean_grid=None (same-grid coupling: ocean on the atm grid), exposing the
    state/q_v/ocean_state that cmip_column_state reads.

    Also LOCKS the CMIP DEPLOY propagation (the done-criterion's CMIP half): a
    turbulence_override on the atm config must reach the COUPLED atmosphere — the
    coupled driver builds its atmosphere from the SAME atm_config (CoupledESMDriver:
    ``self._atm = ModelDriver(atm_config)``), NOT from coupled_preset (ocean/coupling
    only), so the LES-informed correction is NOT silently dropped in coupled mode."""
    from legoesm.atmosphere.physics.turbulence.config import (
        CLUBBLiteConfig,
        TurbulenceConfig,
    )
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    build_cmip, _ = make_base_driver_builder(
        "cmip", coupled_preset=PRESETS["aquaplanet"](), ocean_grid=None)
    # a deployed-shape per-column override (the real corrected_turbulence_override form).
    override = TurbulenceConfig(
        scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=jnp.full((8, 16), 0.77)))
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite", turbulence_override=override)
    driver = build_cmip(cfg)
    assert isinstance(driver, CoupledESMDriver)
    # cmip_column_state reads state + q_v from the coupled driver, and the SST via the
    # public get_sst_sic accessor (iter 335 — the atm-grid coupled SST, the same path AMIP
    # uses), NOT the raw ocean_state.T_sfc.
    assert driver.state is not None and driver.q_v is not None
    assert tuple(driver.ocean_state.T_sfc.data.shape) == (8, 16)
    # DIRECT test of the iter-335 public get_sst_sic on the REAL coupled driver: it returns
    # the coupled SST on the ATMOSPHERE grid (same-grid here ⇒ atm column shape), a physical
    # ocean temperature — so the SST env tag always aligns with the atm columns.
    sst, _sic = driver.get_sst_sic(0.0)
    sst_arr = jnp.asarray(getattr(sst, "data", sst))
    assert tuple(sst_arr.shape) == (8, 16)                     # atm-grid, matches columns
    assert bool(jnp.all(jnp.isfinite(sst_arr)))
    assert 200.0 < float(jnp.mean(sst_arr)) < 350.0           # a physical SST [K]
    # CMIP DEPLOY PROPAGATION: the override survived onto the coupled atmosphere config.
    assert driver.atm_config.turbulence_override is override
    assert float(driver.atm_config.turbulence_override.clubb_lite.C_K[0, 0]) == 0.77


@pytest.mark.slow
def test_build_correction_campaign_checkpoint_passthrough():
    """build_correction_campaign forwards checkpoint_callback + start_round to
    run_correction_campaign (the restart wiring main() uses)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    calls = []

    def ckpt(round_idx, res, field):
        calls.append((round_idx, np.asarray(field).copy(), res))

    # initial_field (resume base): every non-worst column keeps this value, so the
    # checkpointed accumulated field proves initial_field was forwarded + used.
    # Gate OFF: this is the forwarding/wiring test (the gate is tested separately),
    # so the round is accepted and the worst column gets a correction on top of base.
    init_field = jnp.full((8, 16), 0.55)
    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1,
        initial_field=init_field, start_round=7, checkpoint_callback=ckpt,
        accept_only_if_improved=False)

    assert len(calls) == 1
    round_idx, field, res = calls[0]
    assert round_idx == 7            # start_round forwarded; callback fired once
    flat = field.reshape(-1)
    worst = 4 * 16 + 8               # the +6 K worst column (row-major)
    others = np.delete(flat, worst)
    # Non-worst columns keep initial_field (0.55), NOT the default background 0.4
    # — so initial_field was genuinely forwarded + used as the round's base.
    np.testing.assert_allclose(others, 0.55)
    # Resume contract (the CLI checkpoint fix relies on this): the persisted
    # accumulated field IS the accepted state and its flattened form equals the
    # accepted config's per-column C_K — so rebuilding the config FROM the field on
    # resume cannot desync. (Accepted round ⇒ res.updated_config.C_K matches too.)
    np.testing.assert_allclose(flat, np.asarray(res.updated_config.C_K))
    np.testing.assert_allclose(
        flat, np.asarray(result.final_config.C_K).reshape(-1))


@pytest.mark.slow
def test_build_correction_campaign_mpas_resume_accumulates_on_cells():
    """The restartable campaign (§1) works on the MPAS cell layout: an
    initial_field (nCells,) is the round-0 base, so every non-worst CELL keeps it
    and the worst CELL is corrected ON TOP — the (ncol,)-field accumulation +
    grid.grid_shape_2d=(nCells,) resume path is grid-agnostic."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    calls = []
    init_field = jnp.full((mesh.nCells,), 0.55)       # the resume base
    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        initial_field=init_field, start_round=7,
        checkpoint_callback=lambda r, res, f: calls.append((r, np.asarray(f).copy())),
        accept_only_if_improved=False)

    assert len(calls) == 1 and calls[0][0] == 7       # start_round forwarded
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (mesh.nCells,)
    # Non-worst cells keep the resume base; the worst cell is corrected on top.
    np.testing.assert_allclose(np.delete(ck, bias_cell), 0.55)
    assert not np.isclose(float(ck[bias_cell]), 0.55)


@pytest.mark.slow
def test_build_correction_campaign_environment_strategy():
    """feedback_strategy='environment' runs end-to-end through the campaign
    (column_environment_grid computes the full-grid env each round) → a
    per-column C_K."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1, feedback_strategy="environment",
        accept_only_if_improved=False)  # wiring test; gate tested separately

    assert len(result.iterations) == 1
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (8 * 16,)             # per-column (env-generalized) field
    assert np.all(np.isfinite(ck))


def test_build_multi_correction_campaign_rejects_unknown_and_empty_coefficients():
    """build_multi_correction_campaign fails LOUD on an UNKNOWN (typo'd) or EMPTY
    --coefficients (dispatch hardening) — a bad coefficient name is NOT silently dropped,
    so the simultaneous campaign corrects EXACTLY the requested set or refuses. The
    validation fires BEFORE the heavy build, so minimal fake args suffice. (Confirms the
    iter-173 'silent filter' concern was a FALSE alarm: _run_multi_main passes the FULL
    tuple — typo included — to this builder, which validates it.)"""
    from types import SimpleNamespace

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig

    # The unknown-coefficient guard fires AFTER grid_shape (reference.T.shape) and the
    # lat/lon defaulting, so the fake reference exposes .T.shape and lat_deg/lon_deg are
    # passed explicitly (skipping grid.grid_lat). validate_reference=False skips the
    # default-on physical check. The empty-tuple guard fires FIRST, before any of these.
    fake_ref = SimpleNamespace(T=SimpleNamespace(shape=(2, 3, 5)))
    common = dict(
        base_atm_config=_base_config(), build_base_driver=(lambda c: None),
        extract_column_state=(lambda d, day, dt: None), reference=fake_ref,
        sigma=object(), grid=object(), area_weights=jnp.ones((1, 1)),
        run_les_fn=(lambda s: None), n_iterations=1,
        les_config=ColumnLESConfig(), n_worst=1, validate_reference=False,
        lat_deg=jnp.zeros((2, 3)), lon_deg=jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_correction_campaign(**common, coefficients=("C_K", "bogus"))
    with pytest.raises(ValueError, match="non-empty tuple"):
        build_multi_correction_campaign(**common, coefficients=())
    # A DUPLICATE (e.g. --coefficients C_K,C_K, a typo) doubles the LES cost / breaks the
    # distributed path → fail loud at construction (iter 251). The empty guard fires first.
    with pytest.raises(ValueError, match=r"duplicate\(s\) \['C_K'\]"):
        build_multi_correction_campaign(**common, coefficients=("C_K", "Pr_t", "C_K"))


def test_build_multi_correction_campaign_corrects_both_coefficients():
    """build_multi_correction_campaign co-corrects C_K AND Pr_t from ONE LES run
    per column: it sets diagnosis_methods, auto-populates l_mix_max, and the multi
    campaign produces in-bounds per-column fields for BOTH coefficients."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t"), accept_only_if_improved=False)

    assert set(result.final_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert ck.shape == (8 * 16,) and prt.shape == (8 * 16,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2     # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds


def test_multi_campaign_auto_populates_clubb_l_mix_max_from_tuned_config(monkeypatch):
    """The MULTI-coefficient path's l_mix_max threading guard (the iter-379 single-path
    analog for ``build_multi_correction_campaign``, line ~767).  C_eps's diagnosis,
    like C_K's, evaluates the GCM mixing length, so a hardcoded ``l_mix_max`` would
    bias BOTH co-corrected coefficients whenever the GCM value is tuned.  The
    'corrects_both_coefficients' wiring test uses the DEFAULT 100.0 (vacuous).  Here a
    TUNED ``initial_clubb.l_mix_max=175.0`` with ``coefficients=("C_K","C_eps")`` (both
    need ℓ) must reach ``make_les_diagnose_fn``'s ``les_config.clubb_l_mix_max``."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    tuned = 175.0                                    # ≠ the 100.0 default ⇒ non-vacuous
    assert tuned != float(CLUBBLiteConfig().l_mix_max)
    captured = {}

    def spy_make_les_diagnose_fn(*_a, les_config, **_kw):
        captured["l_mix_max"] = les_config.clubb_l_mix_max
        return lambda record, ctx: None

    import legoesm.training.correction_loop as cl

    monkeypatch.setattr(rcc, "make_les_diagnose_fn", spy_make_les_diagnose_fn)
    # build_multi_correction_campaign imports run_multi_correction_campaign at call
    # time (function-scope), so patch the SOURCE module, not the rcc namespace.
    monkeypatch.setattr(cl, "run_multi_correction_campaign",
                        lambda *_a, **_kw: "SENTINEL")

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    out = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,         # noqa: ARG005
        reference=model_state, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "C_eps"),               # both evaluate the GCM mixing length
        initial_clubb=CLUBBLiteConfig(l_mix_max=tuned))

    assert out == "SENTINEL"
    assert captured["l_mix_max"] == pytest.approx(tuned)


def test_build_correction_campaign_c_eps_single_method():
    """Single-coefficient c_eps is wired: build_correction_campaign selects the
    clubb_lite_C_eps promotion + C_eps background + auto-populates l_mix_max."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME, diagnosis_method="c_eps"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched


def test_build_multi_correction_campaign_three_coefficients_with_c_eps():
    """C_K + Pr_t + C_eps co-corrected from one LES run; C_eps closes the wp2-
    identification gap. All three end up in-bounds per-column fields."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6   # C_eps bounds


def test_build_multi_correction_campaign_sequential_staged():
    """sequential=True routes the staged (block-coordinate-descent) mode through
    build_multi_correction_campaign; per-coefficient fractions are reported."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_eps", "C_K", "Pr_t"), sequential=True,
        accept_only_if_improved=False)
    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    assert result.iterations[0].step_fractions_by_key is not None


def test_build_multi_correction_campaign_rejects_unknown_coefficient():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1,
            coefficients=("C_K", "bogus"))


# --- Distributed multi-coefficient campaign wrapper (iter 95) ----------------
def _mock_layout(owned, local_cells, n_global=None):
    lc = np.asarray(local_cells)
    if n_global is None:                              # default: tightest mesh spanning lc
        n_global = int(lc.max()) + 1 if lc.size else 0
    return SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned),
        partition=SimpleNamespace(local_cells=lc, nCells_global=n_global))


def _global_cell_state(ncells, nlev=4):
    # PHYSICALLY plausible values (the distributed kwargs now pre-validate the global
    # reference via validate_reference_physical) — distinct per cell for slice checks.
    rng = np.arange(ncells * nlev, dtype=float).reshape(ncells, nlev)
    return ColumnState(
        T=jnp.asarray(280.0 + 0.01 * rng), q_v=jnp.full((ncells, nlev), 5.0e-3),
        u=jnp.zeros((ncells, nlev)), v=jnp.zeros((ncells, nlev)),
        p_s=jnp.full((ncells,), 1.0e5) + jnp.arange(float(ncells)),
        sst_K=jnp.arange(float(ncells)) + 290.0)


def test_distributed_campaign_kwargs_composes_hooks_and_slices():
    """``_distributed_campaign_kwargs`` (the shared composition for BOTH the single +
    multi distributed wrappers) builds the three hooks from the layout AND slices the
    GLOBAL reference + area_weights to the rank's local cells — order-preserving."""
    from functools import partial

    from legoesm.parallel.reductions import global_sum_mpi
    from legoesm.training.distributed_manifest import gather_global_worst_columns

    # 4 global cells; this rank's local_cells = [2, 3, 1] (owned 2,3; halo 1).
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    area = jnp.arange(4.0) + 1.0
    kw = _distributed_campaign_kwargs(layout, ref, area, n_worst=1, base_valid_mask=None)
    np.testing.assert_array_equal(np.asarray(kw["valid_mask"]), [True, True, False])
    assert kw["global_reduce"] is global_sum_mpi
    assert isinstance(kw["manifest_reducer"], partial)
    assert kw["manifest_reducer"].func is gather_global_worst_columns
    assert kw["n_worst"] == 1
    # reference + area sliced to local_cells [2,3,1] (order preserved).
    np.testing.assert_array_equal(
        np.asarray(kw["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    np.testing.assert_array_equal(np.asarray(kw["area_weights"]), [3.0, 4.0, 2.0])
    # the GLOBAL reference is pre-validated here, so the inner builder skips the
    # per-slice re-validation (collective-safe — iter 99).
    assert kw["validate_reference"] is False


def test_distributed_campaign_kwargs_validates_global_reference_units():
    """``_distributed_campaign_kwargs`` pre-validates the GLOBAL reference for a
    units/sign error (identical on every rank ⇒ collective-safe) BEFORE slicing, so a
    distributed run fails fast instead of correcting against a fake bias (iter 99)."""
    layout = _mock_layout([True, True, False], [2, 3, 1])
    bad = _global_cell_state(4)._replace(
        p_s=jnp.full((4,), 1013.0))                  # hPa, not Pa
    with pytest.raises(ValueError, match=r"global reference.*p_s outside"):
        _distributed_campaign_kwargs(
            layout, bad, jnp.ones(4), n_worst=1, base_valid_mask=None)


def test_distributed_campaign_kwargs_shape_check_precedes_physical():
    """The DETERMINISTIC reference-shape check runs BEFORE the value-dependent physical
    check: a mis-passed rank-LOCAL reference (wrong cell count) AND a bad physical value
    fails on the SHAPE (identical on every rank ⇒ collective-safe), not the physical
    bound (which could differ per rank) — iter 99 Codex ordering fix."""
    layout = _mock_layout([True, True, False], [2, 3, 1])   # n_global defaults to 4
    # a 3-cell reference (wrong shape) that ALSO has a units-bad p_s.
    local_ref = _global_cell_state(3)._replace(p_s=jnp.full((3,), 1013.0))
    with pytest.raises(ValueError, match="has 3 cells but the partitioned mesh has 4"):
        _distributed_campaign_kwargs(
            layout, local_ref, jnp.ones(4), n_worst=1, base_valid_mask=None)


def test_distributed_campaign_kwargs_rejects_mesh_mismatch():
    """A GLOBAL area_weights / reference whose cell-count ≠ the partitioned mesh is
    REJECTED LOUDLY — the JAX gather would otherwise silently clamp out-of-range cell
    ids (too-short) or mis-align (wrong mesh), corrupting the rank's compare/weights
    on a multi-day run (iter 97; EXACT-N vs nCells_global also catches too-LONG)."""
    layout = _mock_layout([True, True, False], [2, 3, 4], n_global=5)
    ref5 = _global_cell_state(5)
    # area_weights shorter than the global mesh (5).
    with pytest.raises(ValueError, match="area_weights has 4 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(4), n_worst=1, base_valid_mask=None)
    # area_weights LONGER than the global mesh — exact check (a bounds check misses this).
    with pytest.raises(ValueError, match="area_weights has 6 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(6), n_worst=1, base_valid_mask=None)
    # a non-1-D area_weights is rejected (must be per-cell).
    with pytest.raises(ValueError, match="must be 1-D"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones((5, 2)), n_worst=1, base_valid_mask=None)
    # reference cell-count ≠ the mesh (here a reference for 6 cells, mesh has 5) —
    # caught by the deterministic shape check that precedes the physical check.
    with pytest.raises(ValueError, match="reference has 6 cells.*partitioned mesh has 5"):
        _distributed_campaign_kwargs(
            layout, _global_cell_state(6), jnp.ones(5), n_worst=1, base_valid_mask=None)


def test_build_distributed_multi_forwards_composed_kwargs(monkeypatch):
    """``build_distributed_multi_correction_campaign`` forwards the composed
    distributed kwargs (the three hooks + the rank-local reference/area slice) AND the
    multi-coefficient ``campaign_kwargs`` (e.g. ``coefficients``) to
    ``build_multi_correction_campaign`` — the multi sibling of the iter-89 single
    wrapper, sharing the SAME composition (no MPI: only the wiring is exercised)."""
    captured = {}

    def _fake_multi(**kwargs):
        captured.update(kwargs)
        return "MULTI_RESULT"

    monkeypatch.setattr(
        "scripts.run.run_correction_campaign.build_multi_correction_campaign",
        _fake_multi)
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    out = build_distributed_multi_correction_campaign(
        layout=layout, reference=ref, area_weights=jnp.arange(4.0) + 1.0, n_worst=1,
        base_atm_config="CFG", coefficients=("C_K", "Pr_t", "C_eps"))
    assert out == "MULTI_RESULT"
    # the distributed hooks + rank-local slice reached build_multi...
    np.testing.assert_array_equal(np.asarray(captured["valid_mask"]), [True, True, False])
    assert captured["manifest_reducer"] is not None and captured["global_reduce"] is not None
    np.testing.assert_array_equal(
        np.asarray(captured["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    # ...and the multi-only kwargs pass through.
    assert captured["coefficients"] == ("C_K", "Pr_t", "C_eps")
    assert captured["base_atm_config"] == "CFG"


def test_build_distributed_multi_rejects_duplicate_hook_kwarg():
    """A caller cannot set the distributed hooks inconsistently — passing one in
    campaign_kwargs collides with the supplied one (TypeError)."""
    layout = _mock_layout([True, True], [0, 1])
    ref = _global_cell_state(2)
    with pytest.raises(TypeError):
        build_distributed_multi_correction_campaign(
            layout=layout, reference=ref, area_weights=jnp.ones(2), n_worst=1,
            valid_mask=jnp.ones(2, dtype=bool))   # duplicate of the supplied hook


@pytest.mark.parametrize(
    "multi, target",
    [(False, "build_distributed_correction_campaign"),
     (True, "build_distributed_multi_correction_campaign")])
def test_build_distributed_mpas_campaign_wires_local_mesh(monkeypatch, multi, target):
    """``build_distributed_mpas_campaign`` (iter 96, the one-call RUNNABLE entry point)
    partitions the GLOBAL mesh and forwards the rank-LOCAL mesh as BOTH the campaign
    ``grid`` and the bound ``build_base_driver`` — and dispatches to the single vs
    multi distributed wrapper on ``multi`` (no MPI: the layout build + both wrappers
    are monkeypatched, only the wiring is exercised)."""
    import scripts.run.run_correction_campaign as rcc

    layout = SimpleNamespace(local_mesh="LOCAL_MESH")
    seen_global = {}

    def _fake_make_layout(global_mesh, rank, n_ranks):
        seen_global.update(global_mesh=global_mesh, rank=rank, n_ranks=n_ranks)
        return layout

    monkeypatch.setattr(
        "legoesm.parallel.voronoi_mpi.make_voronoi_partition_layout", _fake_make_layout)

    captured = {}

    def _fake_wrapper(**kwargs):
        captured.update(kwargs)
        return "WRAPPED"

    # patch BOTH wrappers; only ``target`` should actually be called.
    for name in ("build_distributed_correction_campaign",
                 "build_distributed_multi_correction_campaign"):
        monkeypatch.setattr(rcc, name,
                            _fake_wrapper if name == target else _boom_wrapper)

    driver_calls = []

    def _build_local_driver(cfg, local_mesh):
        driver_calls.append((cfg, local_mesh))
        return "DRIVER"

    out = rcc.build_distributed_mpas_campaign(
        global_mesh="GMESH", rank=0, n_ranks=1, reference="REF", area_weights="AREA",
        n_worst=2, build_local_driver=_build_local_driver, multi=multi,
        validate_partition=False,                    # mock layout + no MPI collective
        base_atm_config="CFG")

    assert out == "WRAPPED"
    assert seen_global == dict(global_mesh="GMESH", rank=0, n_ranks=1)
    # the rank-LOCAL mesh (not the global one) is wired as the grid + into the driver.
    assert captured["grid"] == "LOCAL_MESH"
    assert captured["layout"] is layout
    assert captured["reference"] == "REF" and captured["n_worst"] == 2
    assert captured["base_atm_config"] == "CFG"        # campaign_kwargs pass through
    # the bound build_base_driver binds local_mesh and forwards the user's builder.
    assert captured["build_base_driver"]("the_cfg") == "DRIVER"
    assert driver_calls == [("the_cfg", "LOCAL_MESH")]


def _boom_wrapper(**kwargs):                            # the wrapper that must NOT run
    raise AssertionError("wrong distributed wrapper dispatched")


def test_build_arg_parser_defaults():
    """The CLI parser (the HPC entry point, previously untested) has the documented
    defaults: abort/clamp/gate are ON, mode=amip (iter 102)."""
    args = _build_arg_parser().parse_args(["--config", "base.json", "--era5-zarr", "era5.zarr"])
    assert args.mode == "amip"
    assert args.iterations == 3 and args.n_worst == 20 and args.patience == 2
    assert args.bias_tol is None
    assert args.keep_dry_rounds is False          # default → dry-abort ON
    assert args.allow_unphysical_coeff is False   # default → bounds clamp ON
    assert args.keep_worsening_rounds is False    # default → monotonic gate ON
    assert args.feedback_strategy == "static"
    # --les-dt default 0.5 keeps the acoustic Courant < 1 at dx=50 m (iter 103); the
    # old 1.0 default gave C_a≈1.16 and is now rejected by the LES CFL pre-flight.
    assert args.les_dt == 0.5


def test_build_arg_parser_required_and_choices():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])                                   # --config required
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "x", "--era5-zarr", "z", "--mode", "bogus"])  # bad choice


def test_campaign_knobs_from_args_maps_flags():
    """The shared arg→build-kwargs mapping handles the boolean NEGATIONS + the CSV
    step-fractions parse correctly (an inverted flag = a silent HPC bug; iter 102)."""
    parser = _build_arg_parser()
    knobs = _campaign_knobs_from_args(parser.parse_args(["--config", "x", "--era5-zarr", "z"]))
    assert knobs["stop_on_no_valid_diagnoses"] is True    # NOT --keep-dry-rounds
    assert knobs["clip_to_bounds"] is True                # NOT --allow-unphysical-coeff
    assert knobs["accept_only_if_improved"] is True       # NOT --keep-worsening-rounds
    assert knobs["step_fractions"] is None
    assert knobs["patience"] == 2 and knobs["n_worst"] == 20 and knobs["bias_tol"] is None
    assert knobs["spinup_days"] == 0.0                    # default = keep everything

    flagged = _campaign_knobs_from_args(parser.parse_args([
        "--config", "x", "--era5-zarr", "z", "--keep-dry-rounds", "--allow-unphysical-coeff",
        "--keep-worsening-rounds", "--step-fractions", "1.0,0.5,0.25",
        "--bias-tol", "1e-3", "--patience", "5", "--n-worst", "8",
        "--feedback-strategy", "environment", "--les-budget", "4", "--spinup-days", "30"]))
    assert flagged["spinup_days"] == 30.0                 # --spinup-days reaches the build kwargs
    assert flagged["stop_on_no_valid_diagnoses"] is False
    assert flagged["clip_to_bounds"] is False
    assert flagged["accept_only_if_improved"] is False
    assert flagged["step_fractions"] == [1.0, 0.5, 0.25]
    assert flagged["bias_tol"] == pytest.approx(1e-3) and flagged["patience"] == 5
    assert flagged["n_worst"] == 8 and flagged["les_budget"] == 4
    assert flagged["feedback_strategy"] == "environment"


def test_spinup_warning_line_fires_for_multiday_run_without_exclusion():
    """A multi-day climatology with --spinup-days 0 warns the operator (the time-mean would
    include the un-equilibrated spin-up, iter 445/446); a set --spinup-days OR a short test
    run is silent (no noise on a quick smoke)."""
    from scripts.run.run_correction_campaign import _SPINUP_WARN_DAYS, _spinup_warning_line

    w = _spinup_warning_line(0.0, _SPINUP_WARN_DAYS)       # multi-day, no exclusion → warns
    assert w is not None and "spin-up" in w and f"{_SPINUP_WARN_DAYS}-day" in w
    assert _spinup_warning_line(10.0, 90) is None          # --spinup-days set → silent
    assert _spinup_warning_line(0.0, _SPINUP_WARN_DAYS - 1) is None   # short test → silent


def test_days_below_cadence_warning_fires_at_launch():
    """REGRESSION (iter 499/500): days <= the diagnostic cadence (output.diag_days) warns at
    LAUNCH — the model climatology time-mean fires no segment boundary and a REAL run fails
    loud, but the --dry-run (which never runs the time-mean) would otherwise PASS it (the
    iter-484 dry-run-false-confidence class). The launch note loop runs before the dry-run
    short-circuit, so the dry-run surfaces it too. getattr-guarded for stubbed configs."""
    from scripts.run.run_correction_campaign import _days_below_cadence_warning

    w = _days_below_cadence_warning(5, 5)                  # days == cadence → no boundary → warns
    assert w is not None and "diagnostic cadence" in w and "no segment boundary" in w
    assert _days_below_cadence_warning(3, 5) is not None   # days < cadence → warns
    assert _days_below_cadence_warning(200, 5) is None     # a real window → silent
    assert _days_below_cadence_warning(0, 5) is None       # missing/zero → silent (stub-safe)
    assert _days_below_cadence_warning(6, None) is None    # no cadence (stub) → silent


def test_run_multi_main_forwards_valid_mask_to_the_builder(monkeypatch):
    """The multi-coefficient main forwards the ocean-only valid_mask to
    build_multi_correction_campaign (iter 466 fix): it previously dropped it silently, so
    `--ocean-only --coefficients C_K,Pr_t` ranked ALL columns instead of ocean-only."""
    import jax.numpy as jnp
    import pytest

    import scripts.run.run_correction_campaign as mod

    captured = {}

    class _StopError(Exception):
        pass

    def _spy(**kwargs):
        captured.update(kwargs)
        raise _StopError()                         # stop before the dry-run report

    monkeypatch.setattr(mod, "build_multi_correction_campaign", _spy)

    grid = SimpleNamespace(grid_shape_2d=(2, 2), grid_area=jnp.ones(4))
    args = SimpleNamespace(
        coefficients="C_K,Pr_t", resume=None, checkpoint=None, iterations=1,
        staged=False, surface_flux=False, dry_run=True,
        n_worst=4, les_budget=None, feedback_strategy="static",
        keep_worsening_rounds=False, step_fractions=None, allow_unphysical_coeff=False,
        bias_tol=None, patience=2, keep_dry_rounds=False, spinup_days=0.0)
    mask = jnp.array([True, False, True, False])
    with pytest.raises(_StopError):
        mod._run_multi_main(
            args, base_cfg=object(), grid=grid, sigma=object(), reference=object(),
            build_base_driver=lambda c: object(), extract_fn=lambda *a: object(),
            run_les=object(), valid_mask=mask)
    assert captured["valid_mask"] is mask          # forwarded, not dropped


def test_surface_flux_land_warning_fires_without_ocean_only():
    """--surface-flux on a config WITH a land mask but WITHOUT --ocean-only warns (iter 465):
    a land worst-column would get a surface flux from a non-ocean SST. Silent when paired with
    --ocean-only, without a land mask (aquaplanet), or with --surface-flux off."""
    from scripts.run.run_correction_campaign import _surface_flux_land_warning

    w = _surface_flux_land_warning(True, False, True)      # land mask, no ocean-only → warns
    assert w is not None and "OCEAN" in w and "--ocean-only" in w
    assert _surface_flux_land_warning(True, True, True) is None      # paired → silent
    assert _surface_flux_land_warning(True, False, False) is None    # aquaplanet (no mask) → silent
    assert _surface_flux_land_warning(False, False, True) is None    # surface-flux off → silent


def test_offline_forcing_window_warning_fires_when_run_exceeds_one_month():
    """A run longer than the single-month offline AMIP forcing warns about the SST/insolation
    DESYNC (get_forcing_at_time cyclically repeats the month, iter 458); a sub-month window is
    silent. The AMIP default days=200 trips it."""
    from scripts.run.run_correction_campaign import (
        _OFFLINE_FORCING_SPAN_DAYS,
        _offline_forcing_window_warning,
    )

    w = _offline_forcing_window_warning(200)               # AMIP default >> 1 month → warns
    assert w is not None and "DESYNC" in w and "SINGLE-MONTH" in w
    assert _offline_forcing_window_warning(_OFFLINE_FORCING_SPAN_DAYS) is None   # fits → silent
    assert _offline_forcing_window_warning(20) is None     # sub-month → silent
    assert _offline_forcing_window_warning(None) is None   # unknown → silent
    assert _offline_forcing_window_warning(_OFFLINE_FORCING_SPAN_DAYS + 1) is not None


def test_effective_config_sidecar_persists_runtime_injections(tmp_path):
    """The campaign persists the EFFECTIVE config (base + runtime injections) as a sidecar next
    to --out (iter 464), so a runtime-flag campaign's calibration config is recorded + the
    deploy/re-run can reproduce the SST boundary + insolation the C_K was tuned for."""
    import json

    from legoesm.driver.config import ExperimentConfig, experiment_config_from_dict

    from scripts.run.run_correction_campaign import (
        _effective_config_path,
        _write_effective_config,
    )

    out = str(tmp_path / "corrected.json")
    assert _effective_config_path(out) == str(tmp_path / "corrected.effective_config.json")

    # an effective config carrying the runtime injections (forcing + insolation offset)
    cfg = ExperimentConfig(insolation_start_doy=244.0, dataset="custom",
                           forcing_path="/archive/era5_amip.nc")
    path = _write_effective_config(cfg, out)
    assert path == _effective_config_path(out)
    loaded = experiment_config_from_dict(json.load(open(path)))
    assert loaded.insolation_start_doy == 244.0          # --align-insolation injection recorded
    assert loaded.dataset == "custom"                    # --amip-forcing injection recorded
    assert loaded.forcing_path == "/archive/era5_amip.nc"


def test_offline_reference_window_warning_fires_for_short_reference_vs_long_model():
    """A SHORT offline ERA5 reference vs a multi-day model climatology warns about
    weather-vs-climate (iter 463); a well-matched window or a short test run is silent. The
    AMIP default (days=200, --era5-n-times 1) trips it."""
    from scripts.run.run_correction_campaign import _offline_reference_window_warning

    w = _offline_reference_window_warning(200, 1)          # default snapshot vs 200-day mean
    assert w is not None and "WEATHER to CLIMATE" in w
    assert _offline_reference_window_warning(30, 24) is not None     # 1 day vs 30-day model
    assert _offline_reference_window_warning(30, 720) is None        # 30 days hourly → matched
    assert _offline_reference_window_warning(30, 360) is None        # >= half the window → silent
    assert _offline_reference_window_warning(3, 1) is None           # short test → silent
    assert _offline_reference_window_warning(None, 1) is None        # unknown → silent


def test_insolation_season_note_warns_off_season_offline_date():
    """An off-season (Apr–Sep) OFFLINE ERA5 date warns about the model's January-based insolation
    mismatch and surfaces BOTH fixes: the exact ``insolation_start_doy=<doy>`` (iter 449) and the
    January-window workaround. A January window, the Zarr path (no date), or a malformed date is
    silent (no crash)."""
    from scripts.run.run_correction_campaign import _insolation_season_note

    w = _insolation_season_note("20170901")                # September → off-season → warns
    assert w is not None and "insolation" in w.lower() and "January" in w
    # the note reports Sep 1's exact noleap day-of-year (244) and the config-field fix
    assert "244" in w and "insolation_start_doy=244" in w
    assert _insolation_season_note("20170115") is None     # January → aligned → silent
    assert _insolation_season_note("20171115") is None     # November (near-January half) → silent
    assert _insolation_season_note(None) is None           # Zarr path (no offline date) → silent
    assert _insolation_season_note("bad") is None           # malformed → silent (no crash)
    assert _insolation_season_note("201709") is None        # date w/o day-of-month → silent


def test_align_insolation_sets_start_doy_from_offline_date():
    """--align-insolation derives insolation_start_doy from the offline ERA5 date (iter 450);
    OFF by default it leaves the config untouched; ON without/with a bad date fails LOUD."""
    import pytest
    from legoesm.driver.config import ExperimentConfig

    from scripts.run.run_correction_campaign import (
        _build_arg_parser,
        _maybe_align_insolation,
    )

    base = ExperimentConfig()
    assert base.insolation_start_doy is None

    # flag OFF (default) => unchanged, even with an offline date present
    off = SimpleNamespace(align_insolation=False, local_era5_date="20170901")
    assert _maybe_align_insolation(off, base) is base

    # flag ON + Sep 1 => insolation_start_doy = noleap day-of-year 244
    on = SimpleNamespace(align_insolation=True, local_era5_date="20170901")
    aligned = _maybe_align_insolation(on, base)
    assert aligned.insolation_start_doy == 244.0
    aligned.validate_strict()                          # the derived value is valid

    # flag ON but no offline date / malformed date => fail loud
    for bad in (None, "", "2017", "20171301"):         # incl. month 13 (bad date)
        with pytest.raises(SystemExit, match="align-insolation"):
            _maybe_align_insolation(
                SimpleNamespace(align_insolation=True, local_era5_date=bad), base)

    # parser default is OFF
    assert _build_arg_parser().parse_args(
        ["--config", "x.json"]).align_insolation is False
    assert _build_arg_parser().parse_args(
        ["--config", "x.json", "--align-insolation"]).align_insolation is True


def test_configure_jax_compilation_cache(tmp_path):
    """The persistent compilation-cache config (iter 453): empty dir => no-op (None, no JAX
    mutation); a dir => sets jax_compilation_cache_dir + the min-compile-time threshold;
    parser defaults (env-driven dir, 30 s)."""
    import jax

    from scripts.run.run_correction_campaign import (
        _build_arg_parser,
        _configure_jax_compilation_cache,
    )

    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    # empty => no-op, no mutation
    assert _configure_jax_compilation_cache("", 30.0) is None
    assert jax.config.jax_compilation_cache_dir == before_dir
    try:
        d = str(tmp_path / "jaxcache")
        assert _configure_jax_compilation_cache(d, 45.0) == d
        assert jax.config.jax_compilation_cache_dir == d
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 45.0
    finally:                                    # restore global JAX state for other tests
        jax.config.update("jax_compilation_cache_dir", before_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before_secs)

    a = _build_arg_parser().parse_args(["--config", "x.json"])
    assert a.cache_min_compile_secs == 30.0          # default threshold
    assert isinstance(a.compilation_cache_dir, str)  # env-driven (empty when unset)


def test_ocean_only_mask_from_land_fraction_and_fail_loud():
    """--ocean-only builds a valid_mask from the model's static land fraction (iter 451);
    OFF by default => None (rank all); an all-land grid fails loud; parser defaults are off."""
    import jax.numpy as jnp
    import legoesm.driver.model_driver as md
    import pytest

    from scripts.run.run_correction_campaign import (
        _build_arg_parser,
        _maybe_ocean_mask,
    )

    # flag OFF (default) => None (rank ALL columns), no driver probe
    off = SimpleNamespace(ocean_only=False, max_land_fraction=0.5)
    assert _maybe_ocean_mask(off, base_cfg=object()) is None

    # ON: probe the driver's static land fraction -> ocean mask. Patch ModelDriver so the
    # test is cheap + grid-agnostic (the probe itself is covered in test_model_driver_static_phis;
    # the helper's function-scope `from ... import ModelDriver` resolves the patched module attr).
    class _FakeDriver:
        def __init__(self, _cfg):
            pass

        def static_land_fraction(self):
            return jnp.array([[0.0, 0.6], [0.4, 1.0]])   # 2 ocean (<=0.5), 2 land

    orig = md.ModelDriver
    md.ModelDriver = _FakeDriver
    try:
        on = SimpleNamespace(ocean_only=True, max_land_fraction=0.5, mode="amip")
        mask = _maybe_ocean_mask(on, base_cfg=object())
        assert mask is not None and int(jnp.sum(mask)) == 2     # 2 ocean columns ranked

        # REGRESSION (iter 484): the mask must be GRID-SHAPED (broadcastable to the 2D
        # per-column score), NOT a flat (n_columns,) array. A flat mask is REJECTED by the
        # harness check on a STRUCTURED grid (n_columns does not broadcast to (nlat,nlon)),
        # which silently broke --ocean-only on lat-lon/cubed-sphere/Gaussian until the dry-run
        # surfaced it. sum() alone (shape-agnostic) passed despite the bug, so assert the SHAPE
        # AND that the harness broadcastability check accepts it.
        from scripts.run.run_correction_campaign import assert_per_column_fields_match_grid
        assert mask.shape == (2, 2)
        assert_per_column_fields_match_grid(
            (2, 2), area_weights=jnp.ones((2, 2)), valid_mask=mask)   # must NOT raise

        # CMIP applicability caveat (iter 457): print a NOTE that ocean-only is most meaningful
        # for AMIP (prescribed SST pins the ocean); silent for AMIP.
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _maybe_ocean_mask(SimpleNamespace(
                ocean_only=True, max_land_fraction=0.5, mode="cmip"), base_cfg=object())
        assert "cmip" in buf.getvalue() and "interactive" in buf.getvalue().lower()
        buf_amip = io.StringIO()
        with redirect_stdout(buf_amip):
            _maybe_ocean_mask(on, base_cfg=object())
        assert "interactive" not in buf_amip.getvalue().lower()   # no CMIP note for AMIP

        class _AllLand(_FakeDriver):                            # no ocean => fail loud
            def static_land_fraction(self):
                return jnp.ones((2, 2))
        md.ModelDriver = _AllLand
        with pytest.raises(SystemExit, match="ocean-only"):
            _maybe_ocean_mask(on, base_cfg=object())
    finally:
        md.ModelDriver = orig

    # parser defaults: off, 0.5
    a = _build_arg_parser().parse_args(["--config", "x.json"])
    assert a.ocean_only is False and a.max_land_fraction == 0.5
    a2 = _build_arg_parser().parse_args(
        ["--config", "x.json", "--ocean-only", "--max-land-fraction", "0.0"])
    assert a2.ocean_only is True and a2.max_land_fraction == 0.0


def test_campaign_knobs_reject_degenerate_counts():
    """A non-positive --n-worst (ranks NOTHING) or --les-budget (runs NO LES) must FAIL
    LOUD at construction — caught by the launch dry-run, not after a multi-day no-op run
    (the iter-201 / iter-388 'no silent no-op' convention; iter 249).  ``--les-budget``
    unset (None = no cap) stays valid."""
    parser = _build_arg_parser()

    def _knobs(extra):
        return _campaign_knobs_from_args(
            parser.parse_args(["--config", "x", "--era5-zarr", "z", *extra]))

    for bad in (["--n-worst", "0"], ["--n-worst", "-3"]):
        with pytest.raises(SystemExit, match=r"--n-worst .* must be >= 1"):
            _knobs(bad)
    with pytest.raises(SystemExit, match=r"--les-budget .* must be >= 1"):
        _knobs(["--les-budget", "0"])
    # the valid defaults (n_worst=20, les_budget unset) and an explicit positive budget pass.
    assert _knobs([])["n_worst"] == 20
    assert _knobs(["--n-worst", "1", "--les-budget", "1"])["les_budget"] == 1


def test_campaign_knobs_are_valid_kwargs_for_both_builders():
    """Every knob the CLI maps MUST be a real kwarg of BOTH builders — so a renamed /
    removed builder param fails LOUDLY here, not silently on an HPC launch (this is
    the test that would have caught the iter-101 'CLI forgot to forward the flag'
    drift; iter 102)."""
    import inspect

    knobs = set(_campaign_knobs_from_args(
        _build_arg_parser().parse_args(["--config", "x", "--era5-zarr", "z"])))
    assert "stop_on_no_valid_diagnoses" in knobs            # the iter-101 flag
    for fn in (build_correction_campaign, build_multi_correction_campaign):
        params = set(inspect.signature(fn).parameters)
        missing = knobs - params
        assert not missing, f"{fn.__name__} missing CLI knobs: {missing}"


def test_builders_forward_campaign_knobs_to_run_loop(monkeypatch):
    """Closes the builder→run-loop hop (the subset test only proves CLI→builder): each
    builder must FORWARD the run-bound knobs to run_*correction_campaign. The loop is
    monkeypatched to capture kwargs; sentinel values are asserted. (n_worst is NOT here
    — it is consumed by compose_compare_fn, a different hop.) iter 102 Codex."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    import scripts.run.run_correction_campaign as rcc

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    ms = _full_grid_state()                                   # physical → passes validate
    common = dict(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(ms),        # noqa: ARG005
        extract_column_state=lambda d, day, dt: d.state,      # noqa: ARG005
        reference=ms, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=4, n_worst=1,
        run_les_fn=_mock_run_les_sheared)
    # sentinel knobs (non-default so a dropped forward is visible).
    knobs = dict(les_budget=3, feedback_strategy="environment",
                 accept_only_if_improved=False, step_fractions=[0.5],
                 clip_to_bounds=False, bias_tol=0.123, patience=9,
                 stop_on_no_valid_diagnoses=False)

    # run_correction_campaign is a module-top import (patch on rcc);
    # run_multi_correction_campaign is a function-scope import (patch the SOURCE).
    cap_single = {}
    monkeypatch.setattr(rcc, "run_correction_campaign",
                        lambda *a, **k: (cap_single.update(k), "R")[1])
    assert build_correction_campaign(
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        **common, **knobs) == "R"
    for key, val in knobs.items():
        assert cap_single[key] == val, f"single builder dropped {key}"

    cap_multi = {}
    monkeypatch.setattr(
        "legoesm.training.correction_loop.run_multi_correction_campaign",
        lambda *a, **k: (cap_multi.update(k), "M")[1])
    assert build_multi_correction_campaign(
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        coefficients=("C_K", "Pr_t"), **common, **knobs) == "M"
    for key, val in knobs.items():
        assert cap_multi[key] == val, f"multi builder dropped {key}"


def _bias_imp(base, upd):
    from legoesm.training.bias_metrics import BiasImprovement
    frac = (base - upd) / base if base != 0.0 else 0.0
    return BiasImprovement(
        baseline_bias=jnp.asarray(base), updated_bias=jnp.asarray(upd),
        absolute_reduction=jnp.asarray(base - upd),
        fractional_improvement=jnp.asarray(frac), improved=jnp.asarray(upd < base))


def test_campaign_output_dict_single_round_trips_into_deploy():
    """The single-coefficient campaign OUTPUT dict, after a JSON round-trip, loads
    through the DEPLOY path (corrected_clubb_config) into a per-column CLUBBLiteConfig
    with the SAME C_K — locking the write/read format of the clause-5 'update the
    parameters' plumbing against drift (iter 105)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.6),
            worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)

    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "n_columns": 4},
        summary=summary, health=health, corrected_field="C_K")
    loaded = json.loads(json.dumps(out))                  # the REAL on-disk round trip
    cfg = corrected_clubb_config(loaded)
    np.testing.assert_allclose(np.asarray(cfg.C_K).reshape(-1), np.asarray(field))
    # the diagnostics survive too (the file is also the campaign's human report).
    assert loaded["summary"]["stop_reason"] == "converged"
    assert loaded["health"]["status"] == health.status
    assert loaded["grid"]["grid_type"] == "latlon"


def test_output_dict_records_averaging_provenance():
    """The output JSON records the comparison's ERA5 averaging window (iter 267) so the
    empirical result is self-describing — a bias vs a single SNAPSHOT (n_times=1) is a
    different quantity than vs an N-time climatology. Absent ``averaging`` keeps the
    legacy shape (no key); a provided block round-trips through the JSON."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult

    from scripts.run.run_correction_campaign import _averaging_provenance

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.6),
            worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)
    base = dict(grid_provenance={}, summary=summary, health=health, corrected_field="C_K")
    # absent ⇒ no averaging key (back-compat with the round-trip / deploy contract).
    assert "averaging" not in build_campaign_output_dict(res, **base)
    # the helper resolves the window from args; a climatology run records n_times>1.
    av = _averaging_provenance(SimpleNamespace(era5_time_idx=12, era5_n_times=30))
    assert av == {"era5_time_idx": 12, "era5_n_times": 30}
    out = json.loads(json.dumps(build_campaign_output_dict(res, averaging=av, **base)))
    assert out["averaging"] == {"era5_time_idx": 12, "era5_n_times": 30}


def test_output_dict_records_les_provenance():
    """The output JSON records the spin-off LES config (iter 368) — esp. whether a
    prescribed SURFACE-FLUX BC was used (iter 364/365), which changes HOW the closure was
    diagnosed — so the correction is self-describing.  Absent ⇒ no ``les_config`` key
    (back-compat with the deploy contract); a provided block round-trips through JSON and
    the DEPLOY loader IGNORES the extra key (reads only the corrected field)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    from scripts.run.run_correction_campaign import _les_provenance

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.6),
            worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)
    base = dict(grid_provenance={}, summary=summary, health=health, corrected_field="C_K")
    assert "les_config" not in build_campaign_output_dict(res, **base)   # absent ⇒ no key

    prov = _les_provenance(SimpleNamespace(surface_flux=True, les_budget=8,
                                           les_dt=0.5, les_hours=2.0))
    assert prov == {"surface_flux": True, "les_budget": 8, "les_dt_s": 0.5, "les_hours": 2.0}
    # default-OFF run records surface_flux=False (distinguishable from a surface-flux run).
    assert _les_provenance(SimpleNamespace(surface_flux=False, les_budget=None,
                                           les_dt=0.5, les_hours=2.0))["surface_flux"] is False

    out = json.loads(json.dumps(build_campaign_output_dict(res, les_provenance=prov, **base)))
    assert out["les_config"]["surface_flux"] is True
    # the deploy loader IGNORES the extra les_config key (reads only the corrected field).
    cfg = corrected_clubb_config(out)
    np.testing.assert_allclose(np.asarray(cfg.C_K), np.asarray(field), rtol=1e-12)


def test_averaging_provenance_records_the_model_window_when_config_given():
    """With base_cfg (the campaign output + dry-run), _averaging_provenance ALSO records the
    MODEL-side averaging window — days / diag_days / n_samples — so the iter-267 window
    alignment is TWO-sided (the deployer confirms the model climatology window vs the ERA5
    one, both time-means, not a snapshot). Without base_cfg (the cross-grid kernel write) it
    omits them; diag_days=0 → n_samples None, not a div-by-zero (iter 313)."""
    from scripts.run.run_correction_campaign import _averaging_provenance

    args = SimpleNamespace(era5_time_idx=0, era5_n_times=30)
    cfg = SimpleNamespace(days=200, output=SimpleNamespace(diag_days=5))
    av = _averaging_provenance(args, cfg)
    assert av["era5_n_times"] == 30
    assert av["model_days"] == 200 and av["model_diag_days"] == 5
    assert av["model_n_samples"] == 40                       # 200 // 5
    assert "model_days" not in _averaging_provenance(args)   # no base_cfg ⇒ ERA5-only block
    cfg0 = SimpleNamespace(days=200, output=SimpleNamespace(diag_days=0))
    assert _averaging_provenance(args, cfg0)["model_n_samples"] is None  # no div-by-zero


def test_averaging_provenance_records_era5_n_days_for_the_offline_reference():
    """The OFFLINE multi-day reference coverage (era5_n_days, iter 460/470) is recorded in the
    provenance + surfaced in the dry-run line, so a deployer/operator sees the reference's day
    span. The Zarr path (no --local-era5-dir) omits it (it spans times itself)."""
    from scripts.run.run_correction_campaign import (
        _averaging_provenance,
        _dry_run_era5_line,
    )

    offline = SimpleNamespace(era5_time_idx=0, era5_n_times=720,
                              local_era5_dir="/rda/ERA5", era5_n_days=30)
    av = _averaging_provenance(offline)
    assert av["era5_n_days"] == 30 and av["era5_n_times"] == 720
    assert "over 30 day(s) of offline ERA5" in _dry_run_era5_line(av)

    zarr = SimpleNamespace(era5_time_idx=0, era5_n_times=4, local_era5_dir=None)
    assert "era5_n_days" not in _averaging_provenance(zarr)         # Zarr: N/A
    assert "offline ERA5" not in _dry_run_era5_line(_averaging_provenance(zarr))


def test_campaign_output_dict_diverged_run_is_strict_json():
    """The WHOLE build_campaign_output_dict for a DIVERGED run (NaN bias) must be standard
    JSON — no NaN/Infinity tokens anywhere.

    `test_summary_to_json_diverged_biases_serialize_as_null` covers the summary COMPONENT;
    this locks the actual on-disk ARTIFACT (summary + health + corrected field + grid) an
    operator's plotters / deploy reader will `json.load`.  A non-finite metric leaking
    through ANY branch would make that file unparseable under the strict parser — caught
    here end-to-end (the corrected field is finite by construction; the NaN risk is the
    bias/stat branches)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult

    field = jnp.array([0.42, 0.55, 0.61, 0.73])               # the field is finite by construction
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(5.0, float("nan")),   # DIVERGED: NaN final bias
            worst_column_change=jnp.asarray(float("nan")),
            feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=0),),
        final_field=field.reshape(2, 2), accepted=(False,), stop_reason="max_iterations")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)
    assert not health.ok                                      # the diverged run is not 'improved'

    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "n_columns": 4},
        summary=summary, health=health, corrected_field="C_K")
    # The strict parser raises on ANY NaN/Infinity token anywhere in the artifact.
    json.loads(json.dumps(out), parse_constant=_reject_nonstandard)


def test_output_dict_rejects_non_finite_corrected_field():
    """A NON-FINITE corrected coefficient (an un-clipped ill-posed diagnosis that slipped
    the accept gate) fails LOUD at the write boundary (iter 270) — it would otherwise
    serialize as a NaN/Infinity token (unparseable output file) AND deploy a coefficient
    that blows up the production run. NOT sanitized to null (unlike the bias trajectory):
    a NaN coefficient is a bug to surface. The finite field by construction still writes."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult

    def _res(field):
        return CampaignResult(
            final_config=CLUBBLiteConfig(C_K=field),
            iterations=(CorrectionResult(
                updated_config=None, bias=_bias_imp(1.0, 0.6),
                worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
                n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
            final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")

    bad = jnp.array([0.42, float("nan"), 0.61, float("inf")])
    res = _res(bad)
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    base = dict(grid_provenance={}, summary=summary, health=campaign_health(summary))
    with pytest.raises(ValueError, match=r"corrected field 'C_K' has 2 non-finite"):
        build_campaign_output_dict(res, corrected_field="C_K", **base)
    # the finite field still writes fine.
    ok = _res(jnp.array([0.42, 0.55, 0.61, 0.73]))
    ok_summary = summarize_campaign(ok, promotion_key="clubb_lite_C_K")
    out = build_campaign_output_dict(
        ok, corrected_field="C_K", grid_provenance={}, summary=ok_summary,
        health=campaign_health(ok_summary))
    assert len(out["C_K"]) == 4


def test_campaign_output_dict_multi_round_trips_into_deploy():
    """The multi-coefficient ``"fields"`` shape round-trips into corrected_clubb_config
    recovering BOTH C_K and Pr_t per column (iter 105)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import (
        MultiCampaignResult,
        MultiCorrectionResult,
    )
    from legoesm.training.deploy_correction import corrected_clubb_config

    ck = jnp.array([0.42, 0.55, 0.61, 0.73])
    prt = jnp.array([0.78, 0.81, 0.83, 0.80])
    res = MultiCampaignResult(
        final_config=CLUBBLiteConfig(C_K=ck, Pr_t=prt),
        iterations=(MultiCorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.5),
            worst_column_change=jnp.asarray(0.0), feedback_fields={},
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_fields={"clubb_lite_C_K": ck.reshape(2, 2),
                      "clubb_lite_Pr_t": prt.reshape(2, 2)},
        accepted=(True,), stop_reason="max_iterations")
    summary = summarize_campaign(res)
    health = campaign_health(summary)

    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon", "n_columns": 4},
        summary=summary, health=health, coefficients=("C_K", "Pr_t"))
    loaded = json.loads(json.dumps(out))
    assert loaded["coefficients"] == ["C_K", "Pr_t"]
    cfg = corrected_clubb_config(loaded)
    np.testing.assert_allclose(np.asarray(cfg.C_K).reshape(-1), np.asarray(ck))
    np.testing.assert_allclose(np.asarray(cfg.Pr_t).reshape(-1), np.asarray(prt))


def test_campaign_output_dict_requires_exactly_one_shape():
    """Dispatch hardening: pass exactly one of corrected_field / coefficients."""
    res = SimpleNamespace(iterations=(), accepted=(), final_config=None,
                          final_fields={})
    summary = SimpleNamespace()
    health = SimpleNamespace(status="x", message="y")
    with pytest.raises(ValueError, match="EXACTLY one"):
        build_campaign_output_dict(res, grid_provenance={}, summary=summary,
                                   health=health)                       # neither
    with pytest.raises(ValueError, match="EXACTLY one"):
        build_campaign_output_dict(res, grid_provenance={}, summary=summary,
                                   health=health,
                                   corrected_field="C_K", coefficients=("C_K",))  # both


def test_campaign_output_dict_scalar_single_field_is_rejected_by_deploy():
    """A no-op / zero-round campaign that corrected NOTHING leaves a SCALAR
    final_config field; the output writes it as a bare float (NO reshape), so the
    deploy loader's 1-D assertion REJECTS it loudly instead of silently accepting a
    1-column array (Codex iter 105 — the un-masked failure path)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    res = CampaignResult(
        final_config=CLUBBLiteConfig(),                  # scalar default C_K (no-op)
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 1.0),  # never improved
            worst_column_change=jnp.asarray(0.0), feedback_field=jnp.zeros((2, 2)),
            n_corrected=0, n_diagnosed=0, n_diagnoses_valid=0),),
        final_field=jnp.zeros((2, 2)), accepted=(True,), stop_reason="max_iterations")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    out = build_campaign_output_dict(
        res, grid_provenance={"grid_type": "latlon"}, summary=summary,
        health=campaign_health(summary), corrected_field="C_K")
    loaded = json.loads(json.dumps(out))
    assert not isinstance(loaded["C_K"], list)            # a bare float, NOT [0.4]
    with pytest.raises(ValueError, match="1-D per-column"):
        corrected_clubb_config(loaded)


def test_fast_validation_les_regime_is_tiny_valid_and_shared():
    """The shared WIRING-smoke regime (iter 504) is a single tiny box for BOTH the shallow
    and deep selectors, passes the production regime validator, and is FAR smaller than the
    production default — so it is unmistakably a composition pre-flight, not a science
    regime. Centralising it here means the real-ERA5 full-loop check + the OSSE --quick
    smoke share ONE definition (no duplicated LES dimensions)."""
    from legoesm.atmosphere.dynamics.les.les_regime import (
        LESRegimeConfig,
        validate_regime_config,
    )

    regime = fast_validation_les_regime()
    validate_regime_config(regime)                        # must not raise (self-validating)
    assert regime.shallow == regime.deep                  # one tiny box for both selectors
    tiny = regime.shallow
    assert (tiny.nx, tiny.ny, tiny.nlev) == (8, 8, 8)
    assert tiny.dz_sfc_m * tiny.nlev < tiny.domain_top_m  # the invariant the validator checks

    # Dramatically cheaper than the production default (the whole point of the fast smoke).
    prod = LESRegimeConfig()
    cells = lambda r: r.nx * r.ny * r.nlev  # noqa: E731
    assert cells(tiny) * 100 < cells(prod.shallow)
    assert cells(tiny) * 100 < cells(prod.deep)


def _mk_breakdown(turbulent=True, finite=True, thermo=True, moisture=True, rh=True):
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import LESRealismBreakdown
    flags = (turbulent, finite, thermo, moisture, rh)
    return LESRealismBreakdown(
        turbulent=jnp.asarray(turbulent), finite=jnp.asarray(finite),
        thermo_consistent=jnp.asarray(thermo), moisture_physical=jnp.asarray(moisture),
        rh_ok=jnp.asarray(rh), overall=jnp.asarray(all(flags)))


def test_realism_capture_records_breakdown_and_passes_state_through(monkeypatch):
    """_RealismCapture (iter 512) wraps run_les_fn: it returns the LES state UNCHANGED (pure
    observation) and stores the per-column realism breakdown for the end-of-run report."""
    import legoesm.atmosphere.dynamics.les.column_les_diagnosis as cld

    sentinel_state = object()
    monkeypatch.setattr(cld, "column_les_realism_breakdown",
                        lambda state, hc: _mk_breakdown(turbulent=(hc == "GOOD")))
    cap = _RealismCapture(lambda setup: sentinel_state)
    out = cap(SimpleNamespace(height_coord="GOOD"))
    assert out is sentinel_state                       # state passed through unchanged
    cap(SimpleNamespace(height_coord="DEAD"))          # a laminar one
    assert len(cap.breakdowns) == 2
    assert bool(cap.breakdowns[0].overall) and not bool(cap.breakdowns[1].overall)


def test_realism_campaign_summary_line():
    """The campaign-aggregate realism line: None when nothing captured, an all-realistic
    line when none rejected, else the per-mode rejection counts (iter 512)."""
    assert _realism_campaign_summary_line(None) is None
    assert _realism_campaign_summary_line([]) is None

    allgood = [_mk_breakdown(), _mk_breakdown(), _mk_breakdown()]
    line = _realism_campaign_summary_line(allgood)
    assert line is not None and "all 3 spin-offs realistic" in line

    mixed = [_mk_breakdown(), _mk_breakdown(turbulent=False),
             _mk_breakdown(moisture=False), _mk_breakdown(turbulent=False)]
    line = _realism_campaign_summary_line(mixed)
    assert "1/4 realistic" in line and "3 rejected" in line
    assert "2x laminar" in line and "1x moisture-runaway" in line
    # The line is SHARED by the campaign + OSSE (prefix=), so its tail must be GENERIC: it must
    # NOT name a campaign-only verdict (the OSSE has no `no_valid_diagnoses`) — iter 523.
    assert "no_valid_diagnoses" not in line
    assert "[osse]" in _realism_campaign_summary_line(mixed, prefix="[osse]")


def test_output_dict_records_les_realism():
    """The output JSON records the campaign-aggregate LES-realism rejection counts (iter 514)
    for machine-readable post-run analysis; absent when no LES ran (None ⇒ no key), present +
    JSON-round-tripping when given, and IGNORED by the deploy loader (an extra key)."""
    import json

    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.campaign_summary import campaign_health, summarize_campaign
    from legoesm.training.correction_loop import CampaignResult, CorrectionResult
    from legoesm.training.deploy_correction import corrected_clubb_config

    field = jnp.array([0.42, 0.55, 0.61, 0.73])
    res = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(CorrectionResult(
            updated_config=None, bias=_bias_imp(1.0, 0.6),
            worst_column_change=jnp.asarray(0.0), feedback_field=field.reshape(2, 2),
            n_corrected=2, n_diagnosed=2, n_diagnoses_valid=2),),
        final_field=field.reshape(2, 2), accepted=(True,), stop_reason="converged")
    summary = summarize_campaign(res, promotion_key="clubb_lite_C_K")
    health = campaign_health(summary)
    base = dict(grid_provenance={}, summary=summary, health=health, corrected_field="C_K")

    assert _realism_summary_dict(None) is None and _realism_summary_dict([]) is None
    rd = _realism_summary_dict([_mk_breakdown(), _mk_breakdown(turbulent=False)])
    assert rd["n_total"] == 2 and rd["n_realistic"] == 1 and rd["n_not_turbulent"] == 1
    assert all(isinstance(v, int) for v in rd.values())            # JSON-safe ints

    assert "les_realism" not in build_campaign_output_dict(res, **base)   # absent ⇒ no key
    out = json.loads(json.dumps(build_campaign_output_dict(res, les_realism=rd, **base)))
    assert out["les_realism"]["n_not_turbulent"] == 1
    cfg = corrected_clubb_config(out)                              # deploy IGNORES the extra key
    np.testing.assert_allclose(np.asarray(cfg.C_K), np.asarray(field), rtol=1e-12)


def test_idealized_radiation_low_leverage_warning():
    """The launch warning fires for IDEALIZED radiation (gray/none) — where the bias has tiny
    C_K leverage (iters 412/514) — and is SILENT for the realistic spectral schemes
    (rrtmgp/rrtmg), so an operator does not misread a gray 'stalled' verdict (iter 515)."""
    from scripts.run.run_correction_campaign import (
        _idealized_radiation_low_leverage_warning as warn,
    )

    assert warn("rrtmgp") is None and warn("rrtmg") is None      # realistic → silent
    assert warn(None) is None                                    # missing → silent (stub-safe)
    for idealized in ("gray", "none", "GRAY"):
        msg = warn(idealized)
        assert msg is not None and "IDEALIZED" in msg and "rrtmgp" in msg


def test_print_realism_summary_uses_prefix_and_skips_empty(capsys):
    """print_realism_summary (iter 520) reads a _RealismCapture-wrapped run_les's breakdowns
    and prints the aggregate line with the given prefix (so the OSSE tags it [osse], the
    campaign [campaign]); prints NOTHING when nothing was captured (a dry-run / no LES)."""
    print_realism_summary(SimpleNamespace(breakdowns=[]))      # empty → no print
    print_realism_summary(SimpleNamespace())                   # no .breakdowns → no print
    assert capsys.readouterr().out == ""

    run_les = SimpleNamespace(breakdowns=[_mk_breakdown(), _mk_breakdown(turbulent=False)])
    print_realism_summary(run_les, prefix="[osse]")
    out = capsys.readouterr().out
    assert "[osse] LES realism:" in out and "1x laminar" in out


def test_realism_capture_is_wired_into_build_run_setup(monkeypatch, capsys):
    """_build_run_setup wraps run_les in a _RealismCapture (iter 520) so BOTH the campaign and
    the OSSE capture realism without per-main wrapping; and it surfaces the iter-515 IDEALIZED-
    radiation launch warning (the base config here is gray) — both verified on one
    _build_run_setup call without running the heavy setup body (the heavy deps are
    monkeypatched)."""
    import scripts.run.run_correction_campaign as rcc

    monkeypatch.setattr(rcc, "load_base_config_and_grid", lambda cfg: (SimpleNamespace(
        days=10, output=SimpleNamespace(diag_days=5), radiation="gray", land_mask_path=""),
        object(), object()))
    monkeypatch.setattr(rcc, "make_base_driver_builder",
                        lambda *a, **k: (lambda c: None, lambda *x, **y: None))
    monkeypatch.setattr(rcc, "_les_n_steps", lambda h, d: 1)
    monkeypatch.setattr(rcc, "resolve_orographic_phis", lambda *a, **k: None)
    args = SimpleNamespace(config="c.json", mode="amip", coupled_preset=None,
                           les_hours=2.0, les_dt=0.5, orographic_forcing="off",
                           align_insolation=False, local_era5_date=None,
                           amip_forcing_n_months=1, amip_forcing_from_local_era5=False,
                           local_era5_dir=None, surface_flux=False, ocean_only=False,
                           spinup_days=0.0)
    run_les = rcc._build_run_setup(args)[5]
    assert isinstance(run_les, rcc._RealismCapture)
    assert run_les.breakdowns == []
    # The shared preamble surfaces the iter-515 idealized-radiation warning at LAUNCH (gray →
    # the bias is C_K-insensitive), for BOTH the campaign and the OSSE.
    out = capsys.readouterr().out
    assert "IDEALIZED" in out and "rrtmgp" in out


def test_reduce_realism_summary_mpi():
    """reduce_realism_summary_mpi (iter 528) collective-sums a per-rank RealismRejectionSummary
    into the global one — verified WITHOUT mpirun by injecting the reduce: identity (single
    rank) leaves it unchanged; a doubling reduce (a 2-rank sim) doubles every count; the result
    is a RealismRejectionSummary of plain ints."""
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import RealismRejectionSummary

    s = RealismRejectionSummary(
        n_total=4, n_realistic=1, n_rejected=3, n_not_turbulent=2, n_not_finite=1,
        n_thermo_drift=0, n_moisture_runaway=0, n_supersaturated=0)

    assert reduce_realism_summary_mpi(s, lambda x: x) == s          # single rank → unchanged
    g = reduce_realism_summary_mpi(s, lambda x: x * 2)             # 2-rank sim → 2x
    assert isinstance(g, RealismRejectionSummary)
    assert g.n_total == 8 and g.n_rejected == 6 and g.n_not_turbulent == 4
    assert all(isinstance(v, int) for v in g)                      # global counts are ints


def test_realism_summary_public_list_entry():
    """realism_summary (promoted public, iter 528) turns a LIST of per-column breakdowns (the
    distributed operator's `_RealismCapture.breakdowns`) into a RealismRejectionSummary, or
    None when empty — the entry the distributed pattern (runbook §7) feeds to
    reduce_realism_summary_mpi."""
    assert realism_summary(None) is None and realism_summary([]) is None
    s = realism_summary([_mk_breakdown(), _mk_breakdown(turbulent=False)])
    assert s.n_total == 2 and s.n_realistic == 1 and s.n_not_turbulent == 1
