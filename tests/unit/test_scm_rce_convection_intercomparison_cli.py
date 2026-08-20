"""``--subsidence-solve`` kernel-ARM plumbing in the convection intercomparison.

Ranking convection schemes against a CRM is confounded when the schemes do not
share a vertical transport kernel, so the driver is run TWICE: once with every
mass-flux scheme forced onto the conservative kernel (PRIMARY, isolates scheme
physics) and once with the shipped defaults (SECONDARY, what users get today).
The override itself is the campaign's shared selector and is tested in
``test_scm_rce_subsidence_solve_override.py``; what is tested HERE is the
driver-side plumbing that keeps the two arms from contaminating each other:

* the flag exists on the real parser, defaults to ``as_shipped``, and rejects a
  typo (a typo must not silently produce the as-shipped arm);
* the arm is part of ``_run_signature``, so one arm's per-scheme checkpoint can
  never be silently reused for the other — that would FABRICATE a number rather
  than merely reuse a stale one.  This is the anti-confound gate: it fails if
  anyone drops ``subsidence_solve`` from ``_SIGNATURE_FIELDS``;
* the arm and the per-scheme override status are recorded on every artifact
  (per-scheme JSON, ``intercomparison.csv`` row, ``summary.md``), so no score
  can be read without knowing which kernel produced it.

All checks are pure-CPU and fast: no CRM reference data, no SCM integration.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

import scripts.run.run_scm_rce_convection_intercomparison as driver

camp = driver.camp


# --------------------------------------------------------------------------- #
# Helpers — synthetic diagnostics; nothing here runs the SCM.
# --------------------------------------------------------------------------- #
def _args(*argv):
    """Parse through the module's REAL parser.

    Deliberately not a hand-rolled ``argparse.Namespace``: a hand-rolled one
    would keep passing after the flag is renamed or removed.
    """
    return driver.build_parser().parse_args(list(argv))


def _run(score: float = 1.0):
    return camp.RunDiagnostics(
        label="x", config={}, status="ok", reason="",
        T_rmse=score, qv_rmse=score, cloud_rmse=score, precip_rmse=score,
        score=score,
        drift_T_rmse_K=0.1, drift_qv_rmse=1e-4, drift_qcond_rmse=1e-5,
        moist_adiabat_mean_abs_K=2.0, moist_adiabat_max_abs_K=10.0,
        cold_point_T_K=195.0, cold_point_z_km=17.0,
        T_profile=[300.0], qv_profile=[0.01], qcond_profile=[0.0],
    )


def _result(scheme="tiedtke", *, arm="implicit_flux", status=None, score=1.0):
    return driver.SchemeResult(
        scheme=scheme, prior=_run(score + 1.0), tuned=_run(score), records=[],
        subsidence_solve=arm,
        subsidence_solve_status=(
            status if status is not None else f"forced:{scheme}={arm}"),
    )


# A reference stub complete enough for the PHYSICAL-unit RMSE columns, which
# the summary/CSV writers now compute.  Deliberately not a bare
# SimpleNamespace(precip_ref_mm_day=...): the writers legitimately need the
# reference profiles, and a stub that made them silently return NaN would let
# a broken column pass its own test.  One level, so the mass-weighted RMSE of
# the single-level _run() profiles is exactly |profile - ref|.
def _ref_stub(precip_ref_mm_day: float = 7.16):
    return types.SimpleNamespace(
        precip_ref_mm_day=precip_ref_mm_day,
        mass_weights=[1.0],
        T_ref=[298.0], qv_ref=[0.012], qcond_ref=[1.0e-5],
    )


_SUMMARY_META = dict(
    radiation="rrtmgp", dt=600.0, days=100.0, analysis_days=5.0,
    tune_evals=48, surface_wind_m_s=5.0, large_scale_forcing="none",
    last_reference_files=5,
)


# --------------------------------------------------------------------------- #
# (a) the flag itself
# --------------------------------------------------------------------------- #
def test_flag_defaults_to_as_shipped():
    """Default is the SECONDARY arm: adding the flag must not silently change
    what an existing command line does."""
    assert _args().subsidence_solve == "as_shipped"


@pytest.mark.parametrize("mode", camp.SUBSIDENCE_SOLVE_MODES)
def test_flag_accepts_every_campaign_mode(mode):
    """The CLI choices track the campaign's mode tuple, so a mode added there is
    reachable here rather than silently unusable."""
    assert _args("--subsidence-solve", mode).subsidence_solve == mode


def test_flag_rejects_unknown_mode():
    """Dispatch-hardening: a typo must be a hard error, not a fallback to the
    as-shipped arm reported as the matched-kernel one."""
    with pytest.raises(SystemExit):
        _args("--subsidence-solve", "implicitflux")


# --------------------------------------------------------------------------- #
# (d) THE ANTI-CONFOUND GATE — the arm is part of the checkpoint signature
# --------------------------------------------------------------------------- #
def test_subsidence_solve_is_a_signature_field():
    assert "subsidence_solve" in driver._SIGNATURE_FIELDS


def test_run_signature_differs_between_kernel_arms():
    """A per-scheme checkpoint from one arm must NEVER be reusable in the other.

    Reuse across arms would not merely be stale — it would report the shipped
    kernel's score as the matched-kernel result (or vice versa), fabricating the
    very comparison the two arms exist to make honest.  This assertion fails if
    ``subsidence_solve`` is dropped from ``_SIGNATURE_FIELDS``.
    """
    shipped = driver._run_signature(_args("--subsidence-solve", "as_shipped"))
    matched = driver._run_signature(_args("--subsidence-solve", "implicit_flux"))
    assert shipped != matched
    assert shipped["subsidence_solve"] == "as_shipped"
    assert matched["subsidence_solve"] == "implicit_flux"
    # everything ELSE is identical: the arm is the only variable (controlled
    # comparison — if the two namespaces differed elsewhere the inequality above
    # would be vacuous).
    assert {k: v for k, v in shipped.items() if k != "subsidence_solve"} == {
        k: v for k, v in matched.items() if k != "subsidence_solve"}
    json.dumps(shipped)  # signature stays JSON-serializable


def test_signature_compatible_rejects_cross_arm_checkpoint(tmp_path):
    """End-to-end of the compute-path guard: a checkpoint stamped by one arm is
    incompatible with the other, so ``main`` recomputes instead of reusing."""
    shipped = driver._run_signature(_args("--subsidence-solve", "as_shipped"))
    matched = driver._run_signature(_args("--subsidence-solve", "implicit_flux"))
    assert not driver._signature_compatible(shipped, matched)
    assert not driver._signature_compatible(matched, shipped)
    assert driver._signature_compatible(matched, matched)

    driver.save_scheme_result(tmp_path, _result("tiedtke", arm="implicit_flux"),
                              matched)
    ckpt = driver._scheme_json_path(tmp_path, "tiedtke")
    assert driver._checkpoint_signature(ckpt) == matched
    assert not driver._signature_compatible(
        driver._checkpoint_signature(ckpt), shipped)


# --------------------------------------------------------------------------- #
# (b)/(c) the arm is threaded and recorded on every artifact
# --------------------------------------------------------------------------- #
def test_evaluate_scheme_takes_a_keyword_only_subsidence_solve():
    """Keyword-only, so a positional call site can never bind the arm by
    accident."""
    import inspect

    param = inspect.signature(driver.evaluate_scheme).parameters["subsidence_solve"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default == "as_shipped"


def test_scheme_result_roundtrip_preserves_arm_fields(tmp_path):
    res = _result("zhang_mcfarlane", arm="implicit_flux")
    driver.save_scheme_result(tmp_path, res)
    payload = json.loads(driver._scheme_json_path(tmp_path, "zhang_mcfarlane").read_text())
    assert payload["subsidence_solve"] == "implicit_flux"
    assert payload["subsidence_solve_status"] == "forced:zhang_mcfarlane=implicit_flux"

    loaded = driver.load_scheme_result(
        driver._scheme_json_path(tmp_path, "zhang_mcfarlane"))
    assert loaded.subsidence_solve == res.subsidence_solve
    assert loaded.subsidence_solve_status == res.subsidence_solve_status


def test_legacy_checkpoint_loads_as_shipped_with_empty_status(tmp_path):
    """Checkpoints written before the flag existed were necessarily run with the
    shipped defaults; the empty status marks them as never explicitly stamped."""
    path = tmp_path / "scheme_dca.json"
    path.write_text(json.dumps({
        "scheme": "dca", "signature": {},
        "prior": driver.asdict(_run(2.0)), "tuned": driver.asdict(_run(1.0)),
        "records": [],
    }))
    loaded = driver.load_scheme_result(path)
    assert loaded.subsidence_solve == "as_shipped"
    assert loaded.subsidence_solve_status == ""


def test_row_and_csv_fields_include_the_arm_columns(tmp_path):
    assert "subsidence_solve" in driver.CSV_FIELDS
    assert "subsidence_solve_status" in driver.CSV_FIELDS
    row = driver._row(_result("tiedtke", arm="implicit_flux"))
    assert row["subsidence_solve"] == "implicit_flux"
    assert row["subsidence_solve_status"] == "forced:tiedtke=implicit_flux"
    # every declared column is produced (DictWriter would raise otherwise)
    assert set(driver.CSV_FIELDS) <= set(row)

    csv_path = tmp_path / "intercomparison.csv"
    driver.write_csv(csv_path, [_result("tiedtke", arm="implicit_flux")])
    header, first = csv_path.read_text().splitlines()[:2]
    assert "subsidence_solve" in header.split(",")
    assert "implicit_flux" in first


def test_summary_names_the_kernel_arm(tmp_path):
    """No table in summary.md may be readable without its arm."""
    path = tmp_path / "summary.md"
    driver.write_summary(
        path, _ref_stub(),
        [_result("tiedtke", arm="implicit_flux"),
         _result("dca", arm="implicit_flux", status="not_applicable:dca", score=2.0)],
        meta=dict(_SUMMARY_META),
    )
    text = path.read_text()
    assert "--subsidence-solve implicit_flux" in text
    assert "PRIMARY" in text
    # per-scheme status, including the honestly-unmatched scheme
    assert "forced:tiedtke=implicit_flux" in text
    assert "not_applicable:dca" in text


def test_summary_labels_the_as_shipped_arm_as_secondary(tmp_path):
    path = tmp_path / "summary.md"
    driver.write_summary(
        path, _ref_stub(),
        [_result("tiedtke", arm="as_shipped", status="as_shipped")],
        meta=dict(_SUMMARY_META),
    )
    text = path.read_text()
    assert "--subsidence-solve as_shipped" in text
    assert "SECONDARY" in text


def test_summary_flags_a_mixed_arm_table_as_confounded(tmp_path):
    """If an outdir ever ends up holding both arms' checkpoints, the merged table
    must say it is confounded rather than mislabel itself with one arm."""
    path = tmp_path / "summary.md"
    driver.write_summary(
        path, _ref_stub(),
        [_result("tiedtke", arm="implicit_flux"),
         _result("bechtold", arm="as_shipped", status="as_shipped", score=2.0)],
        meta=dict(_SUMMARY_META),
    )
    text = path.read_text()
    assert "CONFOUNDED" in text
    assert "MIXED(" in text


# --------------------------------------------------------------------------- #
# Saturation treatment: --microphysics / --hard-saturation-adjustment
#
# The column's saturation treatment is an EXPERIMENT variable, not a cosmetic
# one: the IFS/SAM homogeneous-freezing ice allowance exists only inside the
# ice-carrying schemes (morrison/thompson/p3), and the in-scheme liquid guard
# changes the condensation rate on every step.  These gates assert (a) both
# reach the config the SCM actually runs, and (b) both invalidate a checkpoint,
# so a kessler run can never be merged into a morrison table.
# --------------------------------------------------------------------------- #

def test_microphysics_defaults_to_the_campaign_baseline():
    assert _args().microphysics == driver.camp.BASELINE_SCHEMES["microphysics"]
    assert _args().hard_saturation_adjustment is False


def test_microphysics_accepts_an_ice_scheme_and_rejects_an_unknown_one():
    assert _args("--microphysics", "morrison").microphysics == "morrison"
    with pytest.raises(SystemExit):
        _args("--microphysics", "not_a_scheme")


def test_hard_saturation_adjustment_round_trips_both_ways():
    assert _args("--hard-saturation-adjustment").hard_saturation_adjustment is True
    assert (_args("--hard-saturation-adjustment",
                  "--no-hard-saturation-adjustment")
            .hard_saturation_adjustment is False)


@pytest.mark.parametrize("field", ["microphysics", "hard_saturation_adjustment"])
def test_saturation_treatment_is_a_signature_field(field):
    assert field in driver._SIGNATURE_FIELDS


def test_run_signature_differs_between_microphysics_choices():
    a = driver._run_signature(_args("--microphysics", "kessler"))
    b = driver._run_signature(_args("--microphysics", "morrison"))
    assert a != b
    assert {k: v for k, v in a.items() if k != "microphysics"} == \
           {k: v for k, v in b.items() if k != "microphysics"}


def test_run_signature_differs_when_the_liquid_guard_is_on():
    a = driver._run_signature(_args())
    b = driver._run_signature(_args("--hard-saturation-adjustment"))
    assert a != b
    assert {k: v for k, v in a.items() if k != "hard_saturation_adjustment"} == \
           {k: v for k, v in b.items() if k != "hard_saturation_adjustment"}


def test_make_physics_config_threads_the_scheme_and_the_liquid_guard():
    """The flags must reach the sub-config the scheme reads, not just the args."""
    camp = driver.camp
    cfg = camp.make_physics_config(microphysics="morrison")
    assert cfg.microphysics.scheme == "morrison"
    assert cfg.microphysics.morrison.hard_saturation_adjustment is False
    on = camp.make_physics_config(
        microphysics="morrison", hard_saturation_adjustment=True)
    assert on.microphysics.morrison.hard_saturation_adjustment is True


def test_make_physics_config_refuses_a_silently_inert_threshold():
    camp = driver.camp
    with pytest.raises(ValueError, match="silently inert"):
        camp.make_physics_config(microphysics="kessler",
                                 hard_sat_adjust_threshold=1.05)


def test_make_physics_config_raises_for_a_scheme_without_the_guard():
    """sdm resolves super-saturation explicitly and has no guard field; asking
    for it must fail loudly rather than run un-guarded."""
    camp = driver.camp
    with pytest.raises(ValueError):
        camp.make_physics_config(microphysics="sdm",
                                 hard_saturation_adjustment=True)


# --------------------------------------------------------------------------- #
# Physical-unit RMSE columns (K, g/kg) — what the figures and the paper table
# read.  The normalised scores above are divided by the reference's own
# mass-weighted spread, which is right for the optimiser and meaningless in a
# caption.
# --------------------------------------------------------------------------- #

def test_row_carries_physical_rmse_when_a_reference_is_supplied():
    ref = _ref_stub()
    row = driver._row(_result(score=1.0), ref)
    # One level, unit weight => the mass-weighted RMSE is the plain difference.
    assert row["tuned_T_rmse_K"] == pytest.approx(abs(300.0 - 298.0), rel=1e-9)
    assert row["tuned_qv_rmse_g_kg"] == pytest.approx(
        abs(0.01 - 0.012) * 1000.0, rel=1e-9)
    assert row["tuned_qcond_rmse_g_kg"] == pytest.approx(
        abs(0.0 - 1.0e-5) * 1000.0, rel=1e-9)
    # a-priori and tuned use the SAME profiles in the stub, so they agree here;
    # what matters is that both columns are present and finite.
    assert np.isfinite(row["apriori_T_rmse_K"])


def test_row_without_a_reference_is_nan_not_zero():
    """A missing reference must not look like a perfect fit."""
    row = driver._row(_result(score=1.0))
    for key in ("apriori_T_rmse_K", "tuned_T_rmse_K",
                "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg"):
        assert np.isnan(row[key]), (key, row[key])


def test_csv_header_matches_the_paper_figure_script():
    """scripts/plot/plot_scm_rce_convection_paper.py reads these names."""
    for name in ("apriori_T_rmse_K", "tuned_T_rmse_K",
                 "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg"):
        assert name in driver.CSV_FIELDS


# --------------------------------------------------------------------------- #
# Merge gate.  Before this existed, the driver aggregated whatever checkpoints
# happened to be present: ONE surviving scheme was enough to produce a
# "ranking" table and figures.  The guard that was supposed to prevent it lived
# in an older sbatch, not in the driver.
# --------------------------------------------------------------------------- #

def _sig(**over):
    base = driver._run_signature(_args())
    base.update(over)
    return base


def _results_for(schemes, sig=None):
    out = []
    for s in schemes:
        r = _result(s)
        r.signature = dict(sig if sig is not None else _sig())
        out.append(r)
    return out


def test_merge_refuses_a_subset_of_the_schemes():
    partial = _results_for(list(driver.CONVECTION_SCHEMES)[:3])
    with pytest.raises(SystemExit, match="MERGE REFUSED"):
        driver._guard_merge_inputs(partial, _sig(), allow_partial=False)


def test_merge_accepts_the_complete_set():
    full = _results_for(driver.CONVECTION_SCHEMES)
    driver._guard_merge_inputs(full, _sig(), allow_partial=False)


def test_allow_partial_warns_instead_of_refusing(capsys):
    partial = _results_for(list(driver.CONVECTION_SCHEMES)[:2])
    driver._guard_merge_inputs(partial, _sig(), allow_partial=True)
    assert "MERGE REFUSED" in capsys.readouterr().out


def test_merge_refuses_a_checkpoint_from_another_reference():
    """The physical-unit columns are computed at merge time against the CURRENT
    reference, so a stale profile yields a believable wrong number, not NaN."""
    full = _results_for(driver.CONVECTION_SCHEMES)
    full[4].signature = _sig(reference_dir="/some/other/crm")
    with pytest.raises(SystemExit, match="different protocol"):
        driver._guard_merge_inputs(full, _sig(), allow_partial=False)


def test_merge_only_compares_checkpoints_with_each_other():
    """--merge-only has no meaningful current signature (the merge job does not
    repeat the arm's twenty flags), so agreement is required AMONG the
    checkpoints. Passing the arm's signature as the baseline instead would
    refuse every real campaign."""
    arm_sig = _sig(tune_seed=20260810, microphysics="morrison",
                   subsidence_solve="implicit_flux")
    full = _results_for(driver.CONVECTION_SCHEMES, sig=arm_sig)
    driver._guard_merge_inputs(full, None, allow_partial=False)
    # ... and one odd checkpoint out is still caught
    full[7].signature = _sig(tune_seed=20260810, microphysics="kessler",
                             subsidence_solve="implicit_flux")
    with pytest.raises(SystemExit, match="different protocol"):
        driver._guard_merge_inputs(full, None, allow_partial=False)


def test_merge_only_still_requires_the_scored_reference_to_match():
    """The one flag the merge job DOES supply is the reference, and it is the
    one the physical-unit columns are computed against."""
    arm_sig = _sig(reference_dir="/oracle/sam300")
    full = _results_for(driver.CONVECTION_SCHEMES, sig=arm_sig)
    driver._guard_merge_inputs(full, None, allow_partial=False,
                               reference_dir="/oracle/sam300")
    with pytest.raises(SystemExit, match="different reference"):
        driver._guard_merge_inputs(full, None, allow_partial=False,
                                   reference_dir="/oracle/somethingelse")


def test_merge_refuses_an_unstamped_checkpoint():
    full = _results_for(driver.CONVECTION_SCHEMES)
    full[0].signature = {}
    with pytest.raises(SystemExit, match="unstamped"):
        driver._guard_merge_inputs(full, _sig(), allow_partial=False)


def test_a_per_scheme_tuning_budget_is_not_a_mismatch():
    """The budget scales with each scheme's parameter count BY DESIGN; treating
    that as a protocol difference would refuse every real campaign."""
    full = _results_for(driver.CONVECTION_SCHEMES)
    for i, r in enumerate(full):
        r.signature = _sig(tune_evals=48 + 12 * i)
    driver._guard_merge_inputs(full, _sig(tune_evals=48), allow_partial=False)


def test_the_row_reports_the_scheme_s_own_budget():
    r = _result("bechtold")
    r.signature = _sig(tune_evals=228)
    r.records = [object()] * 19
    row = driver._row(r, _ref_stub())
    assert row["tune_evals"] == 228
    assert row["evals_per_param"] == pytest.approx(12.0)


def test_run_meta_carries_no_campaign_wide_tuning_budget():
    """It varies per scheme, so a single value in the shared file would be the
    last finisher's — and every scheme would be reported under it."""
    src = (driver.Path(__file__).resolve().parents[2] / "scripts" / "run"
           / "run_scm_rce_convection_intercomparison.py").read_text()
    meta_block = src.split("    meta = dict(")[1].split(")\n")[0]
    assert "tune_evals" not in meta_block


# --------------------------------------------------------------------------- #
# The boundary-layer SST anchor.  BL_TOP_M = 0.0 reads as "disabled" but the
# mask is `z_above_lowest <= BL_TOP_M` and z is exactly 0 at the lowest level,
# so it anchors that level and the sensible heat flux is identically zero.
# --------------------------------------------------------------------------- #

def test_bl_anchor_defaults_to_the_shipped_value():
    """Unchanged by default: the 2026-08-10 arms must stay reproducible."""
    assert _args().bl_anchor_top_m == driver.camp.DEFAULT_SCM_RCE_BL_TOP_M
    assert driver.camp.DEFAULT_SCM_RCE_BL_TOP_M == 0.0


def test_a_negative_depth_disables_the_anchor():
    assert _args("--bl-anchor-top-m", "-1").bl_anchor_top_m == -1.0


def test_the_anchor_is_a_signature_field():
    """A column with no sensible heat flux is a different experiment, so its
    checkpoint must not merge into a table built without the anchor."""
    assert "bl_anchor_top_m" in driver._SIGNATURE_FIELDS
    a = driver._run_signature(_args())
    b = driver._run_signature(_args("--bl-anchor-top-m", "-1"))
    assert a != b
    assert {k: v for k, v in a.items() if k != "bl_anchor_top_m"} == \
           {k: v for k, v in b.items() if k != "bl_anchor_top_m"}


def test_zero_depth_selects_the_lowest_level_and_negative_selects_none():
    """The arithmetic the flag exists for, asserted directly: z_above_lowest is
    0 at the lowest level, so `0 <= 0` anchors it and `0 <= -1` does not."""
    import numpy as _np
    z_above_lowest = _np.array([2000.0, 500.0, 0.0])   # top -> surface
    assert (z_above_lowest <= 0.0).tolist() == [False, False, True]
    assert (z_above_lowest <= -1.0).tolist() == [False, False, False]


def test_both_consumers_accept_every_shared_kwarg():
    """evaluate_scheme builds ONE `common` dict and splats it into BOTH
    run_cached and tune_category_winner. Adding a knob to one and not the other
    is a TypeError that only appears once a real arm runs -- it cost ten array
    tasks and eight minutes each (job 9371827,
    "tune_category_winner() got an unexpected keyword argument
    'bl_anchor_top_m'"), because no test exercised that path end to end.

    The `common` keys are read from the DRIVER's source, so a knob added there
    tomorrow is covered without editing this test.
    """
    import inspect
    import re

    src = (driver.Path(__file__).resolve().parents[2] / "scripts" / "run"
           / "run_scm_rce_convection_intercomparison.py").read_text()
    block = src.split("    common = dict(")[1].split("\n    )")[0]
    keys = set(re.findall(r"^\s*(\w+)\s*=", block, flags=re.M))
    assert "days" in keys and "bl_anchor_top_m" in keys, keys

    for fn in (driver.camp.run_cached, driver.camp.tune_category_winner):
        params = set(inspect.signature(fn).parameters)
        missing = sorted(keys - params)
        assert not missing, f"{fn.__name__} cannot accept {missing}"
