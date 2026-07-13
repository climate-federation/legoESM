"""Unit tests for the SCM RCE convection-scheme intercomparison driver.

Fast tests use synthetic ``RunDiagnostics``/``ReferenceProfiles`` so no SCM model
is run; they exercise the reporting + plotting + CLI-validation logic. The
end-to-end SCM evaluation is covered by the campaign's own tests.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = _load(
    "run_scm_rce_convection_intercomparison",
    "scripts/run/run_scm_rce_convection_intercomparison.py",
)
camp = driver.camp


NLEV = 8


def _ref():
    z = np.linspace(0.0, 16_000.0, NLEV)
    return camp.ReferenceProfiles(
        z_m=z,
        sigma_half=np.linspace(1.0, 0.0, NLEV),
        sigma_full=np.linspace(1.0, 0.0, NLEV),
        mass_weights=np.full(NLEV, 1.0 / NLEV),
        T_ref=np.linspace(300.0, 200.0, NLEV),
        qv_ref=np.linspace(0.018, 1e-6, NLEV),
        qcond_ref=np.full(NLEV, 1e-4),
        files_used=["vol_a.npz"],
        sfc_cross_check={},
        precip_ref_mm_day=7.16,
    )


def _run(score, *, physical=True, offset=0.0, nlev=NLEV):
    # physical=True -> diagnostics inside every campaign threshold; else violate one.
    return camp.RunDiagnostics(
        label="x", config={"convection": "dca"}, status="ok", reason="",
        T_rmse=score, qv_rmse=score, cloud_rmse=score, precip_rmse=score,
        score=score,
        drift_T_rmse_K=0.1, drift_qv_rmse=1e-4, drift_qcond_rmse=1e-5,
        precip_mm_day=6.0, precip_ref_mm_day=7.16, realism_status="not_checked",
        moist_adiabat_mean_abs_K=2.0 if physical else 50.0,
        moist_adiabat_max_abs_K=10.0 if physical else 100.0,
        cold_point_T_K=195.0 if physical else 250.0,
        cold_point_z_km=17.0,
        T_profile=(np.linspace(300.0, 200.0, nlev) + offset).tolist(),
        qv_profile=(np.linspace(0.018, 1e-6, nlev)).tolist(),
        qcond_profile=np.full(nlev, 1e-4).tolist(),
    )


def _record(default=1.0, tuned=0.5):
    return camp.TuneRecord(
        category="convection", scheme="dca", scheme_key="dca", parameter="tau_s",
        default=default, tuned=tuned, lower=0.0, upper=2.0, units="s",
        score_default=2.0, score_tuned=1.0,
    )


def _results():
    return [
        driver.SchemeResult("dca", _run(2.0), _run(1.0), [_record()]),
        driver.SchemeResult(
            "kuo", _run(3.0, physical=False), _run(2.5, physical=False), []
        ),
    ]


def test_convection_schemes_match_campaign():
    assert driver.CONVECTION_SCHEMES == camp.SCHEME_SWEEPS["convection"]


def test_physical_verdict_mirrors_thresholds():
    assert driver._physical_verdict(_run(1.0, physical=True)) == "physical"
    assert driver._physical_verdict(_run(1.0, physical=False)) == "unphysical"
    assert driver._physical_verdict(_run(float("inf"))) == "nonfinite"


def test_verdict_honors_campaign_failed_status():
    # A campaign-FAILED run (status != "ok") whose realism scalars would
    # otherwise pass must still read "unphysical" — the campaign's hard checks
    # (negative q_v, invalid precip, ...) win over the stored realism scalars.
    run = dataclasses.replace(
        _run(1.0, physical=True), status="failed", reason="negative q_v (min=-1e-3)"
    )
    assert driver._physical_verdict(run) == "unphysical"


def test_checkpoint_signature_guard_mechanism(tmp_path):
    # The --quick-then-full footgun is caught on the COMPUTE path: a checkpoint
    # stamped with one signature is recomputed (not reused) under a different
    # one, while a matching or legacy/unstamped checkpoint is reused.  The merge
    # itself always aggregates whatever is present.
    res = driver.SchemeResult("dca", _run(2.0), _run(1.0), [_record()])
    quick_sig = {"days": 0.05, "tune_evals": 2}
    driver.save_scheme_result(tmp_path, res, quick_sig)
    ckpt = driver._scheme_json_path(tmp_path, "dca")

    assert driver._checkpoint_signature(ckpt) == quick_sig
    assert driver._signature_compatible(driver._checkpoint_signature(ckpt), quick_sig)
    assert not driver._signature_compatible(
        driver._checkpoint_signature(ckpt), {"days": 100.0, "tune_evals": 40})

    driver.save_scheme_result(tmp_path, res)  # no signature -> legacy {}
    assert driver._checkpoint_signature(ckpt) == {}
    assert driver._signature_compatible({}, {"days": 100.0})  # grandfathered

    # Merge aggregates unconditionally (guard is on the compute path, not here).
    assert [r.scheme for r in driver.load_all_scheme_results(tmp_path, ["dca"])] == ["dca"]


def test_run_signature_includes_seed_and_stringifies_paths(tmp_path):
    import argparse
    args = argparse.Namespace(**{k: 0 for k in driver._SIGNATURE_FIELDS})
    args.tune_seed = 7
    args.reference_dir = tmp_path  # a Path — must be stringified for JSON
    sig = driver._run_signature(args)
    assert sig["tune_seed"] == 7
    assert sig["reference_dir"] == str(tmp_path)
    json.dumps(sig)  # signature must be JSON-serializable


def test_plot_all_skips_scheme_without_profiles(tmp_path):
    # A crashed scheme has empty profiles; it must be dropped from the montage
    # even when a valid scheme sorts ahead of it (else matplotlib raises on the
    # shape mismatch).
    ref = _ref()
    good = driver.SchemeResult("dca", _run(1.0), _run(0.5), [])
    empty = dataclasses.replace(
        _run(float("inf")), T_profile=[], qv_profile=[], qcond_profile=[])
    crashed = driver.SchemeResult("kuo", empty, empty, [])
    out = tmp_path / "all.png"
    driver.plot_all(out, ref, [crashed, good])  # must not raise
    assert out.exists()


@pytest.mark.parametrize(
    "mean,mx,coldT,coldz",
    [
        (2.0, 10.0, 195.0, 17.0),   # all inside -> physical
        (50.0, 10.0, 195.0, 17.0),  # moist-adiabat mean fail
        (2.0, 100.0, 195.0, 17.0),  # moist-adiabat max fail
        (2.0, 10.0, 250.0, 17.0),   # cold-point T too warm
        (2.0, 10.0, 195.0, 30.0),   # cold-point z too high
        (float("nan"), 10.0, 195.0, 17.0),  # NaN mean -> fail closed
    ],
)
def test_verdict_agrees_with_campaign_realism_gate(mean, mx, coldT, coldz):
    # For runs with enough free-troposphere levels (the only condition our scalar
    # verdict can't see) and equilibrated drift, our verdict must match the
    # campaign's own realism gate on the shared moist-adiabat / cold-point checks.
    diag = {
        "mean_abs_K": mean, "max_abs_K": mx,
        "cold_point_T_K": coldT, "cold_point_z_km": coldz,
        "n_free_trop_levels": 12,  # >= MIN_FREE_TROP_LEVELS, so uncovered check passes
    }
    campaign_physical = len(camp.realism_reasons_from_diagnostics(diag)) == 0
    run = camp.RunDiagnostics(
        label="x", config={}, status="ok", reason="",
        T_rmse=1.0, qv_rmse=1.0, cloud_rmse=1.0, precip_rmse=1.0, score=1.0,
        drift_T_rmse_K=0.1, drift_qv_rmse=1e-4, drift_qcond_rmse=1e-5,  # equilibrated
        moist_adiabat_mean_abs_K=mean, moist_adiabat_max_abs_K=mx,
        cold_point_T_K=coldT, cold_point_z_km=coldz,
        T_profile=[300.0], qv_profile=[0.01], qcond_profile=[0.0],
    )
    verdict_physical = driver._physical_verdict(run) == "physical"
    assert verdict_physical == campaign_physical


def test_verdict_flags_equilibrium_drift():
    # Drift beyond the campaign equilibrium tolerance -> unphysical even if the
    # realism (moist-adiabat/cold-point) diagnostics are perfect.
    run = camp.RunDiagnostics(
        label="x", config={}, status="ok", reason="",
        T_rmse=1.0, qv_rmse=1.0, cloud_rmse=1.0, precip_rmse=1.0, score=1.0,
        drift_T_rmse_K=camp.EQUIL_T_TOL_K + 1.0,  # over tolerance
        drift_qv_rmse=1e-4, drift_qcond_rmse=1e-5,
        moist_adiabat_mean_abs_K=2.0, moist_adiabat_max_abs_K=10.0,
        cold_point_T_K=195.0, cold_point_z_km=17.0,
        T_profile=[300.0], qv_profile=[0.01], qcond_profile=[0.0],
    )
    assert driver._physical_verdict(run) == "unphysical"


def test_unknown_scheme_raises():
    with pytest.raises(SystemExit):
        driver.main(["--schemes", "not_a_scheme"])


def test_row_improvement_pct():
    row = driver._row(driver.SchemeResult("dca", _run(2.0), _run(1.0), []))
    assert row["score_improvement_pct"] == pytest.approx(50.0)
    # non-finite prior score -> nan improvement, no crash
    row_inf = driver._row(
        driver.SchemeResult("dca", _run(float("inf")), _run(1.0), [])
    )
    assert np.isnan(row_inf["score_improvement_pct"])


def test_write_csv_sorted_by_tuned_score(tmp_path):
    path = tmp_path / "ic.csv"
    driver.write_csv(path, _results())
    lines = path.read_text().splitlines()
    assert lines[0].startswith("scheme,")
    # dca (tuned 1.0) ranks before kuo (tuned 2.5)
    assert lines[1].startswith("dca,")
    assert lines[2].startswith("kuo,")


def test_write_tuned_parameters(tmp_path):
    path = tmp_path / "p.json"
    driver.write_tuned_parameters(path, _results())
    import json

    payload = json.loads(path.read_text())
    assert "dca" in payload and "dca.tau_s" in payload["dca"]
    assert payload["dca"]["dca.tau_s"]["tuned"] == 0.5
    assert "kuo" not in payload  # no records -> omitted


def test_write_summary_contents(tmp_path):
    path = tmp_path / "s.md"
    driver.write_summary(path, _ref(), _results(), meta=dict(
        radiation="rrtmgp", dt=600.0, days=100.0, analysis_days=5.0,
        tune_evals=48, surface_wind_m_s=5.0, large_scale_forcing="none",
        last_reference_files=5,
    ))
    text = path.read_text()
    assert "A priori vs tuned RMSE" in text
    assert "2→1" in text  # dca score prior->tuned
    assert "dca.tau_s" in text
    assert "physical" in text  # dca physical verdict reported
    assert "unphysical" in text  # kuo unphysical verdict reported


def test_scheme_result_roundtrip(tmp_path):
    res = driver.SchemeResult("dca", _run(2.0), _run(1.0), [_record()])
    driver.save_scheme_result(tmp_path, res)
    loaded = driver.load_scheme_result(tmp_path / "scheme_dca.json")
    assert loaded.scheme == "dca"
    assert loaded.prior.score == pytest.approx(2.0)
    assert loaded.tuned.score == pytest.approx(1.0)
    assert loaded.records[0].parameter == "tau_s"
    # load_all picks up only present checkpoints, in canonical scheme order
    got = driver.load_all_scheme_results(tmp_path, ("dca", "kuo"))
    assert [r.scheme for r in got] == ["dca"]


@pytest.mark.skipif(
    not (camp.DEFAULT_REFERENCE_DIR / "snapshots3d").exists(),
    reason="CRM RCEMIP1 reference volumes not present",
)
def test_merge_only_builds_summary(tmp_path):
    ref = camp.build_reference_profiles(
        camp.DEFAULT_REFERENCE_DIR, camp.DEFAULT_LAST_REFERENCE_FILES,
        precip_analysis_days=5.0,
    )
    nlev = ref.z_m.shape[0]
    for scheme, prior_s, tuned_s in (("dca", 2.0, 1.0), ("kuo", 3.0, 2.5)):
        driver.save_scheme_result(tmp_path, driver.SchemeResult(
            scheme, _run(prior_s, nlev=nlev), _run(tuned_s, nlev=nlev), []
        ))
    rc = driver.main(["--merge-only", "--outdir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "intercomparison.csv").exists()
    assert (tmp_path / "profiles_all_convection.png").stat().st_size > 0


def test_plots_written(tmp_path):
    ref = _ref()
    results = _results()
    driver.plot_scheme(tmp_path / "profiles_dca.png", ref, results[0])
    driver.plot_all(tmp_path / "profiles_all.png", ref, results)
    assert (tmp_path / "profiles_dca.png").stat().st_size > 0
    assert (tmp_path / "profiles_all.png").stat().st_size > 0
