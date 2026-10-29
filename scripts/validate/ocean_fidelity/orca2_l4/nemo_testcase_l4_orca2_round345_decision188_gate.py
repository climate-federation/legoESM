#!/usr/bin/env python3
"""Apply Decision 188 to Decision-160 monthly metric reports.

Inputs are either round-342 month-by-month reports or the original ORCA2X
30-day pilot report.  Metric physics remains in the Decision-160 tools; this
module only validates their report schema and applies the frozen bars.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round342_s4_year_metrics as year_metrics,
)


class GateError(RuntimeError):
    """Raised when a metric report cannot support a Decision-188 verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


TEMPERATURE_KEYS = ("SST", "T_0-100m", "T_100-1000m", "T_>1000m")
SALINITY_KEYS = ("SSS", "S_0-100m", "S_100-1000m", "S_>1000m")
FIXED_RMS_BARS = {
    **{key: 0.01 for key in TEMPERATURE_KEYS},
    **{key: 0.002 for key in SALINITY_KEYS},
    "SSH": 0.001,
    "MLD": 1.0,
}
PSI_RMS_BAR_SV = 0.1
PSI_PATTERN_CORR_MIN = 0.999
DRAKE_BAR_SV = 0.5
KE_ENVELOPE_MULTIPLIER = 3.0
KE_FLOOR_FRACTION = 0.01


def _finite(value: Any, label: str) -> float:
    out = float(value)
    require(math.isfinite(out), f"{label} is not finite: {out}")
    return out


def _verify_bars_match_decision160_reporter() -> None:
    """Refuse if the reused Decision-160 reporter's named bars drift."""
    for key, bar in FIXED_RMS_BARS.items():
        reported = year_metrics.DECISION_185_REFERENCE_BARS[key]
        require(_finite(reported["reference_bar"], f"{key} reporter bar") == bar,
                f"Decision-160 reporter bar for {key} drifted from Decision 188")
    psi = year_metrics.DECISION_185_REFERENCE_BARS["PSI"]
    require(_finite(psi["reference_bar_rms_sv"], "PSI RMS reporter bar") == PSI_RMS_BAR_SV,
            "Decision-160 PSI RMS bar drifted from Decision 188")
    require(
        _finite(psi["reference_bar_pattern_corr_min"], "PSI corr reporter bar")
        == PSI_PATTERN_CORR_MIN,
        "Decision-160 PSI correlation bar drifted from Decision 188",
    )
    drake = year_metrics.DECISION_185_REFERENCE_BARS["DRAKE"]
    require(_finite(drake["reference_bar"], "Drake reporter bar") == DRAKE_BAR_SV,
            "Decision-160 Drake bar drifted from Decision 188")


def _year_months(report: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("format") == "orca2-r342-s4-year-metrics-v2",
            f"unsupported year metric format: {report.get('format')!r}")
    require(report.get("state_averaging_before_nonlinear_metrics") is False,
            "Decision-160 nonlinear metrics must be evaluated before monthly averaging")
    require(report.get("bias_convention") == "candidate - NEMO",
            "year report bias convention is not candidate - NEMO")
    raw_months = report.get("months", [])
    require(len(raw_months) == len(year_metrics.NOLEAP_MONTHS),
            "year report must contain all 12 no-leap months")
    months = []
    for month in raw_months:
        metrics = month.get("metrics")
        require(month.get("status") == "EVALUATED_COMPLETE_MONTH" and metrics is not None,
                f"{month.get('label', '<unlabelled>')}: year month is not complete")
        months.append({
            "label": str(month["label"]),
            "status": str(month["status"]),
            "available_days": list(month.get("available_days", [])),
            "saved_kts": list(month.get("saved_kts", [])),
            "metrics": metrics,
        })
    return months


def _pilot_month(report: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("format") == "orca2x-r1-pilot-metrics-v1",
            f"unsupported pilot metric format: {report.get('format')!r}")
    convention = report.get("bias_convention", "candidate - NEMO")
    require(convention == "candidate - NEMO",
            "pilot report bias convention is not candidate - NEMO")
    monthly = report.get("monthly")
    daily = report.get("daily")
    require(isinstance(monthly, dict) and isinstance(daily, list) and bool(daily),
            "ORCA2X pilot report lacks monthly or daily Decision-160 metrics")
    metrics: dict[str, Any] = {}
    for key in (*FIXED_RMS_BARS, "PSI"):
        require(key in monthly, f"ORCA2X pilot monthly metrics missing {key}")
        metrics[key] = dict(monthly[key])
    require("DRAKE" in monthly, "ORCA2X pilot monthly metrics missing DRAKE")
    drake = monthly["DRAKE"]
    metrics["DRAKE"] = {
        "abs_mean_diff_sv": abs(
            _finite(drake["cand_mean_sv"], "pilot Drake candidate mean")
            - _finite(drake["nemo_mean_sv"], "pilot Drake NEMO mean")
        )
    }
    nemo_ke = sum(
        _finite(row["KE"]["nemo_mean"], "pilot daily KE NEMO mean")
        for row in daily
    ) / len(daily)
    require(nemo_ke > 0.0, f"pilot NEMO monthly mean KE must be positive, got {nemo_ke}")
    metrics["KE"] = {
        "relative_rms_fraction": _finite(monthly["KE"]["rms"], "pilot KE RMS") / nemo_ke,
        "nemo_mean": nemo_ke,
    }
    days = [int(row.get("day", index + 1)) for index, row in enumerate(daily)]
    require(days == list(range(1, len(days) + 1)),
            f"ORCA2X pilot days must be contiguous from one, got {days[:3]}..{days[-3:]}")
    return [{
        "label": f"PILOT_DAYS_1_{len(days)}",
        "status": "EVALUATED_PARTIAL_MONTH",
        "available_days": days,
        "saved_kts": [day * year_metrics.STEPS_PER_DAY for day in days],
        "metrics": metrics,
    }]


def normalize_months(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize either supported Decision-160 report to evaluated months."""
    fmt = report.get("format")
    if fmt == "orca2-r342-s4-year-metrics-v2":
        return _year_months(report)
    if fmt == "orca2x-r1-pilot-metrics-v1":
        return _pilot_month(report)
    raise GateError(f"unsupported Decision-160 metric report format: {fmt!r}")


def _score_month(
    candidate: dict[str, Any],
    envelope: dict[str, Any],
) -> dict[str, Any]:
    cmetrics = candidate["metrics"]
    emetrics = envelope["metrics"]
    rows: dict[str, Any] = {}
    for key, bar in FIXED_RMS_BARS.items():
        require(key in cmetrics, f"{candidate['label']}: candidate metrics missing {key}")
        value = _finite(cmetrics[key]["rms"], f"{candidate['label']} {key} RMS")
        rows[key] = {"value": value, "bar": bar, "operator": "<=", "pass": value <= bar}

    require("PSI" in cmetrics, f"{candidate['label']}: candidate metrics missing PSI")
    psi_rms = _finite(cmetrics["PSI"].get("rms_sv", cmetrics["PSI"].get("rms")),
                      f"{candidate['label']} PSI RMS")
    psi_corr = _finite(cmetrics["PSI"]["pattern_corr"],
                       f"{candidate['label']} PSI correlation")
    rows["PSI"] = {
        "rms_sv": psi_rms,
        "rms_bar_sv": PSI_RMS_BAR_SV,
        "pattern_corr": psi_corr,
        "pattern_corr_min": PSI_PATTERN_CORR_MIN,
        "pass": psi_rms <= PSI_RMS_BAR_SV and psi_corr >= PSI_PATTERN_CORR_MIN,
    }

    require("DRAKE" in cmetrics, f"{candidate['label']}: candidate metrics missing DRAKE")
    drake = _finite(
        cmetrics["DRAKE"].get("abs_mean_diff_sv", cmetrics["DRAKE"].get("abs_diff_sv")),
        f"{candidate['label']} Drake absolute mean difference",
    )
    rows["DRAKE"] = {
        "value_sv": drake,
        "bar_sv": DRAKE_BAR_SV,
        "operator": "<=",
        "pass": drake <= DRAKE_BAR_SV,
    }

    require("KE" in cmetrics and "KE" in emetrics,
            f"{candidate['label']}: candidate or envelope metrics missing KE")
    candidate_ke = _finite(cmetrics["KE"]["relative_rms_fraction"],
                           f"{candidate['label']} candidate KE relative RMS")
    envelope_ke = _finite(emetrics["KE"]["relative_rms_fraction"],
                          f"{candidate['label']} envelope KE relative RMS")
    require(candidate_ke >= 0.0 and envelope_ke >= 0.0,
            f"{candidate['label']}: KE relative RMS cannot be negative")
    ke_bar = max(KE_FLOOR_FRACTION, KE_ENVELOPE_MULTIPLIER * envelope_ke)
    rows["KE"] = {
        "relative_rms_fraction": candidate_ke,
        "envelope_relative_rms_fraction": envelope_ke,
        "envelope_multiplier": KE_ENVELOPE_MULTIPLIER,
        "floor_fraction": KE_FLOOR_FRACTION,
        "bar_fraction": ke_bar,
        "operator": "<=",
        "pass": candidate_ke <= ke_bar,
    }
    passed = all(bool(row["pass"]) for row in rows.values())
    return {"label": candidate["label"], "status": candidate["status"],
            "metrics": rows, "verdict": "PASS" if passed else "FAIL"}


def _verify_envelope_admission(
    envelope_report: dict[str, Any],
    envelope_admission: dict[str, Any],
) -> None:
    """Bind the metric envelope to the admitted nonzero fp64 restart perturbation."""
    require(envelope_admission.get("status") == "PASS_R345_S4_YEAR_SST_PLUS_ONE_ULP_ARM"
            and envelope_admission.get("admitted") is True,
            "envelope lacks a passed round-345 perturbed-restart admission")
    perturbation = envelope_admission.get("restart_sst_perturbation", {})
    require(
        perturbation.get("status") == "PASS_R345_DAY1_RESTART_SST_PLUS_ONE_FP64_ULP"
        and perturbation.get("variable") == "tn"
        and perturbation.get("dtype") == "float64"
        and perturbation.get("restart_kt") == 8
        and perturbation.get("first_run_step") == 9
        and perturbation.get("exact_ulp_distance") == 1
        and int(perturbation.get("changed_values", 0)) > 0
        and perturbation.get("changed_values") == perturbation.get("selected_values")
        and perturbation.get("unselected_restart_values_and_metadata_bit_identical") is True,
        "envelope admission is zero, non-perturbed, or not the Decision-188 fp64 restart pattern",
    )
    require(envelope_admission.get("binary_equal_arm_a") is True,
            "envelope binary differs from admitted arm A")
    identity = envelope_admission.get("arm_a_input_identity", {})
    require(identity.get("forcing_and_damping_untouched") is True,
            "envelope forcing or damping inputs are not proven equal to arm A")
    control = envelope_admission.get("unperturbed_restart_control", {})
    control_input = control.get("unperturbed_restart_control_input", {})
    continuation = envelope_admission.get(
        "restart_continuation_control",
        control.get("restart_continuation_control", {}),
    )
    require(
        control.get("status") == "PASS_R345_UNPERTURBED_RESTART_CONTROL_ARM"
        and control.get("admitted") is True
        and control.get("binary_equal_arm_a") is True
        and control_input.get("status")
        == "PASS_R345_DAY1_RESTART_UNPERTURBED_CONTROL"
        and control_input.get("restart_pair_byte_identical") is True
        and continuation.get("status") == "PASS_R345_UNPERTURBED_RESTART_CONTROL"
        and continuation.get("all_daily_restarts_byte_identical") is True
        and continuation.get("restart_kt_first") == 8
        and continuation.get("restart_kt_last") == 2920
        and continuation.get("daily_restart_pairs_compared") == 365
        and continuation.get("restart_files_compared") == 730,
        "envelope lacks the passed 365-day unperturbed restart-continuation control",
    )
    require(
        Path(str(continuation.get("baseline_arm", ""))).resolve()
        == Path(str(identity.get("baseline_arm", ""))).resolve(),
        "restart control and envelope admission name different arm-A baselines",
    )
    if envelope_report.get("format") == "orca2-r342-s4-year-metrics-v2":
        require(envelope_report.get("candidate_is_nemo_restart") is True,
                "year envelope candidate must be a NEMO restart arm")
        candidate_root = Path(str(envelope_report.get("candidate_root", ""))).resolve()
        nemo_root = Path(str(envelope_report.get("nemo_root", ""))).resolve()
        admitted_root = Path(str(envelope_admission.get("run_dir", ""))).resolve()
        baseline_root = Path(str(identity.get("baseline_arm", ""))).resolve()
        require(candidate_root == admitted_root and nemo_root == baseline_root,
                "envelope report roots do not match its perturbed-arm admission")
        require(candidate_root != nemo_root,
                "envelope compares the baseline arm with itself")


def evaluate_decision188(
    candidate_report: dict[str, Any],
    envelope_report: dict[str, Any],
    envelope_admission: dict[str, Any],
) -> dict[str, Any]:
    """Apply the Decision-188 bars to matching candidate/envelope months."""
    _verify_bars_match_decision160_reporter()
    _verify_envelope_admission(envelope_report, envelope_admission)
    if candidate_report.get("format") == "orca2-r342-s4-year-metrics-v2":
        candidate_nemo_root = candidate_report.get("nemo_root")
        envelope_nemo_root = envelope_report.get("nemo_root")
        require(bool(candidate_nemo_root) and candidate_nemo_root == envelope_nemo_root,
                "year candidate and envelope must name the same NEMO baseline root")
    candidate_months = normalize_months(candidate_report)
    envelope_months = normalize_months(envelope_report)
    require(len(candidate_months) == len(envelope_months),
            "candidate and envelope reports have different evaluated-month counts")
    scored = []
    nonzero_envelope_months = 0
    for candidate, envelope in zip(candidate_months, envelope_months, strict=True):
        require(candidate["label"] == envelope["label"],
                f"month labels differ: {candidate['label']} != {envelope['label']}")
        require(candidate["available_days"] == envelope["available_days"],
                f"{candidate['label']}: candidate and envelope day windows differ")
        require(candidate["saved_kts"] == envelope["saved_kts"],
                f"{candidate['label']}: candidate and envelope saved steps differ")
        envelope_ke = _finite(
            envelope["metrics"]["KE"]["relative_rms_fraction"],
            f"{envelope['label']} envelope KE relative RMS",
        )
        require(envelope_ke > 0.0,
                f"{envelope['label']}: Decision-188 KE envelope must be > 0")
        nonzero_envelope_months += 1
        scored.append(_score_month(candidate, envelope))
    passed = all(month["verdict"] == "PASS" for month in scored)
    return {
        "format": "orca2-decision188-gate-v1",
        "decision": 188,
        "bar_mode": "DECISION_188_PASS_FAIL",
        "candidate_metric_format": candidate_report.get("format"),
        "envelope_metric_format": envelope_report.get("format"),
        "months": scored,
        "months_evaluated": len(scored),
        "nonzero_envelope_months": nonzero_envelope_months,
        "verdict": "PASS" if passed else "FAIL",
    }


def minimal_year_report(
    *, sst: float, ke: float,
    nemo_root: str = "/synthetic/nemo-baseline",
    candidate_root: str = "/synthetic/perturbed-nemo",
) -> dict[str, Any]:
    metrics = {key: {"rms": 0.0} for key in FIXED_RMS_BARS}
    metrics["SST"]["rms"] = sst
    metrics.update({
        "PSI": {"rms_sv": 0.0, "pattern_corr": 1.0},
        "DRAKE": {"abs_mean_diff_sv": 0.0},
        "KE": {"relative_rms_fraction": ke},
    })
    return {
        "format": "orca2-r342-s4-year-metrics-v2",
        "bias_convention": "candidate - NEMO",
        "nemo_root": nemo_root,
        "candidate_root": candidate_root,
        "candidate_is_nemo_restart": True,
        "state_averaging_before_nonlinear_metrics": False,
        "months": [
            {
                "label": label,
                "status": "EVALUATED_COMPLETE_MONTH",
                "available_days": [month_index],
                "saved_kts": [month_index * year_metrics.STEPS_PER_DAY],
                "metrics": copy.deepcopy(metrics),
            }
            for month_index, label, _length in year_metrics.NOLEAP_MONTHS
        ],
    }


def minimal_envelope_admission(*, changed_values: int = 4) -> dict[str, Any]:
    return {
        "status": "PASS_R345_S4_YEAR_SST_PLUS_ONE_ULP_ARM",
        "admitted": True,
        "run_dir": "/synthetic/perturbed-nemo",
        "binary_equal_arm_a": True,
        "arm_a_input_identity": {
            "baseline_arm": "/synthetic/nemo-baseline",
            "forcing_and_damping_untouched": True,
        },
        "restart_sst_perturbation": {
            "status": "PASS_R345_DAY1_RESTART_SST_PLUS_ONE_FP64_ULP",
            "variable": "tn",
            "dtype": "float64",
            "restart_kt": 8,
            "first_run_step": 9,
            "exact_ulp_distance": 1,
            "selected_values": changed_values,
            "changed_values": changed_values,
            "unselected_restart_values_and_metadata_bit_identical": True,
        },
        "unperturbed_restart_control": {
            "status": "PASS_R345_UNPERTURBED_RESTART_CONTROL_ARM",
            "admitted": True,
            "run_dir": "/synthetic/unperturbed-control",
            "binary_equal_arm_a": True,
            "unperturbed_restart_control_input": {
                "status": "PASS_R345_DAY1_RESTART_UNPERTURBED_CONTROL",
                "restart_pair_byte_identical": True,
            },
        },
        "restart_continuation_control": {
            "status": "PASS_R345_UNPERTURBED_RESTART_CONTROL",
            "baseline_arm": "/synthetic/nemo-baseline",
            "control_arm": "/synthetic/unperturbed-control",
            "restart_kt_first": 8,
            "restart_kt_last": 2920,
            "daily_restart_pairs_compared": 365,
            "restart_files_compared": 730,
            "all_daily_restarts_byte_identical": True,
        },
    }


def run_plants() -> dict[str, Any]:
    """Run the three Decision-188 non-vacuity plants."""
    admission = minimal_envelope_admission()
    envelope_floor = minimal_year_report(sst=0.0, ke=0.001)
    at_bar = evaluate_decision188(
        minimal_year_report(sst=0.01, ke=0.01), envelope_floor, admission)
    over_bar = evaluate_decision188(
        minimal_year_report(sst=math.nextafter(0.01, math.inf), ke=0.0),
        envelope_floor, admission)
    inside_envelope = evaluate_decision188(
        minimal_year_report(sst=0.0, ke=0.02),
        minimal_year_report(sst=0.0, ke=0.007),
        admission,
    )
    over_floor = evaluate_decision188(
        minimal_year_report(sst=0.0, ke=0.011), envelope_floor, admission)
    zero_caught = False
    try:
        evaluate_decision188(
            minimal_year_report(sst=0.0, ke=0.0),
            minimal_year_report(sst=0.0, ke=0.0), admission)
    except GateError:
        zero_caught = True
    partial_zero_caught = False
    try:
        partial_zero = minimal_year_report(sst=0.0, ke=0.001)
        partial_zero["months"][5]["metrics"]["KE"]["relative_rms_fraction"] = 0.0
        evaluate_decision188(
            minimal_year_report(sst=0.0, ke=0.0), partial_zero, admission)
    except GateError:
        partial_zero_caught = True
    nonperturbed_caught = False
    try:
        evaluate_decision188(
            minimal_year_report(sst=0.0, ke=0.0), envelope_floor,
            minimal_envelope_admission(changed_values=0))
    except GateError:
        nonperturbed_caught = True
    control_caught = False
    try:
        no_control = minimal_envelope_admission()
        no_control.pop("unperturbed_restart_control")
        evaluate_decision188(
            minimal_year_report(sst=0.0, ke=0.0), envelope_floor, no_control)
    except GateError:
        control_caught = True
    require(at_bar["verdict"] == "PASS", "at-bar plant did not pass")
    require(over_bar["verdict"] == "FAIL", "over-bar plant did not fail")
    require(inside_envelope["verdict"] == "PASS", "inside-envelope KE plant did not pass")
    require(over_floor["verdict"] == "FAIL", "KE floor plant did not fail")
    require(zero_caught, "zero-envelope plant did not refuse")
    require(partial_zero_caught, "partial-zero-envelope plant did not refuse")
    require(nonperturbed_caught, "non-perturbed-envelope plant did not refuse")
    require(control_caught, "missing unperturbed-control plant did not refuse")
    return {
        "fixed_bar_plant": "PASS",
        "ke_inside_3x_envelope_plant": "PASS",
        "ke_floor_plant": "PASS",
        "zero_envelope_refusal_plant": "PASS",
        "partial_zero_envelope_refusal_plant": "PASS",
        "nonperturbed_envelope_refusal_plant": "PASS",
        "missing_unperturbed_control_refusal_plant": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--envelope-metrics", type=Path)
    parser.add_argument("--envelope-admission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(run_plants(), indent=2, sort_keys=True))
        return 0
    require(args.candidate_metrics is not None and args.envelope_metrics is not None
            and args.envelope_admission is not None,
            "--candidate-metrics, --envelope-metrics, and --envelope-admission are required")
    candidate = json.loads(args.candidate_metrics.read_text(encoding="utf-8"))
    envelope = json.loads(args.envelope_metrics.read_text(encoding="utf-8"))
    admission = json.loads(args.envelope_admission.read_text(encoding="utf-8"))
    result = evaluate_decision188(candidate, envelope, admission)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
