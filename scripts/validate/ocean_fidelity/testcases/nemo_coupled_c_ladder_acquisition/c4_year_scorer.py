#!/usr/bin/env python3
"""Score 12 C4 monthly carry checkpoints with Decision 188 plus SI3 totals."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round345_decision188_gate as decision188,
)
from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
    c4_year_runner as runner,
)


ICE_ENVELOPE_MULTIPLIER = decision188.KE_ENVELOPE_MULTIPLIER
ICE_RELATIVE_FLOOR = decision188.KE_FLOOR_FRACTION


class C4ScoreError(RuntimeError):
    """Raised when checkpoint evidence cannot support a C4 verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C4ScoreError(message)


def _finite(value: Any, label: str) -> float:
    output = float(value)
    require(math.isfinite(output), f"{label} is not finite: {output}")
    return output


def load_checkpoint_report(checkpoint_dir: Path) -> dict[str, Any]:
    """Assemble one D188-compatible report from exactly 12 month checkpoints."""

    paths = sorted(checkpoint_dir.glob("c4_month??_step*.npz"))
    require(len(paths) == 12,
            f"C4 checkpoint set must contain exactly 12 months, found {len(paths)}")
    months: dict[int, dict[str, Any]] = {}
    reference_root = None
    for path in paths:
        metadata, report = runner.inspect_checkpoint(path)
        try:
            month = runner.validate_month_checkpoint(path, metadata, report)
        except RuntimeError as exc:
            raise C4ScoreError(str(exc)) from exc
        require(month not in months, f"duplicate C4 checkpoint month {month}")
        require(report.get("format") == runner.C4_METRIC_FORMAT,
                f"{path.name}: unsupported C4 metric format {report.get('format')!r}")
        require(report.get("month") == month,
                f"{path.name}: metric month disagrees with checkpoint metadata")
        require(report.get("status") == "EVALUATED_COMPLETE_MONTH",
                f"{path.name}: C4 month is not complete")
        root = str(Path(metadata["reference_root"]).resolve())
        if reference_root is None:
            reference_root = root
        require(root == reference_root, f"{path.name}: C4 reference root changed between months")
        months[month] = report
    require(sorted(months) == list(range(1, 13)),
            f"C4 checkpoint months are {sorted(months)}, expected 1..12")
    return {
        "format": "orca2-r342-s4-year-metrics-v2",
        "bias_convention": "candidate - NEMO",
        "nemo_root": reference_root,
        "candidate_root": str(checkpoint_dir.resolve()),
        "candidate_is_nemo_restart": False,
        "state_averaging_before_nonlinear_metrics": False,
        "months": [months[index] for index in range(1, 13)],
    }


def _score_ice_month(
    candidate: dict[str, Any], envelope: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    rows = {}
    for key in runner.ICE_KEYS:
        require(key in candidate["metrics"], f"{candidate['label']}: missing candidate {key}")
        require(key in envelope["metrics"], f"{candidate['label']}: missing envelope {key}")
        source = candidate["metrics"][key]
        envelope_source = envelope["metrics"][key]
        nemo_mean = _finite(source["nemo_monthly_mean"], f"{key} NEMO mean")
        candidate_mean = _finite(source["candidate_monthly_mean"], f"{key} candidate mean")
        envelope_nemo_mean = _finite(
            envelope_source["nemo_monthly_mean"], f"{key} envelope NEMO mean"
        )
        envelope_mean = _finite(
            envelope_source["candidate_monthly_mean"], f"{key} envelope candidate mean"
        )
        require(nemo_mean != 0.0 and envelope_nemo_mean != 0.0,
                f"{candidate['label']}: {key} NEMO monthly mean is zero")
        value = abs(candidate_mean - nemo_mean) / abs(nemo_mean)
        envelope_value = abs(envelope_mean - envelope_nemo_mean) / abs(envelope_nemo_mean)
        reported = _finite(source["relative_abs_difference"], f"{key} reported error")
        envelope_reported = _finite(
            envelope_source["relative_abs_difference"], f"{key} envelope reported error"
        )
        require(math.isclose(value, reported, rel_tol=1.0e-12, abs_tol=1.0e-15),
                f"{candidate['label']}: {key} derived error disagrees with monthly means")
        require(
            math.isclose(
                envelope_value, envelope_reported, rel_tol=1.0e-12, abs_tol=1.0e-15
            ),
            f"{candidate['label']}: {key} envelope error disagrees with monthly means",
        )
        require(value >= 0.0 and envelope_value > 0.0,
                f"{candidate['label']}: {key} candidate must be nonnegative and envelope positive")
        bar = max(ICE_RELATIVE_FLOOR, ICE_ENVELOPE_MULTIPLIER * envelope_value)
        rows[key] = {
            "nemo_monthly_mean": nemo_mean,
            "candidate_monthly_mean": candidate_mean,
            "relative_abs_difference": value,
            "envelope_relative_abs_difference": envelope_value,
            "envelope_multiplier": ICE_ENVELOPE_MULTIPLIER,
            "floor_fraction": ICE_RELATIVE_FLOOR,
            "bar_fraction": bar,
            "units": source["units"],
            "operator": "<=",
            "pass": value <= bar,
        }
    return rows


def score_c4(
    candidate_report: dict[str, Any],
    envelope_report: dict[str, Any],
    envelope_admission: dict[str, Any],
) -> dict[str, Any]:
    """Apply the executable D188 ocean card and its scalar envelope rule to SI3."""

    try:
        ocean = decision188.evaluate_decision188(
            candidate_report, envelope_report, envelope_admission
        )
        candidate_months = decision188.normalize_months(candidate_report)
        envelope_months = decision188.normalize_months(envelope_report)
    except decision188.GateError as exc:
        raise C4ScoreError(str(exc)) from exc
    require(len(ocean["months"]) == 12, "C4 D188 result did not evaluate 12 months")

    months = []
    first_failure = None
    for ocean_month, candidate, envelope in zip(
        ocean["months"], candidate_months, envelope_months, strict=True
    ):
        ice = _score_ice_month(candidate, envelope)
        quantities = copy.deepcopy(ocean_month["metrics"])
        quantities.update(ice)
        failed = next((name for name, row in quantities.items() if not bool(row["pass"])), None)
        verdict = "PASS" if failed is None else "FAIL"
        if first_failure is None and failed is not None:
            first_failure = {"month": candidate["label"], "quantity": failed}
        months.append({
            "label": candidate["label"],
            "status": candidate["status"],
            "quantities": quantities,
            "verdict": verdict,
        })
    if first_failure is None:
        status = "PASS_C4_12_OF_12"
    else:
        status = f"FAIL_C4_{first_failure['month']}_{first_failure['quantity']}"
    return {
        "format": "nemo-coupled-c4-score-v1",
        "decision": 228,
        "ocean_card": "DECISION_188",
        "ice_rule": "max(0.01, 3 * admitted-envelope relative absolute monthly-mean error)",
        "hemispheres": {
            "NH": "gphit > 0 degrees",
            "SH": "gphit <= 0 degrees",
        },
        "months": months,
        "months_passed": sum(month["verdict"] == "PASS" for month in months),
        "first_failure": first_failure,
        "status": status,
    }


def _add_synthetic_ice(report: dict[str, Any], relative_error: float) -> dict[str, Any]:
    output = copy.deepcopy(report)
    for month in output["months"]:
        for key in runner.ICE_KEYS:
            month["metrics"][key] = {
                "metric": "relative_abs_monthly_mean",
                "nemo_monthly_mean": 10.0,
                "candidate_monthly_mean": 10.0 * (1.0 + relative_error),
                "absolute_difference": 10.0 * relative_error,
                "relative_abs_difference": relative_error,
                "units": "synthetic",
            }
    return output


def run_plants() -> dict[str, str]:
    admission = decision188.minimal_envelope_admission()
    candidate = _add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.0), 0.0
    )
    envelope = _add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.001), 0.001
    )
    control = score_c4(candidate, envelope, admission)
    require(control["status"] == "PASS_C4_12_OF_12", "C4 synthetic control did not pass")

    ocean_plant = copy.deepcopy(candidate)
    ocean_plant["months"][3]["metrics"]["SST"]["rms"] = math.nextafter(0.01, math.inf)
    ocean_result = score_c4(ocean_plant, envelope, admission)
    require(ocean_result["status"].startswith("FAIL_C4_M04_Apr_SST"),
            "one-month ocean plant did not fail C4")

    ice_plant = copy.deepcopy(candidate)
    ice_metric = ice_plant["months"][7]["metrics"]["ICE_AREA_NH"]
    ice_metric["candidate_monthly_mean"] = 10.2
    ice_metric["absolute_difference"] = 0.2
    ice_metric["relative_abs_difference"] = 0.02
    ice_result = score_c4(ice_plant, envelope, admission)
    require(ice_result["status"].startswith("FAIL_C4_M08_Aug_ICE_AREA_NH"),
            "one-month ice-area plant did not fail C4")
    return {
        "control": "PASS_C4_12_OF_12",
        "one_month_ocean_quantity_plant": ocean_result["status"],
        "one_month_ice_area_plant": ice_result["status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--envelope-metrics", type=Path)
    parser.add_argument("--envelope-admission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(run_plants(), indent=2, sort_keys=True))
        return 0
    require(args.checkpoint_dir is not None, "--checkpoint-dir is required")
    require(args.envelope_metrics is not None, "--envelope-metrics is required")
    require(args.envelope_admission is not None, "--envelope-admission is required")
    candidate = load_checkpoint_report(args.checkpoint_dir)
    envelope = json.loads(args.envelope_metrics.read_text(encoding="utf-8"))
    admission = json.loads(args.envelope_admission.read_text(encoding="utf-8"))
    result = score_c4(candidate, envelope, admission)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS_C4_12_OF_12" else 1


if __name__ == "__main__":
    raise SystemExit(main())
