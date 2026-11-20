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
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise C4ScoreError(f"{label} is missing or not numeric: {value!r}") from exc
    require(math.isfinite(output), f"{label} is not finite: {output}")
    return output


def load_checkpoint_report(checkpoint_dir: Path) -> dict[str, Any]:
    """Assemble one D188-compatible report from exactly 12 month checkpoints."""

    paths = sorted(checkpoint_dir.glob("c4_month??_step*.npz"))
    require(len(paths) == 12,
            f"C4 checkpoint set must contain exactly 12 months, found {len(paths)}")
    months: dict[int, dict[str, Any]] = {}
    reference_root = None
    input_identity = None
    metric_format = None
    for path in paths:
        metadata, report = runner.inspect_checkpoint(path)
        try:
            month = runner.validate_month_checkpoint(path, metadata, report)
        except RuntimeError as exc:
            raise C4ScoreError(str(exc)) from exc
        require(month not in months, f"duplicate C4 checkpoint month {month}")
        require(report.get("format") in runner.SUPPORTED_C4_METRIC_FORMATS,
                f"{path.name}: unsupported C4 metric format {report.get('format')!r}")
        require(report.get("month") == month,
                f"{path.name}: metric month disagrees with checkpoint metadata")
        require(report.get("status") == "EVALUATED_COMPLETE_MONTH",
                f"{path.name}: C4 month is not complete")
        root = str(Path(metadata["reference_root"]).resolve())
        if reference_root is None:
            reference_root = root
        require(root == reference_root, f"{path.name}: C4 reference root changed between months")
        identity = metadata.get("input_identity")
        if input_identity is None:
            input_identity = identity
        require(identity == input_identity,
                f"{path.name}: C4 input_identity changed between months")
        if metric_format is None:
            metric_format = report["format"]
        require(report["format"] == metric_format,
                f"{path.name}: C4 metric format changed between months")
        months[month] = report
    require(sorted(months) == list(range(1, 13)),
            f"C4 checkpoint months are {sorted(months)}, expected 1..12")
    try:
        current_identity = runner.checkpoint_input_identity(Path(reference_root))
    except RuntimeError as exc:
        raise C4ScoreError(str(exc)) from exc
    require(runner._resume_identity_is_compatible(input_identity, current_identity),
            "C4 deck receipt or coupled program changed since checkpoint acquisition")
    assembled = {
        "format": "orca2-r342-s4-year-metrics-v2",
        "bias_convention": "candidate - NEMO",
        "nemo_root": reference_root,
        "candidate_root": str(checkpoint_dir.resolve()),
        "candidate_is_nemo_restart": False,
        "state_averaging_before_nonlinear_metrics": False,
        "months": [months[index] for index in range(1, 13)],
        "checkpoint_input_identity": input_identity,
        "checkpoint_metric_format": metric_format,
    }
    if metric_format == runner.LEGACY_C4_METRIC_FORMAT:
        assembled["legacy_metric_limitations"] = [
            "ice volume values are normalized from km3 to 1e3 km3 on read",
            "mean daily relative ice errors were not stored and require scoring replay",
            "ICE_SST/ICE_SSS used the legacy standalone Decision-171 mask",
        ]
        for month in assembled["months"]:
            for key in ("ICE_VOLUME_NH", "ICE_VOLUME_SH"):
                row = month["metrics"].get(key)
                if row is None or row.get("units") != "km3":
                    continue
                for field in (
                    "nemo_monthly_mean", "candidate_monthly_mean", "absolute_difference",
                ):
                    row[field] = float(row[field]) / 1000.0
                row["units"] = "1e3 km3"
    return assembled


def _score_ice_month(
    candidate: dict[str, Any], envelope: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Report both ice errors and both candidate bars without choosing a verdict."""

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
        require(nemo_mean == envelope_nemo_mean,
                f"{candidate['label']}: {key} envelope and scored NEMO means differ")
        require(candidate_mean >= 0.0 and envelope_mean >= 0.0,
                f"{candidate['label']}: {key} candidate monthly mean is negative")
        value = abs(candidate_mean - nemo_mean) / abs(nemo_mean)
        envelope_value = abs(envelope_mean - envelope_nemo_mean) / abs(envelope_nemo_mean)
        daily_value = _finite(
            source.get("mean_daily_relative_abs_difference"),
            f"{key} mean daily relative error",
        )
        envelope_daily_value = _finite(
            envelope_source.get("mean_daily_relative_abs_difference"),
            f"{key} envelope mean daily relative error",
        )
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
        require(envelope_value > 0.0 and envelope_daily_value > 0.0,
                f"{candidate['label']}: {key} envelope errors must be positive")
        require(daily_value >= 0.0,
                f"{candidate['label']}: {key} mean daily relative error is negative")
        rows[key] = {
            "nemo_monthly_mean": nemo_mean,
            "candidate_monthly_mean": candidate_mean,
            "units": source["units"],
            "envelope_multiplier": ICE_ENVELOPE_MULTIPLIER,
            "floor_fraction": ICE_RELATIVE_FLOOR,
            "monthly_mean_relative_difference": {
                "candidate_error": value,
                "envelope_error": envelope_value,
                "candidate_bars": {
                    "fixed_0p01": ICE_RELATIVE_FLOOR,
                    "three_times_envelope": ICE_ENVELOPE_MULTIPLIER * envelope_value,
                },
                "would_pass_fixed_0p01": value <= ICE_RELATIVE_FLOOR,
                "would_pass_three_times_envelope": (
                    value <= ICE_ENVELOPE_MULTIPLIER * envelope_value
                ),
            },
            "mean_daily_relative_difference": {
                "candidate_error": daily_value,
                "envelope_error": envelope_daily_value,
                "candidate_bars": {
                    "fixed_0p01": ICE_RELATIVE_FLOOR,
                    "three_times_envelope": (
                        ICE_ENVELOPE_MULTIPLIER * envelope_daily_value
                    ),
                },
                "would_pass_fixed_0p01": daily_value <= ICE_RELATIVE_FLOOR,
                "would_pass_three_times_envelope": (
                    daily_value <= ICE_ENVELOPE_MULTIPLIER * envelope_daily_value
                ),
            },
            "verdict": "PENDING_USER_ICE_BAR",
        }
    return rows


def score_c4(
    candidate_report: dict[str, Any],
    envelope_report: dict[str, Any],
    envelope_admission: dict[str, Any],
) -> dict[str, Any]:
    """Apply D188 to ocean and leave the two SI3 bar candidates pending."""

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
    first_ocean_failure = None
    for ocean_month, candidate, envelope in zip(
        ocean["months"], candidate_months, envelope_months, strict=True
    ):
        ice = _score_ice_month(candidate, envelope)
        quantities = copy.deepcopy(ocean_month["metrics"])
        quantities.update(ice)
        failed = next(
            (name for name, row in ocean_month["metrics"].items() if not bool(row["pass"])),
            None,
        )
        if first_ocean_failure is None and failed is not None:
            first_ocean_failure = {"month": candidate["label"], "quantity": failed}
        months.append({
            "label": candidate["label"],
            "status": candidate["status"],
            "quantities": quantities,
            "ocean_d188_verdict": ocean_month["verdict"],
            "ice_bar_status": "PENDING_USER_ICE_BAR",
            "verdict": "PENDING_USER_ICE_BAR",
        })
    return {
        "format": "nemo-coupled-c4-score-v1",
        "decision": 228,
        "ocean_card": "DECISION_188",
        "ice_bar_decision": "PENDING_USER",
        "ice_bar_candidates": ["fixed 0.01", "3 * admitted-envelope error"],
        "ice_error_definitions": [
            "monthly-mean relative difference",
            "mean of daily relative differences",
        ],
        "hemispheres": {
            "NH": "gphit > 0 degrees",
            "SH": "gphit <= 0 degrees",
        },
        "months": months,
        "ocean_d188": {
            "verdict": ocean["verdict"],
            "months_passed": sum(
                month["ocean_d188_verdict"] == "PASS" for month in months
            ),
            "first_failure": first_ocean_failure,
        },
        "status": "PENDING_USER_ICE_BAR",
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
                "mean_daily_relative_abs_difference": relative_error,
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
    require(control["status"] == "PENDING_USER_ICE_BAR",
            "C4 synthetic control did not withhold the ice verdict")
    require(control["ocean_d188"]["verdict"] == "PASS",
            "C4 synthetic ocean control did not pass D188")

    ocean_plant = copy.deepcopy(candidate)
    ocean_plant["months"][3]["metrics"]["SST"]["rms"] = math.nextafter(0.01, math.inf)
    ocean_result = score_c4(ocean_plant, envelope, admission)
    require(ocean_result["ocean_d188"]["first_failure"]
            == {"month": "M04_Apr", "quantity": "SST"},
            "one-month ocean plant did not fail C4")

    ice_plant = copy.deepcopy(candidate)
    ice_metric = ice_plant["months"][7]["metrics"]["ICE_AREA_NH"]
    ice_metric["candidate_monthly_mean"] = 10.2
    ice_metric["absolute_difference"] = 0.2
    ice_metric["relative_abs_difference"] = 0.02
    ice_metric["mean_daily_relative_abs_difference"] = 0.02
    ice_result = score_c4(ice_plant, envelope, admission)
    ice_row = ice_result["months"][7]["quantities"]["ICE_AREA_NH"]
    require(
        not ice_row["monthly_mean_relative_difference"]["would_pass_fixed_0p01"]
        and not ice_row["monthly_mean_relative_difference"][
            "would_pass_three_times_envelope"
        ],
        "one-month ice-area plant did not fire both candidate bars",
    )
    return {
        "control": "PENDING_USER_ICE_BAR",
        "ocean_control": control["ocean_d188"]["verdict"],
        "one_month_ocean_quantity_plant": "FAIL_D188_M04_Apr_SST",
        "one_month_ice_area_plant": "BOTH_CANDIDATE_BARS_FIRE",
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
    return 2 if result["status"] == "PENDING_USER_ICE_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
