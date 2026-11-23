#!/usr/bin/env python3
"""Score 12 C4 monthly carry checkpoints with Decisions 235 and 233."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    require(not isinstance(value, bool),
            f"{label} is boolean, not numeric: {value!r}")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise C4ScoreError(f"{label} is missing or not numeric: {value!r}") from exc
    require(math.isfinite(output), f"{label} is not finite: {output}")
    return output


def load_checkpoint_report(
    checkpoint_dir: Path, *, carry_template: Any | None = None
) -> dict[str, Any]:
    """Assemble one D188-compatible report from exactly 12 month checkpoints."""

    paths = sorted(checkpoint_dir.glob("c4_month??_step*.npz"))
    require(len(paths) == 12,
            f"C4 checkpoint set must contain exactly 12 months, found {len(paths)}")
    months: dict[int, dict[str, Any]] = {}
    reference_root = None
    input_identity = None
    metric_format = None
    math_mode = None
    checkpoint_metadata = []
    for path in paths:
        try:
            metadata, report = runner.inspect_checkpoint(path)
        except RuntimeError as exc:
            raise C4ScoreError(str(exc)) from exc
        checkpoint_metadata.append((path, metadata))
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
        try:
            checkpoint_math = runner.checkpoint_math(metadata)
        except RuntimeError as exc:
            raise C4ScoreError(str(exc)) from exc
        if math_mode is None:
            math_mode = checkpoint_math
        require(checkpoint_math == math_mode,
                f"{path.name}: C4 math selection changed between months")
        if metric_format is None:
            metric_format = report["format"]
        require(report["format"] == metric_format,
                f"{path.name}: C4 metric format changed between months")
        months[month] = report
    require(sorted(months) == list(range(1, 13)),
            f"C4 checkpoint months are {sorted(months)}, expected 1..12")
    if carry_template is None:
        try:
            carry_template = runner.c_ladder_scorer._c2_cold_start(
                Path(reference_root), n_steps=2, math_mode=math_mode
            )[0]
        except (RuntimeError, ValueError) as exc:
            raise C4ScoreError(str(exc)) from exc
    _payload, expected_tree = runner._carry_payload(carry_template)
    for path, metadata in checkpoint_metadata:
        if metadata.get("input_identity") == runner.IN_FLIGHT_44512_INPUT_IDENTITY:
            leaf_count = runner.IN_FLIGHT_44512_LEAF_COUNT
            schema = metadata.get("leaf_schema")
            require(isinstance(schema, list) and len(schema) == leaf_count,
                    f"{path.name}: in-flight checkpoint must contain exactly "
                    f"{leaf_count} leaves")
            require(metadata.get("leaf_count") in (None, leaf_count),
                    f"{path.name}: in-flight checkpoint must contain exactly "
                    f"{leaf_count} leaves")
            require(metadata.get("treedef", "").count("*") == leaf_count,
                    f"{path.name}: in-flight checkpoint must contain exactly "
                    f"{leaf_count} leaves")
            require(
                [str(row.get("name", "")) for row in schema]
                == [f"carry_{index:05d}" for index in range(leaf_count)],
                f"{path.name}: in-flight checkpoint carry layout is not the "
                f"admitted {leaf_count}-leaf layout",
            )
        else:
            require(metadata.get("leaf_schema") == expected_tree["leaf_schema"],
                    f"{path.name}: carry schema differs from the deck template")
            require(metadata.get("leaf_count") == expected_tree["leaf_count"],
                    f"{path.name}: carry leaf count differs from the deck template")
            require(metadata.get("treedef") == expected_tree["treedef"],
                    f"{path.name}: carry treedef differs from the deck template")
    try:
        current_identity = runner.checkpoint_input_identity(Path(reference_root))
    except RuntimeError as exc:
        raise C4ScoreError(str(exc)) from exc
    require(runner.checkpoint_identity_is_scorable(input_identity, current_identity),
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
        "math": math_mode,
    }
    return assembled


def _score_ice_month(
    candidate: dict[str, Any], envelope: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Apply D233 to the four daily-based monthly ice errors."""

    rows = {}
    expected_units = {
        "ICE_AREA_NH": "1e6 km2",
        "ICE_AREA_SH": "1e6 km2",
        "ICE_VOLUME_NH": "1e3 km3",
        "ICE_VOLUME_SH": "1e3 km3",
    }
    for key in runner.ICE_KEYS:
        require(key in candidate["metrics"], f"{candidate['label']}: missing candidate {key}")
        require(key in envelope["metrics"], f"{candidate['label']}: missing envelope {key}")
        source = candidate["metrics"][key]
        envelope_source = envelope["metrics"][key]
        require(source.get("units") == expected_units[key],
                f"{candidate['label']}: {key} candidate units are not {expected_units[key]}")
        require(envelope_source.get("units") == expected_units[key],
                f"{candidate['label']}: {key} envelope units are not {expected_units[key]}")
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
        require(envelope_daily_value > 0.0,
                f"{candidate['label']}: {key} daily envelope error must be positive")
        require(daily_value >= 0.0,
                f"{candidate['label']}: {key} mean daily relative error is negative")
        daily_bar = max(
            ICE_RELATIVE_FLOOR,
            ICE_ENVELOPE_MULTIPLIER * envelope_daily_value,
        )
        passed = daily_value <= daily_bar
        rows[key] = {
            "nemo_monthly_mean": nemo_mean,
            "candidate_monthly_mean": candidate_mean,
            "units": source["units"],
            "envelope_multiplier": ICE_ENVELOPE_MULTIPLIER,
            "floor_fraction": ICE_RELATIVE_FLOOR,
            "monthly_mean_relative_difference": {
                "candidate_error": value,
                "envelope_error": envelope_value,
                "verdict_bearing": False,
            },
            "mean_daily_relative_abs_difference": {
                "candidate_error": daily_value,
                "envelope_error": envelope_daily_value,
                "bar": daily_bar,
                "operator": "<=",
                "pass": passed,
                "verdict_bearing": True,
            },
            "pass": passed,
            "verdict": "PASS" if passed else "FAIL",
        }
    return rows


def _score_ocean_month_decision235(
    candidate: dict[str, Any],
    envelope: dict[str, Any],
    decision188_month: dict[str, Any],
) -> dict[str, Any]:
    """Apply the C4-only D235 twin envelope to D188's fixed ocean bars."""

    rows = copy.deepcopy(decision188_month["metrics"])
    candidate_metrics = candidate["metrics"]
    envelope_metrics = envelope["metrics"]
    for key, fixed_bar in decision188.FIXED_RMS_BARS.items():
        require(key in candidate_metrics,
                f"{candidate['label']}: candidate metrics missing {key}")
        require(key in envelope_metrics, f"{candidate['label']}: twin metrics missing {key}")
        twin_value = _finite(
            envelope_metrics[key].get("rms"),
            f"{candidate['label']} {key} twin RMS",
        )
        require(twin_value >= 0.0,
                f"{candidate['label']}: {key} twin RMS is negative")
        candidate_value = _finite(
            candidate_metrics[key].get("rms"),
            f"{candidate['label']} {key} candidate RMS",
        )
        require(candidate_value >= 0.0,
                f"{candidate['label']}: {key} candidate RMS is negative")
        scaled_twin = decision188.KE_ENVELOPE_MULTIPLIER * twin_value
        applied_bar = max(fixed_bar, scaled_twin)
        binding_side = (
            "twin" if scaled_twin > fixed_bar
            else "fixed" if scaled_twin < fixed_bar
            else "fixed_and_twin"
        )
        rows[key].update({
            "candidate_value": candidate_value,
            "fixed_bar": fixed_bar,
            "twin_value": twin_value,
            "applied_bar": applied_bar,
            "binding_side": binding_side,
            "fraction_of_applied_bar": candidate_value / applied_bar,
            "bar": applied_bar,
            "pass": candidate_value <= applied_bar,
        })

    require("PSI" in envelope_metrics,
            f"{candidate['label']}: twin metrics missing PSI")
    require("rms_sv" in candidate_metrics["PSI"],
            f"{candidate['label']}: candidate PSI missing rms_sv")
    require("rms_sv" in envelope_metrics["PSI"],
            f"{candidate['label']}: twin PSI missing rms_sv")
    twin_psi_rms = _finite(
        envelope_metrics["PSI"]["rms_sv"],
        f"{candidate['label']} PSI twin RMS",
    )
    require(twin_psi_rms >= 0.0,
            f"{candidate['label']}: PSI twin RMS is negative")
    candidate_psi_rms = _finite(
        candidate_metrics["PSI"]["rms_sv"],
        f"{candidate['label']} PSI candidate RMS",
    )
    require(candidate_psi_rms >= 0.0,
            f"{candidate['label']}: PSI candidate RMS is negative")
    scaled_twin_psi_rms = decision188.KE_ENVELOPE_MULTIPLIER * twin_psi_rms
    applied_psi_rms_bar = max(decision188.PSI_RMS_BAR_SV, scaled_twin_psi_rms)
    twin_psi_corr = _finite(
        envelope_metrics["PSI"].get("pattern_corr"),
        f"{candidate['label']} PSI twin pattern correlation",
    )
    require(-1.0 <= twin_psi_corr <= 1.0,
            f"{candidate['label']}: PSI twin pattern correlation must be in [-1, 1]")
    twin_psi_decorrelation = 1.0 - twin_psi_corr
    fixed_psi_decorrelation_bar = 1.0 - decision188.PSI_PATTERN_CORR_MIN
    scaled_twin_psi_decorrelation = (
        decision188.KE_ENVELOPE_MULTIPLIER * twin_psi_decorrelation
    )
    applied_psi_decorrelation_bar = max(
        fixed_psi_decorrelation_bar, scaled_twin_psi_decorrelation
    )
    candidate_psi_corr = _finite(
        candidate_metrics["PSI"].get("pattern_corr"),
        f"{candidate['label']} PSI candidate pattern correlation",
    )
    require(
        -1.0 <= candidate_psi_corr <= 1.0,
        f"{candidate['label']}: PSI candidate pattern correlation must be in [-1, 1]",
    )
    candidate_psi_decorrelation = 1.0 - candidate_psi_corr
    rows["PSI"].update({
        "rms_candidate_value_sv": candidate_psi_rms,
        "rms_fixed_bar_sv": decision188.PSI_RMS_BAR_SV,
        "rms_twin_value_sv": twin_psi_rms,
        "rms_applied_bar_sv": applied_psi_rms_bar,
        "rms_binding_side": (
            "twin" if scaled_twin_psi_rms > decision188.PSI_RMS_BAR_SV
            else "fixed" if scaled_twin_psi_rms < decision188.PSI_RMS_BAR_SV
            else "fixed_and_twin"
        ),
        "rms_fraction_of_applied_bar": candidate_psi_rms / applied_psi_rms_bar,
        "pattern_decorrelation": candidate_psi_decorrelation,
        "pattern_decorrelation_candidate_value": candidate_psi_decorrelation,
        "pattern_decorrelation_fixed_bar": fixed_psi_decorrelation_bar,
        "pattern_decorrelation_twin_value": twin_psi_decorrelation,
        "pattern_decorrelation_applied_bar": applied_psi_decorrelation_bar,
        "pattern_decorrelation_binding_side": (
            "twin"
            if scaled_twin_psi_decorrelation > fixed_psi_decorrelation_bar
            else "fixed"
            if scaled_twin_psi_decorrelation < fixed_psi_decorrelation_bar
            else "fixed_and_twin"
        ),
        "pattern_decorrelation_fraction_of_applied_bar": (
            candidate_psi_decorrelation / applied_psi_decorrelation_bar
        ),
        "rms_bar_sv": applied_psi_rms_bar,
        "pattern_corr_min": 1.0 - applied_psi_decorrelation_bar,
        "pass": (
            candidate_psi_rms <= applied_psi_rms_bar
            and candidate_psi_decorrelation <= applied_psi_decorrelation_bar
        ),
    })

    require("DRAKE" in envelope_metrics,
            f"{candidate['label']}: twin metrics missing DRAKE")
    require("abs_mean_diff_sv" in candidate_metrics["DRAKE"],
            f"{candidate['label']}: candidate DRAKE missing abs_mean_diff_sv")
    require("abs_mean_diff_sv" in envelope_metrics["DRAKE"],
            f"{candidate['label']}: twin DRAKE missing abs_mean_diff_sv")
    twin_drake = _finite(
        envelope_metrics["DRAKE"]["abs_mean_diff_sv"],
        f"{candidate['label']} DRAKE twin absolute mean difference",
    )
    require(twin_drake >= 0.0,
            f"{candidate['label']}: DRAKE twin value is negative")
    candidate_drake = _finite(
        candidate_metrics["DRAKE"]["abs_mean_diff_sv"],
        f"{candidate['label']} DRAKE candidate absolute mean difference",
    )
    require(candidate_drake >= 0.0,
            f"{candidate['label']}: DRAKE candidate value is negative")
    scaled_twin_drake = decision188.KE_ENVELOPE_MULTIPLIER * twin_drake
    applied_drake_bar = max(decision188.DRAKE_BAR_SV, scaled_twin_drake)
    rows["DRAKE"].update({
        "candidate_value_sv": candidate_drake,
        "fixed_bar_sv": decision188.DRAKE_BAR_SV,
        "twin_value_sv": twin_drake,
        "applied_bar_sv": applied_drake_bar,
        "binding_side": (
            "twin" if scaled_twin_drake > decision188.DRAKE_BAR_SV
            else "fixed" if scaled_twin_drake < decision188.DRAKE_BAR_SV
            else "fixed_and_twin"
        ),
        "fraction_of_applied_bar": candidate_drake / applied_drake_bar,
        "bar_sv": applied_drake_bar,
        "pass": candidate_drake <= applied_drake_bar,
    })

    passed = all(bool(row["pass"]) for row in rows.values())
    return {
        "label": decision188_month["label"],
        "status": decision188_month["status"],
        "metrics": rows,
        "verdict": "PASS" if passed else "FAIL",
    }


def score_c4(
    candidate_report: dict[str, Any],
    envelope_report: dict[str, Any],
    envelope_admission: dict[str, Any],
    *,
    envelope_metrics_digest: str | None = None,
) -> dict[str, Any]:
    """Apply D235 to ocean and D233 to four SI3 quantities in every month."""

    math_mode = candidate_report.get("math")
    require(math_mode in runner.MATH_CHOICES,
            f"candidate math must be one of {runner.MATH_CHOICES}, got {math_mode!r}")

    if envelope_metrics_digest is None:
        envelope_metrics_digest = decision188.envelope_metrics_sha256(envelope_report)
    candidate_mask_sources = sorted({
        month.get("ice_sst_sss_mask_source", "UNSTAMPED")
        for month in candidate_report.get("months", [])
    })
    envelope_mask_sources = sorted({
        month.get("ice_sst_sss_mask_source", "UNSTAMPED")
        for month in envelope_report.get("months", [])
    })
    try:
        candidate_months = decision188.normalize_months(candidate_report)
        envelope_months = decision188.normalize_months(envelope_report)
        for candidate, envelope in zip(
            candidate_months, envelope_months, strict=False
        ):
            candidate_ke = candidate.get("metrics", {}).get("KE", {})
            if "relative_rms_fraction" in candidate_ke:
                _finite(
                    candidate_ke["relative_rms_fraction"],
                    f"{candidate['label']} KE candidate relative RMS",
                )
            envelope_ke = envelope.get("metrics", {}).get("KE", {})
            if "relative_rms_fraction" in envelope_ke:
                _finite(
                    envelope_ke["relative_rms_fraction"],
                    f"{candidate['label']} KE twin relative RMS",
                )
        ocean = decision188.evaluate_decision188(
            candidate_report,
            envelope_report,
            envelope_admission,
            envelope_metrics_digest=envelope_metrics_digest,
        )
    except decision188.GateError as exc:
        raise C4ScoreError(str(exc)) from exc
    require(len(ocean["months"]) == 12, "C4 D188 result did not evaluate 12 months")

    months = []
    first_ocean_failure = None
    first_failure = None
    for ocean_month, candidate, envelope in zip(
        ocean["months"], candidate_months, envelope_months, strict=True
    ):
        ocean_month = _score_ocean_month_decision235(
            candidate, envelope, ocean_month
        )
        ice = _score_ice_month(candidate, envelope)
        quantities = copy.deepcopy(ocean_month["metrics"])
        quantities.update(ice)
        failed_ocean = next(
            (name for name, row in ocean_month["metrics"].items() if not bool(row["pass"])),
            None,
        )
        if first_ocean_failure is None and failed_ocean is not None:
            first_ocean_failure = {
                "month": candidate["label"], "quantity": failed_ocean,
            }
        failed_ice = next(
            (name for name in runner.ICE_KEYS if not bool(ice[name]["pass"])),
            None,
        )
        failed = failed_ocean if failed_ocean is not None else failed_ice
        if first_failure is None and failed is not None:
            first_failure = {"month": candidate["label"], "quantity": failed}
        months.append({
            "label": candidate["label"],
            "status": candidate["status"],
            "quantities": quantities,
            "ocean_d188_verdict": ocean_month["verdict"],
            "ice_d233_verdict": "PASS" if failed_ice is None else "FAIL",
            "verdict": "PASS" if failed is None else "FAIL",
        })
    passed = first_failure is None
    return {
        "format": "nemo-coupled-c4-score-v2",
        "math": math_mode,
        "decision": 228,
        "ocean_card": "DECISION_188",
        "ocean_bar_decision": 235,
        "ocean_bar": {
            "formula": (
                "applied_bar = max(Decision-188 fixed bar, "
                "KE_ENVELOPE_MULTIPLIER * admitted coupled NEMO twin value "
                "for the same month and metric)"
            ),
            "psi_pattern_formula": (
                "allowed_decorrelation = max(1 - PSI_PATTERN_CORR_MIN, "
                "KE_ENVELOPE_MULTIPLIER * "
                "(1 - admitted coupled NEMO twin pattern_corr))"
            ),
            "fixed_bar_source_decision": 188,
            "envelope_multiplier": decision188.KE_ENVELOPE_MULTIPLIER,
            "verdict_bearing_quantities": [
                *decision188.FIXED_RMS_BARS,
                "PSI",
                "DRAKE",
                "KE",
            ],
            "unchanged_quantities": ["KE"],
        },
        "ocean_reference_only_metrics": {
            "ICE_SST": {
                "verdict_bearing": False,
                "reference_bar_rms": 0.01,
                "required_mask": "Decision-188 ice_cover_clim.nc",
            },
            "ICE_SSS": {
                "verdict_bearing": False,
                "reference_bar_rms": 0.002,
                "required_mask": "Decision-188 ice_cover_clim.nc",
            },
            "candidate_reported_masks": candidate_mask_sources,
            "envelope_reported_masks": envelope_mask_sources,
            "reason": "Decision 188 excludes both keys from FIXED_RMS_BARS",
        },
        "ice_bar_decision": 233,
        "ice_bar": {
            "formula": (
                "mean_daily_relative_abs_difference <= "
                "max(KE_FLOOR_FRACTION, KE_ENVELOPE_MULTIPLIER * "
                "admitted_coupled_NEMO_twin_mean_daily_relative_abs_difference)"
            ),
            "constants_source": "Decision 188",
            "floor_fraction": ICE_RELATIVE_FLOOR,
            "envelope_multiplier": ICE_ENVELOPE_MULTIPLIER,
        },
        "monthly_mean_relative_difference_verdict_bearing": False,
        "hemispheres": {
            "NH": "gphit > 0 degrees",
            "SH": "gphit <= 0 degrees",
        },
        "months": months,
        "ocean_d188": {
            "fixed_bar_source_decision": 188,
            "bar_decision": 235,
            "verdict": "PASS" if first_ocean_failure is None else "FAIL",
            "months_passed": sum(
                month["ocean_d188_verdict"] == "PASS" for month in months
            ),
            "first_failure": first_ocean_failure,
        },
        "months_passed": sum(month["verdict"] == "PASS" for month in months),
        "first_failure": first_failure,
        "status": "PASS_C4_12_OF_12" if passed else "FAIL_C4",
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
                "units": "1e6 km2" if "AREA" in key else "1e3 km3",
            }
    return output


def run_plants() -> dict[str, str]:
    candidate = _add_synthetic_ice(
        decision188.minimal_year_report(sst=0.0, ke=0.0), 0.0
    )
    candidate["math"] = "libm-cpu"
    envelope = _add_synthetic_ice(
        decision188.minimal_coupled_c4_year_report(sst=0.0, ke=0.001), 0.001
    )
    for month in envelope["months"]:
        for key, fixed_bar in decision188.FIXED_RMS_BARS.items():
            month["metrics"][key]["rms"] = fixed_bar / 10.0
        month["metrics"]["PSI"]["rms_sv"] = 0.02
        month["metrics"]["PSI"]["pattern_corr"] = 0.9998
        month["metrics"]["DRAKE"]["abs_mean_diff_sv"] = 0.1
    admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=envelope
    )
    control = score_c4(candidate, envelope, admission)
    require(control["status"] == "PASS_C4_12_OF_12",
            "C4 synthetic control did not pass all 12 months")
    require(control["ocean_d188"]["verdict"] == "PASS",
            "C4 synthetic ocean control did not pass D188")

    ocean_plant = copy.deepcopy(candidate)
    ocean_plant["months"][3]["metrics"]["SST"]["rms"] = math.nextafter(0.01, math.inf)
    ocean_result = score_c4(ocean_plant, envelope, admission)
    require(ocean_result["status"] == "FAIL_C4",
            "one-month ocean plant did not return a failing process status")
    require(ocean_result["first_failure"]
            == {"month": "M04_Apr", "quantity": "SST"},
            "one-month ocean plant did not fail C4")

    adaptive_envelope = copy.deepcopy(envelope)
    adaptive_envelope["months"][1]["metrics"]["SST"]["rms"] = 0.007
    adaptive_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=adaptive_envelope
    )
    between = copy.deepcopy(candidate)
    between["months"][1]["metrics"]["SST"]["rms"] = 0.02
    between_result = score_c4(between, adaptive_envelope, adaptive_admission)
    require(between_result["status"] == "PASS_C4_12_OF_12",
            "value between the fixed and twin-bound bars did not pass")
    between_row = between_result["months"][1]["quantities"]["SST"]
    require(
        between_row["fixed_bar"] == decision188.FIXED_RMS_BARS["SST"]
        and between_row["twin_value"] == 0.007
        and between_row["applied_bar"]
        == decision188.KE_ENVELOPE_MULTIPLIER * 0.007
        and between_row["binding_side"] == "twin",
        "twin-bound pass plant did not report its bar inputs",
    )

    above_applied = copy.deepcopy(between)
    above_applied["months"][1]["metrics"]["SST"]["rms"] = math.nextafter(
        between_row["applied_bar"], math.inf
    )
    above_applied_result = score_c4(
        above_applied, adaptive_envelope, adaptive_admission
    )
    require(above_applied_result["status"] == "FAIL_C4",
            "value above the applied ocean bar passed")
    require(
        above_applied_result["first_failure"]
        == {"month": "M02_Feb", "quantity": "SST"},
        "above-applied-bar plant did not name M02 SST",
    )

    fixed_envelope = copy.deepcopy(envelope)
    fixed_envelope["months"][2]["metrics"]["SST"]["rms"] = 0.001
    fixed_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=fixed_envelope
    )
    fixed_candidate = copy.deepcopy(candidate)
    fixed_candidate["months"][2]["metrics"]["SST"]["rms"] = 0.009
    fixed_result = score_c4(fixed_candidate, fixed_envelope, fixed_admission)
    fixed_row = fixed_result["months"][2]["quantities"]["SST"]
    require(
        fixed_result["status"] == "PASS_C4_12_OF_12"
        and fixed_row["binding_side"] == "fixed"
        and fixed_row["applied_bar"] == decision188.FIXED_RMS_BARS["SST"],
        "fixed-bound ocean plant did not use the Decision-188 bar",
    )

    fixed_other_envelope = copy.deepcopy(envelope)
    fixed_other_envelope["months"][8]["metrics"]["PSI"]["rms_sv"] = 0.02
    fixed_other_envelope["months"][9]["metrics"]["PSI"]["pattern_corr"] = 0.9998
    fixed_other_envelope["months"][10]["metrics"]["DRAKE"][
        "abs_mean_diff_sv"
    ] = 0.1
    fixed_other_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=fixed_other_envelope
    )
    fixed_other_candidate = copy.deepcopy(candidate)
    fixed_other_candidate["months"][8]["metrics"]["PSI"]["rms_sv"] = 0.08
    fixed_other_candidate["months"][9]["metrics"]["PSI"]["pattern_corr"] = 0.9992
    fixed_other_candidate["months"][10]["metrics"]["DRAKE"][
        "abs_mean_diff_sv"
    ] = 0.4
    fixed_other_result = score_c4(
        fixed_other_candidate, fixed_other_envelope, fixed_other_admission
    )
    fixed_psi_rms_row = fixed_other_result["months"][8]["quantities"]["PSI"]
    fixed_psi_pattern_row = fixed_other_result["months"][9]["quantities"]["PSI"]
    fixed_drake_row = fixed_other_result["months"][10]["quantities"]["DRAKE"]
    require(
        fixed_other_result["status"] == "PASS_C4_12_OF_12"
        and fixed_psi_rms_row["rms_binding_side"] == "fixed"
        and fixed_psi_pattern_row["pattern_decorrelation_binding_side"] == "fixed"
        and fixed_drake_row["binding_side"] == "fixed",
        "fixed-bound PSI/DRAKE plants did not use the Decision-188 bars",
    )

    adaptive_other_envelope = copy.deepcopy(envelope)
    adaptive_other_envelope["months"][4]["metrics"]["PSI"]["rms_sv"] = 0.05
    adaptive_other_envelope["months"][5]["metrics"]["PSI"]["pattern_corr"] = 0.9
    adaptive_other_envelope["months"][6]["metrics"]["DRAKE"][
        "abs_mean_diff_sv"
    ] = 0.2
    adaptive_other_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=adaptive_other_envelope
    )

    psi_rms_between = copy.deepcopy(candidate)
    psi_rms_between["months"][4]["metrics"]["PSI"]["rms_sv"] = 0.125
    psi_rms_between_result = score_c4(
        psi_rms_between, adaptive_other_envelope, adaptive_other_admission
    )
    psi_rms_between_row = psi_rms_between_result["months"][4]["quantities"]["PSI"]
    require(
        psi_rms_between_result["status"] == "PASS_C4_12_OF_12"
        and psi_rms_between_row["rms_binding_side"] == "twin",
        "PSI RMS value between the fixed and twin-bound bars did not pass",
    )
    psi_rms_above = copy.deepcopy(psi_rms_between)
    psi_rms_above["months"][4]["metrics"]["PSI"]["rms_sv"] = math.nextafter(
        psi_rms_between_row["rms_applied_bar_sv"], math.inf
    )
    psi_rms_above_result = score_c4(
        psi_rms_above, adaptive_other_envelope, adaptive_other_admission
    )
    require(
        psi_rms_above_result["status"] == "FAIL_C4"
        and psi_rms_above_result["first_failure"]
        == {"month": "M05_May", "quantity": "PSI"},
        "above-applied-bar PSI RMS plant did not name M05 PSI",
    )

    psi_pattern_between = copy.deepcopy(candidate)
    psi_pattern_between["months"][5]["metrics"]["PSI"]["pattern_corr"] = 0.8
    psi_pattern_between_result = score_c4(
        psi_pattern_between, adaptive_other_envelope, adaptive_other_admission
    )
    psi_pattern_between_row = psi_pattern_between_result["months"][5][
        "quantities"
    ]["PSI"]
    require(
        psi_pattern_between_result["status"] == "PASS_C4_12_OF_12"
        and psi_pattern_between_row["pattern_decorrelation_binding_side"] == "twin",
        "PSI decorrelation between the fixed and twin-bound bars did not pass",
    )
    psi_pattern_above = copy.deepcopy(psi_pattern_between)
    psi_pattern_above_value = math.nextafter(
        psi_pattern_between_row["pattern_decorrelation_applied_bar"], math.inf
    )
    psi_pattern_above["months"][5]["metrics"]["PSI"]["pattern_corr"] = (
        1.0 - psi_pattern_above_value
    )
    psi_pattern_above_result = score_c4(
        psi_pattern_above, adaptive_other_envelope, adaptive_other_admission
    )
    require(
        psi_pattern_above_result["status"] == "FAIL_C4"
        and psi_pattern_above_result["first_failure"]
        == {"month": "M06_Jun", "quantity": "PSI"},
        "above-applied-bar PSI decorrelation plant did not name M06 PSI",
    )

    drake_between = copy.deepcopy(candidate)
    drake_between["months"][6]["metrics"]["DRAKE"]["abs_mean_diff_sv"] = 0.55
    drake_between_result = score_c4(
        drake_between, adaptive_other_envelope, adaptive_other_admission
    )
    drake_between_row = drake_between_result["months"][6]["quantities"]["DRAKE"]
    require(
        drake_between_result["status"] == "PASS_C4_12_OF_12"
        and drake_between_row["binding_side"] == "twin",
        "DRAKE value between the fixed and twin-bound bars did not pass",
    )
    drake_above = copy.deepcopy(drake_between)
    drake_above["months"][6]["metrics"]["DRAKE"][
        "abs_mean_diff_sv"
    ] = math.nextafter(drake_between_row["applied_bar_sv"], math.inf)
    drake_above_result = score_c4(
        drake_above, adaptive_other_envelope, adaptive_other_admission
    )
    require(
        drake_above_result["status"] == "FAIL_C4"
        and drake_above_result["first_failure"]
        == {"month": "M07_Jul", "quantity": "DRAKE"},
        "above-applied-bar DRAKE plant did not name M07 DRAKE",
    )

    refusal_cases = []
    negative_rms = copy.deepcopy(candidate)
    negative_rms["months"][0]["metrics"]["SST"]["rms"] = -0.001
    refusal_cases.append((
        "negative candidate RMS", negative_rms, envelope, admission,
        "SST candidate RMS is negative",
    ))
    negative_psi_rms = copy.deepcopy(candidate)
    negative_psi_rms["months"][0]["metrics"]["PSI"]["rms_sv"] = -0.001
    refusal_cases.append((
        "negative candidate PSI RMS", negative_psi_rms, envelope, admission,
        "PSI candidate RMS is negative",
    ))
    negative_drake = copy.deepcopy(candidate)
    negative_drake["months"][0]["metrics"]["DRAKE"]["abs_mean_diff_sv"] = -0.001
    refusal_cases.append((
        "negative candidate DRAKE", negative_drake, envelope, admission,
        "DRAKE candidate value is negative",
    ))
    candidate_corr = copy.deepcopy(candidate)
    candidate_corr["months"][0]["metrics"]["PSI"]["pattern_corr"] = 1.001
    refusal_cases.append((
        "candidate PSI correlation outside [-1, 1]",
        candidate_corr, envelope, admission,
        "PSI candidate pattern correlation must be in [-1, 1]",
    ))
    twin_corr = copy.deepcopy(envelope)
    twin_corr["months"][0]["metrics"]["PSI"]["pattern_corr"] = -1.001
    twin_corr_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=twin_corr
    )
    refusal_cases.append((
        "twin PSI correlation outside [-1, 1]",
        candidate, twin_corr, twin_corr_admission,
        "PSI twin pattern correlation must be in [-1, 1]",
    ))
    boolean_number = copy.deepcopy(candidate)
    boolean_number["months"][0]["metrics"]["SST"]["rms"] = True
    refusal_cases.append((
        "boolean numeric value", boolean_number, envelope, admission,
        "SST candidate RMS is boolean, not numeric",
    ))
    candidate_ke_boolean = copy.deepcopy(candidate)
    candidate_ke_boolean["months"][0]["metrics"]["KE"][
        "relative_rms_fraction"
    ] = False
    refusal_cases.append((
        "candidate KE boolean", candidate_ke_boolean, envelope, admission,
        "KE candidate relative RMS is boolean, not numeric",
    ))
    twin_ke_boolean = copy.deepcopy(envelope)
    twin_ke_boolean["months"][0]["metrics"]["KE"][
        "relative_rms_fraction"
    ] = True
    twin_ke_boolean_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=twin_ke_boolean
    )
    refusal_cases.append((
        "twin KE boolean", candidate, twin_ke_boolean, twin_ke_boolean_admission,
        "KE twin relative RMS is boolean, not numeric",
    ))
    refusal_results = {}
    for name, refusal_candidate, refusal_envelope, refusal_admission, message in refusal_cases:
        caught = False
        try:
            score_c4(refusal_candidate, refusal_envelope, refusal_admission)
        except C4ScoreError as exc:
            caught = message in str(exc)
        require(caught, f"{name} was not refused with the expected error")
        refusal_results[name] = "PASS"

    corrupt_envelope = copy.deepcopy(envelope)
    corrupt_envelope["months"][3]["metrics"]["SST"]["rms"] = -0.001
    corrupt_admission = decision188.minimal_coupled_c4_envelope_admission(
        envelope_report=corrupt_envelope
    )
    corrupt_caught = False
    try:
        score_c4(candidate, corrupt_envelope, corrupt_admission)
    except C4ScoreError:
        corrupt_caught = True
    require(corrupt_caught, "negative ocean twin value was not refused")

    ice_bar = max(
        ICE_RELATIVE_FLOOR,
        ICE_ENVELOPE_MULTIPLIER * 0.001,
    )
    above = math.nextafter(ice_bar, math.inf)
    below = math.nextafter(ice_bar, 0.0)

    ice_plant = copy.deepcopy(candidate)
    ice_metric = ice_plant["months"][7]["metrics"]["ICE_AREA_NH"]
    ice_metric["candidate_monthly_mean"] = 10.0 * (1.0 + above)
    ice_metric["absolute_difference"] = 10.0 * above
    ice_metric["relative_abs_difference"] = above
    ice_metric["mean_daily_relative_abs_difference"] = above
    ice_result = score_c4(ice_plant, envelope, admission)
    require(ice_result["status"] == "FAIL_C4", "above-bar ice plant passed")
    require(
        ice_result["first_failure"]
        == {"month": "M08_Aug", "quantity": "ICE_AREA_NH"},
        "above-bar ice plant did not name M08 ICE_AREA_NH",
    )

    below_plant = copy.deepcopy(candidate)
    below_metric = below_plant["months"][7]["metrics"]["ICE_AREA_NH"]
    below_metric["candidate_monthly_mean"] = 10.0 * (1.0 + below)
    below_metric["absolute_difference"] = 10.0 * below
    below_metric["relative_abs_difference"] = below
    below_metric["mean_daily_relative_abs_difference"] = below
    below_result = score_c4(below_plant, envelope, admission)
    require(below_result["status"] == "PASS_C4_12_OF_12",
            "below-bar ice plant failed")

    cancellation_plant = copy.deepcopy(candidate)
    cancellation_metric = cancellation_plant["months"][7]["metrics"][
        "ICE_AREA_NH"
    ]
    cancellation_metric["mean_daily_relative_abs_difference"] = above
    cancellation_result = score_c4(cancellation_plant, envelope, admission)
    require(cancellation_result["status"] == "FAIL_C4",
            "daily cancellation plant passed")
    require(
        cancellation_result["first_failure"]
        == {"month": "M08_Aug", "quantity": "ICE_AREA_NH"},
        "daily cancellation plant did not name M08 ICE_AREA_NH",
    )
    return {
        "control": "PASS_C4_12_OF_12",
        "ocean_control": control["ocean_d188"]["verdict"],
        "one_month_ocean_quantity_plant": "FAIL_C4_M04_Apr_SST",
        "ocean_between_fixed_and_twin_bound_plant": "PASS_C4_12_OF_12",
        "ocean_above_applied_bar_plant": "FAIL_C4_M02_Feb_SST",
        "ocean_fixed_bound_plant": "PASS_C4_12_OF_12",
        "psi_rms_fixed_bound_plant": "PASS_C4_12_OF_12",
        "psi_pattern_fixed_bound_plant": "PASS_C4_12_OF_12",
        "drake_fixed_bound_plant": "PASS_C4_12_OF_12",
        "psi_rms_between_fixed_and_twin_bound_plant": "PASS_C4_12_OF_12",
        "psi_rms_above_applied_bar_plant": "FAIL_C4_M05_May_PSI",
        "psi_pattern_between_fixed_and_twin_bound_plant": "PASS_C4_12_OF_12",
        "psi_pattern_above_applied_bar_plant": "FAIL_C4_M06_Jun_PSI",
        "drake_between_fixed_and_twin_bound_plant": "PASS_C4_12_OF_12",
        "drake_above_applied_bar_plant": "FAIL_C4_M07_Jul_DRAKE",
        "negative_candidate_rms_refusal_plant": refusal_results[
            "negative candidate RMS"
        ],
        "negative_candidate_psi_rms_refusal_plant": refusal_results[
            "negative candidate PSI RMS"
        ],
        "negative_candidate_drake_refusal_plant": refusal_results[
            "negative candidate DRAKE"
        ],
        "candidate_psi_correlation_refusal_plant": refusal_results[
            "candidate PSI correlation outside [-1, 1]"
        ],
        "twin_psi_correlation_refusal_plant": refusal_results[
            "twin PSI correlation outside [-1, 1]"
        ],
        "boolean_numeric_refusal_plant": refusal_results["boolean numeric value"],
        "candidate_ke_boolean_refusal_plant": refusal_results[
            "candidate KE boolean"
        ],
        "twin_ke_boolean_refusal_plant": refusal_results["twin KE boolean"],
        "corrupt_ocean_twin_refusal_plant": "PASS",
        "ice_above_bar_plant": "FAIL_C4_M08_Aug_ICE_AREA_NH",
        "ice_below_bar_plant": "PASS_C4_12_OF_12",
        "daily_cancellation_plant": "FAIL_C4_M08_Aug_ICE_AREA_NH",
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
    envelope_bytes = args.envelope_metrics.read_bytes()
    envelope = json.loads(envelope_bytes)
    admission = json.loads(args.envelope_admission.read_text(encoding="utf-8"))
    result = score_c4(
        candidate,
        envelope,
        admission,
        envelope_metrics_digest=hashlib.sha256(envelope_bytes).hexdigest(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS_C4_12_OF_12" else 1


if __name__ == "__main__":
    raise SystemExit(main())
