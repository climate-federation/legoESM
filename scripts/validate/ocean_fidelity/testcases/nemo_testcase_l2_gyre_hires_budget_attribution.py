#!/usr/bin/env python3
"""Rank the registered GYRE surface-V budget without hiding term boundaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_hires_budget_preflight import (  # noqa: E402
    BUDGET_DAY_KEY,
    BUDGET_MASK_KEY,
    BUDGET_TERM_KEYS,
    END_DAY,
    REGISTERED_CLOSURE_BAR_ABS_M_S,
    REGISTERED_TERMS,
    START_DAY,
    audit_budget_contract,
    sha256,
)
from nemo_testcase_l2_gyre_hires_divergence import wet_span_bins  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

COMPONENT_TERMS = REGISTERED_TERMS[:-1]
VERTICAL_EXTERNAL = ("vertical_mixing", "external_mode_correction")
TWIN_CLEAR_FACTOR = 4.0


def load_budget(
    root: Path,
    provider: str,
    nemo_precision_flavour: str = "vector-math-conda",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Reopen one admitted provider and return its digest-bound arrays."""
    audit = audit_budget_contract(root, provider, nemo_precision_flavour)
    require(audit["status"] == "PASS", f"{provider} budget contract refuses")
    contract = audit["contract"]
    artifact = root / contract["data_artifact"]
    require(sha256(artifact) == contract["data_sha256"],
            f"{provider} data digest moved after admission")
    with np.load(artifact, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    return audit, arrays


def row_metrics(
    legoesm: np.ndarray, nemo: np.ndarray, selection: np.ndarray,
) -> dict[str, float]:
    """Return the round-12 registered score for one matched term."""
    lego_values = np.asarray(legoesm, dtype=np.float64)[:, selection]
    nemo_values = np.asarray(nemo, dtype=np.float64)[:, selection]
    require(lego_values.shape == nemo_values.shape and lego_values.size > 0,
            "attribution row shapes differ or the region is empty")
    require(np.all(np.isfinite(lego_values))
            and np.all(np.isfinite(nemo_values)),
            "attribution row has non-finite active values")
    difference = lego_values - nemo_values
    nemo_rms = float(np.sqrt(np.mean(nemo_values * nemo_values,
                                     dtype=np.float64)))
    require(nemo_rms > 0.0, "registered relative score has a zero denominator")
    signed_difference = float(np.mean(difference, dtype=np.float64))
    return {
        "legoesm_signed_mean_m_s": float(np.mean(
            lego_values, dtype=np.float64)),
        "nemo_signed_mean_m_s": float(np.mean(
            nemo_values, dtype=np.float64)),
        "legoesm_rms_m_s": float(np.sqrt(np.mean(
            lego_values * lego_values, dtype=np.float64))),
        "nemo_rms_m_s": nemo_rms,
        "signed_legoesm_minus_nemo_mean_m_s": signed_difference,
        "difference_rms_m_s": float(np.sqrt(np.mean(
            difference * difference, dtype=np.float64))),
        "registered_relative_score": abs(signed_difference) / nemo_rms,
    }


def component_closure(
    legoesm: dict[str, np.ndarray], nemo: dict[str, np.ndarray], mask: np.ndarray,
) -> dict[str, float | bool]:
    """Close the cross-provider component difference to the total difference."""
    component_difference = sum(
        legoesm[f"surface_v_{term}_daily_mean"]
        - nemo[f"surface_v_{term}_daily_mean"]
        for term in COMPONENT_TERMS
    )
    total_difference = (
        legoesm["surface_v_total_increment_daily_mean"]
        - nemo["surface_v_total_increment_daily_mean"])
    residual = component_difference - total_difference
    maximum = float(np.max(np.abs(residual[:, mask])))
    return {
        "max_abs_m_s": maximum,
        "rms_m_s": float(np.sqrt(np.mean(
            residual[:, mask] ** 2, dtype=np.float64))),
        "bar_abs_m_s": REGISTERED_CLOSURE_BAR_ABS_M_S,
        "passes": maximum <= REGISTERED_CLOSURE_BAR_ABS_M_S,
    }


def state_envelope_context(path: Path) -> dict:
    """Carry forward only the preregistered field/region twin evidence."""
    report = json.loads(path.read_text())
    require(report.get("schema") == "gyre-r4-smag-divergence-envelope-v3",
            "state-envelope report schema differs")
    require(report["definition"]["regional_window_days"] == [START_DAY, END_DAY],
            "state-envelope report window differs")
    day30 = next(
        row for row in report["first_four_times_twin"]["rows"]
        if row["field"] == "surface_v")
    leading = report["leading_region_by_tied_field"]["surface_v"]
    growth = report["growth_days_30_180_by_tied_field"]["fields"]["surface_v"]
    require(leading["region"] == "western_boundary",
            "state-envelope leading surface-V region moved")
    return {
        "artifact": str(path.resolve()),
        "sha256": sha256(path),
        "surface_v_day30_candidate_to_twin_ratio": day30["ratio"],
        "western_boundary_signed_excess_share": leading[
            "signed_excess_share"],
        "days_30_180_growth_classification": growth["classification"],
        "scope": (
            "inherited field/region context only; this state-space envelope "
            "does not provide a per-term arithmetic envelope"),
    }


def build_report(
    legoesm_root: Path,
    nemo_root: Path,
    admission_path: Path,
    state_envelope_path: Path,
    nemo_twin_root: Path | None = None,
) -> dict:
    """Compute raw registered rows and the source-aligned diagnostic ranking."""
    admission = json.loads(admission_path.read_text())
    require(admission.get("verdict") == "PASS",
            "paired budget admission did not pass")
    lego_audit, lego = load_budget(legoesm_root, "legoesm")
    nemo_audit, nemo = load_budget(nemo_root, "nemo")
    require(np.array_equal(lego[BUDGET_DAY_KEY], nemo[BUDGET_DAY_KEY]),
            "provider day axes differ")
    require(np.array_equal(
        lego[BUDGET_DAY_KEY], np.arange(START_DAY, END_DAY + 1)),
        "provider day axis is not exactly the registered window")
    require(np.array_equal(lego[BUDGET_MASK_KEY], nemo[BUDGET_MASK_KEY]),
            "provider native surface-V masks differ")
    mask = np.asarray(lego[BUDGET_MASK_KEY], dtype=bool)
    western_boundary = wet_span_bins(mask) == 0
    require(np.any(western_boundary), "western-boundary wet-span bin is empty")

    closure = component_closure(lego, nemo, mask)
    require(bool(closure["passes"]),
            "cross-provider component difference does not close")

    raw_rows = []
    for term, key in zip(REGISTERED_TERMS, BUDGET_TERM_KEYS, strict=True):
        raw_rows.append({
            "term": term,
            **row_metrics(lego[key], nemo[key], western_boundary),
        })
    raw_components = raw_rows[:-1]
    raw_rank = sorted(
        raw_components, key=lambda row: row["registered_relative_score"],
        reverse=True)

    lego_group = sum(
        lego[f"surface_v_{term}_daily_mean"] for term in VERTICAL_EXTERNAL)
    nemo_group = sum(
        nemo[f"surface_v_{term}_daily_mean"] for term in VERTICAL_EXTERNAL)
    aligned_rows = [
        row for row in raw_components if row["term"] not in VERTICAL_EXTERNAL]
    aligned_rows.append({
        "term": "implicit_zdf_plus_external_mode",
        **row_metrics(lego_group, nemo_group, western_boundary),
    })
    aligned_rank = sorted(
        aligned_rows, key=lambda row: row["registered_relative_score"],
        reverse=True)
    aligned_lego_arrays = {
        term: lego[f"surface_v_{term}_daily_mean"]
        for term in COMPONENT_TERMS if term not in VERTICAL_EXTERNAL}
    aligned_nemo_arrays = {
        term: nemo[f"surface_v_{term}_daily_mean"]
        for term in COMPONENT_TERMS if term not in VERTICAL_EXTERNAL}
    aligned_lego_arrays["implicit_zdf_plus_external_mode"] = lego_group
    aligned_nemo_arrays["implicit_zdf_plus_external_mode"] = nemo_group

    twin_envelope = {
        "status": "UNMEASURED",
        "clear_factor": TWIN_CLEAR_FACTOR,
        "reason": (
            "the scalar-math NEMO twin has daily states but no matched "
            "online per-term budget; nonlinear daily states cannot "
            "reconstruct these production terms"),
    }
    if nemo_twin_root is not None:
        twin_audit, twin = load_budget(
            nemo_twin_root, "nemo", "scalar-math-conda")
        require(np.array_equal(twin[BUDGET_DAY_KEY], nemo[BUDGET_DAY_KEY]),
                "scalar/vector NEMO budget day axes differ")
        require(np.array_equal(twin[BUDGET_MASK_KEY], mask),
                "scalar/vector NEMO native surface-V masks differ")
        aligned_twin_arrays = {
            term: twin[f"surface_v_{term}_daily_mean"]
            for term in COMPONENT_TERMS if term not in VERTICAL_EXTERNAL}
        aligned_twin_arrays["implicit_zdf_plus_external_mode"] = sum(
            twin[f"surface_v_{term}_daily_mean"]
            for term in VERTICAL_EXTERNAL)
        rows = []
        for candidate_row in aligned_rank:
            term = candidate_row["term"]
            arithmetic = row_metrics(
                aligned_twin_arrays[term], aligned_nemo_arrays[term],
                western_boundary)
            candidate_abs = abs(candidate_row[
                "signed_legoesm_minus_nemo_mean_m_s"])
            twin_abs = abs(arithmetic[
                "signed_legoesm_minus_nemo_mean_m_s"])
            if twin_abs == 0.0:
                ratio = None
                clears = candidate_abs > 0.0
            else:
                ratio = candidate_abs / twin_abs
                clears = ratio >= TWIN_CLEAR_FACTOR
            rows.append({
                "term": term,
                "candidate_abs_signed_mean_difference_m_s": candidate_abs,
                "scalar_minus_vector_abs_signed_mean_difference_m_s": twin_abs,
                "candidate_to_twin_ratio": ratio,
                "clears_four_times_twin": clears,
                "scalar_minus_vector_registered_relative_score": arithmetic[
                    "registered_relative_score"],
            })
        twin_envelope = {
            "status": "MEASURED",
            "clear_factor": TWIN_CLEAR_FACTOR,
            "provider": {
                "root": str(nemo_twin_root.resolve()),
                "producer_commit": twin_audit["contract"]["producer_commit"],
                "precision_flavour": twin_audit["contract"][
                    "precision_flavour"],
                "data_sha256": twin_audit["contract"]["data_sha256"],
            },
            "rows": rows,
            "first_source_aligned_candidate_clears": rows[0][
                "clears_four_times_twin"],
        }

    raw_by_term = {row["term"]: row for row in raw_rows}
    vertical_difference = raw_by_term["vertical_mixing"][
        "signed_legoesm_minus_nemo_mean_m_s"]
    external_difference = raw_by_term["external_mode_correction"][
        "signed_legoesm_minus_nemo_mean_m_s"]
    pair_remainder = vertical_difference + external_difference
    cancellation = 1.0 - abs(pair_remainder) / max(
        abs(vertical_difference), abs(external_difference))

    return {
        "schema": "legoesm-gyre-hires-budget-attribution-v1",
        "verdict": (
            "HELD_NO_TERM_TWIN_ENVELOPE" if nemo_twin_root is None
            else "TERM_TWIN_ENVELOPE_MEASURED"),
        "window_days": [START_DAY, END_DAY],
        "field": "native surface V",
        "region": "western-boundary wet-span bin",
        "active_region_cells": int(np.count_nonzero(western_boundary)),
        "instrument": {
            "source": str(Path(__file__).resolve()),
            "source_sha256": sha256(Path(__file__)),
            "git_commit": subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[4]),
                 "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
        },
        "provider_contracts": {
            "legoesm": {
                "root": str(legoesm_root.resolve()),
                "producer_commit": lego_audit["contract"]["producer_commit"],
                "precision_flavour": lego_audit["contract"][
                    "precision_flavour"],
                "data_sha256": lego_audit["contract"]["data_sha256"],
                "closure_max_abs_m_s": lego_audit["contract"][
                    "closure_max_abs_m_s"],
            },
            "nemo": {
                "root": str(nemo_root.resolve()),
                "producer_commit": nemo_audit["contract"]["producer_commit"],
                "precision_flavour": nemo_audit["contract"][
                    "precision_flavour"],
                "data_sha256": nemo_audit["contract"]["data_sha256"],
                "closure_max_abs_m_s": nemo_audit["contract"][
                    "closure_max_abs_m_s"],
            },
            "admission_artifact": str(admission_path.resolve()),
            "admission_sha256": sha256(admission_path),
        },
        "cross_provider_component_closure": closure,
        "registered_raw_rows": raw_rows,
        "registered_raw_rank": [row["term"] for row in raw_rank],
        "registered_numerical_carrier": raw_rank[0]["term"],
        "boundary_alignment": {
            "status": "RAW_VERTICAL_AND_EXTERNAL_ROWS_NOT_LIKE_FOR_LIKE",
            "vertical_external_signed_difference_cancellation_fraction": (
                cancellation),
            "vertical_external_signed_difference_remainder_m_s": (
                pair_remainder),
            "aligned_package": "implicit_zdf_plus_external_mode",
            "reason": (
                "NEMO records the temporary prognostic barotropic removal in "
                "vertical_mixing and restores it in external_mode_correction; "
                "legoESM re-adds that mean before its vertical boundary. Only "
                "the pair has the same source extent."),
        },
        "source_aligned_diagnostic_rows": aligned_rows,
        "source_aligned_diagnostic_rank": [
            row["term"] for row in aligned_rank],
        "first_source_aligned_candidate": aligned_rank[0]["term"],
        "state_twin_envelope_context": state_envelope_context(
            state_envelope_path),
        "term_twin_envelope": twin_envelope,
        "physics_owner": "UNMEASURED",
        "shared_path_change": "NONE",
        "first_candidate_to_test": aligned_rank[0]["term"],
        "smallest_proving_test": (
            "At one common developed entry state, capture stage-3 "
            "total_vorticity_coriolis in legoESM and scalar/vector NEMO; "
            "substitute NEMO's already-formed vorticity operand into legoESM "
            "and require this row to fall to the scalar/vector NEMO envelope "
            "without moving any untouched row."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legoesm-budget-root", type=Path, required=True)
    parser.add_argument("--nemo-budget-root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--state-envelope", type=Path, required=True)
    parser.add_argument("--nemo-twin-budget-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.legoesm_budget_root,
        args.nemo_budget_root,
        args.admission,
        args.state_envelope,
        args.nemo_twin_budget_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
