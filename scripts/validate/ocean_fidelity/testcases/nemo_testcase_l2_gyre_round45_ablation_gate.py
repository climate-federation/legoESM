#!/usr/bin/env python3
"""Fail-closed classifier for the preregistered round-45 GYRE ZAD arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp


ROUND45 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round45")
ROUND44 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round44")
TRAJECTORY_COMMIT = "2167274466daf576e7a243219ca22a322de3230d"
STAGE_COMMIT = "d2a761b95e33cbd67cb951b58711ab00b268804d"


class GateError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise GateError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    with path.open() as handle:
        return json.load(handle)


def residual_path(path: Path) -> Path:
    return path.with_suffix(".residuals.npz")


def kt2_uv(report: dict) -> tuple[float, float]:
    rows = {row["name"].rsplit(".", 1)[-1]: row for row in report["steps"][1]["rows"]}
    return rows["u"]["absolute_max"], rows["v"]["absolute_max"]


def run(root: Path, round44: Path, *, plant: bool = False) -> dict:
    stamp = worktree_stamp()
    files = {
        "candidate": root / "GYRE_candidate_trajectory.json",
        "legacy_internal_zad": root / "GYRE_ablate_internal_zad_trajectory.json",
        "stage_folded": root / "GYRE_ablate_stage_folded_trajectory.json",
        "implicit_matrix": root / "GYRE_ablate_implicit_matrix_trajectory.json",
        "poststep_remnant": root / "GYRE_ablate_poststep_remnant_trajectory.json",
    }
    reports = {name: read_json(path) for name, path in files.items()}
    for name, report in reports.items():
        require(report["worktree"]["clean"], f"{name}: dirty producer")
        require(report["worktree"]["commit"] == TRAJECTORY_COMMIT,
                f"{name}: unregistered producer")
        require(report["round45_ablation"] == name,
                f"{name}: wrong arm label")

    shipped_path = round44 / "restamp_GYRE_kt1_10.residuals.npz"
    rejected_path = round44 / "correct_GYRE_kt1_10.residuals.npz"
    candidate_digest = digest(residual_path(files["candidate"]))
    if plant:
        candidate_digest = "PLANTED" + candidate_digest
    require(candidate_digest == digest(rejected_path),
            "candidate does not reproduce the rejected round-44 trajectory")
    require(digest(residual_path(files["legacy_internal_zad"])) == digest(shipped_path),
            "legacy internal ZAD arm does not reproduce shipped trajectory")
    for name in ("stage_folded", "implicit_matrix", "poststep_remnant"):
        require(digest(residual_path(files[name])) == digest(rejected_path),
                f"{name}: arm was not inert")

    comparison = read_json(round44 / "correct_before_after_GYRE_compare.json")
    require(comparison["status"] == "FAIL", "round-44 Rule-12 result changed")
    require(len(comparison["violations"]) == 55, "expected 55 Rule-12 violations")

    candidate_stage_path = root / "GYRE_candidate_momentum_stages.json"
    shipped_stage_path = root / "GYRE_shipped_momentum_stages.json"
    candidate_stage = read_json(candidate_stage_path)
    shipped_stage = read_json(shipped_stage_path)
    for report in (candidate_stage, shipped_stage):
        require(report["worktree"]["clean"], "dirty stage producer")
        require(report["worktree"]["commit"] == STAGE_COMMIT,
                "unregistered stage producer")
    stage_rows = []
    first_departure = None
    c_npz = np.load(residual_path(candidate_stage_path))
    s_npz = np.load(residual_path(shipped_stage_path))
    n_captures = len(c_npz.files) // 3
    for candidate_row, shipped_row in zip(
            candidate_stage["rows"], shipped_stage["rows"], strict=True):
        require(candidate_row["name"] == shipped_row["name"],
                "stage row registry mismatch")
        # Residual capture also contains trajectory/control rows.  Resolve the
        # stage row by its independently reported population, oracle scale and
        # both candidate errors; never assume capture position.
        matches = []
        for index in range(n_captures):
            oracle = c_npz[f"r{index:05d}_oracle"]
            candidate_values = c_npz[f"r{index:05d}_candidate"]
            shipped_oracle = s_npz[f"r{index:05d}_oracle"]
            shipped_values = s_npz[f"r{index:05d}_candidate"]
            if candidate_values.size != candidate_row["n"]:
                continue
            candidate_error = float(np.max(np.abs(candidate_values - oracle)))
            shipped_error = float(np.max(np.abs(shipped_values - shipped_oracle)))
            oracle_scale = float(np.max(np.abs(oracle)))
            if (candidate_error == candidate_row["absolute_max"]
                    and shipped_error == shipped_row["absolute_max"]
                    and oracle_scale == candidate_row["reference_max_abs"]):
                matches.append(index)
        require(matches, f"no captured array for {candidate_row['name']}")
        key = f"r{matches[0]:05d}_candidate"
        movement = float(np.max(np.abs(c_npz[key] - s_npz[key])))
        row = {
            "name": candidate_row["name"],
            "candidate_absolute_max": candidate_row["absolute_max"],
            "shipped_absolute_max": shipped_row["absolute_max"],
            "candidate_vs_shipped_max": movement,
        }
        stage_rows.append(row)
        if movement and first_departure is None:
            first_departure = row
    require(first_departure is not None, "no candidate/shipped stage departure")
    require(".stage3.u" in first_departure["name"],
            f"unexpected first stage departure: {first_departure['name']}")

    candidate_uv = kt2_uv(reports["candidate"])
    shipped_uv = kt2_uv(reports["legacy_internal_zad"])
    table = []
    for name in files:
        uv = kt2_uv(reports[name])
        inert = digest(residual_path(files[name])) == candidate_digest
        table.append({
            "arm": name,
            "kt2_u": uv[0],
            "kt2_v": uv[1],
            "first_over_bar": reports[name]["first_over_bar"],
            "candidate_residuals_bit_identical": inert,
            "rule12_worsening_rows_vs_shipped": 55 if inert else 0,
            "meets_both_partner_criteria": False,
        })

    worst = max(comparison["field_moves"], key=lambda row: row[
        "max_oracle_residual_worsening_ulps"])
    return {
        "worktree": stamp,
        "format": "nemo-testcase-l2-gyre-round45-ablation-v1",
        "status": "REFUTED",
        "partner": "NONE_IN_PREREGISTERED_SET",
        "reason": "no arm keeps kt2 U/V at bar and removes all 55 worsening rows",
        "candidate_kt2_uv": {"u": candidate_uv[0], "v": candidate_uv[1]},
        "shipped_kt2_uv": {"u": shipped_uv[0], "v": shipped_uv[1]},
        "candidate_first_over_bar": reports["candidate"]["first_over_bar"],
        "ablation_table": table,
        "stage_rows": stage_rows,
        "first_candidate_vs_shipped_departure": first_departure,
        "worst_rule12_row": worst,
        "rule12": {
            "status": comparison["status"],
            "violations": len(comparison["violations"]),
            "largest_worsening_ulps": comparison[
                "largest_oracle_residual_worsening_ulps"],
        },
        "source": {
            "nemo_wzv": "stprk3_stg.f90:327-333",
            "nemo_dynadv_call": "stprk3_stg.f90:466-472",
            "nemo_zad_dispatch": "dynadv.f90:152-176",
            "nemo_zad_statement": "dynzad.f90:105-137",
            "legoesm_internal_reconstruction": "ocean_pe_latlon_cgrid.py:4749-4770",
            "legoesm_folded_term_gate": "ocean_model_latlon_cgrid.py:5927-5937",
            "legoesm_implicit_carrier": "ocean_model_latlon_cgrid.py:6877-6894",
            "legoesm_poststep_remnant": "ocean_model_latlon_cgrid.py:6905-6927",
        },
        "next_owner": {
            "status": "UNMEASURED_WITH_SPEC",
            "boundary": "kt=2 stage program producing the kt=3 T/S/U/V/ssh entry",
            "needed": "kt=2 stages 1-3 source-order operands, RHS after HPG/VOR/KEG/ZAD/LDF/ZDF, tracer completion, and external-mode frames",
            "run_spec": str(root / "run.sh"),
        },
        "artifacts": {str(path): digest(path) for path in [
            *files.values(), *(residual_path(path) for path in files.values()),
            candidate_stage_path, shipped_stage_path,
            residual_path(candidate_stage_path), residual_path(shipped_stage_path),
            round44 / "correct_before_after_GYRE_compare.json",
        ]},
        "plant": "candidate digest mismatch" if plant else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROUND45)
    parser.add_argument("--round44", type=Path, default=ROUND44)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.root, args.round44, plant=args.plant)
    except (GateError, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 1 if report["status"] == "REFUTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
