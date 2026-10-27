#!/usr/bin/env python3
"""Fail-closed gate for the round-245 SMT-3 100-day measurement."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_l1_vortex_smt_round242_100day_gate import (  # noqa: E402
    validate_visuals,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round245")
ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/"
    "oracle_vortex_smt3/day100")
SMT2_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round244")
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 60, 100)
FIELDS = ("T", "u", "v", "ssh")
PRE_LANDING_DAY100_T_RMS_K = 6.813785267886451e-04
SMT2_DAY100_T_RMS_K = 8.1037591477894766e-06


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text())


def validate_admission(oracle: Path) -> dict:
    stem = "vortex_round224_smt3_vec_100d_admission"
    admitted = _load(oracle / f"{stem}.json")
    require(admitted.get("status") == "ADMITTED",
            "SMT-3 daily admission is not ADMITTED")
    require(admitted.get("restart_byte_identical") is True,
            "SMT-3 reference restart is not byte-identical")
    plant_status = {}
    for kind in ("header", "field-name", "truncated"):
        planted = _load(oracle / f"{stem}_plant_{kind}.json")
        plant_status[kind] = planted.get("status")
        require(plant_status[kind] == "REFUSED",
                f"SMT-3 admission {kind} plant did not refuse")
    return {
        "status": admitted["status"],
        "restart_byte_identical": True,
        "records": len(admitted.get("records", [])),
        "plant_status": plant_status,
    }


def validate(score: dict, smt2_score: dict, *, root: Path = ROOT,
             oracle: Path = ORACLE) -> dict:
    require(score.get("format") == "nemo-testcase-l1-vortex-round210-100day-v1",
            "unexpected score format")
    require(set(score.get("cards", {})) == {"smt3"},
            "score must contain exactly the SMT-3 card")
    card = score["cards"]["smt3"]
    require(card.get("case") == "VORTEX_SMT3_VEC-zps", "wrong SMT-3 card")
    require(card["kt1_10_sanity"]["status"] == "REPRODUCED",
            "certified kt=1..10 registry did not reproduce")
    require(card["kt1_10_sanity"]["mismatches"] == [],
            "short-run calibration contains mismatches")
    require(card.get("days") == list(range(1, 101)),
            "daily score registry is incomplete")

    admission = validate_admission(oracle)
    restarts = sorted(oracle.glob("*_restart.nc"))
    require(len(restarts) == 100,
            f"expected 100 daily NEMO restarts, found {len(restarts)}")
    restart_steps = [int(path.name.split("_")[-2]) for path in restarts]
    require(restart_steps == list(range(30, 3001, 30)),
            "NEMO daily restart cadence differs from 30..3000 by 30")
    snapshots = sorted((root / "lego_smt3").glob("day[0-9][0-9][0-9].npz"))
    require(len(snapshots) == 100,
            f"expected 100 legoESM snapshots, found {len(snapshots)}")

    rows = card["rows"]
    require(set(rows) == {str(day) for day in range(1, 101)},
            "daily row registry is incomplete")
    for day in range(1, 101):
        row = rows[str(day)]
        require(set(row) == {f"{field}_{metric}" for field in FIELDS
                             for metric in ("rms", "max")},
                f"day {day}: field/metric registry differs")
        for name, value in row.items():
            require(math.isfinite(float(value)) and float(value) >= 0.0,
                    f"day {day} {name} is not finite and nonnegative")

    smt2 = smt2_score.get("cards", {}).get("smt2")
    require(smt2 is not None, "SMT-2 comparison reference is missing")
    require(smt2.get("case") == "VORTEX_SMT2_VEC-zps",
            "wrong SMT-2 comparison reference")
    require(smt2["rows"]["100"]["T_rms"] == SMT2_DAY100_T_RMS_K,
            "SMT-2 day-100 T reference moved")

    day100_T = float(rows["100"]["T_rms"])
    require(float(rows["100"]["ssh_max"]) > 0.0,
            "fixed day-100 SSH-difference scale is zero")
    predictions = {
        "day100_T_below_pre_landing": {
            "status": ("CONFIRMED" if day100_T < PRE_LANDING_DAY100_T_RMS_K
                       else "REFUTED"),
            "measured_K": day100_T,
            "bound_K": PRE_LANDING_DAY100_T_RMS_K,
        },
        "day100_T_above_smt2": {
            "status": ("CONFIRMED" if day100_T > SMT2_DAY100_T_RMS_K
                       else "REFUTED"),
            "measured_K": day100_T,
            "bound_K": SMT2_DAY100_T_RMS_K,
        },
    }

    table = {}
    for day in CHECKPOINTS:
        smt3_row = rows[str(day)]
        smt2_row = smt2["rows"][str(day)]
        table[str(day)] = {
            "smt3": smt3_row,
            "smt2": smt2_row,
            "ratio_smt3_to_smt2": {
                name: float(smt3_row[name]) / float(smt2_row[name])
                for name in smt3_row
            },
        }

    return {
        "format": "nemo-testcase-l1-vortex-smt-round245-gate-v1",
        "oracle_root": str(oracle),
        "admission": admission,
        "nemo_daily_restarts": len(restarts),
        "lego_daily_snapshots": len(snapshots),
        "short_run_status": card["kt1_10_sanity"]["status"],
        "day100_T_rms_K": day100_T,
        "day100_T_ratio_to_smt2": day100_T / SMT2_DAY100_T_RMS_K,
        "predictions": predictions,
        "checkpoints": table,
        "visuals": validate_visuals(root, "smt3"),
        "status": "PASS",
    }


def plant(score: dict, smt2_score: dict, kind: str) -> None:
    if kind == "checkpoint":
        del score["cards"]["smt3"]["rows"]["60"]
    elif kind == "upper-bound":
        score["cards"]["smt3"]["rows"]["100"]["T_rms"] = (
            PRE_LANDING_DAY100_T_RMS_K * 1.01)
    elif kind == "lower-bound":
        score["cards"]["smt3"]["rows"]["100"]["T_rms"] = (
            SMT2_DAY100_T_RMS_K * 0.99)
    else:  # pragma: no cover - argparse owns this branch
        raise GateError(f"unknown plant {kind}")
    report = validate(score, smt2_score)
    if kind == "upper-bound":
        require(report["predictions"]["day100_T_below_pre_landing"]["status"]
                == "CONFIRMED", "upper-bound prediction changed to REFUTED")
    if kind == "lower-bound":
        require(report["predictions"]["day100_T_above_smt2"]["status"]
                == "CONFIRMED", "lower-bound prediction changed to REFUTED")
    raise GateError(f"{kind} plant did not fire")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--smt2-root", type=Path, default=SMT2_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant",
                        choices=("checkpoint", "upper-bound", "lower-bound"))
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp

        try:
            stamp = worktree_stamp()
        except RuntimeError as error:
            raise GateError(str(error)) from error
        score = _load(args.root / "round210_scores.json")
        smt2_score = _load(args.smt2_root / "round210_scores.json")
        if args.plant:
            plant(score, smt2_score, args.plant)
        report = validate(score, smt2_score, root=args.root,
                          oracle=args.oracle)
        report["gate_worktree"] = stamp
    except GateError as error:
        prefix = "STATUS PLANT-FIRED" if args.plant else "REFUSE"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 1
    output = args.output or args.root / "round245_gate.json"
    output.write_text(json.dumps(report, indent=2))
    print("STATUS PASS: "
          f"short={report['short_run_status']} "
          f"day100_T_rms={report['day100_T_rms_K']:.17e} K "
          f"T_ratio={report['day100_T_ratio_to_smt2']:.9f} "
          f"restarts={report['nemo_daily_restarts']} "
          f"frames={report['visuals']['gif_frames']}")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
