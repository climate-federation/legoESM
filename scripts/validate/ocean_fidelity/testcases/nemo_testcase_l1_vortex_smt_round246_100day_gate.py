#!/usr/bin/env python3
"""Fail-closed gate for the round-246 SMT-4 100-day measurement."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_l1_vortex_smt_round242_100day_gate import (  # noqa: E402
    validate_visuals,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round246")
ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
    "oracle_vortex_smt4/day100")
SMT3_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round245")
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 60, 100)
FIELDS = ("T", "u", "v", "ssh")
PREVIOUS_DAY100_T_RMS_K = 2.5527080520554426e-04
SMT3_DAY100_T_RMS_K = 1.7729713625071864e-04


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text())


def validate_admission(oracle: Path) -> dict:
    stem = "vortex_round237_smt4_vec_100d_admission"
    admitted = _load(oracle / f"{stem}.json")
    require(admitted.get("status") == "ADMITTED",
            "SMT-4 daily admission is not ADMITTED")
    require(admitted.get("restart_byte_identical") is True,
            "SMT-4 reference restart is not byte-identical")
    plant_status = {}
    for kind in ("header", "field-name", "truncated"):
        planted = _load(oracle / f"{stem}_plant_{kind}.json")
        plant_status[kind] = planted.get("status")
        require(plant_status[kind] == "REFUSED",
                f"SMT-4 admission {kind} plant did not refuse")
    return {
        "status": admitted["status"],
        "restart_byte_identical": True,
        "records": len(admitted.get("records", [])),
        "plant_status": plant_status,
    }


def validate(score: dict, smt3_score: dict, *, root: Path = ROOT,
             oracle: Path = ORACLE) -> dict:
    require(score.get("format") == "nemo-testcase-l1-vortex-round210-100day-v1",
            "unexpected score format")
    require(set(score.get("cards", {})) == {"smt4"},
            "score must contain exactly the SMT-4 card")
    card = score["cards"]["smt4"]
    require(card.get("case") == "VORTEX_SMT4_VEC-zps", "wrong SMT-4 card")
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
    snapshots = sorted((root / "lego_smt4").glob("day[0-9][0-9][0-9].npz"))
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

    smt3 = smt3_score.get("cards", {}).get("smt3")
    require(smt3 is not None, "SMT-3 comparison reference is missing")
    require(smt3.get("case") == "VORTEX_SMT3_VEC-zps",
            "wrong SMT-3 comparison reference")
    require(smt3["rows"]["100"]["T_rms"] == SMT3_DAY100_T_RMS_K,
            "SMT-3 day-100 T reference moved")

    day100_T = float(rows["100"]["T_rms"])
    require(float(rows["100"]["ssh_max"]) > 0.0,
            "fixed day-100 SSH-difference scale is zero")
    predictions = {
        "day100_T_reproduces_round238": {
            "status": ("CONFIRMED" if day100_T == PREVIOUS_DAY100_T_RMS_K
                       else "REFUTED"),
            "measured_K": day100_T,
            "expected_K": PREVIOUS_DAY100_T_RMS_K,
        },
        "day100_T_above_smt3": {
            "status": ("CONFIRMED" if day100_T > SMT3_DAY100_T_RMS_K
                       else "REFUTED"),
            "measured_K": day100_T,
            "bound_K": SMT3_DAY100_T_RMS_K,
        },
    }

    table = {}
    for day in CHECKPOINTS:
        smt4_row = rows[str(day)]
        smt3_row = smt3["rows"][str(day)]
        table[str(day)] = {
            "smt4": smt4_row,
            "smt3": smt3_row,
            "ratio_smt4_to_smt3": {
                name: float(smt4_row[name]) / float(smt3_row[name])
                for name in smt4_row
            },
        }

    return {
        "format": "nemo-testcase-l1-vortex-smt-round246-gate-v1",
        "oracle_root": str(oracle),
        "admission": admission,
        "nemo_daily_restarts": len(restarts),
        "lego_daily_snapshots": len(snapshots),
        "short_run_status": card["kt1_10_sanity"]["status"],
        "day100_T_rms_K": day100_T,
        "day100_T_ratio_to_smt3": day100_T / SMT3_DAY100_T_RMS_K,
        "predictions": predictions,
        "checkpoints": table,
        "visuals": validate_visuals(root, "smt4"),
        "status": "PASS",
    }


def plant(score: dict, smt3_score: dict, kind: str) -> None:
    if kind == "checkpoint":
        del score["cards"]["smt4"]["rows"]["60"]
    elif kind == "reproduction":
        score["cards"]["smt4"]["rows"]["100"]["T_rms"] = float(
            np.nextafter(PREVIOUS_DAY100_T_RMS_K, math.inf))
    elif kind == "ordering":
        score["cards"]["smt4"]["rows"]["100"]["T_rms"] = (
            SMT3_DAY100_T_RMS_K * 0.99)
    else:  # pragma: no cover - argparse owns this branch
        raise GateError(f"unknown plant {kind}")
    report = validate(score, smt3_score)
    if kind == "reproduction":
        require(report["predictions"]["day100_T_reproduces_round238"]["status"]
                == "CONFIRMED", "reproduction prediction changed to REFUTED")
    if kind == "ordering":
        require(report["predictions"]["day100_T_above_smt3"]["status"]
                == "CONFIRMED", "SMT-3 ordering prediction changed to REFUTED")
    raise GateError(f"{kind} plant did not fire")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--smt3-root", type=Path, default=SMT3_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant",
                        choices=("checkpoint", "reproduction", "ordering"))
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp

        try:
            stamp = worktree_stamp()
        except RuntimeError as error:
            raise GateError(str(error)) from error
        score = _load(args.root / "round210_scores.json")
        smt3_score = _load(args.smt3_root / "round210_scores.json")
        if args.plant:
            plant(score, smt3_score, args.plant)
        report = validate(score, smt3_score, root=args.root,
                          oracle=args.oracle)
        report["gate_worktree"] = stamp
    except GateError as error:
        prefix = "STATUS PLANT-FIRED" if args.plant else "REFUSE"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 1
    output = args.output or args.root / "round246_gate.json"
    output.write_text(json.dumps(report, indent=2))
    print("STATUS PASS: "
          f"short={report['short_run_status']} "
          f"day100_T_rms={report['day100_T_rms_K']:.17e} K "
          f"T_ratio={report['day100_T_ratio_to_smt3']:.9f} "
          f"restarts={report['nemo_daily_restarts']} "
          f"frames={report['visuals']['gif_frames']}")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
