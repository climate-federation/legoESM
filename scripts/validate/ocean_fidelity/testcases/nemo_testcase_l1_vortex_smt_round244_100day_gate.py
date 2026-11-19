#!/usr/bin/env python3
"""Fail-closed gate for the round-244 SMT-2 100-day measurement."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_l1_vortex_round210_100day_comparison import (  # noqa: E402
    load_nemo,
)
from nemo_testcase_l1_vortex_smt_round242_100day_gate import (  # noqa: E402
    validate_visuals,
)
from nemo_testcase_phase3_trajectory_gate import expected_masks  # noqa: E402


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round244")
ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/"
    "oracle_vortex_smt2/day100")
SMT1_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round242")
SMT1_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/"
    "oracle_vortex_smt1/day100")
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 60, 100)
FIELDS = ("T", "u", "v", "ssh")
SMT1_DAY100_T_RMS_K = 4.3321114781972461e-05
DAY100_T_FACTOR_BOUND = 2.0


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text())


def nemo_day100_u_max(oracle: Path, case: str) -> float:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    card = build_nemo_testcase_card(case)
    mask = expected_masks(card)["u"]
    fields = load_nemo(oracle, 100, card.recipe.z_coord.n_levels)
    return float(np.max(np.abs(fields["u"][mask])))


def validate(
    score: dict,
    smt1_score: dict,
    *,
    root: Path = ROOT,
    oracle: Path = ORACLE,
    nemo_u_maxima: tuple[float, float] | None = None,
) -> dict:
    require(score.get("format") == "nemo-testcase-l1-vortex-round210-100day-v1",
            "unexpected score format")
    require(set(score.get("cards", {})) == {"smt2"},
            "score must contain exactly the SMT-2 card")
    card = score["cards"]["smt2"]
    require(card.get("case") == "VORTEX_SMT2_VEC-zps", "wrong SMT-2 card")
    require(card["kt1_10_sanity"]["status"] == "REPRODUCED",
            "certified kt=1..10 registry did not reproduce")
    require(card["kt1_10_sanity"]["mismatches"] == [],
            "short-run calibration contains mismatches")
    require(card.get("days") == list(range(1, 101)),
            "daily score registry is incomplete")

    restarts = sorted(oracle.glob("*_restart.nc"))
    require(len(restarts) == 100,
            f"expected 100 daily NEMO restarts, found {len(restarts)}")
    restart_steps = [int(path.name.split("_")[-2]) for path in restarts]
    require(restart_steps == list(range(30, 3001, 30)),
            "NEMO daily restart cadence differs from 30..3000 by 30")
    snapshots = sorted((root / "lego_smt2").glob("day[0-9][0-9][0-9].npz"))
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

    smt1 = smt1_score.get("cards", {}).get("smt1")
    require(smt1 is not None, "SMT-1 comparison reference is missing")
    require(smt1["case"] == "VORTEX_SMT1_VEC-zps", "wrong SMT-1 reference")
    require(smt1["rows"]["100"]["T_rms"] == SMT1_DAY100_T_RMS_K,
            "SMT-1 day-100 T reference moved")
    day100_T = float(rows["100"]["T_rms"])
    T_ratio = day100_T / SMT1_DAY100_T_RMS_K
    require(float(rows["100"]["ssh_max"]) > 0.0,
            "fixed day-100 SSH-difference scale is zero")

    if nemo_u_maxima is None:
        smt2_nemo_u = nemo_day100_u_max(oracle, "VORTEX_SMT2_VEC-zps")
        smt1_nemo_u = nemo_day100_u_max(SMT1_ORACLE,
                                        "VORTEX_SMT1_VEC-zps")
    else:
        smt2_nemo_u, smt1_nemo_u = nemo_u_maxima
    require(math.isfinite(smt2_nemo_u) and math.isfinite(smt1_nemo_u),
            "NEMO day-100 U maximum is not finite")

    predictions = {
        "day100_T_within_2x_smt1": {
            "status": ("CONFIRMED" if T_ratio <= DAY100_T_FACTOR_BOUND
                       else "REFUTED"),
            "measured_ratio": T_ratio,
            "bound": DAY100_T_FACTOR_BOUND,
        },
        "nemo_day100_u_max_below_smt1": {
            "status": ("CONFIRMED" if smt2_nemo_u < smt1_nemo_u
                       else "REFUTED"),
            "smt2_m_s": smt2_nemo_u,
            "smt1_m_s": smt1_nemo_u,
        },
    }

    table = {}
    for day in CHECKPOINTS:
        smt2_row = rows[str(day)]
        smt1_row = smt1["rows"][str(day)]
        table[str(day)] = {
            "smt2": smt2_row,
            "smt1": smt1_row,
            "ratio_smt2_to_smt1": {
                name: float(smt2_row[name]) / float(smt1_row[name])
                for name in smt2_row
            },
        }

    return {
        "format": "nemo-testcase-l1-vortex-smt-round244-gate-v1",
        "oracle_root": str(oracle),
        "nemo_daily_restarts": len(restarts),
        "lego_daily_snapshots": len(snapshots),
        "short_run_status": card["kt1_10_sanity"]["status"],
        "day100_T_rms_K": day100_T,
        "day100_T_ratio_to_smt1": T_ratio,
        "nemo_day100_u_max_abs_m_s": {
            "smt2": smt2_nemo_u,
            "smt1": smt1_nemo_u,
        },
        "predictions": predictions,
        "checkpoints": table,
        "visuals": validate_visuals(root, "smt2"),
        "status": "PASS",
    }


def plant(score: dict, smt1_score: dict, kind: str) -> None:
    if kind == "checkpoint":
        del score["cards"]["smt2"]["rows"]["60"]
        kwargs = {}
    elif kind == "bound":
        score["cards"]["smt2"]["rows"]["100"]["T_rms"] = (
            DAY100_T_FACTOR_BOUND * SMT1_DAY100_T_RMS_K * 1.01)
        kwargs = {}
    elif kind == "nemo-u":
        kwargs = {"nemo_u_maxima": (2.0, 1.0)}
    else:  # pragma: no cover - argparse owns this branch
        raise GateError(f"unknown plant {kind}")
    report = validate(score, smt1_score, **kwargs)
    if kind == "bound":
        require(report["predictions"]["day100_T_within_2x_smt1"]["status"]
                == "CONFIRMED", "day-100 T prediction changed to REFUTED")
    if kind == "nemo-u":
        require(report["predictions"]["nemo_day100_u_max_below_smt1"]["status"]
                == "CONFIRMED", "NEMO U prediction changed to REFUTED")
    raise GateError(f"{kind} plant did not fire")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--smt1-root", type=Path, default=SMT1_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("checkpoint", "bound", "nemo-u"))
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp

        try:
            stamp = worktree_stamp()
        except RuntimeError as error:
            raise GateError(str(error)) from error
        score = _load(args.root / "round210_scores.json")
        smt1_score = _load(args.smt1_root / "round210_scores.json")
        if args.plant:
            plant(score, smt1_score, args.plant)
        report = validate(score, smt1_score, root=args.root,
                          oracle=args.oracle)
        report["gate_worktree"] = stamp
    except GateError as error:
        prefix = "STATUS PLANT-FIRED" if args.plant else "REFUSE"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 1
    output = args.output or args.root / "round244_gate.json"
    output.write_text(json.dumps(report, indent=2))
    print("STATUS PASS: "
          f"short={report['short_run_status']} "
          f"day100_T_rms={report['day100_T_rms_K']:.17e} K "
          f"T_ratio={report['day100_T_ratio_to_smt1']:.9f} "
          f"restarts={report['nemo_daily_restarts']} "
          f"frames={report['visuals']['gif_frames']}")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
