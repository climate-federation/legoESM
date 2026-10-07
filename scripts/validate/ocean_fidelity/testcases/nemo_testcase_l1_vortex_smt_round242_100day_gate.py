#!/usr/bin/env python3
"""Fail-closed gate for the round-242 SMT-1 100-day measurement."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round242")
ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/"
    "oracle_vortex_smt1/day100")
FLAT_REFERENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round210/"
    "round210_scores.json")
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 60, 100)
FIELDS = ("T", "u", "v", "ssh")
DAY100_T_BOUND_K = 1.0e-3
MEASUREMENT_COMMIT = "8e0065ed08e63ae285061dd50f2190070033eb4e"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text())


def _validate_visuals(root: Path) -> dict:
    from PIL import Image

    mp4 = root / "vortex_smt1_100d.mp4"
    gif = root / "vortex_smt1_100d.gif"
    montage = root / "vortex_smt1_frames.png"
    for path in (mp4, gif, montage):
        require(path.is_file(), f"missing visual artifact: {path}")
        require(path.stat().st_size > 1000, f"empty visual artifact: {path}")
    require(b"ftyp" in mp4.read_bytes()[:64], "MP4 lacks an ftyp header")
    with Image.open(gif) as handle:
        require(getattr(handle, "n_frames", 1) == 100,
                "GIF does not contain exactly 100 frames")
        gif_size = handle.size
    with Image.open(montage) as handle:
        require(handle.width > 0 and handle.height > 0,
                "montage has invalid dimensions")
        montage_size = handle.size
    return {
        "mp4_bytes": mp4.stat().st_size,
        "gif_bytes": gif.stat().st_size,
        "gif_frames": 100,
        "gif_size": list(gif_size),
        "montage_bytes": montage.stat().st_size,
        "montage_size": list(montage_size),
    }


def validate(score: dict, flat: dict, *, root: Path = ROOT,
             oracle: Path = ORACLE) -> dict:
    require(score.get("format") == "nemo-testcase-l1-vortex-round210-100day-v1",
            "unexpected score format")
    require(set(score.get("cards", {})) == {"smt1"},
            "score must contain exactly the SMT-1 card")
    card = score["cards"]["smt1"]
    require(card.get("case") == "VORTEX_SMT1_VEC-zps",
            "wrong SMT-1 card")
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
    snapshots = sorted((root / "lego_smt1").glob("day[0-9][0-9][0-9].npz"))
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
    require(rows["100"]["T_rms"] < DAY100_T_BOUND_K,
            f"day-100 T RMS {rows['100']['T_rms']} exceeds frozen bound")
    require(rows["100"]["ssh_max"] > 0.0,
            "fixed day-100 SSH-difference scale is zero")

    require("vec" in flat.get("cards", {}),
            "flat-vector reference is missing")
    flat_rows = flat["cards"]["vec"]["rows"]
    table = {}
    for day in CHECKPOINTS:
        smt = rows[str(day)]
        vec = flat_rows[str(day)]
        table[str(day)] = {
            "smt1": smt,
            "flat_vector": vec,
            "ratio_smt1_to_flat_vector": {
                name: float(smt[name]) / float(vec[name])
                for name in smt
            },
        }

    visuals = _validate_visuals(root)
    return {
        "format": "nemo-testcase-l1-vortex-smt-round242-gate-v1",
        "measurement_commit": MEASUREMENT_COMMIT,
        "oracle_root": str(oracle),
        "nemo_daily_restarts": len(restarts),
        "lego_daily_snapshots": len(snapshots),
        "short_run_status": card["kt1_10_sanity"]["status"],
        "day100_T_rms_K": rows["100"]["T_rms"],
        "day100_T_bound_K": DAY100_T_BOUND_K,
        "checkpoints": table,
        "visuals": visuals,
        "status": "PASS",
    }


def plant(score: dict, flat: dict, kind: str, *, root: Path = ROOT,
          oracle: Path = ORACLE) -> None:
    if kind == "checkpoint":
        del score["cards"]["smt1"]["rows"]["60"]
    elif kind == "bound":
        score["cards"]["smt1"]["rows"]["100"]["T_rms"] = DAY100_T_BOUND_K
    else:  # pragma: no cover - argparse owns this branch
        raise GateError(f"unknown plant {kind}")
    validate(score, flat, root=root, oracle=oracle)
    raise GateError(f"{kind} plant did not fire")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--flat-reference", type=Path, default=FLAT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("checkpoint", "bound"))
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp

        try:
            stamp = worktree_stamp()
        except RuntimeError as error:
            raise GateError(str(error)) from error
        score = _load(args.root / "round210_scores.json")
        flat = _load(args.flat_reference)
        if args.plant:
            plant(score, flat, args.plant, root=args.root, oracle=args.oracle)
        report = validate(score, flat, root=args.root, oracle=args.oracle)
        report["gate_worktree"] = stamp
    except GateError as error:
        prefix = "STATUS PLANT-FIRED" if args.plant else "REFUSE"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 1
    output = args.output or args.root / "round242_gate.json"
    output.write_text(json.dumps(report, indent=2))
    print("STATUS PASS: "
          f"short={report['short_run_status']} "
          f"day100_T_rms={report['day100_T_rms_K']:.17e} K "
          f"restarts={report['nemo_daily_restarts']} "
          f"frames={report['visuals']['gif_frames']}")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
