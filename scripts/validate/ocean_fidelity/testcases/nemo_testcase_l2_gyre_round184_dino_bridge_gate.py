#!/usr/bin/env python3
"""Admit the Round-184 DINO developed-state bridge measurements.

This gate reads the established Round-182 production trace logs.  It requires
both DINO cards, in carried and recompute arms, to keep every active
prognostic family finite for the first two production steps.  For the MLF card
it additionally requires both full day-one artifacts to be stable and records
every day-one surface-field move between the two N2-routing arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROW = re.compile(
    r"TRACE_STEP_OUT_ACTIVE_CHECK_ACTIVE "
    r"eta_nonfinite=(?P<eta_bad>\d+) T_nonfinite=(?P<T_bad>\d+) "
    r"S_nonfinite=(?P<S_bad>\d+) u_nonfinite=(?P<u_bad>\d+) "
    r"v_nonfinite=(?P<v_bad>\d+).*?"
    r"eta_maxabs=(?P<eta_max>\S+) T_maxabs=(?P<T_max>\S+) "
    r"S_maxabs=(?P<S_max>\S+) u_maxabs=(?P<u_max>\S+) "
    r"v_maxabs=(?P<v_max>\S+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_trace(path: Path, expected_commit: str) -> list[dict]:
    text = path.read_text()
    require(
        f"PROVENANCE: HEAD={expected_commit} dirty_tracked_files=0" in text,
        f"{path}: missing clean expected commit stamp",
    )
    rows = []
    for step, match in enumerate(ROW.finditer(text), start=1):
        rows.append({
            "step": step,
            "nonfinite": {
                name: int(match.group(f"{name}_bad"))
                for name in ("eta", "T", "S", "u", "v")
            },
            "max_abs": {
                name: float(match.group(f"{name}_max"))
                for name in ("eta", "T", "S", "u", "v")
            },
        })
    require(len(rows) >= 2, f"{path}: fewer than two production-step rows")
    return rows


def read_mlf(path: Path, expected_commit: str) -> dict:
    with np.load(path, allow_pickle=False) as artifact:
        require(int(np.asarray(artifact["stable"]).item()) == 1,
                f"{path}: MLF day-one artifact is not stable")
        producer = str(np.asarray(artifact["producer_git_sha"]).item())
        dirty = int(np.asarray(
            artifact["producer_dirty_tracked_files"]).item())
        require(producer == expected_commit and dirty == 0,
                f"{path}: artifact provenance mismatch")
        return {
            name: np.asarray(artifact[name]).copy()
            for name in ("eta", "sst", "u", "v")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kamm-carried-log", type=Path, required=True)
    parser.add_argument("--kamm-control-log", type=Path, required=True)
    parser.add_argument("--mlf-carried-log", type=Path, required=True)
    parser.add_argument("--mlf-control-log", type=Path, required=True)
    parser.add_argument("--mlf-carried", type=Path, required=True)
    parser.add_argument("--mlf-control", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-nonfinite", action="store_true")
    args = parser.parse_args()

    logs = {
        "nemo_dino_kamm.carried": read_trace(
            args.kamm_carried_log, args.expect_commit),
        "nemo_dino_kamm.recompute": read_trace(
            args.kamm_control_log, args.expect_commit),
        "nemo_dino_kamm_mlf.carried": read_trace(
            args.mlf_carried_log, args.expect_commit),
        "nemo_dino_kamm_mlf.recompute": read_trace(
            args.mlf_control_log, args.expect_commit),
    }
    if args.plant_nonfinite:
        logs["nemo_dino_kamm.carried"][0]["nonfinite"]["T"] = 1

    for label, rows in logs.items():
        for row in rows[:2]:
            require(not any(row["nonfinite"].values()),
                    f"{label} step {row['step']} has a non-finite active cell")

    carried = read_mlf(args.mlf_carried, args.expect_commit)
    control = read_mlf(args.mlf_control, args.expect_commit)
    moves = []
    for name in carried:
        left = control[name]
        right = carried[name]
        require(left.shape == right.shape and left.dtype == right.dtype,
                f"MLF {name}: schema mismatch")
        unequal = left.view(np.uint32) != right.view(np.uint32)
        moves.append({
            "field": name,
            "cells": int(left.size),
            "cells_unequal": int(np.count_nonzero(unequal)),
            "max_abs_move": float(np.max(np.abs(right - left))),
            "control_sha256": sha256_array(left),
            "carried_sha256": sha256_array(right),
        })

    report = {
        "format": "nemo-testcase-l2-gyre-round184-dino-bridge-v1",
        "status": "PASS",
        "commit": args.expect_commit,
        "registered_steps": {
            label: rows[:2] for label, rows in logs.items()
        },
        "mlf_day1_stable": {"carried": True, "recompute": True},
        "mlf_day1_field_moves": moves,
        "plant": "nonfinite" if args.plant_nonfinite else None,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.plant_nonfinite:
        print("STATUS PLANT-MISSED")
        return 2
    print("STATUS PASS: both DINO cards finite through two developed steps; "
          "both MLF arms stable through day one")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"REFUSE: {exc}")
        print("STATUS PLANT-FIRED" if "--plant-nonfinite" in __import__("sys").argv
              else "STATUS FAIL")
        raise SystemExit(1)
