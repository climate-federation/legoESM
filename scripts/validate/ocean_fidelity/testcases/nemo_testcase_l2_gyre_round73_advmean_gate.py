#!/usr/bin/env python3
"""Fail-closed gate for the round-73 kt=2 external transport-mean record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_round14_advmean as round14  # noqa: E402

RECORD = "oracle_bt_advmean_operands_kt00000002.bin"
N2 = 36 * 26
N_CYCLE = 50
N_SUBSTEP_FIELDS = len(round14.ADVMEAN_SUBSTEP_FIELDS)
EXPECTED_SIZE = (
    16 + 6 * 4 + 8 + N_CYCLE * 8 + 2 * N2 * 8
    + N_CYCLE * (4 + 8 + N_SUBSTEP_FIELDS * N2 * 8)
    + 4 * N2 * 8
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REFUSE: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bits_equal(left, right) -> bool:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _row(candidate, oracle) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    delta = candidate - oracle
    return {
        "bit_exact": _bits_equal(candidate, oracle),
        "differing_cells": int(np.count_nonzero(
            candidate.view(np.uint64) != oracle.view(np.uint64))),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
    }


def validate_fields(fields: dict, *, replay_ulp: bool = False) -> dict:
    rows = {"u": [], "v": []}
    for step in range(50):
        weight = np.float64(fields["weight"][step])
        for face, reciprocal in (("u", fields["r1_e2u"]),
                                 ("v", fields["r1_e1v"])):
            increment = (weight * fields[f"metric_{face}"][step]) * reciprocal
            replay = fields[f"sum_{face}_entry"][step] + increment
            target = np.array(fields[f"sum_{face}_exit"][step], copy=True)
            if replay_ulp and step == 0 and face == "u":
                live = np.argwhere(np.isfinite(target) & (target != 0.0))
                require(live.size > 0, "replay plant requires nonzero U target")
                index = tuple(live[0])
                target[index] = np.nextafter(target[index], np.inf)
            rows[face].append(_row(replay, target))
    for face in ("u", "v"):
        replay = fields[f"sum_{face}_exit"][-1] / np.float64(fields["divisor"])
        rows[f"{face}_normalized"] = _row(replay, fields[f"pre_lbc_{face}"])
    require(all(row["bit_exact"] for face in ("u", "v") for row in rows[face])
            and rows["u_normalized"]["bit_exact"]
            and rows["v_normalized"]["bit_exact"],
            "compiled advective-mean replay is not bit-exact")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant", choices=("none", "stamp", "header", "truncation", "replay-ulp"),
        default="none")
    args = parser.parse_args()

    path = args.root / RECORD
    stamp = args.root / f"{RECORD}.stamp"
    require(path.is_file() and stamp.is_file(), "record or stamp missing")
    producer = (args.root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_commit, "producer_commit.txt mismatch")
    parts = stamp.read_text().strip().split()
    expected_commit = "planted-wrong-commit" if args.plant == "stamp" else producer
    require(parts == [sha256(path), expected_commit, RECORD], "record stamp mismatch")
    observed_size = path.stat().st_size - (1 if args.plant == "truncation" else 0)
    require(observed_size == EXPECTED_SIZE, "record byte size is not exact")
    fields = round14.read_advmean(
        path, expected_kt=(3 if args.plant == "header" else 2))
    rows = validate_fields(fields, replay_ulp=args.plant == "replay-ulp")
    report = {
        "format": "nemo-l2-round73-advmean-v1",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "record_sha256": sha256(path),
        "record_size": path.stat().st_size,
        "rows": rows,
        "plant": args.plant,
        "status": "AT-BAR",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ROUND73_ADVMEAN_RECORD_AT_BAR")


if __name__ == "__main__":
    main()
