#!/usr/bin/env python3
"""Fail-closed gate for the GYRE round-76 kt=2 U-midpoint record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

RECORD = "oracle_bt_uamid_operands_kt00000002.bin"
MAGIC = "NEMO_L2_UAMID_1"
N_CYCLE = 50
JPI, JPJ = 36, 26
NTSI, NTEI, NTSJ, NTEJ = 3, 34, 3, 24
NX, NY = NTEI - NTSI + 1, NTEJ - NTSJ + 1
FIELDS = ("un_e", "ub_e", "ubb_e", "ua_e")
EXPECTED_SIZE = 16 + 10 * 4 + N_CYCLE * (4 + 3 * 8 + len(FIELDS) * NX * NY * 8)


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
    require(candidate.shape == oracle.shape, "replay shape mismatch")
    require(np.all(np.isfinite(candidate)), "replay candidate is non-finite")
    require(np.all(np.isfinite(oracle)), "replay target is non-finite")
    delta = candidate - oracle
    return {
        "bit_exact": _bits_equal(candidate, oracle),
        "differing_cells": int(
            np.count_nonzero(candidate.view(np.uint64) != oracle.view(np.uint64))
        ),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
    }


def read_record(path: Path, *, expected_kt: int = 2) -> dict:
    require(time_level_for_dump(path.name) == "now", "record level is not NOW")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=10i", handle.read(40))
        version, kt, ncycle, jpi, jpj, bits, ntsi, ntei, ntsj, ntej = header
        require(
            (magic, version, kt, ncycle, jpi, jpj, bits, ntsi, ntei, ntsj, ntej)
            == (
                MAGIC,
                1,
                expected_kt,
                N_CYCLE,
                JPI,
                JPJ,
                64,
                NTSI,
                NTEI,
                NTSJ,
                NTEJ,
            ),
            f"bad header {(magic, *header)}",
        )
        rows = {name: [] for name in ("coefficients", *FIELDS)}
        count = NX * NY
        for expected in range(1, N_CYCLE + 1):
            raw_jn = handle.read(4)
            require(len(raw_jn) == 4, f"truncated substep {expected}")
            (jn,) = struct.unpack("=i", raw_jn)
            require(jn == expected, f"substep {jn} != {expected}")
            coefficients = np.fromfile(handle, dtype=np.float64, count=3)
            require(coefficients.size == 3, f"truncated coefficients {expected}")
            rows["coefficients"].append(coefficients)
            for name in FIELDS:
                values = np.fromfile(handle, dtype=np.float64, count=count)
                require(values.size == count, f"truncated {name} {expected}")
                rows[name].append(values.reshape((NX, NY), order="F").T)
        require(handle.read(1) == b"", "trailing payload")
    return {
        "header": {
            "version": version,
            "kt": kt,
            "ncycle": ncycle,
            "jpi": jpi,
            "jpj": jpj,
            "bits": bits,
            "ntsi": ntsi,
            "ntei": ntei,
            "ntsj": ntsj,
            "ntej": ntej,
            "registry_level": "now",
        },
        **{name: np.stack(values) for name, values in rows.items()},
    }


def validate_fields(fields: dict, *, replay_ulp: bool = False) -> list[dict]:
    rows = []
    for index in range(N_CYCLE):
        za1, za2, za3 = fields["coefficients"][index]
        first = za1 * fields["un_e"][index]
        second = za2 * fields["ub_e"][index]
        third = za3 * fields["ubb_e"][index]
        replay = (first + second) + third
        target = np.array(fields["ua_e"][index], copy=True)
        if replay_ulp and index == 0:
            live = np.argwhere(np.isfinite(target) & (target != 0.0))
            require(live.size > 0, "replay plant requires a nonzero U target")
            location = tuple(live[0])
            target[location] = np.nextafter(target[location], np.inf)
        rows.append(_row(replay, target))
    require(all(row["bit_exact"] for row in rows), "compiled U midpoint replay is not bit-exact")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant",
        choices=("none", "stamp", "header", "truncation", "replay-ulp"),
        default="none",
    )
    args = parser.parse_args()
    path = args.root / RECORD
    stamp = args.root / f"{RECORD}.stamp"
    require(path.is_file() and stamp.is_file(), "record or stamp missing")
    producer = (args.root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_commit, "producer_commit.txt mismatch")
    expected_commit = "planted-wrong-commit" if args.plant == "stamp" else producer
    require(
        stamp.read_text().split() == [sha256(path), expected_commit, RECORD],
        "record stamp mismatch",
    )
    observed_size = path.stat().st_size - (1 if args.plant == "truncation" else 0)
    require(observed_size == EXPECTED_SIZE, "record byte size is not exact")
    fields = read_record(path, expected_kt=(3 if args.plant == "header" else 2))
    rows = validate_fields(fields, replay_ulp=args.plant == "replay-ulp")
    report = {
        "format": "nemo-testcase-l2-gyre-round76-uamid-v1",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "record_sha256": sha256(path),
        "record_size": path.stat().st_size,
        "header": fields["header"],
        "rows": rows,
        "plant": args.plant,
        "status": "AT-BAR",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ROUND76_UAMID_RECORD_AT_BAR")


if __name__ == "__main__":
    main()
