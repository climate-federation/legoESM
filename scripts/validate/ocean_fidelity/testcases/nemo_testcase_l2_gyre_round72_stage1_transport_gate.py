#!/usr/bin/env python3
"""Fail-closed gate for the round-72 kt=2 stage-1 transport operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

DIMS = (36, 26, 31)
RECORD = "oracle_rkstage1_transport_operands_kt00000002.bin"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REFUSE: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _read(
    path: Path, *, truncate: bool = False, header_delta: bool = False,
) -> dict[str, np.ndarray | tuple]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    if truncate:
        values = values[:-1]
    if header_delta:
        header = (header[0], header[1] + 1, *header[2:])
    require(magic == "NEMO_L2_TRPOP_2", f"bad magic {magic!r}")
    require(header == (2, 2, 1, 3, *DIMS, 64), f"bad header {header}")
    nx, ny, nz = DIMS
    n2, n3 = nx * ny, nx * ny * nz
    require(values.size == 10 * n2 + 8 * n3, "bad payload size")
    layout = (
        ("e2u", n2, 2), ("e3u", n3, 3), ("uu", n3, 3),
        ("zub", n2, 2), ("umask", n3, 3), ("zFu", n3, 3),
        ("e1v", n2, 2), ("e3v", n3, 3), ("vv", n3, 3),
        ("zvb", n2, 2), ("vmask", n3, 3), ("zFv", n3, 3),
        ("un_adv", n2, 2), ("r1_hu", n2, 2), ("uu_b", n2, 2),
        ("vn_adv", n2, 2), ("r1_hv", n2, 2), ("vv_b", n2, 2),
    )
    fields: dict[str, np.ndarray | tuple] = {"header": header}
    offset = 0
    for name, size, rank in layout:
        raw = values[offset : offset + size]
        shape = (nx, ny, nz) if rank == 3 else (nx, ny)
        fields[name] = raw.reshape(shape, order="F")[2:-2, 2:-2]
        offset += size
    return fields


def _row(candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray) -> dict:
    candidate, oracle = candidate[active], oracle[active]
    require(np.all(np.isfinite(candidate)), "non-finite replay value")
    require(np.all(np.isfinite(oracle)), "non-finite recorded value")
    delta = candidate - oracle
    return {
        "bit_exact": _bits_equal(candidate, oracle),
        "differing_cells": int(np.count_nonzero(
            candidate.view(np.uint64) != oracle.view(np.uint64))),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
    }


def validate(path: Path, *, replay_ulp: bool = False) -> dict:
    fields = _read(path)
    rows = {}
    for face in ("u", "v"):
        mask = np.asarray(fields[f"{face}mask"])[..., :-1] != 0.0
        mask2 = np.any(mask, axis=-1)
        adv = np.asarray(fields[f"{face}n_adv"])
        inverse_depth = np.asarray(fields[f"r1_h{face}"])
        barotropic = np.asarray(fields[f"{face}{face}_b"])
        correction = adv * inverse_depth - barotropic
        rows[f"z{face}b"] = _row(
            correction, np.asarray(fields[f"z{face}b"]), mask2)
        velocity = np.asarray(fields[f"{face}{face}"])[..., :-1]
        stored_mask = np.asarray(fields[f"{face}mask"])[..., :-1]
        corrected = velocity + np.asarray(fields[f"z{face}b"])[..., None] * stored_mask
        metric = np.asarray(fields["e2u" if face == "u" else "e1v"])
        thickness = np.asarray(fields[f"e3{face}"])[..., :-1]
        replay = (metric[..., None] * thickness) * corrected
        target = np.asarray(fields[f"zF{face}"])[..., :-1].copy()
        if replay_ulp and face == "u":
            index = tuple(np.argwhere(mask)[0])
            target[index] = np.nextafter(target[index], np.inf)
        rows[f"zF{face}"] = _row(replay, target, mask)
    require(all(row["bit_exact"] for row in rows.values()),
            f"compiled transport replay is not bit-exact: {rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
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
    require(len(parts) == 3, "malformed stamp")
    expected_commit = "planted-wrong-commit" if args.plant == "stamp" else args.expect_commit
    require(parts == [sha256(path), expected_commit, RECORD], "record stamp mismatch")
    if args.plant == "header":
        _read(path, header_delta=True)
    if args.plant == "truncation":
        _read(path, truncate=True)
    rows = validate(path, replay_ulp=args.plant == "replay-ulp")
    report = {
        "format": "nemo-l2-round72-stage1-transport-v1",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "record_sha256": sha256(path),
        "rows": rows,
        "plant": args.plant,
        "status": "AT-BAR",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ROUND72_STAGE1_TRANSPORT_RECORD_AT_BAR")


if __name__ == "__main__":
    main()
