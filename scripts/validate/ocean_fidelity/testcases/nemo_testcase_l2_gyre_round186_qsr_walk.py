#!/usr/bin/env python3
"""Admit the Round-186 developed ``qsr_2BD`` operand/replay record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np


MAGIC = b"NEMO_L2_R186QSR1"
# At step 1080 the persistent three-slot rotation enters stage 3 with Kmm=2
# and Krhs=1 (stprk3.f90's two in-stage swaps plus its end-of-step swap).
HEADER = (1, 1080, 3, 2, 1, 36, 26, 31, 64)
HEADER_INTS = len(HEADER)
JPI, JPJ, JPK = 36, 26, 31
# qsr is allocated on NEMO's no-halo domain (Nis0:Nie0,Njs0:Nje0), while
# r3t and every 3-D field in the same WRITE retain the full local domain.
# With nn_hls=2 that no-halo extent is (36 - 4) by (26 - 4).
NN_HLS = 2
QSR_NI, QSR_NJ = JPI - 2 * NN_HLS, JPJ - 2 * NN_HLS
VALUE_COUNT = JPK + QSR_NI * QSR_NJ + JPI * JPJ + 5 * JPI * JPJ * JPK
RECORD_BYTES = 16 + 4 * HEADER_INTS + 8 * VALUE_COUNT


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> dict:
    raw = path.read_bytes()
    require(len(raw) == RECORD_BYTES,
            f"{path}: {len(raw)} bytes, expected {RECORD_BYTES}")
    require(raw[:16] == MAGIC, f"{path}: bad magic {raw[:16]!r}")
    header = struct.unpack(f"={HEADER_INTS}i", raw[16:16 + 4 * HEADER_INTS])
    require(header == HEADER, f"{path}: header is {header}, expected {HEADER}")
    values = np.frombuffer(raw, dtype="=f8", offset=16 + 4 * HEADER_INTS)
    require(values.size == VALUE_COUNT, f"{path}: bad value count")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    offset = 0

    def take(count: int, shape: tuple[int, ...], name: str) -> np.ndarray:
        nonlocal offset
        result = values[offset:offset + count].reshape(shape, order="F").copy()
        offset += count
        require(result.size == count, f"{path}: truncated {name}")
        return result

    n2 = JPI * JPJ
    n3 = n2 * JPK
    result = {
        "gdepw_1d": take(JPK, (JPK,), "gdepw_1d"),
        "qsr": take(QSR_NI * QSR_NJ, (QSR_NI, QSR_NJ), "qsr"),
        "r3t_Kmm": take(n2, (JPI, JPJ), "r3t_Kmm"),
        "e3t_3d": take(n3, (JPI, JPJ, JPK), "e3t_3d"),
        "tmask": take(n3, (JPI, JPJ, JPK), "tmask"),
        "wmask": take(n3, (JPI, JPJ, JPK), "wmask"),
        "actual_increment": take(n3, (JPI, JPJ, JPK), "actual_increment"),
        "replay_increment": take(n3, (JPI, JPJ, JPK), "replay_increment"),
    }
    require(offset == VALUE_COUNT, f"{path}: unread values")
    result.update(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        header=list(header), bytes=len(raw),
        qsr_bounds_1based=[1 + NN_HLS, JPI - NN_HLS,
                           1 + NN_HLS, JPJ - NN_HLS],
    )
    return result


def admit(root: Path, expect_commit: str, plant: str | None = None) -> dict:
    require(plant in (None, "actual-increment-ulp"), f"unknown plant {plant}")
    record_path = root / "oracle_qsr_walk_kt00001080.bin"
    record = read_record(record_path)
    producer = (root / "producer_commit.txt").read_text().strip()
    require(producer == expect_commit,
            f"producer commit {producer}, expected {expect_commit}")
    stamp_words = (root / "qsr_record.stamp").read_text().split()
    require(stamp_words == [record["sha256"], producer, record_path.name],
            "qsr record stamp mismatch")

    actual = record["actual_increment"].copy()
    replay = record["replay_increment"]
    if plant == "actual-increment-ulp":
        actual.flat[0] = np.nextafter(actual.flat[0], np.inf)
    unequal = int(np.count_nonzero(actual.view(np.uint64) != replay.view(np.uint64)))
    max_abs = float(np.max(np.abs(actual - replay)))
    if plant:
        require(unequal > 0, "actual-increment ULP plant did not move replay row")
        raise GateError(f"STATUS PLANT-FIRED: {plant}; unequal={unequal}")
    require(unequal == 0,
            f"post-call qsr replay is not BIT: {unequal} cells, max {max_abs}")
    return {
        "format": "gyre-round186-qsr-walk-admission-v1",
        "status": "PASS",
        "producer_commit": producer,
        "record": {"path": record["path"], "sha256": record["sha256"],
                   "bytes": record["bytes"], "header": record["header"],
                   "qsr_bounds_1based": record["qsr_bounds_1based"]},
        "calibration": {"cells_unequal": unequal, "max_abs_K_s-1": max_abs},
        "worktree": {"commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("actual-increment-ulp",))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        report = admit(args.root, args.expect_commit, args.plant)
    except GateError as exc:
        print(str(exc))
        return 1
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("STATUS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
