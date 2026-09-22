#!/usr/bin/env python3
"""Admit the passive Round-146 developed RHS-family record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np


RECORD = "oracle_developed_rhs_families_kt00001081.bin"
MAGIC = "NEMO_L2_R146FAM"
FAMILIES = ("hpg", "ldf", "vor", "keg", "zad")
FIELDS = tuple(f"after_{family}_{face}" for family in FAMILIES for face in ("u", "v"))
NX, NY, NZ = 36, 26, 31
COUNT = NX * NY * NZ
EXPECTED_SIZE = 16 + 8 * 4 + len(FIELDS) * 4 + len(FIELDS) * COUNT * 8
ROUND140_RECORD = "oracle_developed_rhs_kt00001081.bin"
ROUND140_EXPECTED_SHA = "3523ad8eba0b53f4462e85237d85a066195437e6ad1c867e4cb0179d1764a8f5"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(payload: bytes, offset: int, count: int, label: str):
    end = offset + count
    require(end <= len(payload), f"truncated {label}")
    return payload[offset:end], end


def read_record_bytes(payload: bytes) -> dict:
    require(len(payload) == EXPECTED_SIZE,
            f"record is {len(payload)} bytes, expected {EXPECTED_SIZE}")
    raw, offset = _take(payload, 0, 16, "magic")
    try:
        magic = raw.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise GateError("magic is not ASCII") from error
    raw, offset = _take(payload, offset, 8 * 4, "header")
    version, kt, kbb, krhs, nx, ny, nz, bits = struct.unpack("=8i", raw)
    raw, offset = _take(payload, offset, len(FIELDS) * 4, "sizes")
    sizes = struct.unpack(f"={len(FIELDS)}i", raw)
    require((magic, version, kt, kbb, krhs, nx, ny, nz, bits)
            == (MAGIC, 1, 1081, 1, 3, NX, NY, NZ, 64),
            "bad header " + repr((magic, version, kt, kbb, krhs, nx, ny, nz, bits)))
    require(sizes == (COUNT,) * len(FIELDS), "bad field sizes")
    fields = {}
    for name in FIELDS:
        raw, offset = _take(payload, offset, COUNT * 8, name)
        values = np.frombuffer(raw, dtype=np.float64).copy()
        require(values.size == COUNT and np.all(np.isfinite(values)),
                f"bad values in {name}")
        fields[name] = values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)
    require(offset == len(payload), "trailing payload")
    require(tuple(fields) == FIELDS, "field registry changed")
    return {"header": {"magic": magic, "version": version, "kt": kt,
                       "Kbb": kbb, "Krhs": krhs, "nx": nx, "ny": ny,
                       "nz": nz, "bits": bits}, "fields": fields}


def _read_round140_completed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ROUND140_EXPECTED_SHA,
            "Round-140 completed-RHS ancestry changed")
    # Header 16 + 8 ints + 7 sizes, then e3u, rhs_u, umask, e3v, rhs_v.
    offset = 16 + 8 * 4 + 7 * 4
    arrays = []
    for _ in range(5):
        raw, offset = _take(payload, offset, COUNT * 8, "Round-140 field")
        arrays.append(np.frombuffer(raw, dtype=np.float64).copy().reshape(
            (NX, NY, NZ), order="F").transpose(1, 0, 2))
    return arrays[1], arrays[4]


def _identity(a: np.ndarray, b: np.ndarray) -> dict:
    unequal = a.view(np.uint64) != b.view(np.uint64)
    return {"bit_exact": not bool(np.any(unequal)),
            "differing_cells": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.0}


def _git_stamp(repo: Path, expected: str) -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo, text=True)
    require(head == expected, f"commit {head} != expected {expected}")
    require(not status, "worktree is dirty")
    return {"commit": head, "clean": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round140-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("none", "header", "truncation",
                                             "final-ulp", "missing-field"),
                        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = (args.root / RECORD).read_bytes()
        if args.plant == "header":
            changed = bytearray(payload)
            changed[16:20] = struct.pack("=i", 2)
            payload = bytes(changed)
        elif args.plant == "truncation":
            payload = payload[:-1]
        record = read_record_bytes(payload)
        if args.plant == "missing-field":
            del record["fields"][FIELDS[-1]]
            require(tuple(record["fields"]) == FIELDS, "field registry changed")
        completed_u, completed_v = _read_round140_completed(
            args.round140_root / ROUND140_RECORD)
        if args.plant == "final-ulp":
            record["fields"]["after_zad_u"][2, 2, 0] = np.nextafter(
                record["fields"]["after_zad_u"][2, 2, 0], np.inf)
        final_rows = {
            "u": _identity(record["fields"]["after_zad_u"], completed_u),
            "v": _identity(record["fields"]["after_zad_v"], completed_v),
        }
        require(all(row["bit_exact"] for row in final_rows.values()),
                "final ZAD boundary differs from admitted completed RHS")
        stamp = _git_stamp(args.repo, args.expect_commit)
        result = {"format": "nemo-testcase-l2-gyre-round146-rhs-family-v1",
                  "status": "PASS", "worktree": stamp,
                  "record_sha256": hashlib.sha256(
                      (args.root / RECORD).read_bytes()).hexdigest(),
                  "expected_size": EXPECTED_SIZE,
                  "fields": list(FIELDS), "final_vs_round140": final_rows}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("ROUND146 RHS FAMILY RECORD PASS")
        return 0
    except (GateError, OSError, ValueError) as error:
        result = {"format": "nemo-testcase-l2-gyre-round146-rhs-family-v1",
                  "status": "PLANT-FIRED" if args.plant != "none" else "FAIL",
                  "plant": args.plant, "error": str(error)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        marker = "STATUS PLANT-FIRED" if args.plant != "none" else "STATUS FAIL"
        print(f"ROUND146 RHS FAMILY {args.plant.upper()} {marker}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
