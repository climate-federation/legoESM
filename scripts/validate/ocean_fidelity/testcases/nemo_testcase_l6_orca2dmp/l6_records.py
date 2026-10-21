#!/usr/bin/env python3
"""Fail-closed validator for the grid-ladder L6 damping acquisition."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
from pathlib import Path

import numpy as np


MAGIC = b"NEMO_L6_DMP__1  "
HEADER = struct.Struct("=10i")


def decode(path: Path, expected_kt: int | None = None):
    raw = path.read_bytes()
    if len(raw) < 16 + HEADER.size:
        raise ValueError(f"truncated damping record: {path}")
    if raw[:16] != MAGIC:
        raise ValueError(f"bad damping magic: {path}")
    header = HEADER.unpack_from(raw, 16)
    version, kt, _kbb, _kmm, _krhs, jpi, jpj, jpkm1, bits, nfields = header
    if (version, bits, nfields) != (1, 64, 10) or min(jpi, jpj, jpkm1) <= 0:
        raise ValueError(f"bad damping header: {path}: {header}")
    if expected_kt is not None and kt != expected_kt:
        raise ValueError(f"bad damping kt: {path}: {kt} != {expected_kt}")
    cell_count = jpi * jpj * jpkm1
    expected_size = 16 + HEADER.size + nfields * cell_count * (bits // 8)
    if len(raw) != expected_size:
        raise ValueError(f"bad damping EOF: {path}: {len(raw)} != {expected_size}")
    fields = np.frombuffer(
        raw, dtype=np.float64, count=nfields * cell_count,
        offset=16 + HEADER.size).reshape(nfields, cell_count)
    if not np.isfinite(fields).all():
        raise ValueError(f"non-finite damping payload: {path}")
    wet, rate = fields[:2]
    defined = wet > 0.0
    if not np.any(defined) or not np.any(rate[defined] > 0.0):
        raise ValueError(f"damping did not select any wet cell: {path}")
    if np.any(fields[:, ~defined] != 0.0):
        raise ValueError(f"observer wrote an undefined/dry cell: {path}")
    if np.any(rate[defined] < 0.0):
        raise ValueError(f"negative damping coefficient: {path}")
    acted = np.zeros(cell_count, dtype=bool)
    for offset, tracer in ((2, "temperature"), (6, "salinity")):
        target, before, rhs_before, rhs_after = fields[offset:offset + 4]
        expected_after = rhs_before + rate * (target - before)
        if not np.array_equal(rhs_after[defined], expected_after[defined]):
            raise ValueError(f"{tracer} damping statement mismatch: {path}")
        if kt == 1:
            target_bits = target.view(np.uint64)
            before_bits = before.view(np.uint64)
            rhs_before_bits = rhs_before.view(np.uint64)
            rhs_after_bits = rhs_after.view(np.uint64)
            if not np.array_equal(target_bits[defined], before_bits[defined]):
                raise ValueError(
                    f"kt=1 {tracer} target differs from before state: {path}")
            if not np.array_equal(
                    rhs_after_bits[defined], rhs_before_bits[defined]):
                raise ValueError(
                    f"kt=1 {tracer} damping is not bitwise inert: {path}")
        acted |= rhs_after.view(np.uint64) != rhs_before.view(np.uint64)
    if kt >= 2 and not np.any(acted & defined & (rate > 0.0)):
        raise ValueError(f"damping source was not acting: {path}")
    return header, fields


def check_records(root: Path, steps: int) -> list[Path]:
    expected = [root / f"oracle_damping_kt{kt:08d}.bin"
                for kt in range(1, steps + 1)]
    actual = sorted(root.glob("oracle_damping_kt*.bin"))
    if actual != expected:
        raise ValueError(f"damping inventory mismatch: {root}")
    for kt, path in enumerate(expected, 1):
        decode(path, kt)
    return expected


def check_log(log: Path, deck: Path) -> None:
    deck_text = deck.read_text()
    log_text = log.read_text(errors="replace")
    deck_match = re.search(r"^\s*nn_zdmp\s*=\s*([^!,\s]+)", deck_text,
                           flags=re.MULTILINE | re.IGNORECASE)
    log_match = re.search(r"nn_zdmp\s*=\s*([^\s]+)", log_text,
                          flags=re.IGNORECASE)
    if deck_match is None or log_match is None:
        raise ValueError("nn_zdmp missing from deck or ocean.output")
    if float(log_match.group(1).replace("D", "E").replace("d", "e")) != float(
            deck_match.group(1).replace("D", "E").replace("d", "e")):
        raise ValueError("ocean.output nn_zdmp differs from namelist")
    if re.search(r"ln_tradmp\s*=\s*T\s*$", log_text,
                 flags=re.MULTILINE | re.IGNORECASE) is None:
        raise ValueError("ocean.output does not resolve ln_tradmp=T")
    required = {
        "active damping banner":
            r"tracer\s+damping\s+as\s+specified\s+by\s+mask",
        "resto record read":
            r"read\s+resto\s*\(\s*rec:\s*1\s*\)\s+in\s+resto\.nc\s+ok",
    }
    for label, pattern in required.items():
        if re.search(pattern, log_text, flags=re.IGNORECASE) is None:
            raise ValueError(f"ocean.output lacks {label}")


def admit(roots: list[Path], steps: int) -> None:
    inventories = [check_records(root, steps) for root in roots]
    for kt in range(steps):
        payloads = [paths[kt].read_bytes() for paths in inventories]
        if payloads[1:] != payloads[:-1]:
            raise ValueError(f"damping twins differ at kt={kt + 1}")
    digest = hashlib.sha256()
    for path in inventories[0]:
        digest.update(path.read_bytes())
    print(f"L6_DAMPING_RECORDS_PASS steps=1..{steps} sha256={digest.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    records = sub.add_parser("check-records")
    records.add_argument("root", type=Path)
    records.add_argument("steps", type=int)
    log = sub.add_parser("check-log")
    log.add_argument("log", type=Path)
    log.add_argument("deck", type=Path)
    admission = sub.add_parser("admit")
    admission.add_argument("steps", type=int)
    admission.add_argument("roots", nargs=3, type=Path)
    args = parser.parse_args()
    if args.mode == "check-records":
        check_records(args.root, args.steps)
    elif args.mode == "check-log":
        check_log(args.log, args.deck)
    else:
        admit(args.roots, args.steps)


if __name__ == "__main__":
    main()
