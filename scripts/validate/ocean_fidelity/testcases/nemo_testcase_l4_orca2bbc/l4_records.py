#!/usr/bin/env python3
"""Fail-closed validator for the grid-ladder L4 geothermal acquisition."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_BBC__1  "
HEADER = struct.Struct("=8i")


def decode(path: Path, expected_kt: int | None = None) -> tuple[tuple[int, ...], np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 16 + HEADER.size:
        raise ValueError(f"truncated geothermal record: {path}")
    if raw[:16] != MAGIC:
        raise ValueError(f"bad geothermal magic: {path}")
    header = HEADER.unpack_from(raw, 16)
    version, kt, _kmm, _krhs, jpi, jpj, bits, nfields = header
    if version != 1 or bits != 64 or nfields != 3 or jpi <= 0 or jpj <= 0:
        raise ValueError(f"bad geothermal header: {path}: {header}")
    if expected_kt is not None and kt != expected_kt:
        raise ValueError(f"bad geothermal kt: {path}: {kt} != {expected_kt}")
    count = nfields * jpi * jpj
    expected = 16 + HEADER.size + count * (bits // 8)
    if len(raw) != expected:
        raise ValueError(f"bad geothermal EOF: {path}: {len(raw)} != {expected}")
    values = np.frombuffer(raw, dtype=np.float64, count=count,
                           offset=16 + HEADER.size).reshape(nfields, jpi * jpj)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite geothermal payload: {path}")
    qgh, e3t, trend = values
    wet = e3t > 0.0
    if not np.any(wet) or not np.any(qgh[wet] > 0.0) or not np.any(trend[wet] > 0.0):
        raise ValueError(f"geothermal heating did not act: {path}")
    if np.any(qgh[~wet] != 0.0) or np.any(trend[~wet] != 0.0):
        raise ValueError(f"noncanonical dry geothermal slots: {path}")
    expected_trend = np.zeros_like(trend)
    expected_trend[wet] = qgh[wet] / e3t[wet]
    if not np.array_equal(trend, expected_trend):
        raise ValueError(f"geothermal statement mismatch: {path}")
    return header, values


def check_records(root: Path, steps: int) -> list[Path]:
    expected = [root / f"oracle_geothermal_kt{kt:08d}.bin"
                for kt in range(1, steps + 1)]
    actual = sorted(root.glob("oracle_geothermal_kt*.bin"))
    if actual != expected:
        raise ValueError(f"geothermal inventory mismatch: {root}")
    for kt, path in enumerate(expected, 1):
        decode(path, kt)
    return expected


def check_log(log: Path, deck: Path) -> None:
    deck_text = deck.read_text()
    log_text = log.read_text(errors="replace")
    deck_match = re.search(r"^\s*nn_geoflx\s*=\s*([^!,\s]+)", deck_text,
                           flags=re.MULTILINE | re.IGNORECASE)
    log_match = re.search(r"nn_geoflx\s*=\s*([^\s]+)", log_text,
                          flags=re.IGNORECASE)
    if deck_match is None or log_match is None:
        raise ValueError("nn_geoflx missing from deck or ocean.output")
    # Numeric log contracts compare parsed numbers, never formatted strings.
    if float(log_match.group(1).replace("D", "E").replace("d", "e")) != float(
            deck_match.group(1).replace("D", "E").replace("d", "e")):
        raise ValueError("ocean.output nn_geoflx differs from namelist")
    for phrase in ("Apply a geothermal heating at ocean bottom",
                   "variable geothermal heat flux"):
        if phrase not in log_text:
            raise ValueError(f"ocean.output lacks {phrase!r}")


def admit(roots: list[Path], steps: int) -> None:
    inventories = [check_records(root, steps) for root in roots]
    for kt in range(steps):
        payloads = [paths[kt].read_bytes() for paths in inventories]
        if payloads[1:] != payloads[:-1]:
            raise ValueError(f"geothermal twins differ at kt={kt + 1}")
    digest = hashlib.sha256()
    for path in inventories[0]:
        digest.update(path.read_bytes())
    print(f"L4_GEOTHERMAL_RECORDS_PASS steps=1..{steps} sha256={digest.hexdigest()}")


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
