#!/usr/bin/env python3
"""Fail-closed reader/admission checks for L8's seven Krhs boundaries."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np


MAGIC = b"NEMO_L2_RKTR3_1 "
HEADER = struct.Struct("=11i")
NAME = "oracle_rktracer_stage3_kt00000001.bin"
BOUNDARIES = ("zero", "advection", "sbc", "qsr", "ldf", "bbc", "bbl", "dmp")


def decode(path: Path, expected_kt: int | None = None):
    raw = path.read_bytes()
    if len(raw) < len(MAGIC) + HEADER.size:
        raise ValueError(f"truncated L8 record: {path}")
    if raw[:16] != MAGIC:
        raise ValueError(f"bad L8 magic: {path}")
    header = HEADER.unpack_from(raw, 16)
    version, kt, stage, _kbb, _kmm, _krhs, _kaa, jpi, jpj, jpk, bits = header
    if (version, stage, bits) != (1, 3, 64) or min(jpi, jpj, jpk) <= 0:
        raise ValueError(f"bad L8 header: {header!r}")
    if expected_kt is not None and kt != expected_kt:
        raise ValueError(f"bad L8 kt: {kt} != {expected_kt}")
    count = jpi * jpj * jpk
    expected = 16 + HEADER.size + (22 * count + 3 * jpi * jpj) * 8
    if len(raw) != expected:
        raise ValueError(f"bad L8 EOF: {len(raw)} != {expected}")
    values = np.frombuffer(raw, dtype="=f8", offset=16 + HEADER.size)
    fields = values[:22 * count].reshape(22, count)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite L8 payload: {path}")
    out = {}
    for index, boundary in enumerate(BOUNDARIES):
        out[f"{boundary}_T"] = fields[2 * index]
        out[f"{boundary}_S"] = fields[2 * index + 1]
    return header, out


def check(path: Path, expected_kt: int, acting: bool = False):
    _, fields = decode(path, expected_kt)
    wet = (fields["advection_T"] != 0.0) | (fields["advection_S"] != 0.0)
    if not np.any(wet):
        raise ValueError("L8 record has no defined wet payload")
    if acting:
        for left, right in zip(BOUNDARIES, BOUNDARIES[1:]):
            changed = ((fields[f"{left}_T"].view(np.uint64)
                        != fields[f"{right}_T"].view(np.uint64))
                       | (fields[f"{left}_S"].view(np.uint64)
                          != fields[f"{right}_S"].view(np.uint64)))
            if right == "dmp" and expected_kt == 1:
                if np.any(changed):
                    raise ValueError("kt=1 damping must be bitwise inert")
            elif not np.any(changed & wet):
                raise ValueError(f"L8 {right} boundary did not act")
    return fields


def inventory(root: Path, steps: int, acting: bool = False):
    expected = [root / f"oracle_rktracer_stage3_kt{kt:08d}.bin"
                for kt in range(1, steps + 1)]
    actual = sorted(root.glob("oracle_rktracer_stage3_kt*.bin"))
    if actual != expected:
        raise ValueError(f"L8 inventory mismatch: {root}")
    for kt, path in enumerate(expected, 1):
        check(path, kt, acting=acting)
    return expected


def admit(roots: list[Path], steps: int):
    paths = [inventory(root, steps) for root in roots]
    digest = hashlib.sha256()
    for index in range(steps):
        payloads = [group[index].read_bytes() for group in paths]
        if payloads[1:] != payloads[:-1]:
            raise ValueError(f"L8 twins differ at kt={index + 1}")
        digest.update(payloads[0])
    print(f"L8_RECORDS_PASS steps=1..{steps} sha256={digest.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("check")
    one.add_argument("path", type=Path)
    one.add_argument("kt", type=int)
    one.add_argument("--acting", action="store_true")
    many = sub.add_parser("inventory")
    many.add_argument("root", type=Path)
    many.add_argument("steps", type=int)
    many.add_argument("--acting", action="store_true")
    admission = sub.add_parser("admit")
    admission.add_argument("steps", type=int)
    admission.add_argument("roots", nargs=3, type=Path)
    args = parser.parse_args()
    if args.mode == "check":
        check(args.path, args.kt, args.acting)
    elif args.mode == "inventory":
        inventory(args.root, args.steps, args.acting)
    else:
        admit(args.roots, args.steps)


if __name__ == "__main__":
    main()
