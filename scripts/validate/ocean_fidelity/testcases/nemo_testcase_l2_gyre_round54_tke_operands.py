#!/usr/bin/env python3
"""Fail-closed reader for the round-54 WRITE-only kt=2 TKE record."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L2_R54TKE1 "
FIELDS = (
    "rn_Dt", "en_entry", "avm_entry", "avt_entry", "dissl_entry",
    "rn2", "rn2b", "sh2", "e3t_Kmm", "e3w_Kmm", "matrix_diag",
    "matrix_upper", "matrix_lower", "rhs_pre_sweep", "en_post_sweep",
    "mxl_momentum", "mxl_dissipation", "pdlr", "avm_output",
    "avt_output", "dissl_output",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_record(path: Path, *, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    offset = 0

    def take(count: int) -> bytes:
        nonlocal offset
        require(offset + count <= len(raw), "record is truncated")
        value = raw[offset:offset + count]
        offset += count
        return value

    magic = take(16)
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"wrong magic {magic!r}")
    header = struct.unpack("=13i", take(13 * 4))
    keys = ("version", "kt", "Kbb", "Kmm", "jpi", "jpj", "jpk", "jpkm1",
            "ntsi", "ntei", "ntsj", "ntej", "real_bits")
    head = dict(zip(keys, header, strict=True))
    require(head["version"] == 1 and head["kt"] == 2,
            f"unexpected version/kt {head['version']}/{head['kt']}")
    require((head["jpi"], head["jpj"], head["jpk"], head["jpkm1"])
            == (32, 22, 31, 30), f"unexpected GYRE shape {head}")
    require(head["real_bits"] == 64, "record is not fp64")

    arrays = {}
    for expected in FIELDS:
        label = take(16).decode("ascii").rstrip()
        require(label == expected, f"field order mismatch: {label!r} != {expected!r}")
        ndim, n1, n2, n3 = struct.unpack("=4i", take(16))
        require(ndim in (0, 3), f"invalid rank for {label}: {ndim}")
        shape = () if ndim == 0 else (n1, n2, n3)
        count = 1 if ndim == 0 else n1 * n2 * n3
        value = np.frombuffer(take(count * 8), dtype="=f8").copy()
        if shape:
            value = value.reshape(shape, order="F")
        else:
            value = value.reshape(()).item()
        arrays[label] = value
    require(offset == len(raw), f"record has {len(raw) - offset} trailing bytes")

    if plant == "nan":
        arrays["avt_output"] = np.array(arrays["avt_output"], copy=True)
        arrays["avt_output"].flat[0] = np.nan
    for name, value in arrays.items():
        require(np.all(np.isfinite(value)), f"{name} contains NaN/Inf")
    require(arrays["rn_Dt"] == 14400.0, f"wrong rn_Dt {arrays['rn_Dt']}")
    for name in ("en_entry", "avm_entry", "avt_entry", "dissl_entry",
                 "en_post_sweep", "mxl_momentum", "mxl_dissipation",
                 "avm_output", "avt_output", "dissl_output"):
        require(np.min(arrays[name]) >= 0.0, f"{name} has a negative value")
    return {"header": head, "arrays": arrays}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--producer-commit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("header", "truncation", "nan", "stamp"))
    args = parser.parse_args(argv)
    try:
        expected = args.expect_commit.lower()
        producer = args.producer_commit.read_text().strip().lower()
        if args.plant == "stamp":
            producer = "0" * 40
        require(len(expected) == 40 and producer == expected,
                f"producer stamp mismatch: {producer} != {expected}")
        rec = read_record(args.record, plant=args.plant)
        summary = {}
        for name, value in rec["arrays"].items():
            arr = np.asarray(value)
            summary[name] = {
                "shape": list(arr.shape),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        report = {
            "format": "gyre-round54-tke-operands-v1",
            "producer_commit": producer,
            "record": str(args.record),
            "header": rec["header"],
            "fields": summary,
            "plant": args.plant,
            "status": "PASS",
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(payload)
        print(payload, end="")
        return 0
    except (GateError, OSError, UnicodeDecodeError, struct.error, ValueError) as exc:
        print(f"STATUS FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
