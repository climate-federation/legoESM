#!/usr/bin/env python3
"""Fail-closed admission gate for the Round-117 direct forcing records.

The source card writes the kt=2 slow-forcing producer in ``stp2d`` and the
directly consumed pre-loop fields in ``dynspg_ts``.  This gate parses both
records through physical EOF, proves their shared boundary bit-for-bit,
replays the compiled final subtract, and checks the duplicate final forcing
against the admitted Round-81 external-step record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round81_btstep_gate as round81,
)

SLOW_RECORD = "oracle_slow_forcing_kt00000002.bin"
PRELOOP_RECORD = "oracle_preloop_forcing_kt00000002.bin"
SLOW_MAGIC = "NEMO_L2_SLOW_2"
PRELOOP_MAGIC = "NEMO_L2_R117PF1"
JPI, JPJ, JPK = 36, 26, 31
OWNED_X, OWNED_Y = JPI - 4, JPJ - 4
FULL_2D_COUNT = JPI * JPJ
OWNED_2D_COUNT = OWNED_X * OWNED_Y
FULL_3D_COUNT = JPI * JPJ * JPK
SLOW_HEADER = (2, 2, 3, 1, JPI, JPJ, JPK, 64)
SLOW_SIZES = (FULL_3D_COUNT,) * 6 + (OWNED_2D_COUNT,)
SLOW_EXPECTED_SIZE = (
    16 + 8 * 4 + 7 * 4
    + 6 * FULL_3D_COUNT * 8
    + 6 * OWNED_2D_COUNT * 8
    + 8 * FULL_2D_COUNT * 8
    + 8
)
PRELOOP_HEADER = (1, 2, 3, JPI, JPJ, 64, 18, OWNED_2D_COUNT)
PRELOOP_FULL_FIELDS = (
    "u_kmm", "v_kmm", "u_mask", "v_mask", "cor_u", "cor_v",
)
PRELOOP_FIELD_LAYOUT = (
    ("u_kmm", "full"), ("v_kmm", "full"),
    ("incoming_u", "owned"), ("incoming_v", "owned"),
    ("u_mask", "full"), ("v_mask", "full"),
    ("ffu_nw", "owned"), ("ffu_ne", "owned"),
    ("ffu_sw", "owned"), ("ffu_se", "owned"),
    ("ffv_nw", "owned"), ("ffv_ne", "owned"),
    ("ffv_sw", "owned"), ("ffv_se", "owned"),
    ("cor_u", "full"), ("cor_v", "full"),
    ("final_u", "owned"), ("final_v", "owned"),
)
PRELOOP_EXPECTED_SIZE = (
    16 + 8 * 4
    + len(PRELOOP_FULL_FIELDS) * FULL_2D_COUNT * 8
    + (len(PRELOOP_FIELD_LAYOUT) - len(PRELOOP_FULL_FIELDS))
    * OWNED_2D_COUNT * 8
)


class GateError(RuntimeError):
    """A named fail-closed refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _take(stream: io.BytesIO, count: int, label: str) -> bytes:
    payload = stream.read(count)
    require(len(payload) == count, f"record truncated in {label}")
    return payload


def _full2(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="=f8").reshape(
        (JPI, JPJ), order="F").T.copy()


def _owned2(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="=f8").reshape(
        (OWNED_X, OWNED_Y), order="F").T.copy()


def _full3(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="=f8").reshape(
        (JPI, JPJ, JPK), order="F").transpose(1, 0, 2).copy()


def read_slow_record(path: Path) -> dict[str, object]:
    """Parse the exact mixed-layout ``NEMO_L2_SLOW_2`` record."""
    raw = path.read_bytes()
    require(
        len(raw) == SLOW_EXPECTED_SIZE,
        f"{path}: {len(raw)} bytes != {SLOW_EXPECTED_SIZE}",
    )
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "slow magic").decode("ascii").rstrip()
    header = struct.unpack("=8i", _take(stream, 32, "slow header"))
    sizes = struct.unpack("=7i", _take(stream, 28, "slow sizes"))
    require(magic == SLOW_MAGIC, f"bad slow magic {magic!r}")
    require(header == SLOW_HEADER, f"slow header {header} != {SLOW_HEADER}")
    require(sizes == SLOW_SIZES, f"slow sizes {sizes} != {SLOW_SIZES}")

    fields: dict[str, object] = {}
    for name in ("e3u", "krhs_u", "umask", "e3v", "krhs_v", "vmask"):
        fields[name] = _full3(_take(
            stream, FULL_3D_COUNT * 8, f"slow {name}"))
    for name in ("depth_u", "depth_v"):
        fields[name] = _owned2(_take(
            stream, OWNED_2D_COUNT * 8, f"slow {name}"))
    for name in ("r1_hu0", "r1_hv0"):
        fields[name] = _full2(_take(
            stream, FULL_2D_COUNT * 8, f"slow {name}"))
    for name in ("post_drag_u", "post_drag_v"):
        fields[name] = _owned2(_take(
            stream, OWNED_2D_COUNT * 8, f"slow {name}"))
    for name in ("cd_u", "cd_v"):
        fields[name] = _full2(_take(
            stream, FULL_2D_COUNT * 8, f"slow {name}"))
    fields["r1_rho0"] = struct.unpack(
        "=d", _take(stream, 8, "slow r1_rho0"))[0]
    for name in ("utau", "vtau", "r1_hu", "r1_hv"):
        fields[name] = _full2(_take(
            stream, FULL_2D_COUNT * 8, f"slow {name}"))
    for name in ("post_wind_u", "post_wind_v"):
        fields[name] = _owned2(_take(
            stream, OWNED_2D_COUNT * 8, f"slow {name}"))
    require(stream.read(1) == b"", "slow record has trailing payload")
    for name, value in fields.items():
        require(np.all(np.isfinite(value)), f"slow {name} is non-finite")
    return {
        "header": header, "sizes": sizes, "fields": fields,
        "bytes": len(raw), "sha256": sha256(path),
    }


def read_preloop_record(
    path: Path, *, plant: str = "none",
) -> dict[str, object]:
    """Parse the exact full-domain/owned-domain pre-loop record."""
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    require(
        len(raw) == PRELOOP_EXPECTED_SIZE,
        f"{path}: {len(raw)} bytes != {PRELOOP_EXPECTED_SIZE}",
    )
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "pre-loop magic").decode("ascii").rstrip()
    header = list(struct.unpack("=8i", _take(stream, 32, "pre-loop header")))
    if plant == "header":
        header[6] += 1
    header_tuple = tuple(header)
    require(magic == PRELOOP_MAGIC, f"bad pre-loop magic {magic!r}")
    require(
        header_tuple == PRELOOP_HEADER,
        f"pre-loop header {header_tuple} != {PRELOOP_HEADER}",
    )
    fields = {}
    layout_fields = (
        PRELOOP_FIELD_LAYOUT[:-1]
        if plant == "layout" else PRELOOP_FIELD_LAYOUT
    )
    for name, layout in layout_fields:
        if layout == "full":
            fields[name] = _full2(_take(
                stream, FULL_2D_COUNT * 8, f"pre-loop {name}"))
        else:
            fields[name] = _owned2(_take(
                stream, OWNED_2D_COUNT * 8, f"pre-loop {name}"))
        require(np.all(np.isfinite(fields[name])),
                f"pre-loop {name} is non-finite")
    require(stream.read(1) == b"", "pre-loop record has trailing payload")
    return {
        "header": header_tuple, "fields": fields,
        "bytes": len(raw), "sha256": sha256(path),
    }


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, f"comparison shape {left.shape} != {right.shape}")
    unequal = left.view(np.uint64) != right.view(np.uint64)
    return {
        "bit_exact": bool(not np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "cells": int(left.size),
        "absolute_max": float(np.max(np.abs(left - right), initial=0.0)),
    }


def _check_stamp(root: Path, record_name: str, expected_commit: str) -> None:
    record = root / record_name
    stamp = record.with_name(record.name + ".stamp")
    parts = stamp.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, f"malformed stamp for {record_name}")
    require(parts[0] == sha256(record), f"stamp digest moved for {record_name}")
    require(parts[1] == expected_commit, f"stamp producer moved for {record_name}")
    require(parts[2] == record_name, f"stamp filename moved for {record_name}")


def _check_binary(root: Path) -> str:
    fields = (root / "binary.sha256").read_text(encoding="utf-8").split()
    require(bool(fields), "malformed binary.sha256")
    digest = sha256(root / "nemo")
    require(fields[0] == digest, "run binary differs from binary.sha256")
    return digest


def measure(args: argparse.Namespace) -> dict[str, object]:
    stamp = worktree_stamp()
    require(stamp.get("clean") is True,
            "round117 gate requires a clean producer worktree")
    expected = args.expect_commit.lower()
    require(
        len(expected) == 40 and stamp["commit"].lower() == expected,
        f"commit stamp mismatch: {stamp['commit']} != {expected}",
    )
    record_commit = (
        args.expect_record_commit or args.expect_commit
    ).lower()
    require(len(record_commit) == 40, "record producer commit is not full length")
    producer = (args.root / "producer_commit.txt").read_text(
        encoding="utf-8").strip().lower()
    require(producer == record_commit,
            "producer_commit.txt differs from --expect-record-commit")
    stamp_commit = "0" * 40 if args.plant == "stamp" else record_commit
    for name in (SLOW_RECORD, PRELOOP_RECORD):
        _check_stamp(args.root, name, stamp_commit)

    slow = read_slow_record(args.root / SLOW_RECORD)
    preloop = read_preloop_record(
        args.root / PRELOOP_RECORD, plant=args.plant)
    fields = preloop["fields"]
    if args.plant == "input-ulp":
        active = np.argwhere(fields["u_mask"][2:-2, 2:-2] == 1.0)
        require(active.size > 0, "input-ULP plant found no active U face")
        location = tuple(active[0])
        fields["incoming_u"][location] = np.nextafter(
            fields["incoming_u"][location], np.inf)

    for face in ("u", "v"):
        mask = fields[f"{face}_mask"]
        require(np.all((mask == 0.0) | (mask == 1.0)),
                f"{face}-mask is not binary")

    reference = round81.read_record(
        args.reference_root / round81.RECORD, expected_kt=2)
    if args.plant == "reference-ulp":
        reference["slow_u"][0, 0, 0] = np.nextafter(
            reference["slow_u"][0, 0, 0], np.inf)

    rows: dict[str, dict[str, object]] = {}
    for face in ("u", "v"):
        incoming = fields[f"incoming_{face}"]
        cor = fields[f"cor_{face}"][2:-2, 2:-2]
        mask = fields[f"{face}_mask"][2:-2, 2:-2]
        final = fields[f"final_{face}"]
        replay = incoming - cor * mask
        rows[face] = {
            "stp2d_post_wind_to_preloop_incoming": comparison(
                slow["fields"][f"post_wind_{face}"], incoming),
            "compiled_subtract_replay": comparison(replay, final),
            "preloop_final_to_round81_substep1": comparison(
                final, reference[f"slow_{face}"][0]),
        }
    require(
        all(row["bit_exact"] for face in rows.values() for row in face.values()),
        "one or more duplicate-boundary/replay rows are not bit-exact",
    )

    return {
        "format": "nemo-testcase-l2-gyre-round117-preloop-v1",
        "status": "PASS",
        "worktree": stamp,
        "producer_commit": producer,
        "binary_sha256": _check_binary(args.root),
        "records": {
            SLOW_RECORD: {
                "bytes": slow["bytes"], "sha256": slow["sha256"],
                "header": list(slow["header"]),
                "sizes": list(slow["sizes"]),
            },
            PRELOOP_RECORD: {
                "bytes": preloop["bytes"], "sha256": preloop["sha256"],
                "header": list(preloop["header"]),
            },
            "round81_reference": str(args.reference_root / round81.RECORD),
        },
        "rows": rows,
        "plant": args.plant,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plant",
        choices=("none", "stamp", "truncation", "header", "layout",
                 "input-ulp", "reference-ulp"),
        default="none",
    )
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (GateError, OSError, ValueError, struct.error) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}", file=sys.stderr)
        else:
            print(f"REFUSE: {error}", file=sys.stderr)
        return 1
    print(
        "STATUS PASS: round117 pre-loop records "
        f"slow_bytes={report['records'][SLOW_RECORD]['bytes']} "
        f"preloop_bytes={report['records'][PRELOOP_RECORD]['bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
