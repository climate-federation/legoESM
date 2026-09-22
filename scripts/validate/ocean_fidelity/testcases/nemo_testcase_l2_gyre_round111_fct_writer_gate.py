#!/usr/bin/env python3
"""Fail-closed admission gate for the round-111 FCT writer record."""

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

MAGIC = "NEMO_L2_R111F1"
RECORD = "oracle_fct_writers_kt00000002_s3.bin"
HEADER = (1, 2, 3, 2, 1, 1, 36, 26, 31, 64, 34, 3, 3, 2)


class GateError(RuntimeError):
    """A named fail-closed refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(name: str, rank: int, shape: tuple[int, int, int],
         origin: tuple[int, int, int]) -> tuple:
    return name, rank, shape, origin


def expected_rows() -> tuple[tuple, ...]:
    rows = [
        _row("p2dt", 0, (1, 1, 1), (0, 0, 0)),
        _row("transport_u", 3, (35, 25, 30), (1, 1, 1)),
        _row("transport_v", 3, (35, 25, 30), (1, 1, 1)),
        _row("transport_w", 3, (34, 24, 30), (2, 2, 1)),
        _row("e3t_3d", 3, (34, 24, 30), (2, 2, 1)),
        _row("r3t_Kbb", 2, (34, 24, 1), (2, 2, 1)),
        _row("r3t_Kmm", 2, (34, 24, 1), (2, 2, 1)),
        _row("tmask", 3, (34, 24, 30), (2, 2, 1)),
        _row("wmask", 3, (34, 24, 30), (2, 2, 1)),
        _row("r1_e1e2t", 2, (34, 24, 1), (2, 2, 1)),
    ]
    for tracer in ("T", "S"):
        rows.extend((
            _row(f"base_{tracer}", 3, (36, 26, 30), (1, 1, 1)),
            _row(f"rhs_entry_{tracer}", 3, (32, 22, 30), (3, 3, 1)),
            _row(f"first_u_{tracer}", 3, (35, 25, 30), (1, 1, 1)),
            _row(f"first_v_{tracer}", 3, (35, 25, 30), (1, 1, 1)),
            _row(f"first_w_{tracer}", 3, (34, 24, 31), (2, 2, 1)),
            _row(f"first_div_{tracer}", 3, (34, 24, 30), (2, 2, 1)),
            _row(f"midpoint_{tracer}", 3, (34, 24, 30), (2, 2, 1)),
            _row(f"average_u_{tracer}", 3, (33, 23, 30), (2, 2, 1)),
            _row(f"average_v_{tracer}", 3, (33, 23, 30), (2, 2, 1)),
            _row(f"average_w_{tracer}", 3, (34, 24, 31), (2, 2, 1)),
            _row(f"final_div_{tracer}", 3, (32, 22, 30), (3, 3, 1)),
            _row(f"rhs_after_{tracer}", 3, (32, 22, 30), (3, 3, 1)),
        ))
    return tuple(rows)


EXPECTED_ROWS = expected_rows()


def _take(stream: io.BytesIO, count: int, label: str) -> bytes:
    data = stream.read(count)
    require(len(data) == count, f"record truncated in {label}")
    return data


def read_self_describing_record(
    path: Path,
    *,
    magic_expected: str,
    header_expected: tuple[int, ...],
    rows_expected: tuple[tuple, ...],
    truncate: bool = False,
    drop_last: bool = False,
) -> dict:
    """Read the shared ordered-field stream used by the FCT writer probes."""
    raw = path.read_bytes()
    if truncate:
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    header_bytes = len(header_expected) * 4
    header = struct.unpack(
        f"={len(header_expected)}i", _take(stream, header_bytes, "header"))
    require(magic == magic_expected, f"bad magic {magic!r}")
    require(header == header_expected,
            f"header {header} != {header_expected}")

    fields: dict[str, dict] = {}
    observed = []
    field_count = header[10] - int(drop_last)
    for index in range(field_count):
        name = _take(stream, 16, f"field {index} name").decode("ascii").rstrip()
        rank, nx, ny, nz, i0, j0, k0 = struct.unpack(
            "=7i", _take(stream, 28, f"{name} metadata"))
        count = nx * ny * nz
        payload = _take(stream, count * 8, f"{name} payload")
        values = np.frombuffer(payload, dtype="=f8").reshape(
            (nx, ny, nz), order="F").copy()
        require(np.all(np.isfinite(values)), f"{name} contains non-finite values")
        require(name not in fields, f"duplicate field {name}")
        fields[name] = {
            "rank": rank,
            "shape": (nx, ny, nz),
            "origin": (i0, j0, k0),
            "values": values,
        }
        observed.append((name, rank, (nx, ny, nz), (i0, j0, k0)))
    require(stream.read(1) == b"", "record has trailing bytes")
    require(tuple(observed) == rows_expected[:field_count],
            "field order, rank, extent, or origin differs from the source card")
    return {"header": header, "fields": fields, "sha256": sha256(path)}


def read_record(path: Path, *, truncate: bool = False) -> dict:
    return read_self_describing_record(
        path,
        magic_expected=MAGIC,
        header_expected=HEADER,
        rows_expected=EXPECTED_ROWS,
        truncate=truncate,
    )


def check_stamp(root: Path, expected_commit: str) -> None:
    record = root / RECORD
    stamp = record.with_name(record.name + ".stamp")
    parts = stamp.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, "malformed record stamp")
    require(parts[0] == sha256(record), "record stamp digest mismatch")
    require(parts[1] == expected_commit, "record stamp producer mismatch")
    require(parts[2] == record.name, "record stamp filename mismatch")


def check_binary(root: Path) -> str:
    parts = (root / "binary.sha256").read_text(encoding="utf-8").split()
    require(len(parts) >= 1, "malformed binary.sha256")
    digest = sha256(root / "nemo")
    require(parts[0] == digest, "run binary differs from binary.sha256")
    return digest


def measure(args: argparse.Namespace) -> dict:
    producer = (args.root / "producer_commit.txt").read_text(
        encoding="utf-8").strip()
    require(producer == args.expect_commit,
            "producer_commit.txt differs from --expect-commit")
    stamp_commit = (
        "planted-wrong-commit" if args.plant == "stamp"
        else args.expect_commit
    )
    check_stamp(args.root, stamp_commit)
    record = read_record(
        args.root / RECORD, truncate=args.plant == "truncation")
    fields = record["fields"]
    if args.plant == "rhs-entry-ulp":
        entry = fields["rhs_entry_T"]["values"]
        entry.flat[0] = np.nextafter(entry.flat[0], np.inf)

    zero_rows = {}
    for tracer in ("T", "S"):
        values = fields[f"rhs_entry_{tracer}"]["values"]
        zero_rows[tracer] = bool(np.all(values.view("=u8") == 0))
    require(all(zero_rows.values()),
            f"stage-3 Krhs entry is not positive-zero bitwise: {zero_rows}")
    require(fields["p2dt"]["values"].item() == 14400.0,
            "stage-3 p2dt is not the resolved 14400 s")
    for mask in ("tmask", "wmask"):
        values = fields[mask]["values"]
        require(np.all((values == 0.0) | (values == 1.0)),
                f"{mask} is not binary")

    return {
        "format": "nemo-testcase-l2-gyre-round111-fct-writers-v1",
        "status": "PASS",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "binary_sha256": check_binary(args.root),
        "record": {
            "name": RECORD,
            "sha256": record["sha256"],
            "header": list(record["header"]),
            "bytes": (args.root / RECORD).stat().st_size,
            "field_count": len(fields),
        },
        "controls": {"rhs_entries_positive_zero": zero_rows},
        "plant": args.plant,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plant", choices=("none", "stamp", "truncation", "rhs-entry-ulp"),
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
        "STATUS PASS: round111 FCT writer record "
        f"fields={report['record']['field_count']} bytes={report['record']['bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
