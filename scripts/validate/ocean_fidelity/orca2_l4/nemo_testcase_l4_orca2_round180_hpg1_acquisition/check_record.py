#!/usr/bin/env python3
"""Admit the self-describing, rank-complete rung-0 kt=1 HPG record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R180HP1".ljust(16, b" ")
KT = 1
LEVELS = (1, 3)
NAMES = (
    "rhd", "e3w", "gdept_z0", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v",
    "sum_u", "sum_v", "r1_e1u", "r1_e2v",
)
THREE_D = frozenset(NAMES[:9])
PLANTS = (
    "none", "header", "field-name", "field-rank", "field-dims", "truncation",
    "swapped-rank", "rhs-byte", "restart-byte",
)


class Refusal(RuntimeError):
    """A fail-closed admission refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_record(path: Path, plant: str = "none") -> dict[str, object]:
    """Parse the record from its own header and retain every numeric payload."""
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 80, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    header = struct.unpack_from("=16i", raw, 16)
    (version, kt, kmm, krhs, rank, nx, ny, nz, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require(
        (version, kt, kmm, krhs, bits) == (1, KT, *LEVELS, 64),
        f"{path.name}: version/step/levels/precision moved",
    )
    require(nx > 0 and ny > 0 and nz > 0, f"{path.name}: invalid local shape")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: invalid owned bounds")
    require(nfields == len(NAMES), f"{path.name}: field count moved")

    offset = 80
    fields: dict[str, np.ndarray] = {}
    payload_rows: dict[str, dict[str, object]] = {}
    for index, expected in enumerate(NAMES):
        require(offset + 36 <= len(raw), f"{path.name}: truncated field header")
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        field_rank, ndim, n1, n2, n3 = struct.unpack_from("=5i", raw, offset)
        offset += 20
        if plant == "field-name" and index == 0:
            name = NAMES[1]
        if plant == "field-rank" and index == 0:
            field_rank = 1 - rank
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name == expected, f"{path.name}: bad field {index} {name!r}")
        require(field_rank == rank, f"{path.name}: {name} rank moved")
        expected_dims = (3, nx, ny, nz) if name in THREE_D else (2, nx, ny, 1)
        require((ndim, n1, n2, n3) == expected_dims,
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset,
        ).copy().reshape((n1, n2, n3), order="F")
        owned = values[ntsi - 1:ntei, ntsj - 1:ntej, :]
        require(bool(np.isfinite(owned).all()),
                f"{path.name}: non-finite owned payload in {name}")
        fields[name] = values
        payload_rows[name] = {
            "rank": field_rank,
            "dims": [ndim, n1, n2, n3],
            "owned_max_abs": float(np.max(np.abs(owned), initial=0.0)),
            "sha256": _digest(raw[offset:end]),
        }
        offset = end
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        "rank": rank, "kt": kt, "levels": [kmm, krhs],
        "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "field_rows": payload_rows, "bytes": len(raw), "sha256": _digest(raw),
    }


def _public(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "fields"}


def _same_file(left: Path, right: Path, *, plant: bool = False) -> dict[str, object]:
    left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
    if plant:
        right_bytes = right_bytes[:-1] + bytes((right_bytes[-1] ^ 1,))
    require(left_bytes == right_bytes, f"additions-only identity moved: {left.name}")
    return {"name": left.name, "sha256": _digest(right_bytes)}


def run(root: Path, baseline: Path, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r180_hpg1_rank{expected_rank:04d}_kt00000001.bin"
        record = read_record(
            path, plant if expected_rank == 0 and plant not in {
                "rhs-byte", "restart-byte",
            } else "none",
        )
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        records.append(record)

    require(len({tuple(row["shape"]) for row in records}) == 1,
            "rank-local record shapes differ")
    coverage = np.zeros((148, 180), dtype=np.int8)
    for record in records:
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                "owned slab outside the ORCA2 global domain")
        coverage[j0:j1, i0:i1] += 1
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")

    rhs_rows = []
    for rank in (0, 1):
        name = f"oracle_r92_rhs_rank{rank:04d}_kt00000001.bin"
        rhs_rows.append(_same_file(
            baseline / name, root / name,
            plant=plant == "rhs-byte" and rank == 0,
        ))

    restarts = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restarts) == 20, "baseline lacks 20 ocean restarts")
    restart_rows = [
        _same_file(
            source, root / source.name,
            plant=plant == "restart-byte" and index == 0,
        )
        for index, source in enumerate(restarts)
    ]
    return {
        "status": "PASS_R180_HPG1_ADMISSION",
        "claim_label": "independent hierarchy rung 0",
        "rank_coverage": "exactly-once",
        "records": [_public(record) for record in records],
        "inherited_rhs_comparisons": rhs_rows,
        "terminal_restart_comparisons": restart_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root, args.baseline, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, Refusal) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R180_HPG1_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
