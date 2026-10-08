#!/usr/bin/env python3
"""Admit the self-describing, rank-complete ORCA2 kt=8 HPG record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R172HP8".ljust(16, b" ")
KT = 8
NAMES = (
    "rhd", "e3w", "gdept_z0", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v",
    "sum_u", "sum_v", "r1_e1u", "r1_e2v",
)
THREE_D = frozenset(NAMES[:9])
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "restart-byte",
)


class Refusal(RuntimeError):
    """A fail-closed admission refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def read_record(path: Path, plant: str = "none") -> dict:
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
    require(version == 1 and kt == KT and bits == 64,
            f"{path.name}: version/step/precision moved")
    require(nx > 0 and ny > 0 and nz > 0, f"{path.name}: invalid local shape")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: invalid owned bounds")
    require(nfields == len(NAMES), f"{path.name}: field count moved")

    offset = 80
    fields: dict[str, dict] = {}
    for index, expected in enumerate(NAMES):
        require(offset + 32 <= len(raw), f"{path.name}: truncated field header")
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        if plant == "field-name" and index == 0:
            name = NAMES[1]
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name == expected, f"{path.name}: bad field {index} {name!r}")
        expected_dims = (3, nx, ny, nz) if name in THREE_D else (2, nx, ny, 1)
        require((ndim, n1, n2, n3) == expected_dims,
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        values = values.reshape((n1, n2, n3), order="F")
        owned = values[ntsi - 1:ntei, ntsj - 1:ntej, :]
        require(bool(np.isfinite(owned).all()),
                f"{path.name}: non-finite owned payload in {name}")
        fields[name] = {
            "dims": [ndim, n1, n2, n3],
            "owned_max_abs": float(np.max(np.abs(owned), initial=0.0)),
            "sha256": hashlib.sha256(raw[offset:end]).hexdigest(),
        }
        offset = end
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        "rank": rank, "kt": kt, "levels": [kmm, krhs],
        "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r172_hpg8_rank{expected_rank:04d}_kt00000008.bin"
        record = read_record(
            path, plant if expected_rank == 0 and plant != "restart-byte" else "none")
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        records.append(record)

    shapes = {tuple(row["shape"]) for row in records}
    require(len(shapes) == 1, "rank-local record shapes differ")
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

    restarts = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restarts) == 20, "baseline lacks 20 ocean restarts")
    restart_rows = []
    for index, source in enumerate(restarts):
        target = root / source.name
        require(target.is_file(), f"missing target restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"additions-only acquisition changed {source.name}")
        restart_rows.append({
            "name": source.name, "sha256": hashlib.sha256(right).hexdigest(),
        })
    return {
        "status": "PASS_R172_HPG8_ADMISSION",
        "rank_coverage": "exactly-once",
        "records": records,
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
    print("STATUS PASS_R172_HPG8_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
