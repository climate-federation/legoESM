#!/usr/bin/env python3
"""Admission gate for the self-describing round-92 rung-0 RHS record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"NEMO_L4_R92RHS1".ljust(16, b" ")
NAMES = tuple(
    f"after_{boundary}_{face}"
    for boundary in ("hpg", "ldf", "vor", "keg", "zad")
    for face in ("u", "v")
)
PLANTS = ("none", "header", "field-name", "truncation", "swapped-rank", "restart-byte")


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def _i4(raw: bytes, offset: int, count: int) -> tuple[tuple[int, ...], int]:
    end = offset + 4 * count
    require(end <= len(raw), "truncated integer header")
    return struct.unpack_from(f"={count}i", raw, offset), end


def read_record(path: Path, *, plant: str = "none") -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 80, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    header, offset = _i4(raw, 16, 16)
    (version, kt, kbb, krhs, rank, nx, ny, nz, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require(version == 1 and kt == 1 and bits == 64,
            f"{path.name}: bad version/step/precision")
    require((nx, ny, nz) == (94, 152, 31), f"{path.name}: bad local shape")
    require((ntsi, ntsj, ntei, ntej) == (3, 3, 92, 150),
            f"{path.name}: bad owned bounds")
    require(nfields == len(NAMES), f"{path.name}: expected ten fields")
    fields: dict[str, dict] = {}
    for index in range(nfields):
        require(offset + 32 <= len(raw), f"{path.name}: truncated field header")
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        dims, offset = _i4(raw, offset, 4)
        ndim, n1, n2, n3 = dims
        if plant == "field-name" and index == 0:
            name = "planted_name"
        require(name == NAMES[index],
                f"{path.name}: field {index} is {name!r}, expected {NAMES[index]!r}")
        require((ndim, n1, n2, n3) == (3, nx, ny, nz),
                f"{path.name}: {name} dimensions disagree with header")
        count = n1 * n2 * n3
        end = offset + 8 * count
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        owned = values.reshape((n1, n2, n3), order="F")[
            ntsi - 1:ntei, ntsj - 1:ntej, :
        ]
        require(bool(np.isfinite(owned).all()),
                f"{path.name}: nonfinite owned value in {name}")
        fields[name] = {
            "count": count,
            "sha256": hashlib.sha256(raw[offset:end]).hexdigest(),
        }
        offset = end
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        "rank": rank,
        "kt": kt,
        "levels": {"Kbb": kbb, "Krhs": krhs},
        "shape": [nx, ny, nz],
        "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej],
        "fields": fields,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str) -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    records = []
    coverage = np.zeros((148, 180), dtype=np.int8)
    for expected_rank in (0, 1):
        path = root / f"oracle_r92_rhs_rank{expected_rank:04d}_kt00000001.bin"
        record = read_record(
            path,
            plant=plant if expected_rank == 0 and plant != "restart-byte" else "none",
        )
        require(record["rank"] == expected_rank,
                f"{path.name}: rank tag does not match filename")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned slab lies outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
    require(bool(np.all(coverage == 1)),
            "rank-owned RHS slabs do not cover the domain exactly once")

    restart_names = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restart_names) == 20, "baseline does not carry 20 ocean restarts")
    restart_rows = []
    for index, source in enumerate(restart_names):
        target = root / source.name
        require(target.is_file(), f"target restart missing: {source.name}")
        source_raw = source.read_bytes()
        target_raw = target.read_bytes()
        if plant == "restart-byte" and index == 0:
            target_raw = target_raw[:-1] + bytes([target_raw[-1] ^ 1])
        require(source_raw == target_raw,
                f"write-only acquisition changed {source.name}")
        restart_rows.append({
            "name": source.name,
            "sha256": hashlib.sha256(target_raw).hexdigest(),
        })
    return {
        "status": "PASS_R92_RHS_ADMISSION",
        "records": records,
        "rank_coverage": "exactly-once",
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
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R92_RHS_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
