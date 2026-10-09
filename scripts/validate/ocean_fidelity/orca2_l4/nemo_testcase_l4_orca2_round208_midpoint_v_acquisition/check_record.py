#!/usr/bin/env python3
"""Admit the rank-complete OMT-0 substep-1 flux-form V operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R208MV1".ljust(16, b" ")
NAMES = (
    "hv_e", "zhv_bck", "hv_kmm", "vn_e", "zv_spg", "zhvp2_e",
    "zv_trd", "zv_frc", "z1_hv", "va_pre_lbc",
)
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "restart-byte", "nonfinite",
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
    (version, kt, jn, rank, nx, ny, nz, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields, reserved) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require((version, kt, jn, nz, bits, nfields, reserved) ==
            (1, 1, 1, 1, 64, len(NAMES), 0),
            f"{path.name}: header values moved")
    require(nx > 0 and ny > 0, f"{path.name}: invalid shape")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: invalid owned bounds")
    offset = 80
    fields = {}
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
        require((ndim, n1, n2, n3) == (2, nx, ny, 1),
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        values = values.reshape((n1, n2), order="F")
        owned = values[ntsi - 1:ntei, ntsj - 1:ntej]
        if plant == "nonfinite" and index == 0:
            owned = owned.copy()
            owned.flat[0] = np.nan
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
        "rank": rank, "shape": [nx, ny], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r208_midv_rank{expected_rank:04d}_kt00000001.bin"
        record = read_record(
            path, plant if expected_rank == 0 and plant != "restart-byte" else "none")
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        records.append(record)
    coverage = np.zeros((148, 180), dtype=np.int8)
    for record in records:
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                "owned slab outside ORCA2 global domain")
        coverage[j0:j1, i0:i1] += 1
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")
    restarts = sorted(baseline.glob("ORCA2_00000010_restart_????.nc"))
    require(len(restarts) == 2, "baseline lacks two terminal restarts")
    restart_rows = []
    for index, source in enumerate(restarts):
        target = root / source.name
        require(target.is_file(), f"missing target restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"additions-only acquisition changed {source.name}")
        restart_rows.append({
            "name": source.name, "sha256": hashlib.sha256(right).hexdigest()})
    return {
        "status": "PASS_R208_MIDPOINT_V_RECORD",
        "rank_coverage": "exactly-once", "records": records,
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
    print("STATUS PASS_R208_MIDPOINT_V_RECORD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
