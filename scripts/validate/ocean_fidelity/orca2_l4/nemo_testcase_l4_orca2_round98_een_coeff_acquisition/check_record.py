#!/usr/bin/env python3
"""Admit the rank-complete self-describing round-98 EEN coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L4_R98EEN1"
HEADER = struct.Struct("=16i")
GROUP = struct.Struct("=4i")
FIELDS = ("ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
          "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne")
PLANTS = ("none", "header", "field-name", "field-dims", "truncation",
          "missing-field", "swapped-rank", "restart-byte")


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def read_record(path: Path, plant: str = "none") -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 16 + HEADER.size, f"{path.name}: record too short")
    magic = raw[:16].decode("ascii", "replace").rstrip(" \x00")
    if plant == "header":
        magic = "X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    header = list(HEADER.unpack_from(raw, 16))
    (version, kt, kmm, scheme, rank, nx, ny, nz, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require((version, kt, scheme, bits, nfields) == (1, 1, 3, 64, 8),
            f"{path.name}: invalid identity {header}")
    require(kmm > 0 and nx > 0 and ny > 0 and nz > 0,
            f"{path.name}: non-positive level or shape")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: owned bounds outside local domain")
    offset = 16 + HEADER.size
    groups = {}
    for index in range(nfields):
        require(offset + 16 + GROUP.size <= len(raw),
                f"{path.name}: truncated field header {index}")
        name = raw[offset:offset + 16].decode("ascii", "replace").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = GROUP.unpack_from(raw, offset)
        offset += GROUP.size
        if plant == "field-name" and index == 0:
            name = "missing"
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name and name not in groups,
                f"{path.name}: empty or duplicate field {name!r}")
        require((ndim, n1, n2, n3) == (2, nx, ny, 1),
                f"{path.name}: {name!r} dimensions moved")
        end = offset + 8 * n1 * n2
        require(end <= len(raw), f"{path.name}: truncated payload {name!r}")
        values = np.frombuffer(raw, dtype="=f8", count=n1 * n2, offset=offset)
        require(bool(np.isfinite(values).all()), f"{path.name}: nonfinite {name!r}")
        groups[name] = [n1, n2]
        offset = end
    require(offset == len(raw), f"{path.name}: trailing bytes")
    if plant == "missing-field":
        groups.pop(FIELDS[-1], None)
    require(set(groups) == set(FIELDS),
            f"{path.name}: fields {sorted(groups)} != {list(FIELDS)}")
    return {
        "rank": rank, "kt": kt, "kmm": kmm, "scheme": scheme,
        "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": groups,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str) -> dict:
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r98_een_coeff_rank{expected_rank:04d}_kt00000001.bin"
        applied = plant if expected_rank == 0 and plant not in ("none", "restart-byte") else "none"
        row = read_record(path, applied)
        require(row["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(row)
    require(bool(np.all(coverage == 1)), "rank slabs do not cover domain exactly once")
    restarts = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restarts) == 20, "baseline lacks 20 ocean restarts")
    restart_rows = []
    for index, source in enumerate(restarts):
        target = root / source.name
        require(target.is_file(), f"missing target restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"write-only acquisition changed {source.name}")
        restart_rows.append({"name": source.name,
                             "sha256": hashlib.sha256(right).hexdigest()})
    return {"status": "PASS_R98_EEN_COEFF_ADMISSION",
            "rank_coverage": "exactly-once", "records": records,
            "terminal_restart_comparisons": restart_rows}


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
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R98_EEN_COEFF_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
