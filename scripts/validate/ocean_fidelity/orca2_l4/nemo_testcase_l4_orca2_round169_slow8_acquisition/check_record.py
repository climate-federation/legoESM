#!/usr/bin/env python3
"""Admit the self-describing rank-complete ORCA2 kt=8 slow record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R169SLW8"
KT = 8
NAMES = (
    "e3u", "rhs_u", "umask", "e3v", "rhs_v", "vmask",
    "r1_hu0", "r1_hv0", "depth_u", "depth_v",
    "drag_u", "drag_v", "cd_u", "cd_v", "r1_rho0",
    "tau_u", "tau_v", "r1_hu", "r1_hv", "wind_u", "wind_v",
    "final_u", "final_v",
)
THREE_D = frozenset(("e3u", "rhs_u", "umask", "e3v", "rhs_v", "vmask"))
SCALARS = frozenset(("r1_rho0",))
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "restart-byte",
)


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def _owned_slice(value: np.ndarray, header: dict) -> np.ndarray:
    nx, ny = header["shape"][:2]
    ntsi, ntsj, ntei, ntej = header["owned"]
    ioff = (nx - value.shape[0]) // 2
    joff = (ny - value.shape[1]) // 2
    i0, i1 = ntsi - 1 - ioff, ntei - ioff
    j0, j1 = ntsj - 1 - joff, ntej - joff
    require(0 <= i0 < i1 <= value.shape[0]
            and 0 <= j0 < j1 <= value.shape[1],
            "field does not contain the owned slab")
    return value[i0:i1, j0:j1, ...]


def read_record(path: Path, plant: str = "none") -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 84, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    header_values = struct.unpack_from("=17i", raw, 16)
    (version, kt, kbb, kaa, krhs, rank, nx, ny, nz, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header_values
    if plant == "swapped-rank":
        rank = 1 - rank
    require(version == 1 and kt == KT, f"{path.name}: header version/kt moved")
    require(bits == 64 and nfields == len(NAMES),
            f"{path.name}: precision/field count moved")
    require((nx, ny, nz) == (94, 152, 31),
            f"{path.name}: local shape moved")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: owned bounds outside local domain")
    header = {
        "rank": rank, "kt": kt, "levels": [kbb, kaa, krhs],
        "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej],
    }
    offset = 84
    names = []
    fields = {}
    for index in range(nfields):
        require(offset + 32 <= len(raw), f"{path.name}: truncated field header")
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        if plant == "field-name" and index == 0:
            name = NAMES[1]
        if plant == "field-dims" and index == 0:
            n1 += 1
        expected_ndim = 3 if name in THREE_D else (1 if name in SCALARS else 2)
        require(name and name not in fields and ndim == expected_ndim,
                f"{path.name}: bad field {index} {name!r}")
        require(n1 > 0 and n2 > 0 and n3 > 0,
                f"{path.name}: non-positive dimensions for {name!r}")
        if ndim == 1:
            require((n1, n2, n3) == (1, 1, 1),
                    f"{path.name}: scalar dimensions moved for {name!r}")
            shape = (1,)
            count = 1
        elif ndim == 2:
            require(n1 <= nx and n2 <= ny and n3 == 1,
                    f"{path.name}: 2-D dimensions moved for {name!r}")
            shape = (n1, n2)
            count = n1 * n2
        else:
            require(n1 <= nx and n2 <= ny and n3 == nz,
                    f"{path.name}: 3-D dimensions moved for {name!r}")
            shape = (n1, n2, n3)
            count = n1 * n2 * n3
        end = offset + 8 * count
        require(end <= len(raw), f"{path.name}: truncated payload for {name!r}")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        if ndim == 1:
            owned = values
        else:
            value = values.reshape(shape, order="F")
            owned = _owned_slice(value, header)
        require(bool(np.isfinite(owned).all()),
                f"{path.name}: non-finite owned payload for {name!r}")
        names.append(name)
        fields[name] = {
            "dims": [ndim, n1, n2, n3],
            "sha256": hashlib.sha256(raw[offset:end]).hexdigest(),
        }
        offset = end
    require(tuple(names) == NAMES, f"{path.name}: field order moved")
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        **header, "fields": fields, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r169_slow8_rank{expected_rank:04d}_kt00000008.bin"
        record = read_record(
            path, plant if expected_rank == 0 and plant != "restart-byte" else "none")
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
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
        require(left == right, f"write-only acquisition changed {source.name}")
        restart_rows.append({
            "name": source.name,
            "sha256": hashlib.sha256(right).hexdigest(),
        })
    return {
        "status": "PASS_R169_SLOW8_ADMISSION",
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
    print("STATUS PASS_R169_SLOW8_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
