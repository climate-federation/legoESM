#!/usr/bin/env python3
"""Admit the self-describing rank-complete ORCA2 kt=1 stage-1 record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R175S1".ljust(16, b" ")
NAMES_2D = ("ext_ssh", "ext_r3t", "ext_ub", "ext_vb")
NAMES_3D = (
    "rhs_u", "rhs_v", "pre_u", "pre_v", "cor_u", "cor_v",
    "trp_u", "trp_v", "trp_w", "adv_t", "adv_s", "sbc_t", "sbc_s",
    "qco_t", "qco_s", "final_u", "final_v", "final_t", "final_s",
)
NAMES = NAMES_2D + NAMES_3D
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "coverage", "restart-byte",
)


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_record(path: Path, plant: str = "none") -> dict[str, object]:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 92, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    header = struct.unpack_from("=19i", raw, 16)
    (version, kt, stage, kbb, kmm, krhs, kaa, rank, nx, ny, nz,
     nimpp, njmpp, ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require((version, kt, stage, bits) == (1, 1, 1, 64),
            f"{path.name}: version/step/stage/precision moved")
    require((nx, ny, nz) == (94, 152, 31), f"{path.name}: local shape moved")
    require((ntsi, ntsj, ntei, ntej) == (3, 3, 92, 150),
            f"{path.name}: owned bounds moved")
    require(nfields == len(NAMES), f"{path.name}: field count moved")
    offset = 92
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
        expected_dims = (2, nx, ny, 1) if expected in NAMES_2D else (3, nx, ny, nz)
        require(name == expected, f"{path.name}: field {index} is {name!r}")
        require((ndim, n1, n2, n3) == expected_dims,
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        reshaped = values.reshape((n1, n2, n3), order="F")
        owned = reshaped[ntsi - 1:ntei, ntsj - 1:ntej, :]
        require(bool(np.isfinite(owned).all()),
                f"{path.name}: non-finite owned payload in {name}")
        fields[name] = {
            "ndim": ndim, "dims": [n1, n2, n3],
            "payload_sha256": digest(raw[offset:end]),
        }
        offset = end
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        "rank": rank, "levels": [kbb, kmm, krhs, kaa],
        "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "bytes": len(raw), "sha256": digest(raw),
    }


def run(root: Path, baseline: Path, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r175_stage1_rank{expected_rank:04d}_kt00000001.bin"
        local_plant = plant if expected_rank == 0 and plant not in {
            "coverage", "restart-byte"} else "none"
        record = read_record(path, local_plant)
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        if plant == "coverage" and expected_rank == 1:
            i0 -= 1
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
        restart_rows.append({"name": source.name, "sha256": digest(right)})
    terminal = [row for row in restart_rows if "00000010" in row["name"]]
    require(len(terminal) == 2, "terminal kt=10 restart census moved")
    return {
        "status": "PASS_R175_STAGE1_ADMISSION",
        "rank_coverage": "exactly-once", "records": records,
        "restart_comparisons": restart_rows, "terminal_kt10": terminal,
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
    print("STATUS PASS_R175_STAGE1_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
