#!/usr/bin/env python3
"""Admit the self-describing, rank-complete OMT-4 kt=7 W record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R299W7".ljust(16, b" ")
NAMES_2D = (
    "ext_ssh", "ext_r3t", "ext_ub", "ext_vb", "w_r1area", "w_area",
    "w_r3bb", "w_r3mm", "w_r3aa",
)
NAMES_3D = (
    "rhs_u", "rhs_v", "pre_u", "pre_v", "cor_u", "cor_v",
    "trp_u", "trp_v", "trp_w", "w_e3t0", "w_tmask", "w_ww",
    "adv_t", "adv_s", "sbc_t", "sbc_s", "qco_t", "qco_s",
    "final_u", "final_v", "final_t", "final_s",
)
NAMES = (
    "ext_ssh", "ext_r3t", "ext_ub", "ext_vb",
    "rhs_u", "rhs_v", "pre_u", "pre_v", "cor_u", "cor_v",
    "trp_u", "trp_v", "trp_w", "w_r1area", "w_area", "w_e3t0",
    "w_r3bb", "w_r3mm", "w_r3aa", "w_tmask", "w_ww",
    "adv_t", "adv_s", "sbc_t", "sbc_s", "qco_t", "qco_s",
    "final_u", "final_v", "final_t", "final_s",
)
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "coverage", "frame-byte", "restart-byte",
)


class Refusal(RuntimeError):
    """The record or its additions-only proof is inadmissible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_record(path: Path, plant: str = "none") -> dict[str, object]:
    """Parse the writer's header; names and payload sizes come from the file."""
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
    require((version, kt, stage, bits) == (1, 7, 1, 64),
            f"{path.name}: version/step/stage/precision moved")
    require((kbb, kmm, krhs, kaa) == (1, 1, 3, 3),
            f"{path.name}: RK levels moved")
    require((nx, ny, nz) == (94, 152, 31), f"{path.name}: local shape moved")
    require((ntsi, ntsj, ntei, ntej) == (3, 3, 92, 150),
            f"{path.name}: owned bounds moved")
    require(nfields == len(NAMES), f"{path.name}: field count moved")
    fields: dict[str, object] = {}
    offset = 92
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
        expected_dims = ((2, nx, ny, 1) if expected in NAMES_2D
                         else (3, nx, ny, nz))
        require(name == expected, f"{path.name}: field {index} is {name!r}")
        require((ndim, n1, n2, n3) == expected_dims,
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        values = values.reshape((n1, n2, n3), order="F")
        owned = values[ntsi - 1:ntei, ntsj - 1:ntej]
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


def compare_family(root: Path, baseline: Path, pattern: str, expected: int,
                   plant: bool) -> list[dict[str, str]]:
    sources = sorted(baseline.glob(pattern))
    require(len(sources) == expected,
            f"baseline {pattern} count {len(sources)} != {expected}")
    rows = []
    for index, source in enumerate(sources):
        target = root / source.name
        require(target.is_file(), f"missing target {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"additions-only build changed {source.name}")
        rows.append({"name": source.name, "sha256": digest(right)})
    return rows


def run(root: Path, baseline: Path, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    schema_plants = {"header", "field-name", "field-dims", "truncation",
                     "swapped-rank"}
    for expected_rank in (0, 1):
        path = root / f"oracle_r299_w_rank{expected_rank:04d}_kt00000007.bin"
        local = plant if expected_rank == 0 and plant in schema_plants else "none"
        record = read_record(path, local)
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
    frames = compare_family(root, baseline, "oracle_r84_frame_*.bin", 80,
                            plant == "frame-byte")
    restarts = compare_family(root, baseline,
                              "ORCA2_00000010_restart_????.nc", 2,
                              plant == "restart-byte")
    return {
        "format": "nemo-testcase-l4-orca2-round299-w-record-v1",
        "status": "PASS_R299_OMT4_KT7_W_ADMISSION",
        "schema": "NEMO_L4_R299W7", "rank_coverage": "exactly-once",
        "records": records, "existing_frame_comparisons": frames,
        "terminal_restart_comparisons": restarts, "baseline": str(baseline),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
