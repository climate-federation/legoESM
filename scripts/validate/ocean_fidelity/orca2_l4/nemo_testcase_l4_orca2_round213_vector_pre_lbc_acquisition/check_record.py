#!/usr/bin/env python3
"""Admit the self-describing, rank-complete OMT-1 vector pre-LBC record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R213VV1".ljust(16, b" ")
NAMES = ("zv_frc", "ssvmask", "va_pre_lbc")
PLANTS = (
    "none", "header", "rk-level", "field-name", "field-dims", "truncation",
    "swapped-rank", "nonfinite", "frame-byte", "restart-byte",
)


class Refusal(RuntimeError):
    """A fail-closed record refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def read_record(path: Path, plant: str = "none") -> dict[str, object]:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 80, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    header = struct.unpack_from("=16i", raw, 16)
    (version, kt, jn, kmm, krhs, rank, nx, ny, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    if plant == "rk-level":
        krhs = 2
    require((version, kt, jn, bits, nfields) == (1, 1, 1, 64, len(NAMES)),
            f"{path.name}: version/step/substep/precision/count moved")
    # stprk3.F90 calls stp_2D(kstp,Nbb,Nbb,Naa,Nrhs), whose executing
    # stp2d.F90 call passes Kbb,Kbb,Krhs to dyn_spg_ts.  At kt=1 those live
    # indices are (Kmm,Krhs)=(1,3); they are not consecutive stage numbers.
    require((kmm, krhs) == (1, 3), f"{path.name}: RK levels moved")
    require(nx > 0 and ny > 0, f"{path.name}: invalid local shape")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: invalid owned bounds")

    offset = 80
    fields: dict[str, dict[str, object]] = {}
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
        expected_shape = ((ntei - ntsi + 1, ntej - ntsj + 1)
                          if expected == "zv_frc" else (nx, ny))
        require((ndim, n1, n2, n3) == (2, *expected_shape, 1),
                f"{path.name}: {name} dimensions moved")
        count = n1 * n2
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name} payload")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        values = values.reshape((n1, n2), order="F")
        owned = (values if expected == "zv_frc" else
                 values[ntsi - 1:ntei, ntsj - 1:ntej])
        finite = bool(np.isfinite(owned).all())
        if plant == "nonfinite" and index == 0:
            finite = False
        require(finite, f"{path.name}: non-finite owned payload in {name}")
        fields[name] = {
            "dims": [ndim, n1, n2, n3],
            "owned_max_abs": float(np.max(np.abs(owned), initial=0.0)),
            "sha256": hashlib.sha256(raw[offset:end]).hexdigest(),
        }
        offset = end
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    return {
        "rank": rank, "kt": kt, "substep": jn, "levels": [kmm, krhs],
        "shape": [nx, ny], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compare_family(root: Path, baseline: Path, pattern: str,
                   expected: int, plant: str) -> list[dict[str, object]]:
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
        require(left == right, f"additions-only acquisition changed {source.name}")
        rows.append({"name": source.name,
                     "sha256": hashlib.sha256(right).hexdigest()})
    return rows


def run(root: Path, baseline: Path, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    records = []
    for expected_rank in (0, 1):
        path = root / (
            f"oracle_r213_vector_rank{expected_rank:04d}_kt00000001_jn001.bin")
        record_plant = plant if expected_rank == 0 and plant in {
            "header", "rk-level", "field-name", "field-dims", "truncation",
            "swapped-rank", "nonfinite",
        } else "none"
        record = read_record(path, record_plant)
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

    frames = compare_family(
        root, baseline, "oracle_r84_frame_*.bin", 64,
        "frame-byte" if plant == "frame-byte" else "")
    restarts = compare_family(
        root, baseline, "ORCA2_00000008_restart_????.nc", 2,
        "restart-byte" if plant == "restart-byte" else "")
    return {
        "status": "PASS_R213_OMT1_VECTOR_PRE_LBC_ADMISSION",
        "rank_coverage": "exactly-once", "records": records,
        "existing_frame_comparisons": frames,
        "terminal_restart_comparisons": restarts,
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
    print("STATUS PASS_R213_OMT1_VECTOR_PRE_LBC_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
