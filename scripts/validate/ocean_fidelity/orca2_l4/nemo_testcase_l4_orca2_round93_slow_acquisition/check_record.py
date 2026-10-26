#!/usr/bin/env python3
"""Admission gate for the self-describing round-93 slow-forcing record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R93SLOW1"
NAMES = (
    "depth_u", "depth_v", "drag_u", "drag_v", "cd_u", "cd_v",
    "wind_u", "wind_v", "final_u", "final_v", "ssh_rhs", "ssh_after",
    "ub_after", "vb_after",
)
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "restart-byte",
)


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def read_record(
    path: Path, plant: str = "none", *, include_owned_values: bool = False,
) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 80, f"{path.name}: record too short")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    header = struct.unpack_from("=16i", raw, 16)
    (version, kt, kbb, kaa, krhs, rank, nx, ny, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require(version == 1, f"{path.name}: unsupported version {version}")
    require(kt == 1, f"{path.name}: unexpected timestep {kt}")
    require(bits == 64, f"{path.name}: precision is not fp64")
    require(nx > 0 and ny > 0, f"{path.name}: non-positive local domain")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: owned bounds outside local domain")
    require(nfields == len(NAMES),
            f"{path.name}: field count {nfields} != {len(NAMES)}")
    offset = 80
    fields = {}
    field_shapes = {}
    owned_values = {}
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
        require(name and name not in fields,
                f"{path.name}: empty or duplicate field {index} {name!r}")
        require(ndim == 2 and n1 > 0 and n2 > 0 and n3 == 1,
                f"{path.name}: bad dimensions for {name!r}")
        require(nx >= n1 and ny >= n2
                and (nx - n1) % 2 == 0 and (ny - n2) % 2 == 0,
                f"{path.name}: {name!r} is not centred in the local domain")
        end = offset + 8 * n1 * n2
        require(end <= len(raw), f"{path.name}: truncated {name}")
        values = np.frombuffer(raw, dtype="=f8", count=n1 * n2, offset=offset)
        ioff, joff = (nx - n1) // 2, (ny - n2) // 2
        i0, i1 = ntsi - 1 - ioff, ntei - ioff
        j0, j1 = ntsj - 1 - joff, ntej - joff
        require(0 <= i0 < i1 <= n1 and 0 <= j0 < j1 <= n2,
                f"{path.name}: {name!r} does not contain the owned slab")
        owned = values.reshape((n1, n2), order="F")[i0:i1, j0:j1].T
        require(bool(np.isfinite(owned).all()), f"{path.name}: nonfinite {name}")
        fields[name] = hashlib.sha256(raw[offset:end]).hexdigest()
        field_shapes[name] = [n2, n1]
        if include_owned_values:
            owned_values[name] = np.array(owned, copy=True)
        offset = end
    require(set(fields) == set(NAMES),
            f"{path.name}: field names differ from the required registry")
    require(offset == len(raw), f"{path.name}: trailing or truncated bytes")
    result = {
        "rank": rank, "kt": kt, "levels": [kbb, kaa, krhs],
        "shape": [nx, ny], "origin": [nimpp, njmpp],
        "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
        "field_shapes": field_shapes,
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if include_owned_values:
        result["owned_values"] = owned_values
    return result


def run(root: Path, baseline: Path, plant: str) -> dict:
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r93_slow_rank{expected_rank:04d}_kt00000001.bin"
        record = read_record(path, plant if expected_rank == 0 and plant != "restart-byte" else "none")
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: slab outside domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
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
        restart_rows.append({"name": source.name, "sha256": hashlib.sha256(right).hexdigest()})
    return {"status": "PASS_R93_SLOW_ADMISSION", "rank_coverage": "exactly-once",
            "records": records, "terminal_restart_comparisons": restart_rows}


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
    print("STATUS PASS_R93_SLOW_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
