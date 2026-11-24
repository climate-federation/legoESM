#!/usr/bin/env python3
"""Admit the self-describing rank-complete round-181 HPG fold record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L4_R181HPF".ljust(16, b" ")
NAMES = ("north_e3w", "north_rhd", "current_e3w", "current_rhd", "zhpj")
PLANTS = ("none", "header", "field-name", "field-dims", "truncation", "restart-byte")


class Refusal(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Refusal(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_record(path: Path, plant: str = "none") -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 80, f"{path.name}: short record")
    magic = raw[:16]
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic")
    h = struct.unpack_from("=16i", raw, 16)
    version, kt, kmm, krhs, rank, nx, ny, nz, nimpp, njmpp, ntsi, ntsj, ntei, ntej, bits, nfields = h
    require((version, kt, kmm, krhs, bits, nfields) == (1, 1, 1, 3, 64, len(NAMES)),
            f"{path.name}: header moved")
    offset = 80
    fields = {}
    for index, expected in enumerate(NAMES):
        require(offset + 36 <= len(raw), f"{path.name}: truncated field header")
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        field_rank, ndim, n1, n2, n3 = struct.unpack_from("=5i", raw, offset)
        offset += 20
        if plant == "field-name" and index == 0:
            name = NAMES[1]
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name == expected and field_rank == rank, f"{path.name}: bad field {index}")
        require((ndim, n1, n2, n3) == (3, nx, ny, nz), f"{path.name}: {name} dims moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name}")
        fields[name] = np.frombuffer(raw, "=f8", count, offset).copy().reshape(
            (n1, n2, n3), order="F")
        offset = end
    require(offset == len(raw), f"{path.name}: trailing bytes")
    return {"rank": rank, "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
            "owned": [ntsi, ntsj, ntei, ntej], "fields": fields,
            "sha256": digest(raw), "bytes": len(raw)}


def run(root: Path, baseline: Path, plant: str) -> dict:
    records = [read_record(
        root / f"oracle_r181_hpgfold_rank{rank:04d}_kt00000001.bin",
        plant if rank == 0 and plant != "restart-byte" else "none")
        for rank in (0, 1)]
    coverage = np.zeros((148, 180), np.int8)
    for expected, record in enumerate(records):
        require(record["rank"] == expected, "rank moved")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148), "latitude placement moved")
        coverage[j0:j1, i0:i1] += 1
    require(bool(np.all(coverage == 1)), "rank coverage is not exactly once")
    restarts = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restarts) == 20, "baseline restart census moved")
    for index, source in enumerate(restarts):
        left, right = source.read_bytes(), (root / source.name).read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes((right[-1] ^ 1,))
        require(left == right, f"restart moved: {source.name}")
    return {"status": "PASS_R181_HPG_FOLD_ADMISSION",
            "claim_label": "independent hierarchy rung 0",
            "rank_coverage": "exactly-once",
            "records": [{k: r[k] for k in ("rank", "shape", "origin", "owned", "sha256", "bytes")}
                        for r in records], "restart_identities": 20}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--plant", choices=PLANTS, default="none")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    try:
        result = run(a.root, a.baseline, a.plant)
        require(a.plant == "none", f"{a.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, Refusal) as error:
        print(f"STATUS {'PLANT-FIRED' if a.plant != 'none' else 'REFUSE'} {a.plant}: {error}")
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output:
        a.output.write_text(text)
    print(text, end="")
    print("STATUS PASS_R181_HPG_FOLD_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
