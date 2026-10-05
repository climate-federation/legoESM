#!/usr/bin/env python3
"""Admit rank-complete EEN pre-scale accumulators and scale factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L4_R104EA1"
FINAL_MAGIC = "NEMO_L4_R98EEN1"
HEADER = struct.Struct("=14i")
FINAL_HEADER = struct.Struct("=16i")
GROUP = struct.Struct("=4i")
ACC = ("acc_u_nw", "acc_u_ne", "acc_u_sw", "acc_u_se",
       "acc_v_sw", "acc_v_se", "acc_v_nw", "acc_v_ne")
SCL = tuple(name.replace("acc_", "scl_") for name in ACC)
FIELDS = ACC + SCL
FINAL = tuple(name.replace("acc_u_", "ffu_").replace("acc_v_", "ffv_")
              for name in ACC)
PLANTS = ("none", "header", "field-name", "field-dims", "truncation",
          "missing-field", "zero-sign", "swapped-rank", "restart-byte")


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def _groups(raw: bytes, offset: int, nfields: int, owned: tuple[int, int],
            path: Path, plant: str) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for index in range(nfields):
        require(offset + 16 + GROUP.size <= len(raw),
                f"{path.name}: truncated field header {index}")
        name = raw[offset:offset + 16].decode("ascii", "replace").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = GROUP.unpack_from(raw, offset)
        offset += GROUP.size
        if plant == "field-name" and index == 0:
            name = "wrong"
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name and name not in groups, f"{path.name}: duplicate field {name!r}")
        require((ndim, n1, n2, n3) == (2, *owned, 1),
                f"{path.name}: bad dimensions for {name!r}: {(ndim, n1, n2, n3)}")
        end = offset + 8 * n1 * n2
        require(end <= len(raw), f"{path.name}: truncated payload {name!r}")
        values = np.frombuffer(raw, dtype="=f8", count=n1 * n2,
                               offset=offset).copy().reshape((n1, n2), order="F")
        require(bool(np.isfinite(values).all()), f"{path.name}: nonfinite {name!r}")
        groups[name] = values
        offset = end
    require(offset == len(raw), f"{path.name}: trailing bytes")
    return groups


def read_operand(path: Path, plant: str = "none") -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 16 + HEADER.size, f"{path.name}: too short")
    magic = raw[:16].decode("ascii", "replace").rstrip(" \x00")
    if plant == "header":
        magic = "X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    header = list(HEADER.unpack_from(raw, 16))
    version, kt, rank, nx, ny, nz, nimpp, njmpp, ntsi, ntsj, ntei, ntej, bits, nfields = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require((version, kt, bits, nfields) == (1, 1, 64, 16),
            f"{path.name}: invalid identity {header}")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny and nz > 0,
            f"{path.name}: invalid owned bounds")
    owned = (ntei - ntsi + 1, ntej - ntsj + 1)
    groups = _groups(raw, 16 + HEADER.size, nfields, owned, path, plant)
    if plant == "missing-field":
        groups.pop(FIELDS[-1], None)
    require(set(groups) == set(FIELDS),
            f"{path.name}: fields {sorted(groups)} != {list(FIELDS)}")
    return {"rank": rank, "kt": kt, "shape": [nx, ny, nz],
            "origin": [nimpp, njmpp], "owned": [ntsi, ntsj, ntei, ntej],
            "groups": groups, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def read_final(path: Path, owned: tuple[int, int]) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    require(len(raw) >= 16 + FINAL_HEADER.size, f"{path.name}: too short")
    magic = raw[:16].decode("ascii", "replace").rstrip(" \x00")
    require(magic == FINAL_MAGIC, f"{path.name}: bad final magic {magic!r}")
    header = FINAL_HEADER.unpack_from(raw, 16)
    require(header[0] == 1 and header[1] == 1 and header[-1] == 8,
            f"{path.name}: bad final identity")
    groups = _groups(raw, 16 + FINAL_HEADER.size, 8, owned, path, "none")
    require(set(groups) == set(FINAL), f"{path.name}: bad final fields")
    return groups


def run(root: Path, final_root: Path, baseline: Path, plant: str) -> dict:
    coverage = np.zeros((148, 180), dtype=np.int8)
    rows = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r104_een_accum_rank{expected_rank:04d}_kt00000001.bin"
        applied = plant if expected_rank == 0 and plant not in ("none", "restart-byte") else "none"
        row = read_operand(path, applied)
        require(row["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        owned = (ntei - ntsi + 1, ntej - ntsj + 1)
        final = read_final(final_root / f"oracle_r98_een_coeff_rank{expected_rank:04d}_kt00000001.bin", owned)
        bit_differences = {}
        for index, (acc_name, scl_name, final_name) in enumerate(zip(ACC, SCL, FINAL)):
            acc = row["groups"][acc_name].copy()
            if plant == "zero-sign" and expected_rank == 0 and index == 0:
                zeros = np.flatnonzero(acc.view(np.uint64).ravel(order="F") & np.uint64(0x7fffffffffffffff) == 0)
                require(zeros.size > 0, "zero-sign plant found no zero accumulator")
                bits = acc.ravel(order="F").copy().view(np.uint64)
                bits[zeros[0]] ^= np.uint64(1 << 63)
                acc = bits.view(np.float64).reshape(acc.shape, order="F")
            rebuilt = row["groups"][scl_name] * acc
            unequal = int(np.count_nonzero(rebuilt.view(np.uint64) != final[final_name].view(np.uint64)))
            require(unequal == 0,
                    f"{path.name}: {final_name} != {scl_name} * {acc_name} at {unequal} cells")
            bit_differences[final_name] = unequal
        rows.append({key: value for key, value in row.items() if key != "groups"} |
                    {"reconstructed_final_bit_differences": bit_differences})
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
    return {"status": "PASS_R104_EEN_ACCUM_ADMISSION", "rank_coverage": "exactly-once",
            "records": rows, "terminal_restart_comparisons": restart_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root, args.final_root, args.baseline, args.plant)
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
    print("STATUS PASS_R104_EEN_ACCUM_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
