#!/usr/bin/env python3
"""Admit the rank-complete round-110 EEN fraction/halo record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round107_een_step_acquisition import (
    check_record as step_record,
)


MAGIC = b"NEMO_L4_R110EF1"
HEADER = struct.Struct("=14i")
GROUP = struct.Struct("=4i")
FIELDS = (
    "west_ff", "west_e3f0", "west_r3f", "west_mask", "west_denom", "frac_west",
    "center_ff", "center_e3f0", "center_r3f", "center_mask", "center_denom", "frac_center",
    "south_ff", "south_e3f0", "south_r3f", "south_mask", "south_denom", "frac_south",
    "sum_west_center", "sum_all", "mbku",
)
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation", "missing-field",
    "duplicate-rank", "bottom", "denominator", "quotient", "sum", "inherited",
    "restart-byte",
)


class Refusal(RuntimeError):
    """The record is incomplete, malformed, active, or arithmetically inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    return left.shape == right.shape and bool(np.array_equal(
        left.view(np.uint64), right.view(np.uint64)))


def read_record(path: Path, raw_override: bytes | None = None) -> dict:
    raw = path.read_bytes() if raw_override is None else raw_override
    require(len(raw) >= 16 + HEADER.size, f"{path.name}: truncated header")
    require(raw[:16].rstrip(b" \0") == MAGIC, f"{path.name}: bad magic")
    values = HEADER.unpack_from(raw, 16)
    (version, kt, rank, jpi, jpj, jpk, nimpp, njmpp, ntsi, ntsj,
     ntei, ntej, bits, nfields) = values
    require((version, kt, bits, nfields) == (1, 1, 64, len(FIELDS)),
            f"{path.name}: header contract moved")
    require(jpi > 0 and jpj > 0 and jpk > 1, f"{path.name}: invalid local shape")
    require(1 <= ntsi <= ntei <= jpi and 1 <= ntsj <= ntej <= jpj,
            f"{path.name}: invalid owned bounds")
    offset = 16 + HEADER.size
    groups: dict[str, np.ndarray] = {}
    for index in range(nfields):
        require(offset + 16 + GROUP.size <= len(raw),
                f"{path.name}: truncated field header {index}")
        try:
            name = raw[offset:offset + 16].decode("ascii").rstrip(" \0")
        except UnicodeDecodeError as error:
            raise Refusal(f"{path.name}: non-ASCII field header {index}") from error
        offset += 16
        ndim, n1, n2, n3 = GROUP.unpack_from(raw, offset)
        offset += GROUP.size
        require(name and name not in groups,
                f"{path.name}: duplicate/bad field {index} {name!r}")
        require(ndim in (2, 3) and min(n1, n2, n3) > 0,
                f"{path.name}: bad dimensions for {name}")
        count = n1 * n2 * n3
        require(offset + 8 * count <= len(raw), f"{path.name}: truncated payload {name}")
        array = np.frombuffer(raw, dtype="=f8", count=count, offset=offset).copy()
        offset += 8 * count
        groups[name] = array.reshape((n1, n2, n3), order="F")
    require(offset == len(raw), f"{path.name}: parser missed EOF")
    require(tuple(groups) == FIELDS, f"{path.name}: field registry/order moved")
    n1, n2 = ntei - ntsi + 1, ntej - ntsj + 1
    for name, array in groups.items():
        expected = (n1, n2, 1 if name == "mbku" else jpk)
        require(array.shape == expected, f"{path.name}: {name} dimensions moved")
        require(bool(np.all(np.isfinite(array))), f"{path.name}: {name} is non-finite")
    return {
        "path": str(path), "rank": rank, "shape": (jpi, jpj, jpk),
        "origin": (nimpp, njmpp), "owned": (ntsi, ntsj, ntei, ntej),
        "groups": groups, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _restart_identity(root: Path, source_root: Path, plant: str) -> list[dict]:
    rows = []
    for step in range(1, 11):
        for rank in range(2):
            name = f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            candidate = (root / name).read_bytes()
            reference = (source_root / name).read_bytes()
            if plant == "restart-byte" and step == 1 and rank == 0:
                candidate = bytes([candidate[0] ^ 1]) + candidate[1:]
            require(candidate == reference, f"terminal restart moved: {name}")
            rows.append({"name": name, "bytes": len(candidate),
                         "sha256": hashlib.sha256(candidate).hexdigest()})
    return rows


def run(root: Path, source_root: Path, plant: str) -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    arithmetic = []
    for expected_rank in range(2):
        path = root / f"oracle_r110_een_fraction_rank{expected_rank:04d}_kt00000001.bin"
        raw = bytearray(path.read_bytes())
        if plant == "header" and expected_rank == 0:
            raw[0] ^= 1
        elif plant == "field-name" and expected_rank == 0:
            raw[16 + HEADER.size] = ord("x")
        elif plant == "field-dims" and expected_rank == 0:
            offset = 16 + HEADER.size + 16
            ndim, n1, n2, n3 = GROUP.unpack_from(raw, offset)
            GROUP.pack_into(raw, offset, ndim, n1 + 1, n2, n3)
        elif plant == "truncation" and expected_rank == 0:
            raw = raw[:-8]
        elif plant == "missing-field" and expected_rank == 0:
            struct.pack_into("=i", raw, 16 + 13 * 4, len(FIELDS) - 1)
        row = read_record(path, bytes(raw))
        if plant == "duplicate-rank" and expected_rank == 1:
            row["rank"] = 0
        require(row["rank"] == expected_rank, f"{path.name}: rank moved")
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned placement moved")
        coverage[j0:j1, i0:i1] += 1

        groups = row["groups"]
        bottom = np.array(groups["mbku"][..., 0], copy=True)
        if plant == "bottom" and expected_rank == 0:
            bottom[0, 0] = 0.0
        require(bool(np.all(bottom == np.rint(bottom))), f"{path.name}: mbku is not integral")
        require(bool(np.all((bottom >= 1) & (bottom < row["shape"][2]))),
                f"{path.name}: mbku enters dummy level or is invalid")
        levels = np.arange(1, row["shape"][2] + 1)[None, None, :]
        executed = levels <= bottom[..., None]
        for name in FIELDS[:-1]:
            require(bool(np.all(groups[name][~executed] == 0.0)),
                    f"{path.name}: {name} wrote outside mbku")

        test_groups = {name: np.array(value, copy=True) for name, value in groups.items()}
        if plant == "denominator" and expected_rank == 0:
            test_groups["west_denom"].view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        elif plant == "quotient" and expected_rank == 0:
            test_groups["frac_west"].view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        elif plant == "sum" and expected_rank == 0:
            test_groups["sum_all"].view(np.uint64)[0, 0, 0] ^= np.uint64(1)

        for label in ("west", "center", "south"):
            denominator = test_groups[f"{label}_e3f0"] * (
                np.float64(1.0) + test_groups[f"{label}_r3f"] * test_groups[f"{label}_mask"])
            require(bit_equal(test_groups[f"{label}_denom"][executed], denominator[executed]),
                    f"{path.name}: {label} denominator does not replay")
            quotient = test_groups[f"{label}_ff"] / test_groups[f"{label}_denom"]
            require(bit_equal(test_groups[f"frac_{label}"][executed], quotient[executed]),
                    f"{path.name}: {label} quotient does not replay")
        partial = test_groups["frac_west"] + test_groups["frac_center"]
        require(bit_equal(test_groups["sum_west_center"][executed], partial[executed]),
                f"{path.name}: west+center partial sum does not replay")
        total = test_groups["sum_west_center"] + test_groups["frac_south"]
        require(bit_equal(test_groups["sum_all"][executed], total[executed]),
                f"{path.name}: final ordered sum does not replay")

        inherited = step_record.read_record(
            source_root / f"oracle_r107_een_step_rank{expected_rank:04d}_kt00000001.bin")
        inherited_zpvo = np.array(inherited["groups"]["zpvo_nw"], copy=True)
        if plant == "inherited" and expected_rank == 0:
            inherited_zpvo.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        require(bit_equal(test_groups["sum_all"][executed], inherited_zpvo[executed]),
                f"{path.name}: ordered sum moved from admitted zpvo_nw")
        arithmetic.append({"rank": expected_rank, "executed_cells": int(np.count_nonzero(executed)),
                           "denominators_exact": True, "quotients_exact": True,
                           "ordered_sum_exact": True, "admitted_zpvo_exact": True})
        records.append({key: row[key] for key in (
            "rank", "shape", "origin", "owned", "bytes", "sha256")})

    require(bool(np.all(coverage == 1)), "rank slabs do not cover the domain exactly once")
    restarts = _restart_identity(root, source_root, plant)
    return {"status": "PASS_R110_EEN_FRACTION_ADMISSION", "coverage": "exactly-once",
            "records": records, "arithmetic": arithmetic, "terminal_restarts": restarts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root, args.source_root, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, Refusal) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
