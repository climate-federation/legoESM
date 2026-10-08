#!/usr/bin/env python3
"""Fail-closed raw-byte reproducibility gate for the extended ORCA2 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nemo_testcase_l4_orca2_o1_acquisition_gate import exchange


def compare_files(path_a: Path, path_b: Path, *, plant: bool) -> dict:
    digest_a, digest_b = hashlib.sha256(), hashlib.sha256()
    size_a = size_b = 0
    exact = True
    first = True
    with path_a.open("rb") as handle_a, path_b.open("rb") as handle_b:
        while True:
            chunk_a = handle_a.read(1024 * 1024)
            chunk_b = handle_b.read(1024 * 1024)
            if plant and first and chunk_b:
                altered = bytearray(chunk_b)
                altered[0] ^= 1
                chunk_b = bytes(altered)
            first = False
            digest_a.update(chunk_a)
            digest_b.update(chunk_b)
            size_a += len(chunk_a)
            size_b += len(chunk_b)
            exact = exact and chunk_a == chunk_b
            if not chunk_a and not chunk_b:
                break
    return {
        "exact": exact,
        "bytes_a": size_a,
        "bytes_b": size_b,
        "sha256_a": digest_a.hexdigest(),
        "sha256_b": digest_b.hexdigest(),
    }


def inventory() -> set[str]:
    return exchange.phase1.expected_inventory() | {
        exchange.RECORD,
        "oracle_sbcblk_o1_kt00000001.bin",
        "oracle_stage1_wzv_operands_kt00000001.bin",
    }


def transport_operand_diff(path_a: Path, path_b: Path) -> dict[str, int]:
    n2, n3 = 94 * 152, 94 * 152 * 31
    layout = (
        ("e2u", n2), ("e3u", n3), ("uu_Kmm", n3), ("zub", n2),
        ("umask", n3), ("zFu", n3), ("e1v", n2), ("e3v", n3),
        ("vv_Kmm", n3), ("zvb", n2), ("vmask", n3), ("zFv", n3),
        ("un_adv", n2), ("r1_hu_Kmm", n2), ("uu_b_Kmm", n2),
        ("vn_adv", n2), ("r1_hv_Kmm", n2), ("vv_b_Kmm", n2),
    )
    a = np.fromfile(path_a, np.float64, offset=48)
    b = np.fromfile(path_b, np.float64, offset=48)
    if a.size != b.size or a.size != sum(count for _, count in layout):
        return {"schema_error": -1}
    result, offset = {}, 0
    for name, count in layout:
        unequal = int(np.count_nonzero(
            a[offset:offset + count].view(np.uint64)
            != b[offset:offset + count].view(np.uint64)
        ))
        if unequal:
            result[name] = unequal
        offset += count
    return result


def validate(a: Path, b: Path, *, plant: str | None) -> tuple[dict, int]:
    expected = inventory()
    names_a = {path.name for path in a.glob("oracle_*.bin")}
    names_b = {path.name for path in b.glob("oracle_*.bin")}
    rows = {}
    exact = 0
    for name in sorted(expected):
        if name in names_a and name in names_b:
            row = compare_files(a / name, b / name, plant=plant == name)
        else:
            row = {
                "exact": False,
                "bytes_a": (a / name).stat().st_size if name in names_a else 0,
                "bytes_b": (b / name).stat().st_size if name in names_b else 0,
                "sha256_a": None,
                "sha256_b": None,
            }
        same = row["exact"]
        exact += int(same)
        rows[name] = row
    complete = names_a == names_b == expected
    status = "PASS" if complete and exact == len(expected) else "FAIL"
    result = {
        "status": status,
        "rule": f"{len(expected)} / {len(expected)} oracle records raw-byte identical",
        "inventory_complete": complete,
        "expected_count": len(expected),
        "actual_count_a": len(names_a),
        "actual_count_b": len(names_b),
        "raw_exact": exact,
        "raw_total": len(expected),
        "extra_a": sorted(names_a - expected),
        "missing_a": sorted(expected - names_a),
        "extra_b": sorted(names_b - expected),
        "missing_b": sorted(expected - names_b),
        "records": rows,
    }
    transport = "oracle_rkstage1_transport_operands_kt00000001.bin"
    if transport in names_a and transport in names_b:
        result["transport_operand_differing_f64"] = transport_operand_diff(
            a / transport, b / transport
        )
    return result, 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--plant", choices=sorted(inventory()))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result, code = validate(args.arm_a, args.arm_b, plant=args.plant)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
