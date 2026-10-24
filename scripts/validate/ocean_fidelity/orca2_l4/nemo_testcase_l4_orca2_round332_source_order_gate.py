#!/usr/bin/env python3
"""Admit the passive ORCA2 round-332 source-order discriminator."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as dyn_gate,
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
)

MAGIC = "NEMO_L4_R332SO"
HEADER_NAMES = (
    "version", "kt", "kmm", "krhs", "rank", "nx", "ny", "nz",
    "ntsi", "ntsj", "real_bits", "n_fields",
)
FIELDS = ("stage1_v_before", "stage1_v_add", "stage1_v_after",
          "tke_zmxlm", "tke_zmxld")
PLANTS = ("header", "field-order", "truncation", "stamp", "vor", "mxl")
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "dynvor", "zdfphy", "zdftke")
WRITER_UNITS = ("dynzdf_round328_writer", "tke_round331_writer",
                "round332_writer")

GateError = r331.GateError
require = r331.require


def record_name(rank: int) -> str:
    return f"oracle_r332_source_kt00000001_r{rank:04d}.bin"


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r331._take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        r331._take(stream, 4 * len(HEADER_NAMES), "header"))
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["kmm"] = 2
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["kmm"],
             header["krhs"], header["real_bits"], header["n_fields"])
            == (1, 1, 1, 3, 64, len(FIELDS)), "wrong record header")
    fields, checksums, order = {}, {}, []
    for index in range(header["n_fields"]):
        name = r331._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", r331._take(stream, 28, f"{name} metadata"))
        require((rank, n1, n2, n3, i0, j0, k0) == (
            3, header["nx"], header["ny"], header["nz"],
            header["ntsi"], header["ntsj"], 1),
            f"{name}: wrong metadata")
        checksum, = struct.unpack("=Q", r331._take(stream, 8, f"{name} checksum"))
        payload = r331._take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(
            (n1, n2, n3), order="F")
        require(bool(np.isfinite(array).all()), f"{name}: non-finite payload")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == FIELDS, "field order differs from source order")
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_semantics(record: dict) -> None:
    f = record["fields"]
    before = np.ascontiguousarray(f["stage1_v_before"]).view(np.uint64)
    added = np.ascontiguousarray(f["stage1_v_add"])
    after = np.ascontiguousarray(f["stage1_v_after"]).view(np.uint64)
    replay = np.ascontiguousarray(f["stage1_v_before"] + added).view(np.uint64)
    require(bool(np.array_equal(replay, after)),
            "vorticity source-order addition does not replay bitwise")
    changed = before != after
    require(bool(np.any(changed)), "vorticity observer captured no changed bits")
    require(bool(np.any(f["tke_zmxlm"] > 0.0)), "mixing length channel is inactive")
    require(bool(np.any(f["tke_zmxld"] > 0.0)), "dissipation length channel is inactive")


def _plant(record: dict, plant: str) -> None:
    if plant == "vor":
        record["fields"]["stage1_v_after"][:] = record["fields"]["stage1_v_before"]
    elif plant == "mxl":
        record["fields"]["tke_zmxlm"].fill(0.0)


def run_gate(root: Path, zdf_reference_dir: Path, expect_commit: str,
             plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    produced = (root / "producer_commit.txt").read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit,
            f"producer commit {produced} != {expect_commit}")
    # The extension must preserve and re-admit every round-331 boundary.
    prior = r331.run_gate(root, zdf_reference_dir, expect_commit)
    records = {}
    for rank in (0, 1):
        record = read_record(
            root / record_name(rank), plant=plant if rank == 0 else None)
        require(record["header"]["rank"] == rank, "rank tag moved")
        r331._verify_checksums(record)
        records[rank] = record
    if plant in ("vor", "mxl"):
        _plant(records[0], plant)
    for record in records.values():
        _verify_semantics(record)
        h = record["header"]
        prior_record = r331.read_record(root / r331.record_name(h["rank"]))
        require((h["nx"], h["ny"], h["nz"]) == (
            prior_record["header"]["nx"], prior_record["header"]["ny"],
            prior_record["header"]["nz"]), "round-331/332 extents differ")
    signed_zero_change = False
    for record in records.values():
        f = record["fields"]
        before_bits = np.ascontiguousarray(f["stage1_v_before"]).view(np.uint64)
        after_bits = np.ascontiguousarray(f["stage1_v_after"]).view(np.uint64)
        signed_zero_change |= bool(np.any(
            (before_bits != after_bits) &
            (f["stage1_v_before"] == f["stage1_v_after"])))
    require(signed_zero_change,
            "vorticity boundary captured no value-equal bit change")
    return {
        "format": "nemo-testcase-l4-orca2-round332-source-order-v1",
        "status": "PASS_R332_SOURCE_ORDER_ADMISSION",
        "producer_commit": produced,
        "round331_status": prior["status"],
        "records": {record_name(rank): records[rank]["sha256"] for rank in records},
    }


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong observer unit order")
    return dyn_gate.same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS,
        new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--zdf-reference-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--same-program", type=Path)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.same_program:
            entries = [item.split("=", 1) for item in args.observer_patch]
            result = same_program(
                args.same_program, args.reference_bld,
                {unit: Path(path) for unit, path in entries},
                new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(None not in (args.record_dir, args.zdf_reference_dir,
                                 args.expect_commit), "admission arguments incomplete")
            result = run_gate(args.record_dir, args.zdf_reference_dir,
                              args.expect_commit, args.plant)
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
