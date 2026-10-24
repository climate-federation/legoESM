#!/usr/bin/env python3
"""Admit the passive ORCA2 round-332 source-order and raw-V records."""
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
RAW_MAGIC = "NEMO_L4_R332RV"
HEADER_NAMES = (
    "version", "kt", "kmm", "krhs", "rank", "nx", "ny", "nz",
    "ntsi", "ntsj", "real_bits", "n_fields",
)
FIELDS = ("stage1_v_before", "stage1_v_add", "stage1_v_after", "stage1_v_mask",
          "tke_zmxlm", "tke_zmxld", "tke_pdlr")
RAW_HEADER_NAMES = (
    "version", "kt", "kstg", "kaa", "rank", "jpi", "jpj", "jpk",
    "real_bits", "n_fields",
)
RAW_FIELDS = ("raw_postfold_v",)
PLANTS = ("header", "field-order", "truncation", "stamp", "vor", "pdlr",
          "raw-header", "raw-truncation", "raw-checksum")
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "dynvor", "zdfphy", "zdftke")
WRITER_UNITS = ("dynzdf_round328_writer", "tke_round331_writer",
                "round332_writer")

GateError = r331.GateError
require = r331.require


def record_name(rank: int) -> str:
    return f"oracle_r332_source_kt00000001_r{rank:04d}.bin"


def raw_record_name(rank: int) -> str:
    return f"oracle_r332_raw_v_kt00000001_r{rank:04d}.bin"


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


def read_raw_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "raw-truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r331._take(stream, 16, "raw-V magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(RAW_HEADER_NAMES)}i",
        r331._take(stream, 4 * len(RAW_HEADER_NAMES), "raw-V header"))
    header = dict(zip(RAW_HEADER_NAMES, values, strict=True))
    if plant == "raw-header":
        header["kstg"] = 2
    require(magic == RAW_MAGIC, f"raw-V magic {magic!r} != {RAW_MAGIC!r}")
    require((header["version"], header["kt"], header["kstg"], header["kaa"],
             header["real_bits"], header["n_fields"]) ==
            (1, 1, 1, 2, 64, len(RAW_FIELDS)), "wrong raw-V record header")
    fields, checksums, order = {}, {}, []
    shape = (header["jpi"], header["jpj"], header["jpk"])
    for index in range(header["n_fields"]):
        name = r331._take(stream, 24, f"raw-V field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", r331._take(stream, 28, f"{name} metadata"))
        require((rank, n1, n2, n3, i0, j0, k0) == (3, *shape, 1, 1, 1),
                f"{name}: raw-V field lacks exact jpi,jpj,jpk extents")
        checksum, = struct.unpack("=Q", r331._take(stream, 8, f"{name} checksum"))
        payload = r331._take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(shape, order="F")
        require(bool(np.isfinite(array).all()), f"{name}: non-finite payload")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "raw-V record trailing bytes")
    require(tuple(order) == RAW_FIELDS, "raw-V field order differs from source order")
    if plant == "raw-checksum":
        fields[RAW_FIELDS[0]].flat[0] = np.nextafter(
            fields[RAW_FIELDS[0]].flat[0], np.inf)
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_semantics(record: dict, tke_wet: np.ndarray) -> dict[str, int]:
    f = record["fields"]
    vwet = f["stage1_v_mask"] != 0.0
    require(tke_wet.shape == f["stage1_v_before"].shape,
            "TKE wet-mask extents moved")
    require(vwet.shape == f["stage1_v_before"].shape,
            "v wet-mask extents moved")
    require(bool(np.all((f["stage1_v_mask"] == 0.0) |
                        (f["stage1_v_mask"] == 1.0))),
            "v wet mask is not binary")
    before = np.ascontiguousarray(f["stage1_v_before"]).view(np.uint64)
    added = np.ascontiguousarray(f["stage1_v_add"])
    after = np.ascontiguousarray(f["stage1_v_after"]).view(np.uint64)
    replay = np.ascontiguousarray(f["stage1_v_before"] + added).view(np.uint64)
    require(bool(np.array_equal(replay, after)),
            "vorticity source-order addition does not replay bitwise")
    changed = before != after
    value_equal = f["stage1_v_before"] == f["stage1_v_after"]
    require(bool(np.any(f["tke_zmxlm"][tke_wet] > 0.0)),
            "wet mixing length channel is inactive")
    require(bool(np.any(f["tke_zmxld"][tke_wet] > 0.0)),
            "wet dissipation length channel is inactive")
    pdlr_wet = tke_wet.copy()
    pdlr_wet[:, :, 0] = False
    require(bool(np.all((f["tke_pdlr"][pdlr_wet] >= 0.1) &
                        (f["tke_pdlr"][pdlr_wet] <= 1.0))),
            "wet inverse Prandtl ratio is outside [0.1, 1]")
    return {
        "changed_bits": int(np.count_nonzero(changed)),
        "wet_changed_bits": int(np.count_nonzero(changed & vwet)),
        "value_equal_bit_changes": int(np.count_nonzero(changed & value_equal)),
        "wet_value_equal_bit_changes": int(np.count_nonzero(changed & value_equal & vwet)),
    }


def _plant(record: dict, plant: str, tke_wet: np.ndarray) -> None:
    if plant == "vor":
        vwet = record["fields"]["stage1_v_mask"] != 0.0
        require(bool(np.any(vwet)), "vorticity plant lacks a wet v point")
        index = tuple(np.argwhere(vwet)[0])
        value = record["fields"]["stage1_v_after"][index]
        record["fields"]["stage1_v_after"][index] = np.nextafter(value, np.inf)
    elif plant == "pdlr":
        candidates = tke_wet & (record["fields"]["tke_pdlr"] == 1.0)
        candidates[:, :, 0] = False
        require(bool(np.any(candidates)), "pdlr plant lacks a wet upper-bound cell")
        index = tuple(np.argwhere(candidates)[0])
        record["fields"]["tke_pdlr"][index] = np.nextafter(1.0, np.inf)


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
    raw_records = {}
    wet_masks = {}
    for rank in (0, 1):
        record = read_record(
            root / record_name(rank), plant=plant if rank == 0 else None)
        require(record["header"]["rank"] == rank, "rank tag moved")
        r331._verify_checksums(record)
        records[rank] = record
        raw_record = read_raw_record(
            root / raw_record_name(rank),
            plant=(plant if rank == 0 and plant is not None
                   and plant.startswith("raw-") else None))
        require(raw_record["header"]["rank"] == rank, "raw-V rank tag moved")
        r331._verify_checksums(raw_record)
        raw_records[rank] = raw_record
        prior_record = r331.read_record(root / r331.record_name(rank))
        wet_masks[rank] = prior_record["fields"]["tke_wmask"] != 0.0
    if plant in ("vor", "pdlr"):
        _plant(records[0], plant, wet_masks[0])
    source_order_counts = {}
    raw_sign_census = {}
    for rank, record in records.items():
        source_order_counts[str(rank)] = _verify_semantics(record, wet_masks[rank])
        raw_v = raw_records[rank]["fields"][RAW_FIELDS[0]]
        zeros = raw_v == 0.0
        raw_sign_census[str(rank)] = {
            "negative_zero": int(np.count_nonzero(zeros & np.signbit(raw_v))),
            "positive_zero": int(np.count_nonzero(zeros & ~np.signbit(raw_v))),
            "nonzero": int(np.count_nonzero(~zeros)),
        }
        h = record["header"]
        prior_record = r331.read_record(root / r331.record_name(h["rank"]))
        require((h["nx"], h["ny"], h["nz"]) == (
            prior_record["header"]["nx"], prior_record["header"]["ny"],
            prior_record["header"]["nz"] + 1), "round-331/332 extents differ")
    return {
        "format": "nemo-testcase-l4-orca2-round332-source-order-raw-v-v2",
        "status": "PASS_R332_SOURCE_ORDER_ADMISSION",
        "producer_commit": produced,
        "round331_status": prior["status"],
        "source_order_counts": source_order_counts,
        "raw_sign_census": raw_sign_census,
        "records": {record_name(rank): records[rank]["sha256"] for rank in records},
        "raw_records": {
            raw_record_name(rank): raw_records[rank]["sha256"]
            for rank in raw_records},
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
