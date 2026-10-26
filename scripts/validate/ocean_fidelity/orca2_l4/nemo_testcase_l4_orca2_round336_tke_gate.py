#!/usr/bin/env python3
"""Admit the round-331/332 TKE internals extended to ORCA2 kt=2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round336_trazdf_gate as trazdf,
)

MAGIC = "NEMO_L4_R336TK2"
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "rank", "nx", "ny", "nz",
    "ntsi", "ntsj", "real_bits", "n_fields",
)
FIELDS = ("tke_zmxlm", "tke_zmxld", "tke_pdlr")
PLANTS = (
    "header", "field-order", "truncation", "checksum", "stamp",
    "final", "mxl", "pdlr", "time-level",
)

GateError = r331.GateError
require = r331.require


def record_name(rank: int) -> str:
    return f"oracle_r336_tke_mxl_kt00000002_r{rank:04d}.bin"


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r331._take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        r331._take(stream, 4 * len(HEADER_NAMES), "header"),
    )
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant in {"header", "time-level"}:
        header["kmm"] = -1
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require(
        (header["version"], header["kt"], header["kbb"], header["kmm"],
         header["real_bits"], header["n_fields"]) == (1, 2, 3, 3, 64, 3),
        "wrong kt=2 mixing-length header",
    )
    require(min(header["nx"], header["ny"], header["nz"]) > 0,
            "non-positive record extent")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = r331._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", r331._take(stream, 28, f"{name} metadata"))
        require(rank == 3, f"{name}: bad array rank {rank}")
        require((n1, n2, n3) == (header["nx"], header["ny"], header["nz"]),
                f"{name}: wrong extent {(n1, n2, n3)}")
        require((i0, j0, k0) == (header["ntsi"], header["ntsj"], 1),
                f"{name}: wrong origin")
        checksum, = struct.unpack("=Q", r331._take(stream, 8, f"{name} checksum"))
        payload = r331._take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(
            (n1, n2, n3), order="F")
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == FIELDS, "field order differs from TKE source order")
    record = {"header": header, "fields": fields, "checksums": checksums,
              "sha256": hashlib.sha256(raw).hexdigest()}
    if plant == "checksum":
        record["fields"][FIELDS[0]].flat[0] = np.nextafter(
            record["fields"][FIELDS[0]].flat[0], np.inf)
    r331._verify_checksums(record)
    return record


def _wet_index(wet: np.ndarray) -> tuple[np.ndarray, ...]:
    indices = np.nonzero(wet)
    require(bool(indices[0].size), "kt=2 TKE record has no wet interfaces")
    return tuple(axis[:1] for axis in indices)


def verify_records(tke: dict, mxl: dict, phase: dict,
                   plant: str | None = None) -> dict[str, int]:
    require(plant is None or plant in {"final", "mxl", "pdlr"},
            f"unsupported semantic plant {plant}")
    th, mh = tke["header"], mxl["header"]
    require((th["kt"], th["kbb"], th["kmm"], th["krhs"]) == (2, 3, 3, 1),
            "wrong kt=2 TKE solver levels")
    require((th["rank"], th["nx"], th["ny"], th["nz"] + 1) ==
            (mh["rank"], mh["nx"], mh["ny"], mh["nz"]),
            "round-331/332 kt=2 record headers differ")
    r331._verify_source_order(tke)
    wet_full = tke["fields"]["tke_wmask"] != 0.0
    wet_interior = wet_full[..., 1:]
    pdlr_support = wet_full.copy()
    pdlr_support[..., 0] = False
    full_index = _wet_index(wet_full)
    interior_index = _wet_index(wet_interior)
    tf, mf = tke["fields"], mxl["fields"]
    if plant == "final":
        tf["after_tke_avt"][interior_index] = np.nextafter(
            tf["after_tke_avt"][interior_index], np.inf)
    elif plant == "mxl":
        mf["tke_zmxlm"][full_index] = -1.0
    elif plant == "pdlr":
        candidates = pdlr_support & (mf["tke_pdlr"] == 1.0)
        pdlr_index = _wet_index(candidates)
        mf["tke_pdlr"][pdlr_index] = np.nextafter(1.0, np.inf)
    phase_avt = phase["fields"]["avt_after_tke"]
    require(phase_avt.shape == tf["after_tke_avt"].shape,
            "round-336 TKE phase extent differs")
    require(np.array_equal(phase_avt.view(np.uint64),
                           tf["after_tke_avt"].view(np.uint64)),
            "kt=2 TKE final avt differs from admitted zdf_phy handoff")
    require(bool(np.all(mf["tke_zmxlm"][wet_full] >= 0.0)) and
            bool(np.all(mf["tke_zmxld"][wet_full] >= mf["tke_zmxlm"][wet_full])),
            "kt=2 wet mixing lengths violate NEMO's final bounds")
    require(bool(np.all(mf["tke_pdlr"][pdlr_support] >= 0.1)) and
            bool(np.all(mf["tke_pdlr"][pdlr_support] <= 1.0)),
            "kt=2 wet inverse Prandtl ratio is outside [0.1,1]")
    return {
        "wet_interfaces": int(np.count_nonzero(wet_full)),
        "final_avt_unequal": 0,
    }


def run_gate(root: Path, phase_root: Path, expect_commit: str,
             plant: str | None = None) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit,
            f"producer commit {produced} != {expect_commit}")
    phase_stamp = phase_root / "producer_commit.txt"
    require(phase_stamp.is_file(), "phase reference lacks producer_commit.txt")
    phase_producer = phase_stamp.read_text().strip()
    phase_admission = trazdf.run_gate(phase_root, expect_commit=phase_producer)
    require(phase_admission["status"] == "PASS_R336_TRAZDF_ACQUISITION",
            "round-336 phase reference is not admitted")
    counts = {}
    hashes = {}
    for rank in (0, 1):
        tke_path = root / r331.record_name(rank, kt=2)
        mxl_path = root / record_name(rank)
        phase_path = phase_root / trazdf.record_name(rank)
        require(tke_path.is_file(), f"missing {tke_path.name}")
        require(mxl_path.is_file(), f"missing {mxl_path.name}")
        require(phase_path.is_file(), f"missing phase reference {phase_path.name}")
        tke = r331.read_record(tke_path, expected_kt=2)
        r331._verify_checksums(tke)
        mxl = read_record(
            mxl_path,
            plant=plant if rank == 0 and plant in {
                "header", "field-order", "truncation", "checksum", "time-level"
            } else None,
        )
        phase = trazdf.read_record(phase_path)
        trazdf._verify_checksums(phase)
        counts[str(rank)] = verify_records(
            tke, mxl, phase,
            plant=plant if rank == 0 and plant in {"final", "mxl", "pdlr"} else None,
        )
        hashes[str(rank)] = {
            "tke": tke["sha256"], "mixing_length": mxl["sha256"],
            "phase_reference": phase["sha256"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round336-tke-kt2-admission-v1",
        "status": "PASS_R336_TKE_KT2_ADMISSION",
        "producer_commit": produced,
        "phase_reference_producer": phase_producer,
        "counts": counts,
        "sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--phase-reference-dir", required=True, type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_gate(args.record_dir, args.phase_reference_dir,
                          args.expect_commit, args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, ValueError, KeyError) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R336_TKE_KT2_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
