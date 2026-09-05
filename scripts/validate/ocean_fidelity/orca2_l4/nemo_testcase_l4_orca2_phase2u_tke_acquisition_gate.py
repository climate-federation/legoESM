#!/usr/bin/env python3
"""Admit the canonical kt=2 TKE-walk twins and their inherited V2 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)

MAGIC = "NEMO_L4_TKEW_1"
RECORD = "oracle_tke_walk_kt00000002.bin"
HEADER_INTS = 15
FIELDS_3D = (
    "zpelc", "en_post_lc", "pdlr", "zdiag_pre_solve", "zd_lw_pre_solve",
    "zd_up_pre_solve", "en_rhs_pre_solve", "zdiag_after_forward",
    "zd_lw_after_forward", "en_post_solve", "en_post_etau", "mxlm",
    "mxld", "avm_post", "avt_post", "dissl_post",
)
FIELDS_2D = ("zice_fra", "zWlc2", "imlc_real", "zhlc", "zus3")
FIELDS = FIELDS_3D + FIELDS_2D


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tke(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * HEADER_INTS)
        require(len(raw_header) == 4 * HEADER_INTS, "truncated base header")
        header = struct.unpack(f"={HEADER_INTS}i", raw_header)
        (version, kt, kbb, kmm, jpi, jpj, jpk, real_bits, n3, n2, payload,
         nn_eice, nn_etau, nn_mxl, nn_pdl) = header
        require(magic == MAGIC, f"magic {magic!r}")
        require((version, kt, kbb, kmm) == (1, 2, 3, 3),
                f"version/levels {version}/{kt}/{kbb}/{kmm}")
        require((real_bits, n3, n2) == (64, len(FIELDS_3D), len(FIELDS_2D)),
                f"precision/counts {real_bits}/{n3}/{n2}")
        require((nn_eice, nn_etau, nn_mxl, nn_pdl) == (1, 1, 3, 1),
                f"resolved selectors {nn_eice}/{nn_etau}/{nn_mxl}/{nn_pdl}")
        raw_extents = handle.read(4 * 3 * len(FIELDS))
        require(len(raw_extents) == 4 * 3 * len(FIELDS), "truncated extents")
        extents = np.frombuffer(raw_extents, dtype=np.int32).reshape(
            (len(FIELDS), 3))
        expected = [(jpi, jpj, jpk)] * len(FIELDS_3D) + [
            (jpi, jpj, 1)] * len(FIELDS_2D)
        require(tuple(map(tuple, extents.tolist())) == tuple(expected),
                "extent table differs from writer SHAPE expressions")
        derived = sum(int(np.prod(shape)) for shape in expected)
        require(payload == derived, f"payload {payload} != derived {derived}")

        rows = []
        arrays: dict[str, np.ndarray] = {}
        for name, shape in zip(FIELDS, expected, strict=True):
            count = int(np.prod(shape))
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count, f"{name}: truncated payload")
            value = np.frombuffer(raw, dtype=np.float64).reshape(shape, order="F")
            require(np.isfinite(value).all(), f"{name}: non-finite")
            halo = np.ones(shape, dtype=bool)
            halo[2:-2, 2:-2, :] = False
            require(np.all(value[halo] == 0.0), f"{name}: noncanonical halo")
            arrays[name] = value
            rows.append({"name": name, "extent": list(shape), "count": count,
                         "sha256": hashlib.sha256(raw).hexdigest()})
        require(handle.read(1) == b"", "trailing bytes")

    expected_bytes = 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS) + 8 * payload
    require(path.stat().st_size == expected_bytes, "exact-EOF byte count")
    require(np.count_nonzero(arrays["zWlc2"]) > 0, "vacuous Langmuir operand")
    require(np.count_nonzero(arrays["en_post_solve"]) > 0, "vacuous solve target")
    require(np.count_nonzero(arrays["avm_post"]) > 0, "vacuous closure target")
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "magic": magic, "header": list(header), "extent_table": extents.tolist(),
        "extent_table_sha256": hashlib.sha256(raw_extents).hexdigest(),
        "extent_derived_payload_f64": derived, "fields": rows,
        "nonzero": {name: int(np.count_nonzero(arrays[name])) for name in FIELDS},
        "exact_eof": True,
    }


def validate(run_a: Path, run_b: Path, baseline: Path,
             identity_control: Path) -> dict[str, object]:
    names_a = {p.name for p in run_a.glob("oracle_*.bin")}
    names_b = {p.name for p in run_b.glob("oracle_*.bin")}
    names_old = {p.name for p in baseline.glob("oracle_*.bin")}
    require(names_a == names_b == names_old | {RECORD},
            "inventory is not the 100-record V2 root plus one TKE record")
    twins = []
    for name in sorted(names_a):
        require((run_a / name).read_bytes() == (run_b / name).read_bytes(),
                f"twin differs: {name}")
        twins.append({"file": name, "sha256": sha256(run_a / name)})
    inherited = []
    for name in sorted(names_old):
        require((run_a / name).read_bytes() == (baseline / name).read_bytes(),
                f"inherited record differs: {name}")
        inherited.append({"file": name, "sha256": sha256(run_a / name)})
    return {
        "status": "PASS", "schema": read_tke(run_a / RECORD),
        "twin_raw_exact": len(twins), "twin_total": len(twins),
        "inherited_raw_exact": len(inherited), "inherited_total": len(inherited),
        "records": twins,
        "ordinary_output_identity": phase1.validate_identity(identity_control, run_a),
    }


def _overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-p2u-tke-plant-"))
    for item in root.iterdir():
        os.symlink(item, target / item.name)
    target_record = target / RECORD
    target_record.unlink()
    shutil.copyfile(root / RECORD, target_record)
    return target


def _patch(path: Path, offset: int, raw: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(raw)


def run_plant(name: str, run_a: Path, run_b: Path, baseline: Path,
              identity_control: Path) -> None:
    planted = _overlay(run_a)
    record = planted / RECORD
    try:
        if name == "magic":
            _patch(record, 0, b"X")
        elif name == "extent":
            _patch(record, 16 + 4 * HEADER_INTS, struct.pack("=i", 93))
        elif name == "count":
            _patch(record, 16 + 4 * 10, struct.pack("=i", 1))
        elif name == "truncated":
            record.write_bytes(record.read_bytes()[:-8])
        elif name == "trailing":
            with record.open("ab") as handle:
                handle.write(b"X")
        elif name == "canonical":
            payload_offset = 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS)
            _patch(record, payload_offset, struct.pack("=d", 1.0))
        elif name == "twin":
            _patch(record, 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS),
                   struct.pack("=d", 1.0))
            validate(planted, run_b, baseline, identity_control)
            raise GateError("twin plant escaped")
        else:
            raise GateError(f"unknown plant {name}")
        read_tke(record)
        raise GateError(f"{name} plant escaped")
    except GateError as exc:
        if "escaped" in str(exc):
            raise
        print(f"FAIL: binding {name} plant rejected: {exc}")
        raise SystemExit(1)
    finally:
        shutil.rmtree(planted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=("magic", "extent", "count", "truncated",
                                             "trailing", "canonical", "twin"))
    args = parser.parse_args()
    if args.plant:
        run_plant(args.plant, args.run_a, args.run_b, args.baseline,
                  args.identity_control)
    report = validate(args.run_a, args.run_b, args.baseline, args.identity_control)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {report['twin_raw_exact']}/{report['twin_total']} twins; "
          f"{report['inherited_raw_exact']}/{report['inherited_total']} inherited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
