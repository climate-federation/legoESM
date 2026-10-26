#!/usr/bin/env python3
"""Fail-closed admission gate for the round-53 OVERFLOW ENS operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import nemo_testcase_l1_overflow_round50_pair_gate as r50_gate
import numpy as np


class GateError(RuntimeError):
    """The acquisition cannot be admitted."""


MAGIC = "NEMO_L1_R53ENS01"
HEADER = struct.Struct("=12i")
FIELD_HEADER = struct.Struct("=4i")
FIELDS = (
    "zwz_prediv",
    "zwz_postdiv",
    "zuav",
    "zwz_pair_u",
    "product_u",
    "rhs_before_u",
    "rhs_after_u",
    "zvau",
    "zwz_pair_v",
    "product_v",
    "rhs_before_v",
    "rhs_after_v",
)
HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE / "nemo_testcase_l1_overflow_round53_ens_operands"
MODULE = INSTRUMENT / "l1_r53_ens.F90"
PATCH = INSTRUMENT / "dynvor_round53.patch"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/DYN/dynvor.F90"
)
COMPILED = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _worktree_stamp() -> dict:
    try:
        from legoesm.ocean.fidelity.provenance import worktree_stamp
        return worktree_stamp()
    except Exception as error:  # pragma: no cover - environment-specific
        return {"unavailable": str(error)}


def _take(raw: bytes, offset: int, size: int, what: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {what}")
    return raw[offset:end], end


def _compiled_cme_value() -> int:
    source = COMPILED.read_text()
    matches = re.findall(r"np_CME\s*=\s*([0-9]+)", source)
    require(matches, f"{COMPILED}: np_CME declaration missing")
    require(len(set(matches)) == 1, f"{COMPILED}: ambiguous np_CME values")
    return int(matches[0])


def read_record(path: Path) -> dict:
    """Parse the record through physical EOF using only its declared shapes."""
    raw = path.read_bytes()
    chunk, offset = _take(raw, 0, 16, "magic")
    magic = chunk.decode("ascii").rstrip()
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    (version, kt, stage, kmm, kvor, nx, ny, nz, ntsi, ntsj,
     bits, nfields) = HEADER.unpack(chunk)
    require(version == 1, f"{path}: bad version {version}")
    require((kt, stage) == (3, 2), f"{path}: wrong time/stage {(kt, stage)}")
    require(kmm > 0, f"{path}: invalid Kmm {kmm}")
    require(kvor == _compiled_cme_value(), f"{path}: wrong kvor {kvor}")
    require(all(0 < value < 10_000 for value in (nx, ny, nz)),
            f"{path}: invalid declared shape {(nx, ny, nz)}")
    require(ntsi > 0 and ntsj > 0, f"{path}: invalid origin {(ntsi, ntsj)}")
    require(bits == 64, f"{path}: bad wp width {bits}")
    require(nfields == len(FIELDS),
            f"{path}: field count {nfields} != {len(FIELDS)}")

    fields: dict[str, np.ndarray] = {}
    payload_offsets: dict[str, int] = {}
    for expected_name in FIELDS:
        chunk, offset = _take(raw, offset, 16, f"{expected_name} name")
        name = chunk.decode("ascii").rstrip()
        require(name == expected_name,
                f"{path}: expected {expected_name!r}, got {name!r}")
        require(name not in fields, f"{path}: duplicate field {name}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size,
                              f"{name} dimensions")
        ndim, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        require((ndim, dx, dy, dz) == (3, nx, ny, nz),
                f"{path}: {name} bad declared shape {(ndim, dx, dy, dz)}")
        payload_offsets[name] = offset
        count = math.prod((dx, dy, dz))
        chunk, offset = _take(raw, offset, count * (bits // 8),
                              f"{name} payload")
        values = np.frombuffer(chunk, dtype="=f8").reshape(
            (dx, dy, dz), order="F"
        )
        require(np.isfinite(values).all(), f"{path}: {name} is non-finite")
        fields[name] = values
    require(offset == len(raw), f"{path}: trailing bytes after physical EOF")
    require(tuple(fields) == FIELDS, f"{path}: missing/reordered fields")
    return {
        "path": str(path),
        "header": {
            "version": version,
            "kt": kt,
            "stage": stage,
            "Kmm": kmm,
            "kvor": kvor,
            "shape": [nx, ny, nz],
            "origin": [ntsi, ntsj],
            "bits": bits,
            "nfields": nfields,
        },
        "fields": fields,
        "payload_offsets": payload_offsets,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stamp(path: Path, expected_commit: str, digest: str) -> dict:
    stamp_path = Path(f"{path}.stamp")
    parts = stamp_path.read_text().strip().split()
    require(len(parts) == 3, f"{stamp_path}: malformed stamp")
    got_digest, got_commit, got_name = parts
    require(got_name == path.name, f"{stamp_path}: wrong record name")
    require(got_commit == expected_commit, f"{stamp_path}: wrong producer commit")
    require(got_digest == digest, f"{stamp_path}: digest mismatch")
    return {"path": str(stamp_path), "commit": got_commit, "sha256": got_digest}


def _one_ulp_digest(path: Path, payload_offset: int) -> str:
    raw = bytearray(path.read_bytes())
    value = struct.unpack_from("=d", raw, payload_offset)[0]
    planted = np.nextafter(np.float64(value), np.float64(np.inf)).item()
    require(planted != value and np.isfinite(planted),
            f"{path}: payload plant cannot move finite value")
    struct.pack_into("=d", raw, payload_offset, planted)
    return hashlib.sha256(raw).hexdigest()


def preflight() -> dict:
    """Prove that the committed patch is additions-only and hits the run branch."""
    for path in (MODULE, PATCH, SOURCE, COMPILED):
        require(path.is_file(), f"missing {path}")
    removed = [line for line in PATCH.read_text().splitlines()
               if line.startswith("-") and not line.startswith("---")]
    require(not removed, f"patch removes/replaces source lines: {removed[:1]}")
    with tempfile.TemporaryDirectory(prefix="orca2-r53-preflight-") as tmp:
        target = Path(tmp) / "dynvor.F90"
        shutil.copyfile(SOURCE, target)
        result = subprocess.run(
            ["patch", "-s", str(target), str(PATCH)],
            text=True, capture_output=True, check=False,
        )
        require(result.returncode == 0,
                f"patch does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    sentinels = (
        "CALL r53_ens_begin", "CALL r53_ens_zwz(1", "CALL r53_ens_zwz(2",
        "CALL r53_ens_before", "CALL r53_ens_after", "CALL r53_ens_finish",
    )
    require(all(item in patched for item in sentinels),
            "patched source misses a round-53 sentinel")
    compiled = COMPILED.read_text()
    for statement in (
        "CASE ( np_CME )",
        "pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav *",
        "ntot = np_CME",
    ):
        require(statement in compiled, f"compiled branch misses {statement!r}")
    return {
        "status": "PREFLIGHT_PASS",
        "worktree": _worktree_stamp(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "compiled_sha256": hashlib.sha256(COMPILED.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "removed_source_lines": len(removed),
        "sentinels": list(sentinels),
        "compiled_kvor": _compiled_cme_value(),
    }


def admit(
    record: Path,
    parent_record: Path,
    expected_commit: str,
    plant: str | None,
) -> dict:
    parsed = read_record(record)
    parent = r50_gate.read_record(parent_record, "momentum", 2)
    parent_header = parent["header"]
    header = parsed["header"]
    require(header["kt"] == parent_header["kt"], "parent kt mismatch")
    require(header["Kmm"] == parent_header["Kmm"], "parent Kmm mismatch")
    require(header["shape"] == parent_header["shape"], "parent shape mismatch")
    require(header["origin"] == parent_header["origin"], "parent origin mismatch")
    require(header["bits"] == parent_header["bits"], "parent fp width mismatch")
    digest = parsed["sha256"]
    stamp_commit = expected_commit
    if plant == "payload":
        digest = _one_ulp_digest(record, parsed["payload_offsets"][FIELDS[0]])
    elif plant == "stamp":
        stamp_commit = f"{expected_commit}-plant"
    stamp = _stamp(record, stamp_commit, digest)
    return {
        "status": "AT_BAR",
        "producer_commit": expected_commit,
        "record": {key: value for key, value in parsed.items()
                   if key not in ("fields", "payload_offsets")},
        "parent_record": {
            "path": str(parent_record),
            "sha256": parent["sha256"],
            "header": parent_header,
        },
        "stamp": stamp,
        "plant": plant,
        "preflight": preflight(),
        "worktree": _worktree_stamp(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--parent-record", type=Path)
    parser.add_argument("--producer-commit", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("payload", "stamp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            require(args.record is None, "preflight does not accept a record")
            report = preflight()
        else:
            require(args.record is not None, "--record is required")
            require(args.parent_record is not None, "--parent-record is required")
            require(args.producer_commit is not None,
                    "--producer-commit is required")
            require(args.expect_commit is not None, "--expect-commit is required")
            produced = args.producer_commit.read_text().strip()
            require(produced == args.expect_commit,
                    "producer_commit.txt does not match --expect-commit")
            report = admit(
                args.record, args.parent_record, args.expect_commit, args.plant
            )
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateError, r50_gate.GateError, OSError, UnicodeError,
            struct.error, ValueError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
