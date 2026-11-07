#!/usr/bin/env python3
"""Fail-closed admission gate for round-62 OVERFLOW dyn_zdf frames."""

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
import nemo_testcase_l1_overflow_round53_ens_operands_gate as r53_gate
import nemo_testcase_l1_overflow_round56_up3_operands_gate as r56_gate
import numpy as np


class GateError(RuntimeError):
    """The acquisition cannot be admitted."""


MAGIC = "NEMO_L1_R62ZDF01"
HEADER = struct.Struct("=14i")
FIELD_HEADER = struct.Struct("=4i")
FIELDS = (
    "explicit_u", "explicit_v",
    "baro_subtract_u", "baro_subtract_v",
    "baro_drag_u", "baro_drag_v",
    "implicit_solve_u", "implicit_solve_v",
)
HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE / "nemo_testcase_l1_overflow_round62_dynzdf"
MODULE = INSTRUMENT / "l1_r62_dynzdf.F90"
PATCH = INSTRUMENT / "dynzdf_round62.patch"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/DYN/dynzdf.F90"
)
COMPILED = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90"
)
RECORD_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds"
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


def read_record(path: Path) -> dict:
    """Parse through physical EOF using only the record's declared shapes."""
    raw = path.read_bytes()
    chunk, offset = _take(raw, 0, 16, "magic")
    magic = chunk.decode("ascii").rstrip()
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz,
     ntsi, ntsj, bits, nfields) = HEADER.unpack(chunk)
    require(version == 1, f"{path}: bad version {version}")
    require((kt, stage) == (3, 3), f"{path}: wrong time/stage {(kt, stage)}")
    require((kbb, kmm, krhs, kaa) == (1, 2, 3, 3),
            f"{path}: wrong RK slots {(kbb, kmm, krhs, kaa)}")
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
                f"{path}: {name} bad shape {(ndim, dx, dy, dz)}")
        payload_offsets[name] = offset
        chunk, offset = _take(raw, offset, math.prod((dx, dy, dz)) * 8,
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
            "version": version, "kt": kt, "stage": stage,
            "Kbb": kbb, "Kmm": kmm, "Krhs": krhs, "Kaa": kaa,
            "shape": [nx, ny, nz], "origin": [ntsi, ntsj],
            "bits": bits, "nfields": nfields,
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


def _inventory() -> dict:
    """Parse the unique admitted records that extend the round-50 lineage."""
    r50_dir = RECORD_ROOT / "round50/acquisition/oracle_overflow_kt3_pair"
    r53_path = (RECORD_ROOT / "round53/acquisition/oracle_overflow_ens_operands/"
                "oracle_r53_ens_kt00000003_s2.bin")
    r56_path = (RECORD_ROOT / "round56/acquisition/oracle_overflow_up3_operands/"
                "oracle_r56_up3_kt00000003_s2.bin")
    records: dict[str, list[str]] = {}
    for stage in (1, 2, 3):
        for kind in ("momentum", "tracer"):
            path = r50_dir / f"oracle_r50_{kind}_kt00000003_s{stage}.bin"
            parsed = r50_gate.read_record(path, kind, stage)
            records[path.name] = list(parsed["fields"])
    records[r53_path.name] = list(r53_gate.read_record(r53_path)["fields"])
    records[r56_path.name] = list(r56_gate.read_up3_record(r56_path)["fields"])
    required = set(FIELDS)
    carriers = [name for name, fields in records.items()
                if required.issubset(fields)]
    require(not carriers,
            f"existing admitted record already carries dyn_zdf frames: {carriers}")
    return {"records": records, "required_fields": list(FIELDS),
            "complete_carriers": carriers}


def preflight() -> dict:
    """Prove the patch is additions-only and targets the compiled branch."""
    for path in (MODULE, PATCH, SOURCE, COMPILED):
        require(path.is_file(), f"missing {path}")
    removed = [line for line in PATCH.read_text().splitlines()
               if line.startswith("-") and not line.startswith("---")]
    require(not removed, f"patch removes/replaces source lines: {removed[:1]}")
    with tempfile.TemporaryDirectory(prefix="orca2-r62-preflight-") as tmp:
        target = Path(tmp) / "dynzdf.F90"
        shutil.copyfile(SOURCE, target)
        result = subprocess.run(
            ["patch", "-s", str(target), str(PATCH)],
            text=True, capture_output=True, check=False,
        )
        require(result.returncode == 0,
                f"patch does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    sentinels = (
        "CALL r62_zdf_begin", "CALL r62_zdf_row(1",
        "CALL r62_zdf_row(2", "CALL r62_zdf_row(3",
        "CALL r62_zdf_u_solve", "CALL r62_zdf_v_solve",
        "CALL r62_zdf_finish",
    )
    require(all(patched.count(item) == 1 for item in sentinels),
            "patched source misses or duplicates a round-62 sentinel")
    compiled = COMPILED.read_text()
    for statement in (
        "IF( ln_drgimp .AND. ln_dynspg_ts ) THEN",
        "puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b",
        "puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zws",
        "pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - zws",
    ):
        require(statement in compiled, f"compiled branch misses {statement!r}")
    return {
        "status": "PREFLIGHT_PASS", "worktree": _worktree_stamp(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "compiled_sha256": hashlib.sha256(COMPILED.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "removed_source_lines": len(removed), "sentinels": list(sentinels),
        "inventory": _inventory(),
    }


def admit(record: Path, parent_record: Path, expected_commit: str,
          plant: str | None) -> dict:
    parsed = read_record(record)
    parent = r50_gate.read_record(parent_record, "momentum", 3)
    header, parent_header = parsed["header"], parent["header"]
    require(header["kt"] == parent_header["kt"], "parent kt mismatch")
    require(header["shape"] == parent_header["shape"], "parent shape mismatch")
    require(header["origin"] == parent_header["origin"], "parent origin mismatch")
    require(header["bits"] == parent_header["bits"], "parent fp width mismatch")
    require(np.array_equal(parsed["fields"]["implicit_solve_u"],
                           parent["fields"]["raw_kaa_u"]),
            "implicit U endpoint differs from parent raw Kaa")
    require(np.array_equal(parsed["fields"]["implicit_solve_v"],
                           parent["fields"]["raw_kaa_v"]),
            "implicit V endpoint differs from parent raw Kaa")
    digest = parsed["sha256"]
    stamp_commit = expected_commit
    if plant == "payload":
        digest = _one_ulp_digest(record, parsed["payload_offsets"][FIELDS[0]])
    elif plant == "stamp":
        stamp_commit = f"{expected_commit}-plant"
    stamp = _stamp(record, stamp_commit, digest)
    return {
        "status": "AT_BAR", "producer_commit": expected_commit,
        "record": {key: value for key, value in parsed.items()
                   if key not in ("fields", "payload_offsets")},
        "parent_record": {"path": str(parent_record),
                          "sha256": parent["sha256"],
                          "header": parent_header},
        "endpoint_identity": {"u": True, "v": True},
        "stamp": stamp, "plant": plant, "preflight": preflight(),
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
            result = preflight()
        else:
            require(all((args.record, args.parent_record,
                         args.producer_commit, args.expect_commit)),
                    "record admission requires record, parent, and commit stamp")
            producer_commit = args.producer_commit.read_text().strip()
            require(producer_commit == args.expect_commit,
                    "producer commit file does not match expected commit")
            result = admit(args.record, args.parent_record,
                           args.expect_commit, args.plant)
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n")
        return 0
    except (GateError, OSError, ValueError, struct.error) as error:
        print(f"REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
