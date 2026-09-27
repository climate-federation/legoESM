#!/usr/bin/env python3
"""Fail-closed schema/admission gate for the round-50 OVERFLOW kt=3 record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class GateError(RuntimeError):
    """The acquisition cannot be admitted."""


HEADER = struct.Struct("=14i")
FIELD_HEADER = struct.Struct("=4i")
EXPECTED_SHAPE = (202, 3, 100)
EXPECTED_ORIGIN = (3, 3)
SLOTS = {
    1: (1, 1, 3, 3),
    2: (1, 3, 2, 2),
    3: (1, 2, 3, 3),
}
COMMON = (
    "Kbb_T", "Kbb_S", "Kbb_ssh", "Kmm_T", "Kmm_S", "Kmm_ssh",
)
MOMENTUM_FIELDS = {
    1: COMMON + ("rhs_entry_u", "rhs_entry_v", "after_adv_u", "after_adv_v",
                 "pre_zdf_u", "pre_zdf_v", "raw_kaa_u", "raw_kaa_v",
                 "postbar_kaa_u", "postbar_kaa_v"),
    2: COMMON + ("rhs_entry_u", "rhs_entry_v", "rhd", "after_hpg_u",
                 "after_hpg_v", "after_vor_u", "after_vor_v", "after_adv_u",
                 "after_adv_v", "pre_zdf_u", "pre_zdf_v", "raw_kaa_u",
                 "raw_kaa_v", "postbar_kaa_u", "postbar_kaa_v"),
    3: COMMON + ("rhs_entry_u", "rhs_entry_v", "rhd", "after_hpg_u",
                 "after_hpg_v", "after_vor_u", "after_vor_v", "after_adv_u",
                 "after_adv_v", "after_ldf_u", "after_ldf_v", "pre_zdf_u",
                 "pre_zdf_v", "raw_kaa_u", "raw_kaa_v", "postbar_kaa_u",
                 "postbar_kaa_v"),
}
TRACER_FIELDS = {
    1: COMMON + ("rhs_entry_T", "rhs_entry_S", "after_adv_T", "after_adv_S",
                 "after_sbc_T", "after_sbc_S", "Kaa_T", "Kaa_S", "Kaa_ssh"),
    2: COMMON + ("rhs_entry_T", "rhs_entry_S", "after_adv_T", "after_adv_S",
                 "after_sbc_T", "after_sbc_S", "Kaa_T", "Kaa_S", "Kaa_ssh"),
    3: COMMON + ("rhs_entry_T", "rhs_entry_S", "after_adv_T", "after_adv_S",
                 "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
                 "after_ldf_T", "after_ldf_S", "pre_zdf_T", "pre_zdf_S",
                 "Kaa_T", "Kaa_S", "Kaa_ssh"),
}
KINDS = {
    "momentum": ("NEMO_L1_R50MOM1", MOMENTUM_FIELDS),
    "tracer": ("NEMO_L1_R50TRA1", TRACER_FIELDS),
}
HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE / "nemo_testcase_l1_overflow_round50_pair"
MODULE = INSTRUMENT / "l1_r50_pair.F90"
PATCH = INSTRUMENT / "stprk3_stg_round50.patch"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3/MY_SRC/stprk3_stg.F90"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(raw: bytes, offset: int, size: int, what: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {what}")
    return raw[offset:end], end


def read_record(path: Path, kind: str, stage: int) -> dict:
    """Parse one record through physical EOF and reject any schema drift."""
    require(kind in KINDS, f"unknown kind {kind}")
    require(stage in (1, 2, 3), f"bad expected stage {stage}")
    raw = path.read_bytes()
    chunk, offset = _take(raw, 0, 16, "magic")
    magic = chunk.decode("ascii").rstrip()
    expected_magic, schemas = KINDS[kind]
    require(magic == expected_magic, f"{path}: bad magic {magic!r}")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    header = HEADER.unpack(chunk)
    (version, kt, got_stage, kbb, kmm, krhs, kaa, nx, ny, nz,
     ntsi, ntsj, bits, nfields) = header
    require(version == 1, f"{path}: bad version {version}")
    require(kt == 3, f"{path}: bad kt {kt}")
    require(got_stage == stage, f"{path}: bad stage {got_stage}")
    require((kbb, kmm, krhs, kaa) == SLOTS[stage],
            f"{path}: bad slots {(kbb, kmm, krhs, kaa)}")
    require((nx, ny, nz) == EXPECTED_SHAPE,
            f"{path}: bad dimensions {(nx, ny, nz)}")
    require((ntsi, ntsj) == EXPECTED_ORIGIN,
            f"{path}: bad owned origin {(ntsi, ntsj)}")
    require(bits == 64, f"{path}: bad wp width {bits}")
    expected_fields = schemas[stage]
    require(nfields == len(expected_fields),
            f"{path}: field count {nfields} != {len(expected_fields)}")
    fields: dict[str, np.ndarray] = {}
    payload_offsets: dict[str, int] = {}
    for expected_name in expected_fields:
        chunk, offset = _take(raw, offset, 16, f"{expected_name} name")
        name = chunk.decode("ascii").rstrip()
        require(name == expected_name,
                f"{path}: expected {expected_name!r}, got {name!r}")
        require(name not in fields, f"{path}: duplicate field {name}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size,
                              f"{name} dimensions")
        ndim, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        expected_dims = (nx, ny, 1) if name.endswith("ssh") else (nx, ny, nz)
        expected_ndim = 2 if name.endswith("ssh") else 3
        require((ndim, dx, dy, dz) == (expected_ndim, *expected_dims),
                f"{path}: {name} bad shape header {(ndim, dx, dy, dz)}")
        count = math.prod(expected_dims)
        payload_offsets[name] = offset
        chunk, offset = _take(raw, offset, count * 8, f"{name} payload")
        values = np.frombuffer(chunk, dtype="=f8").reshape(expected_dims,
                                                              order="F")
        require(np.isfinite(values).all(), f"{path}: {name} is non-finite")
        fields[name] = values
    require(offset == len(raw), f"{path}: trailing bytes after physical EOF")
    require(tuple(fields) == expected_fields, f"{path}: missing/reordered fields")
    return {
        "path": str(path), "kind": kind, "stage": stage,
        "header": {
            "version": version, "kt": kt, "Kbb": kbb, "Kmm": kmm,
            "Krhs": krhs, "Kaa": kaa, "shape": [nx, ny, nz],
            "origin": [ntsi, ntsj], "bits": bits, "nfields": nfields,
        },
        "fields": fields, "payload_offsets": payload_offsets,
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
    """Prove the committed patch applies and only adds source lines."""
    for path in (MODULE, PATCH, SOURCE):
        require(path.is_file(), f"missing {path}")
    removed = [line for line in PATCH.read_text().splitlines()
               if line.startswith("-") and not line.startswith("---")]
    require(not removed, f"patch removes/replaces source lines: {removed[:1]}")
    with tempfile.TemporaryDirectory(prefix="orca2-r50-preflight-") as tmp:
        target = Path(tmp) / "stprk3_stg.F90"
        shutil.copyfile(SOURCE, target)
        result = subprocess.run(
            ["patch", "-s", str(target), str(PATCH)],
            text=True, capture_output=True, check=False,
        )
        require(result.returncode == 0,
                f"patch does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    sentinels = (
        "CALL r50_mom_begin", "CALL r50_mom_rhd", "'after_hpg'",
        "'after_vor'", "'after_adv'", "'after_ldf'", "'pre_zdf'",
        "'raw_kaa'", "CALL r50_mom_finish", "CALL r50_tra_begin",
        "'after_sbc'", "'after_qsr'", "CALL r50_tra_finish",
    )
    missing = [item for item in sentinels if item not in patched]
    require(not missing, f"patched source misses sentinels {missing}")
    return {
        "status": "PREFLIGHT_PASS",
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "removed_source_lines": len(removed),
        "sentinels": list(sentinels),
    }


def admit(record_dir: Path, expected_commit: str, plant: str | None) -> dict:
    records = []
    stamps = []
    for kind in ("momentum", "tracer"):
        for stage in (1, 2, 3):
            path = record_dir / f"oracle_r50_{kind}_kt00000003_s{stage}.bin"
            require(path.is_file(), f"missing {path}")
            record = read_record(path, kind, stage)
            digest = record["sha256"]
            if plant == "payload" and kind == "momentum" and stage == 1:
                first = next(iter(record["payload_offsets"].values()))
                digest = _one_ulp_digest(path, first)
            if plant == "stamp" and kind == "momentum" and stage == 1:
                stamp_commit = f"{expected_commit}-plant"
            else:
                stamp_commit = expected_commit
            stamps.append(_stamp(path, stamp_commit, digest))
            records.append({key: value for key, value in record.items()
                            if key not in ("fields", "payload_offsets")})
    return {
        "status": "AT_BAR", "producer_commit": expected_commit,
        "records": records, "stamps": stamps,
        "plant": plant, "preflight": preflight(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--producer-commit", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("payload", "stamp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            require(args.record_dir is None and args.producer_commit is None,
                    "preflight does not accept record arguments")
            report = preflight()
        else:
            require(args.record_dir is not None, "--record-dir is required")
            require(args.producer_commit is not None,
                    "--producer-commit is required")
            require(args.expect_commit is not None, "--expect-commit is required")
            produced = args.producer_commit.read_text().strip()
            require(produced == args.expect_commit,
                    "producer_commit.txt does not match --expect-commit")
            report = admit(args.record_dir, args.expect_commit, args.plant)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
