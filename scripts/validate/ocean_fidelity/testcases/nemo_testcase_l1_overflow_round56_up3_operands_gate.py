#!/usr/bin/env python3
"""Fail-closed schema/admission gate for the round-56 UP3 operand record."""

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
from nemo_testcase_l1_overflow_round50_pair_gate import (
    GateError,
    require,
)
from nemo_testcase_l1_overflow_round50_pair_gate import (
    read_record as read_parent,
)

HEADER = struct.Struct("=13i")
FIELD_HEADER = struct.Struct("=4i")
MAGIC = "NEMO_L1_R56UP301"
FIELDS = (
    "transport_u",
    "transport_v",
    "transport_w",
    "kbb_u",
    "kbb_v",
    "kmm_u",
    "kmm_v",
    "rhs_entry_u",
    "rhs_entry_v",
    "curv_uu",
    "curv_vv",
    "curv_uv",
    "curv_vu",
    "pair_u_t",
    "pair_v_t",
    "selected_u_t",
    "selected_v_t",
    "pair_u_f",
    "pair_v_f",
    "selected_u_f",
    "selected_v_f",
    "flux_u_t",
    "flux_v_t",
    "flux_u_f",
    "flux_v_f",
    "hscale_u",
    "hscale_v",
    "h_rhs_before_u",
    "h_rhs_before_v",
    "h_rhs_after_u",
    "h_rhs_after_v",
    "v_curv_u",
    "v_curv_v",
    "v_transport_u",
    "v_transport_v",
    "v_selected_u",
    "v_selected_v",
    "v_flux_prev_u",
    "v_flux_prev_v",
    "v_flux_new_u",
    "v_flux_new_v",
    "v_scale_u",
    "v_scale_v",
    "v_rhs_before_u",
    "v_rhs_before_v",
    "v_rhs_after_u",
    "v_rhs_after_v",
)
HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE / "nemo_testcase_l1_overflow_round56_up3_operands"
MODULE = INSTRUMENT / "l1_r56_up3.F90"
PATCH = INSTRUMENT / "dynadv_up3_round56.patch"
SOURCE = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/DYN/dynadv_up3.F90")


def _take(raw: bytes, offset: int, size: int, what: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {what}")
    return raw[offset:end], end


def read_up3_record(path: Path) -> dict:
    """Parse dimensions and payload lengths from the record through physical EOF."""
    raw = path.read_bytes()
    chunk, offset = _take(raw, 0, 16, "magic")
    magic = chunk.decode("ascii").rstrip()
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    header = HEADER.unpack(chunk)
    (version, kt, stage, kbb, kmm, krhs, nx, ny, nz, origin_i, origin_j, bits, nfields) = header
    require(version == 1, f"{path}: bad version {version}")
    require(
        (kt, stage, kbb, kmm, krhs) == (3, 2, 3, 3, 2),
        f"{path}: bad step/stage/slots {(kt, stage, kbb, kmm, krhs)}",
    )
    require(nx > 0 and ny > 0 and nz > 0, f"{path}: non-positive dimensions {(nx, ny, nz)}")
    require(bits == 64, f"{path}: bad wp width {bits}")
    require(nfields == len(FIELDS), f"{path}: field count {nfields} != {len(FIELDS)}")
    arrays = {}
    payload_offsets = {}
    for expected_name in FIELDS:
        chunk, offset = _take(raw, offset, 16, f"{expected_name} name")
        name = chunk.decode("ascii").rstrip()
        require(name == expected_name, f"{path}: expected {expected_name!r}, got {name!r}")
        require(name not in arrays, f"{path}: duplicate field {name}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size, f"{name} shape header")
        rank, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        require(rank == 3, f"{path}: {name} rank {rank} != 3")
        require(
            (dx, dy, dz) == (nx, ny, nz),
            f"{path}: {name} shape {(dx, dy, dz)} != header {(nx, ny, nz)}",
        )
        count = math.prod((dx, dy, dz))
        payload_offsets[name] = offset
        chunk, offset = _take(raw, offset, count * (bits // 8), f"{name} payload")
        value = np.frombuffer(chunk, dtype="=f8").reshape((dx, dy, dz), order="F")
        require(np.isfinite(value).all(), f"{path}: {name} is non-finite")
        arrays[name] = value
    require(offset == len(raw), f"{path}: trailing bytes after physical EOF")
    return {
        "path": str(path),
        "header": {
            "version": version,
            "kt": kt,
            "stage": stage,
            "Kbb": kbb,
            "Kmm": kmm,
            "Krhs": krhs,
            "shape": [nx, ny, nz],
            "origin": [origin_i, origin_j],
            "bits": bits,
            "nfields": nfields,
        },
        "fields": arrays,
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
    require(
        planted != value and np.isfinite(planted), "payload plant could not move finite fp64 value"
    )
    struct.pack_into("=d", raw, payload_offset, planted)
    return hashlib.sha256(raw).hexdigest()


def preflight() -> dict:
    for path in (MODULE, PATCH, SOURCE):
        require(path.is_file(), f"missing {path}")
    removed = [
        line
        for line in PATCH.read_text().splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    require(not removed, f"patch removes/replaces source lines: {removed[:1]}")
    with tempfile.TemporaryDirectory(prefix="orca2-r56-preflight-") as tmp:
        target = Path(tmp) / "dynadv_up3.F90"
        shutil.copyfile(SOURCE, target)
        result = subprocess.run(
            ["patch", "-s", str(target), str(PATCH)],
            text=True,
            capture_output=True,
            check=False,
        )
        require(result.returncode == 0, f"patch does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    sentinels = (
        "CALL r56_up3_begin",
        "CALL r56_up3_curv",
        "CALL r56_up3_select_t",
        "CALL r56_up3_select_f",
        "CALL r56_up3_hflux",
        "CALL r56_up3_hrhs_before",
        "CALL r56_up3_hrhs_after",
        "CALL r56_up3_vrhs_before",
        "CALL r56_up3_vrhs_after",
        "CALL r56_up3_bottom_before",
        "CALL r56_up3_finish",
    )
    missing = [sentinel for sentinel in sentinels if sentinel not in patched]
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


def _bit_unequal(a: np.ndarray, b: np.ndarray) -> int:
    require(a.shape == b.shape, f"endpoint shape drift {a.shape} != {b.shape}")
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def admit(record: Path, parent_record: Path, expected_commit: str, plant: str | None) -> dict:
    report = read_up3_record(record)
    parent = read_parent(parent_record, "momentum", 2)
    nx, ny, nz = report["header"]["shape"]
    px, py, pz = parent["header"]["shape"]
    origin_i, origin_j = report["header"]["origin"]
    require(nz == pz, "UP3/parent vertical shape drift")
    require(origin_i >= 1 and origin_j >= 1, "UP3 owned origin is not one-based positive")
    require(
        origin_i - 1 + px <= nx and origin_j - 1 + py <= ny,
        "UP3 full field does not contain the parent owned block",
    )
    digest = report["sha256"]
    stamp_commit = expected_commit
    if plant == "payload":
        digest = _one_ulp_digest(record, report["payload_offsets"]["transport_u"])
    elif plant == "stamp":
        stamp_commit = f"{expected_commit}-plant"
    stamp = _stamp(record, stamp_commit, digest)
    fields = report["fields"]
    parent_fields = parent["fields"]

    def owned(value: np.ndarray) -> np.ndarray:
        return value[
            origin_i - 1 : origin_i - 1 + px,
            origin_j - 1 : origin_j - 1 + py,
            :pz,
        ]

    endpoints = {
        "entry_u_unequal": _bit_unequal(owned(fields["rhs_entry_u"]), parent_fields["after_vor_u"]),
        "entry_v_unequal": _bit_unequal(owned(fields["rhs_entry_v"]), parent_fields["after_vor_v"]),
        "final_u_unequal": _bit_unequal(
            owned(fields["v_rhs_after_u"]), parent_fields["after_adv_u"]
        ),
        "final_v_unequal": _bit_unequal(
            owned(fields["v_rhs_after_v"]), parent_fields["after_adv_v"]
        ),
    }
    require(not any(endpoints.values()), f"UP3 endpoint mismatch: {endpoints}")
    return {
        "status": "AT_BAR",
        "record": {
            key: value for key, value in report.items() if key not in ("fields", "payload_offsets")
        },
        "parent_sha256": parent["sha256"],
        "stamp": stamp,
        "endpoints": endpoints,
        "plant": plant,
        "preflight": preflight(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--parent-record", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("payload", "stamp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            require(
                args.record is None and args.parent_record is None,
                "preflight accepts no record arguments",
            )
            report = preflight()
        else:
            require(args.record is not None, "--record is required")
            require(args.parent_record is not None, "--parent-record is required")
            require(args.expect_commit is not None, "--expect-commit is required")
            report = admit(args.record, args.parent_record, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant else 0
    except (GateError, OSError, ValueError, KeyError, struct.error) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
