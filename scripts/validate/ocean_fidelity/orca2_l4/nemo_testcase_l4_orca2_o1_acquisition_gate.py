#!/usr/bin/env python3
"""Accept the WRITE-only ORCA2 O1 mapped-input/open-ocean-bulk record."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2b_exchange_gate as exchange,
)

MAGIC = "NEMO_L4_BLKIO_1"
RECORD = "oracle_sbcblk_o1_kt00000001.bin"
NX, NY, BITS = 90, 148, 64
HEADER_FMT = "=8i"
INPUT_FIELDS = (
    "wndi", "wndj", "tair", "humi", "qsr_down", "qlw_down",
    "precip_raw", "snow_raw", "slp",
)
OUTPUT_FIELDS = (
    "theta_air", "q_air", "precip", "sst", "ssu", "ssv", "tsk",
    "ssq", "cd_du", "sensible", "latent", "evap", "qlwn", "qsr",
    "qns", "emp", "utau", "vtau", "taum", "wndm",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_o1(path: Path, expected_sha256: str | None = None) -> dict:
    """Walk both frames; payload sizes derive only from their write lists."""
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kind, fields in enumerate((INPUT_FIELDS, OUTPUT_FIELDS)):
            raw_magic = handle.read(16)
            require(len(raw_magic) == 16, f"frame {kind}: truncated magic")
            require(raw_magic.decode("ascii").rstrip() == MAGIC,
                    f"frame {kind}: bad magic")
            raw_header = handle.read(struct.calcsize(HEADER_FMT))
            require(len(raw_header) == struct.calcsize(HEADER_FMT),
                    f"frame {kind}: truncated header")
            header = struct.unpack(HEADER_FMT, raw_header)
            expected = (1, 1, kind, NX, NY, len(fields), 0, BITS)
            require(header == expected, f"frame {kind}: header {header} != {expected}")
            count = NX * NY * len(fields)
            payload = np.fromfile(handle, dtype=np.float64, count=count)
            require(payload.size == count, f"frame {kind}: truncated payload")
            require(bool(np.isfinite(payload).all()),
                    f"frame {kind}: non-finite payload")
            rows.append({
                "kind": kind,
                "fields": list(fields),
                "header": list(header),
                "payload_f64": int(payload.size),
            })
        require(handle.read(1) == b"", "trailing payload")
    digest = exchange.sha256(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "O1 record SHA-256 mismatch")
    return {"frames": rows, "bytes": path.stat().st_size, "sha256": digest}


def planted_controls(path: Path) -> dict[str, str]:
    expected_digest = exchange.sha256(path)
    results = {}

    def expect(name: str, mutate, *, bind_digest: bool = False) -> None:
        with tempfile.TemporaryDirectory(prefix=f"orca2-o1-{name}-") as td:
            altered = Path(td) / RECORD
            shutil.copyfile(path, altered)
            mutate(altered)
            try:
                read_o1(altered, expected_digest if bind_digest else None)
            except (GateError, OSError, UnicodeError, struct.error, ValueError):
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"plant {name} did not fail")

    def patch(target: Path, offset: int, value: bytes) -> None:
        with target.open("r+b") as handle:
            handle.seek(offset)
            handle.write(value)

    expect("header_field_count", lambda p: patch(p, 16 + 5 * 4, struct.pack("=i", 8)))

    def one_ulp(target: Path) -> None:
        offset = 16 + struct.calcsize(HEADER_FMT)
        with target.open("r+b") as handle:
            handle.seek(offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))

    expect("one_ulp_payload", one_ulp, bind_digest=True)
    return results


def validate(
    run_dir: Path,
    control: Path,
    legacy_manifest: Path,
    surface_sha256: str,
    o1_sha256: str | None,
    plants: bool,
) -> dict:
    expected = exchange.phase1.expected_inventory() | {exchange.RECORD, RECORD}
    observed = {p.name for p in run_dir.glob("oracle_*.bin")}
    require(
        observed == expected,
        f"record inventory missing={sorted(expected-observed)} "
        f"extra={sorted(observed-expected)}",
    )
    result = {
        "status": "PASS",
        "oracle_label": "VARIANT",
        "legacy_records": exchange.validate_legacy_records(run_dir, legacy_manifest),
        "surface_input": exchange.validate_surface(
            run_dir / exchange.RECORD, surface_sha256, icebergs_off=True),
        "o1": read_o1(run_dir / RECORD, o1_sha256),
        "identity": exchange.phase1.validate_identity(control, run_dir),
    }
    if plants:
        result["o1_planted_controls"] = planted_controls(run_dir / RECORD)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--legacy-record-manifest", type=Path, required=True)
    parser.add_argument("--surface-sha256", required=True)
    parser.add_argument("--o1-sha256")
    parser.add_argument("--plant-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(
            args.run_dir, args.control, args.legacy_record_manifest,
            args.surface_sha256, args.o1_sha256, args.plant_controls,
        )
    except (GateError, exchange.GateError, OSError, UnicodeError,
            struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
        code = 1
    else:
        code = 0
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
