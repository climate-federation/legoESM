#!/usr/bin/env python3
"""Validate the Phase-2b final ORCA2 ocean surface-input record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)

NX, NY, NTR, NCLASSES, HALO = 94, 152, 2, 10, 2
NFULL, NREDUCED_2D, NHALO1, NREDUCED_3D = 20, 12, 1, 2
MAGIC = "NEMO_L4_SBCIN_1"
RECORD = "oracle_ocean_surface_input_kt00000001.bin"
HEADER = (
    1, 1, 1, NX, NY, NTR, NCLASSES, HALO,
    NFULL, NREDUCED_2D, NHALO1, NREDUCED_3D, 64,
)
HEADER_FMT = "=13i"
FIELDS = (
    ("utau", "full"), ("vtau", "full"), ("utauU", "full"),
    ("vtauV", "full"), ("utau_b", "full"), ("vtau_b", "full"),
    ("utau_icb", "full"), ("vtau_icb", "full"),
    ("taum", "reduced"), ("wndm", "reduced"),
    ("qsr", "reduced"), ("qns", "reduced"), ("qns_b", "reduced"),
    ("qsr_tot", "reduced"), ("qns_tot", "reduced"),
    ("emp", "full"), ("emp_b", "full"),
    ("sfx", "reduced"), ("sfx_b", "reduced"),
    ("emp_tot", "reduced"), ("fwfice", "reduced"),
    ("rnf", "full"), ("rnf_b", "full"), ("fwficb", "reduced"),
    ("fr_i", "full"), ("snwice_mass", "full"),
    ("snwice_mass_b", "full"), ("snwice_fmass", "full"),
    ("rCdU_ice", "halo1"), ("icb_calving", "full"),
    ("icb_calving_hflx", "full"), ("icb_floating_melt", "full"),
    ("icb_stored_heat", "full"),
    ("rnf_tsc", "reduced3d"), ("rnf_tsc_b", "reduced3d"),
)


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


def payload_count(header: tuple[int, ...]) -> int:
    (_, _, _, nx, ny, ntr, _, halo,
     nfull, nreduced2d, nhalo1, nreduced3d, bits) = header
    require(bits == 64, f"word size {bits} != 64")
    observed = {
        "full": sum(kind == "full" for _, kind in FIELDS),
        "reduced": sum(kind == "reduced" for _, kind in FIELDS),
        "halo1": sum(kind == "halo1" for _, kind in FIELDS),
        "reduced3d": sum(kind == "reduced3d" for _, kind in FIELDS),
    }
    require((nfull, nreduced2d, nhalo1, nreduced3d) ==
            (observed["full"], observed["reduced"], observed["halo1"], observed["reduced3d"]),
            "allocation-class counts do not match write list")
    reduced = (nx - 2 * halo) * (ny - 2 * halo)
    halo1 = (nx - 2 * (halo - 1)) * (ny - 2 * (halo - 1))
    return nfull * nx * ny + (nreduced2d + nreduced3d * ntr) * reduced + nhalo1 * halo1


def validate_surface(path: Path, expected_sha256: str | None = None) -> dict:
    with path.open("rb", buffering=0) as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated magic")
        require(raw_magic.decode("ascii").rstrip() == MAGIC, "bad magic")
        raw_header = handle.read(struct.calcsize(HEADER_FMT))
        require(len(raw_header) == struct.calcsize(HEADER_FMT), "truncated header")
    header = struct.unpack(HEADER_FMT, raw_header)
    require(header == HEADER, f"header {header} != {HEADER}")
    count = payload_count(header)
    offset = 16 + struct.calcsize(HEADER_FMT)
    expected_bytes = offset + count * 8
    require(path.stat().st_size == expected_bytes,
            f"size {path.stat().st_size} != derived schema {expected_bytes}")
    payload = np.memmap(path, dtype=np.float64, mode="r", offset=offset, shape=(count,))
    require(bool(np.isfinite(payload).all()), "non-finite payload")
    digest = sha256(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "record SHA-256 mismatch")

    cursor = 0
    fields = []
    class_sizes = {
        "full": NX * NY,
        "reduced": (NX - 2 * HALO) * (NY - 2 * HALO),
        "halo1": (NX - 2 * (HALO - 1)) * (NY - 2 * (HALO - 1)),
        "reduced3d": NTR * (NX - 2 * HALO) * (NY - 2 * HALO),
    }
    for name, allocation in FIELDS:
        size = class_sizes[allocation]
        values = payload[cursor:cursor + size]
        row = {"name": name, "allocation": allocation, "values": int(values.size),
               "finite": int(np.isfinite(values).sum())}
        if allocation == "reduced3d":
            row["levels"] = NTR
        fields.append(row)
        cursor += size
    require(cursor == count, f"schema walk consumed {cursor}/{count}")
    return {"magic": MAGIC, "header": list(header), "payload_f64": count,
            "bytes": expected_bytes, "sha256": digest, "fields": fields}


def validate_legacy_records(run_dir: Path) -> dict:
    """Run the frozen 90-stream Phase-1 parser despite this run's one extra file."""
    with tempfile.TemporaryDirectory(prefix="orca2-l4-phase2b-legacy-") as td:
        view = Path(td)
        for name in phase1.expected_inventory():
            os.symlink(run_dir / name, view / name)
        return phase1.validate_records(view)


def planted_controls(path: Path) -> dict[str, str]:
    expected_digest = sha256(path)
    results: dict[str, str] = {}

    def expect(name: str, mutate) -> None:
        with tempfile.TemporaryDirectory(prefix=f"orca2-l4-phase2b-{name}-") as td:
            altered = Path(td) / RECORD
            shutil.copyfile(path, altered)
            mutate(altered)
            try:
                validate_surface(altered, expected_digest)
            except (GateError, OSError, UnicodeError, struct.error, ValueError):
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"plant {name} did not fail")

    def patch(target: Path, offset: int, data: bytes) -> None:
        with target.open("r+b") as handle:
            handle.seek(offset)
            handle.write(data)

    expect("bad_magic", lambda target: patch(target, 0, b"X"))
    expect("bad_derived_count", lambda target: patch(target, 16 + 8 * 4, struct.pack("=i", 19)))
    expect("truncated_payload", lambda target: target.write_bytes(target.read_bytes()[:-8]))
    payload_offset = 16 + struct.calcsize(HEADER_FMT)
    expect("nan_payload", lambda target: patch(target, payload_offset, struct.pack("=d", float("nan"))))

    def one_ulp(target: Path) -> None:
        # First owned payload value; integrity pin binds an otherwise schema-valid mutation.
        with target.open("r+b") as handle:
            handle.seek(payload_offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(payload_offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))

    expect("one_ulp_active_field", one_ulp)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--expected-record-sha256")
    parser.add_argument("--plant-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = args.run_dir / RECORD
        result = {
            "surface_input": validate_surface(record, args.expected_record_sha256),
            "legacy_records": validate_legacy_records(args.run_dir),
        }
        if args.control is not None:
            result["identity"] = phase1.validate_identity(args.control, args.run_dir)
        if args.plant_controls:
            result["planted_controls"] = planted_controls(record)
        result["status"] = "PASS"
    except (GateError, phase1.GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
