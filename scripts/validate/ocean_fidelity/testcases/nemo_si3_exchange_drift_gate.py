#!/usr/bin/env python3
"""Account for C1D SI3 exchange-stream drift and enforce deterministic reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np

MAGIC = b"NEMO_L3XCHG_001 "
HEADER_BYTES = 40


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


class FieldSpec(NamedTuple):
    name: str
    size: str


# Exact order at config-local icestp.F90:257-264.  A2D(0), A2D(1), and full
# sizes are resolved from sbc_ice.F90:124-146 and sbc_oce.F90:184-216.
FIELD_SPECS = tuple(
    [FieldSpec(name, "reduced") for name in (
        "qns_ice", "qsr_ice", "qla_ice", "dqla_ice", "dqns_ice",
        "tn_ice", "alb_ice", "qml_ice", "qcn_ice", "qtr_ice_top",
    )]
    + [FieldSpec("utau_ice", "full"), FieldSpec("vtau_ice", "full")]
    + [FieldSpec(name, "reduced") for name in (
        "emp_ice", "evap_ice", "devap_ice", "qns_oce", "qsr_oce",
        "qemp_oce", "qemp_ice", "qevap_ice", "qprec_ice", "emp_oce",
        "wndm_ice", "sstfrz",
    )]
    + [FieldSpec("rCdU_ice", "halo1")]
    + [FieldSpec(name, "full") for name in (
        "snwice_mass", "snwice_mass_b", "snwice_fmass", "utau", "vtau",
    )]
    + [FieldSpec(name, "reduced") for name in ("taum", "qsr", "qns")]
    + [FieldSpec("emp", "full"), FieldSpec("sfx", "reduced"),
       FieldSpec("fr_i", "full")]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout(nx: int, ny: int, nc: int) -> tuple[list[tuple[FieldSpec, int, int]], int]:
    sizes = {
        "reduced": max(1, nx - 4) * max(1, ny - 4),
        "halo1": max(1, nx - 2) * max(1, ny - 2),
        "full": nx * ny,
    }
    layout: list[tuple[FieldSpec, int, int]] = []
    offset = HEADER_BYTES
    for spec in FIELD_SPECS:
        count = sizes[spec.size] * (nc if spec.name in {
            "qns_ice", "qsr_ice", "qla_ice", "dqla_ice", "dqns_ice",
            "tn_ice", "alb_ice", "qml_ice", "qcn_ice", "qtr_ice_top",
            "evap_ice", "devap_ice", "qevap_ice",
        } else 1)
        layout.append((spec, offset, count))
        offset += count * 8
    return layout, offset


def _read_header(record: memoryview, expected_step: int) -> tuple[int, int, int]:
    require(bytes(record[:16]) == MAGIC, f"bad magic at step {expected_step}")
    version, step, nx, ny, nc, bits = struct.unpack("=6i", record[16:HEADER_BYTES])
    require((version, step, bits) == (1, expected_step, 64),
            f"bad registry at step {expected_step}")
    return nx, ny, nc


def inspect_stream(path: Path) -> dict[str, object]:
    """Validate a stream and return its source-defined geometry and digest."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        first = memoryview(stream.read(HEADER_BYTES))
    nx, ny, nc = _read_header(first, 1)
    layout, record_bytes = _layout(nx, ny, nc)
    require(size % record_bytes == 0, f"truncated stream {path}")
    steps = size // record_bytes
    data = np.memmap(path, dtype=np.uint8, mode="r", shape=(steps, record_bytes))
    for index in range(steps):
        got = memoryview(data[index])
        _read_header(got, index + 1)
    return {
        "path": str(path), "sha256": sha256(path), "bytes": size,
        "steps": steps, "record_bytes": record_bytes,
        "geometry": {"nx": nx, "ny": ny, "nc": nc},
        "fields": [spec.name for spec, _, _ in layout],
    }


def compare_streams(reference: Path, candidate: Path) -> dict[str, object]:
    """Map every differing byte to its frame, field, and active/inactive class."""
    ref_info = inspect_stream(reference)
    candidate_info = inspect_stream(candidate)
    require(ref_info["bytes"] == candidate_info["bytes"], "stream size drift")
    require(ref_info["geometry"] == candidate_info["geometry"], "geometry drift")
    require(ref_info["steps"] == candidate_info["steps"], "step-count drift")
    geometry = ref_info["geometry"]
    nx, ny, nc = (int(geometry[key]) for key in ("nx", "ny", "nc"))
    layout, record_bytes = _layout(nx, ny, nc)
    steps = int(ref_info["steps"])
    ref = np.memmap(reference, dtype=np.uint8, mode="r", shape=(steps, record_bytes))
    got = np.memmap(candidate, dtype=np.uint8, mode="r", shape=(steps, record_bytes))
    changed = ref != got
    field_rows: list[dict[str, object]] = []
    accounted = 0
    active_differences = 0
    # C1D has one active interior cell. Full arrays store 5x5 with its active
    # member at Fortran-linear index 12; A2D(1) stores 3x3 with index 4.
    active_index = {"reduced": 0, "halo1": 4, "full": 12}
    for spec, start, count in layout:
        stop = start + count * 8
        byte_mask = changed[:, start:stop]
        if not np.any(byte_mask):
            continue
        value_mask = np.any(byte_mask.reshape(steps, count, 8), axis=2)
        indices = np.argwhere(value_mask)
        active = int(np.sum(indices[:, 1] == active_index[spec.size]))
        active_differences += active
        byte_count = int(np.sum(byte_mask))
        accounted += byte_count
        field_rows.append({
            "field": spec.name,
            "allocation": spec.size,
            "differing_bytes": byte_count,
            "differing_values": int(indices.shape[0]),
            "differing_frames": int(np.sum(np.any(value_mask, axis=1))),
            "element_indices": sorted(set(int(value) for value in indices[:, 1])),
            "first_step": int(indices[0, 0]) + 1,
            "first_element_index": int(indices[0, 1]),
            "active_differences": active,
            "classification": "inactive/uninitialized" if active == 0 else "real field",
        })
    header_differences = int(np.sum(changed[:, :HEADER_BYTES]))
    accounted += header_differences
    total = int(np.sum(changed))
    require(accounted == total, "unaccounted differing bytes")
    classification = (
        "identical" if total == 0 else
        "inactive/uninitialized" if active_differences == 0 and header_differences == 0 else
        "metadata" if total == header_differences else "real field"
    )
    return {
        "reference": ref_info, "candidate": candidate_info,
        "differing_bytes": total,
        "differing_frames": int(np.sum(np.any(changed, axis=1))),
        "header_differences": header_differences,
        "active_differences": active_differences,
        "accounted_differences": accounted,
        "classification": classification,
        "field_differences": field_rows,
    }


def require_identical(reference: Path, candidate: Path) -> dict[str, object]:
    report = compare_streams(reference, candidate)
    require(report["differing_bytes"] == 0, "exchange streams are not byte-identical")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparisons = []
    for candidate in args.candidates:
        comparison = (require_identical if args.require_identical else compare_streams)(
            args.reference, candidate
        )
        comparisons.append(comparison)
    result = {"bar": "byte-identical", "comparisons": comparisons}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
