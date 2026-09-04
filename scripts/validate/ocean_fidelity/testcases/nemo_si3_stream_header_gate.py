#!/usr/bin/env python3
"""Fail closed when an SI3 WRITE-only stream header disagrees with its payload.

The parser advances only by each record's declared/derived payload size.  A bad
count therefore exposes itself as a bad next magic or trailing/truncated bytes.
Counts derivable from dimensions are recomputed here rather than copied from a
case-specific 3+3-layer literal.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


class HeaderError(RuntimeError):
    pass


def _take(handle, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise HeaderError(f"{label}: wanted {size} bytes, got {len(data)}")
    return data


def _finish(handle, path: Path, records: int) -> dict[str, object]:
    if handle.read(1):
        raise HeaderError(f"{path}: trailing bytes")
    return {"path": str(path), "records": records, "status": "VALID"}


def _counted(path: Path, magic: bytes, nints: int, count_index: int,
             bits_index: int) -> dict[str, object]:
    records = 0
    with path.open("rb") as handle:
        while first := handle.read(16):
            if first != magic:
                raise HeaderError(f"{path}: bad magic at record {records}")
            header = struct.unpack(f"={nints}i", _take(handle, 4*nints, str(path)))
            nval, bits = header[count_index], header[bits_index]
            if nval < 0 or bits not in (32, 64):
                raise HeaderError(f"{path}: invalid count/width {nval}/{bits}")
            _take(handle, nval * (bits // 8), f"{path} payload {records}")
            records += 1
        return _finish(handle, path, records)


def _reassoc(path: Path) -> dict[str, object]:
    records = 0
    with path.open("rb") as handle:
        while first := handle.read(16):
            if first != b"NEMO_L3REA_001  ":
                raise HeaderError(f"{path}: bad magic at record {records}")
            h = struct.unpack("=8i", _take(handle, 32, str(path)))
            version, _step, _cat, npti, ni, ns, bits, claimed = h
            expected = (3 + 3*ni + 2*ns) * npti
            if version != 1 or bits != 64 or claimed != expected:
                raise HeaderError(
                    f"{path}: claimed {claimed}, derived {expected} from {h}")
            _take(handle, expected * 8, f"{path} payload {records}")
            records += 1
        return _finish(handle, path, records)


def _zdf(path: Path) -> dict[str, object]:
    records = 0
    with path.open("rb") as handle:
        while first := handle.read(16):
            if first != b"NEMO_L3ZDF_001  ":
                raise HeaderError(f"{path}: bad magic at record {records}")
            h = struct.unpack("=9i", _take(handle, 36, str(path)))
            version, _step, frame, _iteration, npti, ni, ns, bits, claimed = h
            expected_by_frame = {
                0: 10 + 3*ni + 2*ns,
                1: 5 + 3*ni + 2*ns,
                2: 2 + ni + ns,
                3: 4 * (ni + ns + 1),
                4: 2 * (ni + ns + 1),
                5: 3 + ni + ns,
                6: 4 + ni + ns,
            }
            expected = expected_by_frame.get(frame)
            if version != 1 or npti != 1 or bits != 64 or claimed != expected:
                raise HeaderError(
                    f"{path}: frame {frame} claimed {claimed}, derived {expected}")
            _take(handle, expected * 8, f"{path} payload {records}")
            records += 1
        return _finish(handle, path, records)


def _thd(path: Path) -> dict[str, object]:
    records = 0
    with path.open("rb") as handle:
        while first := handle.read(16):
            if first != b"NEMO_L3THD_001  ":
                raise HeaderError(f"{path}: bad magic at record {records}")
            h = struct.unpack("=11i", _take(handle, 44, str(path)))
            version, _step, _stage, payload, nx, ny, nc, ni, ns, npti, bits = h
            if version != 1 or bits != 64:
                raise HeaderError(f"{path}: invalid header {h}")
            if payload == 0:
                nval = nx*ny*nc*(9 + 2*ni + ns)
            elif payload == 1:
                nval = npti*(4 + 2*ni + ns)
            else:
                raise HeaderError(f"{path}: unknown payload selector {payload}")
            _take(handle, nval * 8, f"{path} payload {records}")
            records += 1
        return _finish(handle, path, records)


def _exchange(path: Path) -> dict[str, object]:
    records = 0
    with path.open("rb") as handle:
        while first := handle.read(16):
            if first != b"NEMO_L3XCHG_001 ":
                raise HeaderError(f"{path}: bad magic at record {records}")
            version, _step, nx, ny, nc, bits = struct.unpack(
                "=6i", _take(handle, 24, str(path)))
            if version != 1 or bits != 64:
                raise HeaderError(f"{path}: invalid header")
            n2 = nx*ny
            nr = max(1, nx-4)*max(1, ny-4)
            nr1 = max(1, nx-2)*max(1, ny-2)
            nval = 13*nr*nc + 13*nr + nr1 + 9*n2
            _take(handle, nval*8, f"{path} payload {records}")
            records += 1
        return _finish(handle, path, records)


READERS = {
    "oracle_si3_reassoc_operands.bin": _reassoc,
    "oracle_si3_zdf_operands.bin": _zdf,
    "oracle_si3_thd_frames.bin": _thd,
    "oracle_si3_exchange_frames.bin": _exchange,
    "oracle_si3_zdf_inputs.bin": lambda p: _counted(
        p, b"NEMO_L3ZIN_002  ", 6, 5, 4),
    "oracle_si3_dh_operands.bin": lambda p: _counted(
        p, b"NEMO_L3DHO_001  ", 6, 5, 4),
    "oracle_si3_dh_remap_operands.bin": lambda p: _counted(
        p, b"NEMO_L3DHR_001  ", 5, 4, 3),
    "oracle_si3_bulk_operands.bin": lambda p: _counted(
        p, b"NEMO_L3BULK_001 ", 5, 3, 4),
    "oracle_rung36_ssm_frames.bin": lambda p: _counted(
        p, b"NEMO_L3SSM__001 ", 7, 5, 6),
}


def validate_root(root: Path) -> dict[str, object]:
    rows = []
    for name, reader in READERS.items():
        path = root / name
        if path.exists():
            rows.append(reader(path))
    if not rows:
        raise HeaderError(f"{root}: no registered streams")
    return {"root": str(root), "streams": rows, "status": "VALID"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_root(args.root)
        code = 0
    except HeaderError as exc:
        result = {"root": str(args.root), "status": "INVALID", "error": str(exc)}
        code = 1
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
