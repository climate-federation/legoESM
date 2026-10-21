#!/usr/bin/env python3
"""Read and police the L7 river-mouth before/after record."""

from __future__ import annotations

import argparse
import struct
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L7_RNFKZ_1 "
TAGS = (b"CLOSURE_AVT     ", b"RIVER_AVT       ", b"EVD_DDM_AVT     ",
        b"IWM_AVT         ")


def read(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        magic = stream.read(16)
        header = stream.read(28)
        if len(header) != 28:
            raise ValueError("short L7 header")
        version, kt, nx, ny, nz, bits, nfields = struct.unpack("=7i", header)
        if magic != MAGIC or (version, kt, bits, nfields) != (1, 1, 64, 4):
            raise ValueError("wrong L7 record identity")
        if min(nx, ny, nz) <= 0:
            raise ValueError("nonpositive L7 shape")
        out: dict[str, np.ndarray] = {}
        count = nx * ny * nz
        for expected in TAGS:
            tag = stream.read(16)
            raw = stream.read(count * 8)
            if tag != expected or len(raw) != count * 8:
                raise ValueError("wrong tag or short L7 field")
            values = np.frombuffer(raw, dtype=np.float64).reshape(
                (nx, ny, nz), order="F").copy()
            if not np.isfinite(values).all():
                raise ValueError("nonfinite L7 observer value")
            out[tag.decode().strip()] = values
        if stream.read(1):
            raise ValueError("trailing bytes in L7 record")
    return out


def write_synthetic(path: Path, *, acting: bool) -> None:
    shape = (4, 3, 5)
    before = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F")
    after = before.copy()
    if acting:
        after[1, 1, 1] += 2.0e-3
    fields = (before, after, after + 1.0e-4, after + 2.0e-4)
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("=7i", 1, 1, *shape, 64, 4))
        for tag, field in zip(TAGS, fields, strict=True):
            stream.write(tag)
            stream.write(np.asarray(field, order="F").tobytes(order="F"))


def check_acting(path: Path) -> None:
    record = read(path)
    if np.count_nonzero(record["RIVER_AVT"] - record["CLOSURE_AVT"]) == 0:
        raise ValueError("river-mouth statement did not act on any defined cell")


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.bin"
        inert = Path(tmp) / "inert.bin"
        write_synthetic(good, acting=True)
        write_synthetic(inert, acting=False)
        check_acting(good)
        try:
            check_acting(inert)
        except ValueError:
            pass
        else:
            raise AssertionError("inert acting-control was accepted")
        raw = good.read_bytes()
        short = Path(tmp) / "short.bin"
        short.write_bytes(raw[:-8])
        try:
            read(short)
        except ValueError:
            pass
        else:
            raise AssertionError("short record was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--check-acting", type=Path)
    parser.add_argument("--compare-defined", nargs=2, type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.check:
        read(args.check)
    elif args.check_acting:
        check_acting(args.check_acting)
    elif args.compare_defined:
        left, right = map(read, args.compare_defined)
        if any(not np.array_equal(left[key], right[key]) for key in left):
            raise SystemExit("defined L7 twins differ")
    else:
        parser.error("choose one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
