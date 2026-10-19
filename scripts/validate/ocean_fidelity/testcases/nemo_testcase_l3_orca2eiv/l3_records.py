#!/usr/bin/env python3
"""Strict reader and synthetic layout self-test for the L3 EIV observer."""

from __future__ import annotations

import argparse
import struct
import tempfile
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L3_EIV_1   "
TAGS = ("BEFORE_U        ", "BEFORE_V        ", "AFTER_U         ",
        "AFTER_V         ", "DELTA_U         ", "DELTA_V         ")


def write_synthetic(path: Path) -> None:
    nx, ny, nz = 3, 4, 2
    values = np.arange(nx * ny * nz, dtype="=f8").reshape(
        (nx, ny, nz), order="F")
    frames = (values, values + 10.0, values + 2.0, values + 13.0,
              np.full_like(values, 2.0), np.full_like(values, 3.0))
    with path.open("wb") as handle:
        handle.write(MAGIC.encode("ascii"))
        handle.write(struct.pack("=9i", 1, 1, 2, 3, nx, ny, nz, 64, 6))
        for tag, frame in zip(TAGS, frames, strict=True):
            handle.write(tag.encode("ascii"))
            handle.write(np.asfortranarray(frame).tobytes(order="F"))


def read(path: Path) -> dict[str, np.ndarray | int]:
    with path.open("rb") as handle:
        magic_raw = handle.read(16)
        if len(magic_raw) != 16 or magic_raw.decode("ascii") != MAGIC:
            raise ValueError("wrong or short 16-byte magic")
        raw = handle.read(36)
        if len(raw) != 36:
            raise ValueError("short header")
        version, kt, kmm, krhs, nx, ny, nz, bits, frames = struct.unpack("=9i", raw)
        if (version, bits, frames) != (1, 64, 6) or min(nx, ny, nz) < 1:
            raise ValueError("unsupported header")
        out: dict[str, np.ndarray | int] = {"kt": kt, "Kmm": kmm, "Krhs": krhs}
        count = nx * ny * nz
        for expected in TAGS:
            tag_raw = handle.read(16)
            if len(tag_raw) != 16 or tag_raw.decode("ascii") != expected:
                raise ValueError(f"missing or wrong frame {expected!r}")
            payload = handle.read(8 * count)
            if len(payload) != 8 * count:
                raise ValueError(f"short frame {expected!r}")
            array = np.frombuffer(payload, dtype="=f8").reshape((nx, ny, nz), order="F")
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite frame {expected!r}")
            out[expected.strip().lower()] = array
        if handle.read(1):
            raise ValueError("extended payload")
    np.testing.assert_array_equal(out["after_u"] - out["before_u"], out["delta_u"])
    np.testing.assert_array_equal(out["after_v"] - out["before_v"], out["delta_v"])
    return out


def selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        good = Path(directory) / "good.bin"
        write_synthetic(good)
        assert read(good)["kt"] == 1
        data = good.read_bytes()
        for name, planted in (
            ("truncated", data[:-8]),
            ("bad_tag", data.replace(TAGS[-1].encode(), b"BROKEN_FRAME____", 1)),
        ):
            bad = Path(directory) / f"{name}.bin"
            bad.write_bytes(planted)
            try:
                read(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"reader accepted {name} plant")


def check(path: Path) -> None:
    record = read(path)
    moved = sum(int(np.count_nonzero(record[name])) for name in ("delta_u", "delta_v"))
    if moved == 0:
        raise ValueError("EIV observer recorded no changed transport value")
    print(f"L3_EIV_ACTING changed_transport_values={moved}")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    selftest() if args.selftest else check(args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
