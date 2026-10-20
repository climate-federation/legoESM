#!/usr/bin/env python3
"""Strict reader and synthetic layout self-test for the L3 EIV observer."""

from __future__ import annotations

import argparse
import struct
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

MAGIC = "NEMO_L3_EIV_1   "
TAGS = ("BEFORE_U        ", "BEFORE_V        ", "AFTER_U         ",
        "AFTER_V         ", "DELTA_U         ", "DELTA_V         ")
HALO = 2


def _synthetic_masks() -> dict[str, np.ndarray]:
    masks = {
        "U": np.zeros((8, 8, 2), dtype=bool),
        "V": np.zeros((8, 8, 2), dtype=bool),
    }
    masks["U"][HALO:-HALO, HALO:-HALO] = True
    masks["V"][HALO:-HALO, HALO:-HALO] = True
    masks["U"][3, 3, 1] = False
    masks["V"][4, 4, 1] = False
    return masks


def _frame_grid(name: str) -> str:
    return "U" if name.strip().endswith("_U") else "V"


def _canonical(values: np.ndarray, defined: np.ndarray) -> np.ndarray:
    """Mirror the observer's zero-first view without reading excluded cells."""

    out = np.zeros_like(values)
    out[defined] = values[defined]
    return out


def write_synthetic(
    path: Path,
    *,
    garbage_at: tuple[int, int, int] | None = None,
) -> None:
    nx, ny, nz = 8, 8, 2
    values = np.arange(nx * ny * nz, dtype="=f8").reshape(
        (nx, ny, nz), order="F")
    raw = [values, values + 10.0, values + 2.0, values + 13.0,
           np.full_like(values, 2.0), np.full_like(values, 3.0)]
    if garbage_at is not None:
        raw[0] = raw[0].copy()
        raw[2] = raw[2].copy()
        raw[0][garbage_at] = 98765.125
        raw[2][garbage_at] = 98767.125
    masks = _synthetic_masks()
    frames = tuple(
        _canonical(frame, masks[_frame_grid(tag)])
        for tag, frame in zip(TAGS, raw, strict=True)
    )
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
        out: dict[str, np.ndarray | int] = {
            "kt": kt, "Kmm": kmm, "Krhs": krhs,
            "nx": nx, "ny": ny, "nz": nz,
        }
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


def defined_masks(mesh: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Map ORCA2's halo-free U/V masks onto the observer's rank-local arrays."""

    nx, ny, nz = shape
    out: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(mesh) as dataset:
        for grid, name in (("U", "umask"), ("V", "vmask")):
            physical = np.ma.filled(dataset.variables[name][:], 0)
            physical = np.squeeze(np.asarray(physical)).astype(bool)
            if physical.shape != (nz, ny - 2 * HALO, nx - 2 * HALO):
                raise ValueError(
                    f"{mesh}: {name} shape {physical.shape} does not match "
                    f"record {(nx, ny, nz)} with halo {HALO}")
            mask = np.zeros(shape, dtype=bool)
            mask[HALO:-HALO, HALO:-HALO] = physical.transpose(2, 1, 0)
            out[grid] = mask
    return out


def validate_canonical(
    record: dict[str, np.ndarray | int],
    masks: dict[str, np.ndarray],
) -> dict[str, int]:
    """Require +0 in every cell outside the scorer's owned wet domain."""

    counts: dict[str, int] = {}
    for tag in TAGS:
        name = tag.strip().lower()
        values = np.asarray(record[name])
        mask = masks[_frame_grid(tag)]
        if values.shape != mask.shape:
            raise ValueError(f"{name}: shape {values.shape} != mask {mask.shape}")
        excluded_bits = np.ascontiguousarray(values[~mask]).view(np.uint64)
        noncanonical = int(np.count_nonzero(excluded_bits))
        if noncanonical:
            raise ValueError(
                f"{name}: {noncanonical} excluded cells are not canonical +0")
        counts[name] = int(mask.sum())
    return counts


def compare_defined(
    left: dict[str, np.ndarray | int],
    right: dict[str, np.ndarray | int],
    masks: dict[str, np.ndarray],
) -> dict[str, int]:
    """Byte-compare twins only on the same owned wet U/V domain as the model."""

    validate_canonical(left, masks)
    validate_canonical(right, masks)
    for name in ("kt", "Kmm", "Krhs", "nx", "ny", "nz"):
        if left[name] != right[name]:
            raise ValueError(f"header {name} differs: {left[name]} != {right[name]}")
    result: dict[str, int] = {}
    for tag in TAGS:
        name = tag.strip().lower()
        mask = masks[_frame_grid(tag)]
        a = np.ascontiguousarray(np.asarray(left[name])[mask]).view(np.uint64)
        b = np.ascontiguousarray(np.asarray(right[name])[mask]).view(np.uint64)
        unequal = int(np.count_nonzero(a != b))
        if unequal:
            raise ValueError(f"{name}: twins differ in {unequal} defined cells")
        result[name] = int(a.size)
    return result


def selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        good = Path(directory) / "good.bin"
        write_synthetic(good)
        good_record = read(good)
        assert good_record["kt"] == 1
        masks = _synthetic_masks()
        validate_canonical(good_record, masks)
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

        # The observer's zero-first view erases garbage in an excluded cell.
        undefined = Path(directory) / "undefined-garbage.bin"
        write_synthetic(undefined, garbage_at=(0, 0, 0))
        if undefined.read_bytes() != data:
            raise AssertionError("excluded-cell garbage escaped canonicalization")
        compare_defined(good_record, read(undefined), masks)

        # A wet owned cell is binding: the same garbage must remain visible.
        defined = Path(directory) / "defined-garbage.bin"
        write_synthetic(defined, garbage_at=(2, 2, 0))
        try:
            compare_defined(good_record, read(defined), masks)
        except ValueError:
            pass
        else:
            raise AssertionError("defined-cell garbage did not separate twins")
    print("L3_RECORD_SELFTEST_OK undefined_garbage=canonicalized "
          "defined_garbage=detected")


def check(path: Path, mesh: Path) -> None:
    record = read(path)
    shape = tuple(int(record[name]) for name in ("nx", "ny", "nz"))
    masks = defined_masks(mesh, shape)
    validate_canonical(record, masks)
    moved = sum(int(np.count_nonzero(
        np.asarray(record[name])[masks[name[-1].upper()]]))
        for name in ("delta_u", "delta_v"))
    if moved == 0:
        raise ValueError("EIV observer recorded no changed transport value")
    print(f"L3_EIV_ACTING changed_transport_values={moved}")


def compare_paths(left: Path, right: Path, mesh: Path) -> None:
    a, b = read(left), read(right)
    shape = tuple(int(a[name]) for name in ("nx", "ny", "nz"))
    if shape != tuple(int(b[name]) for name in ("nx", "ny", "nz")):
        raise ValueError("twin record extents differ")
    counts = compare_defined(a, b, defined_masks(mesh, shape))
    print("L3_DEFINED_TWINS_EQUAL " + " ".join(
        f"{name}={count}" for name, count in counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--check", type=Path)
    group.add_argument("--compare-defined", nargs=2, type=Path,
                       metavar=("LEFT", "RIGHT"))
    parser.add_argument("--mesh", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.mesh is None:
        parser.error("--mesh is required with --check/--compare-defined")
    elif args.check:
        check(args.check, args.mesh)
    else:
        compare_paths(*args.compare_defined, args.mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
