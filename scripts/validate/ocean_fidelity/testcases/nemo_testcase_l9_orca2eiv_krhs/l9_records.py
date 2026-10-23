#!/usr/bin/env python3
"""Strict combined reader for L9's L8 boundaries and L3 EIV witness."""

from __future__ import annotations

import argparse
import importlib.util
import struct
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASES = HERE.parent
L8_PATH = CASES / "nemo_testcase_l8_orca2krhs/l8_records.py"
L3_PATH = CASES / "nemo_testcase_l3_orca2eiv/l3_records.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


L8 = _module("l9_l8_records", L8_PATH)
L3 = _module("l9_l3_records", L3_PATH)


def check_layout(root: Path) -> None:
    L8.decode(root / L8.NAME, expected_kt=1)
    L3.read(root / "oracle_l3_eiv_kt00000001.bin")


def _write_python_layout(root: Path) -> None:
    jpi, jpj, jpk = 3, 2, 2
    count = jpi * jpj * jpk
    with (root / L8.NAME).open("wb") as handle:
        handle.write(L8.MAGIC)
        handle.write(struct.pack("=11i", 1, 1, 3, 2, 3, 4, 1,
                                 jpi, jpj, jpk, 64))
        for n in range(1, 23):
            handle.write(np.full(count, n, dtype="=f8").tobytes())
        for n in range(23, 26):
            handle.write(np.full(jpi * jpj, n, dtype="=f8").tobytes())

    nx, ny, nz = 8, 8, 2
    frames = (1.0, 2.0, 3.0, 5.0, 2.0, 3.0)
    with (root / "oracle_l3_eiv_kt00000001.bin").open("wb") as handle:
        handle.write(L3.MAGIC.encode("ascii"))
        handle.write(struct.pack("=9i", 1, 1, 2, 3, nx, ny, nz, 64, 6))
        for tag, value in zip(L3.TAGS, frames, strict=True):
            handle.write(tag.encode("ascii"))
            handle.write(np.full(nx * ny * nz, value, dtype="=f8").tobytes())


def selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_python_layout(root)
        check_layout(root)
        target = root / L8.NAME
        target.write_bytes(target.read_bytes()[:-8])
        try:
            check_layout(root)
        except ValueError:
            pass
        else:
            raise AssertionError("combined reader accepted truncated L8 frame")
    print("L9_RECORD_SELFTEST_OK combined_layout=true truncation_refused=true")


def check_acting(root: Path, steps: int, mesh: Path) -> None:
    L8.inventory(root, steps, acting=True)
    L3.check(root / "oracle_l3_eiv_kt00000001.bin", mesh)
    print(f"L9_ACTING_PASS steps=1..{steps}")


def inventory(root: Path, steps: int, mesh: Path) -> None:
    L8.inventory(root, steps, acting=False)
    record = L3.read(root / "oracle_l3_eiv_kt00000001.bin")
    shape = tuple(int(record[name]) for name in ("nx", "ny", "nz"))
    L3.validate_canonical(record, L3.defined_masks(mesh, shape))


def admit(roots: list[Path], steps: int, mesh: Path) -> None:
    L8.admit(roots, steps)
    left = roots[0] / "oracle_l3_eiv_kt00000001.bin"
    for root in roots[1:]:
        L3.compare_paths(left, root / left.name, mesh)
    print(f"L9_RECORDS_PASS steps=1..{steps} twins=3")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("selftest")
    layout = sub.add_parser("layout")
    layout.add_argument("root", type=Path)
    acting = sub.add_parser("acting")
    acting.add_argument("root", type=Path)
    acting.add_argument("steps", type=int)
    acting.add_argument("mesh", type=Path)
    one = sub.add_parser("inventory")
    one.add_argument("root", type=Path)
    one.add_argument("steps", type=int)
    one.add_argument("mesh", type=Path)
    admission = sub.add_parser("admit")
    admission.add_argument("steps", type=int)
    admission.add_argument("mesh", type=Path)
    admission.add_argument("roots", nargs=3, type=Path)
    args = parser.parse_args()
    if args.mode == "selftest":
        selftest()
    elif args.mode == "layout":
        check_layout(args.root)
    elif args.mode == "acting":
        check_acting(args.root, args.steps, args.mesh)
    elif args.mode == "inventory":
        inventory(args.root, args.steps, args.mesh)
    else:
        admit(args.roots, args.steps, args.mesh)


if __name__ == "__main__":
    main()
