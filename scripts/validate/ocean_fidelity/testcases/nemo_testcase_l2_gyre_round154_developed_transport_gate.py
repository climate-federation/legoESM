#!/usr/bin/env python3
"""Fail-closed admission for the developed stage-3 transport operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp

MAGIC = "NEMO_L2_R154TRP"
RECORD = "oracle_developed_transport_kt00001081.bin"
HEADER = (1, 1081, 3, 1, 2, 3, 36, 26, 31, 64, 20, 3, 34, 3, 24)
N2 = 36 * 26
N3 = N2 * 31
RECORD_BYTES = 16 + 15 * 4 + 8 * (12 * N2 + 8 * N3)
FIELDS_2D = (
    "e2u", "e1v", "r3u_Kmm", "r3v_Kmm", "zub", "zvb",
    "un_adv", "vn_adv", "r1_hu_0", "r1_hv_0", "uu_b_Kmm", "vv_b_Kmm",
)
FIELDS_3D = (
    "e3u_0", "e3v_0", "umask", "vmask", "uu_Kmm", "vv_Kmm", "zFu", "zFv",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path, *, truncate: bool = False) -> dict:
    payload = path.read_bytes()
    if truncate:
        payload = payload[:-8]
    require(len(payload) == RECORD_BYTES,
            f"{path}: {len(payload)} bytes, expected {RECORD_BYTES}")
    magic = payload[:16].decode("ascii").rstrip()
    header = struct.unpack_from("=15i", payload, 16)
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    require(header == HEADER, f"{path}: bad header {header}")
    offset = 16 + 15 * 4
    arrays = {}
    for name in FIELDS_2D:
        count = N2
        values = np.frombuffer(payload, dtype="=f8", count=count,
                               offset=offset).reshape((36, 26), order="F")
        arrays[name] = values
        offset += 8 * count
    for name in FIELDS_3D:
        count = N3
        values = np.frombuffer(payload, dtype="=f8", count=count,
                               offset=offset).reshape((36, 26, 31), order="F")
        arrays[name] = values
        offset += 8 * count
    require(offset == len(payload), "record reader left trailing bytes")
    require(all(np.all(np.isfinite(value)) for value in arrays.values()),
            "record contains a non-finite operand")
    for name in ("umask", "vmask"):
        require(np.all((arrays[name] == 0.0) | (arrays[name] == 1.0)),
                f"{name} is not binary")
    return {"header": header, "fields": arrays, "sha256": sha256(path)}


def _state_files(root: Path) -> tuple[str, ...]:
    names = {path.name for path in root.glob("*_restart.nc")}
    require(bool(names), f"{root}: no restart files")
    require((root / "mesh_mask.nc").is_file(), f"{root}: no mesh_mask.nc")
    names.add("mesh_mask.nc")
    return tuple(sorted(names))


def _check_state(root: Path, baseline: Path, *, plant=False) -> list[str]:
    names = _state_files(baseline)
    require(_state_files(root) == names, "state-file registry differs")
    for index, name in enumerate(names):
        got = sha256(root / name)
        if plant and index == 0:
            got = "0" * 64
        require(got == sha256(baseline / name),
                f"passive writer moved state file {name}")
    return list(names)


def _check_stream_manifest(root: Path) -> dict:
    manifest = root / "round154_streams.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    expected = sorted(
        path.name for path in root.iterdir()
        if path.is_file() and (
            path.name.startswith("oracle") and path.suffix == ".bin"
            or path.name.endswith("_restart.nc")
            or path.name == "mesh_mask.nc"))
    require(sorted(entries) == expected,
            "self-consistent stream manifest registry differs from run")
    for name in expected:
        require(entries[name] == sha256(root / name),
                f"stream manifest digest differs for {name}")
    return {"count": len(expected), "sha256": sha256(manifest),
            "names": expected}


def _effect_control(record: dict, *, plant=False) -> dict:
    f = record["fields"]
    u = np.array(f["uu_Kmm"], copy=True)
    index = tuple(int(value) for value in np.argwhere(f["umask"] == 1.0)[0])
    if plant:
        u[index] = np.nextafter(u[index], np.inf)
    zub = f["un_adv"] * (f["r1_hu_0"] / (1.0 + f["r3u_Kmm"])) - f["uu_b_Kmm"]
    reconstructed = (f["e2u"][..., None] * f["e3u_0"]
                     * (1.0 + f["r3u_Kmm"][..., None] * f["umask"])
                     * (u + zub[..., None] * f["umask"]))
    original = (f["e2u"][..., None] * f["e3u_0"]
                * (1.0 + f["r3u_Kmm"][..., None] * f["umask"])
                * (f["uu_Kmm"] + zub[..., None] * f["umask"]))
    moved = int(np.count_nonzero(
        reconstructed.view(np.uint64) != original.view(np.uint64)))
    if plant:
        require(moved > 0, "velocity ULP plant moved no reconstructed zFu")
        raise GateError(
            f"velocity ULP moved reconstructed zFu in {moved} cells")
    return {"velocity_index": list(index), "reconstructed_zFu_cells_moved": moved}


def measure(args) -> dict:
    producer = (args.root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_commit, "producer commit mismatch")
    stamp = (args.root / f"{RECORD}.stamp").read_text().split()
    require(len(stamp) == 3, "malformed record stamp")
    expected_stamp = "0" * 64 if args.plant == "stamp" else sha256(args.root / RECORD)
    require(stamp == [expected_stamp, producer, RECORD], "record stamp mismatch")
    record = read_record(args.root / RECORD,
                         truncate=args.plant == "truncation")
    states = _check_state(args.root, args.baseline,
                          plant=args.plant == "restart-byte")
    streams = _check_stream_manifest(args.root)
    effect = _effect_control(record, plant=args.plant == "operand-ulp")
    return {
        "format": "nemo-testcase-l2-gyre-round154-developed-transport-v1",
        "status": "PASS", "producer_commit": producer,
        "worktree": worktree_stamp(),
        "record": {"name": RECORD, "sha256": record["sha256"],
                   "bytes": RECORD_BYTES, "fields": list(record["fields"])},
        "state_files_bit_identical": states,
        "self_consistent_streams": streams,
        "effect_control": effect,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("none", "stamp", "truncation",
                                             "restart-byte", "operand-ulp"),
                        default="none")
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    except (GateError, OSError, ValueError, struct.error) as error:
        marker = "STATUS PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"{marker}: {args.plant}: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"REFUSE: plant {args.plant} stayed green", file=sys.stderr)
        return 2
    print("STATUS PASS: developed transport record fields=20 state=BIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
