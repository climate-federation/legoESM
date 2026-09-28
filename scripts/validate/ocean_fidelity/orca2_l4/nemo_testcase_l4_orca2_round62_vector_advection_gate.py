#!/usr/bin/env python3
"""Admit ORCA2's existing stage-2 vector-advection boundary.

This gate deliberately does not infer either child operator from the combined
``dyn_keg`` + ``dyn_zad`` endpoint.  It proves the card branch and compiled
call order, measures the admitted rank-0 boundary, and reports the exact
missing operands which require the round-62 split acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import xarray as xr

from legoesm.ocean.fidelity.provenance import worktree_stamp


ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
RECORD = "oracle_rkstage2_terms_kt00000001.bin"
EXPECTED_SHA256 = "18a7b4701f167c5fd17f93875c318eb21ea30ce79c523b269ac6d8abd2ebd35d"
MAGIC = "NEMO_L2_RKTRM_1"
HEADER = (1, 1, 2, 3, 2, 94, 152, 31, 64)
NAMES = (
    "before_u", "before_v", "after_hpg_u", "after_hpg_v",
    "after_vorticity_u", "after_vorticity_v",
    "after_advection_u", "after_advection_v",
)
PPROOT = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
)


class GateError(RuntimeError):
    """Fail-closed measurement error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(raw: np.ndarray) -> np.ndarray:
    return raw.reshape((94, 152, 31), order="F").transpose(1, 0, 2).copy()


def read_terms(path: Path, *, plant: str | None = None) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing record: {path}")
    digest = sha256(path)
    if plant == "digest":
        digest = "0" * 64
    require(digest == EXPECTED_SHA256, f"unregistered record digest: {digest}")
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(struct.calcsize("=9i"))
        require(len(raw_header) == struct.calcsize("=9i"), "truncated header")
        header = struct.unpack("=9i", raw_header)
        values = np.fromfile(handle, dtype=np.float64)
        require(handle.read(1) == b"", "trailing payload")
    require(magic == MAGIC, f"bad magic: {magic!r}")
    require(header == HEADER, f"bad header: {header}")
    count = 94 * 152 * 31
    require(values.size == len(NAMES) * count,
            f"payload count {values.size} != {len(NAMES) * count}")
    require(bool(np.isfinite(values).all()), "non-finite payload")
    names = list(NAMES)
    if plant == "field-order":
        names[0], names[1] = names[1], names[0]
    require(tuple(names) == NAMES, f"field order changed: {names}")
    arrays = {
        name: _xyz(values[i * count:(i + 1) * count])
        for i, name in enumerate(names)
    }
    if plant == "payload-ulp":
        arrays["after_advection_u"][2, 2, 0] = np.nextafter(
            arrays["after_advection_u"][2, 2, 0], np.inf
        )
    return arrays


def _compiled_scope() -> dict[str, object]:
    sources = {
        "dispatcher": PPROOT / "dynadv.f90",
        "keg": PPROOT / "dynkeg.f90",
        "zad": PPROOT / "dynzad.f90",
        "caller": PPROOT / "stprk3_stg.f90",
    }
    for path in sources.values():
        require(path.is_file(), f"compiled source missing: {path}")
    dispatcher = sources["dispatcher"].read_text()
    caller = sources["caller"].read_text()
    require(dispatcher.index("CALL dyn_keg") < dispatcher.index("CALL dyn_zad"),
            "compiled vector call order is not KEG then ZAD")
    require("IF( ln_dynadv_vec ) THEN" in caller,
            "compiled stage caller lacks vector arm")
    require("CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)" in caller,
            "compiled stage caller lacks vector dyn_adv call")
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in sources.items()
    }


def _movement(arrays: dict[str, np.ndarray], root: Path) -> list[dict[str, object]]:
    with xr.open_dataset(root / "mesh_mask_0000.nc", decode_cf=False) as mesh:
        masks = {
            face: np.asarray(mesh[f"{face.lower()}mask"][0, :30]).transpose(1, 2, 0) > 0
            for face in ("U", "V")
        }
    rows = []
    for face in ("U", "V"):
        key = face.lower()
        before = arrays[f"after_vorticity_{key}"][2:-2, 2:-2, :30]
        after = arrays[f"after_advection_{key}"][2:-2, 2:-2, :30]
        mask = masks[face]
        require(before.shape == after.shape == mask.shape,
                f"{face}: record/mask shape mismatch")
        unequal = before.view(np.uint64) != after.view(np.uint64)
        delta = np.abs(after - before)
        active_delta = delta[mask]
        row = {
            "face": face,
            "active_wet_cells": int(mask.sum()),
            "bit_unequal_active_wet": int(np.count_nonzero(unequal[mask])),
            "maximum_absolute_change": float(np.max(active_delta)),
            "rms_change": float(np.sqrt(np.mean(active_delta * active_delta))),
        }
        rows.append(row)
    require(any(row["bit_unequal_active_wet"] for row in rows),
            "vector advection made no active-wet movement")
    require(not (rows[0]["bit_unequal_active_wet"] == 651
                 and rows[1]["bit_unequal_active_wet"] == 0),
            "imported OVERFLOW boundary unexpectedly reproduced")
    return rows


def run(root: Path, *, expect_commit: str, plant: str | None = None) -> dict[str, object]:
    stamp = worktree_stamp()
    expected = expect_commit.lower()
    if plant == "stamp":
        expected = "0" * 40
    require(stamp["commit"].lower() == expected,
            f"producer commit {stamp['commit']} != expected {expected}")
    arrays = read_terms(root / RECORD, plant=plant)
    rows = _movement(arrays, root)
    if plant == "payload-ulp":
        require(False, "payload ULP plant fired")
    return {
        "format": "nemo-testcase-l4-orca2-round62-vector-advection-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": stamp,
        "record": str(root / RECORD),
        "record_sha256": sha256(root / RECORD),
        "compiled_sources": _compiled_scope(),
        "movement_rows": rows,
        "child_statement_census": {
            "after_keg": False,
            "ww": False,
            "verdict": "UNMEASURED_WITH_SPEC",
            "first_unresolved_statement": "dyn_keg",
            "second_unresolved_statement": "dyn_zad",
        },
        "imported_overflow_651_framing": "REFUTED",
        "planted_control": plant,
        "status": "STOPPED_FOR_RECORD",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("digest", "field-order", "payload-ulp", "stamp"))
    args = parser.parse_args(argv)
    try:
        report = run(args.root, expect_commit=args.expect_commit, plant=args.plant)
    except GateError as exc:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}: {exc}")
            return 1
        raise
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
