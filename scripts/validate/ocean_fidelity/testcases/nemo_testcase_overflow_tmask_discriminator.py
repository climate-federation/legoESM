#!/usr/bin/env python3
"""Discriminate the OVERFLOW EOS mask from the oracle ``mesh_mask`` tmask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

DEFAULT_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/"
    "overflow_kt1_10/mesh_mask.nc")


def _oracle_tmask(path: Path, nlev: int) -> np.ndarray:
    with netCDF4.Dataset(str(path)) as data:
        raw = np.asarray(data.variables["tmask"][0, :nlev], dtype=np.float64)
    # NetCDF is (k,j,i); legoESM is (j,i,k).
    return raw.transpose(1, 2, 0) > 0.5


def run(mesh: Path, output: Path, plant: bool = False) -> dict:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_overflow_zps_card()
    applied = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    oracle = _oracle_tmask(mesh, applied.shape[-1])
    if applied.shape != oracle.shape:
        raise AssertionError(f"mask shape mismatch {applied.shape} != {oracle.shape}")
    if plant:
        applied = applied.copy()
        applied.flat[0] = ~applied.flat[0]
    differing = np.argwhere(applied != oracle)
    wet_zeroed = oracle & ~applied
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-overflow-tmask-discriminator-v1",
        "mesh_mask": str(mesh),
        "shape_jik": list(applied.shape),
        "oracle_wet_cells": int(np.count_nonzero(oracle)),
        "applied_wet_cells": int(np.count_nonzero(applied)),
        "differing_cells": int(differing.shape[0]),
        "nemo_wet_cells_zeroed_by_applied_mask": int(np.count_nonzero(wet_zeroed)),
        "first_differing_jik": (
            differing[0].astype(int).tolist() if differing.size else None),
        "source": {
            "nemo_mask_application": "src/OCE/TRA/eosbn2.F90:288",
            "nemo_hpg_consumer": "src/OCE/DYN/dynhpg.F90:340-390",
            "legoesm_mask": "OceanPartialCellCoordinate.is_active",
        },
        "plant": plant,
        "verdict": "PASS" if differing.size == 0 else "FAIL",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report = run(args.mesh, args.output, args.plant)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
