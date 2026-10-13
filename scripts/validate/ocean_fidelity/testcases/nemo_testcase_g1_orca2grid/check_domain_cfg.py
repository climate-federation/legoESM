#!/usr/bin/env python3
"""Admit the ORCA2G1 domain_cfg.nc written by NEMO's tools/DOMAINcfg.

make_domain_cfg.exe exits without MPI_Finalize after closing the file, so
mpirun's exit status says nothing about the domain.  This check of the file
itself is what admits it.  ``--plant`` corrupts the in-memory copy and the
check must then exit nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

DIMS = {"x": 180, "y": 148, "nav_lev": 2}
MESH = tuple(p + q for p in ("glam", "gphi", "e1", "e2") for q in "tuvf")
REQUIRED = MESH + (
    "ff_f", "ff_t", "e3t_1d", "e3w_1d", "e3t_0", "e3u_0", "e3v_0", "e3f_0",
    "e3w_0", "e3uw_0", "e3vw_0", "bottom_level", "top_level", "isf_draft",
    "bathy_metry", "jpiglo", "jpjglo", "jpkglo", "jperio", "ln_zco", "ln_zps",
    "ln_sco", "ln_isfcav")
SCALARS = {"jperio": 4, "jpkglo": 2, "ln_zco": 1, "ln_zps": 0}
DEPTH_M = 100.0
PLANTS = ("none", "missing-variable", "bad-level", "thin-cell", "wrong-jperio",
          "wrong-dims", "nonfinite")


class Refuse(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def load(path: Path):
    require(path.is_file(), f"{path} is absent")
    try:
        with netCDF4.Dataset(path, "r") as ds:
            ds.set_auto_mask(False)
            dims = {k: len(v) for k, v in ds.dimensions.items()}
            fields = {k: np.array(v[:]) for k, v in ds.variables.items()}
    except OSError as exc:
        raise Refuse(f"{path} is not a readable netCDF file: {exc}") from exc
    return dims, fields


def apply_plant(dims: dict, fields: dict, plant: str) -> None:
    require(plant in PLANTS, f"unknown plant {plant}")
    wet = np.argwhere(np.squeeze(fields["bottom_level"]) > 0)[0]
    if plant == "missing-variable":
        del fields["e3t_0"]
    elif plant == "bad-level":
        np.squeeze(fields["bottom_level"])[tuple(wet)] = 2
    elif plant == "thin-cell":
        fields["e3t_0"].reshape(DIMS["nav_lev"], DIMS["y"], DIMS["x"])[0][tuple(wet)] = 99.0
    elif plant == "wrong-jperio":
        fields["jperio"] = np.array(6)
    elif plant == "wrong-dims":
        dims["x"] -= 1
    elif plant == "nonfinite":
        fields["e1t"].flat[0] = np.nan


def check(path: Path, shipped: Path | None = None, plant: str = "none") -> dict:
    dims, fields = load(path)
    if plant != "none":
        apply_plant(dims, fields, plant)
    for name, size in DIMS.items():
        require(dims.get(name) == size, f"dimension {name} is {dims.get(name)}, not {size}")
    missing = [name for name in REQUIRED if name not in fields]
    require(not missing, f"missing variables {missing}")
    for name in REQUIRED:
        require(bool(np.isfinite(fields[name]).all()), f"{name} is not finite")
    for name, value in SCALARS.items():
        require(int(fields[name]) == value, f"{name} is {int(fields[name])}, not {value}")
    bottom = np.squeeze(fields["bottom_level"])
    require(bottom.shape == (DIMS["y"], DIMS["x"]), f"bottom_level has shape {bottom.shape}")
    require(set(np.unique(bottom).tolist()) == {0, 1}, "bottom_level is not exactly {0, 1}")
    wet = bottom == 1
    top_e3t = np.squeeze(fields["e3t_0"])[0]
    require(bool((top_e3t[wet] == DEPTH_M).all()), "e3t_0 is not 100 m on every wet cell")
    result = {
        "status": "PASS_G1_DOMAIN_CFG",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dims": dims,
        "wet_cells": int(wet.sum()),
        "scalars": {name: int(fields[name]) for name in (
            "jpiglo", "jpjglo", "jpkglo", "jperio", "ln_zco", "ln_zps", "ln_sco", "ln_isfcav")},
    }
    if shipped is not None:  # reported, not judged
        _, ref = load(shipped)
        result["vs_shipped"] = {
            "coast_is_shipped_bottom_level_gt_0": bool(
                np.array_equal(wet, np.squeeze(ref["bottom_level"]) > 0)),
            "mesh_fields_differing": [name for name in MESH if not np.array_equal(
                np.squeeze(fields[name]), np.squeeze(ref[name]))],
            "coriolis_max_abs_difference": {name: float(np.abs(
                np.squeeze(fields[name]) - np.squeeze(ref[name])).max())
                for name in ("ff_f", "ff_t")},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-cfg", type=Path, required=True)
    parser.add_argument("--shipped", type=Path)
    parser.add_argument("--plant", default="none", choices=PLANTS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.domain_cfg, args.shipped, args.plant)
    except Refuse as exc:
        print(f"REFUSE: {exc}" + ("  STATUS PLANT-FIRED" if args.plant != "none" else ""),
              file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"REFUSE: plant {args.plant} stayed green", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
