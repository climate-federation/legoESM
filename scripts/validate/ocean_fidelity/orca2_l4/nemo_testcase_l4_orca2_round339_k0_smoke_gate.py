#!/usr/bin/env python3
"""Fail-closed admission gate for the round-339 K0 ten-step smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as common,
)

GROUPS = {
    "nn_ice": "namsbc",
    "sn_ice": "namsbc_iif",
    "ln_icebergs": "namberg",
    "ln_drgice_imp": "namdrg",
    "ln_mxl0": "namzdf_tke",
    "nn_mxlice": "namzdf_tke",
}
RESOLVED = {
    "nn_ice": "1",
    "ln_drgice_imp": "F",
    "ln_mxl0": "T",
    "nn_mxlice": "0",
    "nn_itend": "10",
}
PLANTS = ("group", "error-banner", "selector", "nan", "truncation", "manifest")
NO_ICEBERGS_BANNER = "==>>>   No icebergs used"
PLANT_MESSAGES = {
    "group": "sn_ice belongs to ['namsbc'], expected membership in namsbc_iif",
    "error-banner": "ocean.output contains E R R O R",
    "selector": "resolved nn_ice is not 1",
    "nan": "non-finite restart variable",
    "truncation": "truncated restart plant",
    "manifest": "manifest producer commit moved",
}

GateError = common.GateError
require = common.require


def finite_restarts(root: Path, *, plant: str | None = None) -> dict[str, str]:
    import netCDF4

    paths = sorted(set(root.glob("ORCA2_00000010_restart_*.nc")))
    require(len(paths) >= 2, "ten-step restart inventory is incomplete")
    result: dict[str, str] = {}
    for index, path in enumerate(paths):
        if plant == "truncation" and index == 0:
            raise GateError(f"truncated restart plant: {path.name}")
        with netCDF4.Dataset(path) as dataset:
            numeric = [variable for variable in dataset.variables.values()
                       if np.issubdtype(variable.dtype, np.number)]
            require(numeric, f"restart has no numeric variables: {path.name}")
            for variable in numeric:
                values = np.asarray(variable[:])
                if plant == "nan" and index == 0 and values.size:
                    values = values.astype(np.float64, copy=True)
                    values.flat[0] = np.nan
                require(bool(np.all(np.isfinite(values))),
                        f"non-finite restart variable {path.name}:{variable.name}")
        result[path.name] = common.sha256(path)
    return result


def run_gate(root: Path, reference_namelist: Path, *, expected_commit: str | None,
             plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    common.validate_group_membership(reference_namelist, GROUPS)
    staged = root / "staged.namelist_cfg"
    require(staged.is_file(), "missing staged.namelist_cfg")
    if plant == "group":
        text = staged.read_text(encoding="utf-8")
        import re
        block = re.search(r"(?ims)^&namsbc_iif\b.*?^/", text)
        require(block is not None, "staged namsbc_iif group is absent")
        sn_line = re.search(r"(?im)^\s*sn_ice\s*=.*$", block.group(0))
        require(sn_line is not None, "staged sn_ice is absent")
        planted_block = block.group(0).replace(sn_line.group(0), "")
        planted = text[:block.start()] + planted_block + text[block.end():]
        planted = re.sub(r"(?im)^(&namsbc\b.*)$", r"\1\n" + sn_line.group(0), planted, count=1)
        scratch_namelist = root / ".group.namelist_cfg"
        scratch_namelist.write_text(planted, encoding="utf-8")
        try:
            common.validate_group_membership(scratch_namelist, GROUPS)
        finally:
            scratch_namelist.unlink(missing_ok=True)
    else:
        common.validate_group_membership(staged, GROUPS)
    output = root / "ocean.output"
    require(output.is_file(), "missing ocean.output")
    if plant in ("error-banner", "selector"):
        text = output.read_text(encoding="utf-8", errors="replace")
        if plant == "error-banner":
            text += "\nE R R O R planted\n"
        else:
            import re
            text = re.sub(r"(nn_ice\s*=\s*)1(\s)", r"\g<1>2\2", text, count=1)
        scratch = root / f".{plant}.ocean.output"
        scratch.write_text(text, encoding="utf-8")
        try:
            common.validate_ocean_output(scratch, RESOLVED)
        finally:
            scratch.unlink(missing_ok=True)
    else:
        common.validate_ocean_output(output, RESOLVED)
    output_text = output.read_text(encoding="utf-8", errors="replace")
    require(NO_ICEBERGS_BANNER in output_text, "NEMO did not report No icebergs used")
    require("ice_cover_clim.nc" in output_text,
            "resolved ice-cover file is not ice_cover_clim.nc")
    stdout = (root / "run.user.stdout.log").read_text(encoding="utf-8", errors="replace")
    require("STOP 0" in stdout, "NEMO did not report STOP 0")
    require("E R R O R" not in stdout, "stdout contains E R R O R")
    keys = (root / "compiled_keys.txt").read_text(encoding="utf-8")
    require("key_si3" in keys.split(), "key_si3 is absent")
    require("key_xios" not in keys.split(), "key_xios is present")
    manifest = common.verify_manifest(root / "producer_manifest.json",
                                      expected_commit=("0" * 40 if plant == "manifest" else expected_commit))
    require(manifest["my_src_files"] == [],
            f"K0 MY_SRC is not empty: {manifest['my_src_files']}")
    require(any(name.startswith("build_sources/ppsrc/nemo/")
                for name in manifest["build_sources"]),
            "K0 ppsrc inventory is empty")
    restarts = finite_restarts(root, plant=plant)
    return {
        "status": "PASS_R339_K0_SMOKE",
        "producer_commit": manifest["producer_commit"],
        "resolved": RESOLVED,
        "restarts": restarts,
        "scores_physics": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-namelist", type=Path, required=True)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_gate(args.root, args.reference_namelist,
                          expected_commit=args.expect_commit, plant=args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
