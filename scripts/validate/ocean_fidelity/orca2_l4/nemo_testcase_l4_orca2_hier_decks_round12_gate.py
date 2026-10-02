#!/usr/bin/env python3
"""Close rung-1 admission and classify its boundary with main-lane rung 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round11_gate as rung1,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

RUNG1_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung1"
)
RUNG1_RECORD = RUNG1_ROOT / "record"
RUNG1_ADMISSION = RUNG1_ROOT / "rung1_admission.json"
MAIN_RUNG0_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round90/"
    "acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2"
)
REFERENCE_NAMELIST = (
    rung1.COMPILED.parents[2] / "EXP00" / "namelist_ref"
)

RUNG1_ADMISSION_SHA = "4a89741a11b8379cd7e5e27768a314ac2936b06c1b19ec93faf895df9e9a28fa"
REFERENCE_NAMELIST_SHA = "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd"
SOURCE_SHA = {
    "zdfphy": "903e0d1220a2f9b38cdd55b6eece5dbbfed992f12f497155cca8633702e7451e",
    "sbcmod": "f2ce40cd468c056e3b39ccaa74cdf7b76fd3d477ab6aaa08b0bf6d18e807c52b",
    "traqsr": "e35de18655671e28f806c42663330efc552ab838d417438beec4142e6bfd4493",
}

PROTOCOL = {
    "namrun.ln_rst_list",
    "namrun.nn_itend",
    "namrun.nn_stock",
    "namrun.nn_stocklist",
}
ZERO_FILE = {
    "namsbc_flx.sn_emp",
    "namsbc_flx.sn_qsr",
    "namsbc_flx.sn_qtot",
    "namsbc_flx.sn_utau",
    "namsbc_flx.sn_vtau",
}
DAMPING = {"namtra_dmp.ln_tradmp", "namtsd.ln_tsd_dmp"}
OWNER_OFF = {
    "namagrif.ln_spc_dyn",
    "namsbc_ssr.ln_sssr_bnd",
    "namtra_qsr.nn_chldta",
}
REFERENCE_DEFAULT = {
    "namzdf.ln_zdfgls",
    "namzdf.ln_zdfmfc",
    "namzdf.ln_zdfnpc",
    "namzdf.ln_zdfosm",
    "namzdf.ln_zdfric",
    "namzdf.ln_zdfswm",
}
ACTIVE = {"namzdf.nn_havtb"}
PLANTS = (
    "none",
    "admission-pin",
    "diff-class",
    "source-branch",
    "inactive-owner",
    "shape-signal",
)


class GateError(RuntimeError):
    """A frozen round-12 predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(output: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*([FT]|[01])\b", output)
    require(match is not None, f"resolved value missing: {name}")
    return match.group(1)


def _background_shape_signal(*, plant: str) -> dict[str, object]:
    rung_mesh = RUNG1_RECORD / "mesh_mask_0000.nc"
    main_mesh = MAIN_RUNG0_RECORD / "mesh_mask_0000.nc"
    with Dataset(rung_mesh) as rung_ds, Dataset(main_mesh) as main_ds:
        rung_lat = np.asarray(rung_ds.variables["gphit"][0], dtype=np.float64)
        main_lat = np.asarray(main_ds.variables["gphit"][0], dtype=np.float64)
        rung_wet = np.asarray(rung_ds.variables["tmask"][0, 0]) != 0
        main_wet = np.asarray(main_ds.variables["tmask"][0, 0]) != 0
    require(np.array_equal(rung_lat, main_lat), "rung-1/main-rung-0 latitude changed")
    require(np.array_equal(rung_wet, main_wet), "rung-1/main-rung-0 wet mask changed")

    factor = np.ones_like(rung_lat)
    south = (-15.0 <= rung_lat) & (rung_lat < -5.0)
    equator = (-5.0 <= rung_lat) & (rung_lat < 5.0)
    north = (5.0 <= rung_lat) & (rung_lat < 15.0)
    factor[south] = 1.0 - 0.09 * (rung_lat[south] + 15.0)
    factor[equator] = 0.1
    factor[north] = 0.1 + 0.09 * (rung_lat[north] - 5.0)
    affected = rung_wet & (factor != 1.0)
    if plant == "shape-signal":
        affected[:] = False
    require(np.any(affected), "nn_havtb shape has no wet-cell signal")
    background = 1.2e-5
    return {
        "affected_surface_wet_cells_rank0": int(np.count_nonzero(affected)),
        "minimum_factor_rank0": float(np.min(factor[affected])),
        "maximum_background_difference_m2_s_rank0": float(
            np.max(np.abs(background * factor[affected] - background))
        ),
        "mesh_equal": True,
    }


def evaluate(*, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"invalid plant: {plant}")
    inherited = rung1.preflight()

    admission_sha = sha256(RUNG1_ADMISSION)
    if plant == "admission-pin":
        admission_sha = "0" * 64
    require(admission_sha == RUNG1_ADMISSION_SHA, "rung-1 admission changed")
    admission = json.loads(RUNG1_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG1_RECORD", "rung 1 is not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "rung-1 frame count changed")

    differences = inherited["main_rung0_comparison"]["semantic_differences"]
    keys = set(differences)
    if plant == "diff-class":
        keys.add("plant.unclassified")
    groups = PROTOCOL | ZERO_FILE | DAMPING | OWNER_OFF | REFERENCE_DEFAULT | ACTIVE
    require(keys == groups, f"unclassified rung boundary: {sorted(keys ^ groups)}")
    require(differences["namzdf.nn_havtb"] == ["1", "0"], "nn_havtb boundary changed")

    for name, expected in SOURCE_SHA.items():
        actual = sha256(rung1.COMPILED / f"{name}.f90")
        if plant == "source-branch" and name == "zdfphy":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
    require(sha256(REFERENCE_NAMELIST) == REFERENCE_NAMELIST_SHA, "reference namelist changed")

    reference = namelist_values(REFERENCE_NAMELIST)
    for key in REFERENCE_DEFAULT:
        require(
            rung1._semantic_token(reference[key]) == ".false.",
            f"reference default is not false: {key}",
        )

    rung_output = (RUNG1_RECORD / "ocean.output").read_text(errors="strict")
    main_output = (MAIN_RUNG0_RECORD / "ocean.output").read_text(errors="strict")
    owner_values = {
        "rung1_ln_ssr": _resolved(rung_output, "ln_ssr"),
        "main_ln_ssr": _resolved(main_output, "ln_ssr"),
        "rung1_ln_traqsr": _resolved(rung_output, "ln_traqsr"),
        "main_ln_traqsr": _resolved(main_output, "ln_traqsr"),
    }
    if plant == "inactive-owner":
        owner_values["rung1_ln_ssr"] = "T"
    require(set(owner_values.values()) == {"F"}, f"owner-off boundary changed: {owner_values}")
    resolved_havtb = {
        "rung1": _resolved(rung_output, "nn_havtb"),
        "main_rung0": _resolved(main_output, "nn_havtb"),
    }
    require(resolved_havtb == {"rung1": "1", "main_rung0": "0"}, "resolved nn_havtb changed")

    compiled_occurrences = []
    for path in sorted(rung1.COMPILED.glob("*.f90")):
        if b"ln_spc_dyn" in path.read_bytes():
            compiled_occurrences.append(path.name)
    require(not compiled_occurrences, f"ln_spc_dyn reached compiled source: {compiled_occurrences}")

    return {
        "status": "PASS_RUNG1_RECORD_ACTIVE_RUNG0_DIFFERENCE",
        "claim_label": "independent",
        "rung1_admission": "PASS_RUNG1_RECORD",
        "classified_assignments": {
            "protocol": sorted(PROTOCOL),
            "zero_file_name": sorted(ZERO_FILE),
            "intended_damping_boundary": sorted(DAMPING),
            "owner_off": sorted(OWNER_OFF),
            "explicit_reference_default": sorted(REFERENCE_DEFAULT),
            "physics_active_extra": sorted(ACTIVE),
        },
        "resolved_owner_values": owner_values,
        "resolved_nn_havtb": resolved_havtb,
        "nn_havtb_signal": _background_shape_signal(plant=plant),
        "decision_needed": (
            "Choose which deck owns nn_havtb: rung 1 retains shipped ORCA2 value 1, "
            "while main rung 0 uses GYRE-physics value 0."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (GateError, KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
