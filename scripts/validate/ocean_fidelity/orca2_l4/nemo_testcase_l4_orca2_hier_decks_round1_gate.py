#!/usr/bin/env python3
"""Preflight and admit ORCA2 hierarchy rung 10 without rebuilding NEMO."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round69_month_surface_gate as surface,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round1_acquisition"
RUNG_CFG = ACQUISITION / "rung10_namelist_cfg"
RUNG_ICE_CFG = ACQUISITION / "rung10_namelist_ice_cfg"
MANIFEST = ACQUISITION / "rung10_manifest.json"
SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/"
    "acquisition/orca1ice_surface_only_240step_np2"
)
SOURCE_ADMISSION = SOURCE.parent / "month_surface_admission.json"
BUILD = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE"
)
CPP = BUILD / "cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm"
COMPILED = BUILD / "BLD/ppsrc/nemo"

EXPECTED = {
    "source_admission": "9138ce62aca56408884e1e2b74e783ab3622a8e9ad1b12ddb7ec96ce4a700294",
    "deck_manifest": "09a350860ff6eaef17d1f0e18aa8e16c4d929e994d9d6804d0976f531b06f66e",
    "input_manifest": "3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5",
    "namelist_cfg": "036f3cec148b2e89cede910d13189db4cc8e2e74a9b9ecdede7a87a5c14a98ec",
    "namelist_ice_cfg": "6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89",
    "binary": "450410b5c9b1960c4cc8d05d690decca3308f888e70987b444682315bb846572",
    "cpp": "2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67",
    "stprk3": "b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422",
    "writer": "b116511cbefb6d2b6f1d911a77079a4268f343ea3aa1b861d08ac0c8d2bc4bcf",
}
CPP_KEYS = ("key_si3", "key_qco", "key_vco_1d3d", "key_RK3")
RESTART_FIELDS = ("sshn", "un", "vn", "tn", "sn")
MONTH_GRIDS = ("T", "U", "V", "W")
PLANTS = (
    "none",
    "deck-byte",
    "manifest-field",
    "field-name",
    "truncated",
    "missing-frame",
    "operand-ulp",
    "restart-ulp",
    "terminal-nonfinite",
    "resolved-switch",
    "sha-inventory",
)


class GateError(RuntimeError):
    """The rung-10 hierarchy record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_rows(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing SHA-256 manifest: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        require(
            len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
            f"malformed SHA-256 row: {line!r}",
        )
        name = parts[1].lstrip("* ")
        require(name == Path(name).name and name not in rows, f"unsafe/duplicate name: {name}")
        rows[name] = parts[0]
    require(bool(rows), f"empty SHA-256 manifest: {path}")
    return rows


def _check_manifest_targets(root: Path, name: str) -> dict[str, str]:
    rows = _manifest_rows(root / name)
    for filename, digest in rows.items():
        target = root / filename
        require(target.is_file(), f"manifest target is absent: {target}")
        require(sha256(target) == digest, f"manifest target changed: {target}")
    return rows


def _token(value: str) -> str:
    return value.split("!", 1)[0].strip().lower()


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", "deck-byte", "manifest-field"), f"invalid preflight plant {plant}")
    manifest = json.loads(MANIFEST.read_text())
    if plant == "manifest-field":
        manifest["rung"] = 9
    require(manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed")
    require(manifest["rung"] == 10, "manifest rung changed")
    require(tuple(manifest["build"]["cpp_keys"]) == CPP_KEYS, "CPP key inventory changed")
    require(manifest["run"] == {
        "mpi_ranks": 2,
        "first_step": 1,
        "last_step": 240,
        "from_rest": True,
        "initial_state_output": 1,
    }, "run protocol changed")

    require(sha256(SOURCE_ADMISSION) == EXPECTED["source_admission"], "source admission changed")
    require(sha256(SOURCE / "deck_files.sha256") == EXPECTED["deck_manifest"], "source deck manifest changed")
    require(sha256(SOURCE / "input_files.sha256") == EXPECTED["input_manifest"], "source input manifest changed")
    source_deck = _check_manifest_targets(SOURCE, "deck_files.sha256")
    require(source_deck["namelist_cfg"] == EXPECTED["namelist_cfg"], "source namelist pin changed")
    require(source_deck["namelist_ice_cfg"] == EXPECTED["namelist_ice_cfg"], "source ice namelist pin changed")

    cfg_digest = sha256(RUNG_CFG)
    if plant == "deck-byte":
        cfg_digest = "0" * 64
    require(cfg_digest == EXPECTED["namelist_cfg"], "committed rung-10 namelist changed")
    require(sha256(RUNG_ICE_CFG) == EXPECTED["namelist_ice_cfg"], "committed rung-10 ice namelist changed")
    require(RUNG_CFG.read_bytes() == (SOURCE / "namelist_cfg").read_bytes(), "rung-10 ocean namelist is not byte-identical")
    require(RUNG_ICE_CFG.read_bytes() == (SOURCE / "namelist_ice_cfg").read_bytes(), "rung-10 ice namelist is not byte-identical")

    require(sha256(SOURCE / "nemo") == EXPECTED["binary"], "instrumented binary changed")
    require(sha256(CPP) == EXPECTED["cpp"], "CPP-key file changed")
    require(sha256(COMPILED / "stprk3.f90") == EXPECTED["stprk3"], "compiled stprk3 changed")
    require(sha256(COMPILED / "l4_r69_surface.f90") == EXPECTED["writer"], "compiled writer changed")
    cpp_tokens = tuple(CPP.read_text().split("fppkeys", 1)[1].split())
    require(cpp_tokens == CPP_KEYS, f"resolved CPP keys changed: {cpp_tokens}")
    stprk3 = (COMPILED / "stprk3.f90").read_text()
    writer = (COMPILED / "l4_r69_surface.f90").read_text()
    require("CALL sbc        ( kstp, Nbb, Nbb )" in stprk3, "compiled SBC call is absent")
    require("CALL l4_r69_dump( kstp, Nbb )" in stprk3, "compiled writer call is absent")
    require(stprk3.index("CALL sbc        ( kstp, Nbb, Nbb )") < stprk3.index("CALL l4_r69_dump( kstp, Nbb )"), "writer is not after SBC")
    require("STATUS='NEW'" in writer and "STORAGE_SIZE(1._wp) /= 64" in writer, "compiled writer guards changed")

    ocean = namelist_values(RUNG_CFG)
    ice = namelist_values(RUNG_ICE_CFG)
    expected_assignments = {
        "namrun.nn_itend": "240",
        "namrun.nn_stock": "240",
        "namrun.nn_istate": "1",
        "namsbc.nn_ice": "2",
        "namagrif.ln_spc_dyn": ".true.",
    }
    for name, expected in expected_assignments.items():
        require(name in ocean and _token(ocean[name]) == expected, f"rung-10 assignment changed: {name}")
    require("nampar.jpl" in ice and _token(ice["nampar.jpl"]) == "1", "rung-10 jpl is not one")
    return {
        "status": "PREFLIGHT_PASS",
        "rung": 10,
        "deck_difference_lines": 0,
        "binary_sha256": EXPECTED["binary"],
        "cpp_keys": list(CPP_KEYS),
        "ln_spc_dyn_deck_value": True,
        "ln_spc_dyn_compiled_scope": "INERT_WITHOUT_key_agrif",
    }


def validate_frames(root: Path, *, plant: str = "none") -> dict[str, object]:
    expected = {surface._record_name(kt, rank) for kt in range(1, 241) for rank in (0, 1)}
    observed = {path.name for path in root.glob("oracle_*.bin")}
    if plant == "missing-frame":
        observed.discard(sorted(expected)[0])
    require(observed == expected, f"operand inventory differs: missing={sorted(expected-observed)[:2]} extra={sorted(observed-expected)[:2]}")
    comparisons = 0
    finite_fields = 0
    first = True
    for kt in range(1, 241):
        for rank in (0, 1):
            local_plant = plant if first and plant in ("field-name", "truncated") else "none"
            current = surface.read_surface(root / surface._record_name(kt, rank), kt=kt, rank=rank, plant=local_plant)
            for values in current["fields"].values():
                require(bool(np.isfinite(values).all()), f"non-finite operand frame kt={kt} rank={rank}")
                finite_fields += 1
            if kt <= 10:
                reference = surface.read_surface(SOURCE / surface._record_name(kt, rank), kt=kt, rank=rank)
                for name in surface.FIELDS:
                    values = np.asarray(current["fields"][name])
                    if plant == "operand-ulp" and first and name == surface.FIELDS[0]:
                        values = values.copy()
                        values.flat[0] = np.nextafter(values.flat[0], np.inf)
                    require(surface._raw_equal(values, reference["fields"][name]), f"operand differs: kt={kt} rank={rank} field={name}")
                    comparisons += 1
            first = False
    return {
        "status": "FINITE_ALL_STEPS_BIT_EXACT_KT1_10",
        "frames": len(expected),
        "finite_field_payloads": finite_fields,
        "bit_exact_field_comparisons": comparisons,
    }


def _mutated_copy(path: Path, target: Path) -> None:
    shutil.copy2(path, target)
    with Dataset(target, "r+") as dataset:
        for variable in dataset.variables.values():
            if variable.dtype == np.dtype("float64") and variable.size:
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:]).copy()
                values.reshape(-1)[0] = np.nextafter(values.reshape(-1)[0], np.inf)
                variable[:] = values
                return
    raise GateError(f"no mutable fp64 payload in {path}")


def validate_terminal(root: Path, *, plant: str = "none") -> dict[str, object]:
    names = [
        f"ORCA2_00000240_restart{suffix}_{rank:04d}.nc"
        for suffix in ("", "_ice") for rank in (0, 1)
    ]
    for index, name in enumerate(names):
        expected = SOURCE / name
        actual = root / name
        require(expected.is_file() and actual.is_file(), f"missing terminal restart: {name}")
        if plant == "restart-ulp" and index == 0:
            with tempfile.TemporaryDirectory(prefix="orca2-hier-r10-") as temporary:
                changed = Path(temporary) / name
                _mutated_copy(actual, changed)
                phase1._netcdf_equal_except_timestamp(expected, changed)
        else:
            phase1._netcdf_equal_except_timestamp(expected, actual)

    rows = []
    for rank in (0, 1):
        path = root / f"ORCA2_00000240_restart_{rank:04d}.nc"
        with Dataset(path) as dataset:
            require(float(np.asarray(dataset["kt"][:])) == 240.0, f"{path.name}: wrong kt")
            fields = {}
            for field in RESTART_FIELDS:
                variable = dataset[field]
                require(variable.dtype == np.dtype("float64"), f"{path.name}: {field} is not fp64")
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:])
                if plant == "terminal-nonfinite" and rank == 0 and field == RESTART_FIELDS[0]:
                    values = values.copy()
                    values.reshape(-1)[0] = np.nan
                require(bool(np.isfinite(values).all()), f"{path.name}: {field} is non-finite")
                fields[field] = list(values.shape)
        rows.append({"file": path.name, "fields": fields})
    return {"status": "BIT_EXACT_FINITE_FP64", "restart_shards": names, "ocean_fields": rows}


def validate_month_products(root: Path) -> dict[str, object]:
    names = [f"ORCA2_30d_00010101_00010130_grid_{grid}_{rank:04d}.nc" for grid in MONTH_GRIDS for rank in (0, 1)]
    checked = 0
    for name in names:
        path = root / name
        require(path.is_file(), f"missing month product: {name}")
        with Dataset(path) as dataset:
            for variable in dataset.variables.values():
                if np.issubdtype(variable.dtype, np.floating):
                    variable.set_auto_maskandscale(False)
                    require(bool(np.isfinite(np.asarray(variable[:])).all()), f"{name}: {variable.name} is non-finite")
                    checked += 1
    return {"status": "FINITE", "files": names, "floating_variables": checked}


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    ocean = (root / "ocean.output").read_text(errors="strict")
    stdout = (root / "run.user.stdout.log").read_text(errors="strict")
    timing = (root / "run.user.time.log").read_text(errors="strict")
    checks = {
        "stop_0": "STOP 0" in stdout,
        "run_done": "RUN_DONE" in timing,
        "last_step_240": re.search(r"number of the last time step\s+nn_itend\s+=\s+240\b", ocean) is not None,
        "restart_240": re.search(r"frequency of restart file\s+nn_stock\s+=\s+240\b", ocean) is not None,
        "from_rest": re.search(r"restart logical\s+ln_rstart\s+=\s+F\b", ocean) is not None,
        "nn_ice_2": re.search(r"ice management in the sbc.*nn_ice\s+=\s+2\b", ocean) is not None,
        "jpl_1": re.search(r"number of ice\s+categories\s+jpl\s+=\s+1\b", ocean) is not None,
    }
    if plant == "resolved-switch":
        checks["jpl_1"] = False
    require(all(checks.values()), f"resolved deck check failed: {checks}")
    return {"status": "PASS", **checks}


def validate_sha_inventory(root: Path, *, plant: str = "none") -> dict[str, object]:
    rows = _manifest_rows(root / "SHA256SUMS")
    expected = {
        path.name
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if plant == "sha-inventory":
        rows.pop(sorted(rows)[0], None)
    require(set(rows) == expected, f"SHA256SUMS inventory differs: missing={sorted(expected-set(rows))[:2]} extra={sorted(set(rows)-expected)[:2]}")
    for name, digest in rows.items():
        require(sha256(root / name) == digest, f"SHA256SUMS digest differs: {name}")
    return {"status": "COMPLETE", "regular_files": len(rows)}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    if plant in ("deck-byte", "manifest-field"):
        preflight(plant=plant)
    else:
        preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(sha256(root / "nemo") == EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == RUNG_CFG.read_bytes(), "record ocean namelist differs")
    require((root / "namelist_ice_cfg").read_bytes() == RUNG_ICE_CFG.read_bytes(), "record ice namelist differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record hierarchy manifest differs")
    frames = validate_frames(root, plant=plant)
    terminal = validate_terminal(root, plant=plant)
    products = validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung10-record-v1",
        "status": "PASS_RUNG10_RECORD",
        "claim_label": "independent",
        "rung": 10,
        "producer_commit": expect_commit,
        "deck_delta_from_source": [],
        "frames": frames,
        "terminal": terminal,
        "month_products": products,
        "resolved": resolved,
        "sha_inventory": sha_inventory,
        "ln_spc_dyn": {
            "deck_value": True,
            "compiled_scope": "INERT_WITHOUT_key_agrif",
            "reported_by_ocean_output": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report = preflight(plant=args.plant)
        else:
            require(args.record is not None and args.expect_commit, "record and expected commit are required")
            report = validate_record(args.record, expect_commit=args.expect_commit, plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (GateError, surface.GateError, phase1.GateError, KeyError, OSError, TypeError, ValueError) as error:
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
