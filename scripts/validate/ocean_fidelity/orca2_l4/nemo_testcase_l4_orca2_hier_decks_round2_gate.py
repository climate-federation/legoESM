#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-9 no-sea-ice record."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_hier_decks_round1_gate as rung10,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round2_acquisition"
MANIFEST = ACQUISITION / "rung9_manifest.json"
SENTINEL = ACQUISITION / "rung9_unread_namelist_ice_cfg"
RUNG10_CFG = rung10.RUNG_CFG
RUNG10_ICE_CFG = rung10.RUNG_ICE_CFG
RUNG10_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung10/record"
)
RUNG10_ADMISSION = RUNG10_RECORD.parent / "rung10_admission.json"
BUILD = rung10.BUILD
COMPILED = rung10.COMPILED

RUNG9_CFG_SHA = "9177246fef7f2d9da1dcfcf13b1ccb8107b3004b45074159ffab67accc86eb15"
RUNG10_ICE_CFG_SHA = "6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89"
PLANTS = (
    "none",
    "deck-extra",
    "build-pin",
    "ice-artifact",
    "field-name",
    "truncated",
    "missing-frame",
    "frame-nonfinite",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "resolved-consequence",
    "sha-inventory",
)


class GateError(RuntimeError):
    """A frozen rung-9 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rung9_cfg_bytes(*, plant: str = "none") -> bytes:
    source = RUNG10_CFG.read_text()
    old = "   nn_ice      = 2         !  =0 no ice boundary condition"
    new = "   nn_ice      = 0         !  =0 no ice boundary condition"
    require(source.count(old) == 1, "upper-rung nn_ice source line changed")
    result = source.replace(old, new)
    if plant == "deck-extra":
        old_fwb = "   nn_fwb      = 2         !  FreshWater Budget:"
        require(result.count(old_fwb) == 1, "deck-extra plant target changed")
        result = result.replace(old_fwb, old_fwb.replace("= 2", "= 1"))
    return result.encode()


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _ice_assignment_inventory() -> list[str]:
    values = namelist_values(RUNG10_ICE_CFG)
    return sorted(values)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", "deck-extra", "build-pin", "ice-artifact"), f"invalid preflight plant: {plant}")
    rung10.preflight()
    admission = json.loads(RUNG10_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG10_RECORD", "rung-10 admission is not PASS")
    require(admission.get("frames", {}).get("bit_exact_field_comparisons") == 200, "rung-10 bit identity evidence changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed")
    require(manifest["rung"] == 9, "manifest rung changed")
    require(manifest["run"] == {
        "mpi_ranks": 2,
        "first_step": 1,
        "last_step": 240,
        "from_rest": True,
        "initial_state_output": 1,
    }, "run protocol changed")
    require(tuple(manifest["build"]["cpp_keys"]) == rung10.CPP_KEYS, "CPP keys changed")

    cfg = rung9_cfg_bytes(plant=plant)
    require(sha256_bytes(cfg) == RUNG9_CFG_SHA, "rung-9 ocean namelist digest/delta changed")
    ice_digest = rung10.sha256(RUNG10_ICE_CFG)
    if plant == "ice-artifact":
        ice_digest = "0" * 64
    require(ice_digest == RUNG10_ICE_CFG_SHA, "retained ice namelist artifact changed")
    require(SENTINEL.read_text() == "REFUSE_IF_NEMO_READS_RUNG9_ICE_NAMELIST\n", "unread sentinel changed")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="orca2-rung9-deck-") as temporary:
        root = Path(temporary)
        # Use separate paths because namelist_values reads a path, while preserving
        # the exact committed bytes above.
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(RUNG10_CFG.read_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    changed = {
        name: [rung10._token(upper[name]), rung10._token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    require(changed == {"namsbc.nn_ice": ["2", "0"]}, f"rung delta is not one module: {changed}")
    upper_lines = RUNG10_CFG.read_text().splitlines()
    lower_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(upper_lines, lower_lines, strict=True))
        if before != after
    ]
    require(line_delta == [[86, upper_lines[85], lower_lines[85]]], f"unexpected line delta: {line_delta}")

    expected_build = {
        "binary_sha256": rung10.EXPECTED["binary"],
        "cpp_keys_sha256": rung10.EXPECTED["cpp"],
        "compiled_stprk3_sha256": rung10.EXPECTED["stprk3"],
        "compiled_writer_sha256": rung10.EXPECTED["writer"],
    }
    if plant == "build-pin":
        expected_build["binary_sha256"] = "0" * 64
    for key, value in expected_build.items():
        require(manifest["build"][key] == value, f"manifest build pin changed: {key}")

    sources = {
        "sbcmod": (COMPILED / "sbcmod.f90").read_text(),
        "icestp": (COMPILED / "icestp.f90").read_text(),
        "zdftke": (COMPILED / "zdftke.f90").read_text(),
        "zdfdrg": (COMPILED / "zdfdrg.f90").read_text(),
        "sbcfwb": (COMPILED / "sbcfwb.f90").read_text(),
    }
    required = {
        "sbcmod": ("IF( nn_ice == 0 ) THEN", "ELSEIF( nn_ice == 2 ) THEN", "CALL ice_stp"),
        "icestp": ("CALL load_nml( numnam_ice_cfg, 'namelist_ice_cfg'",),
        "zdftke": ("nn_mxlice > 0 .AND. nn_ice == 0", "nn_mxlice = 0"),
        "zdfdrg": ("ln_drgice_imp .AND.   nn_ice /= 2", "ln_drgice_imp = .FALSE."),
        "sbcfwb": ("IF ( nn_ice/=2 ) nn_fwb_voltype = 2",),
    }
    for name, needles in required.items():
        require(all(needle in sources[name] for needle in needles), f"compiled no-ice branch changed: {name}")

    return {
        "status": "PREFLIGHT_PASS_RUNG9",
        "rung": 9,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG10_RECORD",
        "assignment_delta": changed,
        "line_delta": line_delta,
        "retained_ice_namelist_assignments_inert": _ice_assignment_inventory(),
        "compiled_consequences": {
            "ice_fraction": 0,
            "ice_init_called": False,
            "ice_stp_called": False,
            "namelist_ice_cfg_loaded": False,
            "nn_mxlice": 0,
            "ln_drgice_imp": False,
            "nn_fwb_voltype": 2,
        },
        "cpp_keys": list(rung10.CPP_KEYS),
        "binary_sha256": rung10.EXPECTED["binary"],
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung9_cfg_bytes(),
        "namelist_ice_cfg": RUNG10_ICE_CFG.read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(target.is_file() and target.read_bytes() == payload, f"existing deck artifact differs: {target}")
        else:
            target.write_bytes(payload)
    rows = "".join(f"{sha256_bytes(payload)}  {name}\n" for name, payload in expected.items())
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "existing deck SHA256SUMS differs")
    else:
        ledger.write_text(rows)


def validate_frames(root: Path, *, plant: str = "none") -> dict[str, object]:
    expected = {rung10.surface._record_name(kt, rank) for kt in range(1, 241) for rank in (0, 1)}
    observed = {path.name for path in root.glob("oracle_*.bin")}
    if plant == "missing-frame":
        observed.discard(sorted(expected)[0])
    require(observed == expected, f"operand inventory differs: missing={sorted(expected-observed)[:2]} extra={sorted(observed-expected)[:2]}")
    finite = 0
    first = True
    for kt in range(1, 241):
        for rank in (0, 1):
            parser_plant = plant if first and plant in ("field-name", "truncated") else "none"
            frame = rung10.surface.read_surface(root / rung10.surface._record_name(kt, rank), kt=kt, rank=rank, plant=parser_plant)
            for name, payload in frame["fields"].items():
                values = np.asarray(payload)
                if plant == "frame-nonfinite" and first and name == rung10.surface.FIELDS[0]:
                    values = values.copy()
                    values.flat[0] = np.nan
                require(bool(np.isfinite(values).all()), f"non-finite operand kt={kt} rank={rank} field={name}")
                finite += 1
            first = False
    return {"status": "SELF_DESCRIBING_FINITE", "frames": len(expected), "finite_field_payloads": finite}


def validate_terminal(root: Path, *, plant: str = "none") -> dict[str, object]:
    forbidden = sorted(
        path.name for pattern in ("*restart_ice*", "output.init_ice*", "output.namelist.ice")
        for path in root.glob(pattern)
    )
    require(not forbidden, f"ice products exist in no-ice record: {forbidden}")
    rows = []
    for rank in (0, 1):
        path = root / f"ORCA2_00000240_restart_{rank:04d}.nc"
        require(path.is_file(), f"missing terminal ocean restart: {path.name}")
        with Dataset(path) as dataset:
            step = float(np.asarray(dataset["kt"][:]))
            if plant == "terminal-step" and rank == 0:
                step = 239.0
            if step != 240.0:
                message = f"{path.name}: wrong terminal step {step}"
                if plant == "terminal-step":
                    print(f"STATUS PLANT-FIRED: {message}")
                raise GateError(message)
            fields = {}
            for field in rung10.RESTART_FIELDS:
                variable = dataset[field]
                require(variable.dtype == np.dtype("float64"), f"{path.name}: {field} is not fp64")
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:])
                if plant == "terminal-nonfinite" and rank == 0 and field == rung10.RESTART_FIELDS[0]:
                    values = values.copy()
                    values.flat[0] = np.nan
                if not bool(np.isfinite(values).all()):
                    message = f"{path.name}: {field} is non-finite"
                    if plant == "terminal-nonfinite":
                        print(f"STATUS PLANT-FIRED: {message}")
                    raise GateError(message)
                fields[field] = list(values.shape)
        rows.append({"file": path.name, "fields": fields})
    return {"status": "FINITE_FP64_NO_ICE_PRODUCTS", "restart_shards": 2, "ocean_fields": rows}


def validate_resolved(
    root: Path,
    *,
    plant: str = "none",
    tke_active: bool = True,
) -> dict[str, object]:
    ocean = (root / "ocean.output").read_text(errors="strict")
    stdout = (root / "run.user.stdout.log").read_text(errors="strict")
    timing = (root / "run.user.time.log").read_text(errors="strict")
    nn_mxlice_print = re.search(
        r"type of scaling under sea-ice\s+nn_mxlice\s*=\s*0\b", ocean
    )
    fwb_match = re.search(
        r"FreshWater Budget control.*nn_fwb\s*=\s*([0-4])\b", ocean
    )
    require(fwb_match is not None, "resolved nn_fwb print is absent")
    fwb_active = fwb_match.group(1) != "0"
    fwb_voltype_print = "nn_fwb_voltype = 2: Control OCEAN volume" in ocean
    checks = {
        "stop_0": "STOP 0" in stdout,
        "run_done": "RUN_DONE" in timing,
        "last_step_240": re.search(r"number of the last time step\s+nn_itend\s+=\s+240\b", ocean) is not None,
        "restart_240": re.search(r"frequency of restart file\s+nn_stock\s+=\s+240\b", ocean) is not None,
        "from_rest": re.search(r"restart logical\s+ln_rstart\s+=\s+F\b", ocean) is not None,
        "nn_ice_0": re.search(r"ice management in the sbc.*nn_ice\s+=\s+0\b", ocean) is not None,
        "no_si3_init": "Sea Ice Model: SI3" not in ocean,
        "no_ice_namelist_output": "output.namelist.ice" not in ocean,
        "sentinel_unread": SENTINEL.read_text().strip() not in ocean + stdout,
        "nn_mxlice_0": nn_mxlice_print is not None if tke_active else nn_mxlice_print is None,
        "ln_drgice_imp_false": re.search(r"implicit ice-ocean drag\s+ln_drgice_imp\s*=\s*F\b", ocean) is not None,
        "freshwater_budget_print_matches_activity": fwb_voltype_print == fwb_active,
    }
    require((root / "namelist_ice_cfg").read_bytes() == SENTINEL.read_bytes(), "execution ice sentinel differs")
    if plant == "ice-sentinel-read":
        checks["sentinel_unread"] = False
    if plant == "resolved-consequence":
        checks["nn_mxlice_0"] = False
    require(all(checks.values()), f"resolved no-ice checks failed: {checks}")
    return {"status": "PASS_NO_ICE_SENTINEL_UNREAD", **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    if plant in ("deck-extra", "build-pin", "ice-artifact"):
        preflight(plant=plant)
    else:
        preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(rung10.sha256(root / "nemo") == rung10.EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == rung9_cfg_bytes(), "record ocean namelist differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record manifest differs")
    frames = validate_frames(root, plant=plant)
    terminal = validate_terminal(root, plant=plant)
    month = rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung9-record-v1",
        "status": "PASS_RUNG9_RECORD",
        "claim_label": "independent",
        "rung": 9,
        "producer_commit": expect_commit,
        "deck_delta_from_rung10": {"namsbc.nn_ice": [2, 0]},
        "frames": frames,
        "terminal": terminal,
        "month_products": month,
        "resolved": resolved,
        "sha_inventory": sha_inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage-deck", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report = preflight(plant=args.plant)
            if args.stage_deck:
                stage_deck(args.stage_deck)
        else:
            require(args.record is not None and args.expect_commit, "record and expected commit are required")
            report = validate_record(args.record, expect_commit=args.expect_commit, plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (GateError, rung10.GateError, rung10.surface.GateError, KeyError, OSError, TypeError, ValueError) as error:
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
