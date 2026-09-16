#!/usr/bin/env python3
"""Read-only admission inventory for the native ORCA2 fidelity lane.

This probe does not import or execute the ocean model.  It answers the earlier
question that must be closed before a kt=1--3 comparison is meaningful: does
the checked-out tree contain an executable native ORCA2 card, and is a usable
NEMO record already present?  ``--plant missing-record`` is the binding
negative control.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

HISTORICAL_REVISION = "b7ce08cc8afa5cf377922abf198cf1794fab8a73"
PINNED_TKE_SHA256 = "31675493f022f71a609142f53bbe220c111b09e9a9a352926a1aff7358770a52"
TKE_MAGIC = "NEMO_L4_TKEW_1"
TKE_RECORD = "oracle_tke_walk_kt00000002.bin"
TKE_HEADER_INTS = 15
TKE_3D_FIELDS = 17
TKE_2D_FIELDS = 7
TKE_FIELDS = TKE_3D_FIELDS + TKE_2D_FIELDS
REQUIRED_DECK_INPUTS = (
    "ORCA_R2_zps_domcfg.nc",
    "data_1m_potential_temperature_nomask.nc",
    "data_1m_salinity_nomask.nc",
)
UNMEASURED_FEATURES = (
    "staged_gm_eiv",
    "linear_implicit_bottom_drag",
    "internal_wave_mixing",
    "spatial_lateral_viscosity",
    "freshwater_budget_carry",
    "si3_jpl5_layered_prather_state",
)


class InventoryError(RuntimeError):
    """A fail-closed inventory predicate was not met."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(repo_root: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{revision}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        result.returncode == 0,
        f"historical source {revision}:{path} is unavailable: {result.stderr.strip()}",
    )
    return result.stdout


def _function_names(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dispatch_keys(source: str) -> tuple[str, ...]:
    """Return literal keys of the ``builders`` map in testcase dispatch."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "builders" for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            values = []
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    values.append(key.value)
            return tuple(values)
    raise InventoryError("testcase recipe has no literal builders dispatch")


def _historical_unmeasured_features(source: str) -> tuple[str, ...]:
    """Read the historical card guard without importing historical code."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "NEMOTestcaseCard":
            continue
        for keyword in node.keywords:
            if keyword.arg == "unmeasured_features":
                value = ast.literal_eval(keyword.value)
                require(isinstance(value, tuple), "historical feature guard is not a tuple")
                return tuple(str(item) for item in value)
    raise InventoryError("historical ORCA2 card has no unmeasured_features guard")


def _read_tke_schema(path: Path) -> dict[str, Any]:
    """Validate the self-describing Phase-2v TKE frame through exact EOF."""
    require(path.is_file(), f"missing native TKE record {path}")
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated TKE magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * TKE_HEADER_INTS)
        require(len(raw_header) == 4 * TKE_HEADER_INTS, "truncated TKE header")
        header = struct.unpack(f"={TKE_HEADER_INTS}i", raw_header)
        (
            version,
            kt,
            kbb,
            kmm,
            jpi,
            jpj,
            jpk,
            real_bits,
            n3,
            n2,
            payload,
            nn_eice,
            nn_etau,
            nn_mxl,
            nn_pdl,
        ) = header
        require(magic == TKE_MAGIC, f"unexpected TKE magic {magic!r}")
        require((version, kt, kbb, kmm) == (1, 2, 3, 3), "wrong TKE step/levels")
        require(
            (real_bits, n3, n2) == (64, TKE_3D_FIELDS, TKE_2D_FIELDS),
            "wrong TKE precision/field counts",
        )
        require(
            (nn_eice, nn_etau, nn_mxl, nn_pdl) == (1, 1, 3, 1),
            "wrong TKE selector tuple",
        )
        raw_extents = handle.read(12 * TKE_FIELDS)
        require(len(raw_extents) == 12 * TKE_FIELDS, "truncated TKE extents")
        flat = struct.unpack(f"={3 * TKE_FIELDS}i", raw_extents)
        extents = tuple(tuple(flat[3 * i : 3 * i + 3]) for i in range(TKE_FIELDS))
        expected = ((jpi, jpj, jpk),) * TKE_3D_FIELDS + ((jpi, jpj, 1),) * TKE_2D_FIELDS
        require(extents == expected, "TKE extent table disagrees with header")
        derived_payload = sum(a * b * c for a, b, c in expected)
        require(payload == derived_payload, "TKE payload count disagrees with extents")
        expected_bytes = 16 + 4 * TKE_HEADER_INTS + 12 * TKE_FIELDS + 8 * payload
        handle.seek(expected_bytes)
        require(handle.read(1) == b"", "TKE record has trailing bytes")
    require(path.stat().st_size == expected_bytes, "TKE record fails exact EOF")
    actual_sha = sha256(path)
    return {
        "bytes": path.stat().st_size,
        "exact_eof": True,
        "header": list(header),
        "magic": magic,
        "path": str(path),
        "pinned_sha256": PINNED_TKE_SHA256,
        "sha256": actual_sha,
        "sha256_matches_pinned_receipt": actual_sha == PINNED_TKE_SHA256,
    }


def _expected_hash(script: str, name: str) -> str:
    match = re.search(rf"^readonly {re.escape(name)}=([0-9a-f]{{64}})$", script, re.M)
    require(match is not None, f"run.sh has no {name}")
    return match.group(1)


def _complete_run(root: Path) -> dict[str, Any]:
    require(root.is_dir(), f"missing ORCA2 record directory {root}")
    time_log = root / "run.user.time.log"
    time_step = root / "time.step"
    run_script = root / "run.sh"
    for path in (time_log, time_step, run_script, root / "ocean.output"):
        require(path.is_file(), f"missing ORCA2 provenance file {path}")
    log = time_log.read_text()
    require("MPIRUN_RC=0" in log and "RUN DONE" in log, f"incomplete ORCA2 run {root}")
    require(int(time_step.read_text().strip()) == 10, f"ORCA2 run did not reach kt=10: {root}")
    script = run_script.read_text()
    expected_binary = _expected_hash(script, "EXPECTED_BINARY_SHA256")
    expected_deck = _expected_hash(script, "EXPECTED_DECK_MANIFEST_SHA256")
    expected_inputs = _expected_hash(script, "EXPECTED_INPUT_MANIFEST_SHA256")
    actual_binary = sha256(root / "nemo")
    actual_deck = sha256(root / "deck_files.sha256")
    actual_inputs = sha256(root / "input_files.sha256")
    require(actual_binary == expected_binary, f"binary provenance mismatch in {root}")
    require(actual_deck == expected_deck, f"deck provenance mismatch in {root}")
    require(actual_inputs == expected_inputs, f"input provenance mismatch in {root}")
    return {
        "binary_sha256": actual_binary,
        "completed": True,
        "deck_manifest_sha256": actual_deck,
        "input_manifest_sha256": actual_inputs,
        "oracle_record_count": len(tuple(root.glob("oracle_*.bin"))),
        "path": str(root),
        "time_step": 10,
    }


def _required_frames(root: Path) -> tuple[Path, ...]:
    entries = tuple(root / f"oracle_step_entry_kt{kt:08d}.bin" for kt in range(1, 4))
    stages = tuple(
        root / f"oracle_stage_kt{kt:08d}_s{stage}.bin"
        for kt in range(1, 4)
        for stage in range(1, 4)
    )
    return entries + stages + (root / TKE_RECORD,)


def _phase3_inventory(phase3_root: Path) -> dict[str, Any]:
    round_dirs = tuple(
        path for path in phase3_root.iterdir() if path.is_dir() and path.name.startswith("round")
    )
    named = sorted(
        path
        for root in round_dirs
        for path in root.rglob("*")
        if path.is_file() and "orca2" in str(path.relative_to(phase3_root)).lower()
    )
    raw = tuple(path for path in named if path.suffix == ".bin")
    representative = tuple(
        path
        for path in (
            phase3_root / "round22/orca2_probe/orca2-r22-stage-transport-thickness.json",
            phase3_root / "round24/crosscard/orca2_entry_stage.json",
            phase3_root / "round24/crosscard/orca2_production_w.json",
            phase3_root / "round26/orca2/orca2_fourth_card_blocked_round26.json",
            phase3_root / "round27/orca2/orca2_hpg_model_arm.json",
        )
        if path.is_file()
    )
    return {
        "derived_orca2_artifact_count": len(named),
        "raw_orca2_bin_count": len(raw),
        "representative_derived_artifacts": [str(path) for path in representative],
        "status": "DERIVED_ONLY" if named and not raw else "RAW_PRESENT" if raw else "MISSING",
    }


def inventory(
    repo_root: Path,
    oracle_root: Path,
    phase3_root: Path,
    record_a: Path,
    record_b: Path,
) -> dict[str, Any]:
    recipe_path = repo_root / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"
    tripole_path = repo_root / "packages/core/legoesm/grids/tripole.py"
    forcing_path = repo_root / "packages/ocean/legoesm/ocean/forcing/nemo_fld_read.py"
    require(recipe_path.is_file(), f"missing current recipe {recipe_path}")
    require(tripole_path.is_file(), f"missing tripolar grid {tripole_path}")
    current_recipe = recipe_path.read_text()
    dispatch = _dispatch_keys(current_recipe)
    current_functions = _function_names(current_recipe)

    historical_recipe = _git_blob(
        repo_root,
        HISTORICAL_REVISION,
        "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
    )
    historical_functions = _function_names(historical_recipe)
    historical_features = _historical_unmeasured_features(historical_recipe)
    require(historical_features == UNMEASURED_FEATURES, "historical ORCA2 guard drifted")

    configs_root = oracle_root / "cfgs"
    require(configs_root.is_dir(), f"missing oracle config root {configs_root}")
    oracle_configs = []
    for config in sorted(
        (path for path in configs_root.iterdir() if "ORCA2" in path.name),
        key=lambda path: path.name,
    ):
        config_binary = config / "BLD/bin/nemo.exe"
        has_binary = config_binary.is_file() and os.access(config_binary, os.X_OK)
        has_exp00 = (config / "EXP00/namelist_cfg").is_file()
        has_reference = (config / "EXPREF").is_dir()
        oracle_configs.append(
            {
                "binary_executable": has_binary,
                "exp00_namelist_present": has_exp00,
                "name": config.name,
                "path": str(config),
                "reference_config_present": has_reference,
                "status": (
                    "BUILT"
                    if has_binary
                    else "REFERENCE_CONFIG_NOT_BUILT"
                    if has_reference
                    else "MISSING_BUILD"
                ),
            }
        )
    require(oracle_configs, "no ORCA2 oracle configuration found")

    target = oracle_root / "cfgs/ORCA2_OMIP_L4"
    cpp = target / "cpp_ORCA2_OMIP_L4.fcm"
    namelist = target / "EXP00/namelist_cfg"
    binary = target / "BLD/bin/nemo.exe"
    stprk3 = target / "BLD/ppsrc/nemo/stprk3.f90"
    zdftke = target / "BLD/ppsrc/nemo/zdftke.f90"
    for path in (cpp, namelist, binary, stprk3, zdftke):
        require(path.exists(), f"incomplete read-only ORCA2 oracle build: {path}")
    cpp_text = cpp.read_text()
    namelist_text = namelist.read_text()
    require(
        all(key in cpp_text for key in ("key_si3", "key_qco", "key_vco_1d3d", "key_RK3")),
        "ORCA2 oracle keys do not match the native card",
    )
    require("CALL stp_RK3_stg( 3" in stprk3.read_text(), "compiled ORCA2 target lacks RK3 stage 3")
    require(
        "CALL tke_tke" in zdftke.read_text() and "CALL tke_avn" in zdftke.read_text(),
        "compiled ORCA2 target lacks the TKE call pair",
    )
    missing_exp00 = tuple(
        name for name in REQUIRED_DECK_INPUTS if not (target / "EXP00" / name).exists()
    )
    require(
        all(name.removesuffix(".nc") in namelist_text for name in REQUIRED_DECK_INPUTS),
        "ORCA2 namelist no longer names the required grid/T/S deck",
    )

    phase3 = _phase3_inventory(phase3_root)
    run_a = _complete_run(record_a)
    run_b = _complete_run(record_b)
    frames_a = _required_frames(record_a)
    frames_b = _required_frames(record_b)
    for path in frames_a + frames_b:
        require(path.is_file(), f"missing required native ORCA2 frame {path}")
    twin_rows = []
    for left, right in zip(frames_a, frames_b, strict=True):
        left_sha = sha256(left)
        right_sha = sha256(right)
        twin_rows.append(
            {
                "bytes": left.stat().st_size,
                "name": left.name,
                "sha256": left_sha,
                "twin_exact": left_sha == right_sha and left.stat().st_size == right.stat().st_size,
            }
        )
    require(all(row["twin_exact"] for row in twin_rows), "required kt=1--3/TKE twin differs")
    tke = _read_tke_schema(record_a / TKE_RECORD)
    require(tke["sha256_matches_pinned_receipt"], "TKE hash differs from admitted receipt")

    current_has_orca2 = "ORCA2-zps" in dispatch
    current_has_builder = "build_orca2_zps_card" in current_functions
    current_forcing_exists = forcing_path.is_file()
    current_execution_guard = "validate_nemo_testcase_card_for_execution" in current_functions
    historical_guard = "validate_nemo_testcase_card_for_execution" in historical_functions
    run_possible = bool(
        current_has_orca2
        and current_has_builder
        and current_forcing_exists
        and current_execution_guard
    )
    require(not run_possible, "current ORCA2 execution unexpectedly admitted; run kt=1--3 instead")

    record_complete = bool(
        run_a["completed"]
        and run_b["completed"]
        and len(twin_rows) == 13
        and all(row["twin_exact"] for row in twin_rows)
        and tke["exact_eof"]
        and tke["sha256_matches_pinned_receipt"]
    )
    return {
        "acquisition": {
            "needed": not record_complete,
            "reason": (
                "existing Phase-2v native record is complete and reusable"
                if record_complete
                else "no complete native record passed inventory"
            ),
        },
        "current_legoesm": {
            "first_failed_predicate": "current ORCA2 recipe dispatch is absent",
            "forcing_path": str(forcing_path),
            "generic_tripole_function_present": (
                "create_tripole_grid" in _function_names(tripole_path.read_text())
            ),
            "kt1_3_from_rest_possible": run_possible,
            "execution_guard_present": current_execution_guard,
            "orca2_builder_present": current_has_builder,
            "orca2_dispatch_present": current_has_orca2,
            "recipe_dispatch": list(dispatch),
            "status": "MISSING_NATIVE_RECIPE_AND_FORCING",
            "tripole_path": str(tripole_path),
        },
        "first_over_bar": {
            "boundary": "pre-kt1 execution admission",
            "status": "NOT_MEASURED",
            "why": "current ORCA2 recipe dispatch and exact forcing reader are absent",
        },
        "historical_legoesm": {
            "builder_present": "build_orca2_zps_card" in historical_functions,
            "execution_guard_present": historical_guard,
            "forcing_present": bool(
                _git_blob(
                    repo_root,
                    HISTORICAL_REVISION,
                    "packages/ocean/legoesm/ocean/forcing/nemo_fld_read.py",
                )
            ),
            "revision": HISTORICAL_REVISION,
            "status": "BUILT_NOT_RUNNABLE",
            "unmeasured_features": list(historical_features),
        },
        "nemo_oracle_build": {
            "all_orca2_configs": oracle_configs,
            "binary_bytes": binary.stat().st_size,
            "binary_executable": os.access(binary, os.X_OK),
            "binary_path": str(binary),
            "compiled_integrator": "RK3",
            "compiled_tke": True,
            "config_path": str(target),
            "exp00_missing_required_inputs": list(missing_exp00),
            "runnable_from_exp00_without_staging": not missing_exp00,
            "status": "BUILT_INPUT_STAGING_REQUIRED",
        },
        "phase3": phase3,
        "record": {
            "admitted_tke_schema": tke,
            "required_kt1_3_and_tke_twin_rows": twin_rows,
            "run_a": run_a,
            "run_b": run_b,
            "status": "COMPLETE_NATIVE_RECORD",
        },
        "schema": "nemo-testcase-l2-orca2-parallel-inventory-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2"),
    )
    parser.add_argument(
        "--phase3-root",
        type=Path,
        default=Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3"),
    )
    default_record = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/runs/"
        "variant_icebergs_off_phase2v_tke_a_10step_np2"
    )
    parser.add_argument("--record-a", type=Path, default=default_record)
    parser.add_argument(
        "--record-b",
        type=Path,
        default=Path(
            "/data/abyssal/dbalwada/nemo-testcases-l4/runs/"
            "variant_icebergs_off_phase2v_tke_b_10step_np2"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("missing-record",))
    args = parser.parse_args()
    if args.plant == "missing-record":
        args.record_a = args.record_a / "PLANTED_MISSING_RECORD"
    try:
        report = inventory(
            args.repo_root.resolve(),
            args.oracle_root,
            args.phase3_root.resolve(),
            args.record_a.resolve(),
            args.record_b.resolve(),
        )
    except InventoryError as exc:
        print(f"REFUSE: ORCA2 inventory: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "PASS: ORCA2 inventory; "
        f"kt1-3_possible={report['current_legoesm']['kt1_3_from_rest_possible']}; "
        f"acquisition_needed={report['acquisition']['needed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
