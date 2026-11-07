#!/usr/bin/env python3
"""Read-only admission inventory for the native ORCA2 fidelity lane.

This probe does not import or execute the ocean model.  It answers the earlier
question that must be closed before a kt=1--3 comparison is meaningful: does
the checked-out tree contain an executable native ORCA2 card, and is a usable
NEMO record already present?  The ``--plant`` modes bind missing-record,
missing-boundary, and producer-provenance refusal paths.
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
EXPECTED_STREAM_COUNT = 101
NX, NY, NZ, NTR = 94, 152, 31, 2
OWN_X, OWN_Y = NX - 4, NY - 4
ENTRY_FIELDS = ("T", "S", "u", "v", "ssh")
ENTRY_MAGIC = "NEMO_L1_ENTRY_1"
STAGE_MAGIC = "NEMO_L1_STAGE_1"
ZDF_MAGIC = "NEMO_L4_ZSH2_2"
ZDF_RECORD = "oracle_zdf_sh2_operands_kt00000002.bin"
ZDF_3D_FIELDS = (
    "sh2",
    "avm_k_pre",
    "avt_k_pre",
    "en_pre",
    "rn2",
    "rn2b",
    "u_Kbb",
    "u_Kmm",
    "v_Kbb",
    "v_Kmm",
    "e3uw_Kbb",
    "e3uw_Kmm",
    "e3vw_Kbb",
    "e3vw_Kmm",
    "umask",
    "vmask",
    "wumask",
    "wvmask",
    "gdepw_Kmm",
    "e3t_Kmm",
    "e3w_Kmm",
)
ZDF_2D_FIELDS = ("taum", "fr_i", "rCdU_bot", "mbkt_real")
ZDF_3D_ALLOCATIONS = ("reduced", "full", "reduced", "reduced") + ("full",) * 17
ZDF_2D_ALLOCATIONS = ("reduced", "full", "full", "full")
TKE_MAGIC = "NEMO_L4_TKEW_1"
TKE_RECORD = "oracle_tke_walk_kt00000002.bin"
TKE_HEADER_INTS = 15
TKE_3D_FIELD_NAMES = (
    "dissl_pre",
    "zpelc",
    "en_post_lc",
    "pdlr",
    "zdiag_pre_solve",
    "zd_lw_pre_solve",
    "zd_up_pre_solve",
    "en_rhs_pre_solve",
    "zdiag_after_forward",
    "zd_lw_after_forward",
    "en_post_solve",
    "en_post_etau",
    "mxlm",
    "mxld",
    "avm_post",
    "avt_post",
    "dissl_post",
)
TKE_2D_FIELD_NAMES = (
    "zice_fra",
    "zWlc2",
    "imlc_real",
    "zhlc",
    "zus3",
    "htau",
    "hm_i",
)
TKE_FIELD_NAMES = TKE_3D_FIELD_NAMES + TKE_2D_FIELD_NAMES
TKE_3D_FIELDS = len(TKE_3D_FIELD_NAMES)
TKE_2D_FIELDS = len(TKE_2D_FIELD_NAMES)
TKE_FIELDS = len(TKE_FIELD_NAMES)
ROUND101_TKE_CONTRACT = (
    "en_entry",
    "en_after_boundaries",
    "en_after_langmuir",
    "rhs_pre_sweep",
    "en_post_sweep",
)
PHASE2V_TKE_TO_ROUND101 = {
    "en_post_lc": "en_after_langmuir",
    "en_rhs_pre_solve": "rhs_pre_sweep",
    "en_post_solve": "en_post_sweep",
}
MISSING_BOUNDARY_VERDICT = "REUSABLE FOR entry/stage; TKE boundary frame MISSING"
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


def _functions_calling(source: str, callee: str) -> tuple[str, ...]:
    """Return function definitions containing a direct call to ``callee``."""
    callers = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == callee
            for child in ast.walk(node)
        ):
            callers.append(node.name)
    return tuple(sorted(callers))


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


def _entry_stage_specs(root: Path) -> tuple[tuple[Path, int, int | None], ...]:
    entries = tuple(
        (root / f"oracle_step_entry_kt{kt:08d}.bin", kt, None)
        for kt in range(1, 11)
    )
    stages = tuple(
        (root / f"oracle_stage_kt{kt:08d}_s{stage}.bin", kt, stage)
        for kt in range(1, 11)
        for stage in range(1, 4)
    )
    return entries + stages


def _read_entry_stage_schema(path: Path, kt: int, stage: int | None) -> dict[str, Any]:
    """Parse a native entry/stage header and require its exact payload size."""
    require(path.is_file(), f"missing native entry/stage record {path}")
    if stage is None:
        magic, header_ints = ENTRY_MAGIC, 8
        level = 1 if kt % 2 else 3
        expected_header = (1, kt, level, NX, NY, NZ, NTR, 64)
        kind = "entry"
    else:
        magic, header_ints = STAGE_MAGIC, 9
        level = 2 if stage == 2 else (3 if kt % 2 else 1)
        expected_header = (1, kt, stage, level, NX, NY, NZ, NTR, 64)
        kind = "stage"
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path.name}: truncated magic")
        observed_magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * header_ints)
        require(len(raw_header) == 4 * header_ints, f"{path.name}: truncated header")
        header = struct.unpack(f"={header_ints}i", raw_header)
    require(observed_magic == magic, f"{path.name}: unexpected magic {observed_magic!r}")
    require(header == expected_header, f"{path.name}: unexpected header {header}")
    payload = 4 * NX * NY * NZ + NX * NY
    expected_bytes = 16 + 4 * header_ints + 8 * payload
    require(path.stat().st_size == expected_bytes, f"{path.name}: exact EOF mismatch")
    return {
        "bytes": expected_bytes,
        "exact_eof": True,
        "fields": list(ENTRY_FIELDS),
        "header": list(header),
        "kind": kind,
        "magic": observed_magic,
        "name": path.name,
        "sha256": sha256(path),
    }


def _read_zdf_schema(path: Path) -> dict[str, Any]:
    """Parse the Phase-2v kt=2 ZDF operand stream that carries native ``en_pre``."""
    require(path.is_file(), f"missing native ZDF operand record {path}")
    field_count = len(ZDF_3D_FIELDS) + len(ZDF_2D_FIELDS)
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path.name}: truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(13 * 4)
        require(len(raw_header) == 13 * 4, f"{path.name}: truncated header")
        header = struct.unpack("=13i", raw_header)
        raw_extents = handle.read(field_count * 3 * 4)
        require(
            len(raw_extents) == field_count * 3 * 4,
            f"{path.name}: truncated extent table",
        )
        flat = struct.unpack(f"={field_count * 3}i", raw_extents)
    (
        version,
        kt,
        kbb,
        kmm,
        krhs,
        jpi,
        jpj,
        jpk,
        real_bits,
        n3,
        n2,
        payload,
        armed,
    ) = header
    require(magic == ZDF_MAGIC, f"{path.name}: unexpected magic {magic!r}")
    require(
        (version, kt, kbb, kmm, krhs, jpi, jpj, jpk, real_bits, n3, n2, armed)
        == (2, 2, 3, 3, 1, NX, NY, NZ, 64, len(ZDF_3D_FIELDS), len(ZDF_2D_FIELDS), 1),
        f"{path.name}: unexpected header {header}",
    )
    expected_extents = tuple(
        (OWN_X, OWN_Y, NZ) if allocation == "reduced" else (NX, NY, NZ)
        for allocation in ZDF_3D_ALLOCATIONS
    ) + tuple(
        (OWN_X, OWN_Y, 1) if allocation == "reduced" else (NX, NY, 1)
        for allocation in ZDF_2D_ALLOCATIONS
    )
    extents = tuple(tuple(flat[index : index + 3]) for index in range(0, len(flat), 3))
    require(extents == expected_extents, f"{path.name}: extent table mismatch")
    derived_payload = sum(a * b * c for a, b, c in expected_extents)
    require(payload == derived_payload, f"{path.name}: payload count mismatch")
    expected_bytes = 16 + 13 * 4 + field_count * 3 * 4 + payload * 8
    require(path.stat().st_size == expected_bytes, f"{path.name}: exact EOF mismatch")
    return {
        "bytes": expected_bytes,
        "exact_eof": True,
        "fields_2d": list(ZDF_2D_FIELDS),
        "fields_3d": list(ZDF_3D_FIELDS),
        "header": list(header),
        "magic": magic,
        "name": path.name,
        "sha256": sha256(path),
    }


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
        "fields_2d": list(TKE_2D_FIELD_NAMES),
        "fields_3d": list(TKE_3D_FIELD_NAMES),
        "header": list(header),
        "magic": magic,
        "path": str(path),
        "pinned_sha256": PINNED_TKE_SHA256,
        "sha256": actual_sha,
        "sha256_matches_pinned_receipt": actual_sha == PINNED_TKE_SHA256,
    }


def _tke_contract_map(tke: dict[str, Any], zdf: dict[str, Any]) -> dict[str, Any]:
    """Map every Phase-2v TKE name onto the newer statement contract."""
    tke_names = tuple(tke["fields_3d"]) + tuple(tke["fields_2d"])
    zdf_names = tuple(zdf["fields_3d"]) + tuple(zdf["fields_2d"])
    require(tke_names == TKE_FIELD_NAMES, "TKE schema field-name order drifted")
    require("en_pre" in zdf_names, "native ZDF entry no longer carries en_pre")
    mappings = [
        {
            "native_field": "en_pre",
            "record": ZDF_RECORD,
            "round101_field": "en_entry",
            "status": "PRESENT_NATIVE",
        }
    ]
    mappings.extend(
        {
            "native_field": native,
            "record": TKE_RECORD,
            "round101_field": contract,
            "status": "PRESENT_NATIVE",
        }
        for native, contract in PHASE2V_TKE_TO_ROUND101.items()
    )
    available = {row["round101_field"] for row in mappings}
    missing = [field for field in ROUND101_TKE_CONTRACT if field not in available]
    per_field = [
        {
            "classification": (
                "ROUND101_STATEMENT_EQUIVALENT"
                if name in PHASE2V_TKE_TO_ROUND101
                else "SUPPLEMENTAL_PHASE2V_OPERAND"
            ),
            "phase2v_field": name,
            "round101_field": PHASE2V_TKE_TO_ROUND101.get(name),
        }
        for name in tke_names
    ]
    return {
        "contract": list(ROUND101_TKE_CONTRACT),
        "mappings": mappings,
        "missing": missing,
        "phase2v_field_map": per_field,
        "status": "MISSING" if missing else "COMPLETE",
    }


def _reuse_decision(
    *,
    stream_counts: tuple[int, int],
    schemas_valid: bool,
    entry_stage_twins_exact: bool,
    producer_hashes_match: bool,
    available_tke_fields: set[str],
) -> dict[str, Any]:
    """Evaluate the acquisition decision from explicit, plantable predicates."""
    missing = [field for field in ROUND101_TKE_CONTRACT if field not in available_tke_fields]
    entry_stage_reusable = bool(
        stream_counts == (EXPECTED_STREAM_COUNT, EXPECTED_STREAM_COUNT)
        and schemas_valid
        and entry_stage_twins_exact
        and producer_hashes_match
    )
    if not entry_stage_reusable:
        verdict = "NOT REUSABLE FOR entry/stage"
    elif missing == ["en_after_boundaries"]:
        verdict = MISSING_BOUNDARY_VERDICT
    elif missing:
        verdict = "REUSABLE FOR entry/stage; TKE contract fields MISSING"
    else:
        verdict = "REUSABLE FOR entry/stage and TKE statement boundaries"
    return {
        "acquisition_needed": not entry_stage_reusable or bool(missing),
        "entry_stage_reusable": entry_stage_reusable,
        "missing_tke_fields": missing,
        "verdict": verdict,
    }


def _run_decision_plant(
    plant: str,
    *,
    stream_counts: tuple[int, int],
    schemas_valid: bool,
    entry_stage_twins_exact: bool,
    producer_hashes_match: bool,
) -> None:
    """Prove the boundary and provenance predicates can change admission."""
    complete = set(ROUND101_TKE_CONTRACT)
    admitted = _reuse_decision(
        stream_counts=stream_counts,
        schemas_valid=schemas_valid,
        entry_stage_twins_exact=entry_stage_twins_exact,
        producer_hashes_match=producer_hashes_match,
        available_tke_fields=complete,
    )
    if plant == "missing-boundary":
        planted = _reuse_decision(
            stream_counts=stream_counts,
            schemas_valid=schemas_valid,
            entry_stage_twins_exact=entry_stage_twins_exact,
            producer_hashes_match=producer_hashes_match,
            available_tke_fields=complete - {"en_after_boundaries"},
        )
    elif plant == "provenance-mismatch":
        planted = _reuse_decision(
            stream_counts=stream_counts,
            schemas_valid=schemas_valid,
            entry_stage_twins_exact=entry_stage_twins_exact,
            producer_hashes_match=False,
            available_tke_fields=complete,
        )
    else:
        raise InventoryError(f"unknown decision plant {plant!r}")
    require(not admitted["acquisition_needed"], f"{plant} control baseline was not admitted")
    require(planted["acquisition_needed"], f"{plant} did not flip acquisition decision")
    require(planted["verdict"] != admitted["verdict"], f"{plant} did not flip verdict")
    raise InventoryError(
        f"{plant} plant flipped decision to {planted['verdict']}"
    )


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
    plant: str | None = None,
) -> dict[str, Any]:
    recipe_path = repo_root / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"
    tripole_path = repo_root / "packages/core/legoesm/grids/tripole.py"
    forcing_path = repo_root / "packages/ocean/legoesm/ocean/forcing/nemo_fld_read.py"
    recipes_path = repo_root / "packages/ocean/legoesm/ocean/recipes.py"
    nemo_recipe_path = repo_root / "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py"
    run_omip_path = repo_root / "scripts/run/run_omip.py"
    core2_builder_path = repo_root / "scripts/data/build_core2_nyf_zarr.py"
    core2_loader_path = repo_root / "packages/ocean/legoesm/ocean/forcing/core2.py"
    jra55_path = repo_root / "scripts/data/prepare_omip_forcing.py"
    native_fields_path = (
        repo_root / "packages/ocean/legoesm/ocean/forcing/nemo_native_fields.py"
    )
    vertical_path = repo_root / "packages/ocean/legoesm/ocean/vertical.py"
    experiment_root = repo_root / "scripts/experiment"
    acquisition_path = (
        repo_root
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh"
    )
    require(recipe_path.is_file(), f"missing current recipe {recipe_path}")
    require(tripole_path.is_file(), f"missing tripolar grid {tripole_path}")
    for path in (
        recipes_path,
        nemo_recipe_path,
        run_omip_path,
        core2_builder_path,
        core2_loader_path,
        jra55_path,
        native_fields_path,
        vertical_path,
        acquisition_path,
    ):
        require(path.is_file(), f"missing current generic ORCA2 asset {path}")
    require(experiment_root.is_dir(), f"missing experiment directory {experiment_root}")
    current_recipe = recipe_path.read_text()
    dispatch = _dispatch_keys(current_recipe)
    current_functions = _function_names(current_recipe)
    guard_callers = _functions_calling(current_recipe, "validate_nemo_testcase_card")

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
    names_a = {path.name for path in record_a.glob("oracle_*.bin") if path.is_file()}
    names_b = {path.name for path in record_b.glob("oracle_*.bin") if path.is_file()}
    stream_counts = (len(names_a), len(names_b))
    require(
        stream_counts == (EXPECTED_STREAM_COUNT, EXPECTED_STREAM_COUNT),
        f"native stream counts {stream_counts} != {(EXPECTED_STREAM_COUNT,) * 2}",
    )
    require(names_a == names_b, "A/B native stream inventories differ")

    producer_fields = ("binary_sha256", "deck_manifest_sha256", "input_manifest_sha256")
    matched_producer_hashes = {name: run_a[name] for name in producer_fields}
    producer_hashes_match = all(run_a[name] == run_b[name] for name in producer_fields)

    entry_stage_rows = []
    for (left, kt, stage), (right, right_kt, right_stage) in zip(
        _entry_stage_specs(record_a), _entry_stage_specs(record_b), strict=True
    ):
        require((kt, stage) == (right_kt, right_stage), "A/B frame specification drift")
        left_schema = _read_entry_stage_schema(left, kt, stage)
        right_schema = _read_entry_stage_schema(right, kt, stage)
        require(
            left_schema["header"] == right_schema["header"],
            f"{left.name}: A/B schema headers differ",
        )
        entry_stage_rows.append(
            {
                **left_schema,
                "twin_exact": (
                    left_schema["sha256"] == right_schema["sha256"]
                    and left_schema["bytes"] == right_schema["bytes"]
                ),
            }
        )
    entry_stage_twins_exact = all(row["twin_exact"] for row in entry_stage_rows)

    zdf = _read_zdf_schema(record_a / ZDF_RECORD)
    zdf_b = _read_zdf_schema(record_b / ZDF_RECORD)
    require(zdf["header"] == zdf_b["header"], "A/B ZDF schema headers differ")
    zdf_twin_exact = zdf["sha256"] == zdf_b["sha256"]
    tke = _read_tke_schema(record_a / TKE_RECORD)
    tke_b = _read_tke_schema(record_b / TKE_RECORD)
    tke_twin_exact = (
        tke["sha256"] == tke_b["sha256"] and tke["bytes"] == tke_b["bytes"]
    )
    require(tke["sha256_matches_pinned_receipt"], "TKE hash differs from admitted receipt")
    require(tke_b["sha256_matches_pinned_receipt"], "twin TKE hash differs from receipt")
    tke_contract = _tke_contract_map(tke, zdf)
    available_tke_fields = {
        row["round101_field"] for row in tke_contract["mappings"]
    }
    schemas_valid = bool(
        len(entry_stage_rows) == 40
        and all(row["exact_eof"] for row in entry_stage_rows)
        and zdf["exact_eof"]
        and tke["exact_eof"]
    )
    if plant in ("missing-boundary", "provenance-mismatch"):
        _run_decision_plant(
            plant,
            stream_counts=stream_counts,
            schemas_valid=schemas_valid,
            entry_stage_twins_exact=entry_stage_twins_exact,
            producer_hashes_match=producer_hashes_match,
        )
    require(producer_hashes_match, "A/B producer provenance hashes differ")
    require(entry_stage_twins_exact, "native entry/stage A/B twins differ")
    require(zdf_twin_exact, "native ZDF en_entry companion A/B twins differ")
    require(tke_twin_exact, "native TKE A/B twins differ")
    decision = _reuse_decision(
        stream_counts=stream_counts,
        schemas_valid=schemas_valid,
        entry_stage_twins_exact=entry_stage_twins_exact,
        producer_hashes_match=producer_hashes_match,
        available_tke_fields=available_tke_fields,
    )
    require(
        decision["verdict"] == MISSING_BOUNDARY_VERDICT,
        f"unexpected native-record verdict {decision['verdict']}",
    )

    current_has_orca2 = "ORCA2-zps" in dispatch
    current_has_builder = "build_orca2_zps_card" in current_functions
    current_forcing_exists = forcing_path.is_file()
    current_execution_guard = "validate_nemo_testcase_card" in current_functions
    require(
        set(guard_callers)
        == {
            "build_gyre_zco_card",
            "build_lock_exchange_zco_card",
            "build_overflow_zps_card",
        },
        f"current testcase guard callers drifted: {guard_callers}",
    )
    historical_guard = "validate_nemo_testcase_card_for_execution" in historical_functions
    run_possible = bool(
        current_has_orca2
        and current_has_builder
        and current_forcing_exists
        and current_execution_guard
    )
    require(not run_possible, "current ORCA2 execution unexpectedly admitted; run kt=1--3 instead")

    recipes_text = recipes_path.read_text()
    nemo_recipe_functions = _function_names(nemo_recipe_path.read_text())
    run_omip_text = run_omip_path.read_text()
    native_field_functions = _function_names(native_fields_path.read_text())
    vertical_functions = _function_names(vertical_path.read_text())
    generic_assets = {
        "catalog_recipes": {
            "legoesm_nemo_like_v1": "legoesm_nemo_like_v1" in recipes_text,
            "omip_nemo_match_tripole_v1": "omip_nemo_match_tripole_v1" in recipes_text,
        },
        "core2_builder": core2_builder_path.is_file(),
        "core2_loader": "load_core2_nyf" in _function_names(core2_loader_path.read_text()),
        "eorca1_native_readers": all(
            name in native_field_functions
            for name in (
                "load_nemo_sss_restoring_climatology",
                "load_nemo_monthly_init_ts",
                "load_nemo_ice_init",
            )
        ),
        "jra55_preparation": "build_jra55_cache" in jra55_path.read_text(),
        "nemo_rest_constructor": "build_nemo_rest_recipe" in nemo_recipe_functions,
        "nemo_tke_config": "_nemo_tke_config" in nemo_recipe_functions,
        "qco_partial_cell_arithmetic": all(
            name in vertical_functions
            for name in (
                "nemo_qco_live_face_geometry_from_operands",
                "create_partial_cell_coordinate",
            )
        ),
        "tripole_mesh_override": (
            "--tripole-mesh" in run_omip_text
            and "create_tripole_grid" in run_omip_text
            and "tripole_partial_cells" in run_omip_text
        ),
    }
    experiment_orca2_mentions = sorted(
        str(path.relative_to(repo_root))
        for path in experiment_root.rglob("*")
        if path.is_file()
        and any(
            token in path.read_text(errors="replace").lower()
            for token in ("orca2", "orca_r2", "eorca2")
        )
    )
    require(all(generic_assets["catalog_recipes"].values()), "named ocean recipes missing")
    require(
        all(value for key, value in generic_assets.items() if key != "catalog_recipes"),
        "generic from-rest tripolar asset inventory is incomplete",
    )
    require(
        not experiment_orca2_mentions,
        f"unexpected exact ORCA2 experiment assets: {experiment_orca2_mentions}",
    )
    return {
        "acquisition": {
            "needed": decision["acquisition_needed"],
            "reason": decision["verdict"],
            "script": str(acquisition_path),
        },
        "current_legoesm": {
            "first_failed_predicate": "current ORCA2 recipe dispatch is absent",
            "forcing_path": str(forcing_path),
            "generic_from_rest_tripolar_assets": generic_assets,
            "generic_tripole_function_present": (
                "create_tripole_grid" in _function_names(tripole_path.read_text())
            ),
            "kt1_3_from_rest_possible": run_possible,
            "execution_guard_present": current_execution_guard,
            "execution_guard_callers": list(guard_callers),
            "experiment_orca2_assets": experiment_orca2_mentions,
            "orca2_builder_present": current_has_builder,
            "orca2_dispatch_present": current_has_orca2,
            "recipe_dispatch": list(dispatch),
            "status": (
                "EXACT_CERTIFIED_ORCA2_TESTCASE_ASSEMBLY_ABSENT; "
                "GENERIC_EXTERNAL_MESH_TRIPOLE_FROM_REST_CONSTRUCTIBLE"
            ),
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
            "decision": decision,
            "entry_stage_schema_count": len(entry_stage_rows),
            "entry_stage_schemas": entry_stage_rows,
            "entry_stage_schemas_valid": schemas_valid,
            "entry_stage_twins_exact": entry_stage_twins_exact,
            "matched_producer_hashes": matched_producer_hashes,
            "producer_hashes_match": producer_hashes_match,
            "run_a": run_a,
            "run_b": run_b,
            "stream_inventory_matches": names_a == names_b,
            "stream_counts": list(stream_counts),
            "tke_contract": tke_contract,
            "tke_twin_exact": tke_twin_exact,
            "zdf_entry_schema": zdf,
            "zdf_entry_twin_exact": zdf_twin_exact,
            "status": decision["verdict"],
        },
        "schema": "nemo-testcase-l2-orca2-parallel-inventory-v2",
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
    parser.add_argument(
        "--plant",
        choices=("missing-record", "missing-boundary", "provenance-mismatch"),
    )
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
            args.plant,
        )
    except InventoryError as exc:
        print(f"REFUSE: ORCA2 inventory: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: ORCA2 inventory; {report['record']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
