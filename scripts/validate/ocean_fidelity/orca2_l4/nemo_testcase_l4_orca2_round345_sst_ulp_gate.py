#!/usr/bin/env python3
"""Stage, audit, and admit the Decision-188 one-ULP K0 NEMO arm."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _path in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from legoesm.ocean.fidelity.provenance import git_sha, worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round339_common as common,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round342_s4_year_gate as r342,
)

GateError = common.GateError
require = common.require

DOMAIN_FILE = "ORCA_R2_zps_domcfg.nc"
RESTART_VARIABLE = "tn"
RESTART_KT = 8
FIRST_RUN_STEP = 9
PERTURBATION_MANIFEST = "restart_sst_ulp_perturbation.json"
CONTROL_INPUT_MANIFEST = "restart_unperturbed_control_input.json"
INPUT_AUDIT = "round342_arm_a_input_identity.json"
PERTURBED_ARM_NAME = "orca2_k0_year1_sst_plus_1ulp_np2"
CONTROL_ARM_NAME = "orca2_k0_year1_restart_control_np2"
DELAY_RESTART_VARIABLES = frozenset({"DELAY_re8_fwb2", "DELAY_im8_fwb2"})
BASELINE_ARM = (
    Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
    / "round342/s4_year/orca2_k0_year1_a_np2"
)
BASELINE_ADMISSION = BASELINE_ARM.parent / "round342_s4_year_arm_a_admission.json"
PREFLIGHT_SCRATCH_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work"
)
DEFAULT_RUN_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year"
)
DEFAULT_EVIDENCE_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345"
)


def _wet_surface_mask(domain_file: Path) -> np.ndarray:
    require(domain_file.is_file(), f"missing ORCA2 domain file: {domain_file}")
    with netCDF4.Dataset(domain_file, "r") as dataset:
        require("top_level" in dataset.variables,
                f"top_level missing from {domain_file}")
        top = np.asarray(dataset.variables["top_level"][:], dtype=np.int64)
    require(top.ndim == 2, f"top_level must be 2-D, got {top.shape}")
    return top > 0


def _same_array_bits(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _rank_wet_mask(domain_file: Path, rank: int, shape: tuple[int, int]) -> np.ndarray:
    wet = _wet_surface_mask(domain_file)
    require(wet.shape[0] == shape[0] and wet.shape[1] == 2 * shape[1],
            f"domain wet mask {wet.shape} is not two conforming rank slabs of {shape}")
    return wet[:, rank * shape[1]:(rank + 1) * shape[1]]


def verify_restart_pair_perturbation(
    source_arm: Path,
    perturbed_arm: Path,
) -> dict[str, Any]:
    """Prove only fp64 wet-surface ``tn`` differs by +1 ULP in the kt=8 pair."""
    rank_rows = []
    selected_total = 0
    changed_total = 0
    for rank in (0, 1):
        name = f"ORCA2_{RESTART_KT:08d}_restart_{rank:04d}.nc"
        source_file = source_arm / name
        perturbed_file = perturbed_arm / name
        require(source_file.is_file(), f"missing admitted day-1 restart: {source_file}")
        require(perturbed_file.is_file() and not perturbed_file.is_symlink(),
                f"perturbed day-1 restart must be a regular file: {perturbed_file}")
        with netCDF4.Dataset(source_file) as source, netCDF4.Dataset(perturbed_file) as perturbed:
            require(tuple(source.ncattrs()) == tuple(perturbed.ncattrs()),
                    f"{name}: global attribute names changed")
            for attr in source.ncattrs():
                require(_same_array_bits(np.asarray(source.getncattr(attr)),
                                         np.asarray(perturbed.getncattr(attr))),
                        f"{name}: global attribute {attr} changed")
            require(tuple(source.dimensions) == tuple(perturbed.dimensions),
                    f"{name}: dimension inventory changed")
            require(tuple(source.variables) == tuple(perturbed.variables),
                    f"{name}: variable inventory changed")
            for dim_name in source.dimensions:
                require(len(source.dimensions[dim_name]) == len(perturbed.dimensions[dim_name]),
                        f"{name}: dimension {dim_name} changed")
            for variable_name in source.variables:
                src_var = source.variables[variable_name]
                dst_var = perturbed.variables[variable_name]
                src_var.set_auto_maskandscale(False)
                dst_var.set_auto_maskandscale(False)
                require(src_var.dtype == dst_var.dtype and src_var.dimensions == dst_var.dimensions,
                        f"{name}:{variable_name} dtype or dimensions changed")
                require(tuple(src_var.ncattrs()) == tuple(dst_var.ncattrs()),
                        f"{name}:{variable_name} attribute names changed")
                for attr in src_var.ncattrs():
                    require(_same_array_bits(np.asarray(src_var.getncattr(attr)),
                                             np.asarray(dst_var.getncattr(attr))),
                            f"{name}:{variable_name}:{attr} changed")
                source_values = np.asarray(src_var[:])
                perturbed_values = np.asarray(dst_var[:])
                if variable_name != RESTART_VARIABLE:
                    require(_same_array_bits(source_values, perturbed_values),
                            f"{name}: unselected variable {variable_name} changed")
                    continue
                require(src_var.dtype == np.dtype("float64") and src_var.ndim == 4
                        and src_var.shape[0] == 1 and src_var.shape[1] >= 1,
                        f"{name}:{variable_name} is not a double-precision 3-D restart state")
                expected = source_values.copy()
                wet = _rank_wet_mask(
                    perturbed_arm / DOMAIN_FILE, rank, tuple(src_var.shape[-2:])
                )
                selected = wet & np.isfinite(source_values[0, 0])
                expected[0, 0][selected] = np.nextafter(
                    source_values[0, 0][selected], np.float64(np.inf)
                )
                require(_same_array_bits(expected, perturbed_values),
                        f"{name}: restart SST is not the exact +1-fp64-ULP pattern")
                selected_count = int(np.count_nonzero(selected))
                changed_count = int(np.count_nonzero(
                    source_values[0, 0][selected].view(np.uint64)
                    != perturbed_values[0, 0][selected].view(np.uint64)
                ))
                require(selected_count > 0 and changed_count == selected_count,
                        f"{name}: selected {selected_count} cells but changed {changed_count}")
                selected_total += selected_count
                changed_total += changed_count
        source_sha = common.sha256(source_file)
        perturbed_sha = common.sha256(perturbed_file)
        require(source_sha != perturbed_sha, f"{name}: source and perturbed digests are equal")
        rank_rows.append({
            "rank": rank,
            "source_path": str(source_file.resolve()),
            "source_sha256": source_sha,
            "perturbed_file": name,
            "perturbed_sha256": perturbed_sha,
        })
    return {
        "status": "PASS_R345_DAY1_RESTART_SST_PLUS_ONE_FP64_ULP",
        "source_arm": str(source_arm.resolve()),
        "perturbed_arm": ".",
        "restart_kt": RESTART_KT,
        "first_run_step": FIRST_RUN_STEP,
        "variable": RESTART_VARIABLE,
        "dtype": "float64",
        "selection": "tn[0, 0, wet_y, wet_x] in each rank-local restart",
        "direction": "nextafter(value, +infinity)",
        "selected_values": selected_total,
        "changed_values": changed_total,
        "exact_ulp_distance": 1,
        "unselected_restart_values_and_metadata_bit_identical": True,
        "rank_files": rank_rows,
    }


def verify_unperturbed_restart_pair(
    source_arm: Path,
    control_arm: Path,
) -> dict[str, Any]:
    """Prove the control starts from the exact admitted arm-A kt=8 restart bytes."""
    rows = []
    for rank in (0, 1):
        name = f"ORCA2_{RESTART_KT:08d}_restart_{rank:04d}.nc"
        source = source_arm / name
        control = control_arm / name
        require(source.is_file(), f"missing admitted day-1 restart: {source}")
        require(control.is_file() and not control.is_symlink(),
                f"control day-1 restart must be a regular file: {control}")
        source_sha = common.sha256(source)
        control_sha = common.sha256(control)
        require(control_sha == source_sha,
                f"{name}: unperturbed control input differs from admitted arm A")
        rows.append({
            "rank": rank,
            "source_path": str(source.resolve()),
            "source_sha256": source_sha,
            "control_file": name,
            "control_sha256": control_sha,
        })
    return {
        "status": "PASS_R345_DAY1_RESTART_UNPERTURBED_CONTROL",
        "source_arm": str(source_arm.resolve()),
        "control_arm": ".",
        "restart_kt": RESTART_KT,
        "first_run_step": FIRST_RUN_STEP,
        "restart_pair_byte_identical": True,
        "rank_files": rows,
    }


def copy_unperturbed_day1_restart_pair(
    run_dir: Path,
    source_arm: Path = BASELINE_ARM,
) -> dict[str, Any]:
    for rank in (0, 1):
        name = f"ORCA2_{RESTART_KT:08d}_restart_{rank:04d}.nc"
        shutil.copy2(source_arm / name, run_dir / name)
    audit = verify_unperturbed_restart_pair(source_arm, run_dir)
    (run_dir / CONTROL_INPUT_MANIFEST).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def verify_control_input_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / CONTROL_INPUT_MANIFEST
    require(manifest_path.is_file(), f"missing control-input manifest: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    measured = verify_unperturbed_restart_pair(Path(recorded["source_arm"]), run_dir)
    require(recorded == measured, "recorded control-input manifest does not match re-audit")
    return measured


def perturb_day1_restart_pair(run_dir: Path, source_arm: Path = BASELINE_ARM) -> dict[str, Any]:
    """Copy admitted kt=8 restarts and perturb only their wet-surface fp64 ``tn``."""
    wet_global = _wet_surface_mask(run_dir / DOMAIN_FILE)
    for rank in (0, 1):
        name = f"ORCA2_{RESTART_KT:08d}_restart_{rank:04d}.nc"
        source = source_arm / name
        target = run_dir / name
        shutil.copy2(source, target)
        with netCDF4.Dataset(target, "r+") as dataset:
            variable = dataset.variables[RESTART_VARIABLE]
            variable.set_auto_maskandscale(False)
            require(variable.dtype == np.dtype("float64") and variable.shape[0] == 1,
                    f"unexpected {name}:{RESTART_VARIABLE} storage")
            wet = _rank_wet_mask(run_dir / DOMAIN_FILE, rank, tuple(variable.shape[-2:]))
            expected_wet = wet_global[:, rank * wet.shape[1]:(rank + 1) * wet.shape[1]]
            require(np.array_equal(wet, expected_wet),
                    f"{name}: rank wet-mask split is inconsistent")
            surface = np.asarray(variable[0, 0], dtype=np.float64)
            selected = wet & np.isfinite(surface)
            surface[selected] = np.nextafter(surface[selected], np.float64(np.inf))
            variable[0, 0] = surface
    audit = verify_restart_pair_perturbation(source_arm, run_dir)
    (run_dir / PERTURBATION_MANIFEST).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def verify_perturbation_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / PERTURBATION_MANIFEST
    require(manifest_path.is_file(), f"missing perturbation manifest: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    measured = verify_restart_pair_perturbation(Path(recorded["source_arm"]), run_dir)
    require(recorded == measured, "recorded perturbation manifest does not match re-audit")
    return measured


def _patch_restart_namelist(source: Path, destination: Path, expected_steps: int) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8")
    block_match = re.search(r"(?ims)^\s*&namrun\b.*?^\s*/\s*$", text)
    require(block_match is not None, f"missing &namrun block in {source}")
    block = block_match.group(0)
    for name in ("ln_rstart", "nn_rstctl", "cn_ocerst_in"):
        block = re.sub(
            rf"(?im)^\s*{name}\s*=.*$", "", block
        )
    block, count_start = re.subn(
        r"(?im)(\bnn_it000\s*=\s*)\d+", rf"\g<1>{FIRST_RUN_STEP}", block, count=1
    )
    block, count_end = re.subn(
        r"(?im)(\bnn_itend\s*=\s*)\d+", rf"\g<1>{expected_steps}", block, count=1
    )
    block, count_stock = re.subn(
        r"(?im)(\bnn_stock\s*=\s*)\d+", rf"\g<1>{r342.DAILY_STOCK_STRIDE}", block, count=1
    )
    require((count_start, count_end, count_stock) == (1, 1, 1),
            "failed to patch restart step controls in &namrun")
    insertion = (
        "   ln_rstart   = .true.\n"
        "   nn_rstctl    = 2\n"
        f'   cn_ocerst_in = "ORCA2_{RESTART_KT:08d}_restart"\n'
    )
    block = re.sub(r"(?m)^\s*/\s*$", insertion + "/", block, count=1)
    text = text[:block_match.start()] + block + text[block_match.end():]
    destination.write_text(text, encoding="utf-8")
    common.validate_group_membership(destination, {
        "ln_rstart": "namrun", "nn_rstctl": "namrun", "cn_ocerst_in": "namrun",
        "nn_it000": "namrun", "nn_itend": "namrun", "nn_stock": "namrun",
    })
    return {
        "source": str(source.resolve()),
        "source_sha256": common.sha256(source),
        "sha256": common.sha256(destination),
        "ln_rstart": True,
        "nn_rstctl": 2,
        "cn_ocerst_in": f"ORCA2_{RESTART_KT:08d}_restart",
        "nn_it000": FIRST_RUN_STEP,
        "nn_itend": expected_steps,
        "nn_stock": r342.DAILY_STOCK_STRIDE,
    }


def _manifest_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            names.append(parts[1])
    return names


def verify_arm_a_inputs_untouched(
    run_dir: Path,
    baseline_arm: Path = BASELINE_ARM,
) -> dict[str, Any]:
    """Prove every staged deck/forcing/damping input except restart controls equals arm A."""
    names: set[str] = {"ice_cover_clim.nc"}
    for manifest_name in ("deck_files.sha256", "input_files.sha256"):
        names.update(_manifest_names(r342.SOURCE_RUN / manifest_name))
    names.discard("namelist_cfg")
    rows = {}
    for name in sorted(names):
        baseline = baseline_arm / name
        staged = run_dir / name
        require(baseline.exists() and staged.exists(), f"missing arm-A identity input {name}")
        baseline_sha = common.sha256(baseline)
        staged_sha = common.sha256(staged)
        require(staged_sha == baseline_sha, f"staged input {name} differs from admitted arm A")
        require(baseline.is_symlink() == staged.is_symlink(),
                f"staged input {name} changed regular/symlink type")
        if baseline.is_symlink():
            require(os.readlink(baseline) == os.readlink(staged),
                    f"staged input {name} changed symlink target")
        rows[name] = baseline_sha
    result = {
        "status": "PASS_R345_DECK_FORCING_DAMPING_EQUAL_ARM_A",
        "baseline_arm": str(baseline_arm.resolve()),
        "checked_file_count": len(rows),
        "files_sha256": rows,
        "forcing_and_damping_untouched": True,
    }
    (run_dir / INPUT_AUDIT).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def verify_baseline_admission(baseline_arm: Path = BASELINE_ARM) -> dict[str, Any]:
    admission = json.loads(BASELINE_ADMISSION.read_text(encoding="utf-8"))
    manifest = json.loads((baseline_arm / "producer_manifest.json").read_text(encoding="utf-8"))
    require(admission.get("admitted") is True, "round-342 arm A is not admitted")
    require(Path(admission["run_dir"]).resolve() == baseline_arm.resolve(),
            "round-342 admission names a different arm A")
    require(admission["producer_commit"] == manifest["producer_commit"],
            "round-342 admission/manifest producer commits differ")
    actual_binary = common.sha256(baseline_arm / "nemo")
    require(admission["binary_sha256"] == manifest["binary_sha256"] == actual_binary,
            "round-342 admission/manifest/actual binary digests differ")
    day1_sha256 = {}
    for rank in (0, 1):
        key = f"rank_{rank:04d}"
        restart = baseline_arm / f"ORCA2_{RESTART_KT:08d}_restart_{rank:04d}.nc"
        actual_restart = common.sha256(restart)
        require(actual_restart == admission["first_day_restarts"][key]["sha256"],
                f"round-342 admitted day-1 rank {rank} restart digest changed")
        day1_sha256[key] = actual_restart
    return {
        "admission_path": str(BASELINE_ADMISSION.resolve()),
        "producer_commit": manifest["producer_commit"],
        "binary_sha256": actual_binary,
        "day1_restart_sha256": day1_sha256,
    }


def _stage_restart_arm(
    run_dir: Path,
    *,
    perturb: bool,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    marker = run_dir / ".round345_attempt"
    marker_text = marker.read_text(encoding="utf-8") if marker.is_file() else None
    if marker_text is not None:
        marker.unlink()
    try:
        staged = r342.stage_s4_year_run_dir(
            run_dir,
            ice_cover_path,
            build_cfg=build_cfg,
            nn_itend=expected_steps,
            nn_stock=r342.DAILY_STOCK_STRIDE,
        )
    finally:
        if marker_text is not None:
            marker.write_text(marker_text, encoding="utf-8")
    baseline = verify_baseline_admission()
    restart_namelist = _patch_restart_namelist(
        BASELINE_ARM / "staged.namelist_cfg", run_dir / "namelist_cfg", expected_steps
    )
    shutil.copy2(run_dir / "namelist_cfg", run_dir / "staged.namelist_cfg")
    input_identity = verify_arm_a_inputs_untouched(run_dir)
    restart_audit = (
        perturb_day1_restart_pair(run_dir)
        if perturb
        else copy_unperturbed_day1_restart_pair(run_dir)
    )
    staged.update({
        "baseline_admission": baseline,
        "restart_namelist": restart_namelist,
        "arm_a_input_identity": input_identity,
        (
            "restart_sst_perturbation"
            if perturb
            else "unperturbed_restart_control_input"
        ): restart_audit,
    })
    return staged


def stage_perturbed_arm(
    run_dir: Path,
    *,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    return _stage_restart_arm(
        run_dir, perturb=True, ice_cover_path=ice_cover_path,
        build_cfg=build_cfg, expected_steps=expected_steps,
    )


def stage_control_arm(
    run_dir: Path,
    *,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    return _stage_restart_arm(
        run_dir, perturb=False, ice_cover_path=ice_cover_path,
        build_cfg=build_cfg, expected_steps=expected_steps,
    )


def write_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Write and immediately re-hash the retained arm manifest."""
    commit = git_sha(allow_dirty=False)
    files = [
        path for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"producer_manifest.json", ".round345_attempt"}
    ]
    manifest = run_dir / "producer_manifest.json"
    common.write_manifest(
        manifest,
        producer_commit=commit,
        binary=run_dir / "nemo",
        files=files,
        root=run_dir,
        build_source_root=run_dir / "build_sources",
    )
    return common.verify_manifest(manifest, expected_commit=commit)


def verify_restart_continuation_control(
    control_arm: Path,
    *,
    baseline_arm: Path = BASELINE_ARM,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    """Require bitwise-identical common variable data in every continuation restart.

    Arm A registered the delayed ``fwb2`` reduction before the continuation's
    input restart.  The restarted control cannot reconstruct that registration
    from its kt=8 input, so exactly the two delayed-buffer variables are allowed
    only in arm A.  File bytes and NetCDF metadata are deliberately not compared.
    """
    rows = []
    common_variable_comparisons = 0
    first_output_kt = RESTART_KT + r342.DAILY_STOCK_STRIDE
    require(expected_steps >= first_output_kt,
            f"restart continuation ends before first output: {expected_steps}")
    require((expected_steps - RESTART_KT) % r342.DAILY_STOCK_STRIDE == 0,
            f"restart continuation endpoint is off cadence: {expected_steps}")
    for kt in range(first_output_kt, expected_steps + 1, r342.DAILY_STOCK_STRIDE):
        for rank in (0, 1):
            name = f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
            baseline = baseline_arm / name
            control = control_arm / name
            require(baseline.is_file(), f"missing admitted arm-A restart: {baseline}")
            require(control.is_file() and not control.is_symlink(),
                    f"missing regular control restart: {control}")
            with netCDF4.Dataset(baseline, "r") as baseline_ds, netCDF4.Dataset(
                control, "r"
            ) as control_ds:
                baseline_names = set(baseline_ds.variables)
                control_names = set(control_ds.variables)
                baseline_only = baseline_names - control_names
                control_only = control_names - baseline_names
                require(
                    baseline_only == DELAY_RESTART_VARIABLES and not control_only,
                    f"{name}: restart variable inventory mismatch: "
                    f"arm_A_only={sorted(baseline_only)}, "
                    f"control_only={sorted(control_only)}, expected_arm_A_only="
                    f"{sorted(DELAY_RESTART_VARIABLES)}",
                )
                common_names = sorted(baseline_names & control_names)
                require(bool(common_names), f"{name}: no common restart variables")
                for variable_name in common_names:
                    baseline_var = baseline_ds.variables[variable_name]
                    control_var = control_ds.variables[variable_name]
                    baseline_var.set_auto_maskandscale(False)
                    control_var.set_auto_maskandscale(False)
                    baseline_values = np.asarray(baseline_var[:])
                    control_values = np.asarray(control_var[:])
                    require(
                        baseline_values.dtype == control_values.dtype,
                        f"{name}:{variable_name} dtype differs from arm A: "
                        f"{control_values.dtype} != {baseline_values.dtype}",
                    )
                    require(
                        baseline_values.shape == control_values.shape,
                        f"{name}:{variable_name} shape differs from arm A: "
                        f"{control_values.shape} != {baseline_values.shape}",
                    )
                    require(
                        baseline_values.tobytes(order="C")
                        == control_values.tobytes(order="C"),
                        f"{name}:{variable_name} restart variable data differs from arm A",
                    )
                common_variable_comparisons += len(common_names)
            baseline_sha = common.sha256(baseline)
            control_sha = common.sha256(control)
            rows.append({
                "kt": kt,
                "rank": rank,
                "file": name,
                "baseline_sha256": baseline_sha,
                "control_sha256": control_sha,
                "common_variable_count": len(common_names),
                "arm_a_only_variables": sorted(DELAY_RESTART_VARIABLES),
                "control_only_variables": [],
                "unequal_common_variable_data": [],
            })
    expected_pairs = (expected_steps - RESTART_KT) // r342.DAILY_STOCK_STRIDE
    expected_files = expected_pairs * 2
    require(len(rows) == expected_files,
            f"control compared {len(rows)} restart files, expected {expected_files}")
    return {
        "status": "PASS_R345_UNPERTURBED_RESTART_CONTROL",
        "baseline_arm": str(baseline_arm.resolve()),
        "control_arm": str(control_arm.resolve()),
        "restart_kt_first": first_output_kt,
        "restart_kt_last": expected_steps,
        "daily_restart_pairs_compared": expected_pairs,
        "restart_files_compared": len(rows),
        "common_variable_data_comparisons": common_variable_comparisons,
        "unequal_common_variable_data_count": 0,
        "all_common_variable_data_bitwise_identical": True,
        "required_arm_a_only_variables": sorted(DELAY_RESTART_VARIABLES),
        "control_only_variables": [],
        "comparison": "variable dtype + shape + raw C-order data bytes; NetCDF file bytes and metadata excluded",
        "restart_files": rows,
    }


def _copy_netcdf_excluding_variables(
    source: Path,
    destination: Path,
    excluded: set[str] | frozenset[str],
) -> None:
    """Copy a stub restart while omitting named variables from its inventory."""
    temporary = destination.with_name(destination.name + ".variable-copy.tmp")
    if temporary.exists():
        temporary.unlink()
    with netCDF4.Dataset(source, "r") as src, netCDF4.Dataset(
        temporary, "w", format=src.data_model
    ) as dst:
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, None if dimension.isunlimited() else len(dimension))
        dst.setncatts({name: src.getncattr(name) for name in src.ncattrs()})
        for name, src_var in src.variables.items():
            if name in excluded:
                continue
            attrs = {attr: src_var.getncattr(attr) for attr in src_var.ncattrs()}
            fill_value = attrs.pop("_FillValue", None)
            create_kwargs = {} if fill_value is None else {"fill_value": fill_value}
            dst_var = dst.createVariable(
                name, src_var.datatype, src_var.dimensions, **create_kwargs
            )
            dst_var.setncatts(attrs)
            src_var.set_auto_maskandscale(False)
            dst_var.set_auto_maskandscale(False)
            dst_var[...] = src_var[...]
    os.replace(temporary, destination)


def admit_control_arm(
    run_dir: Path,
    *,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
    allow_stub_binary: bool = False,
) -> dict[str, Any]:
    control_input = verify_control_input_manifest(run_dir)
    manifest = r342._verify_existing_run_manifest(run_dir)
    baseline = verify_baseline_admission()
    actual_binary = common.sha256(run_dir / "nemo")
    if not allow_stub_binary:
        require(actual_binary == baseline["binary_sha256"],
                "round-345 control binary differs from admitted round-342 arm A")
    input_identity = verify_arm_a_inputs_untouched(run_dir)
    inventory = r342.verify_no_unexpected_files(run_dir, expected_steps=expected_steps)
    ppsrc_dir = r342._ppsrc_dir(build_cfg)
    r342.verify_namelist_groups_against_ppsrc(
        run_dir / "namelist_cfg", r342.NAMELIST_REF, ppsrc_dir
    )
    expected_resolved = {
        **r342.r339_k0.RESOLVED,
        "nn_it000": str(FIRST_RUN_STEP),
        "nn_itend": str(expected_steps),
        "nn_stock": str(r342.DAILY_STOCK_STRIDE),
        "ln_rstart": "T",
        "nn_rstctl": "2",
        "cn_ocerst_in": f"ORCA2_{RESTART_KT:08d}_restart",
    }
    common.validate_ocean_output(run_dir / "ocean.output", expected_resolved)
    completion = r342.verify_nemo_completion_signal(run_dir, expected_steps=expected_steps)
    continuation = verify_restart_continuation_control(
        run_dir, expected_steps=expected_steps
    )
    return {
        "status": "PASS_R345_UNPERTURBED_RESTART_CONTROL_ARM",
        "admitted": True,
        "run_dir": str(run_dir.resolve()),
        "round": "orca2_r345",
        "stage": "Decision_188_unperturbed_restart_control",
        "restart_kt": RESTART_KT,
        "first_run_step": FIRST_RUN_STEP,
        "expected_steps": expected_steps,
        "inventory": inventory,
        "completion_signal": completion,
        "baseline_admission": baseline,
        "binary_sha256": actual_binary,
        "binary_equal_arm_a": actual_binary == baseline["binary_sha256"],
        "arm_a_input_identity": input_identity,
        "unperturbed_restart_control_input": control_input,
        "restart_continuation_control": continuation,
        "producer_manifest": manifest,
    }


def admit_perturbed_arm(
    run_dir: Path,
    *,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
    allow_stub_binary: bool = False,
    write_output: bool = True,
) -> dict[str, Any]:
    perturbation = verify_perturbation_manifest(run_dir)
    manifest = r342._verify_existing_run_manifest(run_dir)
    baseline = verify_baseline_admission()
    actual_binary = common.sha256(run_dir / "nemo")
    if not allow_stub_binary:
        require(actual_binary == baseline["binary_sha256"],
                "round-345 binary differs from admitted round-342 arm A")
    input_identity = verify_arm_a_inputs_untouched(run_dir)
    inventory = r342.verify_no_unexpected_files(run_dir, expected_steps=expected_steps)
    ppsrc_dir = r342._ppsrc_dir(build_cfg)
    r342.verify_namelist_groups_against_ppsrc(
        run_dir / "namelist_cfg", r342.NAMELIST_REF, ppsrc_dir
    )
    expected_resolved = {
        **r342.r339_k0.RESOLVED,
        "nn_it000": str(FIRST_RUN_STEP),
        "nn_itend": str(expected_steps),
        "nn_stock": str(r342.DAILY_STOCK_STRIDE),
        "ln_rstart": "T",
        "nn_rstctl": "2",
        "cn_ocerst_in": f"ORCA2_{RESTART_KT:08d}_restart",
    }
    common.validate_ocean_output(run_dir / "ocean.output", expected_resolved)
    completion = r342.verify_nemo_completion_signal(run_dir, expected_steps=expected_steps)
    expected_sizes = (r342.R339_RESTART_0.stat().st_size, r342.R339_RESTART_1.stat().st_size)
    for kt in range(r342.DAILY_STOCK_STRIDE, expected_steps + 1, r342.DAILY_STOCK_STRIDE):
        for rank, expected_size in enumerate(expected_sizes):
            r342._verify_single_restart_file(
                run_dir / f"ORCA2_{kt:08d}_restart_{rank:04d}.nc", kt, expected_size
            )
    result = {
        "status": "PASS_R345_S4_YEAR_SST_PLUS_ONE_ULP_ARM",
        "admitted": True,
        "run_dir": str(run_dir.resolve()),
        "round": "orca2_r345",
        "stage": "Decision_188_chaos_envelope_NEMO_arm",
        "restart_kt": RESTART_KT,
        "first_run_step": FIRST_RUN_STEP,
        "expected_steps": expected_steps,
        "daily_restart_pairs": expected_steps // r342.DAILY_STOCK_STRIDE,
        "inventory": inventory,
        "completion_signal": completion,
        "baseline_admission": baseline,
        "binary_sha256": actual_binary,
        "binary_equal_arm_a": actual_binary == baseline["binary_sha256"],
        "arm_a_input_identity": input_identity,
        "restart_sst_perturbation": perturbation,
        "producer_manifest": manifest,
    }
    if write_output:
        output = run_dir.parent / "round345_s4_year_sst_ulp_admission.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def admit_restart_pair(
    run_root: Path,
    *,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
    allow_stub_binary: bool = False,
) -> dict[str, Any]:
    control_arm = run_root / CONTROL_ARM_NAME
    perturbed_arm = run_root / PERTURBED_ARM_NAME
    control = admit_control_arm(
        control_arm, build_cfg=build_cfg, expected_steps=expected_steps,
        allow_stub_binary=allow_stub_binary,
    )
    perturbed = admit_perturbed_arm(
        perturbed_arm, build_cfg=build_cfg, expected_steps=expected_steps,
        allow_stub_binary=allow_stub_binary, write_output=False,
    )
    require(
        control["producer_manifest"]["producer_commit"]
        == perturbed["producer_manifest"]["producer_commit"],
        "control and perturbed arms have different producer commits",
    )
    require(control["binary_sha256"] == perturbed["binary_sha256"],
            "control and perturbed arms have different binary digests")
    perturbed["unperturbed_restart_control"] = control
    perturbed["restart_continuation_control"] = control["restart_continuation_control"]
    output = run_root / "round345_s4_year_sst_ulp_admission.json"
    output.write_text(json.dumps(perturbed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return perturbed


def _append_admit_log(run_root: Path, entry: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "admit_existing.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _retained_arm(refused: Path, arm_name: str, explicit: Path | None) -> Path:
    candidates = sorted(
        path for path in refused.iterdir()
        if path.is_dir() and arm_name in path.name
    )
    if explicit is not None and arm_name in explicit.name:
        return explicit
    require(bool(candidates), f"no retained {arm_name} arm found inside {refused}")
    return candidates[-1]


def admit_existing_with_quarantine(
    run_root_or_retained: Path,
    *,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
    allow_stub_binary: bool = False,
) -> dict[str, Any]:
    """Checklist 9/13 admission with fail-in-place retention and promotion."""
    explicit = None
    if run_root_or_retained.parent.name == "refused":
        explicit = run_root_or_retained
        run_root = run_root_or_retained.parent.parent
    else:
        run_root = run_root_or_retained
    control = run_root / CONTROL_ARM_NAME
    perturbed = run_root / PERTURBED_ARM_NAME
    refused = run_root / "refused"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if explicit is None and control.is_dir() and perturbed.is_dir():
        try:
            result = admit_restart_pair(
                run_root, build_cfg=build_cfg, expected_steps=expected_steps,
                allow_stub_binary=allow_stub_binary,
            )
            _append_admit_log(run_root, {
                "timestamp_utc": timestamp,
                "route": "in_place",
                "status": "PASS",
                "producer_commit": result["producer_manifest"]["producer_commit"],
                "binary_sha256": result["binary_sha256"],
                "control_arm": str(control),
                "perturbed_arm": str(perturbed),
            })
            return result
        except Exception as exc:
            refused.mkdir(parents=True, exist_ok=True)
            moved = []
            for arm in (control, perturbed):
                if arm.exists():
                    destination = refused / f"{timestamp}_{arm.name}"
                    shutil.move(str(arm), str(destination))
                    moved.append(str(destination))
            _append_admit_log(run_root, {
                "timestamp_utc": timestamp,
                "route": "in_place",
                "status": "REFUSED_AND_QUARANTINED",
                "error": str(exc),
                "quarantined": moved,
            })
            (run_root / "round345_s4_year_sst_ulp_admission.json").write_text(
                json.dumps({
                    "status": "REFUSED_R345_ADMIT_EXISTING",
                    "admitted": False,
                    "timestamp_utc": timestamp,
                    "error": str(exc),
                    "quarantined": moved,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise GateError(
                f"--admit-existing refused in-place restart arms ({exc}); "
                f"quarantined to {moved}"
            ) from exc

    require(not control.exists() and not perturbed.exists(),
            f"incomplete canonical round-345 arm pair under {run_root}")
    require(refused.is_dir(), f"missing refused directory: {refused}")
    retained_control = _retained_arm(refused, CONTROL_ARM_NAME, explicit)
    retained_perturbed = _retained_arm(refused, PERTURBED_ARM_NAME, explicit)
    try:
        require(retained_control.resolve().is_relative_to(refused.resolve())
                and retained_perturbed.resolve().is_relative_to(refused.resolve()),
                "retained arms are not inside refused/")
        # Validate manifests, ancestry, binary/file digests, and both scientific
        # arms before any promotion.  No admission receipt is written in refused/.
        r342._verify_existing_run_manifest(retained_control)
        r342._verify_existing_run_manifest(retained_perturbed)
        admit_control_arm(
            retained_control, build_cfg=build_cfg, expected_steps=expected_steps,
            allow_stub_binary=allow_stub_binary,
        )
        admit_perturbed_arm(
            retained_perturbed, build_cfg=build_cfg, expected_steps=expected_steps,
            allow_stub_binary=allow_stub_binary, write_output=False,
        )
        shutil.move(str(retained_control), str(control))
        shutil.move(str(retained_perturbed), str(perturbed))
        result = admit_restart_pair(
            run_root, build_cfg=build_cfg, expected_steps=expected_steps,
            allow_stub_binary=allow_stub_binary,
        )
        result["admit_existing_route"] = "refused"
        result["promoted_from_refused"] = [
            str(retained_control), str(retained_perturbed),
        ]
        _append_admit_log(run_root, {
            "timestamp_utc": timestamp,
            "route": "refused",
            "status": "PASS_PROMOTED",
            "producer_commit": result["producer_manifest"]["producer_commit"],
            "binary_sha256": result["binary_sha256"],
            "promoted_from": [str(retained_control), str(retained_perturbed)],
            "control_arm": str(control),
            "perturbed_arm": str(perturbed),
        })
        return result
    except Exception as exc:
        # Promotion is transactional: if the canonical-path recheck fails,
        # return either moved arm to its original retained refused/ path.
        if control.exists() and not retained_control.exists():
            shutil.move(str(control), str(retained_control))
        if perturbed.exists() and not retained_perturbed.exists():
            shutil.move(str(perturbed), str(retained_perturbed))
        _append_admit_log(run_root, {
            "timestamp_utc": timestamp,
            "route": "refused",
            "status": "REFUSED_RETAINED_IN_PLACE",
            "error": str(exc),
            "retained_control": str(retained_control),
            "retained_perturbed": str(retained_perturbed),
        })
        raise GateError(
            f"--admit-existing refused retained restart arms ({exc}); kept in refused/"
        ) from exc


def write_stub_outputs(
    run_dir: Path,
    *,
    rank: int | None = None,
    plant_unproduced_file: bool = False,
    plant_missing_stop: bool = False,
    plant_short_timestep: int | None = None,
) -> dict[str, Any]:
    """Replace only NEMO, preserving the staged restart input byte-for-byte."""
    ranks = (0, 1) if rank is None else (rank,)
    require(all(local_rank in (0, 1) for local_rank in ranks),
            f"stub rank must be 0 or 1, got {rank}")
    has_control_manifest = (run_dir / CONTROL_INPUT_MANIFEST).is_file()
    has_perturbation_manifest = (run_dir / PERTURBATION_MANIFEST).is_file()
    require(has_control_manifest != has_perturbation_manifest,
            "stub run directory must identify exactly one restart arm")
    control = has_control_manifest
    if control:
        verify_control_input_manifest(run_dir)
    else:
        verify_perturbation_manifest(run_dir)
    input_before = {
        local_rank: common.sha256(
            run_dir / f"ORCA2_{RESTART_KT:08d}_restart_{local_rank:04d}.nc"
        )
        for local_rank in ranks
    }
    result = r342.write_stub_s4_year_outputs(
        run_dir,
        rank=rank,
        first_restart_kt=RESTART_KT + r342.DAILY_STOCK_STRIDE,
        plant_unproduced_file=plant_unproduced_file,
        plant_missing_stop=plant_missing_stop,
        plant_short_timestep=plant_short_timestep,
    )
    expected_steps, _stock = r342.parse_namelist_itend_stock(run_dir / "namelist_cfg")
    for kt in range(
        RESTART_KT + r342.DAILY_STOCK_STRIDE,
        expected_steps + 1,
        r342.DAILY_STOCK_STRIDE,
    ):
        for local_rank in ranks:
            name = f"ORCA2_{kt:08d}_restart_{local_rank:04d}.nc"
            shutil.copy2(BASELINE_ARM / name, run_dir / name)
            if control:
                _copy_netcdf_excluding_variables(
                    run_dir / name, run_dir / name, DELAY_RESTART_VARIABLES
                )
    input_after = {
        local_rank: common.sha256(
            run_dir / f"ORCA2_{RESTART_KT:08d}_restart_{local_rank:04d}.nc"
        )
        for local_rank in ranks
    }
    require(input_after == input_before,
            f"stub changed staged kt={RESTART_KT} restart input: "
            f"before={input_before}, after={input_after}")
    if control:
        verify_control_input_manifest(run_dir)
    else:
        verify_perturbation_manifest(run_dir)

    if rank in (None, 0):
        ocean_output = run_dir / "ocean.output"
        text = ocean_output.read_text(encoding="utf-8")
        replacements = {
            "cn_ocerst_in": f"ORCA2_{RESTART_KT:08d}_restart",
            "ln_rstart": "T",
            "nn_rstctl": "2",
            "nn_it000": str(FIRST_RUN_STEP),
        }
        for name, value in replacements.items():
            text, count = re.subn(
                rf"(?im)(^\s*[^\n]*\b{name}\s*=\s*)\S+", rf"\g<1>{value}", text, count=1
            )
            require(count == 1, f"stub ocean.output lacks {name}")
        ocean_output.write_text(text, encoding="utf-8")
    result["input_restart_sha256_before"] = input_before
    result["input_restart_sha256_after"] = input_after
    return result


def verify_run_sh(run_sh: Path) -> dict[str, Any]:
    require(run_sh.is_file(), f"missing acquisition script: {run_sh}")
    text = run_sh.read_text(encoding="utf-8")
    require("rm -rf" not in text and "pkill" not in text and "killall" not in text,
            "acquisition script contains a destructive or pattern-kill command")
    syntax = subprocess.run(["bash", "-n", str(run_sh)], capture_output=True, text=True)
    require(syntax.returncode == 0, f"bash -n failed: {syntax.stderr}")
    PREFLIGHT_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PREFLIGHT_SCRATCH_ROOT) as tmp:
        planted = Path(tmp) / "run.sh"
        planted.write_text(text + "\nif then\n", encoding="utf-8")
        bad = subprocess.run(["bash", "-n", str(planted)], capture_output=True, text=True)
        require(bad.returncode != 0, "shell syntax plant did not fail")
    return {"passed": True, "syntax_plant_rc": bad.returncode}


def plant_nonancestor_manifest() -> dict[str, Any]:
    """Prove checklist-9 producer commits must be ancestors of admission HEAD."""
    PREFLIGHT_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PREFLIGHT_SCRATCH_ROOT) as tmp:
        root = Path(tmp)
        repo = root / "repo"
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        marker = repo / "marker"
        marker.write_text("main\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "marker"], check=True)
        subprocess.run([
            "git", "-C", str(repo), "-c", "user.name=Round 345 preflight",
            "-c", "user.email=round345@example.invalid", "commit", "-q", "-m", "main",
        ], check=True)
        main_commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(["git", "-C", str(repo), "switch", "-q", "--orphan", "other"], check=True)
        marker.write_text("other\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "marker"], check=True)
        subprocess.run([
            "git", "-C", str(repo), "-c", "user.name=Round 345 preflight",
            "-c", "user.email=round345@example.invalid", "commit", "-q", "-m", "other",
        ], check=True)
        other_commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(["git", "-C", str(repo), "switch", "-q", "main"], check=True)
        run_dir = root / "retained"
        run_dir.mkdir()
        binary = run_dir / "nemo"
        binary.write_bytes(b"stub binary\n")
        (run_dir / "build_sources").mkdir()
        source_file = run_dir / "build_sources" / "source.f90"
        source_file.write_text(
            "PROGRAM source\nEND PROGRAM source\n", encoding="utf-8"
        )
        common.write_manifest(
            run_dir / "producer_manifest.json",
            producer_commit=other_commit,
            binary=binary,
            files=[binary, source_file],
            root=run_dir,
            build_source_root=run_dir / "build_sources",
        )
        caught = False
        error = ""
        try:
            r342._verify_existing_run_manifest(run_dir, repo=repo)
        except GateError as exc:
            caught = True
            error = str(exc)
        require(caught and main_commit != other_commit and "not an ancestor" in error,
                "non-ancestor producer_commit plant did not fail")
    return {"passed": True, "error": error}


def run_preflight(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
) -> dict[str, Any]:
    """Run inherited checklist gates plus paired restart-arm controls."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    r342.verify_clean_worktree(REPO_ROOT)
    run_sh = Path(__file__).resolve().parent / (
        "nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh"
    )
    shell = verify_run_sh(run_sh)
    inherited = r342.run_preflight(
        evidence_dir=evidence_dir / "inherited_round342",
        build_cfg=build_cfg,
        ice_cover_path=ice_cover_path,
    )
    PREFLIGHT_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PREFLIGHT_SCRATCH_ROOT) as tmp:
        stub_root = Path(tmp) / "stub_round345"
        result = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False,
        )
        require(result.returncode == 0,
                f"round345 stub pipeline failed:\n{result.stdout}\n{result.stderr}")
        control_arm = stub_root / CONTROL_ARM_NAME
        perturbed_arm = stub_root / PERTURBED_ARM_NAME
        require(
            (control_arm / "nemo").read_bytes() == (perturbed_arm / "nemo").read_bytes(),
            "control and perturbed stub executables are not byte-identical",
        )
        perturbation_audit = verify_perturbation_manifest(perturbed_arm)
        control_input_audit = verify_control_input_manifest(control_arm)
        continuation_audit = verify_restart_continuation_control(
            control_arm, expected_steps=16
        )
        overwrite = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False,
        )
        require(overwrite.returncode != 0 and control_arm.is_dir() and perturbed_arm.is_dir(),
                "retained restart-arm overwrite plant did not refuse safely")
        require(not (stub_root / "refused").exists(),
                "overwrite refusal moved an already admitted arm")
        env = dict(os.environ, R345_PREFLIGHT_ALLOW_STUB="1")
        in_place = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False, env=env,
        )
        require(in_place.returncode == 0,
                f"in-place --admit-existing failed:\n{in_place.stdout}\n{in_place.stderr}")
        admit_log = (stub_root / "admit_existing.log").read_text(encoding="utf-8")
        require('"route": "in_place"' in admit_log and '"status": "PASS"' in admit_log,
                "admit_existing.log lacks the in-place PASS record")

        restart = perturbed_arm / f"ORCA2_{RESTART_KT:08d}_restart_0000.nc"
        backup = Path(tmp) / "perturbed_restart.preplant"
        shutil.copy2(restart, backup)
        with netCDF4.Dataset(restart, "r+") as dataset:
            variable = dataset.variables[RESTART_VARIABLE]
            variable.set_auto_maskandscale(False)
            value = np.asarray(variable[0, 0], dtype=np.float64)
            wet = _rank_wet_mask(perturbed_arm / DOMAIN_FILE, 0, tuple(value.shape))
            j, i = np.argwhere(wet & np.isfinite(value))[0]
            value[j, i] = np.nextafter(value[j, i], np.float64(np.inf))
            variable[0, 0] = value
        caught = False
        try:
            verify_perturbation_manifest(perturbed_arm)
        except GateError:
            caught = True
        require(caught, "two-ULP perturbation plant did not fail")
        shutil.move(backup, restart)
        verify_perturbation_manifest(perturbed_arm)

        control_restart = control_arm / "ORCA2_00000016_restart_0000.nc"
        control_backup = Path(tmp) / "control_restart.preplant"
        shutil.copy2(control_restart, control_backup)
        with netCDF4.Dataset(control_restart, "r+") as dataset:
            variable = dataset.variables[RESTART_VARIABLE]
            variable.set_auto_maskandscale(False)
            values = np.asarray(variable[:])
            byte_view = values.view(np.uint8).reshape(-1)
            byte_view[0] ^= np.uint8(1)
            variable[:] = values
        control_tn_byte_flip_caught = False
        try:
            verify_restart_continuation_control(control_arm, expected_steps=16)
        except GateError:
            control_tn_byte_flip_caught = True
        require(control_tn_byte_flip_caught,
                "one-byte tn restart-data plant did not fail")
        shutil.copy2(control_backup, control_restart)

        with netCDF4.Dataset(control_restart, "r+") as dataset:
            dataset.createVariable("UNKNOWN_R345_PLANT", "f8").assignValue(1.0)
        control_extra_variable_caught = False
        try:
            verify_restart_continuation_control(control_arm, expected_steps=16)
        except GateError:
            control_extra_variable_caught = True
        require(control_extra_variable_caught,
                "extra unknown restart-variable plant did not fail")
        shutil.copy2(control_backup, control_restart)

        _copy_netcdf_excluding_variables(
            control_backup, control_restart, {"sn"}
        )
        control_missing_non_delay_caught = False
        try:
            verify_restart_continuation_control(control_arm, expected_steps=16)
        except GateError:
            control_missing_non_delay_caught = True
        require(control_missing_non_delay_caught,
                "missing non-DELAY restart-variable plant did not fail")
        shutil.copy2(control_backup, control_restart)
        verify_restart_continuation_control(control_arm, expected_steps=16)

        # A failing in-place admission must quarantine both arms and log the
        # refusal.  Once repaired, the refused/ route must promote both arms.
        failed_backup = Path(tmp) / "failed_admission_restart.preplant"
        shutil.copy2(restart, failed_backup)
        with restart.open("ab") as handle:
            handle.write(b"manifest checksum plant")
        failed_in_place = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False, env=env,
        )
        refused = stub_root / "refused"
        retained_controls = list(refused.glob(f"*{CONTROL_ARM_NAME}*"))
        retained_perturbed = list(refused.glob(f"*{PERTURBED_ARM_NAME}*"))
        require(failed_in_place.returncode != 0
                and len(retained_controls) == 1 and len(retained_perturbed) == 1,
                "failing in-place admission did not quarantine both restart arms")
        require('"status": "REFUSED_AND_QUARANTINED"' in
                (stub_root / "admit_existing.log").read_text(encoding="utf-8"),
                "admit_existing.log lacks the failing in-place admission record")
        shutil.move(failed_backup, retained_perturbed[0] / restart.name)
        require(not (refused / "round345_s4_year_sst_ulp_admission.json").exists(),
                "admission receipt was written inside refused/")
        from_refused = subprocess.run(
            ["bash", str(run_sh), "--admit-existing", str(retained_perturbed[0]),
             "--stub-steps", "16"],
            capture_output=True, text=True, check=False, env=env,
        )
        require(from_refused.returncode == 0
                and control_arm.is_dir() and perturbed_arm.is_dir()
                and not retained_controls[0].exists() and not retained_perturbed[0].exists(),
                f"refused/ --admit-existing failed:\n{from_refused.stdout}\n{from_refused.stderr}")
        admit_log = (stub_root / "admit_existing.log").read_text(encoding="utf-8")
        require('"status": "PASS_PROMOTED"' in admit_log,
                "admit_existing.log lacks the refused/ promotion record")

        refusal_root = Path(tmp) / "stub_refusal"
        refusal = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(refusal_root),
             "--stub-steps", "16", "--plant-stub-missing-stop"],
            capture_output=True, text=True, check=False,
        )
        refused_arms = list((refusal_root / "refused").glob(f"*{CONTROL_ARM_NAME}*"))
        require(refusal.returncode != 0 and len(refused_arms) == 1,
                "missing-STOP refusal did not retain exactly one failed round-345 arm")
        require((refused_arms[0] / "nemo").is_file()
                and (refused_arms[0] / "build_sources").is_dir(),
                "refused arm did not retain its binary and build snapshot")
        ancestry_plant = plant_nonancestor_manifest()
    report = {
        "status": "ORCA2_ROUND345_SST_ULP_PREFLIGHT_READY",
        "round": "orca2_r345",
        "inherited_round342_checklist": inherited,
        "run_sh": shell,
        "stub_end_to_end": True,
        "overwrite_refusal_plant": True,
        "in_place_admit_existing_plant": True,
        "refused_admit_existing_plant": True,
        "refusal_retention_plant": True,
        "two_ulp_plant": True,
        "control_tn_byte_flip_plant": True,
        "control_extra_variable_plant": True,
        "control_missing_non_delay_variable_plant": True,
        "producer_commit_ancestry_plant": ancestry_plant,
        "perturbation_audit": perturbation_audit,
        "control_input_audit": control_input_audit,
        "restart_continuation_control": continuation_audit,
        "worktree": worktree_stamp(allow_dirty=False),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    output = evidence_dir / "round345_sst_ulp_preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-run-dir", type=Path)
    parser.add_argument("--stage-control-run-dir", type=Path)
    parser.add_argument("--admit-run-dir", type=Path)
    parser.add_argument("--admit-existing", type=Path)
    parser.add_argument("--write-run-manifest", type=Path)
    parser.add_argument("--verify-perturbation", type=Path)
    parser.add_argument("--act-as-nemo-binary", action="store_true")
    parser.add_argument("--plant-stub-unproduced-file", action="store_true")
    parser.add_argument("--plant-stub-missing-stop", action="store_true")
    parser.add_argument("--plant-stub-short-timestep", type=int)
    parser.add_argument("--allow-stub-binary", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--build-cfg", type=Path, default=r342.DEFAULT_BUILD_CFG)
    parser.add_argument("--ice-cover", type=Path, default=r342.DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--expected-steps", type=int, default=r342.YEAR_STEPS)
    args = parser.parse_args(argv)
    selected = sum(value is not None and value is not False for value in (
        args.stage_run_dir, args.stage_control_run_dir, args.admit_run_dir,
        args.admit_existing, args.write_run_manifest,
        args.verify_perturbation, args.preflight_only, args.act_as_nemo_binary,
    ))
    require(selected == 1, "select exactly one gate action")
    if args.stage_run_dir is not None:
        result = stage_perturbed_arm(
            args.stage_run_dir, ice_cover_path=args.ice_cover,
            build_cfg=args.build_cfg, expected_steps=args.expected_steps,
        )
    elif args.stage_control_run_dir is not None:
        result = stage_control_arm(
            args.stage_control_run_dir, ice_cover_path=args.ice_cover,
            build_cfg=args.build_cfg, expected_steps=args.expected_steps,
        )
    elif args.admit_run_dir is not None:
        result = admit_perturbed_arm(
            args.admit_run_dir, build_cfg=args.build_cfg,
            expected_steps=args.expected_steps, allow_stub_binary=args.allow_stub_binary,
        )
    elif args.admit_existing is not None:
        result = admit_existing_with_quarantine(
            args.admit_existing, build_cfg=args.build_cfg,
            expected_steps=args.expected_steps, allow_stub_binary=args.allow_stub_binary,
        )
    elif args.write_run_manifest is not None:
        result = write_run_manifest(args.write_run_manifest)
    elif args.verify_perturbation is not None:
        result = verify_perturbation_manifest(args.verify_perturbation)
    elif args.act_as_nemo_binary:
        result = write_stub_outputs(
            Path.cwd(),
            rank=r342.stub_mpi_rank_from_env(),
            plant_unproduced_file=args.plant_stub_unproduced_file,
            plant_missing_stop=args.plant_stub_missing_stop,
            plant_short_timestep=args.plant_stub_short_timestep,
        )
    else:
        result = run_preflight(
            evidence_dir=args.evidence_dir, build_cfg=args.build_cfg,
            ice_cover_path=args.ice_cover,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
