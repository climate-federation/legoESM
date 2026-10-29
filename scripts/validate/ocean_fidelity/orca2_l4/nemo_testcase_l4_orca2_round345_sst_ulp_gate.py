#!/usr/bin/env python3
"""Stage, audit, and admit the Decision-188 one-ULP K0 NEMO arm."""

from __future__ import annotations

import argparse
import json
import os
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

TEMPERATURE_FILE = "data_1m_potential_temperature_nomask.nc"
TEMPERATURE_VARIABLE = "votemper"
DOMAIN_FILE = "ORCA_R2_zps_domcfg.nc"
PERTURBATION_MANIFEST = "initial_sst_ulp_perturbation.json"
PERTURBED_ARM_NAME = "orca2_k0_year1_sst_plus_1ulp_np2"
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


def verify_sst_ulp_perturbation(
    source_file: Path,
    perturbed_file: Path,
    domain_file: Path,
) -> dict[str, Any]:
    """Prove the perturbed input differs only by +1 float32 ULP on wet SST."""
    require(source_file.is_file(), f"missing source temperature file: {source_file}")
    require(perturbed_file.is_file() and not perturbed_file.is_symlink(),
            f"perturbed temperature input must be a regular file: {perturbed_file}")
    wet = _wet_surface_mask(domain_file)
    changed = 0
    selected = 0
    with (
        netCDF4.Dataset(source_file, "r") as source,
        netCDF4.Dataset(perturbed_file, "r") as perturbed,
    ):
        require(set(source.dimensions) == set(perturbed.dimensions),
                "temperature-file dimensions changed")
        require(set(source.variables) == set(perturbed.variables),
                "temperature-file variable inventory changed")
        for name in source.variables:
            src_var = source.variables[name]
            dst_var = perturbed.variables[name]
            src_var.set_auto_maskandscale(False)
            dst_var.set_auto_maskandscale(False)
            require(src_var.dtype == dst_var.dtype and src_var.dimensions == dst_var.dimensions,
                    f"{name}: dtype or dimensions changed")
            require(src_var.ncattrs() == dst_var.ncattrs(), f"{name}: attribute names changed")
            for attr in src_var.ncattrs():
                require(np.array_equal(np.asarray(src_var.getncattr(attr)),
                                       np.asarray(dst_var.getncattr(attr))),
                        f"{name}:{attr} attribute changed")
            if name != TEMPERATURE_VARIABLE:
                require(np.array_equal(np.asarray(src_var[:]), np.asarray(dst_var[:])),
                        f"unselected variable {name} changed")
                continue
            require(src_var.dtype == np.dtype("float32"),
                    f"{name} must be float32 for the one-ULP pattern, got {src_var.dtype}")
            require(src_var.ndim == 4 and src_var.shape[0] == 12 and src_var.shape[1] >= 1
                    and tuple(src_var.shape[-2:]) == wet.shape,
                    f"unexpected {name} layout {src_var.shape} for wet mask {wet.shape}")
            for record in range(src_var.shape[0]):
                src_surface = np.asarray(src_var[record, 0, :, :], dtype=np.float32)
                dst_surface = np.asarray(dst_var[record, 0, :, :], dtype=np.float32)
                finite_wet = wet & np.isfinite(src_surface)
                expected = src_surface.copy()
                expected[finite_wet] = np.nextafter(
                    src_surface[finite_wet], np.float32(np.inf), dtype=np.float32
                )
                require(np.array_equal(dst_surface.view(np.uint32), expected.view(np.uint32)),
                        f"record {record}: wet-SST one-ULP pattern or untouched land bits differ")
                selected += int(np.count_nonzero(finite_wet))
                changed += int(np.count_nonzero(
                    dst_surface[finite_wet].view(np.uint32)
                    != src_surface[finite_wet].view(np.uint32)
                ))
                for level in range(1, src_var.shape[1]):
                    src_level = np.asarray(src_var[record, level, :, :], dtype=np.float32)
                    dst_level = np.asarray(dst_var[record, level, :, :], dtype=np.float32)
                    require(np.array_equal(src_level.view(np.uint32), dst_level.view(np.uint32)),
                            f"record {record} level {level}: non-surface temperature changed")
    require(selected > 0 and changed == selected,
            f"one-ULP pattern selected {selected} values but changed {changed}")
    source_sha = common.sha256(source_file)
    perturbed_sha = common.sha256(perturbed_file)
    require(source_sha != perturbed_sha, "source and perturbed temperature digests are identical")
    return {
        "status": "PASS_R345_INITIAL_SST_PLUS_ONE_ULP",
        "source_resolved_path": str(source_file.resolve()),
        "source_sha256": source_sha,
        "perturbed_sha256": perturbed_sha,
        "domain_file_resolved_path": str(domain_file.resolve()),
        "domain_file_sha256": common.sha256(domain_file),
        "variable": TEMPERATURE_VARIABLE,
        "dtype": "float32",
        "selection": "all finite wet values at votemper[:, 0, :, :]",
        "direction": "nextafter(value, +infinity)",
        "records": 12,
        "selected_values": selected,
        "changed_values": changed,
        "exact_ulp_distance": 1,
        "unselected_values_bit_identical": True,
        "nemo_use": ["cold-start initial T/S", "configured T/S damping target"],
    }


def perturb_initial_sst(run_dir: Path) -> dict[str, Any]:
    """Replace the staged source symlink by the audited one-ULP regular copy."""
    target = run_dir / TEMPERATURE_FILE
    require(target.is_symlink(), f"expected staged source temperature symlink: {target}")
    source = target.resolve(strict=True)
    temporary = run_dir / f".{TEMPERATURE_FILE}.perturbing.{os.getpid()}"
    shutil.copy2(source, temporary)
    with netCDF4.Dataset(temporary, "r+") as dataset:
        variable = dataset.variables[TEMPERATURE_VARIABLE]
        variable.set_auto_maskandscale(False)
        wet = _wet_surface_mask(run_dir / DOMAIN_FILE)
        require(variable.dtype == np.dtype("float32") and variable.ndim == 4
                and variable.shape[0] == 12,
                f"unexpected {TEMPERATURE_VARIABLE} storage: {variable.dtype} {variable.shape}")
        for record in range(variable.shape[0]):
            surface = np.asarray(variable[record, 0, :, :], dtype=np.float32)
            selected = wet & np.isfinite(surface)
            surface[selected] = np.nextafter(
                surface[selected], np.float32(np.inf), dtype=np.float32
            )
            variable[record, 0, :, :] = surface
    target.unlink()
    temporary.replace(target)
    audit = verify_sst_ulp_perturbation(source, target, run_dir / DOMAIN_FILE)
    (run_dir / PERTURBATION_MANIFEST).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def verify_perturbation_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / PERTURBATION_MANIFEST
    require(manifest_path.is_file(), f"missing perturbation manifest: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(recorded["source_resolved_path"])
    measured = verify_sst_ulp_perturbation(
        source, run_dir / TEMPERATURE_FILE, run_dir / DOMAIN_FILE
    )
    require(recorded == measured, "recorded perturbation manifest does not match re-audit")
    return measured


def stage_perturbed_arm(
    run_dir: Path,
    *,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    staged = r342.stage_s4_year_run_dir(
        run_dir,
        ice_cover_path,
        build_cfg=build_cfg,
        nn_itend=expected_steps,
        nn_stock=r342.DAILY_STOCK_STRIDE,
    )
    staged["initial_sst_perturbation"] = perturb_initial_sst(run_dir)
    return staged


def write_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Write and immediately re-hash the retained arm manifest."""
    commit = git_sha(allow_dirty=False)
    files = [
        path for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "producer_manifest.json"
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


def admit_perturbed_arm(
    run_dir: Path,
    *,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    expected_steps: int = r342.YEAR_STEPS,
) -> dict[str, Any]:
    perturbation = verify_perturbation_manifest(run_dir)
    manifest = common.verify_manifest(run_dir / "producer_manifest.json")
    arm = r342.admit_s4_year_single_arm(
        run_dir,
        build_cfg=build_cfg,
        expected_steps=expected_steps,
        day1_reference_run=None,
    )
    result = {
        **arm,
        "status": "PASS_R345_S4_YEAR_SST_PLUS_ONE_ULP_ARM",
        "round": "orca2_r345",
        "stage": "Decision_188_chaos_envelope_NEMO_arm",
        "day1_unperturbed_identity_required": False,
        "initial_sst_perturbation": perturbation,
        "producer_manifest": manifest,
    }
    output = run_dir.parent / "round345_s4_year_sst_ulp_admission.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def verify_run_sh(run_sh: Path) -> dict[str, Any]:
    require(run_sh.is_file(), f"missing acquisition script: {run_sh}")
    text = run_sh.read_text(encoding="utf-8")
    require("rm -rf" not in text and "pkill" not in text and "killall" not in text,
            "acquisition script contains a destructive or pattern-kill command")
    syntax = subprocess.run(["bash", "-n", str(run_sh)], capture_output=True, text=True)
    require(syntax.returncode == 0, f"bash -n failed: {syntax.stderr}")
    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp) / "run.sh"
        planted.write_text(text + "\nif then\n", encoding="utf-8")
        bad = subprocess.run(["bash", "-n", str(planted)], capture_output=True, text=True)
        require(bad.returncode != 0, "shell syntax plant did not fail")
    return {"passed": True, "syntax_plant_rc": bad.returncode}


def run_preflight(
    *,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    build_cfg: Path = r342.DEFAULT_BUILD_CFG,
    ice_cover_path: Path = r342.DEFAULT_ICE_COVER_PATH,
) -> dict[str, Any]:
    """Run inherited checklist gates plus the one-arm perturbation controls."""
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
    with tempfile.TemporaryDirectory() as tmp:
        stub_root = Path(tmp) / "stub_round345"
        result = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False,
        )
        require(result.returncode == 0,
                f"round345 stub pipeline failed:\n{result.stdout}\n{result.stderr}")
        arm = stub_root / PERTURBED_ARM_NAME
        audit = verify_perturbation_manifest(arm)
        overwrite = subprocess.run(
            ["bash", str(run_sh), "--stub-run-dir", str(stub_root), "--stub-steps", "16"],
            capture_output=True, text=True, check=False,
        )
        require(overwrite.returncode != 0 and arm.is_dir(),
                "retained perturbed arm overwrite plant did not refuse safely")
        with netCDF4.Dataset(arm / TEMPERATURE_FILE, "r+") as dataset:
            variable = dataset.variables[TEMPERATURE_VARIABLE]
            variable.set_auto_maskandscale(False)
            value = np.asarray(variable[0, 0, :, :], dtype=np.float32)
            wet = _wet_surface_mask(arm / DOMAIN_FILE)
            j, i = np.argwhere(wet & np.isfinite(value))[0]
            value[j, i] = np.nextafter(value[j, i], np.float32(np.inf), dtype=np.float32)
            variable[0, 0, :, :] = value
        caught = False
        try:
            verify_perturbation_manifest(arm)
        except GateError:
            caught = True
        require(caught, "two-ULP perturbation plant did not fail")
    report = {
        "status": "ORCA2_ROUND345_SST_ULP_PREFLIGHT_READY",
        "round": "orca2_r345",
        "inherited_round342_checklist": inherited,
        "run_sh": shell,
        "stub_end_to_end": True,
        "overwrite_refusal_plant": True,
        "two_ulp_plant": True,
        "perturbation_audit": audit,
        "worktree": worktree_stamp(allow_dirty=False),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    output = evidence_dir / "round345_sst_ulp_preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-run-dir", type=Path)
    parser.add_argument("--admit-run-dir", type=Path)
    parser.add_argument("--write-run-manifest", type=Path)
    parser.add_argument("--verify-perturbation", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--build-cfg", type=Path, default=r342.DEFAULT_BUILD_CFG)
    parser.add_argument("--ice-cover", type=Path, default=r342.DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--expected-steps", type=int, default=r342.YEAR_STEPS)
    args = parser.parse_args(argv)
    selected = sum(value is not None and value is not False for value in (
        args.stage_run_dir, args.admit_run_dir, args.write_run_manifest,
        args.verify_perturbation, args.preflight_only,
    ))
    require(selected == 1, "select exactly one gate action")
    if args.stage_run_dir is not None:
        result = stage_perturbed_arm(
            args.stage_run_dir, ice_cover_path=args.ice_cover,
            build_cfg=args.build_cfg, expected_steps=args.expected_steps,
        )
    elif args.admit_run_dir is not None:
        result = admit_perturbed_arm(
            args.admit_run_dir, build_cfg=args.build_cfg,
            expected_steps=args.expected_steps,
        )
    elif args.write_run_manifest is not None:
        result = write_run_manifest(args.write_run_manifest)
    elif args.verify_perturbation is not None:
        result = verify_perturbation_manifest(args.verify_perturbation)
    else:
        result = run_preflight(
            evidence_dir=args.evidence_dir, build_cfg=args.build_cfg,
            ice_cover_path=args.ice_cover,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
