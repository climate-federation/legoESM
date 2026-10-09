#!/usr/bin/env python3
"""Fail-closed admission gate for the Round-131 daily NEMO record.

The daily-reset experiment needs a completed NEMO restart after every six
model steps through day 360.  This gate inventories the files, checks the
restart schema and time metadata, and proves every 30-day overlap bit-identical
to the immutable monthly year record.  It writes an audit JSON even when the
record is incomplete, then exits nonzero with ``STATUS STOPPED-FOR-RECORD``.

The two self-check plants operate on a complete virtual inventory/schema so
they cannot pass merely because the real record is already incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


class GateError(RuntimeError):
    """Raised when a fail-closed admission condition is not met."""


PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
DEFAULT_DAILY_ROOT = PHASE3 / "year_owners/nemo_seed0"
DEFAULT_MONTHLY_ROOT = PHASE3 / "year_fromrest/nemo_seed0"
REPO_ROOT = Path(__file__).resolve().parents[4]

STEPS_PER_DAY = 6
YEAR_DAYS = 360
REQUIRED_STEPS = tuple(range(STEPS_PER_DAY,
                             STEPS_PER_DAY * YEAR_DAYS + 1,
                             STEPS_PER_DAY))
MONTHLY_STEPS = tuple(range(30 * STEPS_PER_DAY,
                            STEPS_PER_DAY * YEAR_DAYS + 1,
                            30 * STEPS_PER_DAY))
RESTART_RE = re.compile(
    r"^GYRE_OMIP_L2_P3_(?P<step>[0-9]{8})_restart[.]nc$")
REQUIRED_VARIABLES = (
    "tn", "sn",
    "un", "vn", "uu_n", "vv_n",
    "ub_e", "vb_e", "ubb_e", "vbb_e",
    "sshn", "ssha", "sshb_e", "sshbb_e",
    "en", "avm_k", "avt_k", "dissl",
)
PLANTS = ("missing-boundary", "required-variable", "monthly-overlap-ulp")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False,
        capture_output=True, text=True)
    require(result.returncode == 0,
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def worktree_stamp(expect_commit: str) -> dict:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", expect_commit)),
            "--expect-commit must be a full lowercase commit hash")
    head = _run_git("rev-parse", "HEAD")
    require(head == expect_commit,
            f"tool commit {head} != --expect-commit {expect_commit}")
    status = _run_git("status", "--porcelain", "--untracked-files=all")
    require(status == "", "worktree is dirty; record audit is not citable")
    return {"commit": head, "worktree_clean": True}


def restart_path(root: Path, step: int) -> Path:
    return root / f"GYRE_OMIP_L2_P3_{step:08d}_restart.nc"


def inventory(root: Path) -> tuple[dict[int, Path], list[str]]:
    records: dict[int, Path] = {}
    ignored: list[str] = []
    if not root.is_dir():
        return records, ignored
    for path in sorted(root.glob("*restart.nc")):
        match = RESTART_RE.match(path.name)
        if match is None:
            ignored.append(path.name)
            continue
        step = int(match.group("step"))
        require(step not in records,
                f"duplicate restart boundary kt={step}: {path}")
        records[step] = path
    return records, ignored


def validate_inventory(observed_steps, *, plant: str | None = None) -> dict:
    observed = set(int(step) for step in observed_steps)
    if plant == "missing-boundary":
        require(set(REQUIRED_STEPS) <= observed,
                "missing-boundary plant needs a complete virtual inventory")
        observed.remove(REQUIRED_STEPS[-1])
    missing = sorted(set(REQUIRED_STEPS) - observed)
    unexpected = sorted(observed - set(REQUIRED_STEPS))
    if missing:
        raise GateError(
            f"missing {len(missing)} daily boundaries; first kt={missing[0]}")
    if unexpected:
        raise GateError(
            f"unexpected restart boundaries begin at kt={unexpected[0]}")
    return {
        "required_count": len(REQUIRED_STEPS),
        "observed_count": len(observed),
        "missing_steps": missing,
        "unexpected_steps": unexpected,
    }


def validate_required_variables(names, *, plant: str | None = None) -> None:
    available = set(names)
    if plant == "required-variable":
        require(set(REQUIRED_VARIABLES) <= available,
                "required-variable plant needs a complete virtual schema")
        available.remove("dissl")
    missing = sorted(set(REQUIRED_VARIABLES) - available)
    require(not missing, f"restart schema missing variables {missing}")


def _namelist_integer(path: Path, name: str) -> int | None:
    if not path.is_file():
        return None
    pattern = re.compile(
        rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*([0-9]+)", re.MULTILINE)
    match = pattern.search(path.read_text())
    return None if match is None else int(match.group(1))


def _variable_signature(variable) -> dict:
    return {
        "dimensions": list(variable.dimensions),
        "shape": list(variable.shape),
        "dtype": str(variable.dtype),
    }


def _raw_array(variable) -> np.ndarray:
    variable.set_auto_maskandscale(False)
    return np.asarray(variable[:])


def _array_digest(variable) -> str:
    values = np.ascontiguousarray(_raw_array(variable))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(repr(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def require_bit_identical(left: np.ndarray, right: np.ndarray,
                          label: str) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    require(left.dtype == right.dtype and left.shape == right.shape,
            f"{label}: dtype/shape mismatch "
            f"{(left.dtype, left.shape)} != {(right.dtype, right.shape)}")
    require(np.array_equal(left, right, equal_nan=True),
            f"{label}: arrays are not bit-identical")


def validate_ready_stamp(root: Path, stamp_path: Path) -> dict:
    """Verify the acquisition's closed 360-file SHA manifest and stamp."""
    require(stamp_path.is_file(), f"missing acquisition stamp {stamp_path}")
    parts = stamp_path.read_text().strip().split()
    require(len(parts) == 3,
            f"{stamp_path}: expected SHA, producer commit, and manifest name")
    expected_manifest_sha, producer_commit, manifest_name = parts
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha)),
            f"{stamp_path}: invalid manifest SHA-256")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", producer_commit)),
            f"{stamp_path}: invalid producer commit")
    manifest = root / manifest_name
    require(manifest.is_file(), f"missing acquisition manifest {manifest}")
    require(sha256(manifest) == expected_manifest_sha,
            f"{manifest}: checksum disagrees with {stamp_path}")
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split()
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                f"{manifest}: malformed checksum line {line!r}")
        require(fields[1] not in entries,
                f"{manifest}: duplicate path {fields[1]}")
        entries[fields[1]] = fields[0]
    expected_names = {restart_path(root, step).name for step in REQUIRED_STEPS}
    require(set(entries) == expected_names,
            f"{manifest}: checksum paths do not equal the 360 required restarts")
    mismatches = [name for name, digest in entries.items()
                  if sha256(root / name) != digest]
    require(not mismatches,
            "acquisition checksum mismatch begins at "
            f"{mismatches[0] if mismatches else '<none>'}")
    return {
        "path": str(stamp_path), "sha256": sha256(stamp_path),
        "manifest": str(manifest), "manifest_sha256": expected_manifest_sha,
        "producer_commit": producer_commit, "verified_file_count": len(entries),
    }


def _template_schema(monthly_root: Path) -> tuple[dict, Path]:
    path = restart_path(monthly_root, MONTHLY_STEPS[0])
    require(path.is_file(), f"missing monthly schema reference {path}")
    with Dataset(path, "r") as dataset:
        validate_required_variables(dataset.variables)
        schema = {
            name: _variable_signature(dataset.variables[name])
            for name in REQUIRED_VARIABLES
        }
    return schema, path


def _inspect_restart(path: Path, expected_step: int,
                     template: dict) -> list[str]:
    errors: list[str] = []
    try:
        with Dataset(path, "r") as dataset:
            missing = sorted(set(REQUIRED_VARIABLES) - set(dataset.variables))
            if missing:
                errors.append(f"{path.name}: missing variables {missing}")
            for name in REQUIRED_VARIABLES:
                if name not in dataset.variables:
                    continue
                actual = _variable_signature(dataset.variables[name])
                if actual != template[name]:
                    errors.append(
                        f"{path.name}: {name} signature {actual} != "
                        f"template {template[name]}")
            if "kt" not in dataset.variables:
                errors.append(f"{path.name}: missing scalar kt")
            else:
                actual_step = int(np.asarray(dataset.variables["kt"][...]))
                if actual_step != expected_step:
                    errors.append(
                        f"{path.name}: scalar kt={actual_step} != "
                        f"filename step {expected_step}")
            if "adatrj" not in dataset.variables:
                errors.append(f"{path.name}: missing scalar adatrj")
            else:
                actual_day = float(np.asarray(dataset.variables["adatrj"][...]))
                expected_day = expected_step / STEPS_PER_DAY
                if actual_day != expected_day:
                    errors.append(
                        f"{path.name}: adatrj={actual_day} != day "
                        f"{expected_day}")
    except OSError as error:
        errors.append(f"{path.name}: cannot open NetCDF restart: {error}")
    return errors


def _compare_overlap(daily_path: Path, monthly_path: Path) -> dict:
    unequal: list[str] = []
    digests: dict[str, dict[str, str]] = {}
    with Dataset(daily_path, "r") as daily, Dataset(monthly_path, "r") as monthly:
        validate_required_variables(daily.variables)
        validate_required_variables(monthly.variables)
        for name in REQUIRED_VARIABLES:
            daily_values = _raw_array(daily.variables[name])
            monthly_values = _raw_array(monthly.variables[name])
            daily_digest = _array_digest(daily.variables[name])
            monthly_digest = _array_digest(monthly.variables[name])
            digests[name] = {
                "daily": daily_digest,
                "monthly": monthly_digest,
            }
            try:
                require_bit_identical(
                    daily_values, monthly_values,
                    f"{daily_path.name}/{name} monthly overlap")
            except GateError:
                unequal.append(name)
    return {
        "daily_path": str(daily_path),
        "monthly_path": str(monthly_path),
        "daily_file_sha256": sha256(daily_path),
        "monthly_file_sha256": sha256(monthly_path),
        "bit_identical": not unequal,
        "unequal_variables": unequal,
        "variable_sha256": digests,
    }


def audit_record(daily_root: Path, monthly_root: Path,
                 ready_stamp: Path | None = None,
                 worktree: dict | None = None) -> dict:
    records, ignored = inventory(daily_root)
    observed_steps = sorted(records)
    missing_steps = sorted(set(REQUIRED_STEPS) - set(observed_steps))
    unexpected_steps = sorted(set(observed_steps) - set(REQUIRED_STEPS))
    errors: list[str] = []

    stamp_report = None
    if ready_stamp is not None:
        try:
            stamp_report = validate_ready_stamp(daily_root, ready_stamp)
        except GateError as error:
            errors.append(str(error))

    if missing_steps:
        errors.append(
            f"missing {len(missing_steps)} daily boundaries; first "
            f"kt={missing_steps[0]} day={missing_steps[0] // STEPS_PER_DAY}")
    if unexpected_steps:
        errors.append(
            f"unexpected restart boundaries begin at kt={unexpected_steps[0]}")

    namelist = daily_root / "namelist_cfg"
    nn_itend = _namelist_integer(namelist, "nn_itend")
    nn_stock = _namelist_integer(namelist, "nn_stock")
    if nn_itend is None:
        errors.append(f"{namelist}: nn_itend is absent")
    elif nn_itend < REQUIRED_STEPS[-1]:
        errors.append(
            f"nn_itend={nn_itend} stops before required {REQUIRED_STEPS[-1]}")
    if nn_stock != STEPS_PER_DAY:
        errors.append(
            f"nn_stock={nn_stock} != daily cadence {STEPS_PER_DAY}")

    schema_errors: list[str] = []
    try:
        template, template_path = _template_schema(monthly_root)
    except GateError as error:
        template = {}
        template_path = restart_path(monthly_root, MONTHLY_STEPS[0])
        schema_errors.append(str(error))
    if template:
        for step, path in sorted(records.items()):
            schema_errors.extend(_inspect_restart(path, step, template))
    errors.extend(schema_errors)

    overlaps: dict[str, dict] = {}
    missing_overlaps: list[int] = []
    overlap_errors: list[str] = []
    for step in MONTHLY_STEPS:
        daily_path = restart_path(daily_root, step)
        monthly_path = restart_path(monthly_root, step)
        if not daily_path.is_file():
            missing_overlaps.append(step)
            continue
        if not monthly_path.is_file():
            overlap_errors.append(f"missing monthly control {monthly_path}")
            continue
        try:
            comparison = _compare_overlap(daily_path, monthly_path)
        except (GateError, OSError) as error:
            overlap_errors.append(f"kt={step}: {error}")
            continue
        overlaps[str(step)] = comparison
        if not comparison["bit_identical"]:
            overlap_errors.append(
                f"kt={step}: unequal variables "
                f"{comparison['unequal_variables']}")
    if missing_overlaps:
        errors.append(
            f"missing {len(missing_overlaps)} of 12 monthly overlap "
            f"boundaries; first kt={missing_overlaps[0]}")
    errors.extend(overlap_errors)

    admitted = not errors
    first_step = observed_steps[0] if observed_steps else None
    last_step = observed_steps[-1] if observed_steps else None
    return {
        "format": "nemo-testcase-l2-gyre-round131-daily-record-audit-v1",
        "worktree": worktree,
        "status": "ADMITTED" if admitted else "STOPPED_FOR_RECORD",
        "admitted": admitted,
        "daily_root": str(daily_root),
        "monthly_control_root": str(monthly_root),
        "requirements": {
            "steps_per_day": STEPS_PER_DAY,
            "days": YEAR_DAYS,
            "required_boundary_count": len(REQUIRED_STEPS),
            "first_required_step": REQUIRED_STEPS[0],
            "last_required_step": REQUIRED_STEPS[-1],
            "required_variables": list(REQUIRED_VARIABLES),
            "monthly_overlap_steps": list(MONTHLY_STEPS),
        },
        "inventory": {
            "observed_count": len(observed_steps),
            "first_observed_step": first_step,
            "last_observed_step": last_step,
            "observed_steps": observed_steps,
            "missing_count": len(missing_steps),
            "missing_steps": missing_steps,
            "missing_days": [step // STEPS_PER_DAY for step in missing_steps],
            "unexpected_steps": unexpected_steps,
            "ignored_restart_names": ignored,
        },
        "namelist": {
            "path": str(namelist),
            "sha256": sha256(namelist) if namelist.is_file() else None,
            "nn_itend": nn_itend,
            "nn_stock": nn_stock,
        },
        "acquisition_stamp": stamp_report,
        "schema": {
            "template_path": str(template_path),
            "checked_file_count": len(records) if template else 0,
            "errors": schema_errors,
        },
        "monthly_overlap": {
            "required_count": len(MONTHLY_STEPS),
            "compared_count": len(overlaps),
            "missing_steps": missing_overlaps,
            "comparisons": overlaps,
            "errors": overlap_errors,
        },
        "errors": errors,
    }


def self_check(plant: str | None) -> int:
    if plant is None:
        validate_inventory(REQUIRED_STEPS)
        validate_required_variables(REQUIRED_VARIABLES)
        print("SELF-CHECK OK: complete 360-boundary inventory and 18-field "
              "schema pass")
        return 0
    try:
        if plant == "missing-boundary":
            validate_inventory(REQUIRED_STEPS, plant=plant)
        elif plant == "required-variable":
            validate_required_variables(REQUIRED_VARIABLES, plant=plant)
        elif plant == "monthly-overlap-ulp":
            original = np.array([1.0, 2.0], dtype=np.float64)
            moved = original.copy()
            moved[1] = np.nextafter(moved[1], np.inf)
            require_bit_identical(original, moved, "monthly-overlap-ulp plant")
        else:  # pragma: no cover - argparse constrains the value
            raise AssertionError(plant)
    except GateError as error:
        print(f"REFUSE Round-131 {plant}: {error}", file=sys.stderr)
        print(f"STATUS PLANT-FIRED: {plant}")
        return 1
    print(f"REFUSE Round-131 plant {plant} did not fire", file=sys.stderr)
    return 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--self-check", action="store_true")
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--monthly-root", type=Path,
                        default=DEFAULT_MONTHLY_ROOT)
    parser.add_argument("--ready-stamp", type=Path,
                        help="acquisition stamp closing a 360-file SHA manifest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.plant)
    if args.plant is not None:
        parser.error("--plant is valid only with --self-check")
    if args.expect_commit is None:
        parser.error("--audit requires --expect-commit")
    if args.output is None:
        parser.error("--audit requires --output")

    stamp = worktree_stamp(args.expect_commit)
    report = audit_record(args.daily_root, args.monthly_root,
                          ready_stamp=args.ready_stamp, worktree=stamp)
    report["tool"] = stamp
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    inventory_report = report["inventory"]
    overlap_report = report["monthly_overlap"]
    print("required observed missing first_observed last_observed "
          "monthly_overlaps")
    print(f"{report['requirements']['required_boundary_count']:8d} "
          f"{inventory_report['observed_count']:8d} "
          f"{inventory_report['missing_count']:7d} "
          f"{str(inventory_report['first_observed_step']):>14s} "
          f"{str(inventory_report['last_observed_step']):>13s} "
          f"{overlap_report['compared_count']:16d}")
    if not report["admitted"]:
        for error in report["errors"]:
            print(f"REFUSE Round-131 record: {error}", file=sys.stderr)
        print("STATUS STOPPED-FOR-RECORD")
        return 4
    print("STATUS ADMITTED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        sys.exit(2)
