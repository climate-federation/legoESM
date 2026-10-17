#!/usr/bin/env python3
"""Audit whether stored GYRE runs can support the registered momentum budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

START_DAY = 30
END_DAY = 180
FULL_WINDOW = (1, END_DAY)
LEGO_FULL_FIELDS = (
    "temperature_c_3d",
    "salinity_psu_3d",
    "u_m_s_3d",
    "v_m_s_3d",
    "tke_m2_s2_3d",
    "layer_thickness_m_3d",
)
NEMO_REQUIREMENTS = {
    "T": ("votemper", "vosaline", "vovvle3t", "sossheig"),
    "U": ("vozocrtx",),
    "V": ("vomecrty",),
    "W": ("votkeavt", "votkeavm", "vovecrtz"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_legoesm_keys() -> tuple[str, ...]:
    return tuple(
        key
        for name in LEGO_FULL_FIELDS
        for key in (name, f"{name}_daily_mean")
    )


def audit_legoesm_archive(root: Path) -> dict:
    """Return a fail-closed audit of the matched 3-D legoESM archive."""
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    manifest = None
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            errors.append("manifest status is not complete")
        if manifest.get("completed_days", 0) < END_DAY:
            errors.append(f"manifest has fewer than {END_DAY} completed days")
        if manifest.get("per_step") is not True:
            errors.append("manifest does not certify observer-free per-step integration")
        if manifest.get("full_state_window") != list(FULL_WINDOW):
            errors.append(
                f"manifest full_state_window is not {list(FULL_WINDOW)}")
        if manifest.get("full_state_fields") != list(LEGO_FULL_FIELDS):
            errors.append("manifest full_state_fields do not match the registered set")
        namelist = manifest.get("namelist", {})
        if namelist.get("nn_GYRE") != 4:
            errors.append("manifest does not resolve nn_GYRE=4")
        if namelist.get("ln_dynldf_blp") is not True:
            errors.append("manifest is not the r4_smag momentum deck")

    required = set(required_legoesm_keys())
    day_rows = []
    for day in (1, START_DAY, END_DAY):
        path = root / f"day{day:04d}.npz"
        row = {"day": day, "path": str(path), "missing": [], "non_finite": []}
        if not path.is_file():
            errors.append(f"missing {path.name}")
            row["missing"] = sorted(required)
        else:
            with np.load(path) as archive:
                row["missing"] = sorted(required - set(archive.files))
                for key in sorted(required & set(archive.files)):
                    if not bool(np.all(np.isfinite(archive[key]))):
                        row["non_finite"].append(key)
            if row["missing"]:
                errors.append(
                    f"{path.name} lacks {', '.join(row['missing'])}")
            if row["non_finite"]:
                errors.append(
                    f"{path.name} has non-finite {', '.join(row['non_finite'])}")
            row["sha256"] = sha256(path)
        day_rows.append(row)
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "required_fields": list(LEGO_FULL_FIELDS),
        "sampled_days": day_rows,
        "errors": errors,
        "status": "PASS" if not errors else "REFUSE",
    }


def audit_nemo_archive(root: Path) -> dict:
    """Check tile count, variables, cadence, and placement metadata."""
    from netCDF4 import Dataset

    errors: list[str] = []
    grids = {}
    for grid, variables in NEMO_REQUIREMENTS.items():
        paths = sorted(root.glob(f"*_grid_{grid}_*.nc"))
        if len(paths) != 8:
            errors.append(f"grid_{grid} has {len(paths)} tiles, expected 8")
        tile_rows = []
        for path in paths:
            with Dataset(path) as dataset:
                missing = sorted(set(variables) - set(dataset.variables))
                time_count = len(dataset.dimensions.get("time_counter", ()))
                has_position = "DOMAIN_position_first" in dataset.ncattrs()
                has_global = "DOMAIN_size_global" in dataset.ncattrs()
                if missing:
                    errors.append(f"{path.name} lacks {', '.join(missing)}")
                if time_count < END_DAY:
                    errors.append(
                        f"{path.name} has {time_count} daily records, expected >= {END_DAY}")
                if not (has_position and has_global):
                    errors.append(f"{path.name} lacks DOMAIN placement attributes")
                tile_rows.append({
                    "name": path.name,
                    "time_records": time_count,
                    "missing_variables": missing,
                    "domain_position_first": has_position,
                    "domain_size_global": has_global,
                })
        grids[grid] = tile_rows
    manifest = root / "manifest.sha256"
    if not manifest.is_file():
        errors.append("missing manifest.sha256")
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "requirements": {key: list(value) for key, value in NEMO_REQUIREMENTS.items()},
        "grids": grids,
        "errors": errors,
        "status": "PASS" if not errors else "REFUSE",
    }


def build_report(legoesm_root: Path, nemo_root: Path) -> dict:
    legoesm = audit_legoesm_archive(legoesm_root)
    nemo = audit_nemo_archive(nemo_root)
    admitted = legoesm["status"] == nemo["status"] == "PASS"
    return {
        "schema": "legoesm-gyre-hires-momentum-budget-preflight-v1",
        "registered_window_days": [START_DAY, END_DAY],
        "leading_field": "native surface V",
        "leading_region": "western boundary wet-span bin",
        "legoesm": legoesm,
        "nemo": nemo,
        "verdict": "PASS" if admitted else "ACQUISITION_NEEDED",
        "disposition": (
            "archives admit the registered term attribution"
            if admitted else
            "refuse term attribution until every registered 3-D field is present"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legoesm-root", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.legoesm_root, args.nemo_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
