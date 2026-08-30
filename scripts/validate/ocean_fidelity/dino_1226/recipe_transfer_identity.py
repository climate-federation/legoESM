#!/usr/bin/env python3
"""Fail-closed catalog reachability gate for the faithful DINO MLF card.

The oracle path is ``dino_config_for_recipe`` +
``dino_lat_lon_model_config``.  The candidate path is the public recipe
catalog's ``get_recipe`` + ``assemble_ocean_config``.  A frozen ownership
partition prevents omitted scheme fields from being smuggled through the
setup, and the recursive comparison rejects any unequal/missing/extra leaf.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np
from legoesm.grids.latlon import create_mercator_grid
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
)
from legoesm.ocean.recipes import assemble_ocean_config, get_recipe

ORACLE_CARD = "nemo_dino_kamm_mlf"
CATALOG_RECIPE = "nemo_dino_kamm_mlf_v1"
NEGATIVE_RECIPE = "nemo_dino_v1"
ROOT = Path(__file__).resolve().parents[4]

# Frozen complement of the catalog-owned fields on LatLonCGridOceanConfig.
# These are dimensional coefficients, nested physics/setup objects, constants,
# runtime tolerances, and optional arrays.  Any new resolved field is neither
# silently setup-owned nor silently recipe-owned: the partition gate fails and
# requires an explicit decision here or in the catalog entry.
SETUP_OWNED_FIELDS = frozenset({
    "A_h",
    "A_h_cap_boost",
    "A_h_cap_lat_deg",
    "A_h_cap_width_deg",
    "A_h_cos_power",
    "A_h_lat_profile",
    "A_h_lat_profile_v",
    "A_h_merid",
    "A_v",
    "B_h",
    "B_h_barotropic",
    "C_leith",
    "C_smag",
    "C_smag_lap",
    "K_bih",
    "K_h",
    "K_v",
    "R_earth",
    "S_ref",
    "ab2_epsilon",
    "backscatter",
    "barotropic_chebyshev_degree",
    "barotropic_diffusion_dt_ref",
    "barotropic_div_damp",
    "barotropic_implicit_pcg_fixed_iters",
    "barotropic_implicit_pcg_maxiter",
    "barotropic_implicit_pcg_residual_tol",
    "barotropic_implicit_pcg_tol",
    "barotropic_wide_halo_chunk",
    "bebt",
    "bottom_drag_bbl_thickness",
    "bottom_drag_bg_velocity",
    "bottom_drag_cd0",
    "bottom_drag_cdmax",
    "bottom_drag_ke0",
    "bottom_drag_r",
    "bottom_drag_z0",
    "c_sw",
    "constants",
    "dt_mom_ratio",
    "eos_linear",
    "freeze_floor_temp_c",
    "freezing",
    "g",
    "gm_redi",
    "hyperdiff_coeff",
    "max_abs_eta_m",
    "maxvel_barotropic",
    "min_water_column_m",
    "omega",
    "omp25",
    "physics",
    "polar_filter_cutoff_lat_deg",
    "polar_filter_max_wave_speed",
    "polar_filter_safety_factor",
    "prescribed_flow",
    "qg_leith_coeff",
    "qg_leith_deformation_radius_m",
    "rho_0",
    "rigid_lid_cg_maxiter",
    "rigid_lid_cg_tol",
    "runoff_depth_spread_m",
    "runoff_depth_spread_map",
    "salinity_max_psu",
    "salinity_min_psu",
    "slope_foot_alpha",
    "slope_foot_n_levels",
    "slope_foot_threshold",
    "smag_cfl_safety",
    "temperature_max_c",
    "temperature_min_c",
    "tidal_forcing",
    "wall_grid_filter_rate_s",
    "weno_divergence_smoothness",
})


class OwnershipCollisionError(RuntimeError):
    """Recipe and setup attempted to own the same resolved field."""


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def flatten_resolved(value: Any, path: str = "config") -> dict[str, dict[str, Any]]:
    """Flatten a nested config into exact, JSON-safe leaf records."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        rows: dict[str, dict[str, Any]] = {}
        for field in dataclasses.fields(value):
            rows.update(flatten_resolved(getattr(value, field.name),
                                         f"{path}.{field.name}"))
        return rows
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        rows = {}
        for name in value._fields:
            rows.update(flatten_resolved(getattr(value, name), f"{path}.{name}"))
        return rows
    if isinstance(value, Mapping):
        rows = {}
        for name in sorted(value, key=str):
            rows.update(flatten_resolved(value[name], f"{path}.{name}"))
        return rows
    if isinstance(value, (list, tuple)):
        rows = {}
        for index, item in enumerate(value):
            rows.update(flatten_resolved(item, f"{path}[{index}]"))
        return rows
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        if array.ndim:
            return {path: {
                "kind": "array",
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }}
        return {path: {
            "kind": "scalar_array",
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "dtype": str(array.dtype),
            "shape": [],
            "value": _json_scalar(array.item()),
        }}
    return {path: {
        "kind": "scalar",
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": _json_scalar(value),
    }}


def diff_rows(oracle: Any, candidate: Any) -> tuple[
        list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    oracle_rows = flatten_resolved(oracle)
    candidate_rows = flatten_resolved(candidate)
    differences = []
    for path in sorted(set(oracle_rows) | set(candidate_rows)):
        if path not in oracle_rows:
            differences.append({"path": path, "status": "extra_candidate",
                                "candidate": candidate_rows[path]})
        elif path not in candidate_rows:
            differences.append({"path": path, "status": "missing_candidate",
                                "oracle": oracle_rows[path]})
        elif oracle_rows[path] != candidate_rows[path]:
            differences.append({"path": path, "status": "unequal",
                                "oracle": oracle_rows[path],
                                "candidate": candidate_rows[path]})
    return differences, oracle_rows, candidate_rows


def assemble_owned(recipe_bundle: Mapping[str, Any], config_cls: type,
                   setup_params: Mapping[str, Any]):
    collisions = sorted(set(recipe_bundle) & set(setup_params))
    if collisions:
        raise OwnershipCollisionError(
            f"recipe/setup ownership collision at {collisions}")
    return assemble_ocean_config(
        recipe_bundle, config_cls, **dict(setup_params))


def run_identity(oracle_card: str = ORACLE_CARD,
                 catalog_recipe: str = CATALOG_RECIPE,
                 negative_recipe: str = NEGATIVE_RECIPE) -> dict[str, Any]:
    if jax.default_backend() != "cpu" or not bool(jax.config.x64_enabled):
        raise RuntimeError("recipe identity gate requires CPU/JAX fp64")

    grid = create_mercator_grid(
        n_lon=16, lat_max_deg=70.0, lon_west_deg=0.0, lon_east_deg=50.0)
    oracle_cfg = dino_config_for_recipe(oracle_card)
    oracle, _ = dino_lat_lon_model_config(grid, oracle_cfg)
    bundle = get_recipe(catalog_recipe, "latlon")

    resolved_fields = set(type(oracle).flat_fields())
    recipe_fields = set(bundle)
    setup_fields = set(SETUP_OWNED_FIELDS)
    ownership_collisions = sorted(recipe_fields & setup_fields)
    ownership_missing = sorted(resolved_fields - recipe_fields - setup_fields)
    ownership_extra = sorted((recipe_fields | setup_fields) - resolved_fields)
    if ownership_collisions or ownership_missing or ownership_extra:
        raise RuntimeError(
            "invalid frozen ownership partition: "
            f"collisions={ownership_collisions}, missing={ownership_missing}, "
            f"extra={ownership_extra}")

    # Only setup-owned leaves are sourced from the established DINO setup.
    # No catalog-owned value is copied from the oracle into this mapping.
    setup_params = {name: oracle.flat_get(name) for name in sorted(setup_fields)}
    candidate = assemble_owned(bundle, type(oracle), setup_params)
    differences, oracle_rows, candidate_rows = diff_rows(oracle, candidate)

    negative_bundle = get_recipe(negative_recipe, "latlon")
    negative = assemble_owned(negative_bundle, type(oracle), setup_params)
    negative_differences, _, _ = diff_rows(oracle, negative)

    planted = candidate.replace_flat(
        asselin_gamma=float(np.nextafter(candidate.asselin_gamma, np.inf)))
    plant_differences, _, _ = diff_rows(candidate, planted)
    plant_paths = [row["path"] for row in plant_differences]
    expected_plant_path = "config.asselin_gamma"
    plant_fired = plant_paths == [expected_plant_path]

    collision_fired = False
    collision_message = ""
    collision_setup = dict(setup_params)
    collision_setup["outer_integrator"] = oracle.outer_integrator
    try:
        assemble_owned(bundle, type(oracle), collision_setup)
    except OwnershipCollisionError as error:
        collision_fired = True
        collision_message = str(error)

    passed = (not differences and bool(negative_differences) and plant_fired
              and collision_fired)
    return {
        "schema": "dino_recipe_transfer_identity_v1",
        "oracle_card": oracle_card,
        "catalog_recipe": catalog_recipe,
        "negative_recipe": negative_recipe,
        "catalog_bundle": {name: _json_scalar(value)
                           for name, value in sorted(bundle.items())},
        "negative_bundle": {name: _json_scalar(value)
                            for name, value in sorted(negative_bundle.items())},
        "grid": {"n_lon": 16, "lat_max_deg": 70.0,
                 "lon_west_deg": 0.0, "lon_east_deg": 50.0},
        "ownership": {
            "recipe_fields": sorted(recipe_fields),
            "setup_fields": sorted(setup_fields),
            "collisions": ownership_collisions,
            "missing": ownership_missing,
            "extra": ownership_extra,
        },
        "resolved_leaf_count": len(oracle_rows),
        "oracle_leaves": oracle_rows,
        "catalog_leaves": candidate_rows,
        "identity_differences": differences,
        "negative_control": {
            "difference_count": len(negative_differences),
            "differences": negative_differences,
        },
        "plant": {
            "field": "asselin_gamma",
            "expected_path": expected_plant_path,
            "difference_paths": plant_paths,
            "fired": plant_fired,
        },
        "ownership_collision_plant": {
            "field": "outer_integrator",
            "fired": collision_fired,
            "message": collision_message,
        },
        "verdict": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-card", default=ORACLE_CARD)
    parser.add_argument("--catalog-recipe", default=CATALOG_RECIPE)
    parser.add_argument("--negative-recipe", default=NEGATIVE_RECIPE)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual_commit != args.producer_commit:
        raise RuntimeError(
            f"producer mismatch: expected {args.producer_commit}, got {actual_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError(f"tracked-dirty producer:\n{dirty}")

    result = run_identity(
        args.oracle_card, args.catalog_recipe, args.negative_recipe)
    result["receipt"] = {
        "producer_commit": actual_commit,
        "session_id": args.session_id,
        "instrument_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "jax_backend": jax.default_backend(),
        "jax_x64": bool(jax.config.x64_enabled),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"CONFIG_IDENTITY={result['verdict']} "
          f"diff_rows={len(result['identity_differences'])}")
    print("PLANT=FIRED" if result["plant"]["fired"] else "PLANT=FAILED")
    print("OWNERSHIP_COLLISION=FIRED" if
          result["ownership_collision_plant"]["fired"] else
          "OWNERSHIP_COLLISION=FAILED")
    print("NEGATIVE_CONTROL=DIFF "
          f"rows={result['negative_control']['difference_count']}" if
          result["negative_control"]["difference_count"] else
          "NEGATIVE_CONTROL=FAILED")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
