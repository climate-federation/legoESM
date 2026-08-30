#!/usr/bin/env python3
"""From-rest legoESM arm for the preregistered DINO T1 battery.

This driver has no restart, bridge, or NEMO-state input.  It constructs the
NEMO-grid DINO card through legoESM's public analytic grid, vertical-coordinate,
and from-rest state factories.  A NEMO mesh may be supplied only to the already
recorded live reducer; it never enters state construction and is stamped as a
diagnostic dependency.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import ensure_geometry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_config_for_recipe,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_lat_lon_surface_forcing_arrays,
    dino_lat_lon_vertical,
    dino_step_surface_forcing,
    nemo_faithful_dino_config,
)
from legoesm.ocean.vertical import NemoEENBarotropicOperands

import kamm_twin_90d as twin
from recipe_transfer_identity import flatten_resolved

SCHEMA = "dino_standalone_20y_v1"
REDUCER_CONVENTION_SCHEMA = "dino_standalone_reducer_frame_v1"
RECIPE = "nemo_dino_kamm_mlf"
DT_SECONDS = 2700.0
STEPS_PER_DAY = 32
DAYS_PER_YEAR = 360
YEARS = 20
MEMBERS = tuple(range(6))
PERTURB_EPS = 1.0e-14


def claim_admission_reasons() -> tuple[str, ...]:
    """Known build blockers that make a 20-year science arm inadmissible."""
    reasons = []
    cfg = dino_config_for_recipe(RECIPE)
    if not (
            cfg.barotropic_after_reconcile == "nemo_mlf_baro_corr"
            and cfg.barotropic_cold_start_after_reconcile
            == "nemo_mlf_baro_corr"):
        reasons.append("cold-start corrector is not source-equivalent")
    return tuple(reasons)


def sample_days() -> tuple[int, ...]:
    """The frozen 15 annual plus 60 monthly T1 sample dates."""
    annual = tuple(DAYS_PER_YEAR * year for year in range(1, 16))
    monthly = tuple(range(15 * DAYS_PER_YEAR + 30,
                          YEARS * DAYS_PER_YEAR + 1, 30))
    days = annual + monthly
    if len(days) != 75 or len(set(days)) != 75 or days[-1] != 7200:
        raise RuntimeError("internal T1 sample schedule is not 75 unique dates")
    return days


def seed_for_member(member: int) -> int | None:
    if member not in MEMBERS:
        raise ValueError(f"member must be one of {MEMBERS}, got {member}")
    return None if member == 0 else member


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def state_hashes(state) -> dict[str, str | None]:
    """Hash every mutable state channel used by the DINO integrator."""
    names = (
        "T", "S", "eta", "u", "v", "T_before", "S_before", "eta_before",
        "u_before", "v_before", "tke", "tke_avm", "tke_avt", "tke_dissl",
        "tau_x_before_t", "tau_y_before_t",
    )
    out: dict[str, str | None] = {}
    for name in names:
        value = getattr(state, name, None)
        if value is None:
            out[name] = None
        else:
            out[name] = sha256_array(value.data)
    return out


def _expected_reducer_input_shapes(
        native_t_shape: tuple[int, int, int]) -> dict[str, tuple[int, ...]]:
    """Standalone field shapes corresponding to a native NEMO T frame.

    NEMO's reducer frame contains two horizontal halo rings and its terminal
    ``jpk`` level. The public analytic DINO grid contains only the physical
    core and ``jpkm1`` active levels; C-grid U/V retain one extra face.
    """
    ny, nx, nz = native_t_shape
    if ny <= 4 or nx <= 4 or nz <= 1:
        raise ValueError(
            f"native reducer frame is too small: {native_t_shape}")
    core = (ny - 4, nx - 4, nz - 1)
    return {
        "T": core,
        "S": core,
        "eta": core[:2],
        "u": (core[0], core[1] + 1, core[2]),
        "v": (core[0] + 1, core[1], core[2]),
        "land_mask": core[:2],
    }


def expand_standalone_reducer_inputs(
        fields: dict[str, Any], land_mask: Any,
        native_t_shape: tuple[int, int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map analytic-core fields onto the recorded NEMO reducer frame.

    The recorded reducer remains unchanged. This adapter owns only geometry
    convention: two horizontal NEMO halo rings, the dry terminal ``jpk``
    level, and the C-grid U face mapping. Every supplied capture field is
    asserted before any allocation so a bad arm fails at day 0.
    """
    expected = _expected_reducer_input_shapes(native_t_shape)
    supplied: dict[str, Any] = dict(fields)
    supplied["land_mask"] = land_mask
    missing = sorted(set(expected) - set(supplied))
    if missing:
        raise ValueError(
            "standalone reducer input is missing fields: " + ", ".join(missing))
    actual = {name: tuple(np.shape(supplied[name])) for name in expected}
    bad = {name: {"actual": actual[name], "expected": expected[name]}
           for name in expected if actual[name] != expected[name]}
    if bad:
        rows = "; ".join(
            f"{name} actual={row['actual']} expected={row['expected']}"
            for name, row in sorted(bad.items()))
        raise ValueError("standalone reducer shape contract failed: " + rows)

    ny, nx, nz = native_t_shape

    def scalar3(name: str) -> np.ndarray:
        core = np.asarray(supplied[name], dtype=np.float64)
        native = np.zeros(native_t_shape, dtype=np.float64)
        native[2:-2, 2:-2, :-1] = core
        # NEMO's zonal halo columns are periodic copies of the physical core.
        # Meridional halos remain reducer-dry through ``land_mask``.
        native[2:-2, :2, :-1] = core[:, -2:, :]
        native[2:-2, -2:, :-1] = core[:, :2, :]
        return native

    core_land = np.asarray(supplied["land_mask"], dtype=np.float64)
    native_land = np.zeros((ny, nx), dtype=np.float64)
    native_land[2:-2, 2:-2] = core_land
    native_land[2:-2, :2] = core_land[:, -2:]
    native_land[2:-2, -2:] = core_land[:, :2]

    # The recorded reducer first selects lU[:, 1:nx+1]. Embed the analytic
    # grid's east/native faces u[:, 1:] so that this selection's [2:-2] core is
    # exactly the model core; face column 0 is intentionally unused.
    core_u = np.asarray(supplied["u"], dtype=np.float64)[:, 1:, :]
    native_u_faces = np.zeros((ny, nx + 1, nz), dtype=np.float64)
    native_u_faces[2:-2, 3:nx - 1, :-1] = core_u
    native_u_faces[2:-2, 1:3, :-1] = core_u[:, -2:, :]
    native_u_faces[2:-2, nx - 1:nx + 1, :-1] = core_u[:, :2, :]

    expanded = {
        "T": scalar3("T"),
        "S": scalar3("S"),
        "u": native_u_faces,
        "land_mask": native_land,
    }
    receipt = {
        "schema": REDUCER_CONVENTION_SCHEMA,
        "native_t_shape": list(native_t_shape),
        "standalone_input_shapes": {
            name: list(actual[name]) for name in sorted(actual)},
        "asserted_input_names": sorted(expected),
        "horizontal_convention": (
            "standalone physical core -> NEMO [2:-2,2:-2]; two periodic "
            "zonal halo rings restored; meridional halos reducer-dry"),
        "vertical_convention": (
            "standalone jpkm1 -> NEMO [:-1]; terminal jpk level padded dry"),
        "u_convention": (
            "standalone u[:,1:] east/native faces -> reducer-selected "
            "lU[:,1:nx+1][:,2:-2]"),
        "expanded_reducer_shapes": {
            name: list(value.shape) for name, value in sorted(expanded.items())},
    }
    return expanded, receipt


def build_standalone_reducer_adapter(reducer, fields, land_mask):
    """Return the native-frame adapter plus its fail-fast convention receipt."""
    import acc_thermal_wind as A

    native_t_shape = tuple(A.tmask.shape)
    weight_shapes = {
        "tmask": tuple(A.tmask.shape),
        "umask": tuple(A.umask.shape),
        "e3t0": tuple(A.e3t0.shape),
        "gdept0": tuple(A.gdept0.shape),
        "e3t1d": tuple(A.e3t1d.shape),
        "gdept1d": tuple(A.gdept1d.shape),
        "e2u_col": tuple(A.e2u_col.shape),
    }
    expected_weights = {
        "tmask": native_t_shape,
        "umask": native_t_shape,
        "e3t0": native_t_shape,
        "gdept0": native_t_shape,
        "e3t1d": (native_t_shape[2],),
        "gdept1d": (native_t_shape[2],),
        "e2u_col": (native_t_shape[0],),
    }
    bad_weights = {
        name: {"actual": weight_shapes[name], "expected": expected_weights[name]}
        for name in expected_weights
        if weight_shapes[name] != expected_weights[name]
    }
    if bad_weights:
        raise ValueError(
            f"standalone reducer mesh-weight contract failed: {bad_weights}")

    _, receipt = expand_standalone_reducer_inputs(
        fields, land_mask, native_t_shape)

    # Runtime plant: the exact shape gate that was formerly deferred to day
    # 360 must demonstrably fire before the integration starts.
    try:
        expand_standalone_reducer_inputs(
            fields, np.asarray(land_mask)[:, :-1], native_t_shape)
    except ValueError as exc:
        if "land_mask" not in str(exc):
            raise RuntimeError(
                "standalone reducer shape plant fired on the wrong row") from exc
        receipt["shape_mismatch_plant"] = "FIRED"
    else:  # pragma: no cover - the unit test plants this mismatch
        raise RuntimeError("standalone reducer shape mismatch plant did not fire")

    receipt["mesh_weight_shapes"] = {
        name: list(shape) for name, shape in sorted(weight_shapes.items())}

    def adapted(live):
        adapted_fields, _ = expand_standalone_reducer_inputs(
            {name: live[name] for name in ("T", "S", "eta", "u", "v")},
            live["land_mask"], native_t_shape)
        return reducer(adapted_fields)

    return adapted, receipt


def perturb_temperature(state, member: int):
    """Apply the frozen now-level temperature kick and prove its ownership."""
    seed = seed_for_member(member)
    before = state_hashes(state)
    if seed is None:
        after = before
        receipt = {
            "member": member, "seed": None, "epsilon": PERTURB_EPS,
            "changed_channels": [], "max_abs_delta_T": 0.0,
            "max_relative_delta_T": 0.0,
        }
        return state, receipt

    rng = np.random.default_rng(seed)
    t0 = np.asarray(state.T.data, dtype=np.float64)
    factor = 1.0 + PERTURB_EPS * rng.standard_normal(t0.shape)
    perturbed = jnp.asarray(t0 * factor, dtype=state.T.data.dtype)
    state = state._replace(T=state.T.replace(data=perturbed))
    after = state_hashes(state)
    changed = sorted(name for name in before if before[name] != after[name])
    if changed != ["T"]:
        raise RuntimeError(
            f"temperature perturbation ownership violation: changed={changed}")
    delta = np.asarray(perturbed) - t0
    receipt = {
        "member": member, "seed": seed, "epsilon": PERTURB_EPS,
        "changed_channels": changed,
        "max_abs_delta_T": float(np.max(np.abs(delta))),
        "max_relative_delta_T": float(np.max(
            np.abs(delta) / np.maximum(np.abs(t0), 1.0e-30))),
        "before_hashes": before,
        "after_hashes": after,
    }
    return state, receipt


def git_provenance() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[4]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True).stdout.strip())
    return sha, dirty


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_physics_overrides() -> None:
    forbidden = sorted(
        name for name in os.environ
        if name.startswith("DINO_") or name in {"FP64", "LEGOESM_NEMO_E3T"}
    )
    if forbidden:
        raise SystemExit(
            "REFUSING physics/start overrides in standalone T1 arm: "
            + ", ".join(forbidden))


def attach_analytic_nemo_operands(grid, z_coord, state, cfg):
    """Attach raw-grid operands generated by the public DINO grid recipe.

    Literal faithful kernels consume NEMO's native A2D stagger convention.
    The restart bridge normally carries these fields from ``mesh_mask.nc``;
    T1 cannot.  On DINO's analytic full-step z grid the same operands are
    determined by the public vertical ladder, C-grid metrics, and masks.  This
    constructor supplies those values without reading a NEMO file.
    """
    geom = ensure_geometry(
        grid,
        omega=cfg.omega,
        metric_convention=cfg.metric_convention,
        vface_zonal_metric_evaluation=cfg.vface_zonal_metric_evaluation,
        coriolis_placement=cfg.coriolis_placement,
    )
    dtype = np.asarray(state.T.data).dtype
    tmask = (np.asarray(z_coord.is_active, dtype=bool)
             & (np.asarray(state.land_mask.data) > 0.5)[..., None])
    umask = tmask & np.roll(tmask, -1, axis=1)
    vmask = np.zeros_like(tmask)
    vmask[:-1] = tmask[:-1] & tmask[1:]
    fmask = np.zeros_like(tmask)
    fmask[:-1] = (tmask[:-1]
                  & np.roll(tmask[:-1], -1, axis=1)
                  & tmask[1:]
                  & np.roll(tmask[1:], -1, axis=1))

    n_lat, n_lon, n_levels = tmask.shape
    t_depth = np.abs(np.asarray(z_coord.z_full_ref, dtype=dtype))
    w_depth = np.abs(np.asarray(z_coord.z_half_ref[:-1], dtype=dtype))
    e3t_1d = np.asarray(z_coord.dz_ref, dtype=dtype)
    e3w_1d = np.concatenate((
        np.asarray([2.0 * t_depth[0]], dtype=dtype),
        np.diff(t_depth),
    ))
    if not (t_depth.size == w_depth.size == e3t_1d.size == e3w_1d.size
            == n_levels):
        raise RuntimeError("analytic NEMO vertical operand size mismatch")
    shape3 = (n_lat, n_lon, n_levels)
    gdept = np.broadcast_to(t_depth, shape3).copy()
    gdepw = np.broadcast_to(w_depth, shape3).copy()
    e3t = np.broadcast_to(e3t_1d, shape3).copy()
    e3w = np.broadcast_to(e3w_1d, shape3).copy()
    e3u = e3t.copy()
    e3v = e3t.copy()
    e3f = e3t.copy()

    # Native NEMO A2D stores the east/north/NE stagger associated with each T
    # cell.  legoESM's rich geometry retains one redundant west/south face;
    # slices 1: select the corresponding native face exactly.
    e1t = np.asarray(geom.dx_T, dtype=dtype)
    e2t = np.asarray(geom.dy_T, dtype=dtype)
    e1u = np.asarray(geom.dx_u[:, 1:], dtype=dtype)
    e2u = np.asarray(geom.dy_u[:, 1:], dtype=dtype)
    # ``geom.dx_v`` deliberately zeroes the two closed transport boundaries.
    # NEMO's raw metric arrays stay geometrically positive there and apply the
    # wall through vmask.  QCO reciprocals therefore require the unzeroed
    # native V/F metric evaluated at the true Mercator face latitude.
    v_width = (float(grid.radius) * float(grid.dlon)
               * np.asarray(grid.cos_lat_v[1:], dtype=dtype))
    e1v = np.broadcast_to(v_width[:, None], (n_lat, n_lon)).copy()
    e2v = e1v.copy()
    e1f = e1v.copy()
    e2f = e2v.copy()
    hu = np.sum(e3u * umask, axis=-1)
    hv = np.sum(e3v * vmask, axis=-1)
    hf = np.sum(e3f * fmask, axis=-1)
    ff_f = (2.0 * cfg.omega
            * np.sin(np.asarray(grid.lat_v[1:], dtype=dtype)))[:, None]
    ff_f = np.broadcast_to(ff_f, (n_lat, n_lon)).copy()
    een = NemoEENBarotropicOperands(*(
        jnp.asarray(value, dtype=state.T.data.dtype) for value in (
            ff_f, e3u, e3v, e3f,
            umask.astype(dtype), vmask.astype(dtype), fmask.astype(dtype),
            hu, hv, hf,
            e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f,
        )))
    return z_coord._replace(
        nemo_gdept_0=jnp.asarray(gdept, dtype=state.T.data.dtype),
        nemo_gdepw_0=jnp.asarray(gdepw, dtype=state.T.data.dtype),
        nemo_e3t_0=jnp.asarray(e3t, dtype=state.T.data.dtype),
        nemo_e3w_0=jnp.asarray(e3w, dtype=state.T.data.dtype),
        nemo_e3w_mesh_reference=True,
        nemo_hu_0=jnp.asarray(hu, dtype=state.T.data.dtype),
        nemo_hv_0=jnp.asarray(hv, dtype=state.T.data.dtype),
        nemo_e1e2t=jnp.asarray(e1t * e2t, dtype=state.T.data.dtype),
        nemo_e1e2u=jnp.asarray(e1u * e2u, dtype=state.T.data.dtype),
        nemo_e1e2v=jnp.asarray(e1v * e2v, dtype=state.T.data.dtype),
        nemo_e2u=jnp.asarray(e2u, dtype=state.T.data.dtype),
        nemo_e1v=jnp.asarray(e1v, dtype=state.T.data.dtype),
        nemo_een_barotropic=een,
    )


def build_standalone(member: int):
    """Construct the card's own grid, vertical coordinate, state, and model."""
    set_policy(PrecisionPolicy.fp64())
    cfg = nemo_faithful_dino_config(
        base=dino_config_for_recipe(RECIPE))
    grid = dino_lat_lon_grid(cfg)
    z_coord = dino_lat_lon_vertical(grid, cfg)
    state = dino_lat_lon_state(grid, z_coord, cfg)
    if any(getattr(state, name, None) is not None for name in
           ("T_before", "S_before", "eta_before", "u_before", "v_before")):
        raise RuntimeError("from-rest factory unexpectedly populated a bridge level")
    z_coord = attach_analytic_nemo_operands(grid, z_coord, state, cfg)
    state, perturbation = perturb_temperature(state, member)
    geometry = ensure_geometry(
        grid,
        omega=cfg.omega,
        metric_convention=cfg.metric_convention,
        vface_zonal_metric_evaluation=cfg.vface_zonal_metric_evaluation,
        coriolis_placement=cfg.coriolis_placement,
    )._replace(native_lat_T_deg=jnp.degrees(grid.lat2d))
    model_cfg, _ = dino_lat_lon_model_config(geometry, cfg)
    # Despite the card name, the shipped and T3-certified resolved selector is
    # ``leapfrog``.  Its own no-history entry performs the model's Euler-start
    # bootstrap.  The twin harness's historical DINO_OUTER_INTEGRATOR=nemo_mlf
    # ablation is not a card default and is forbidden in this standalone lane.
    if model_cfg.outer_integrator != "leapfrog":
        raise RuntimeError(
            f"T1 card resolved {model_cfg.outer_integrator!r}, expected the "
            "T3-certified leapfrog default")
    model = LatLonCGridOceanModel(geometry, z_coord, model_cfg)
    forcing = dino_lat_lon_surface_forcing_arrays(
        geometry, cfg, wind_lat_deg=geometry.native_lat_T_deg[:, 0])
    step_forcing = dino_step_surface_forcing(forcing)
    return (cfg, geometry, z_coord, state, model_cfg, model, forcing,
            step_forcing, perturbation)


def run(args: argparse.Namespace) -> int:
    _assert_no_physics_overrides()
    producer, dirty = git_provenance()
    if dirty:
        raise SystemExit("REFUSING dirty tracked producer")
    if args.output_dir.exists():
        raise SystemExit(f"REFUSING existing output directory {args.output_dir}")
    claim_steps = YEARS * DAYS_PER_YEAR * STEPS_PER_DAY
    blockers = claim_admission_reasons()
    if args.steps == claim_steps and blockers:
        raise SystemExit(
            "REFUSING claim-length T1 arm while standalone admission is "
            "blocked: " + "; ".join(blockers))
    args.output_dir.mkdir(parents=True)
    snapshots = args.output_dir / "snapshots"
    reductions = args.output_dir / "reductions"
    snapshots.mkdir()
    reductions.mkdir()

    (cfg, grid, z_coord, state, model_cfg, model, forcing, step_forcing,
     perturbation) = build_standalone(args.member)
    if str(np.asarray(state.T.data).dtype) != "float64":
        raise SystemExit("REFUSING non-fp64 materialized standalone state")

    reducer = None
    reducer_status = "disabled for debug run"
    reducer_mesh = None
    reducer_convention_receipt = None
    if args.reducer_mesh is not None:
        reducer_mesh = args.reducer_mesh.resolve()
        if reducer_mesh.name != "mesh_mask.nc" or not reducer_mesh.is_file():
            raise SystemExit("--reducer-mesh must name an existing mesh_mask.nc")
        reducer, reducer_status = twin.build_snapshot_reducer(
            str(reducer_mesh.parent))
        if reducer is None:
            raise SystemExit(f"live reducer unavailable: {reducer_status}")
        day0_fields = {name: getattr(state, name).data
                       for name in ("T", "S", "eta", "u", "v")}
        reducer, reducer_convention_receipt = build_standalone_reducer_adapter(
            reducer, day0_fields, state.land_mask.data)
        _, day0_reduced, day0_status = twin.capture_snapshot(
            day0_fields, snap_dtype=np.float64, reducer=reducer,
            land_mask=state.land_mask.data)
        if day0_reduced is None or day0_status != "ok":
            raise SystemExit(
                f"live reducer day-0 admission failed: {day0_status}")
        missing = sorted(
            set(twin.REDUCED_KEYS + (twin.REDUCED_ROW_KEY,))
            - set(day0_reduced))
        if missing:
            raise SystemExit(
                "live reducer day-0 admission omitted keys: " + ", ".join(missing))
        day0_serialized = {
            key: np.asarray(value, dtype=np.float64).tolist()
            for key, value in sorted(day0_reduced.items())}
        reducer_convention_receipt.update(
            day0_reduction_status="PASS",
            day0_reduction_keys=sorted(day0_reduced),
            day0_reduction_sha256=hashlib.sha256(json.dumps(
                day0_serialized, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
        )
        reducer_status = (
            f"{reducer_status}; convention={REDUCER_CONVENTION_SCHEMA}; "
            "day0=PASS")
        print(
            f"REDUCER_DAY0=PASS convention={REDUCER_CONVENTION_SCHEMA} "
            f"plant={reducer_convention_receipt['shape_mismatch_plant']}",
            flush=True)
    elif args.steps == YEARS * DAYS_PER_YEAR * STEPS_PER_DAY:
        raise SystemExit("a claim-length run requires --reducer-mesh")

    cfg_rows = flatten_resolved(model_cfg)
    cfg_hash = hashlib.sha256(json.dumps(
        cfg_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    initial_hashes = state_hashes(state)
    reducer_receipt_path = args.output_dir / "reducer_convention_receipt.json"
    if reducer_convention_receipt is not None:
        reducer_convention_receipt.update(
            producer_commit=producer,
            reducer_mesh=str(reducer_mesh),
            reducer_mesh_sha256=_file_sha256(reducer_mesh),
        )
        _write_json(reducer_receipt_path, reducer_convention_receipt)
    initial_receipt = {
        "schema": SCHEMA,
        "producer_commit": producer,
        "recipe": RECIPE,
        "member": args.member,
        "twin_start_mode": "standalone",
        "initial_state_factory": "dino_lat_lon_state",
        "grid_factory": "dino_lat_lon_grid(nemo_faithful_dino_config)",
        "vertical_factory": "dino_lat_lon_vertical",
        "bridge_paths": [],
        "restart_paths": [],
        "before_levels_populated": False,
        "seasonal_t0_seconds": 0.0,
        "first_step_policy": model_cfg.outer_integrator,
        "materialized_dtype": str(np.asarray(state.T.data).dtype),
        "resolved_config_sha256": cfg_hash,
        "state_hashes": initial_hashes,
        "perturbation": perturbation,
        "reducer_mesh_diagnostic_only": (
            str(reducer_mesh) if reducer_mesh is not None else None),
        "reducer_convention_schema": (
            REDUCER_CONVENTION_SCHEMA
            if reducer_convention_receipt is not None else None),
        "reducer_convention_receipt_sha256": (
            _file_sha256(reducer_receipt_path)
            if reducer_convention_receipt is not None else None),
    }
    _write_json(args.output_dir / "initial_receipt.json", initial_receipt)

    frozen_days = set(sample_days())
    target_steps = args.steps
    capture_steps = {
        day * STEPS_PER_DAY: day for day in frozen_days
        if day * STEPS_PER_DAY <= target_steps
    }
    dyn = jax.jit(lambda st, rate: model.step(
        st, DT_SECONDS, surface_forcing=step_forcing,
        external_tracer_rate=rate))
    written: list[dict[str, Any]] = []
    started = time.time()
    for step in range(1, target_steps + 1):
        state, rate = apply_dino_lat_lon_surface_forcing(
            state, forcing, z_coord, cfg, DT_SECONDS,
            t_seconds=step * DT_SECONDS, return_rate=True)
        state = dyn(state, rate)
        if step in capture_steps:
            day = capture_steps[step]
            fields = {name: getattr(state, name).data
                      for name in ("T", "S", "eta", "u", "v")}
            stored, reduced, status = twin.capture_snapshot(
                fields, snap_dtype=np.float64, reducer=reducer,
                land_mask=state.land_mask.data)
            if reducer is not None and (reduced is None or status != "ok"):
                raise SystemExit(f"live reduction failed at day {day}: {status}")
            snap = snapshots / f"day_{day:04d}.npz"
            np.savez(snap, day=np.int32(day), step=np.int32(step), **stored)
            row: dict[str, Any] = {
                "day": day, "step": step,
                "snapshot": str(snap.relative_to(args.output_dir)),
                "snapshot_sha256": _file_sha256(snap),
            }
            if reduced is not None:
                red = reductions / f"day_{day:04d}.npz"
                np.savez(red, **{key: np.asarray(value, dtype=np.float64)
                                 for key, value in reduced.items()})
                row.update(
                    reduction=str(red.relative_to(args.output_dir)),
                    reduction_sha256=_file_sha256(red),
                )
            written.append(row)
            print(f"CAPTURE day={day} step={step}", flush=True)

        if step % (DAYS_PER_YEAR * STEPS_PER_DAY) == 0 or step == target_steps:
            finite = all(np.isfinite(np.asarray(getattr(state, name).data)).all()
                         for name in ("T", "S", "eta", "u", "v"))
            print(f"PROGRESS step={step}/{target_steps} finite={finite}", flush=True)
            if not finite:
                raise SystemExit(f"nonfinite standalone state at step {step}")

    expected_days = sorted(day for day in frozen_days
                           if day * STEPS_PER_DAY <= target_steps)
    if [row["day"] for row in written] != expected_days:
        raise SystemExit("snapshot schedule mismatch")
    complete = (target_steps == claim_steps
                and expected_days == list(sample_days()) and not blockers)
    manifest = {
        "schema": SCHEMA,
        "producer_commit": producer,
        "recipe": RECIPE,
        "member": args.member,
        "seed": seed_for_member(args.member),
        "steps_completed": target_steps,
        "claim_steps": claim_steps,
        "claim_admissible": complete,
        "claim_admission_blockers": list(blockers),
        "twin_start_mode": "standalone",
        "barotropic_cold_start_after_reconcile": (
            model_cfg.barotropic.barotropic_cold_start_after_reconcile),
        "bridge_paths": [],
        "restart_paths": [],
        "storage_dtype": "float64",
        "compute_dtype": "float64",
        "sample_days": expected_days,
        "resolved_config_sha256": cfg_hash,
        "initial_receipt_sha256": _file_sha256(
            args.output_dir / "initial_receipt.json"),
        "reducer_status": reducer_status,
        "reducer_convention_schema": (
            REDUCER_CONVENTION_SCHEMA
            if reducer_convention_receipt is not None else None),
        "reducer_convention_receipt_sha256": (
            _file_sha256(reducer_receipt_path)
            if reducer_convention_receipt is not None else None),
        "captures": written,
        "wall_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(f"MANIFEST={args.output_dir / 'manifest.json'}", flush=True)
    print(f"CLAIM_ADMISSIBLE={complete}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--member", type=int, required=True, choices=MEMBERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reducer-mesh", type=Path,
        help="diagnostic-only NEMO mesh_mask.nc used by imported reducers; "
             "never used to construct the standalone state")
    parser.add_argument(
        "--steps", type=int,
        default=YEARS * DAYS_PER_YEAR * STEPS_PER_DAY,
        help="debug truncation only; any value other than 230400 is stamped "
             "claim_admissible=false")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > YEARS * DAYS_PER_YEAR * STEPS_PER_DAY:
        parser.error("--steps must be in [1, 230400]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
