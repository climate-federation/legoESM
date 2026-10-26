#!/usr/bin/env python3
"""Score the preregistered row-1.3 frozen EEN-coefficient peel."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import numpy as np
import spg_substep_chain as inherited
import split_explicit_momentum_chain_round9 as r9
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _build_een_barotropic_inputs,
    een_barotropic_coriolis,
)
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.vertical import compute_layer_thickness
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    sha256,
    stream_manifest,
)

ROUND21_SHA = "c2b890f8ec70847bfc69f565cb254e022d86621b726a1d37aef85059a62e8709"
POINTWISE = 1.0e-15
COEFFICIENTS = tuple(
    f"corcoef_dump_{component}_{corner}.bin"
    for component in ("ffu", "ffv")
    for corner in ("nw", "ne", "sw", "se")
)


def _metric(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    wet = np.asarray(mask, dtype=bool)
    a = np.asarray(candidate, dtype=np.float64)[wet]
    b = np.asarray(oracle, dtype=np.float64)[wet]
    if not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise SystemExit("empty or non-finite registered metric population")
    rms = float(np.sqrt(np.mean(b * b)))
    if rms == 0.0:
        raise SystemExit("zero oracle RMS on registered metric population")
    delta = a - b
    error = float(np.sqrt(np.mean(delta * delta)) / rms)
    maximum = float(np.max(np.abs(delta)) / rms)
    mismatches = int(np.count_nonzero(
        np.ascontiguousarray(a).view(np.uint64)
        != np.ascontiguousarray(b).view(np.uint64)))
    return {
        "n": int(a.size),
        "normalized_rms_error": error,
        "max_error_over_nemo_rms": maximum,
        "bit_mismatch_count": mismatches,
        "gate_status": "AT BAR" if error <= POINTWISE and maximum <= POINTWISE else "DEBT",
    }


def _checkerboards(shape: tuple[int, int], periodic_u: bool = False) -> list[np.ndarray]:
    jj, ii = np.indices(shape)
    si = np.where(ii % 2, -1.0, 1.0)
    sj = np.where(jj % 2, -1.0, 1.0)
    values = [np.ones(shape), si, sj, si * sj]
    if periodic_u:
        for value in values:
            value[:, -1] = value[:, 0]
    return values


def _solve_coefficients(
    outputs: list[np.ndarray],
    patterns: list[np.ndarray],
    component: str,
) -> dict[str, np.ndarray]:
    """Invert four global checkerboards into each local four-point stencil."""
    y = np.stack(outputs, axis=-1)
    nlat, nlon = y.shape[:2]
    matrix = np.empty((nlat, nlon, 4, 4), dtype=np.float64)
    if component == "u":
        # NEMO source order: NW, NE, SW, SE. Lego U[:,1:] is NEMO U.
        for p, value in enumerate(patterns):
            for j in range(nlat):
                for i in range(nlon):
                    east = (i + 1) % nlon
                    matrix[j, i, p] = (
                        value[j + 1, i], value[j + 1, east],
                        value[j, i], value[j, east])
        solved = np.linalg.solve(matrix, y[..., None])[..., 0]
        return {name: solved[..., k] for k, name in enumerate(("nw", "ne", "sw", "se"))}

    # NEMO source order in the signed sum: SW, SE, NW, NE. The last V row is
    # a dry polar wall, so solve it with a harmless identity and exclude it.
    matrix[-1, :, :, :] = np.eye(4)
    y[-1, :, :] = 0.0
    for p, value in enumerate(patterns):
        for j in range(nlat - 1):
            for i in range(nlon):
                matrix[j, i, p] = (
                    value[j, i], value[j, i + 1],
                    value[j + 1, i], value[j + 1, i + 1])
    solved = np.linalg.solve(matrix, -y[..., None])[..., 0]
    return {name: solved[..., k] for k, name in enumerate(("sw", "se", "nw", "ne"))}


def _literal_application(
    ua: np.ndarray,
    va: np.ndarray,
    coeff: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    east_v = np.roll(va, -1, axis=1)
    south_v = np.concatenate([np.zeros_like(va[:1]), va[:-1]], axis=0)
    south_east_v = np.roll(south_v, -1, axis=1)
    north_u = np.concatenate([ua[1:], np.zeros_like(ua[:1])], axis=0)
    west_u = np.roll(ua, 1, axis=1)
    north_west_u = np.roll(north_u, 1, axis=1)
    zu = ((coeff["ffu_nw"] * va + coeff["ffu_ne"] * east_v)
          + (coeff["ffu_sw"] * south_v + coeff["ffu_se"] * south_east_v))
    zv = -((coeff["ffv_sw"] * west_u + coeff["ffv_se"] * ua)
           + (coeff["ffv_nw"] * north_west_u + coeff["ffv_ne"] * north_u))
    return zu, zv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-run", type=Path, required=True)
    parser.add_argument("--off-run", type=Path, required=True)
    parser.add_argument("--bracket", type=Path, required=True)
    parser.add_argument("--bracket-sha", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--round21", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU/fp64 scorer required")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise SystemExit("clean scorer checkout required")
    subprocess.run(["git", "cat-file", "-e", f"{args.producer}^{{commit}}"], cwd=root, check=True)
    model_paths = ["packages/core", "packages/ocean", "src"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", args.producer, head, "--", *model_paths],
        cwd=root, text=True).strip()
    if changed:
        raise SystemExit(f"model differs from SHA-pinned producer {args.producer}: {changed}")
    if sha256(args.round21) != ROUND21_SHA:
        raise SystemExit("round-21 artifact SHA changed")
    round21 = json.loads(args.round21.read_text())
    if round21.get("disposition") != "PGF_AT_BAR_ADVANCE_TO_CORIOLIS":
        raise SystemExit("round-21 ordered stop changed")

    if sha256(args.bracket) != args.bracket_sha:
        raise SystemExit("bracket receipt SHA changed")
    bracket = json.loads(args.bracket.read_text())
    on, off = args.on_run.resolve(), args.off_run.resolve()
    expected = set(COEFFICIENTS)
    if not (
        bracket.get("schema") == "dino-spg-row13-een-coefficient-bracket-v1"
        and Path(bracket.get("on_dir", "")).resolve() == on
        and Path(bracket.get("off_dir", "")).resolve() == off
        and bracket.get("shared_count") == 203
        and bracket.get("shared_exact") is True
        and set(bracket.get("new_streams", [])) == expected
        and all(bracket.get("controls", {}).values())
    ):
        raise SystemExit("invalid coefficient bracket receipt")
    on_manifest, off_manifest = stream_manifest(on), stream_manifest(off)
    if not (
        len(on_manifest) == 211 and len(off_manifest) == 203
        and set(on_manifest) - set(off_manifest) == expected
        and all(files_byte_identical(on / name, off / name) for name in off_manifest)
        and manifest_sha256({name: on_manifest[name] for name in sorted(off_manifest)})
            == bracket["shared_manifest_sha256"]
    ):
        raise SystemExit("run directories changed after exact bracket")
    if not one_bit_file_control(on / sorted(off_manifest)[0]):
        raise SystemExit("one-bit file control failed")

    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(on))
    if (jpi, jpj, hls) != (56, 203, 2):
        raise SystemExit(f"unexpected DINO dimensions {(jpi, jpj, hls)}")
    def load(name: str) -> np.ndarray:
        return inherited._load_full(str(on / name), jpi, jpj, hls)

    ua = load("cor2d_dump_ua_e_in_substep1.bin")
    va = load("cor2d_dump_va_e_in_substep1.bin")
    nemo_u = load("cor2d_dump_zu_trd_substep1.bin")
    nemo_v = load("cor2d_dump_zv_trd_substep1.bin")
    nemo_coeff = {Path(name).stem.removeprefix("corcoef_dump_"): load(name)
                  for name in COEFFICIENTS}

    mesh = read_nemo_mesh_mask(str(on / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(str(on / inherited.RESTART_FILE), nn_hls=0)
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    bridge = bridge_nemo_to_legoesm_topo(
        mesh, restart, periodic_i=True, full_step=True, omega=cfg.omega,
        coriolis_placement=cfg.coriolis_placement,
        carry_native_lat_deg=True)
    model_cfg, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    h_k = compute_layer_thickness(
        bridge.state.eta.data, bridge.state.H_bathy.data, bridge.z_coord,
        min_water_column_m=model_cfg.min_water_column_m)
    pre = _build_een_barotropic_inputs(
        h_k, bridge.geometry, bridge.state.land_mask.data,
        bridge.state.u_mask.data, bridge.state.v_mask.data, jnp.float64,
        metric_complete=True, een_q_boundary=model_cfg.een_q_boundary,
        een_e3f_scheme=model_cfg.een_e3f_scheme, dz_ref=bridge.z_coord.dz_ref)
    umask = np.asarray(mesh.umask)[..., 0] > 0.5
    vmask = np.asarray(mesh.vmask)[..., 0] > 0.5
    if (int(umask.sum()), int(vmask.sum())) != (9758, 9868):
        raise SystemExit("registered U/V populations changed")

    reconstructed_u, reconstructed_v = _literal_application(ua, va, nemo_coeff)
    reconstruction = {
        "u": _metric(reconstructed_u, nemo_u, umask),
        "v": _metric(reconstructed_v, nemo_v, vmask),
    }

    u_patterns = _checkerboards(tuple(bridge.state.u_mask.data.shape), periodic_u=True)
    v_patterns = _checkerboards(tuple(bridge.state.v_mask.data.shape))
    zeros_u = jnp.zeros_like(bridge.state.u_mask.data, dtype=jnp.float64)
    zeros_v = jnp.zeros_like(bridge.state.v_mask.data, dtype=jnp.float64)
    u_outputs = [np.asarray(een_barotropic_coriolis(zeros_u, jnp.asarray(p), pre)[0])[:, 1:]
                 for p in v_patterns]
    v_outputs = [np.asarray(een_barotropic_coriolis(jnp.asarray(p), zeros_v, pre)[1])[1:, :]
                 for p in u_patterns]
    effective_u = _solve_coefficients(u_outputs, v_patterns, "u")
    effective_v = _solve_coefficients(v_outputs, u_patterns, "v")
    coefficients: dict[str, dict[str, Any]] = {}
    for corner in ("nw", "ne", "sw", "se"):
        coefficients[f"ffu_{corner}"] = _metric(
            effective_u[corner], nemo_coeff[f"ffu_{corner}"], umask)
        coefficients[f"ffv_{corner}"] = _metric(
            effective_v[corner], nemo_coeff[f"ffv_{corner}"], vmask)

    prod_u, prod_v = een_barotropic_coriolis(
        jnp.asarray(r9._u_face(ua)), jnp.asarray(r9._v_face(va)), pre)
    production_output = {
        "u": _metric(np.asarray(prod_u)[:, 1:], nemo_u, umask),
        "v": _metric(np.asarray(prod_v)[1:, :], nemo_v, vmask),
    }
    plant = np.array(reconstructed_u, copy=True)
    first = tuple(np.argwhere(umask)[0])
    plant[first] += 1.0e-6 * float(np.sqrt(np.mean(nemo_u[umask] ** 2)))
    controls = {
        "bracket_controls": all(bracket["controls"].values()),
        "literal_identity": _metric(nemo_u, nemo_u, umask)["gate_status"] == "AT BAR",
        "wet_point_plant": _metric(plant, nemo_u, umask)["gate_status"] == "DEBT",
    }
    if not all(controls.values()):
        raise SystemExit(f"control failed: {controls}")

    recon_ok = all(row["gate_status"] == "AT BAR" for row in reconstruction.values())
    debt = [name for name, row in coefficients.items() if row["gate_status"] == "DEBT"]
    output_debt = any(row["gate_status"] == "DEBT" for row in production_output.values())
    if not recon_ok:
        disposition = "INVALID_LITERAL_APPLICATION_RECONSTRUCTION"
    elif debt:
        disposition = "CORIOLIS_COEFFICIENT_COMPOSITION_LOCALIZED"
    elif output_debt:
        disposition = "CORIOLIS_APPLICATION_ASSOCIATION_OWNED"
    else:
        disposition = "CORIOLIS_AT_BAR_RELEASE_BOTTOM_STRESS"

    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round22-v1",
        "session_id": session,
        "git_commit": head,
        "physics_producer": args.producer,
        "producer_model_diff_zero": True,
        "backend": "cpu",
        "jax_enable_x64": True,
        "bars": {"pointwise": POINTWISE},
        "literal_application_reconstruction": reconstruction,
        "production_effective_coefficients": coefficients,
        "production_output": production_output,
        "diverged_coefficients": debt,
        "controls": controls,
        "disposition": disposition,
        "ordered_rows": {
            "1.3_next": (
                "bottom_stress"
                if disposition == "CORIOLIS_AT_BAR_RELEASE_BOTTOM_STRESS"
                else "EEN_coefficients"
            ),
            "1.4": "ORDERED_BLOCKED",
            **{str(row): "ORDERED_BLOCKED" for row in range(2, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "bindings": {
            "round21_sha256": ROUND21_SHA,
            "bracket_sha256": args.bracket_sha,
            "on_manifest_sha256": manifest_sha256(on_manifest),
            "off_manifest_sha256": manifest_sha256(off_manifest),
            "coefficient_sha256": {name: sha256(on / name) for name in COEFFICIENTS},
            "mesh_sha256": sha256(on / "mesh_mask.nc"),
            "restart_sha256": sha256(on / inherited.RESTART_FILE),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print(f"diverged_coefficients={','.join(debt) if debt else 'none'}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
