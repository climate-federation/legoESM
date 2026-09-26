#!/usr/bin/env python3
"""Score the preregistered row-1.3 vertex-f coefficient owner from round 22."""
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
import split_explicit_momentum_chain_round22 as r22
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
from zdf_stream_bracket import manifest_sha256, sha256, stream_manifest

ROUND22_SHA = "c770d68c352682d15464c0bda01cb1c9cde22a857600e237f2658e8c6a97d1e3"
POINTWISE = 1.0e-15


def _arm(
    placement: str,
    mesh: Any,
    restart: Any,
    ua: np.ndarray,
    va: np.ndarray,
    nemo_u: np.ndarray,
    nemo_v: np.ndarray,
    nemo_coeff: dict[str, np.ndarray],
    umask: np.ndarray,
    vmask: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
        coriolis_placement=placement,
    )
    bridge = bridge_nemo_to_legoesm_topo(
        mesh, restart, periodic_i=True, full_step=True, omega=cfg.omega,
        coriolis_placement=placement, carry_native_lat_deg=True,
    )
    model_cfg, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    h_k = compute_layer_thickness(
        bridge.state.eta.data, bridge.state.H_bathy.data, bridge.z_coord,
        min_water_column_m=model_cfg.min_water_column_m,
    )
    pre = _build_een_barotropic_inputs(
        h_k, bridge.geometry, bridge.state.land_mask.data,
        bridge.state.u_mask.data, bridge.state.v_mask.data, jnp.float64,
        metric_complete=True, een_q_boundary=model_cfg.een_q_boundary,
        een_e3f_scheme=model_cfg.een_e3f_scheme,
        dz_ref=bridge.z_coord.dz_ref,
    )
    u_patterns = r22._checkerboards(
        tuple(bridge.state.u_mask.data.shape), periodic_u=True)
    v_patterns = r22._checkerboards(tuple(bridge.state.v_mask.data.shape))
    zeros_u = jnp.zeros_like(bridge.state.u_mask.data, dtype=jnp.float64)
    zeros_v = jnp.zeros_like(bridge.state.v_mask.data, dtype=jnp.float64)
    u_outputs = [
        np.asarray(een_barotropic_coriolis(
            zeros_u, jnp.asarray(pattern), pre)[0])[:, 1:]
        for pattern in v_patterns
    ]
    v_outputs = [
        np.asarray(een_barotropic_coriolis(
            jnp.asarray(pattern), zeros_v, pre)[1])[1:, :]
        for pattern in u_patterns
    ]
    effective_u = r22._solve_coefficients(u_outputs, v_patterns, "u")
    effective_v = r22._solve_coefficients(v_outputs, u_patterns, "v")
    coefficient_arrays = {
        **{f"ffu_{name}": value for name, value in effective_u.items()},
        **{f"ffv_{name}": value for name, value in effective_v.items()},
    }
    coefficients = {
        name: r22._metric(value, nemo_coeff[name],
                          umask if name.startswith("ffu") else vmask)
        for name, value in sorted(coefficient_arrays.items())
    }
    prod_u, prod_v = een_barotropic_coriolis(
        jnp.asarray(r22.r9._u_face(ua)),
        jnp.asarray(r22.r9._v_face(va)), pre,
    )
    output = {
        "u": r22._metric(np.asarray(prod_u)[:, 1:], nemo_u, umask),
        "v": r22._metric(np.asarray(prod_v)[1:, :], nemo_v, vmask),
    }
    # NemoGrid.ff_f(i,j) is the NE F point of T(i,j); bridge f_v row j+1 is
    # that same latitude.  The synthetic southern face has no NEMO peer.
    f_point = r22._metric(
        np.asarray(bridge.geometry.f_v)[1:, :], np.asarray(mesh.ff_f),
        np.ones_like(mesh.ff_f, dtype=bool),
    )
    return {
        "placement": placement,
        "f_point": f_point,
        "effective_coefficients": coefficients,
        "production_output": output,
    }, coefficient_arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round22", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU/fp64 scorer required")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise SystemExit("clean scorer checkout required")
    if sha256(args.round22) != ROUND22_SHA:
        raise SystemExit("round-22 artifact SHA changed")
    prior = json.loads(args.round22.read_text())
    if not (
        prior.get("disposition") == "CORIOLIS_COEFFICIENT_COMPOSITION_LOCALIZED"
        and set(prior.get("diverged_coefficients", []))
            == {f"{c}_{q}" for c in ("ffu", "ffv") for q in ("nw", "ne", "sw", "se")}
    ):
        raise SystemExit("round-22 registered stop changed")
    run = args.run.resolve()
    manifest = stream_manifest(run)
    if manifest_sha256(manifest) != prior["bindings"]["on_manifest_sha256"]:
        raise SystemExit("retained round-22 run manifest changed")
    for name, digest in prior["bindings"]["coefficient_sha256"].items():
        if sha256(run / name) != digest:
            raise SystemExit(f"retained coefficient changed: {name}")
    if sha256(run / "mesh_mask.nc") != prior["bindings"]["mesh_sha256"]:
        raise SystemExit("mesh changed")
    if sha256(run / r22.inherited.RESTART_FILE) != prior["bindings"]["restart_sha256"]:
        raise SystemExit("restart changed")

    jpi, jpj, _jpk, hls, _icycle, _nn_e = r22.inherited._read_dims(str(run))
    if (jpi, jpj, hls) != (56, 203, 2):
        raise SystemExit("DINO dimensions changed")
    def load(name: str) -> np.ndarray:
        return r22.inherited._load_full(str(run / name), jpi, jpj, hls)

    ua = load("cor2d_dump_ua_e_in_substep1.bin")
    va = load("cor2d_dump_va_e_in_substep1.bin")
    nemo_u = load("cor2d_dump_zu_trd_substep1.bin")
    nemo_v = load("cor2d_dump_zv_trd_substep1.bin")
    nemo_coeff = {
        Path(name).stem.removeprefix("corcoef_dump_"): load(name)
        for name in r22.COEFFICIENTS
    }
    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(
        str(run / r22.inherited.RESTART_FILE), nn_hls=0)
    umask = np.asarray(mesh.umask)[..., 0] > 0.5
    vmask = np.asarray(mesh.vmask)[..., 0] > 0.5
    if (int(umask.sum()), int(vmask.sum())) != (9758, 9868):
        raise SystemExit("registered populations changed")

    legacy, _legacy_arrays = _arm(
        "cell_average", mesh, restart, ua, va, nemo_u, nemo_v,
        nemo_coeff, umask, vmask)
    face, face_arrays = _arm(
        "face_latitude", mesh, restart, ua, va, nemo_u, nemo_v,
        nemo_coeff, umask, vmask)

    legacy_reproduces = True
    for name, row in legacy["effective_coefficients"].items():
        old = prior["production_effective_coefficients"][name]
        for key in ("normalized_rms_error", "max_error_over_nemo_rms"):
            denom = max(abs(float(old[key])), np.finfo(np.float64).tiny)
            legacy_reproduces &= abs(float(row[key]) - float(old[key])) / denom <= POINTWISE
    legacy_debt = all(
        row["gate_status"] == "DEBT"
        for row in legacy["effective_coefficients"].values())

    identity = r22._metric(nemo_coeff["ffu_nw"], nemo_coeff["ffu_nw"], umask)
    planted = np.array(nemo_coeff["ffu_nw"], copy=True)
    ij = tuple(np.argwhere(umask)[0])
    steps = 0
    while True:
        planted[ij] = np.nextafter(planted[ij], np.inf)
        steps += 1
        planted_metric = r22._metric(planted, nemo_coeff["ffu_nw"], umask)
        if planted_metric["gate_status"] == "DEBT":
            break
        if steps > 1024:
            raise SystemExit("could not construct decisive nextafter plant")
    controls = {
        "legacy_reproduces_round22": bool(legacy_reproduces),
        "legacy_all_eight_debt": bool(legacy_debt),
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "nextafter_plant_debt": planted_metric["gate_status"] == "DEBT",
        "nextafter_steps": steps,
    }
    if not all(value for key, value in controls.items() if key != "nextafter_steps"):
        raise SystemExit(f"control failed: {controls}")

    face_rows = [*face["effective_coefficients"].values(),
                 *face["production_output"].values()]
    face_exact = all(row["gate_status"] == "AT BAR" for row in face_rows)
    old_sum = sum(row["normalized_rms_error"]
                  for row in legacy["effective_coefficients"].values())
    new_sum = sum(row["normalized_rms_error"]
                  for row in face["effective_coefficients"].values())
    reduction = 1.0 - new_sum / old_sum
    if face_exact and legacy_debt:
        disposition = "VERTEX_F_OWNS_ALL_EIGHT_COEFFICIENTS"
    elif reduction >= 0.9:
        disposition = "VERTEX_F_PARTIAL_OWNER"
    else:
        disposition = "VERTEX_F_REFUTED"

    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round23-v1",
        "session_id": session,
        "git_commit": head,
        "backend": "cpu",
        "jax_enable_x64": True,
        "bars": {"pointwise": POINTWISE, "partial_reduction": 0.9},
        "legacy_control": legacy,
        "face_latitude_arm": face,
        "coefficient_normalized_rms_reduction": reduction,
        "controls": controls,
        "plant_metric": planted_metric,
        "disposition": disposition,
        "ordered_rows": {
            "1.3_next": "bottom_stress" if face_exact else "EEN_coefficient_operands",
            "1.4": "ORDERED_BLOCKED",
            **{str(row): "ORDERED_BLOCKED" for row in range(2, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "bindings": {
            "round22_sha256": ROUND22_SHA,
            "run_manifest_sha256": manifest_sha256(manifest),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print(f"coefficient_normalized_rms_reduction={reduction:.17g}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
