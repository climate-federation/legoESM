#!/usr/bin/env python3
"""Registered production replay of the literal NEMO EEN coefficient builder."""
from __future__ import annotations

import argparse
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

import jax
import jax.numpy as jnp
import numpy as np

import split_explicit_momentum_chain_round22 as r22
import split_explicit_momentum_chain_round24 as r24
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _build_een_barotropic_inputs,
    _nemo_literal_een_coefficients,
    barotropic_coriolis_een_pre_step,
    een_barotropic_coriolis,
)
from legoesm.ocean.experiments.dino import (
    DINO_RECIPES,
    dino_config_for_recipe,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from zdf_stream_bracket import manifest_sha256, sha256, stream_manifest


ROUND25_SHA = "a9db28c458bb6e409ec23b74ea371b2d2e4aeaf1d84de322074ab72b46fca16e"
BAR = 1.0e-15
COEFFICIENTS = tuple(Path(name).stem.removeprefix("corcoef_dump_")
                     for name in r22.COEFFICIENTS)


def _tracked_status(root: Path) -> str:
    """Ignore untracked run products while refusing tracked-tree drift."""
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _array_equal_tree(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (str, bool)) or a is None:
            if a != b:
                return False
        elif not np.array_equal(np.asarray(a), np.asarray(b)):
            return False
    return True


def _boundary_metric(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    """Use the registered relative bar, with an exact gate for zero walls."""
    wet = np.ones_like(oracle, dtype=bool)
    oracle_rms = float(np.sqrt(np.mean(np.asarray(oracle)[wet] ** 2)))
    if oracle_rms > 0.0:
        return r22._metric(candidate, oracle, wet)
    error = np.asarray(candidate)[wet] - np.asarray(oracle)[wet]
    mismatch = int(np.count_nonzero(
        np.asarray(candidate)[wet] != np.asarray(oracle)[wet]))
    return {
        "n": int(error.size),
        "oracle_rms": 0.0,
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "bit_mismatch_count": mismatch,
        "gate_status": "AT BAR" if mismatch == 0 else "DEBT",
        "zero_oracle_rule": "bit_exact",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 scorer required")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if _tracked_status(root):
        raise SystemExit("tracked-clean scorer checkout required")
    if sha256(args.round25) != ROUND25_SHA:
        raise SystemExit("round-25 exact-arm artifact changed")
    prior = json.loads(args.round25.read_text())
    if prior.get("disposition") != "NEMO_SOURCE_ASSOCIATION_OWNS_FINAL_COEFFICIENT_DEBT":
        raise SystemExit("round-25 ownership verdict changed")

    run = args.run.resolve()
    manifest = stream_manifest(run)
    manifest_hash = manifest_sha256(manifest)
    if manifest_hash != prior["bindings"]["run_manifest_sha256"]:
        raise SystemExit("retained coefficient run manifest changed")
    jpi, jpj, _jpk, hls, _icycle, _nn_e = r22.inherited._read_dims(str(run))

    def load(name: str) -> np.ndarray:
        return r22.inherited._load_full(str(run / name), jpi, jpj, hls)

    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(str(run / r22.inherited.RESTART_FILE), nn_hls=0)
    mx = r24._mesh(run / "mesh_mask.nc")
    ht_0 = (mx["e3t_0"] * mx["tmask"]).sum(axis=-1)
    tsurf = mx["tmask"].max(axis=-1)
    r3t = (np.asarray(restart.ssh)
           * tsurf / (ht_0 + 1.0 - tsurf))
    h_k = jnp.asarray(mx["e3t_0"] * (
        1.0 + r3t[..., None] * mx["tmask"]) * mx["tmask"])
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    bridge = bridge_nemo_to_legoesm_topo(
        mesh, restart, periodic_i=True, full_step=True, omega=cfg.omega,
        coriolis_placement="face_latitude", carry_native_lat_deg=True)
    state = bridge.state
    eta = jnp.asarray(restart.ssh)
    ua_native = jnp.asarray(load("cor2d_dump_ua_e_in_substep1.bin"))
    va_native = jnp.asarray(load("cor2d_dump_va_e_in_substep1.bin"))
    ua = jnp.concatenate((ua_native[:, -1:], ua_native), axis=1)
    va = jnp.concatenate((jnp.zeros_like(va_native[:1]), va_native), axis=0)
    oracle_coeff = {
        name: load(f"corcoef_dump_{name}.bin") for name in COEFFICIENTS}
    nemo_u = load("cor2d_dump_zu_trd_substep1.bin")
    nemo_v = load("cor2d_dump_zv_trd_substep1.bin")
    umask = mx["umask"][..., 0] > 0.5
    vmask = mx["vmask"][..., 0] > 0.5

    build_kwargs = dict(
        h_k=h_k, grid=bridge.geometry,
        mask=state.land_mask.data, u_mask=state.u_mask.data,
        v_mask=state.v_mask.data, dtype=jnp.float64,
        metric_complete=True, een_q_boundary="nemo_live",
        een_e3f_scheme="nemo_avg", dz_ref=bridge.z_coord.dz_ref,
        coefficient_evaluation="nemo_literal", z_coord=bridge.z_coord)

    def evaluate(eta_arg, u_arg, v_arg):
        pre = _build_een_barotropic_inputs(
            **build_kwargs, eta=eta_arg)
        cu, cv = een_barotropic_coriolis(u_arg, v_arg, pre)
        return tuple(pre["literal_coefficients"][name]
                     for name in COEFFICIENTS) + (cu[:, 1:], cv[1:, :])

    eager_values = evaluate(eta, ua, va)
    compiled_values = jax.jit(evaluate)(eta, ua, va)

    def score(values):
        coeff = {
            name: r22._metric(np.asarray(value), oracle_coeff[name],
                              umask if name.startswith("ffu") else vmask)
            for name, value in zip(COEFFICIENTS, values[:len(COEFFICIENTS)])}
        output = {
            "u": r22._metric(np.asarray(values[-2]), nemo_u, umask),
            "v": r22._metric(np.asarray(values[-1]), nemo_v, vmask),
        }
        return coeff, output

    eager_coeff, eager_output = score(eager_values)
    jit_coeff, jit_output = score(compiled_values)
    all_rows = [*jit_coeff.values(), *jit_output.values()]

    # Selector scope and byte-pin: the faithful parent and its MLF child are
    # the only literal cards; omission and explicit generic must have identical
    # inputs and arrays.
    literal_cards = sorted(
        name for name in DINO_RECIPES
        if dino_config_for_recipe(name).barotropic_een_coefficient_evaluation
        == "nemo_literal")
    generic_implicit = _build_een_barotropic_inputs(
        h_k, bridge.geometry, state.land_mask.data, state.u_mask.data,
        state.v_mask.data, jnp.float64, metric_complete=True,
        een_q_boundary="nemo_live", een_e3f_scheme="nemo_avg",
        dz_ref=bridge.z_coord.dz_ref)
    generic_explicit = _build_een_barotropic_inputs(
        h_k, bridge.geometry, state.land_mask.data, state.u_mask.data,
        state.v_mask.data, jnp.float64, metric_complete=True,
        een_q_boundary="nemo_live", een_e3f_scheme="nemo_avg",
        dz_ref=bridge.z_coord.dz_ref, coefficient_evaluation="generic")

    # Both public consumers must reuse the same materialized object.  The
    # pre-step wrapper returns the supplied object verbatim; the live operator
    # is then evaluated directly from that same identity.
    shared_pre = _build_een_barotropic_inputs(**build_kwargs, eta=eta)
    _, _, returned_pre = barotropic_coriolis_een_pre_step(
        state.u.data, state.v.data, h_k, bridge.geometry,
        state.land_mask.data, state.u_mask.data, state.v_mask.data,
        jnp.asarray(1.0e-10), jnp.float64, metric_complete=True,
        een_q_boundary="nemo_live", een_e3f_scheme="nemo_avg",
        dz_ref=bridge.z_coord.dz_ref,
        coefficient_evaluation="nemo_literal", eta=eta,
        z_coord=bridge.z_coord, pre=shared_pre, return_pre=True)
    _ = een_barotropic_coriolis(ua, va, returned_pre)

    identity = r22._metric(oracle_coeff["ffu_nw"],
                           oracle_coeff["ffu_nw"], umask)
    plant = np.array(oracle_coeff["ffu_nw"], copy=True)
    wet = np.argwhere(umask)
    plant_at = tuple(wet[np.argmax(np.abs(oracle_coeff["ffu_nw"][umask]))])
    for _ in range(4):
        plant[plant_at] = np.nextafter(plant[plant_at], np.inf)
    planted = r22._metric(plant, oracle_coeff["ffu_nw"], umask)

    seam_candidate = np.asarray(compiled_values[-2])[:, [0, -1]]
    wall_candidate = np.asarray(compiled_values[-1])[[0, -1]]
    pointwise = {
        "u_periodic_seam": _boundary_metric(
            seam_candidate, nemo_u[:, [0, -1]]),
        "v_south_north_walls": _boundary_metric(
            wall_candidate, nemo_v[[0, -1]]),
    }

    direction = jnp.sin(jnp.arange(eta.size, dtype=jnp.float64)).reshape(eta.shape)

    def coefficient_loss(scale):
        c = _nemo_literal_een_coefficients(
            eta + scale * direction, bridge.z_coord, jnp.float64)
        return sum(jnp.sum(value * value) for value in c.values())

    gradient = float(jax.grad(coefficient_loss)(jnp.asarray(0.0)))
    prior_control = prior["arms"]["T0V0P0"]
    prior_debt = all(
        row["gate_status"] == "DEBT"
        for row in [*prior_control["coefficients"].values(),
                    *prior_control["output"].values()])
    controls = {
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "four_nextafter_plant_debt": planted["gate_status"] == "DEBT",
        "association_order_control_debt": prior_debt,
        "two_literal_cards_only": literal_cards == [
            "nemo_dino_kamm", "nemo_dino_kamm_mlf"],
        "generic_omission_byte_identical": _array_equal_tree(
            generic_implicit, generic_explicit),
        "both_consumers_reuse_identity": returned_pre is shared_pre,
        "periodic_seam_at_bar": pointwise["u_periodic_seam"]["gate_status"]
        == "AT BAR",
        "wall_rows_at_bar": pointwise["v_south_north_walls"]["gate_status"]
        == "AT BAR",
        "finite_gradient": bool(np.isfinite(gradient)),
    }
    if not all(controls.values()):
        raise SystemExit(f"round-28 control failed: {controls}")

    exact = all(row["gate_status"] == "AT BAR" for row in all_rows)
    disposition = (
        "PRODUCTION_LITERAL_EEN_COEFFICIENTS_AT_BAR"
        if exact else "PRODUCTION_LITERAL_EEN_COEFFICIENTS_DEBT")
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round28-v1",
        "session_id": session,
        "git_commit": head,
        "backend": "cpu",
        "jax_enable_x64": True,
        "bars": {"normalized_rms": BAR, "max_over_nemo_rms": BAR},
        "nemo_source": {
            "triad": "dynspg_ts.F90:1517-1528,1544-1555",
            "vertical_reduction": "dynspg_ts.F90:1530-1534,1557-1561",
            "post_factor": "dynspg_ts.F90:1535-1538,1562-1565",
        },
        "literal_cards": literal_cards,
        "eager": {"coefficients": eager_coeff, "output": eager_output},
        "jit": {"coefficients": jit_coeff, "output": jit_output},
        "pointwise": pointwise,
        "controls": controls,
        "control_receipts": {
            "four_nextafter_metric": planted,
            "four_nextafter_location": [int(index) for index in plant_at],
            "coefficient_gradient": gradient,
        },
        "disposition": disposition,
        "ordered_rows": {
            "1.3_production_replay": "RELEASED" if exact else "ORDERED_BLOCKED",
            "1.4": "NEXT" if exact else "ORDERED_BLOCKED",
            **{str(row): "ORDERED_BLOCKED" for row in range(2, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "bindings": {
            "round25_sha256": ROUND25_SHA,
            "run_manifest_sha256": manifest_hash,
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, row in sorted(jit_coeff.items()):
        print(f"{name} rms={row['normalized_rms_error']:.17g} "
              f"max={row['max_error_over_nemo_rms']:.17g} "
              f"status={row['gate_status']}")
    for name, row in sorted(jit_output.items()):
        print(f"output_{name} rms={row['normalized_rms_error']:.17g} "
              f"max={row['max_error_over_nemo_rms']:.17g} "
              f"status={row['gate_status']}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
