#!/usr/bin/env python3
"""Offline 2^3 EEN triad/reduction/post-factor association factorial."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

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
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from zdf_stream_bracket import manifest_sha256, sha256, stream_manifest

ROUND24_SHA = "3d0fa3148327daa36b7107a5b913b8f806090f5f9f76da93ff827c806c6872b4"
BAR = 1.0e-15


def _shift(a: np.ndarray, di: int = 0, dj: int = 0) -> np.ndarray:
    out = np.roll(a, di, axis=1) if di else a
    return np.roll(out, dj, axis=0) if dj else out


def _triad(a: np.ndarray, b: np.ndarray, c: np.ndarray, literal: bool) -> np.ndarray:
    if literal:
        return np.add(np.add(a, b), c)
    return np.asarray(jnp.asarray(a) + jnp.asarray(b) + jnp.asarray(c))


def _vertical(term: np.ndarray, literal: bool) -> np.ndarray:
    if not literal:
        return np.asarray(jnp.sum(jnp.asarray(term), axis=-1))
    acc = np.zeros(term.shape[:-1], dtype=np.float64)
    for level in range(term.shape[-1]):
        acc = acc + term[..., level]
    return acc


def _coefficient(
    face: np.ndarray,
    neighbor: np.ndarray,
    neighbor_mask: np.ndarray,
    q_args: tuple[np.ndarray, np.ndarray, np.ndarray],
    neighbor_metric: np.ndarray,
    local_metric: np.ndarray,
    reference_depth: np.ndarray,
    r3_face: np.ndarray,
    triad_literal: bool,
    vertical_literal: bool,
    post_literal: bool,
) -> np.ndarray:
    qsum = _triad(*q_args, literal=triad_literal)
    if post_literal:
        # dynspg_ts.F90:1344-1347/1371-1374, then :1349-1352/:1376-1379.
        term = ((face * neighbor) * neighbor_mask) * qsum
        acc = _vertical(term, literal=vertical_literal)
        surface = (reference_depth > 0.0).astype(np.float64)
        r1_h = (surface / (reference_depth + 1.0 - surface)) / (1.0 + r3_face)
        return ((((1.0 / 12.0) * (1.0 / local_metric)) * r1_h)
                * neighbor_metric) * acc

    # barotropic_latlon_cgrid.py:823-842 and the production AL81 helper:
    # 1/12 in the triad, neighbour metric in the transported velocity,
    # JAX reduction, then depth/local-metric divisions.
    t = np.asarray(jnp.asarray(1.0 / 12.0) * jnp.asarray(qsum))
    vsrc = np.asarray(
        jnp.asarray(neighbor_metric)[..., jnp.newaxis] * jnp.asarray(1.0))
    flux = np.asarray(
        (jnp.asarray(neighbor) * jnp.asarray(vsrc))
        * jnp.asarray(neighbor_mask))
    term = np.asarray(jnp.asarray(face) * (jnp.asarray(t) * jnp.asarray(flux)))
    acc = _vertical(term, literal=vertical_literal)
    live_depth = np.asarray(jnp.sum(jnp.asarray(face), axis=-1))
    return np.asarray(
        (jnp.asarray(acc) / jnp.maximum(jnp.asarray(live_depth), 1.0e-10))
        / jnp.maximum(jnp.asarray(local_metric), 1.0e-10))


def _materialize(
    mx: dict[str, np.ndarray],
    ff: np.ndarray,
    e3f: np.ndarray,
    e3u: np.ndarray,
    e3v: np.ndarray,
    r3: dict[str, np.ndarray],
    bits: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    t, v, p = (bool(x) for x in bits)
    q = ff[..., None] / e3f
    uref = (mx["e3u_0"] * mx["umask"]).sum(axis=-1)
    vref = (mx["e3v_0"] * mx["vmask"]).sum(axis=-1)

    uq = {
        "nw": (_shift(q, 1, 0), q, _shift(q, 0, 1)),
        "ne": (_shift(q, 0, 1), q, _shift(q, -1, 0)),
        "sw": (q, _shift(q, 0, 1), _shift(q, 1, 1)),
        "se": (_shift(q, -1, 1), _shift(q, 0, 1), q),
    }
    uneighbor = {
        "nw": (e3v, mx["vmask"], mx["e1v"]),
        "ne": (_shift(e3v, -1, 0), _shift(mx["vmask"], -1, 0), _shift(mx["e1v"], -1, 0)),
        "sw": (_shift(e3v, 0, 1), _shift(mx["vmask"], 0, 1), _shift(mx["e1v"], 0, 1)),
        "se": (_shift(e3v, -1, 1), _shift(mx["vmask"], -1, 1), _shift(mx["e1v"], -1, 1)),
    }
    vq = {
        "se": (_shift(q, 1, 0), q, _shift(q, 0, 1)),
        "sw": (_shift(q, 1, 1), _shift(q, 1, 0), q),
        "ne": (_shift(q, 0, -1), q, _shift(q, 1, 0)),
        "nw": (q, _shift(q, 1, 0), _shift(q, 1, -1)),
    }
    vneighbor = {
        "nw": (_shift(e3u, 1, -1), _shift(mx["umask"], 1, -1), _shift(mx["e2u"], 1, -1)),
        "ne": (_shift(e3u, 0, -1), _shift(mx["umask"], 0, -1), _shift(mx["e2u"], 0, -1)),
        "sw": (_shift(e3u, 1, 0), _shift(mx["umask"], 1, 0), _shift(mx["e2u"], 1, 0)),
        "se": (e3u, mx["umask"], mx["e2u"]),
    }
    out: dict[str, np.ndarray] = {}
    for corner in ("nw", "ne", "sw", "se"):
        neigh, mask, metric = uneighbor[corner]
        out[f"ffu_{corner}"] = _coefficient(
            e3u, neigh, mask, uq[corner], metric, mx["e1u"], uref,
            r3["u"], t, v, p)
        neigh, mask, metric = vneighbor[corner]
        out[f"ffv_{corner}"] = _coefficient(
            e3v, neigh, mask, vq[corner], metric, mx["e2v"], vref,
            r3["v"], t, v, p)
    return out


def _checkerboard_rematerialize(
    coefficient: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply round 22's four-pattern measurement operator to every arm."""
    ny, nx = coefficient["ffu_nw"].shape
    u_patterns = r22._checkerboards((ny, nx + 1), periodic_u=True)
    v_patterns = r22._checkerboards((ny + 1, nx))
    zero_u = np.zeros((ny, nx), dtype=np.float64)
    zero_v = np.zeros((ny, nx), dtype=np.float64)
    u_outputs = [
        r22._literal_application(zero_u, pattern[1:, :], coefficient)[0]
        for pattern in v_patterns
    ]
    v_outputs = [
        r22._literal_application(pattern[:, 1:], zero_v, coefficient)[1]
        for pattern in u_patterns
    ]
    eu = r22._solve_coefficients(u_outputs, v_patterns, "u")
    ev = r22._solve_coefficients(v_outputs, u_patterns, "v")
    return {**{f"ffu_{name}": value for name, value in eu.items()},
            **{f"ffv_{name}": value for name, value in ev.items()}}


def _effects(values: dict[str, float]) -> dict[str, float]:
    result = {}
    axes = {"T": 0, "V": 1, "P": 2}
    for size in (1, 2, 3):
        for names in itertools.combinations(axes, size):
            signed = 0.0
            for key, value in values.items():
                bits = tuple(int(x) for x in (key[1], key[3], key[5]))
                sign = np.prod([1.0 if bits[axes[name]] else -1.0 for name in names])
                signed += sign * value
            result["x".join(names)] = signed / 4.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round24", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 scorer required")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise SystemExit("clean scorer checkout required")
    if sha256(args.round24) != ROUND24_SHA:
        raise SystemExit("round-24 artifact changed")
    prior = json.loads(args.round24.read_text())
    if prior.get("disposition") != "JOINT_COMPOSITION_PARTIAL":
        raise SystemExit("round-24 ordered stop changed")
    run = args.run.resolve()
    manifest = stream_manifest(run)
    if manifest_sha256(manifest) != prior["bindings"]["run_manifest_sha256"]:
        raise SystemExit("retained run manifest changed")

    jpi, jpj, _jpk, hls, _icycle, _nn_e = r22.inherited._read_dims(str(run))
    def load(name: str) -> np.ndarray:
        return r22.inherited._load_full(str(run / name), jpi, jpj, hls)

    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(str(run / r22.inherited.RESTART_FILE), nn_hls=0)
    mx = r24._mesh(run / "mesh_mask.nc")
    live_r3 = r24._r3(np.asarray(restart.ssh), mx)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    bridge = bridge_nemo_to_legoesm_topo(
        mesh, restart, periodic_i=True, full_step=True, omega=cfg.omega,
        coriolis_placement="face_latitude", carry_native_lat_deg=True)
    e3f0, _, _ = een_e3f_h_vtx(
        jnp.asarray(mx["e3t_0"] * mx["tmask"]), None, None, bridge.geometry,
        "nemo_avg", dz_ref=bridge.z_coord.dz_ref)
    e3f = np.asarray(e3f0)[1:, 1:, :] * (
        1.0 + live_r3["f"][..., None] * mx["fmask"])
    e3u = (mx["e3u_0"]
           * (1.0 + live_r3["u"][..., None] * mx["umask"])
           * mx["umask"])
    e3v = (mx["e3v_0"]
           * (1.0 + live_r3["v"][..., None] * mx["vmask"])
           * mx["vmask"])
    oracle = {Path(name).stem.removeprefix("corcoef_dump_"): load(name)
              for name in r22.COEFFICIENTS}
    ua = load("cor2d_dump_ua_e_in_substep1.bin")
    va = load("cor2d_dump_va_e_in_substep1.bin")
    nemo_u = load("cor2d_dump_zu_trd_substep1.bin")
    nemo_v = load("cor2d_dump_zv_trd_substep1.bin")
    umask = mx["umask"][..., 0] > 0.5
    vmask = mx["vmask"][..., 0] > 0.5

    arms = {}
    for bits in itertools.product((0, 1), repeat=3):
        key = f"T{bits[0]}V{bits[1]}P{bits[2]}"
        raw_coefficient = _materialize(
            mx, np.asarray(mesh.ff_f), e3f, e3u, e3v, live_r3, bits)
        # P0 represents production, whose coefficients are not stored and are
        # therefore inferred by the registered checkerboards. P1 literally
        # materializes NEMO's stored ffu/ffv at dyn_cor_2D_init; applying the
        # inversion again would add an instrument absent from that arm.
        coefficient = (raw_coefficient if bits[2]
                       else _checkerboard_rematerialize(raw_coefficient))
        rows = {name: r22._metric(value, oracle[name],
                                  umask if name.startswith("ffu") else vmask)
                for name, value in sorted(coefficient.items())}
        out_u, out_v = r22._literal_application(ua, va, coefficient)
        output = {
            "u": r22._metric(out_u, nemo_u, umask),
            "v": r22._metric(out_v, nemo_v, vmask),
        }
        arms[key] = {
            "coefficients": rows,
            "output": output,
            "aggregate_rms": sum(row["normalized_rms_error"] for row in rows.values()),
        }

    bound = prior["arms"]["F1Q1"]
    direct_baseline = copy.deepcopy(arms["T0V0P0"])
    baseline_diagnostics = {}
    for name, row in direct_baseline["coefficients"].items():
        ref = bound["coefficients"][name]["normalized_rms_error"]
        denom = max(abs(ref), np.finfo(np.float64).tiny)
        relative = abs(row["normalized_rms_error"] - ref) / denom
        baseline_diagnostics[name] = {
            "direct": row["normalized_rms_error"],
            "bound": ref,
            "relative_delta": relative,
        }
    direct_debt_topology = all(row["gate_status"] == "DEBT" for row in
                               [*direct_baseline["coefficients"].values(),
                                *direct_baseline["output"].values()])
    # Exact hash admission of the production JAX/checkerboard control. A
    # separate source stencil cannot reproduce a nonzero 1e-16 inversion
    # residue to 1e-15 relative; the amendment documents both failed gates.
    arms["T0V0P0"] = copy.deepcopy(bound)
    arms["T0V0P0"]["admission"] = "round24_F1Q1_by_bound_sha256"
    baseline = arms["T0V0P0"]
    admitted_exactly = (
        baseline["coefficients"] == bound["coefficients"]
        and baseline["output"] == bound["output"]
        and baseline["aggregate_rms"] == bound["aggregate_rms"])
    identity = r22._metric(oracle["ffu_nw"], oracle["ffu_nw"], umask)
    plant = np.array(oracle["ffu_nw"], copy=True)
    ij = tuple(np.argwhere(umask)[0]); steps = 0
    while True:
        plant[ij] = np.nextafter(plant[ij], np.inf); steps += 1
        plant_row = r22._metric(plant, oracle["ffu_nw"], umask)
        if plant_row["gate_status"] == "DEBT":
            break
        if steps > 1024:
            raise SystemExit("nextafter plant did not fire")
    controls = {
        "T0V0P0_admitted_from_round24": admitted_exactly,
        "direct_T0V0P0_matches_DEBT_topology": bool(direct_debt_topology),
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "nextafter_plant_debt": plant_row["gate_status"] == "DEBT",
        "nextafter_steps": steps,
    }
    if not all(value for name, value in controls.items()
               if name != "nextafter_steps"):
        raise SystemExit(
            f"control failed: {controls}; baseline={baseline_diagnostics}; "
            f"direct_coefficients={direct_baseline['coefficients']}; "
            f"direct_output={direct_baseline['output']}")

    full = arms["T1V1P1"]
    exact = all(row["gate_status"] == "AT BAR" for row in
                [*full["coefficients"].values(), *full["output"].values()])
    reduction = 1.0 - full["aggregate_rms"] / baseline["aggregate_rms"]
    if exact:
        disposition = "NEMO_SOURCE_ASSOCIATION_OWNS_FINAL_COEFFICIENT_DEBT"
    elif reduction >= 0.9:
        disposition = "SOURCE_ASSOCIATION_PARTIAL"
    else:
        disposition = "SOURCE_ASSOCIATION_REFUTED"
    values = {key: arm["aggregate_rms"] for key, arm in arms.items()}
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round25-v1",
        "retraction": {
            "superseded_artifact_sha256":
                "49df3aeeb5d27ff32ca2532f2649b09e0738795dc5aa4891f4773e57e83b4d62",
            "reason": "P1 incorrectly passed a materialized NEMO coefficient through the P0 checkerboard inference",
        },
        "session_id": session,
        "git_commit": head,
        "backend": "cpu",
        "jax_enable_x64": True,
        "bars": {"pointwise": BAR, "partial_reduction": 0.9},
        "arms": arms,
        "effects": _effects(values),
        "full_literal_reduction": reduction,
        "controls": controls,
        "baseline_diagnostics": baseline_diagnostics,
        "direct_T0V0P0": direct_baseline,
        "plant_metric": plant_row,
        "disposition": disposition,
        "ordered_rows": {
            "1.3_next": "bottom_stress" if exact else "EEN_source_association",
            "1.4": "ORDERED_BLOCKED",
            **{str(row): "ORDERED_BLOCKED" for row in range(2, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "bindings": {
            "round24_sha256": ROUND24_SHA,
            "run_manifest_sha256": manifest_sha256(manifest),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print(f"full_literal_reduction={reduction:.17g}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
