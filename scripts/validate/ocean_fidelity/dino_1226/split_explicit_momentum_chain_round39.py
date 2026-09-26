#!/usr/bin/env python3
"""Run the retained-input 2^3 residual ZAD operand factorial."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import split_explicit_momentum_chain_round38 as r38
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import (
    _u_east_to_face_periodic,
    _v_north_to_face,
)
from zu_frc_term_walk import _load_full_3d


ROUND38_SHA = "51737434a48a2c625d0ac5472ab4a5914eca32a4a5d615034abb57ec993fb79b"
ROUND38_DISPOSITION = "WZV_REFUTED_AS_ZAD_MAJORITY_OWNER"


def _literal_zad(velocity: np.ndarray, w: np.ndarray, thickness: np.ndarray,
                 area_t: np.ndarray, area_face: np.ndarray,
                 component: str) -> np.ndarray:
    """Literal `dynzad.F90:83-119` on halo-stripped NEMO staggering."""
    if velocity.shape != (199, 52, 36) or w.shape != (199, 52, 36):
        raise SystemExit(f"literal {component} operand shape changed")
    out = np.zeros((199, 52, 35), dtype=np.float64)
    carried = np.zeros((199, 52), dtype=np.float64)
    for k in range(34):  # Fortran jk=1..jpk-2
        zwf = area_t * w[..., k + 1]
        if component == "u":
            neighbour = np.roll(zwf, -1, axis=1)
        else:
            neighbour = np.empty_like(zwf)
            neighbour[:-1] = zwf[1:]
            neighbour[-1] = zwf[-1]
        fresh = (neighbour + zwf) * (
            velocity[..., k] - velocity[..., k + 1])
        factor = (-np.float64(0.25) / area_face) / thickness[..., k]
        out[..., k] = factor * (carried + fresh)
        carried = fresh
    factor = (-np.float64(0.25) / area_face) / thickness[..., 34]
    out[..., 34] = factor * carried
    return out


def _metric_equal(left: dict, right: dict) -> bool:
    keys = ("normalized_rms_error", "correlation", "rms_ratio",
            "mean_abs_ratio", "per_element_max_error_over_nemo_rms",
            "gate_status")
    return all(left[key] == right[key] for key in keys)


def _normalized_max(left, right, mask=None) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    active = np.ones(right.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    scale = max(float(np.sqrt(np.mean(right[active] ** 2))),
                np.finfo(np.float64).tiny)
    return float(np.max(np.abs(left[active] - right[active])) / scale)


def _contrasts(errors: dict[str, float]) -> dict[str, float]:
    result = {}
    factors = ("W", "H", "A")
    for order in range(1, 4):
        for names in itertools.combinations(factors, order):
            values = []
            for key, error in errors.items():
                bits = dict(zip(factors, (int(key[1]), int(key[3]), int(key[5]))))
                sign = np.prod([1.0 if bits[name] else -1.0 for name in names])
                values.append(sign * error)
            result["x".join(names)] = float(sum(values) / 4.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--restart-file", default="DINO_00005760_restart.nc")
    parser.add_argument("--round38", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r38._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    round38_path = args.round38.resolve()
    if r38._sha(round38_path) != ROUND38_SHA:
        raise SystemExit("official round-38 receipt changed")
    prior = json.loads(round38_path.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != ROUND38_DISPOSITION):
        raise SystemExit("round 38 does not open this factorial")

    run = args.run_stepdump.resolve()
    for name, expected in r38.EXPECTED.items():
        if r38._sha(run / name) != expected:
            raise SystemExit(f"retained operand changed: {name}")
    bridge, cfg, model_cfg, model, forcing, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file=args.restart_file,
    )
    if (model_cfg.vertical_momentum_scheme != "nemo_advective"
            or model_cfg.zad_bottom_face_mask != "nemo_faithful"):
        raise SystemExit("faithful ZAD selector/default changed")
    ldf_state = (state.T_before.data, state.S_before.data,
                 state.u_before.data, state.v_before.data)

    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe

    captured: dict[str, object] = {}
    real_bc = pe._bc_vertical_and_depthmean_velocity

    def spy(*values, **keywords):
        result = real_bc(*values, **keywords)
        if not captured:
            captured.update({
                "h_u": result[0], "h_v": result[1], "w": result[3],
                "u": values[1], "v": values[2],
                "u_active": values[3], "v_active": values[4],
                "grid": values[5],
            })
        return result

    pe._bc_vertical_and_depthmean_velocity = spy
    try:
        with jax.disable_jit():
            model.tendencies_with_diagnostics(
                state, surface_forcing=sf, dt=twin.DT, ldf_state=ldf_state)
    finally:
        pe._bc_vertical_and_depthmean_velocity = real_bc
    if not captured or pe._bc_vertical_and_depthmean_velocity is not real_bc:
        raise SystemExit("production operand capture/restoration failed")

    with netCDF4.Dataset(run / "mesh_mask.nc") as dataset:
        area_t = np.asarray(dataset["e1t"][0]) * np.asarray(dataset["e2t"][0])
        area_u = np.asarray(dataset["e1u"][0]) * np.asarray(dataset["e2u"][0])
        area_v = np.asarray(dataset["e1v"][0]) * np.asarray(dataset["e2v"][0])
        e3u0 = np.moveaxis(np.asarray(dataset["e3u_0"][0]), 0, -1)
        e3v0 = np.moveaxis(np.asarray(dataset["e3v_0"][0]), 0, -1)
        umask36 = np.moveaxis(np.asarray(dataset["umask"][0]), 0, -1)
        vmask36 = np.moveaxis(np.asarray(dataset["vmask"][0]), 0, -1)
        tmask36 = np.moveaxis(np.asarray(dataset["tmask"][0]), 0, -1)
    umask, vmask = umask36[..., :35] > 0.5, vmask36[..., :35] > 0.5
    masks = {"u": umask, "v": vmask}
    oracle = {
        "u": _load_full_3d(str(run / "zad_dump_du.bin"), 56, 203, 35, 2),
        "v": _load_full_3d(str(run / "zad_dump_dv.bin"), 56, 203, 35, 2),
    }

    restart = read_nemo_restart(str(run / args.restart_file), nn_hls=0)
    ssh = np.asarray(restart.ssh)
    hu0 = np.sum(e3u0 * umask36, axis=-1)
    hv0 = np.sum(e3v0 * vmask36, axis=-1)
    weighted = area_t * ssh
    east = np.roll(weighted, -1, axis=1)
    north = np.empty_like(weighted)
    north[:-1] = weighted[1:]
    north[-1] = weighted[-1]
    r3u = (np.float64(0.5) * (weighted + east)
           / np.where(hu0 * area_u != 0.0, hu0 * area_u, 1.0))
    r3v = (np.float64(0.5) * (weighted + north)
           / np.where(hv0 * area_v != 0.0, hv0 * area_v, 1.0))
    e3_nemo_raw = {
        "u": e3u0 * (1.0 + r3u[..., None] * umask36),
        "v": e3v0 * (1.0 + r3v[..., None] * vmask36),
    }
    e3_nemo_native = {
        "u": jnp.asarray(_u_east_to_face_periodic(e3_nemo_raw["u"])),
        "v": jnp.asarray(_v_north_to_face(e3_nemo_raw["v"])),
    }

    production_w = jnp.asarray(captured["w"])
    nemo_w_raw = r38._load_ww(run / "wzv_dump_ww_call1.bin")
    nemo_w_native = np.zeros(np.asarray(production_w).shape, dtype=np.float64)
    nemo_w_native[..., :36] = nemo_w_raw
    w_native = {0: production_w, 1: jnp.asarray(nemo_w_native)}
    w_raw = {0: np.asarray(production_w)[..., :36], 1: nemo_w_raw}

    native = {
        "u": {
            "velocity": captured["u"], "h0": captured["h_u"],
            "h1": e3_nemo_native["u"], "active": captured["u_active"],
        },
        "v": {
            "velocity": captured["v"], "h0": captured["h_v"],
            "h1": e3_nemo_native["v"], "active": captured["v_active"],
        },
    }
    raw = {
        "u": {
            "velocity": np.asarray(captured["u"])[:, 1:, :],
            "h0": np.asarray(captured["h_u"])[:, 1:, :],
            "h1": e3_nemo_raw["u"], "area": area_u,
        },
        "v": {
            "velocity": np.asarray(captured["v"])[1:, :, :],
            "h0": np.asarray(captured["h_v"])[1:, :, :],
            "h1": e3_nemo_raw["v"], "area": area_v,
        },
    }

    alignment = {
        "u_velocity_vs_restart_max_abs": float(np.max(
            np.abs(raw["u"]["velocity"] - np.asarray(restart.u)))),
        "v_velocity_vs_restart_max_abs": float(np.max(
            np.abs(raw["v"]["velocity"] - np.asarray(restart.v)))),
        "t_area_normalized_max": _normalized_max(
            np.asarray(captured["grid"].area_T), area_t,
            tmask36[..., 0] > 0.5),
        "u_area_normalized_max": _normalized_max(
            np.asarray(captured["grid"].dx_u * captured["grid"].dy_u)[:, 1:],
            area_u, umask36[..., 0] > 0.5),
        "v_area_normalized_max": _normalized_max(
            np.asarray(captured["grid"].dx_v * captured["grid"].dy_v)[1:, :],
            area_v, vmask36[..., 0] > 0.5),
    }
    if any(value > 1e-15 for value in alignment.values()):
        raise SystemExit(f"velocity/metric alignment changed: {alignment}")

    rows: dict[str, dict[str, dict]] = {}
    for w_bit, h_bit, a_bit in itertools.product((0, 1), repeat=3):
        name = f"W{w_bit}H{h_bit}A{a_bit}"
        rows[name] = {}
        for component in ("u", "v"):
            if a_bit == 0:
                with jax.disable_jit():
                    value = r38._direct(
                        native[component]["velocity"], w_native[w_bit],
                        native[component][f"h{h_bit}"],
                        native[component]["active"], captured["grid"],
                        component, model_cfg.zad_bottom_face_mask)
                candidate = r38._crop(component, value)
            else:
                candidate = _literal_zad(
                    raw[component]["velocity"], w_raw[w_bit],
                    raw[component][f"h{h_bit}"], area_t,
                    raw[component]["area"], component)
            rows[name][component] = r38._metric(
                candidate, oracle[component], masks[component])

    controls = {
        "hook_restored": pe._bc_vertical_and_depthmean_velocity is real_bc,
        "production_reproduces_round38_u": _metric_equal(
            rows["W0H0A0"]["u"], prior["arms"]["production"]["u"]),
        "production_reproduces_round38_v": _metric_equal(
            rows["W0H0A0"]["v"], prior["arms"]["production"]["v"]),
        "call1_reproduces_round38_u": _metric_equal(
            rows["W1H0A0"]["u"], prior["arms"]["nemo_ww_call1"]["u"]),
        "call1_reproduces_round38_v": _metric_equal(
            rows["W1H0A0"]["v"], prior["arms"]["nemo_ww_call1"]["v"]),
    }
    for component in ("u", "v"):
        wet, truth = masks[component], oracle[component]
        controls[f"identity_{component}"] = (
            r38._metric(truth, truth, wet)["gate_status"] == "AT BAR")
        controls[f"two_bar_{component}"] = r38._two_bar_plant(truth, wet)
        controls[f"roll_{component}"] = (
            r38._metric(np.roll(truth, 1, axis=1), truth, wet)["gate_status"]
            != "AT BAR")
        controls[f"sign_{component}"] = (
            r38._metric(-truth, truth, wet)["gate_status"] != "AT BAR")
        nan = np.array(truth, copy=True)
        nan[tuple(np.argwhere(wet)[0])] = np.nan
        controls[f"wet_nan_{component}"] = not np.isfinite(nan[wet]).all()
    controls = {name: bool(value) for name, value in controls.items()}

    errors = {
        component: {name: row[component]["normalized_rms_error"]
                    for name, row in rows.items()}
        for component in ("u", "v")
    }
    effects = {component: _contrasts(errors[component])
               for component in ("u", "v")}
    residual = {component: errors[component]["W1H0A0"]
                for component in ("u", "v")}
    conditional_removal = {
        "H_given_W": {component: 1.0 - errors[component]["W1H1A0"]
                       / residual[component] for component in ("u", "v")},
        "A_given_W": {component: 1.0 - errors[component]["W1H0A1"]
                       / residual[component] for component in ("u", "v")},
        "H_and_A_given_W": {
            component: 1.0 - errors[component]["W1H1A1"] / residual[component]
            for component in ("u", "v")},
    }
    valid = all(controls.values())
    full_at_bar = all(rows["W1H1A1"][component]["gate_status"] == "AT BAR"
                      for component in ("u", "v"))
    owners = [name for name, item in conditional_removal.items()
              if all(value >= 0.90 for value in item.values())]
    if not valid:
        disposition = "INVALID"
    elif full_at_bar and owners:
        disposition = f"ZAD_RESIDUAL_CLOSED_BY_{owners[0].upper()}"
    elif full_at_bar:
        disposition = "ZAD_RESIDUAL_CLOSED_BY_THREE_FACTOR_COMPOSITION"
    else:
        disposition = "ZAD_RESIDUAL_OPEN_AFTER_FULL_LITERAL_ARM"

    paths = {
        "round38": round38_path,
        "restart": run / args.restart_file,
        "mesh_mask": run / "mesh_mask.nc",
        "wzv_call1": run / "wzv_dump_ww_call1.bin",
        "zad_u": run / "zad_dump_du.bin",
        "zad_v": run / "zad_dump_dv.bin",
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round39.md",
        "scorer": Path(__file__).resolve(),
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round39-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "factors": {"W": "NEMO call1 ww", "H": "NEMO live e3u/e3v(Kmm)",
                    "A": "dynzad.F90:83-119 literal source association"},
        "alignment": alignment,
        "arms": rows,
        "signed_factorial_effects": effects,
        "conditional_residual_removal": conditional_removal,
        "controls": controls,
        "bindings": {name: r38._sha(path.resolve()) for name, path in paths.items()},
        "disposition": disposition,
        "ordered_next": 5 if full_at_bar else 4,
    }
    if r38._tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, row in rows.items():
        print(name, {component: row[component]["normalized_rms_error"]
                     for component in ("u", "v")})
    print(f"conditional_residual_removal={conditional_removal}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
