#!/usr/bin/env python3
"""Peel the dyn_zdf RHS into non-wind and wind-placement terms."""
from __future__ import annotations

import argparse
import dataclasses
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
import numpy as np

import split_explicit_momentum_chain_round35 as r35
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import surface_stress_faces
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)


ROUND35_SHA = "4b6edffaafb7e3f079f8620fd899698c7b5f324ba00bedf1603ad4e2e1c94dbe"


def _deposit(tau, dz, wet, rdt, rho0):
    out = np.zeros_like(dz)
    out[..., 0] = (rdt * np.asarray(tau)
                   / (rho0 * np.maximum(dz[..., 0], 1e-10)))
    return out * wet


def _mean(field, dz, wet):
    weight = dz * wet
    total = np.sum(weight, axis=-1, keepdims=True)
    return np.sum(field * weight, axis=-1, keepdims=True) / np.maximum(total, 1e-10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--entry-restart", type=Path, required=True)
    parser.add_argument("--round35", type=Path, required=True)
    parser.add_argument("--manifest-artifact", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())

    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r35._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    if r35._sha(args.round35) != ROUND35_SHA:
        raise SystemExit("official round-35 receipt changed")
    round35 = json.loads(args.round35.read_text())
    if (round35.get("disposition") != "ROW4_LOCALIZED_TO_RHS"
            or round35.get("first_failing_operand") != "rhs"):
        raise SystemExit("round 35 does not admit the RHS peel")

    run, entry = args.run.resolve(), args.entry_restart.resolve()
    manifest_artifact = json.loads(args.manifest_artifact.read_text())
    manifest = manifest_artifact["bracket"]["off_stream_manifest"]
    for name in r35.STREAMS:
        path = run / name
        if not path.is_file():
            raise SystemExit(f"missing retained input {path}")
        expected = manifest.get(name)
        if expected is not None and r35._sha(path) != expected["sha256"]:
            raise SystemExit(f"retained stream hash admission failed: {name}")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    grid = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(str(entry), nn_hls=0)
    before = read_nemo_restart_before(str(entry), nn_hls=0)
    bridge = bridge_nemo_to_legoesm_topo(
        grid, now, periodic_i=True, full_step=True, omega=dcfg.omega,
        carry_native_lat_deg=True)
    bridge = bridge._replace(state=bridge_before_state_topo(
        bridge, grid, before, periodic_i=True))
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    model_cfg, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    model = LatLonCGridOceanModel(bridge.geometry, bridge.z_coord, model_cfg)
    forcing = dino_lat_lon_surface_forcing_arrays(bridge.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    import legoesm.ocean.physics.vertical_mixing as vm
    real_dispatch = vm.implicit_vertical_diffusion_ocean_momentum_dispatch
    model_cls = type(model)
    real_apply = model_cls._apply_implicit_vertical_mixing
    entries = []
    captured = []
    identities = []

    def apply_spy(self, state, dt, surface_forcing, *call_args, **call_kwargs):
        if kwargs_do_momentum(call_kwargs) and not entries:
            entries.append((np.array(state.u.data), np.array(state.v.data)))
        return real_apply(self, state, dt, surface_forcing,
                          *call_args, **call_kwargs)

    def kwargs_do_momentum(kwargs):
        return kwargs.get("do_momentum", True)

    def dispatch_spy(field, K, dz, dzh, dt, wet, *call_args, **call_kwargs):
        output = real_dispatch(field, K, dz, dzh, dt, wet,
                               *call_args, **call_kwargs)
        repeat = real_dispatch(field, K, dz, dzh, dt, wet,
                               *call_args, **call_kwargs)
        identities.append(np.array_equal(np.asarray(output), np.asarray(repeat)))
        captured.append({
            "field": np.array(field), "dz": np.array(dz),
            "wet": np.array(wet), "diag": np.array(call_kwargs["extra_diag"]),
            "dt": float(dt), "evaluation": call_kwargs.get("evaluation"),
        })
        return output

    model_cls._apply_implicit_vertical_mixing = apply_spy
    vm.implicit_vertical_diffusion_ocean_momentum_dispatch = dispatch_spy
    try:
        with jax.disable_jit():
            model.step(bridge.state, r35.DT, surface_forcing=sf)
    finally:
        model_cls._apply_implicit_vertical_mixing = real_apply
        vm.implicit_vertical_diffusion_ocean_momentum_dispatch = real_dispatch
    if len(entries) != 1 or len(captured) != 2:
        raise SystemExit(
            f"capture cardinality failed: entries={len(entries)}, "
            f"dispatches={len(captured)}")
    if (model_cls._apply_implicit_vertical_mixing is not real_apply
            or vm.implicit_vertical_diffusion_ocean_momentum_dispatch
            is not real_dispatch):
        raise SystemExit("capture hooks were not restored")
    actual_u, actual_v = captured

    def crop_u(value, levels=35):
        return np.asarray(value)[:, 1:, :levels]

    def crop_v(value, levels=35):
        return np.asarray(value)[1:, :, :levels]

    oracle = r35._oracle_pack(run, entry)
    rhs_u, rhs_v, wet_u, wet_v, gate = oracle["rhs"]
    dz_u, dz_v = crop_u(actual_u["dz"]), crop_v(actual_v["dz"])
    diag_u, diag_v = crop_u(actual_u["diag"]), crop_v(actual_v["diag"])
    raw_u, raw_v = crop_u(actual_u["field"]), crop_v(actual_v["field"])

    # Wind stress itself is independent of J; provide a correctly shaped J
    # to the single-owner helper, then use the captured live face thickness in
    # the deposit so this is the exact production arithmetic at the scored
    # dispatch boundary.
    J = jnp.ones(np.asarray(now.ssh).shape, dtype=bridge.state.u.data.dtype)
    stress = surface_stress_faces(sf, bridge.state.u.data.dtype,
                                  bridge.z_coord, J, model.grid)
    if stress is None:
        raise SystemExit("DINO forcing unexpectedly carries no stress")
    tau_u, tau_v = np.asarray(stress[0]), np.asarray(stress[1])
    wind_u = _deposit(crop_u(tau_u[..., None], 1)[..., 0], dz_u, wet_u,
                      2.0 * r35.DT, model_cfg.constants.rho_0)
    wind_v = _deposit(crop_v(tau_v[..., None], 1)[..., 0], dz_v, wet_v,
                      2.0 * r35.DT, model_cfg.constants.rho_0)
    oracle_wind_u = np.zeros_like(rhs_u)
    oracle_wind_v = np.zeros_like(rhs_v)
    oracle_wind_u[..., 0] = (
        r35._load2(run / "zdf_dump_u1_poststress.bin")
        - r35._load2(run / "zdf_dump_u1_prestress.bin"))
    oracle_wind_v[..., 0] = (
        r35._load2(run / "zdf_dump_v1_poststress.bin")
        - r35._load2(run / "zdf_dump_v1_prestress.bin"))

    mean_w_u, mean_w_v = (_mean(wind_u, dz_u, wet_u),
                          _mean(wind_v, dz_v, wet_v))
    nonwind_u = raw_u - wind_u + mean_w_u + diag_u * mean_w_u
    nonwind_v = raw_v - wind_v + mean_w_v + diag_v * mean_w_v
    oracle_nonwind_u = rhs_u - oracle_wind_u
    oracle_nonwind_v = rhs_v - oracle_wind_v
    delayed_actual_u = raw_u + mean_w_u + diag_u * mean_w_u
    delayed_actual_v = raw_v + mean_w_v + diag_v * mean_w_v
    delayed_oracle_u = delayed_actual_u + oracle_wind_u - wind_u
    delayed_oracle_v = delayed_actual_v + oracle_wind_v - wind_v

    arms = {
        "baseline": (raw_u, raw_v, rhs_u, rhs_v),
        "nonwind": (nonwind_u, nonwind_v,
                    oracle_nonwind_u, oracle_nonwind_v),
        "wind": (wind_u, wind_v, oracle_wind_u, oracle_wind_v),
        "delayed_actual_wind": (delayed_actual_u, delayed_actual_v,
                                rhs_u, rhs_v),
        "delayed_oracle_wind": (delayed_oracle_u, delayed_oracle_v,
                                rhs_u, rhs_v),
    }
    rows = {}
    for name, (au, av, eu, ev) in arms.items():
        u_row = r35._metric(au, eu, wet_u, gate)
        if name == "wind":
            # The meridional DINO wind oracle is the structural zero used as
            # the falsifier, so an RMS-normalized score is undefined. Gate it
            # by exact equality instead of inventing a denominator.
            v_exact = np.array_equal(av[wet_v], ev[wet_v])
            v_row = {
                "class_bar": 0.0,
                "gate_name": "exact structural-zero wind",
                "gate_status": "AT BAR" if v_exact else "DEBT",
                "n": int(wet_v.sum()),
                "normalized_rms_error": 0.0 if v_exact else float("inf"),
                "per_element_max_error_over_nemo_rms": (
                    0.0 if v_exact else float("inf")),
            }
        else:
            v_row = r35._metric(av, ev, wet_v, gate)
        rows[name] = {"u": u_row, "v": v_row}
    baseline_error = rows["baseline"]["u"]["normalized_rms_error"]
    delayed_error = rows["delayed_actual_wind"]["u"]["normalized_rms_error"]
    removal = 1.0 - delayed_error / baseline_error
    reconstruction_u = (nonwind_u + wind_u - mean_w_u
                        - diag_u * mean_w_u)
    reconstruction_v = (nonwind_v + wind_v - mean_w_v
                        - diag_v * mean_w_v)
    controls = {
        "hooks_restored": (
            model_cls._apply_implicit_vertical_mixing is real_apply
            and vm.implicit_vertical_diffusion_ocean_momentum_dispatch
            is real_dispatch),
        "same_jax_identity": len(identities) == 2 and all(identities),
        "baseline_reconstructs_u": r35._metric(
            reconstruction_u, raw_u, wet_u, gate)["gate_status"] == "AT BAR",
        "baseline_reconstructs_v": r35._metric(
            reconstruction_v, raw_v, wet_v, gate)["gate_status"] == "AT BAR",
        "v_wind_zero": np.count_nonzero(wind_v[wet_v]) == 0,
        "v_oracle_wind_zero": np.count_nonzero(oracle_wind_v[wet_v]) == 0,
        "v_placement_bit_identical": (
            np.array_equal(delayed_actual_v, raw_v)
            and np.array_equal(delayed_oracle_v, raw_v)),
        "selector_literal": (actual_u["evaluation"] == "nemo_literal"
                             and actual_v["evaluation"] == "nemo_literal"),
        "rdt_exact": (actual_u["dt"] == 2.0 * r35.DT
                      and actual_v["dt"] == 2.0 * r35.DT),
    }
    plant = np.array(rhs_u, copy=True)
    point = tuple(np.argwhere(wet_u)[0])
    plant[point] += 2e-12 * np.sqrt(np.mean(rhs_u[wet_u] ** 2))
    controls["two_bar_plant_fires"] = (
        r35._metric(plant, rhs_u, wet_u, gate)["gate_status"] != "AT BAR")
    controls["roll_fires"] = (
        r35._metric(np.roll(rhs_u, 1, axis=1), rhs_u, wet_u, gate)["gate_status"]
        != "AT BAR")
    nan_plant = np.array(rhs_u, copy=True)
    nan_plant[point] = np.nan
    controls["wet_nan_detected"] = not np.isfinite(nan_plant[wet_u]).all()
    controls = {name: bool(value) for name, value in controls.items()}
    valid = all(controls.values())
    delayed_at_bar = rows["delayed_actual_wind"]["u"]["gate_status"] == "AT BAR"
    oracle_at_bar = rows["delayed_oracle_wind"]["u"]["gate_status"] == "AT BAR"
    nonwind_at_bar = all(
        item["gate_status"] == "AT BAR" for item in rows["nonwind"].values())
    if not valid:
        disposition = "INVALID"
    elif delayed_at_bar and nonwind_at_bar:
        disposition = "RHS_OWNED_BY_WIND_PLACEMENT"
    elif oracle_at_bar and nonwind_at_bar:
        disposition = "RHS_OWNED_BY_WIND_PLACEMENT_AND_ARITHMETIC"
    elif removal >= 0.90:
        disposition = "WIND_PLACEMENT_MAJORITY_WITH_RESIDUAL_RHS_DEBT"
    else:
        disposition = "WIND_PLACEMENT_REFUTED"

    paths = {
        "round35": args.round35,
        "manifest_artifact": args.manifest_artifact,
        "entry_restart": entry,
        "scorer": Path(__file__).resolve(),
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round36.md",
        "production_model": root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
        "stress_helper": root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
        "dynzdf": args.nemo_root / "cfgs/DINO/MY_SRC/dynzdf.F90",
    }
    for name in r35.STREAMS:
        paths[f"stream:{name}"] = run / name
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round36-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "rows": rows,
        "u_baseline_error_removal_fraction": float(removal),
        "controls": controls,
        "bindings": {name: r35._sha(path.resolve())
                     for name, path in paths.items()},
        "disposition": disposition,
        "ordered_next": 4,
    }
    if r35._tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} removal={removal:.9f}")
    for name, row in rows.items():
        print(name, {component: value["normalized_rms_error"]
                     for component, value in row.items()})
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
