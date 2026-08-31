#!/usr/bin/env python3
"""Score production dyn_zdf at the raw, barotropic-free dispatch boundary."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
import netCDF4
import numpy as np

import split_explicit_momentum_chain_round29 as r29
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
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
from zu_frc_term_walk import _load_full_3d


ROUND32_SHA = "0b284d7c8880646850daee86b2b85fb13fb65540402e2e9a02760b0ed60ba86f"
CHAIN_END_SHA = "ce6fff6690ff6fbfa1023b4cf3eafbd36d0b86938ef47c5f7524accc410d4975"
DT = 2700.0
STREAMS = (
    "stp_dump_07_dynspg_kt00005761_u.bin",
    "stp_dump_07_dynspg_kt00005761_v.bin",
    "stp_dump_07_dynspg_kt00005761_ub.bin",
    "stp_dump_07_dynspg_kt00005761_vb.bin",
    "stp_dump_08_dynzdf_kt00005761_u.bin",
    "stp_dump_08_dynzdf_kt00005761_v.bin",
    "zdf_dump_u1_prestress.bin",
    "zdf_dump_u1_poststress.bin",
    "zdf_dump_v1_prestress.bin",
    "zdf_dump_v1_poststress.bin",
    "seq_dump_r3u_aaa_kt00005761.bin",
    "seq_dump_r3v_aaa_kt00005761.bin",
    "drg_dump_rCdU_bot.bin",
    "dump_avm.bin",
    "mesh_mask.nc",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _load2(path: Path, jpi=56, jpj=203, hls=2) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size == jpi * jpj:
        return raw.reshape(jpj, jpi)[hls:-hls, hls:-hls]
    ni, nj = jpi - 2 * hls, jpj - 2 * hls
    if raw.size == ni * nj:
        return raw.reshape(nj, ni)
    raise SystemExit(f"{path}: unexpected 2-D element count {raw.size}")


def _metric(candidate, oracle, mask, gate=None):
    return r29._score(np.asarray(candidate), np.asarray(oracle),
                      np.asarray(mask, dtype=bool), gate)


def _oracle_pack(run: Path, entry: Path):
    jpi, jpj, jpk, hls = 56, 203, 36, 2
    l3 = lambda name: _load_full_3d(
        str(run / name), jpi, jpj, jpk - 1, hls)
    now = read_nemo_restart(str(entry), nn_hls=0)
    before = read_nemo_restart_before(str(entry), nn_hls=0)
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        def z3(name):
            return np.moveaxis(np.asarray(ds[name][0]), 0, -1)
        tmask = z3("tmask") > 0.5
        umask = z3("umask") > 0.5
        vmask = z3("vmask") > 0.5
        e3t0, e3u0, e3v0 = z3("e3t_0"), z3("e3u_0"), z3("e3v_0")
        e3uw0, e3vw0 = z3("e3uw_0"), z3("e3vw_0")
        e1e2t = np.asarray(ds["e1t"][0]) * np.asarray(ds["e2t"][0])
        e1e2u = np.asarray(ds["e1u"][0]) * np.asarray(ds["e2u"][0])
        e1e2v = np.asarray(ds["e1v"][0]) * np.asarray(ds["e2v"][0])
    wu, wv = umask[..., :35], vmask[..., :35]
    hu0 = np.sum(e3u0 * umask, axis=-1)
    hv0 = np.sum(e3v0 * vmask, axis=-1)
    ssh = np.asarray(now.ssh)
    r3u_now = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=1))
               / np.where(hu0 * e1e2u != 0.0, hu0 * e1e2u, 1.0))
    r3v_now = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=0))
               / np.where(hv0 * e1e2v != 0.0, hv0 * e1e2v, 1.0))
    r3ua = _load2(run / "seq_dump_r3u_aaa_kt00005761.bin")
    r3va = _load2(run / "seq_dump_r3v_aaa_kt00005761.bin")
    e3uaa = e3u0 * (1.0 + r3ua[..., None] * umask)
    e3vaa = e3v0 * (1.0 + r3va[..., None] * vmask)
    e3uwmm = e3uw0 * (1.0 + r3u_now[..., None] * umask)
    e3vwmm = e3vw0 * (1.0 + r3v_now[..., None] * vmask)

    rdt = 2.0 * DT
    kr_u = l3("stp_dump_07_dynspg_kt00005761_u.bin")
    kr_v = l3("stp_dump_07_dynspg_kt00005761_v.bin")
    ub = _load2(run / "stp_dump_07_dynspg_kt00005761_ub.bin")
    vb = _load2(run / "stp_dump_07_dynspg_kt00005761_vb.bin")
    rhs_u = (np.asarray(before.u)[..., :35] + rdt * kr_u - ub[..., None]) * wu
    rhs_v = (np.asarray(before.v)[..., :35] + rdt * kr_v - vb[..., None]) * wv
    rcd = _load2(run / "drg_dump_rCdU_bot.bin")
    drag_u = -0.5 * (rcd + np.roll(rcd, -1, axis=1))
    drag_v = -0.5 * (rcd + np.roll(rcd, -1, axis=0))
    kk = np.arange(35)[None, None, :]
    bot_u = wu & (kk == wu.sum(-1)[..., None] - 1)
    bot_v = wv & (kk == wv.sum(-1)[..., None] - 1)
    diag_u = rdt * drag_u[..., None] / np.maximum(e3uaa[..., :35], 1e-10) * bot_u
    diag_v = rdt * drag_v[..., None] / np.maximum(e3vaa[..., :35], 1e-10) * bot_v
    rhs_u -= diag_u * ub[..., None]
    rhs_v -= diag_v * vb[..., None]
    rhs_u[..., 0] += (_load2(run / "zdf_dump_u1_poststress.bin")
                      - _load2(run / "zdf_dump_u1_prestress.bin"))
    rhs_v[..., 0] += (_load2(run / "zdf_dump_v1_poststress.bin")
                      - _load2(run / "zdf_dump_v1_prestress.bin"))
    avm = l3("dump_avm.bin")
    ku = (0.5 * (avm + np.roll(avm, -1, axis=1)))[..., 1:35]
    kv = (0.5 * (avm + np.roll(avm, -1, axis=0)))[..., 1:35]
    ku *= wu[..., 1:] & wu[..., :-1]
    kv *= wv[..., 1:] & wv[..., :-1]
    return {
        "rhs": (rhs_u, rhs_v, wu, wv,
                "dyn_zdf (momentum implicit vertical solve)"),
        "viscosity": (ku, kv, wu[..., 1:] & wu[..., :-1],
                      wv[..., 1:] & wv[..., :-1], None),
        "cell_thickness": (e3uaa[..., :35], e3vaa[..., :35], wu, wv, None),
        "interface_thickness": (e3uwmm[..., 1:35], e3vwmm[..., 1:35],
                                wu[..., 1:] & wu[..., :-1],
                                wv[..., 1:] & wv[..., :-1], None),
        "drag_diagonal": (diag_u, diag_v, wu, wv, None),
        "output": (l3("stp_dump_08_dynzdf_kt00005761_u.bin"),
                   l3("stp_dump_08_dynzdf_kt00005761_v.bin"),
                   wu, wv, "dyn_zdf (momentum implicit vertical solve)"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--entry-restart", type=Path, required=True)
    parser.add_argument("--round32", type=Path, required=True)
    parser.add_argument("--chain-end", type=Path, required=True)
    parser.add_argument("--manifest-artifact", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())

    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    if _sha(args.round32) != ROUND32_SHA:
        raise SystemExit("round-32 certification changed")
    round32 = json.loads(args.round32.read_text())
    if (round32.get("disposition") != "NO_STRICT_SSH_FAILURE"
            or any(value is not None for value in
                   round32["first_strict_failure_substep"].values())):
        raise SystemExit("round-32 receipt does not release row 4")
    if _sha(args.chain_end) != CHAIN_END_SHA:
        raise SystemExit("ZDF chain-end receipt changed")
    chain_end = json.loads(args.chain_end.read_text())
    if (chain_end.get("disposition") != "VERIFIED"
            or chain_end["rows"]["31"]["disposition"] != "VERIFIED"):
        raise SystemExit("row-31 literal kernel receipt is not promotable")

    run, entry = args.run.resolve(), args.entry_restart.resolve()
    manifest_artifact = json.loads(args.manifest_artifact.read_text())
    manifest = manifest_artifact["bracket"]["off_stream_manifest"]
    for name in STREAMS:
        path = run / name
        if not path.is_file():
            raise SystemExit(f"missing retained input {path}")
        expected = manifest.get(name)
        if expected is not None and _sha(path) != expected["sha256"]:
            raise SystemExit(f"retained stream hash admission failed: {name}")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    grid = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(str(entry), nn_hls=0)
    before = read_nemo_restart_before(str(entry), nn_hls=0)
    bridge = bridge_nemo_to_legoesm_topo(
        grid, now, periodic_i=True, full_step=True, omega=dcfg.omega,
        carry_native_lat_deg=True)
    if bridge.state.u_before is not None:
        raise SystemExit("bridge unexpectedly already has a before level")
    bridge = bridge._replace(state=bridge_before_state_topo(
        bridge, grid, before, periodic_i=True))
    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    model_cfg, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    model = LatLonCGridOceanModel(bridge.geometry, bridge.z_coord, model_cfg)
    forcing = dino_lat_lon_surface_forcing_arrays(bridge.geometry, cfg)
    surface_forcing = dino_step_surface_forcing(forcing)

    import legoesm.ocean.physics.vertical_mixing as vm
    real_dispatch = vm.implicit_vertical_diffusion_ocean_momentum_dispatch
    captured = []
    identity_checks = []

    def capture(field, K, dz, dzh, dt, wet, *call_args, **call_kwargs):
        output = real_dispatch(field, K, dz, dzh, dt, wet,
                               *call_args, **call_kwargs)
        # Replay before converting anything to NumPy: NumPy inputs take a
        # different eager coercion route and are not a byte-identity control
        # for the production JAX call.
        identity_output = real_dispatch(field, K, dz, dzh, dt, wet,
                                        *call_args, **call_kwargs)
        identity_checks.append(np.array_equal(np.asarray(identity_output),
                                              np.asarray(output)))
        captured.append({
            "field": np.array(field), "K": np.array(K),
            "dz": np.array(dz), "dzh": np.array(dzh),
            "dt": float(dt), "wet": np.array(wet),
            "evaluation": call_kwargs.get("evaluation"),
            "extra_diag": np.array(call_kwargs.get("extra_diag")),
            "output": np.array(output),
        })
        return output

    vm.implicit_vertical_diffusion_ocean_momentum_dispatch = capture
    try:
        with jax.disable_jit():
            model.step(bridge.state, DT, surface_forcing=surface_forcing)
    finally:
        vm.implicit_vertical_diffusion_ocean_momentum_dispatch = real_dispatch
    if vm.implicit_vertical_diffusion_ocean_momentum_dispatch is not real_dispatch:
        raise SystemExit("dispatch hook restoration failed")
    if len(captured) != 2:
        raise SystemExit(f"expected exactly U/V dispatch calls, got {len(captured)}")
    actual_u, actual_v = captured
    if (actual_u["field"].shape[:2] != (199, 53)
            or actual_v["field"].shape[:2] != (200, 52)):
        raise SystemExit(
            f"ambiguous U/V dispatch order: {actual_u['field'].shape}, "
            f"{actual_v['field'].shape}")

    def crop_u(value, levels):
        return np.asarray(value)[:, 1:, :levels]

    def crop_v(value, levels):
        return np.asarray(value)[1:, :, :levels]

    reference = _oracle_pack(run, entry)
    actual = {
        "rhs": (crop_u(actual_u["field"], 35), crop_v(actual_v["field"], 35)),
        "viscosity": (crop_u(actual_u["K"], 34), crop_v(actual_v["K"], 34)),
        "cell_thickness": (crop_u(actual_u["dz"], 35), crop_v(actual_v["dz"], 35)),
        "interface_thickness": (crop_u(actual_u["dzh"], 34),
                                crop_v(actual_v["dzh"], 34)),
        "drag_diagonal": (crop_u(actual_u["extra_diag"], 35),
                          crop_v(actual_v["extra_diag"], 35)),
        "output": (crop_u(actual_u["output"], 35),
                   crop_v(actual_v["output"], 35)),
    }
    rows = {}
    first_failure = None
    selector_row = {
        "u": actual_u["evaluation"] == "nemo_literal",
        "v": actual_v["evaluation"] == "nemo_literal",
        "dt_u_exact": actual_u["dt"] == 2.0 * DT,
        "dt_v_exact": actual_v["dt"] == 2.0 * DT,
        "wet_u_exact": np.array_equal(crop_u(actual_u["wet"], 35),
                                      reference["rhs"][2]),
        "wet_v_exact": np.array_equal(crop_v(actual_v["wet"], 35),
                                      reference["rhs"][3]),
    }
    rows["selector_dt_wet"] = selector_row
    if not all(selector_row.values()):
        first_failure = "selector_dt_wet"
    for name in ("rhs", "viscosity", "cell_thickness",
                 "interface_thickness", "drag_diagonal", "output"):
        eu, ev, mu, mv, gate = reference[name]
        au, av = actual[name]
        rows[name] = {"u": _metric(au, eu, mu, gate),
                      "v": _metric(av, ev, mv, gate)}
        if first_failure is None and any(
                item["gate_status"] != "AT BAR" for item in rows[name].values()):
            first_failure = name

    controls = {
        "dispatch_restored": (
            vm.implicit_vertical_diffusion_ocean_momentum_dispatch is real_dispatch),
        "wrong_dt_rejected": actual_u["dt"] != DT,
        "all_capture_finite": all(np.isfinite(item[key]).all()
                                  for item in captured
                                  for key in ("field", "K", "dz", "dzh",
                                              "extra_diag", "output")),
    }
    controls["identity_dispatch_exact"] = (
        len(identity_checks) == 2 and all(identity_checks))
    rhs_u, _, wet_u, _, rhs_gate = reference["rhs"]
    rhs_plant = np.array(rhs_u, copy=True)
    point = tuple(np.argwhere(wet_u)[0])
    rhs_plant[point] += 2e-12 * np.sqrt(np.mean(rhs_u[wet_u] ** 2))
    controls["two_bar_rhs_plant_fires"] = (
        _metric(rhs_plant, rhs_u, wet_u, rhs_gate)["gate_status"] != "AT BAR")
    controls["one_cell_roll_fires"] = (
        _metric(np.roll(rhs_u, 1, axis=1), rhs_u, wet_u, rhs_gate)["gate_status"]
        != "AT BAR")
    nan_plant = np.array(rhs_u, copy=True)
    nan_plant[point] = np.nan
    controls["wet_nan_detected"] = not np.isfinite(nan_plant[wet_u]).all()
    out_u, _, out_wet_u, _, out_gate = reference["output"]
    out_plant = np.array(out_u, copy=True)
    out_point = tuple(np.argwhere(out_wet_u)[0])
    out_plant[out_point] += 2e-12 * np.sqrt(np.mean(out_u[out_wet_u] ** 2))
    controls["two_bar_output_plant_fires"] = (
        _metric(out_plant, out_u, out_wet_u, out_gate)["gate_status"] != "AT BAR")
    valid = all(controls.values())
    output_at_bar = all(
        item["gate_status"] == "AT BAR" for item in rows["output"].values())
    input_names = ("selector_dt_wet", "rhs", "viscosity", "cell_thickness",
                   "interface_thickness", "drag_diagonal")
    inputs_at_bar = all(
        all(row.values()) if name == "selector_dt_wet" else
        all(item["gate_status"] == "AT BAR" for item in row.values())
        for name in input_names for row in (rows[name],))
    if not valid:
        disposition = "INVALID"
    elif inputs_at_bar and output_at_bar:
        disposition = "ROW4_DYN_ZDF_AT_BAR"
    elif not inputs_at_bar and output_at_bar:
        disposition = "ROW4_OUTPUT_AT_BAR_WITH_BOUNDED_INPUT_DEBT"
    elif inputs_at_bar and not output_at_bar:
        disposition = "INVALID_KERNEL_CONTRADICTION"
    else:
        disposition = f"ROW4_LOCALIZED_TO_{first_failure.upper()}"

    nemo_sources = {
        "dynzdf": args.nemo_root / "cfgs/DINO/MY_SRC/dynzdf.F90",
        "stpmlf": args.nemo_root / "cfgs/DINO/MY_SRC/stpmlf.F90",
    }
    paths = {
        "round32": args.round32,
        "chain_end": args.chain_end,
        "manifest_artifact": args.manifest_artifact,
        "entry_restart": entry,
        "scorer": Path(__file__).resolve(),
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round35.md",
        "production_model": root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
        **nemo_sources,
    }
    for name in STREAMS:
        paths[f"stream:{name}"] = run / name
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round35-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "capture_boundary": "raw momentum dispatch before barotropic mean readd",
        "dispatch_call_count": len(captured),
        "first_failing_operand": first_failure,
        "rows": rows,
        "controls": controls,
        "bindings": {name: _sha(path.resolve()) for name, path in paths.items()},
        "disposition": disposition,
        "ordered_next": 5 if output_at_bar else 4,
    }
    if _tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_failing_operand={first_failure}")
    for name, row in rows.items():
        if name == "selector_dt_wet":
            print(name, row)
        else:
            print(name, {key: value["gate_status"] for key, value in row.items()})
    return 0 if valid and disposition != "INVALID_KERNEL_CONTRADICTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
