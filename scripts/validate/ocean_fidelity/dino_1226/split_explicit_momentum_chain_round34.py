#!/usr/bin/env python3
"""Capture full-step dyn_zdf dispatch operands and compare to row-31 pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
import split_explicit_momentum_chain_round33 as r33
import spg_substep_chain as inherited
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_restart, read_nemo_restart_before,
)
from zu_frc_term_walk import _load_full_3d


ROUND33_SHA = "20ebb1d9921b7ec6f7eb9b90c1da73278a89e7a16f38b94ab52860a046724b01"
CHAIN_END_SHA = "ce6fff6690ff6fbfa1023b4cf3eafbd36d0b86938ef47c5f7524accc410d4975"


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
    if raw.size == (jpi - 2 * hls) * (jpj - 2 * hls):
        return raw.reshape(jpj - 2 * hls, jpi - 2 * hls)
    raise SystemExit(f"{path}: unexpected 2-D element count {raw.size}")


def _metric(candidate, oracle, mask, gate):
    return r29._score(np.asarray(candidate), np.asarray(oracle),
                      np.asarray(mask, dtype=bool), gate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--entry-restart", type=Path, required=True)
    parser.add_argument("--round32", type=Path, required=True)
    parser.add_argument("--round33", type=Path, required=True)
    parser.add_argument("--chain-end", type=Path, required=True)
    parser.add_argument("--manifest-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        "RETRACTED: round 34 captured the invalid round-33 fed arm after its "
        "double barotropic-mean subtraction. Its operand table does not score "
        "the production row-4 boundary. Use round 35."
    )
    set_policy(PrecisionPolicy.fp64())

    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    if _sha(args.round33) != ROUND33_SHA:
        raise SystemExit("round-33 receipt changed")
    row33 = json.loads(args.round33.read_text())
    if row33.get("disposition") != "ROW4_DYN_ZDF_DIVERGED":
        raise SystemExit("round-33 receipt does not open the operand peel")
    if _sha(args.chain_end) != CHAIN_END_SHA:
        raise SystemExit("ZDF chain-end receipt changed")
    chain = json.loads(args.chain_end.read_text())
    if (chain.get("disposition") != "VERIFIED"
            or chain["rows"]["31"]["disposition"] != "VERIFIED"):
        raise SystemExit("row-31 oracle pack is not promotable")

    import legoesm.ocean.physics.vertical_mixing as vm
    real_dispatch = vm.implicit_vertical_diffusion_ocean_momentum_dispatch
    captured = []

    def capture_dispatch(field, K, dz, dzh, dt, wet, *call_args, **call_kwargs):
        captured.append({
            "field": np.asarray(field), "K": np.asarray(K),
            "dz": np.asarray(dz), "dzh": np.asarray(dzh),
            "dt": float(dt), "wet": np.asarray(wet),
            "evaluation": call_kwargs.get("evaluation"),
            "extra_diag": np.asarray(call_kwargs.get("extra_diag")),
        })
        return real_dispatch(field, K, dz, dzh, dt, wet,
                             *call_args, **call_kwargs)

    temp_row33 = Path("/tmp/dino_split_explicit_momentum_chain_round34_replay.json")
    old_argv = list(sys.argv)
    vm.implicit_vertical_diffusion_ocean_momentum_dispatch = capture_dispatch
    sys.argv = [
        r33.__file__, "--run", str(args.run),
        "--entry-restart", str(args.entry_restart),
        "--round32", str(args.round32),
        "--manifest-artifact", str(args.manifest_artifact),
        "--output", str(temp_row33),
    ]
    try:
        replay_code = r33.main()
    finally:
        sys.argv = old_argv
        vm.implicit_vertical_diffusion_ocean_momentum_dispatch = real_dispatch
    if replay_code != 0 or len(captured) != 8:
        raise SystemExit(
            f"round-33 replay/dispatch count failed: exit={replay_code}, "
            f"calls={len(captured)}")
    if vm.implicit_vertical_diffusion_ocean_momentum_dispatch is not real_dispatch:
        raise SystemExit("dispatch hook restoration failed")
    # Four S17 arms, with U then V in each; final pair is fed-plus-stress.
    actual_u, actual_v = captured[-2:]
    if not (actual_u["field"].shape[:2] == (199, 53)
            and actual_v["field"].shape[:2] == (200, 52)):
        raise SystemExit(
            f"ambiguous U/V dispatch order: {actual_u['field'].shape}, "
            f"{actual_v['field'].shape}")

    run = args.run.resolve()
    entry = args.entry_restart.resolve()
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
    ht0 = np.sum(e3t0 * tmask, axis=-1)
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

    rdt = 5400.0
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
    Ku = (0.5 * (avm + np.roll(avm, -1, axis=1)))[..., 1:35]
    Kv = (0.5 * (avm + np.roll(avm, -1, axis=0)))[..., 1:35]
    Ku *= (wu[..., 1:] & wu[..., :-1])
    Kv *= (wv[..., 1:] & wv[..., :-1])

    def crop_u(value, levels):
        return np.asarray(value)[:, 1:, :levels]

    def crop_v(value, levels):
        return np.asarray(value)[1:, :, :levels]

    reference = {
        "rhs": (rhs_u, rhs_v, wu, wv,
                "dyn_zdf (momentum implicit vertical solve)"),
        "viscosity": (Ku, Kv, wu[..., 1:] & wu[..., :-1],
                      wv[..., 1:] & wv[..., :-1], None),
        "cell_thickness": (e3uaa[..., :35], e3vaa[..., :35], wu, wv, None),
        "interface_thickness": (e3uwmm[..., 1:35], e3vwmm[..., 1:35],
                                wu[..., 1:] & wu[..., :-1],
                                wv[..., 1:] & wv[..., :-1], None),
        "drag_diagonal": (diag_u, diag_v, wu, wv, None),
    }
    captured_values = {
        "rhs": (crop_u(actual_u["field"], 35), crop_v(actual_v["field"], 35)),
        "viscosity": (crop_u(actual_u["K"], 34), crop_v(actual_v["K"], 34)),
        "cell_thickness": (crop_u(actual_u["dz"], 35), crop_v(actual_v["dz"], 35)),
        "interface_thickness": (crop_u(actual_u["dzh"], 34), crop_v(actual_v["dzh"], 34)),
        "drag_diagonal": (crop_u(actual_u["extra_diag"], 35),
                          crop_v(actual_v["extra_diag"], 35)),
    }
    rows = {}
    first_failure = None
    for index, name in enumerate(reference, start=1):
        eu, ev, mu, mv, gate = reference[name]
        au, av = captured_values[name]
        rows[name] = {
            "u": _metric(au, eu, mu, gate),
            "v": _metric(av, ev, mv, gate),
        }
        if first_failure is None and any(
                row["gate_status"] != "AT BAR" for row in rows[name].values()):
            first_failure = name

    replay = json.loads(temp_row33.read_text())
    scalar_controls = {
        "selector_u": actual_u["evaluation"] == "nemo_literal",
        "selector_v": actual_v["evaluation"] == "nemo_literal",
        "dt_u_exact": actual_u["dt"] == rdt,
        "dt_v_exact": actual_v["dt"] == rdt,
        "wrong_dt_fails": 2700.0 != actual_u["dt"],
        "dispatch_restored": (
            vm.implicit_vertical_diffusion_ocean_momentum_dispatch is real_dispatch),
        "round33_replay_model_diff_zero": (
            replay.get("disposition") == row33.get("disposition")
            and replay.get("rows") == row33.get("rows")),
    }
    plant = np.array(rhs_u, copy=True)
    plant[tuple(np.argwhere(wu)[0])] += 2e-12 * np.sqrt(np.mean(rhs_u[wu] ** 2))
    scalar_controls["two_bar_rhs_plant_fires"] = (
        _metric(plant, rhs_u, wu,
                "dyn_zdf (momentum implicit vertical solve)")["gate_status"]
        != "AT BAR")
    scalar_controls["one_cell_roll_fires"] = (
        _metric(np.roll(rhs_u, 1, axis=1), rhs_u, wu,
                "dyn_zdf (momentum implicit vertical solve)")["gate_status"]
        != "AT BAR")
    nan_plant = np.array(rhs_u, copy=True)
    nan_plant[tuple(np.argwhere(wu)[0])] = np.nan
    scalar_controls["wet_nan_detected"] = not np.isfinite(nan_plant[wu]).all()
    valid = all(scalar_controls.values())
    disposition = ("INVALID" if not valid else
                   f"LOCALIZED_TO_{first_failure.upper()}" if first_failure else
                   "ALL_DISPATCH_OPERANDS_AT_BAR")

    paths = {
        "round32": args.round32, "round33": args.round33,
        "chain_end": args.chain_end,
        "manifest_artifact": args.manifest_artifact,
        "entry_restart": entry, "mesh": run / "mesh_mask.nc",
        "scorer": Path(__file__).resolve(),
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round34.md",
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round34-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dispatch_call_count": len(captured),
        "first_failing_operand": first_failure,
        "rows": rows,
        "controls": scalar_controls,
        "bindings": {name: _sha(path.resolve()) for name, path in paths.items()},
        "disposition": disposition,
        "ordered_next": 4 if first_failure else 5,
    }
    if _tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_failing_operand={first_failure}")
    for name, components in rows.items():
        print(name, {k: v["gate_status"] for k, v in components.items()})
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
