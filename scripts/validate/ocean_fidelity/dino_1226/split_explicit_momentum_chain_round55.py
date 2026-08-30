#!/usr/bin/env python3
"""Replay the current production tracer-entry ladder against held row 8."""
from __future__ import annotations

import argparse
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
import jax.numpy as jnp
import numpy as np

import kamm_twin_90d as twin
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gm_module
import zdf_chain_sweep as sweep
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import upwind_to_u_points
from legoesm.ocean.vertical import compute_layer_thickness


ROUND54_SHA = "a1b76177330b83d7bb21c7f35c9e606d10d36a4b98d46b3b570f53588188eaa3"
# 2026-08-30 round-58 population correction: supersedes the invalid
# e3u!=0 round-56 admission hash; same production arm on exact NEMO umask.
ROUND56_SHA = "c114565363360439e155deec96882828e553887ba49b1bfbfd940c50c6998e29"
ROUND59_SHA = "85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677"
ROUND60_SHA = "b3ef5c0534ff1348dbdb581686aa602cc1d9eca9ef61336ca0b4130217e54e2d"
ROUND61_SHA = "b8f7a376a0a13fd384cb4195cf27db8ff6f82c71e576bb8d449451903d3bd07c"
ROUND62_SHA = "290caa5bb3c3187d5ea13fe62276f299bd47953b556615dfab3f4443d1597a86"
ROUND63_SHA = "3a1a25ca328761b1bcbeb87953751a3a15b1ac00852b2ff62fd4223d107d24e0"
ROUND64_SHA = "8858d60a51b07181e290fead087e4bab69c0d15e271bbdecb7d458ba12d4e4c7"
RAW_ARTIFACT_SHA = "ec4885a1e7c059872f1b575c5f93f00c0e6538b65613eede71f082fac24885ea"
HELD_SHA = {
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "fct_entry_dump_un.bin": "89d5c37cb77e03ee0f6ab50ba0814461a7459a13c31fb221329f4cf9b5ec829d",
    "fct_entry_dump_e2u.bin": "d55f36aa52f6686c4d6a152c81ee7b8360ca415342810b68dc232baefb631052",
    "fct_entry_dump_e3u.bin": "d071b046cf5e841d2475b753b3c834ad02e69128e73a59c7dd15576a47d7b4f8",
    "fct_entry_dump_e2e3u.bin": "75fdac4eeaa0791f304b604fad626109cea73d98e3f79dbb455d12881b386a59",
    "fct_entry_dump_pu_euler.bin": "b0bcfbc94bbb0a203e475774b83e93fce8884f949f3364d79716704abbd68bde",
    "fct_entry_dump_pu_bolus.bin": "69dfe222dbbf460f9ae86f657b2b9ca114e4203ac8beb34d00d76f0e8f3f58d5",
    "fct_entry_dump_pu_total.bin": "4006ee7f7c8c140f1e4b3dfcdec5a4a3573b322cd86edfff245a206e05c5d270",
    "fct_dump_zwx_up.bin": "ecb7caf9cb610c274cc206877fc484ffe6a278cf558db73d6cf10aac184b715d",
    "spg_dump_zu_frc.bin": "13138faa46149968c009f0d6c20956b8d2457a904667686a1f6cb1a93d7bcca6",
    "spg_dump_zv_frc.bin": "81d01381ad5164b1a0cc3fb5f22590f471b0e24bb9b68084290e0cb9e2ef4c0e",
    "spg_dump_un_adv_final.bin": "4d8e7a6445ba465c8229954b443c467862381409805163f2045f5854017917ab",
    "spg_dump_vn_adv_final.bin": "ed1e28aa27e1c49d237e07a07707220251b3a1845a599d21b06df18b8425d088",
}
BOLUS_HELD_SHA = {
    "eiv_dump_u.bin": "a0e7b0f0a84cb87bd5e059c7161d261016f2528ba127df666a37966395c0fc00",
    "eiv_dump_psi_uw.bin": "feb5ba7a1e4882cb43c71db049fb78ecd2e2301c9573777bf5a0e1738518101b",
    "eiv_dump_wslpi.bin": "e3073a8501e8046732e301d58781dcaf35a6a84d9071309353b3de0bad08fb64",
    "eiv_dump_aeiu.bin": "8146cf02d33e8013bf240623948b42bd8b2cda8ee15c11847393ad298a5a60e8",
}
FOCUS = [(11, 1), (12, 1), (13, 1), (13, 23)]
POINTWISE_BAR = 1.0e-15
ACCUMULATION_BAR = 1.0e-12


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _load(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    expected = 35 * 203 * 56
    if raw.size != expected:
        raise SystemExit(
            f"{path}: full-halo writer contract is (35,203,56), got {raw.size}")
    return np.moveaxis(raw.reshape(35, 203, 56)[:, 2:-2, 2:-2], 0, -1)


def _controls(oracle: np.ndarray, wet: np.ndarray, bar: float) -> dict:
    identity = sweep.metrics(oracle, oracle, wet, FOCUS, bar)
    point = np.array(oracle, copy=True)
    idx = tuple(int(x) for x in np.argwhere(wet)[0])
    scale = identity["reference_rms"]
    point[idx] += max(4.0 * bar * scale, 4.0 * abs(float(np.spacing(point[idx]))))
    return {
        "identity": bool(identity["pass"]),
        "wet_point": not sweep.metrics(point, oracle, wet, FOCUS, bar)["pass"],
        "meridional_roll": not sweep.metrics(
            np.roll(oracle, 1, axis=0), oracle, wet, FOCUS, bar)["pass"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-dir", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round54", type=Path, required=True)
    parser.add_argument("--round56", type=Path)
    parser.add_argument("--round59", type=Path)
    parser.add_argument("--round60", type=Path)
    parser.add_argument("--round61", type=Path)
    parser.add_argument("--round62", type=Path)
    parser.add_argument("--round63", type=Path)
    parser.add_argument("--round64", type=Path)
    parser.add_argument("--hold-slow-forcing", action="store_true")
    parser.add_argument("--oracle-transport", action="store_true")
    parser.add_argument("--capture-cycle", action="store_true")
    parser.add_argument("--direct-cycle-entry", action="store_true")
    parser.add_argument("--live-thickness-entry", action="store_true")
    parser.add_argument("--capture-bolus-operands", action="store_true")
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round54.resolve()) != ROUND54_SHA:
        raise SystemExit("official round-54 receipt changed")
    prior = json.loads(args.round54.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "MOMENTUM_TAIL_AT_BAR_UPSTREAM_TRANSPORT_EXACT"):
        raise SystemExit("round 54 does not release the tracer tail")
    prior56 = None
    if args.hold_slow_forcing:
        if args.round56 is None or _sha(args.round56.resolve()) != ROUND56_SHA:
            raise SystemExit("official round-56 red receipt required")
        prior56 = json.loads(args.round56.read_text())
        if (prior56.get("session_id") != session or prior56.get("disposition")
                != "TRACER_ENTRY_DIVERGED_8.3"):
            raise SystemExit("round 56 does not admit the forcing substitution")
    elif args.round56 is not None:
        raise SystemExit("--round56 is only valid with --hold-slow-forcing")
    prior59 = prior60 = None
    if args.oracle_transport:
        if not args.hold_slow_forcing:
            raise SystemExit("--oracle-transport requires --hold-slow-forcing")
        if (args.round59 is None
                or _sha(args.round59.resolve()) != ROUND59_SHA):
            raise SystemExit("official round-59 held receipt required")
        if (args.round60 is None
                or _sha(args.round60.resolve()) != ROUND60_SHA):
            raise SystemExit("official round-60 receipt required")
        prior59 = json.loads(args.round59.read_text())
        prior60 = json.loads(args.round60.read_text())
        if (prior59.get("disposition") != "TRACER_ENTRY_DIVERGED_8.3"
                or prior59["rows"][0]["metrics"]["n_diverged_columns"] != 104):
            raise SystemExit("round 59 does not admit the conditional replay")
        if (prior60.get("disposition")
                != "ROW8_3_LITERAL_ACCUMULATOR_AT_BAR_UPSTREAM_OPERANDS_OPEN"):
            raise SystemExit("round 60 does not release the oracle transport arm")
    elif args.round59 is not None or args.round60 is not None:
        raise SystemExit("--round59/--round60 require --oracle-transport")
    prior61 = None
    if args.capture_cycle:
        if not args.oracle_transport:
            raise SystemExit("--capture-cycle requires --oracle-transport")
        if (args.round61 is None
                or _sha(args.round61.resolve()) != ROUND61_SHA):
            raise SystemExit("official round-61 null receipt required")
        prior61 = json.loads(args.round61.read_text())
        if (prior61.get("disposition") != "TRACER_ENTRY_DIVERGED_8.3"
                or prior61["rows"][0]["metrics"]["n_diverged_columns"] != 104):
            raise SystemExit("round 61 does not admit the Kmm-cycle capture")
    elif args.round61 is not None:
        raise SystemExit("--round61 requires --capture-cycle")
    prior62 = None
    if args.direct_cycle_entry:
        if not args.capture_cycle:
            raise SystemExit("--direct-cycle-entry requires --capture-cycle")
        if (args.round62 is None
                or _sha(args.round62.resolve()) != ROUND62_SHA):
            raise SystemExit("official round-62 capture receipt required")
        prior62 = json.loads(args.round62.read_text())
        capture62 = prior62.get("cycle_capture_metrics", {})
        if (prior62.get("disposition") != "ROW8_3_CYCLE_CAPTURE_UNRESOLVED"
                or not capture62.get("production_Hu_avg", {}).get("pass")
                or not capture62.get("consumed_Hu_avg", {}).get("pass")
                or not capture62.get("cycle_corrected_u", {}).get("pass")):
            raise SystemExit("round 62 does not release the direct-cycle score")
    elif args.round62 is not None:
        raise SystemExit("--round62 requires --direct-cycle-entry")
    prior63 = None
    if args.live_thickness_entry:
        if not args.direct_cycle_entry:
            raise SystemExit(
                "--live-thickness-entry requires --direct-cycle-entry")
        if (args.round63 is None
                or _sha(args.round63.resolve()) != ROUND63_SHA):
            raise SystemExit("official round-63 thickness receipt required")
        prior63 = json.loads(args.round63.read_text())
        if (prior63.get("disposition") != "TRACER_ENTRY_DIVERGED_8.5"
                or prior63["rows"][0]["status"] != "AT_BAR"
                or prior63["rows"][1]["status"] != "AT_BAR"
                or prior63["rows"][2]["metrics"]["n_diverged_columns"]
                != 9758):
            raise SystemExit("round 63 does not release the live-thickness score")
    elif args.round63 is not None:
        raise SystemExit("--round63 requires --live-thickness-entry")
    prior64 = None
    if args.capture_bolus_operands:
        if not args.live_thickness_entry:
            raise SystemExit(
                "--capture-bolus-operands requires --live-thickness-entry")
        if (args.round64 is None
                or _sha(args.round64.resolve()) != ROUND64_SHA):
            raise SystemExit("official round-64 row-8.8 receipt required")
        prior64 = json.loads(args.round64.read_text())
        if (prior64.get("disposition") != "TRACER_ENTRY_DIVERGED_8.8"
                or any(prior64["rows"][i]["status"] != "AT_BAR"
                       for i in range(5))
                or prior64["rows"][5]["metrics"]["n_diverged_columns"]
                != 8938):
            raise SystemExit("round 64 does not release the bolus operand peel")
    elif args.round64 is not None:
        raise SystemExit("--round64 requires --capture-bolus-operands")
    if _sha(args.raw_artifact.resolve()) != RAW_ARTIFACT_SHA:
        raise SystemExit("admitted held row-8 artifact changed")
    held = args.held_dir.resolve()
    for name, expected in HELD_SHA.items():
        if not (held / name).is_file() or _sha(held / name) != expected:
            raise SystemExit(f"held row-8 input changed: {name}")
    if args.capture_bolus_operands:
        for name, expected in BOLUS_HELD_SHA.items():
            if not (held / name).is_file() or _sha(held / name) != expected:
                raise SystemExit(f"held GM operand changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(held),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    state = model._seed_tke_preclosure_carry(state)
    captured = []
    original = model_module.add_bolus_to_advecting_flux
    original_solver = model_module.barotropic_substeps_latlon_cgrid
    original_cycle = model_module.nemo_qco_kmm_velocity_cycle
    original_thickness = model_module.nemo_qco_live_face_thicknesses
    solver_calls = 0
    transport_substitutions = 0
    solver_transport_captures = []
    cycle_captures = []
    thickness_captures = []
    bolus_captures = []
    held_u_native = np.fromfile(held / "spg_dump_zu_frc.bin", dtype="<f8")
    held_v_native = np.fromfile(held / "spg_dump_zv_frc.bin", dtype="<f8")
    if held_u_native.size != 199 * 52 or held_v_native.size != 199 * 52:
        raise SystemExit("held slow forcing must be cited-interior (199,52)")
    held_u = np.concatenate(
        [held_u_native.reshape(199, 52)[:, -1:], held_u_native.reshape(199, 52)],
        axis=1)
    held_v = np.concatenate(
        [np.zeros((1, 52)), held_v_native.reshape(199, 52)], axis=0)
    oracle_u_native = np.fromfile(
        held / "spg_dump_un_adv_final.bin", dtype="<f8")
    oracle_v_native = np.fromfile(
        held / "spg_dump_vn_adv_final.bin", dtype="<f8")
    if oracle_u_native.size != 203 * 56 or oracle_v_native.size != 203 * 56:
        raise SystemExit("final transport writers must be full-halo (203,56)")
    oracle_u_native = oracle_u_native.reshape(203, 56)[2:-2, 2:-2]
    oracle_v_native = oracle_v_native.reshape(203, 56)[2:-2, 2:-2]
    oracle_u = np.concatenate([
        oracle_u_native[:, -1:], oracle_u_native], axis=1)
    oracle_v = np.concatenate([np.zeros((1, 52)), oracle_v_native], axis=0)

    def observe(bolus, mass_flux_u, mass_flux_v, u_mask, v_mask, grid, z_coord):
        result = original(
            bolus, mass_flux_u, mass_flux_v, u_mask, v_mask, grid, z_coord)
        captured.append((mass_flux_u, result[0], u_mask))
        return result

    def held_solver(state_arg, dt_s, n_substeps, grid, z_coord, config, **kwargs):
        nonlocal solver_calls, transport_substitutions
        kwargs = dict(kwargs)
        is_target = kwargs.get("eta_init") is not None
        if is_target:
            solver_calls += 1
            kwargs["F_slow_u"] = jnp.asarray(
                held_u, dtype=kwargs["F_slow_u"].dtype)
            kwargs["F_slow_v"] = jnp.asarray(
                held_v, dtype=kwargs["F_slow_v"].dtype)
        result = original_solver(
            state_arg, dt_s, n_substeps, grid, z_coord, config, **kwargs)
        if is_target and args.capture_cycle:
            solver_transport_captures.append(tuple(
                np.asarray(value) for value in result[1]))
        if is_target and args.oracle_transport:
            transport_substitutions += 1
            solved_state, _ = result
            result = (solved_state, (
                jnp.asarray(oracle_u, dtype=state_arg.eta.data.dtype),
                jnp.asarray(oracle_v, dtype=state_arg.eta.data.dtype)))
        return result

    def observed_cycle(*cycle_args, **cycle_kwargs):
        result = original_cycle(*cycle_args, **cycle_kwargs)
        cycle_captures.append({
            "un_adv": np.asarray(cycle_args[3]),
            "vn_adv": np.asarray(cycle_args[4]),
            "corrected_u": np.asarray(result[0]),
            "corrected_v": np.asarray(result[1]),
        })
        return result

    def observed_thickness(*thickness_args, **thickness_kwargs):
        result = original_thickness(*thickness_args, **thickness_kwargs)
        thickness_captures.append(tuple(np.asarray(value) for value in result))
        return result

    original_bolus = gm_module.nemo_eiv_bolus_transport

    def observed_bolus(*bolus_args, **bolus_kwargs):
        result = original_bolus(*bolus_args, **bolus_kwargs)
        bolus_captures.append((
            tuple(np.asarray(value) if hasattr(value, "shape") else value
                  for value in bolus_args),
            dict(bolus_kwargs),
            tuple(np.asarray(value) for value in result),
        ))
        return result

    model_module.add_bolus_to_advecting_flux = observe
    if args.hold_slow_forcing:
        model_module.barotropic_substeps_latlon_cgrid = held_solver
    if args.capture_cycle:
        model_module.nemo_qco_kmm_velocity_cycle = observed_cycle
    if args.live_thickness_entry:
        model_module.nemo_qco_live_face_thicknesses = observed_thickness
    if args.capture_bolus_operands:
        gm_module.nemo_eiv_bolus_transport = observed_bolus
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.add_bolus_to_advecting_flux = original
        model_module.barotropic_substeps_latlon_cgrid = original_solver
        model_module.nemo_qco_kmm_velocity_cycle = original_cycle
        model_module.nemo_qco_live_face_thicknesses = original_thickness
        gm_module.nemo_eiv_bolus_transport = original_bolus
    restored = (model_module.add_bolus_to_advecting_flux is original
                and model_module.barotropic_substeps_latlon_cgrid is original_solver
                and model_module.nemo_qco_kmm_velocity_cycle is original_cycle
                and model_module.nemo_qco_live_face_thicknesses
                is original_thickness
                and gm_module.nemo_eiv_bolus_transport is original_bolus)
    if len(captured) != 1:
        raise SystemExit(f"expected one single-pass tracer handoff, got {len(captured)}")

    base_u, total_u, u_mask = (np.asarray(x) for x in captured[0])
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=cfg.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k))
    u_corrected = np.divide(
        base_u, h_u, out=np.zeros_like(base_u),
        where=(np.asarray(u_mask, dtype=bool) & (h_u != 0.0)))
    e2u = np.asarray(model.grid.dy_u)[:, :, None]
    proxy_un = u_corrected[:, 1:, :35]
    values = {
        "un": proxy_un,
        "e2u": np.broadcast_to(e2u[:, 1:, :], u_corrected[:, 1:, :35].shape),
        "e3u": h_u[:, 1:, :35],
        "e2e3u": (e2u * h_u)[:, 1:, :35],
        "pu_euler": (e2u * base_u)[:, 1:, :35],
        "pu_bolus": (e2u * (total_u - base_u))[:, 1:, :35],
        "pu_total": (e2u * total_u)[:, 1:, :35],
    }
    donor = np.asarray(upwind_to_u_points(state.T_before.data, total_u))
    values["upstream"] = (e2u * total_u * donor)[:, 1:, :35]
    oracle = {
        "un": _load(held / "fct_entry_dump_un.bin"),
        "e2u": _load(held / "fct_entry_dump_e2u.bin"),
        "e3u": _load(held / "fct_entry_dump_e3u.bin"),
        "e2e3u": _load(held / "fct_entry_dump_e2e3u.bin"),
        "pu_euler": _load(held / "fct_entry_dump_pu_euler.bin"),
        "pu_bolus": _load(held / "fct_entry_dump_pu_bolus.bin"),
        "pu_total": _load(held / "fct_entry_dump_pu_total.bin"),
        "upstream": _load(held / "fct_dump_zwx_up.bin"),
    }
    # Registry population is NEMO's real 3-D umask, not nonzero geometric
    # thickness.  The latter admits 590 dry surface faces because e3u remains
    # defined under land and produced the invalid 10,348-column round-55--57
    # population.  The captured production mask is the bridge of that umask.
    wet = np.asarray(u_mask, dtype=bool)[:, 1:, :35]
    if int(np.any(wet, axis=-1).sum()) != 9758 or int(wet.sum()) != 336338:
        raise SystemExit("registered U population changed from 9758/336338")
    capture_metrics = {}
    if args.capture_cycle:
        if len(solver_transport_captures) != 1 or not cycle_captures:
            raise SystemExit(
                "expected one solver transport and at least one Kmm-cycle capture")
        wet2 = np.any(wet, axis=-1)
        wet2_3d = wet2[..., None]
        oracle_transport_3d = oracle_u_native[..., None]
        production_hu = solver_transport_captures[0][0][:, 1:]
        consumed_hu = cycle_captures[0]["un_adv"][:, 1:]
        capture_metrics = {
            "production_Hu_avg": sweep.metrics(
                production_hu[..., None], oracle_transport_3d,
                wet2_3d, FOCUS, POINTWISE_BAR),
            "consumed_Hu_avg": sweep.metrics(
                consumed_hu[..., None], oracle_transport_3d,
                wet2_3d, FOCUS, POINTWISE_BAR),
            "cycle_corrected_u": sweep.metrics(
                cycle_captures[0]["corrected_u"][:, 1:, :35],
                oracle["un"], wet, FOCUS, POINTWISE_BAR),
        }
        if args.direct_cycle_entry:
            values["un"] = cycle_captures[0]["corrected_u"][:, 1:, :35]
    if args.live_thickness_entry:
        if len(thickness_captures) != 1:
            raise SystemExit(
                "expected exactly one tracer live-thickness capture")
        live_u_raw = thickness_captures[0][0][..., :35]
        values["e3u"] = live_u_raw
        values["e2e3u"] = e2u[:, 1:, :] * live_u_raw
    bolus_operand_metrics = {}
    if args.capture_bolus_operands:
        if not bolus_captures:
            raise SystemExit("expected at least one GM bolus producer capture")
        bargs, bkwargs, bresult = bolus_captures[0]
        if not all(np.array_equal(capture[2][0], bresult[0])
                   for capture in bolus_captures[1:]):
            raise SystemExit("multiple GM bolus calls produced different U transports")
        kappa, slope_kp1, _, e2u_native, _, u_mask_native, _, act, act_below = bargs[:9]
        out_shape = tuple(bargs[9])
        kappa = np.asarray(kappa)
        if kappa.ndim == 2:
            kappa = np.broadcast_to(kappa[..., None], out_shape)
        elif kappa.ndim != 3:
            kappa = np.broadcast_to(kappa, out_shape)
        if not bkwargs.get("kappa_face_average", False):
            raise SystemExit("round 65 requires production face-averaged kappa")
        aeiu = 0.5 * (kappa + np.roll(kappa, -1, axis=1))
        slope_sum = slope_kp1 + np.roll(slope_kp1, -1, axis=1)
        aeiu_sum = aeiu + np.roll(aeiu, -1, axis=2)
        wumask = (u_mask_native[:, 1:, None] * act
                  * np.roll(act, -1, axis=1) * act_below
                  * np.roll(act_below, -1, axis=1))
        current_psi = -(e2u_native[..., None] * (0.5 * slope_sum)
                        * (0.5 * aeiu_sum) * wumask)
        literal_psi = (((-0.25 * e2u_native[..., None]) * slope_sum)
                       * aeiu_sum) * wumask
        oracle_psi = _load(held / "eiv_dump_psi_uw.bin")
        oracle_u_eiv = -_load(held / "eiv_dump_u.bin")
        oracle_wslpi = _load(held / "eiv_dump_wslpi.bin")
        oracle_aeiu = _load(held / "eiv_dump_aeiu.bin")
        zero_level = np.zeros_like(oracle_wslpi[..., :1])
        oracle_slope_kp1 = np.concatenate(
            [oracle_wslpi[..., 1:], zero_level], axis=2)
        oracle_aeiu_kp1 = np.concatenate(
            [oracle_aeiu[..., 1:], np.zeros_like(oracle_aeiu[..., :1])], axis=2)
        oracle_slope_sum = oracle_slope_kp1 + np.roll(
            oracle_slope_kp1, -1, axis=1)
        oracle_aeiu_sum = oracle_aeiu + oracle_aeiu_kp1
        own_oracle_slope = (((-0.25 * e2u_native[..., None])
                             * oracle_slope_sum) * aeiu_sum) * wumask
        own_oracle_aeiu = (((-0.25 * e2u_native[..., None])
                            * slope_sum) * oracle_aeiu_sum) * wumask
        oracle_both = (((-0.25 * e2u_native[..., None])
                        * oracle_slope_sum) * oracle_aeiu_sum) * wumask
        wet_psi = np.asarray(wumask[..., :35], dtype=bool)
        if int(np.any(wet_psi, axis=-1).sum()) != 9758:
            raise SystemExit("GM psi registered U-column population changed")
        def bm(value, oracle_value=oracle_psi):
            return sweep.metrics(
                np.asarray(value)[..., :35], oracle_value, wet_psi,
                FOCUS, ACCUMULATION_BAR)
        bolus_operand_metrics = {
            "captured_u_increment": sweep.metrics(
                bresult[0][..., :35], oracle_u_eiv, wet_psi,
                FOCUS, ACCUMULATION_BAR),
            "aeiu_face": bm(aeiu, oracle_aeiu),
            "wslpi_kp1_face_sum": bm(slope_sum, oracle_slope_sum),
            "psi_current_normalized": bm(current_psi),
            "psi_literal_same_operands": bm(literal_psi),
            "factorial_own_slope_own_aeiu": bm(literal_psi),
            "factorial_oracle_slope_own_aeiu": bm(own_oracle_slope),
            "factorial_own_slope_oracle_aeiu": bm(own_oracle_aeiu),
            "factorial_oracle_slope_oracle_aeiu": bm(oracle_both),
        }
    specs = (
        ("8.3", "uu(Kmm) / zptu", "traadv.F90:301-304", "un", POINTWISE_BAR),
        ("8.4", "e2u", "traadv.F90:329", "e2u", POINTWISE_BAR),
        ("8.5", "e3u(Kmm)", "traadv.F90:329", "e3u", POINTWISE_BAR),
        ("8.6", "e2u*e3u(Kmm)", "traadv.F90:329", "e2e3u", POINTWISE_BAR),
        ("8.7", "Eulerian pU", "traadv.F90:328-331", "pu_euler", ACCUMULATION_BAR),
        ("8.8", "GM U increment", "ldftra.F90:894-902", "pu_bolus", ACCUMULATION_BAR),
        ("8.9", "total pU entering FCT", "traadv.F90:343-362", "pu_total", ACCUMULATION_BAR),
        ("8.10", "temperature U upstream flux", "traadv_fct.F90:528-530", "upstream", ACCUMULATION_BAR),
    )
    rows = []
    blocked = False
    first = None
    controls = {
        "hook_restored": restored,
        "solver_substitution_count": (
            solver_calls == 1 if args.hold_slow_forcing else solver_calls == 0),
        "transport_substitution_count": (
            transport_substitutions == 1 if args.oracle_transport
            else transport_substitutions == 0),
        "round59_production_debt_admitted": (
            prior59 is None
            or prior59["rows"][0]["metrics"]["n_diverged_columns"] == 104),
        "round60_literal_local_exact": (
            prior60 is None
            or prior60["metrics"]["raw_metric"]["pass"]),
        "round61_null_admitted": (
            prior61 is None
            or prior61["rows"][0]["metrics"]["n_diverged_columns"] == 104),
        "round62_direct_cycle_exact": (
            prior62 is None
            or prior62["cycle_capture_metrics"]["cycle_corrected_u"]["pass"]),
        "round63_thickness_debt_admitted": (
            prior63 is None
            or prior63["rows"][2]["metrics"]["n_diverged_columns"] == 9758),
        "cycle_capture_count": (
            len(cycle_captures) >= 1 if args.capture_cycle
            else len(cycle_captures) == 0),
        "solver_transport_capture_count": (
            len(solver_transport_captures) == 1 if args.capture_cycle
            else len(solver_transport_captures) == 0),
        "consumed_transport_at_bar": (
            capture_metrics["consumed_Hu_avg"]["pass"]
            if args.capture_cycle else True),
        "mass_flux_division_proxy_red_104": (
            prior63["controls"]["mass_flux_division_proxy_red_104"]
            if args.live_thickness_entry else
            sweep.metrics(proxy_un, oracle["un"], wet, FOCUS,
                          POINTWISE_BAR)["n_diverged_columns"] == 104
            if args.direct_cycle_entry else True),
        "live_thickness_capture_count": (
            len(thickness_captures) == 1 if args.live_thickness_entry
            else len(thickness_captures) == 0),
        "generic_min_thickness_plant_red_9758": (
            sweep.metrics(h_u[:, 1:, :35], oracle["e3u"], wet, FOCUS,
                          POINTWISE_BAR)["n_diverged_columns"] == 9758
            if args.live_thickness_entry else True),
        "round64_row8_8_debt_admitted": (
            prior64 is None
            or prior64["rows"][5]["metrics"]["n_diverged_columns"] == 8938),
        "bolus_capture_count": (
            len(bolus_captures) >= 1 if args.capture_bolus_operands
            else len(bolus_captures) == 0),
        "round56_unheld_red": (
            prior56 is None
            or prior56["rows"][0]["status"] == "DIVERGED"),
    }
    for subrow, name, source, key, bar in specs:
        row = {"subrow": subrow, "name": name, "nemo_source": source, "bar": bar}
        if blocked:
            row["status"] = "ORDERED_BLOCKED"
        else:
            metric = sweep.metrics(values[key], oracle[key], wet, FOCUS, bar)
            row.update(status="AT_BAR" if metric["pass"] else "DIVERGED",
                       metrics=metric)
            controls.update({f"{subrow}_{k}": v for k, v in
                             _controls(oracle[key], wet, bar).items()})
            if not metric["pass"]:
                first = subrow
                blocked = True
        rows.append(row)
    controls["gm_sign_plant"] = not sweep.metrics(
        -values["pu_bolus"], oracle["pu_bolus"], wet, FOCUS,
        ACCUMULATION_BAR)["pass"]
    if args.capture_bolus_operands:
        controls.update({
            "bolus_current_normalized_red":
                not bolus_operand_metrics["psi_current_normalized"]["pass"],
            "bolus_sign_plant": not sweep.metrics(
                -literal_psi[..., :35], oracle_psi, wet_psi, FOCUS,
                ACCUMULATION_BAR)["pass"],
            "bolus_meridional_roll_plant": not sweep.metrics(
                np.roll(literal_psi[..., :35], 1, axis=0), oracle_psi,
                wet_psi, FOCUS, ACCUMULATION_BAR)["pass"],
        })
    controls["all_scored_finite"] = all(
        row.get("status") == "ORDERED_BLOCKED"
        or row["metrics"]["n_nonfinite_wet_elements"] == 0 for row in rows)
    controls["registered_population_exact"] = (
        int(np.any(wet, axis=-1).sum()) == 9758 and int(wet.sum()) == 336338)
    valid = all(controls.values())
    disposition = (("TRACER_ENTRY_ROW8_AT_BAR_LIVE_QCO"
                    if args.live_thickness_entry else
                    "TRACER_ENTRY_ROW8_AT_BAR_DIRECT_KMM"
                    if args.direct_cycle_entry else
                    "TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT"
                    if args.oracle_transport else
                    "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT"
                    if args.hold_slow_forcing else "TRACER_ENTRY_ROW8_AT_BAR")
                   if valid and first is None
                   else "INVALID" if not valid else f"TRACER_ENTRY_DIVERGED_{first}")
    if args.capture_cycle and valid and not args.direct_cycle_entry:
        production_at_bar = capture_metrics["production_Hu_avg"]["pass"]
        consumed_at_bar = capture_metrics["consumed_Hu_avg"]["pass"]
        cycle_at_bar = capture_metrics["cycle_corrected_u"]["pass"]
        if production_at_bar and consumed_at_bar and not cycle_at_bar:
            disposition = "ROW8_3_ACCUMULATOR_EXONERATED_KMM_COMPOSITION_OPEN"
        elif not consumed_at_bar:
            disposition = "INVALID_ORACLE_TRANSPORT_LAYOUT"
        elif not production_at_bar and not cycle_at_bar:
            disposition = "INVALID_TRANSPORT_SUBSTITUTION_PLUMBING"
        else:
            disposition = "ROW8_3_CYCLE_CAPTURE_UNRESOLVED"
    if args.capture_bolus_operands and valid:
        aeiu_at_bar = bolus_operand_metrics["aeiu_face"]["pass"]
        slope_at_bar = bolus_operand_metrics["wslpi_kp1_face_sum"]["pass"]
        literal_at_bar = bolus_operand_metrics["psi_literal_same_operands"]["pass"]
        current_red = not bolus_operand_metrics["psi_current_normalized"]["pass"]
        if aeiu_at_bar and slope_at_bar and literal_at_bar and current_red:
            disposition = "ROW8_8_LOCALIZED_TO_PSI_ASSOCIATION"
        elif not slope_at_bar:
            disposition = "ROW8_8_LOCALIZED_TO_WSPLPI_OPERAND"
        elif not aeiu_at_bar:
            disposition = "ROW8_8_LOCALIZED_TO_AEIU_OPERAND"
        else:
            disposition = "ROW8_8_PSI_COMPOSITION_OPEN"
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": ("dino-split-explicit-momentum-chain-round64-v1"
                   if args.live_thickness_entry else
                   "dino-split-explicit-momentum-chain-round63-v1"
                   if args.direct_cycle_entry else
                   "dino-split-explicit-momentum-chain-round62-v1"
                   if args.capture_cycle else
                   "dino-split-explicit-momentum-chain-round61-v1"
                   if args.oracle_transport else
                   "dino-split-explicit-momentum-chain-round59-v1"),
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "disposition": disposition,
        "first_diverged_subrow": first,
        "rows": rows,
        "controls": controls,
        **({"cycle_capture_metrics": capture_metrics}
           if args.capture_cycle else {}),
        **({"bolus_operand_metrics": bolus_operand_metrics}
           if args.capture_bolus_operands else {}),
        "focus_ji": [list(x) for x in FOCUS],
        "bindings": {
            "round54": _sha(args.round54.resolve()),
            **({"round56": _sha(args.round56.resolve())}
               if args.hold_slow_forcing else {}),
            **({"round59": _sha(args.round59.resolve()),
                "round60": _sha(args.round60.resolve())}
               if args.oracle_transport else {}),
            **({"round61": _sha(args.round61.resolve())}
               if args.capture_cycle else {}),
            **({"round62": _sha(args.round62.resolve())}
               if args.direct_cycle_entry else {}),
            **({"round63": _sha(args.round63.resolve())}
               if args.live_thickness_entry else {}),
            **({"round64": _sha(args.round64.resolve()),
                **{name: _sha(held / name) for name in BOLUS_HELD_SHA}}
               if args.capture_bolus_operands else {}),
            "held_raw_artifact": _sha(args.raw_artifact.resolve()),
            **{name: _sha(held / name) for name in HELD_SHA},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity" /
                ("PREREG_split_explicit_momentum_chain_round65.md"
                 if args.capture_bolus_operands else
                 "PREREG_split_explicit_momentum_chain_round64.md"
                 if args.live_thickness_entry else
                 "PREREG_split_explicit_momentum_chain_round63.md"
                 if args.direct_cycle_entry else
                 "PREREG_split_explicit_momentum_chain_round62.md"
                 if args.capture_cycle else
                 "PREREG_split_explicit_momentum_chain_round61.md"
                 if args.oracle_transport else
                 "PREREG_split_explicit_momentum_chain_round59.md")),
            "production_model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "production_barotropic": _sha(root / "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_traadv": _sha(nemo / "src/OCE/TRA/traadv.F90"),
            "nemo_traadv_fct": _sha(nemo / "src/OCE/TRA/traadv_fct.F90"),
            "nemo_ldftra": _sha(nemo / "cfgs/DINO/MY_SRC/ldftra.F90"),
        },
        "arm": ("bolus_operand_capture" if args.capture_bolus_operands else
                "live_thickness_entry" if args.live_thickness_entry else
                "direct_cycle_entry" if args.direct_cycle_entry else
                "cycle_capture" if args.capture_cycle else
                "oracle_transport" if args.oracle_transport else
                "held_slow_forcing" if args.hold_slow_forcing else
                "production"),
        "ordered_next": ("redi_t" if disposition in {
            "TRACER_ENTRY_ROW8_AT_BAR",
            "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT",
            "TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT",
            "TRACER_ENTRY_ROW8_AT_BAR_DIRECT_KMM",
            "TRACER_ENTRY_ROW8_AT_BAR_LIVE_QCO"} else first),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_diverged_subrow={first}")
    for row in rows:
        if "metrics" in row:
            print(row["subrow"], row["status"],
                  row["metrics"]["n_diverged_columns"],
                  row["metrics"]["max_column_error"])
        else:
            print(row["subrow"], row["status"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
