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
    parser.add_argument("--hold-slow-forcing", action="store_true")
    parser.add_argument("--oracle-transport", action="store_true")
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
    if _sha(args.raw_artifact.resolve()) != RAW_ARTIFACT_SHA:
        raise SystemExit("admitted held row-8 artifact changed")
    held = args.held_dir.resolve()
    for name, expected in HELD_SHA.items():
        if not (held / name).is_file() or _sha(held / name) != expected:
            raise SystemExit(f"held row-8 input changed: {name}")

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
    solver_calls = 0
    transport_substitutions = 0
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
        if is_target and args.oracle_transport:
            transport_substitutions += 1
            solved_state, _ = result
            result = (solved_state, (
                jnp.asarray(oracle_u, dtype=state_arg.eta.data.dtype),
                jnp.asarray(oracle_v, dtype=state_arg.eta.data.dtype)))
        return result

    model_module.add_bolus_to_advecting_flux = observe
    if args.hold_slow_forcing:
        model_module.barotropic_substeps_latlon_cgrid = held_solver
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.add_bolus_to_advecting_flux = original
        model_module.barotropic_substeps_latlon_cgrid = original_solver
    restored = (model_module.add_bolus_to_advecting_flux is original
                and model_module.barotropic_substeps_latlon_cgrid is original_solver)
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
    values = {
        "un": u_corrected[:, 1:, :35],
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
    controls["all_scored_finite"] = all(
        row.get("status") == "ORDERED_BLOCKED"
        or row["metrics"]["n_nonfinite_wet_elements"] == 0 for row in rows)
    controls["registered_population_exact"] = (
        int(np.any(wet, axis=-1).sum()) == 9758 and int(wet.sum()) == 336338)
    valid = all(controls.values())
    disposition = (("TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT"
                    if args.oracle_transport else
                    "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT"
                    if args.hold_slow_forcing else "TRACER_ENTRY_ROW8_AT_BAR")
                   if valid and first is None
                   else "INVALID" if not valid else f"TRACER_ENTRY_DIVERGED_{first}")
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": ("dino-split-explicit-momentum-chain-round61-v1"
                   if args.oracle_transport else
                   "dino-split-explicit-momentum-chain-round59-v1"),
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "disposition": disposition,
        "first_diverged_subrow": first,
        "rows": rows,
        "controls": controls,
        "focus_ji": [list(x) for x in FOCUS],
        "bindings": {
            "round54": _sha(args.round54.resolve()),
            **({"round56": _sha(args.round56.resolve())}
               if args.hold_slow_forcing else {}),
            **({"round59": _sha(args.round59.resolve()),
                "round60": _sha(args.round60.resolve())}
               if args.oracle_transport else {}),
            "held_raw_artifact": _sha(args.raw_artifact.resolve()),
            **{name: _sha(held / name) for name in HELD_SHA},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity" /
                ("PREREG_split_explicit_momentum_chain_round61.md"
                 if args.oracle_transport else
                 "PREREG_split_explicit_momentum_chain_round59.md")),
            "production_model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "production_barotropic": _sha(root / "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_traadv": _sha(nemo / "src/OCE/TRA/traadv.F90"),
            "nemo_traadv_fct": _sha(nemo / "src/OCE/TRA/traadv_fct.F90"),
            "nemo_ldftra": _sha(nemo / "cfgs/DINO/MY_SRC/ldftra.F90"),
        },
        "arm": ("oracle_transport" if args.oracle_transport else
                "held_slow_forcing" if args.hold_slow_forcing else
                "production"),
        "ordered_next": ("redi_t" if disposition in {
            "TRACER_ENTRY_ROW8_AT_BAR",
            "TRACER_ENTRY_ROW8_AT_BAR_UPSTREAM_FORCING_EXACT",
            "TRACER_ENTRY_ROW8_AT_BAR_ORACLE_TRANSPORT"} else first),
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
