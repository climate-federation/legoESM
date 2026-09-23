#!/usr/bin/env python3
"""Certify the production raw-Kaa/live-QCO row-6 carry."""
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
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import split_explicit_momentum_chain_round47 as r47
import split_explicit_momentum_chain_round48 as r48
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND49_SHA = "501f15eb8b2893b2f04fd73e8256ab9d2099cec865b7673198e3f359718b7018"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_model(native: np.ndarray, component: str) -> np.ndarray:
    if component == "u":
        value = np.concatenate([native[:, -1:], native], axis=1)
    else:
        value = np.concatenate([np.zeros_like(native[:1]), native], axis=0)
    return value[..., None]


def _native_target(value, component: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    expected = (199, 53) if component == "u" else (200, 52)
    if array.shape != expected:
        raise SystemExit(
            f"production {component}-target shape changed: {array.shape}")
    return array[:, 1:] if component == "u" else array[1:, :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round49", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round49.resolve()) != ROUND49_SHA:
        raise SystemExit("official round-49 receipt changed")
    prior = json.loads(args.round49.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "ROW6_LOCALIZED_TO_LIVE_QCO_ASSOCIATION_GIVEN_ORACLE_TARGET"):
        raise SystemExit("round 49 does not release production certification")

    run = args.run_stepdump.resolve()
    retained = {**r47.EXPECTED, **r48.TARGET_SHA}
    for name, expected in retained.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained round-50 input changed: {name}")
    before = {
        c: r47._load3(run / f"baro_dump_{c}_before_kt00005761.bin")
        for c in ("u", "v")
    }
    after = {
        c: r47._load3(run / f"baro_dump_{c}_after_kt00005761.bin")
        for c in ("u", "v")
    }
    target = {
        "u": r48._load2(run / "spg_dump_puu_b_final.bin"),
        "v": r48._load2(run / "spg_dump_pvv_b_final.bin"),
    }
    held_u = r47.r46._u_face(r47.r46._native(run / "spg_dump_zu_frc.bin"))
    held_v = r47.r46._v_face(r47.r46._native(run / "spg_dump_zv_frc.bin"))
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        masks = {
            "u": np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > .5,
            "v": np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > .5,
        }
    if (int(masks["u"][..., 0].sum()) != 9758
            or int(masks["v"][..., 0].sum()) != 9868):
        raise SystemExit("row-6 active populations changed")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    if cfg.barotropic.barotropic_after_reconcile != "nemo_mlf_baro_corr":
        raise SystemExit("faithful row-6 selector changed")
    state = model._seed_tke_preclosure_carry(state)

    real_solver = model_module.barotropic_substeps_latlon_cgrid
    model_class = type(model)
    real_reconcile = model_class._apply_after_level_reconcile
    solver_calls = 0
    hook_calls = 0
    captured: dict[str, object] = {}

    def held_solver(s, dt_s, n_substeps, grid, z_coord, config, **kwargs):
        nonlocal solver_calls
        kwargs = dict(kwargs)
        if kwargs.get("eta_init") is not None:
            solver_calls += 1
            kwargs["F_slow_u"] = jnp.asarray(
                held_u, dtype=kwargs["F_slow_u"].dtype)
            kwargs["F_slow_v"] = jnp.asarray(
                held_v, dtype=kwargs["F_slow_v"].dtype)
        return real_solver(s, dt_s, n_substeps, grid, z_coord, config, **kwargs)

    def capture(self, naa, state_arg, btu, btv, um3, vm3, grid,
                kaa_eta_raw=None, z_coord=None, config=None):
        nonlocal hook_calls
        hook_calls += 1
        if kaa_eta_raw is None:
            raise SystemExit("production row-6 path did not carry raw Kaa SSH")
        nemo_input = naa._replace(
            u=naa.u.replace(data=jnp.asarray(r47._to_model(before["u"], "u"))),
            v=naa.v.replace(data=jnp.asarray(r47._to_model(before["v"], "v"))))
        arms = {
            "P_production_target": (btu, btv),
            "T_oracle_target": (
                jnp.asarray(_target_model(target["u"], "u")),
                jnp.asarray(_target_model(target["v"], "v"))),
        }
        for name, (target_u, target_v) in arms.items():
            out = real_reconcile(
                self, nemo_input, state_arg, target_u, target_v,
                um3, vm3, grid, kaa_eta_raw=kaa_eta_raw,
                z_coord=z_coord, config=config)
            captured[name] = {
                c: r47._from_model(getattr(out, c).data, c)
                for c in ("u", "v")
            }
        captured["target_production"] = {
            "u": _native_target(btu, "u"),
            "v": _native_target(btv, "v"),
        }
        captured["kaa_eta_raw"] = np.asarray(kaa_eta_raw, dtype=np.float64)
        captured["eta_entry"] = np.asarray(state_arg.eta.data, dtype=np.float64)
        return real_reconcile(
            self, naa, state_arg, btu, btv, um3, vm3, grid,
            kaa_eta_raw=kaa_eta_raw, z_coord=z_coord, config=config)

    model_module.barotropic_substeps_latlon_cgrid = held_solver
    model_class._apply_after_level_reconcile = capture
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.barotropic_substeps_latlon_cgrid = real_solver
        model_class._apply_after_level_reconcile = real_reconcile
    restored = (model_module.barotropic_substeps_latlon_cgrid is real_solver
                and model_class._apply_after_level_reconcile is real_reconcile)
    if solver_calls != 1 or hook_calls != 1 or not restored:
        raise SystemExit("row-50 hook/solver/restoration contract failed")

    scored = {
        name: {c: r47._metric(values[c], after[c], masks[c])
               for c in ("u", "v")}
        for name, values in captured.items() if name[0] in ("P", "T")
    }
    raw = captured["kaa_eta_raw"]
    stale = captured["eta_entry"]
    controls = {
        "identity_u_at_bar": r47._metric(
            after["u"], after["u"], masks["u"])["gate_status"] == "AT BAR",
        "identity_v_at_bar": r47._metric(
            after["v"], after["v"], masks["v"])["gate_status"] == "AT BAR",
        "u_point_plant_fires": r47._plant(after["u"], masks["u"]),
        "v_point_plant_fires": r47._plant(after["v"], masks["v"]),
        "u_roll_plant_fires": r47._metric(
            np.roll(after["u"], 1, axis=1), after["u"], masks["u"]
        )["gate_status"] != "AT BAR",
        "v_roll_plant_fires": r47._metric(
            np.roll(after["v"], 1, axis=1), after["v"], masks["v"]
        )["gate_status"] != "AT BAR",
        "raw_kaa_finite": bool(np.all(np.isfinite(raw))),
        "raw_kaa_differs_from_stale_entry": bool(np.any(raw != stale)),
        "nemo_u_correction_nonzero": bool(
            np.any((after["u"] - before["u"])[masks["u"]] != 0.0)),
        "nemo_v_correction_nonzero": bool(
            np.any((after["v"] - before["v"])[masks["v"]] != 0.0)),
        "solver_calls_one": solver_calls == 1,
        "hook_calls_one": hook_calls == 1,
        "hooks_restored": restored,
    }
    valid = all(controls.values())
    t_at_bar = all(
        scored["T_oracle_target"][c]["gate_status"] == "AT BAR"
        for c in ("u", "v"))
    if not valid:
        disposition = "INVALID"
    elif t_at_bar:
        disposition = "ROW6_MLF_BARO_CORR_AT_BAR_UPSTREAM_TARGET_EXACT"
    elif scored["T_oracle_target"]["u"]["gate_status"] != "AT BAR":
        disposition = "ROW6_MLF_BARO_CORR_U_DIVERGED"
    else:
        disposition = "ROW6_MLF_BARO_CORR_V_DIVERGED"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round50-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 6,
        "arms": scored,
        "primary_target_qualification": {
            c: r47._metric(captured["target_production"][c], target[c],
                           masks[c][..., 0])
            for c in ("u", "v")
        },
        "controls": controls,
        "bindings": {
            "round49": _sha(args.round49.resolve()),
            **{name: _sha(run / name) for name in retained},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round50.md"),
            "model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "kernel": _sha(root / "packages/ocean/legoesm/ocean/dynamics/barotropic_common.py"),
            "geometry": _sha(root / "packages/ocean/legoesm/ocean/vertical.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "ordered_next": (
            "free_surface_filter" if disposition
            == "ROW6_MLF_BARO_CORR_AT_BAR_UPSTREAM_TARGET_EXACT" else 6),
    }
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, rows in scored.items():
        print(name, rows["u"]["normalized_rms_error"],
              rows["u"]["per_element_max_error_over_nemo_rms"],
              rows["v"]["normalized_rms_error"],
              rows["v"]["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
