#!/usr/bin/env python3
"""Certify production mlf_baro_corr from retained day-180 operands."""
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
import split_explicit_momentum_chain_round46 as r46
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND46_SHA = "a5c68a419877b61605e7cfa1e3e67caf0b5b8d0e77c32c2718e5e2fd71a252f2"
EXPECTED = {
    "baro_dump_u_before_kt00005761.bin": "dd6aa8b6f3794a5aa3b3bffa2c1f50548792b9adaff1d0ff76e5e29ba42ff8d0",
    "baro_dump_u_after_kt00005761.bin": "cd9434ab613ee6e2fed0dd5245383d50b365c2347d4c5cc334e262221e673163",
    "baro_dump_v_before_kt00005761.bin": "ac510a24ae30eab9298cbd57972022dc92c4a0c6e8faf4a196ec5f800ff049df",
    "baro_dump_v_after_kt00005761.bin": "068b26d53d3bcb88d2853ae62ddb9ebfdc9d063d7241ec747ada8dedbdae8b1f",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
    **r46.HELD_SHA,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load3(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 35 * 203 * 56:
        raise SystemExit(f"{path}: expected full-halo (35,203,56) stream")
    return np.moveaxis(raw.reshape(35, 203, 56)[:, 2:-2, 2:-2], 0, -1)


def _to_model(native: np.ndarray, component: str) -> np.ndarray:
    if component == "u":
        redundant = np.concatenate([native[:, -1:, :], native], axis=1)
    else:
        redundant = np.concatenate([np.zeros_like(native[:1]), native], axis=0)
    return np.concatenate(
        [redundant, np.zeros(redundant.shape[:2] + (1,), dtype=redundant.dtype)],
        axis=-1)


def _from_model(value, component: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array[:, 1:, :35] if component == "u" else array[1:, :, :35]


def _metric(candidate, oracle, mask):
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if candidate.shape != oracle.shape or mask.shape != oracle.shape:
        raise SystemExit("row-6 metric shape mismatch")
    if not np.all(np.isfinite(candidate[mask])):
        raise SystemExit("row-6 candidate contains non-finite wet values")
    ref = oracle[mask]
    delta = candidate[mask] - ref
    ref_rms = float(np.sqrt(np.mean(ref * ref, dtype=np.float64)))
    err_rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    norm = err_rms / ref_rms
    maximum = float(np.max(np.abs(delta))) / ref_rms
    return {
        "active_count": int(mask.sum()),
        "oracle_rms": ref_rms,
        "error_rms": err_rms,
        "normalized_rms_error": norm,
        "per_element_max_error_over_nemo_rms": maximum,
        "correlation": float(np.corrcoef(candidate[mask], ref)[0, 1]),
        "gate_status": "AT BAR" if norm <= 1e-15 and maximum <= 1e-15 else "DEBT",
    }


def _plant(oracle, mask):
    value = np.array(oracle, copy=True)
    points = np.argwhere(mask)
    point = tuple(points[int(np.argmax(np.abs(oracle[mask])))])
    rms = float(np.sqrt(np.mean(oracle[mask] ** 2, dtype=np.float64)))
    value[point] += 2e-15 * rms
    return _metric(value, oracle, mask)["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round46", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round46.resolve()) != ROUND46_SHA:
        raise SystemExit("official round-46 receipt changed")
    prior = json.loads(args.round46.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "ROW5_WZV_CALL2_AT_BAR_UPSTREAM_EXACT"):
        raise SystemExit("round 46 does not release row 6")
    run = args.run_stepdump.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained row-6 input changed: {name}")

    before = {c: _load3(run / f"baro_dump_{c}_before_kt00005761.bin")
              for c in ("u", "v")}
    after = {c: _load3(run / f"baro_dump_{c}_after_kt00005761.bin")
             for c in ("u", "v")}
    held_u = r46._u_face(r46._native(run / "spg_dump_zu_frc.bin"))
    held_v = r46._v_face(r46._native(run / "spg_dump_zv_frc.bin"))
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        umask = np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > 0.5
        vmask = np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > 0.5
    if (umask.shape != before["u"].shape or vmask.shape != before["v"].shape
            or int(umask[..., 0].sum()) != 9758
            or int(vmask[..., 0].sum()) != 9868):
        raise SystemExit("row-6 cited-interior mask contract changed")

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
    captured = {}

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

    def capture_reconcile(self, naa, state_arg, btu, btv, um3, vm3, grid,
                          z_coord=None, config=None):
        nonlocal hook_calls
        hook_calls += 1
        nemo_input = naa._replace(
            u=naa.u.replace(data=jnp.asarray(_to_model(before["u"], "u"))),
            v=naa.v.replace(data=jnp.asarray(_to_model(before["v"], "v"))))
        local = real_reconcile(
            self, nemo_input, state_arg, btu, btv, um3, vm3, grid,
            z_coord=z_coord, config=config)
        null = real_reconcile(
            self, nemo_input, state_arg, btu, btv, um3, vm3, grid,
            z_coord=z_coord, config=config)
        production = real_reconcile(
            self, naa, state_arg, btu, btv, um3, vm3, grid,
            z_coord=z_coord, config=config)
        captured.update({
            "u": _from_model(local.u.data, "u"),
            "v": _from_model(local.v.data, "v"),
            "null_u": np.asarray(null.u.data),
            "null_v": np.asarray(null.v.data),
            "local_u": np.asarray(local.u.data),
            "local_v": np.asarray(local.v.data),
            "production_u": np.asarray(production.u.data),
            "production_v": np.asarray(production.v.data),
        })
        return production

    model_module.barotropic_substeps_latlon_cgrid = held_solver
    model_class._apply_after_level_reconcile = capture_reconcile
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.barotropic_substeps_latlon_cgrid = real_solver
        model_class._apply_after_level_reconcile = real_reconcile
    restored = (model_module.barotropic_substeps_latlon_cgrid is real_solver
                and model_class._apply_after_level_reconcile is real_reconcile)
    if solver_calls != 1 or hook_calls != 1 or not captured or not restored:
        raise SystemExit("row-6 hook/solver/restoration contract failed")

    rows = {"u": _metric(captured["u"], after["u"], umask),
            "v": _metric(captured["v"], after["v"], vmask)}
    controls = {
        "identity_u_at_bar": _metric(after["u"], after["u"], umask)["gate_status"] == "AT BAR",
        "identity_v_at_bar": _metric(after["v"], after["v"], vmask)["gate_status"] == "AT BAR",
        "u_point_plant_fires": _plant(after["u"], umask),
        "v_point_plant_fires": _plant(after["v"], vmask),
        "u_roll_plant_fires": _metric(np.roll(after["u"], 1, axis=1), after["u"], umask)["gate_status"] != "AT BAR",
        "v_roll_plant_fires": _metric(np.roll(after["v"], 1, axis=1), after["v"], vmask)["gate_status"] != "AT BAR",
        "unchanged_input_null_u_byte_exact": np.array_equal(captured["null_u"], captured["local_u"]),
        "unchanged_input_null_v_byte_exact": np.array_equal(captured["null_v"], captured["local_v"]),
        "nemo_u_correction_nonzero": bool(np.any((after["u"] - before["u"])[umask] != 0.0)),
        "nemo_v_correction_nonzero": bool(np.any((after["v"] - before["v"])[vmask] != 0.0)),
        "solver_calls_one": solver_calls == 1,
        "hook_calls_one": hook_calls == 1,
        "hooks_restored": restored,
    }
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif rows["u"]["gate_status"] != "AT BAR":
        disposition = "ROW6_MLF_BARO_CORR_U_DIVERGED"
    elif rows["v"]["gate_status"] != "AT BAR":
        disposition = "ROW6_MLF_BARO_CORR_V_DIVERGED"
    else:
        disposition = "ROW6_MLF_BARO_CORR_AT_BAR"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round47-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 6,
        "row6": rows,
        "controls": controls,
        "bindings": {
            "round46": _sha(args.round46.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round47.md"),
            "model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "kernel": _sha(root / "packages/ocean/legoesm/ocean/dynamics/barotropic_common.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "ordered_next": "free_surface_filter" if disposition == "ROW6_MLF_BARO_CORR_AT_BAR" else 6,
    }
    if r46.r45.r44._tracked(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for component in ("u", "v"):
        row = rows[component]
        print(component.upper(), row["gate_status"], row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
