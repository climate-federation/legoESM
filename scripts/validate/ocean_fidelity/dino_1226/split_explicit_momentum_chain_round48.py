#!/usr/bin/env python3
"""Row-6 target x source-association factorial from retained operands."""
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
import split_explicit_momentum_chain_round47 as r47
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND47_SHA = "60a4b7d8f45b8cc9a611deb4d4e641e5952c67960f6edff2ddade51771a3e4aa"
TARGET_SHA = {
    "spg_dump_puu_b_final.bin": "14d2085b34d027f13c6a489c1f1c39a09d1fd3cd56d1e3e1edb718fed458f50a",
    "spg_dump_pvv_b_final.bin": "d3d2fca6d60cc9015ca120bf4a7e523f8bca18049198a44fb8fe8ac395b41ac3",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load2(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 203 * 56:
        raise SystemExit(f"{path}: expected full-halo (203,56) stream")
    return raw.reshape(203, 56)[2:-2, 2:-2]


def _target_model(native, component):
    return (np.concatenate([native[:, -1:], native], axis=1)
            if component == "u" else
            np.concatenate([np.zeros_like(native[:1]), native], axis=0))


def _native_target(value, component):
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    expected = (199, 53) if component == "u" else (200, 52)
    if array.shape != expected:
        raise SystemExit(
            f"production {component}-target shape changed: {array.shape}")
    return array[:, 1:] if component == "u" else array[1:, :]


def _literal(before, target_native, component, z_coord, mask):
    e3 = jnp.asarray(z_coord.nemo_e3t_0, dtype=jnp.float64)[..., :35]
    h0 = jnp.asarray(
        z_coord.nemo_hu_0 if component == "u" else z_coord.nemo_hv_0,
        dtype=jnp.float64)
    field = jnp.asarray(before, dtype=jnp.float64)
    wet = jnp.asarray(mask, dtype=jnp.float64)
    acc = jnp.zeros_like(h0)
    for jk in range(35):
        acc = jax.lax.optimization_barrier(
            acc + e3[..., jk] * field[..., jk] * wet[..., jk])
    wet2 = wet[..., 0]
    r1 = jax.lax.optimization_barrier(wet2 / (h0 + 1.0 - wet2))
    mean = jax.lax.optimization_barrier(acc * r1)
    target = jnp.asarray(target_native, dtype=jnp.float64)
    return jax.lax.optimization_barrier(
        field - mean[..., None] + target[..., None]) * wet


def _energy(candidate, oracle, mask):
    delta = np.asarray(candidate)[mask] - np.asarray(oracle)[mask]
    return float(np.sum(delta * delta, dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round47", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round47.resolve()) != ROUND47_SHA:
        raise SystemExit("official round-47 receipt changed")
    prior = json.loads(args.round47.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "ROW6_MLF_BARO_CORR_U_DIVERGED"):
        raise SystemExit("round 47 does not open the row-6 factorial")
    run = args.run_stepdump.resolve()
    for name, expected in {**r47.EXPECTED, **TARGET_SHA}.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained factorial input changed: {name}")

    before = {c: r47._load3(run / f"baro_dump_{c}_before_kt00005761.bin")
              for c in ("u", "v")}
    after = {c: r47._load3(run / f"baro_dump_{c}_after_kt00005761.bin")
             for c in ("u", "v")}
    target1 = {"u": _load2(run / "spg_dump_puu_b_final.bin"),
               "v": _load2(run / "spg_dump_pvv_b_final.bin")}
    held_u = r47.r46._u_face(r47.r46._native(run / "spg_dump_zu_frc.bin"))
    held_v = r47.r46._v_face(r47.r46._native(run / "spg_dump_zv_frc.bin"))
    import netCDF4
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        masks = {
            "u": np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > 0.5,
            "v": np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > 0.5,
        }

    set_policy(PrecisionPolicy.fp64())
    _, _, cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    state = model._seed_tke_preclosure_carry(state)
    real_solver = model_module.barotropic_substeps_latlon_cgrid
    model_class = type(model)
    real_reconcile = model_class._apply_after_level_reconcile
    solver_calls = hook_calls = 0
    captured = {}

    def held_solver(s, dt_s, n_substeps, grid, z_coord, config, **kwargs):
        nonlocal solver_calls
        kwargs = dict(kwargs)
        if kwargs.get("eta_init") is not None:
            solver_calls += 1
            kwargs["F_slow_u"] = jnp.asarray(held_u, dtype=kwargs["F_slow_u"].dtype)
            kwargs["F_slow_v"] = jnp.asarray(held_v, dtype=kwargs["F_slow_v"].dtype)
        return real_solver(s, dt_s, n_substeps, grid, z_coord, config, **kwargs)

    def capture(self, naa, state_arg, btu, btv, um3, vm3, grid,
                z_coord=None, config=None):
        nonlocal hook_calls
        hook_calls += 1
        inputs = {
            "u": naa.u.replace(data=jnp.asarray(r47._to_model(before["u"], "u"))),
            "v": naa.v.replace(data=jnp.asarray(r47._to_model(before["v"], "v"))),
        }
        nemo_input = naa._replace(**inputs)
        targets0 = {"u": btu, "v": btv}
        for ti, targets in ((0, targets0), (1, {
                "u": jnp.asarray(_target_model(target1["u"], "u"))[..., None],
                "v": jnp.asarray(_target_model(target1["v"], "v"))[..., None]})):
            out = real_reconcile(
                self, nemo_input, state_arg, targets["u"], targets["v"],
                um3, vm3, grid, z_coord=z_coord, config=config)
            captured[f"T{ti}A0"] = {
                c: r47._from_model(getattr(out, c).data, c) for c in ("u", "v")}
            target_native = {
                c: _native_target(targets[c], c) for c in ("u", "v")
            }
            captured[f"T{ti}A1"] = {
                c: np.asarray(_literal(
                    before[c], target_native[c], c,
                    self.z_coord if z_coord is None else z_coord, masks[c]))
                for c in ("u", "v")}
        captured["target0"] = {
            c: _native_target(value, c)
            for c, value in (("u", btu), ("v", btv))}
        return real_reconcile(
            self, naa, state_arg, btu, btv, um3, vm3, grid,
            z_coord=z_coord, config=config)

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
        raise SystemExit("factorial hook/restoration contract failed")

    arms = {
        name: {c: r47._metric(values[c], after[c], masks[c])
               for c in ("u", "v")}
        for name, values in captured.items() if name.startswith("T")
    }
    energy = {name: sum(_energy(captured[name][c], after[c], masks[c])
                        for c in ("u", "v")) for name in arms}
    base = energy["T0A0"]
    removal = {name: 1.0 - value / base for name, value in energy.items()}
    controls = {
        "literal_joint_u_at_bar": arms["T1A1"]["u"]["gate_status"] == "AT BAR",
        "literal_joint_v_at_bar": arms["T1A1"]["v"]["gate_status"] == "AT BAR",
        "round47_baseline_reproduced_u": arms["T0A0"]["u"] == prior["row6"]["u"],
        "round47_baseline_reproduced_v": arms["T0A0"]["v"] == prior["row6"]["v"],
        "solver_calls_one": solver_calls == 1,
        "hook_calls_one": hook_calls == 1,
        "hooks_restored": restored,
    }
    valid = all(controls.values())
    t_remove, a_remove = removal["T1A0"], removal["T0A1"]
    if not valid:
        disposition = "OPEN_UNRESOLVED"
    elif t_remove >= .95 and a_remove < .05:
        disposition = "ROW6_LOCALIZED_TO_PRIMARY_TARGET"
    elif a_remove >= .95 and t_remove < .05:
        disposition = "ROW6_LOCALIZED_TO_SOURCE_ASSOCIATION"
    elif ((.05 <= t_remove < .95 and .05 <= a_remove < .95)
          or t_remove < 0 or a_remove < 0):
        disposition = "ROW6_TARGET_X_ASSOCIATION_COMPOSITION"
    else:
        disposition = "ROW6_OPERANDS_BOUNDED"

    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round48-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 6,
        "arms": arms,
        "squared_error": energy,
        "squared_error_removal": removal,
        "target0_vs_target1": {
            c: r47._metric(captured["target0"][c], target1[c], masks[c][..., 0])
            for c in ("u", "v")},
        "controls": controls,
        "bindings": {
            "round47": _sha(args.round47.resolve()),
            **{name: _sha(run / name) for name in {**r47.EXPECTED, **TARGET_SHA}},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round48.md"),
        },
        "disposition": disposition,
        "ordered_next": 6,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name in ("T0A0", "T1A0", "T0A1", "T1A1"):
        print(name, arms[name]["u"]["per_element_max_error_over_nemo_rms"],
              arms[name]["v"]["per_element_max_error_over_nemo_rms"], removal[name])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
