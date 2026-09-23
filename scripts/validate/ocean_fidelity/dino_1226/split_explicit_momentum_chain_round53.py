#!/usr/bin/env python3
"""Certify production Kmm execute/undo cycle and momentum filter."""
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
import split_explicit_momentum_chain_round52 as r52
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND52_SHA = "b1ad9f8d3ee58b68806180ac23109d10fac0b2e2766462b7e59fe5bc188f4d07"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_model(native, component):
    if component == "u":
        value = np.concatenate([native[:, -1:], native], axis=1)
    else:
        value = np.concatenate([np.zeros_like(native[:1]), native], axis=0)
    return jnp.asarray(value[..., None])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round52", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round52.resolve()) != ROUND52_SHA:
        raise SystemExit("official round-52 receipt changed")
    prior = json.loads(args.round52.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "MOMENTUM_TAIL_DIVERGED_KMM_REWRITE_U"):
        raise SystemExit("round 52 does not release production certification")

    run = args.run_stepdump.resolve()
    retained = {**r52.EXPECTED, **r48.TARGET_SHA,
                "spg_dump_zu_frc.bin": r47.r46.HELD_SHA["spg_dump_zu_frc.bin"],
                "spg_dump_zv_frc.bin": r47.r46.HELD_SHA["spg_dump_zv_frc.bin"]}
    for name, expected in retained.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained round-53 input changed: {name}")
    before = {
        c: r47._load3(run / f"baro_dump_{c}_before_kt00005761.bin")
        for c in ("u", "v")
    }
    target = {
        "u": r48._load2(run / "spg_dump_puu_b_final.bin"),
        "v": r48._load2(run / "spg_dump_pvv_b_final.bin"),
    }
    kmm_oracle = {
        "u": r47._load3(run / "atf_dump_uu_before.bin"),
        "v": r47._load3(run / "atf_dump_vv_before.bin"),
    }
    filtered_oracle = {
        "u": r47._load3(run / "atf_dump_uu_after.bin"),
        "v": r47._load3(run / "atf_dump_vv_after.bin"),
    }
    held_u = r47.r46._u_face(r47.r46._native(run / "spg_dump_zu_frc.bin"))
    held_v = r47.r46._v_face(r47.r46._native(run / "spg_dump_zv_frc.bin"))
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        masks = {
            "u": np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > .5,
            "v": np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > .5,
        }

    set_policy(PrecisionPolicy.fp64())
    _, _, _, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    state = model._seed_tke_preclosure_carry(state)
    real_solver = model_module.barotropic_substeps_latlon_cgrid
    model_class = type(model)
    real_reconcile = model_class._apply_after_level_reconcile
    real_cycle = model_module.nemo_qco_kmm_velocity_cycle
    solver_calls = reconcile_calls = cycle_calls = 0
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

    def exact_reconcile(self, naa, state_arg, btu, btv, um3, vm3, grid,
                        kaa_eta_raw=None, z_coord=None, config=None):
        nonlocal reconcile_calls
        reconcile_calls += 1
        nemo_input = naa._replace(
            u=naa.u.replace(data=jnp.asarray(r47._to_model(before["u"], "u"))),
            v=naa.v.replace(data=jnp.asarray(r47._to_model(before["v"], "v"))))
        return real_reconcile(
            self, nemo_input, state_arg,
            _target_model(target["u"], "u"),
            _target_model(target["v"], "v"),
            um3, vm3, grid, kaa_eta_raw=kaa_eta_raw,
            z_coord=z_coord, config=config)

    def capture_cycle(*cycle_args, **cycle_kwargs):
        nonlocal cycle_calls
        cycle_calls += 1
        values = real_cycle(*cycle_args, **cycle_kwargs)
        captured["kmm_u"] = r47._from_model(values[2], "u")
        captured["kmm_v"] = r47._from_model(values[3], "v")
        return values

    model_module.barotropic_substeps_latlon_cgrid = held_solver
    model_class._apply_after_level_reconcile = exact_reconcile
    model_module.nemo_qco_kmm_velocity_cycle = capture_cycle
    try:
        with jax.disable_jit():
            result = model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.barotropic_substeps_latlon_cgrid = real_solver
        model_class._apply_after_level_reconcile = real_reconcile
        model_module.nemo_qco_kmm_velocity_cycle = real_cycle
    restored = (model_module.barotropic_substeps_latlon_cgrid is real_solver
                and model_class._apply_after_level_reconcile is real_reconcile
                and model_module.nemo_qco_kmm_velocity_cycle is real_cycle)
    captured["filtered_u"] = r47._from_model(result.u_before.data, "u")
    captured["filtered_v"] = r47._from_model(result.v_before.data, "v")

    rows = {
        "kmm_rewrite_u": r47._metric(captured["kmm_u"], kmm_oracle["u"], masks["u"]),
        "kmm_rewrite_v": r47._metric(captured["kmm_v"], kmm_oracle["v"], masks["v"]),
        "dyn_atf_u": r47._metric(captured["filtered_u"], filtered_oracle["u"], masks["u"]),
        "dyn_atf_v": r47._metric(captured["filtered_v"], filtered_oracle["v"], masks["v"]),
    }
    controls = {
        "round52_skipped_cycle_u_debt": prior["rows"]["kmm_rewrite_u"]["gate_status"] == "DEBT",
        "round52_skipped_cycle_v_debt": prior["rows"]["kmm_rewrite_v"]["gate_status"] == "DEBT",
        "identity_u_at_bar": r47._metric(filtered_oracle["u"], filtered_oracle["u"], masks["u"])["gate_status"] == "AT BAR",
        "identity_v_at_bar": r47._metric(filtered_oracle["v"], filtered_oracle["v"], masks["v"])["gate_status"] == "AT BAR",
        "u_point_plant_fires": r47._plant(filtered_oracle["u"], masks["u"]),
        "v_point_plant_fires": r47._plant(filtered_oracle["v"], masks["v"]),
        "u_roll_plant_fires": r47._metric(np.roll(filtered_oracle["u"], 1, axis=1), filtered_oracle["u"], masks["u"])["gate_status"] != "AT BAR",
        "v_roll_plant_fires": r47._metric(np.roll(filtered_oracle["v"], 1, axis=1), filtered_oracle["v"], masks["v"])["gate_status"] != "AT BAR",
        "solver_calls_one": solver_calls == 1,
        "reconcile_calls_one": reconcile_calls == 1,
        "cycle_calls_one": cycle_calls == 1,
        "hooks_restored": restored,
    }
    valid = all(controls.values())
    first = next((n for n, row in rows.items()
                  if row["gate_status"] != "AT BAR"), None)
    disposition = ("MOMENTUM_TAIL_AT_BAR_UPSTREAM_EXACT"
                   if valid and first is None else "INVALID" if not valid
                   else f"MOMENTUM_TAIL_DIVERGED_{first.upper()}")
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round53-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "chain": "momentum_rhs_tail",
        "rows": rows,
        "controls": controls,
        "bindings": {
            "round52": _sha(args.round52.resolve()),
            **{name: _sha(run / name) for name in retained},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round53.md"),
            "model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "momentum": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "first_diverged_row": first,
        "ordered_next": "tracer_tail" if disposition
                        == "MOMENTUM_TAIL_AT_BAR_UPSTREAM_EXACT" else first,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_diverged_row={first}")
    for name, row in rows.items():
        print(name, row["gate_status"], row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
