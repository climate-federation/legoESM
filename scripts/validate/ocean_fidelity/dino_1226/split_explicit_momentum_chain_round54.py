#!/usr/bin/env python3
"""Substitute retained NEMO transports into the production Kmm cycle."""
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
import split_explicit_momentum_chain_round47 as r47
import split_explicit_momentum_chain_round48 as r48
import split_explicit_momentum_chain_round52 as r52
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_qco_kmm_velocity_cycle,
)


ROUND53_SHA = "f6e2ea972dbc4594cae704e4641f4f20e92b03ba25bd890a610ee2ea11c1e142"
TRANSPORT_SHA = {
    "spg_dump_un_adv_final.bin": "6f56785155886c7618b8c32fd3f4b9f391325ccdc6a4e7cc11bdbd654170a6fb",
    "spg_dump_vn_adv_final.bin": "9abb35a32437797aecc60c2e6fea28f2abc1b609662b26b8a669c124f661dab0",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_transport(native, component):
    if component == "u":
        return np.concatenate([native[:, -1:], native], axis=1)
    return np.concatenate([np.zeros_like(native[:1]), native], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round53", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round53.resolve()) != ROUND53_SHA:
        raise SystemExit("official round-53 receipt changed")
    prior = json.loads(args.round53.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "MOMENTUM_TAIL_DIVERGED_KMM_REWRITE_U"):
        raise SystemExit("round 53 does not release the transport substitution")
    run = args.run_stepdump.resolve()
    retained = {**r52.EXPECTED, **r48.TARGET_SHA, **TRANSPORT_SHA}
    for name, expected in retained.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained round-54 input changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    _, _, cfg, model, _, _, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        masks = {
            "u": np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > .5,
            "v": np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > .5,
        }
    un_adv = _to_transport(
        r48._load2(run / "spg_dump_un_adv_final.bin"), "u")
    vn_adv = _to_transport(
        r48._load2(run / "spg_dump_vn_adv_final.bin"), "v")
    u_mask3 = state.u_mask.data[..., None]
    v_mask3 = state.v_mask.data[..., None]
    if hasattr(model.z_coord, "is_active"):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
        au, av = compute_face_masks_3d(model.z_coord.is_active, model.grid)
        u_mask3 = u_mask3 * au.astype(u_mask3.dtype)
        v_mask3 = v_mask3 * av.astype(v_mask3.dtype)
    _, _, restored_u, restored_v = nemo_qco_kmm_velocity_cycle(
        state.eta.data, state.u.data, state.v.data,
        jnp.asarray(un_adv), jnp.asarray(vn_adv), model.z_coord,
        u_mask3, v_mask3)
    restored = {
        "u": r47._from_model(restored_u, "u"),
        "v": r47._from_model(restored_v, "v"),
    }
    kmm_oracle = {
        "u": r47._load3(run / "atf_dump_uu_before.bin"),
        "v": r47._load3(run / "atf_dump_vv_before.bin"),
    }
    filtered_oracle = {
        "u": r47._load3(run / "atf_dump_uu_after.bin"),
        "v": r47._load3(run / "atf_dump_vv_after.bin"),
    }
    kbb = {
        "u": r47._from_model(state.u_before.data, "u"),
        "v": r47._from_model(state.v_before.data, "v"),
    }
    kaa = {
        "u": r47._load3(run / "seq_dump_postlbc_u_aaa_kt00005761.bin"),
        "v": r47._load3(run / "seq_dump_postlbc_v_aaa_kt00005761.bin"),
    }
    filtered = {
        c: np.asarray(jnp.asarray(restored[c]) + cfg.asselin_gamma * (
            jnp.asarray(kbb[c]) - 2.0 * jnp.asarray(restored[c])
            + jnp.asarray(kaa[c])))
        for c in ("u", "v")
    }
    rows = {}
    for c in ("u", "v"):
        rows[f"kmm_rewrite_{c}"] = r47._metric(
            restored[c], kmm_oracle[c], masks[c])
        rows[f"dyn_atf_{c}"] = r47._metric(
            filtered[c], filtered_oracle[c], masks[c])
    controls = {
        "round53_production_u_debt": prior["rows"]["kmm_rewrite_u"]["gate_status"] == "DEBT",
        "round53_production_v_debt": prior["rows"]["kmm_rewrite_v"]["gate_status"] == "DEBT",
        "identity_u_at_bar": r47._metric(filtered_oracle["u"], filtered_oracle["u"], masks["u"])["gate_status"] == "AT BAR",
        "identity_v_at_bar": r47._metric(filtered_oracle["v"], filtered_oracle["v"], masks["v"])["gate_status"] == "AT BAR",
        "u_point_plant_fires": r47._plant(filtered_oracle["u"], masks["u"]),
        "v_point_plant_fires": r47._plant(filtered_oracle["v"], masks["v"]),
        "u_roll_plant_fires": r47._metric(np.roll(filtered_oracle["u"], 1, axis=1), filtered_oracle["u"], masks["u"])["gate_status"] != "AT BAR",
        "v_roll_plant_fires": r47._metric(np.roll(filtered_oracle["v"], 1, axis=1), filtered_oracle["v"], masks["v"])["gate_status"] != "AT BAR",
        "all_rows_finite": all(
            np.isfinite(row["normalized_rms_error"])
            and np.isfinite(row["per_element_max_error_over_nemo_rms"])
            for row in rows.values()),
        "all_populations_nonempty": all(
            int(row.get("active_count", 0)) > 0 for row in rows.values()),
    }
    valid = all(controls.values())
    order = ("kmm_rewrite_u", "kmm_rewrite_v", "dyn_atf_u", "dyn_atf_v")
    first = next((n for n in order if rows[n]["gate_status"] != "AT BAR"), None)
    disposition = ("MOMENTUM_TAIL_AT_BAR_UPSTREAM_TRANSPORT_EXACT"
                   if valid and first is None else "INVALID" if not valid
                   else f"MOMENTUM_TAIL_DIVERGED_{first.upper()}")
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round54-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "rows": rows,
        "controls": controls,
        "bindings": {
            "round53": _sha(args.round53.resolve()),
            **{name: _sha(run / name) for name in retained},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round54.md"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "production_model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "production_momentum": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
        },
        "disposition": disposition,
        "first_diverged_row": first,
        "ordered_next": "tracer_tail" if disposition
                        == "MOMENTUM_TAIL_AT_BAR_UPSTREAM_TRANSPORT_EXACT" else first,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_diverged_row={first}")
    for name in order:
        row = rows[name]
        print(name, row["gate_status"], row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
