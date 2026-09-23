#!/usr/bin/env python3
"""Score the production coupled QCO ZAD implementation at day 180."""
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
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import split_explicit_momentum_chain_round38 as r38
import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from zu_frc_term_walk import _load_full_3d


ROUND39_SHA = "7104c30ad692241f13c828ecd75eadaee284762e64b619d45243471c35b18748"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-stepdump", type=Path, required=True)
    p.add_argument("--run-traj", type=Path, required=True)
    p.add_argument("--round39", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round39) != ROUND39_SHA:
        raise SystemExit("official round-39 receipt changed")
    prior = json.loads(args.round39.read_text())
    if prior.get("disposition") != "ZAD_RESIDUAL_CLOSED_BY_H_GIVEN_W":
        raise SystemExit("round 39 does not admit production implementation")
    set_policy(PrecisionPolicy.fp64())
    _, _, model_cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()),
        str(args.run_stepdump.resolve()), bridge_before=True,
        restart_file="DINO_00005760_restart.nc")
    if (model_cfg.zad_qco_evaluation != "nemo_literal"
            or model_cfg.vertical_momentum_scheme != "nemo_advective"):
        raise SystemExit("coupled production selector/default changed")
    ldf = (state.T_before.data, state.S_before.data,
           state.u_before.data, state.v_before.data)
    with jax.disable_jit():
        _, diag = model.tendencies_with_diagnostics(
            state, surface_forcing=sf, dt=twin.DT, ldf_state=ldf)
        um3, vm3 = compute_face_masks_3d(model.z_coord.is_active, model.grid)
        qww, qhu, qhv = pe._nemo_qco_zad_operands(
            state.eta.data, state.eta_before.data, state.u.data, state.v.data,
            model.grid, model.z_coord, um3, vm3, model.z_coord.is_active,
            2.0 * twin.DT)

    run = args.run_stepdump.resolve()
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        area_t = np.asarray(ds["e1t"][0]) * np.asarray(ds["e2t"][0])
        area_u = np.asarray(ds["e1u"][0]) * np.asarray(ds["e2u"][0])
        area_v = np.asarray(ds["e1v"][0]) * np.asarray(ds["e2v"][0])
        e3u0 = np.moveaxis(np.asarray(ds["e3u_0"][0]), 0, -1)
        e3v0 = np.moveaxis(np.asarray(ds["e3v_0"][0]), 0, -1)
        umask_full = np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)
        vmask_full = np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)
        masks = {
            "u": umask_full[..., :35] > 0.5,
            "v": vmask_full[..., :35] > 0.5,
        }
    weighted = area_t * np.asarray(state.eta.data)
    hu0 = np.sum(e3u0 * umask_full, axis=-1)
    hv0 = np.sum(e3v0 * vmask_full, axis=-1)
    r3u = (0.5 * (weighted + np.roll(weighted, -1, axis=1))
           / np.where(hu0 > 0.0, hu0, 1.0) / area_u)
    north = np.empty_like(weighted)
    north[:-1] = weighted[1:]
    north[-1] = weighted[-1]
    r3v = 0.5 * (weighted + north) / np.where(hv0 > 0.0, hv0, 1.0) / area_v
    expected_hu = e3u0 * (1.0 + r3u[..., None] * umask_full)
    expected_hv = e3v0 * (1.0 + r3v[..., None] * vmask_full)
    ww_oracle = r38._load_ww(run / "wzv_dump_ww_call1.bin")
    tmask = np.asarray(model.z_coord.is_active)
    ww_diff = np.asarray(qww)[..., :36] - ww_oracle
    operand_replay = {
        "ww_normalized_rms_error": float(
            np.sqrt(np.mean(ww_diff[tmask] ** 2))
            / np.sqrt(np.mean(ww_oracle[tmask] ** 2))),
        "ww_max_abs_error": float(np.max(np.abs(ww_diff[tmask]))),
        "live_u_wet_max_abs_error": float(np.max(np.abs(
            np.asarray(qhu)[:, 1:][umask_full > 0.5]
            - expected_hu[umask_full > 0.5]))),
        "live_v_wet_max_abs_error": float(np.max(np.abs(
            np.asarray(qhv)[1:][vmask_full > 0.5]
            - expected_hv[vmask_full > 0.5]))),
    }
    rows = {}
    controls = {}
    for component in ("u", "v"):
        oracle = _load_full_3d(
            str(run / f"zad_dump_d{component}.bin"), 56, 203, 35, 2)
        candidate = r38._crop(
            component, getattr(diag, f"vertadv_{component}").data)
        rows[component] = r38._metric(candidate, oracle, masks[component])
        controls[f"identity_{component}"] = (
            r38._metric(oracle, oracle, masks[component])["gate_status"] == "AT BAR")
        controls[f"two_bar_{component}"] = r38._two_bar_plant(
            oracle, masks[component])
        controls[f"sign_{component}"] = (
            r38._metric(-oracle, oracle, masks[component])["gate_status"] != "AT BAR")
        controls[f"roll_{component}"] = (
            r38._metric(np.roll(oracle, 1, axis=1), oracle,
                        masks[component])["gate_status"] != "AT BAR")
    # Retained red-capable planted half-state: H alone worsened both arms.
    for component in ("u", "v"):
        h_only = prior["arms"]["W0H1A0"][component]["normalized_rms_error"]
        generic = prior["arms"]["W0H0A0"][component]["normalized_rms_error"]
        controls[f"thickness_only_worsens_{component}"] = h_only > generic
    controls = {key: bool(value) for key, value in controls.items()}
    valid = all(controls.values())
    at_bar = all(rows[c]["gate_status"] == "AT BAR" for c in ("u", "v"))
    if not valid:
        disposition = "INVALID"
    elif at_bar:
        disposition = "ROW4_ZAD_AT_BAR"
    else:
        disposition = "OPEN_SOURCE_ASSOCIATION_AMPLIFICATION"
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round40-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "rows": rows,
        "operand_replay": operand_replay,
        "controls": controls,
        "round39_joint_arm": prior["arms"]["W1H1A1"],
        "bindings": {
            "round39": _sha(args.round39.resolve()),
            "mesh_mask": _sha(run / "mesh_mask.nc"),
            "zad_u": _sha(run / "zad_dump_du.bin"),
            "zad_v": _sha(run / "zad_dump_dv.bin"),
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round40.md"),
        },
        "disposition": disposition,
        "ordered_next": 5 if at_bar else 4,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for component in ("u", "v"):
        print(component, rows[component]["normalized_rms_error"],
              rows[component]["gate_status"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
