#!/usr/bin/env python3
"""Gate Decision 54 against ORCA2's compiled ``dynldf_lev`` statements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round11_dynldf_operator_gate as r11,
)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _live_production_ldf(deck_root: Path, record_root: Path, kt: int):
    """Return the production literal LDF term on NEMO's recorded entry."""
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_ldf_metric_reciprocals_cgrid,
        nemo_qco_live_vorticity_e3f_cgrid,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    _, card = ladder.card_fields(deck_root)
    cfg = card.recipe.model_config
    require(cfg.momentum_time_integrator == "rk3_ws", "ORCA2 no longer uses WS-RK3")
    require(cfg.lateral_viscosity_e3_weighting == "nemo_e3", "ORCA2 lost NEMO e3 weighting")
    require(cfg.lateral_viscosity_coefficient_source == "nemo_ahm_3d_file", "ORCA2 lost file-read ahm")

    entry = r11.read_entry_frame(record_root, kt)
    state = card.recipe.initial_state
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        u=state.u.replace(data=jnp.asarray(r11._nemo_u_to_legoesm(entry["u"]), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(r11._nemo_v_to_legoesm(entry["v"]), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )
    grid, zc = card.recipe.grid, card.recipe.z_coord
    tmask = jnp.asarray(zc.is_active).astype(jnp.float64)
    umask, vmask = compute_face_masks_3d(tmask, grid)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    e3t = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    e3u, e3v, _, _ = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, umask, vmask, grid, z_coord=zc)
    e3f = nemo_qco_live_vorticity_e3f_cgrid(
        state.eta.data, zc, state.eta.data.dtype, grid=grid,
        e3t_0=h_ref, tmask=tmask)
    bundle = (e3t, e3u, e3v, e3f, e3u, e3v)
    metric_reciprocals = nemo_ldf_metric_reciprocals_cgrid(
        zc, grid, state.eta.data.dtype)

    model = LatLonCGridOceanModel(grid, zc, cfg)
    result = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        ldf_state=(state.T.data, state.S.data, state.u.data, state.v.data),
        ldf_thickness_operands=bundle,
        ldf_metric_reciprocal_operands=metric_reciprocals,
        return_nemo_operator_components=True)
    components = result[2]
    du = np.asarray(components["ldf_u"].data, dtype=np.float64)[:, 1:, :r11.NZ]
    dv = np.asarray(components["ldf_v"].data, dtype=np.float64)[1:, :, :r11.NZ]
    live = {
        "e3t": np.asarray(e3t, dtype=np.float64)[:, :, :r11.NZ],
        "e3u": np.asarray(e3u, dtype=np.float64)[:, 1:, :r11.NZ],
        "e3v": np.asarray(e3v, dtype=np.float64)[1:, :, :r11.NZ],
        "e3f": np.asarray(e3f, dtype=np.float64)[1:, 1:, :r11.NZ],
        "r1_area_f": np.asarray(
            metric_reciprocals[1], dtype=np.float64)[1:, 1:],
    }
    return du, dv, card, entry, live


def run(deck_root: Path, record_root: Path, kt: int, *, plant: bool) -> dict:
    stamp = worktree_stamp()
    mesh = r11._stitch(
        record_root, "mesh_mask_{rank:04d}.nc",
        ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
         "e3t_0", "e3u_0", "e3v_0", "e3f_0", "tmask", "umask", "vmask"))
    coeff = r11._stitch(
        record_root, "output.init_{rank:04d}.nc", ("ahmt", "ahmf"))
    candidate_u, candidate_v, card, entry, live = _live_production_ldf(
        deck_root, record_root, kt)
    if plant:
        candidate_u = r11._plant_one_value(candidate_u)
    oracle_u, oracle_v = r11.nemo_dynldf_lev_lap_rot(
        mesh, coeff["ahmt"], coeff["ahmf"], entry["u"], entry["v"], entry["ssh"])
    live_u, live_v = r11.nemo_dynldf_lev_lap_rot(
        mesh, coeff["ahmt"], coeff["ahmf"], entry["u"], entry["v"], entry["ssh"],
        thickness_override=(live["e3u"], live["e3v"], live["e3f"]),
        e3t_override=live["e3t"])
    weight_u = mesh["umask"] * np.isfinite(oracle_u)
    weight_v = mesh["vmask"] * np.isfinite(oracle_v)
    score_u = r11.score(candidate_u, oracle_u, weight_u)
    score_v = r11.score(candidate_v, oracle_v, weight_v)

    raw = card.recipe.z_coord.nemo_een_barotropic
    expected_r1_area = 1.0 / (
        np.asarray(raw.e1f) * np.asarray(raw.e2f))
    area_exact = np.array_equal(live["r1_area_f"], expected_r1_area)
    predictions = {
        "R24-P1": True,
        "R24-P2": score_u["bit_identical"] and score_v["bit_identical"],
        "R24-P7": not plant,
    }
    status = "AT-BAR" if all(predictions.values()) and area_exact else "DEBT"
    return {
        "status": status,
        "claim_label": f"given NEMO's entry (kt={kt} recorded state)",
        "worktree": stamp,
        "card": card.case,
        "dtype": str(card.recipe.grid.area_q.dtype),
        "compiled_citations": {
            "file_read": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353",
            "single_mask": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393",
            "curl_and_area": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125",
            "live_thickness": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:127-140",
        },
        "vertex_area_reciprocal": {
            "cells": int(expected_r1_area.size),
            "unequal": int(np.count_nonzero(
                live["r1_area_f"] != expected_r1_area)),
            "bit_identical": area_exact,
        },
        "u_momentum": score_u,
        "v_momentum": score_v,
        "live_operand_replay_u": r11.score(candidate_u, live_u, weight_u),
        "live_operand_replay_v": r11.score(candidate_v, live_v, weight_v),
        "predictions": {key: ("CONFIRMED" if value else "REFUTED")
                        for key, value in predictions.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--kt", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.kt, plant=args.plant)
    except (GateError, OSError, ValueError, RuntimeError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 0 if result["status"] == "AT-BAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
