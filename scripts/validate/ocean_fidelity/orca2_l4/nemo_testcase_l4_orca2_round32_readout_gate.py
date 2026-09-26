#!/usr/bin/env python3
"""Round 32 gates: EEN landing outcome, LDF replay, and ``hf_0`` read-out."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def score(left, right) -> dict:
    left = np.ascontiguousarray(np.asarray(left, np.float64))
    right = np.ascontiguousarray(np.asarray(right, np.float64))
    require(left.shape == right.shape, f"shape mismatch {left.shape} != {right.shape}")
    require(np.isfinite(left).all() and np.isfinite(right).all(),
            "non-finite value in comparison")
    unequal = left.view(np.uint64) != right.view(np.uint64)
    delta = np.abs(left - right)
    return {
        "cells": int(left.size),
        "unequal": int(unequal.sum()),
        "bit_identical": not bool(unequal.any()),
        "max_abs": float(delta.max(initial=0.0)),
    }


def _policy() -> None:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 32 is CPU-only")


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text())


def _row(document: dict, kt: int, checkpoint: str, field: str) -> dict:
    for item in document["candidate_trajectory"]["checkpoints"]:
        if item["kt"] == kt and item["checkpoint"] == checkpoint:
            return item["rows"][field]
    raise GateError(f"missing score kt={kt}:{checkpoint}:{field}")


EXPECTED_KT10 = {"u": 0.4230544199344075, "v": 0.6838675521949865}


def evaluate_outcome(parent_path: Path, arm_path: Path,
                     *, plant: str | None = None) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round25_outcome_gate as r25,
    )

    parent = _read(parent_path)
    arm = _read(arm_path)
    require(parent["status"] == arm["status"] == "LADDER_MEASURED",
            "both ORCA2 arms must be LADDER_MEASURED")
    require(len(parent["candidate_trajectory"]["checkpoints"]) == 40,
            "parent is not the ten-step/40-checkpoint ladder")
    if plant == "truncate":
        arm = copy.deepcopy(arm)
        arm["candidate_trajectory"]["checkpoints"] = (
            arm["candidate_trajectory"]["checkpoints"][:12])
    elif plant == "velocity":
        arm = copy.deepcopy(arm)
        _row(arm, 10, "stage3", "u")["max_abs"] = 1.0

    comparison = r25._compare(parent, arm, "e3f0vor_zero_substitution")
    velocities = {
        field: {
            "parent": float(_row(parent, 10, "stage3", field)["max_abs"]),
            "arm": float(_row(arm, 10, "stage3", field)["max_abs"]),
            "expected": expected,
        }
        for field, expected in EXPECTED_KT10.items()
    }
    expected_shape = (
        comparison["checkpoint_count"] == 40
        and comparison["moved_row_count"] == 80
        and comparison["toward"] == 1
        and comparison["away"] == 2
        and comparison["same_maximum"] == 77
        and comparison["first_moved_row"] == "kt=5:stage2:S"
        and comparison["at_bar_rows_left"] == []
        and comparison["first_non_bit_statement_unchanged"]
    )
    expected_velocity = all(
        row["arm"] == row["expected"] for row in velocities.values())
    status = "LANDED" if expected_shape and expected_velocity else "HELD"
    reasons = []
    if not expected_shape:
        reasons.append("the registered 80-row ladder outcome changed")
    if not expected_velocity:
        reasons.append("the registered kt=10 velocity values changed")
    return {
        "status": status,
        "claim_label": "independent with Decision-52 SSH",
        "parent_commit": parent["worktree"]["commit"],
        "arm_commit": arm["worktree"]["commit"],
        "ladder": {key: value for key, value in comparison.items()
                   if key != "moved_rows"},
        "moved_rows": comparison["moved_rows"],
        "kt10_stage3_velocity_maxima": velocities,
        "reasons": reasons,
        "plant": plant,
        "worktree": worktree_stamp(),
    }


def capture_ldf_replay(deck_root: Path, record_root: Path,
                       *, plant: bool = False) -> dict:
    """Re-run round 24's literal compiled LDF replay on the landed routing."""
    import jax.numpy as jnp
    from jax import lax

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_ldf_reference_e3f,
        nemo_qco_live_vorticity_e3f_cgrid,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round11_dynldf_operator_gate as r11,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    _policy()
    _, card = ladder.card_fields(deck_root)
    cfg = card.recipe.model_config
    entry = r11.read_entry_frame(record_root, 2)
    state = card.recipe.initial_state._replace(
        T=card.recipe.initial_state.T.replace(
            data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=card.recipe.initial_state.S.replace(
            data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        u=card.recipe.initial_state.u.replace(data=jnp.asarray(
            r11._nemo_u_to_legoesm(entry["u"]), dtype=jnp.float64)),
        v=card.recipe.initial_state.v.replace(data=jnp.asarray(
            r11._nemo_v_to_legoesm(entry["v"]), dtype=jnp.float64)),
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )
    grid, zc = card.recipe.grid, card.recipe.z_coord
    tmask = jnp.asarray(zc.is_active, dtype=jnp.float64)
    umask, vmask = compute_face_masks_3d(tmask, grid)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    e3t = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    e3u, e3v, _, _ = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, umask, vmask, grid)
    e3f = nemo_qco_live_vorticity_e3f_cgrid(
        state.eta.data, zc, state.eta.data.dtype, grid=grid,
        e3t_0=h_ref, tmask=tmask, reference_e3f=nemo_ldf_reference_e3f(zc))
    bundle = (e3t, e3u, e3v, e3f, e3u, e3v)
    # Recreate round 24's six *stored* NEMO metric reciprocals locally.  The
    # experimental production seam that once built these was correctly
    # removed; this read-only replay still needs the exact operands in order
    # to compare the landed arithmetic with the compiled statements.
    raw = zc.nemo_een_barotropic
    barrier = lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=state.eta.data.dtype)

    def reciprocal(value):
        value = barrier(jnp.asarray(value, dtype=state.eta.data.dtype))
        return barrier(jnp.where(value > 0.0, one / value, 0.0))

    def west(value):
        return jnp.concatenate([value[:, -1:], value], axis=1)

    def south(value):
        return jnp.concatenate([jnp.zeros_like(value[:1]), value], axis=0)

    r1_t = reciprocal(barrier(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)))
    r1_f_native = reciprocal(
        barrier(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f)))
    with_south = jnp.concatenate([r1_f_native[:1], r1_f_native], axis=0)
    metric = (
        r1_t,
        west(with_south),
        west(reciprocal(raw.e1u)),
        south(reciprocal(raw.e2v)),
        west(reciprocal(raw.e2u)),
        south(reciprocal(raw.e1v)),
    )
    model = LatLonCGridOceanModel(grid, zc, cfg)
    result = model.tendencies(
        state, dt=card.dt_s, momentum_only=True,
        ldf_state=(state.T.data, state.S.data, state.u.data, state.v.data),
        ldf_thickness_operands=bundle,
        ldf_metric_reciprocal_operands=metric,
        return_nemo_operator_components=True)
    candidate_u = np.asarray(
        result[2]["ldf_u"].data, np.float64)[:, 1:, :r11.NZ]
    candidate_v = np.asarray(
        result[2]["ldf_v"].data, np.float64)[1:, :, :r11.NZ]
    if plant:
        candidate_u = r11._plant_one_value(candidate_u)

    mesh = r11._stitch(
        record_root, "mesh_mask_{rank:04d}.nc",
        ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
         "e3t_0", "e3u_0", "e3v_0", "e3f_0", "tmask", "umask", "vmask"))
    coeff = r11._stitch(
        record_root, "output.init_{rank:04d}.nc", ("ahmt", "ahmf"))
    oracle_u, oracle_v = r11.nemo_dynldf_lev_lap_rot(
        mesh, coeff["ahmt"], coeff["ahmf"], entry["u"], entry["v"],
        entry["ssh"])
    score_u = r11.score(candidate_u, oracle_u,
                        mesh["umask"] * np.isfinite(oracle_u))
    score_v = r11.score(candidate_v, oracle_v,
                        mesh["vmask"] * np.isfinite(oracle_v))
    status = "PASS" if score_u["bit_identical"] and score_v["bit_identical"] else "HELD"
    return {
        "status": status,
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "u_momentum": score_u,
        "v_momentum": score_v,
        "plant": plant,
        "worktree": worktree_stamp(),
        "citations": {
            "file_read": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353",
            "single_mask": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393",
            "operator": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140",
        },
    }


def _hf0_builder(original, *, use_carried_hf0: bool, hf0_override=None):
    def builder(eta, z_coord, dtype, nn_e3f_typ=0, *, grid=None,
                e3t_0=None, tmask=None, reference_e3f=None):
        if not use_carried_hf0:
            return original(
                eta, z_coord, dtype, nn_e3f_typ, grid=grid, e3t_0=e3t_0,
                tmask=tmask, reference_e3f=reference_e3f)

        import jax.numpy as jnp
        from jax import lax

        from legoesm.grids.latlon import ensure_geometry
        from legoesm.ocean.vertical import (
            nemo_dynvor_e3f_0vor,
            nemo_fe3mask_from_tmask,
            nemo_t_fold_f_owned,
        )

        raw = getattr(z_coord, "nemo_een_barotropic", None)
        require(raw is not None and getattr(raw, "hf_0", None) is not None,
                "the card carries no NEMO hf_0 operand")
        if e3t_0 is None:
            e3t_0 = getattr(z_coord, "nemo_e3t_0", None)
        if tmask is None:
            tmask = getattr(z_coord, "is_active", None)
        require(e3t_0 is not None and tmask is not None and grid is not None,
                "literal NEMO e3f_vor requires e3t_0, tmask, and grid")
        geom = ensure_geometry(grid)
        b = lax.optimization_barrier
        one = jnp.asarray(1.0, dtype=dtype)
        quarter = jnp.asarray(0.25, dtype=dtype)
        eta = jnp.asarray(eta, dtype=dtype)
        e3t0 = jnp.asarray(e3t_0, dtype=dtype)
        tmask = jnp.asarray(tmask, dtype=dtype)

        def east(value):
            return jnp.roll(value, -1, axis=1)

        def north(value):
            return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

        mesh_e3f = getattr(raw, "e3f_0", None)
        e3f0vor = nemo_dynvor_e3f_0vor(
            e3t0, tmask, grid=grid, dtype=dtype, nn_e3f_typ=nn_e3f_typ,
            substitute_e3f=mesh_e3f)
        area_eta = b(jnp.asarray(geom.area_T, dtype=dtype) * eta)
        area_eta_n = north(area_eta)
        quad = b(b(area_eta + east(area_eta))
                 + b(area_eta_n + east(area_eta_n)))
        fe3mask = nemo_fe3mask_from_tmask(tmask, grid=grid)
        hf0 = jnp.asarray(
            raw.hf_0 if hf0_override is None else hf0_override, dtype=dtype)
        wet_f = (hf0 > 0.0).astype(dtype)
        r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
        area_f = b(jnp.asarray(geom.area_q[1:, 1:], dtype=dtype))
        r3f = b(b(quarter * quad) * r1_hf0 / area_f)
        r3f = nemo_t_fold_f_owned(r3f, grid)
        reference = (e3f0vor if reference_e3f is None
                     else jnp.asarray(reference_e3f, dtype=dtype))
        native = b(reference * b(one + r3f[..., None] * fe3mask))
        with_south = jnp.concatenate([native[:1], native], axis=0)
        return jnp.concatenate([with_south[:, -1:], with_south], axis=1)

    return builder


def _capture_een_variant(deck_root: Path, record_root: Path,
                         *, use_carried_hf0: bool, hf0_override=None) -> dict:
    import jax

    from legoesm.ocean import vertical
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round27_consumer_gate as r27,
    )

    calls = []
    original_builder = vertical.nemo_qco_live_vorticity_e3f_cgrid
    original_flux = pe.pv_flux_al81_partial_cell

    def observed(zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d,
                 vtx_mask, f_vtx=None, eps_h=1.0e-10,
                 q_boundary="neumann_fill", metric_widths=None):
        result = original_flux(
            zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask,
            f_vtx=f_vtx, eps_h=eps_h, q_boundary=q_boundary,
            metric_widths=metric_widths)

        def save(denominator, out_u, out_v):
            calls.append((np.asarray(denominator, np.float64).copy(),
                          np.asarray(out_u, np.float64).copy(),
                          np.asarray(out_v, np.float64).copy()))

        jax.debug.callback(save, h_vtx, result[0], result[1], ordered=True)
        return result

    vertical.nemo_qco_live_vorticity_e3f_cgrid = _hf0_builder(
        original_builder, use_carried_hf0=use_carried_hf0,
        hf0_override=hf0_override)
    pe.pv_flux_al81_partial_cell = observed
    try:
        exposed_u, exposed_v = r27._stage2_vorticity(deck_root, record_root)
    finally:
        vertical.nemo_qco_live_vorticity_e3f_cgrid = original_builder
        pe.pv_flux_al81_partial_cell = original_flux
    require(calls, "the production EEN observer captured no calls")
    return {"calls": calls, "exposed_u": exposed_u, "exposed_v": exposed_v}


def capture_hf0(deck_root: Path, record_root: Path,
                *, plant: bool = False) -> dict:
    import jax.numpy as jnp

    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_dynvor_e3f_0vor,
        nemo_fe3mask_from_tmask,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    _policy()
    _, card = ladder.card_fields(deck_root)
    zc = card.recipe.z_coord
    raw = zc.nemo_een_barotropic
    tmask = jnp.asarray(zc.is_active, jnp.float64)
    state = card.recipe.initial_state
    e3t0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, zc,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    e3f0vor = nemo_dynvor_e3f_0vor(
        e3t0, tmask, grid=card.recipe.grid, dtype=jnp.float64,
        substitute_e3f=raw.e3f_0)
    fe3mask = nemo_fe3mask_from_tmask(tmask, grid=card.recipe.grid)
    reconstructed = np.asarray(jnp.sum(e3f0vor * fe3mask, axis=-1), np.float64)
    carried = np.asarray(raw.hf_0, np.float64)
    selected_hf0 = reconstructed if plant else carried
    parent = _capture_een_variant(
        deck_root, record_root, use_carried_hf0=False)
    arm = _capture_een_variant(
        deck_root, record_root, use_carried_hf0=True,
        hf0_override=selected_hf0)
    if plant:
        carried = reconstructed
    hf_score = score(carried, reconstructed)
    require(hf_score["unequal"] > 0,
            "carried hf_0 equals the reconstructed column depth; the read-out is vacuous")
    first = {
        "denominator": score(arm["calls"][0][0], parent["calls"][0][0]),
        "output_u": score(arm["calls"][0][1], parent["calls"][0][1]),
        "output_v": score(arm["calls"][0][2], parent["calls"][0][2]),
    }
    exposed = {"u": score(arm["exposed_u"], parent["exposed_u"]),
               "v": score(arm["exposed_v"], parent["exposed_v"])}
    thresholds = {"u": 5.153126997217792e-09,
                  "v": 2.5500881043307236e-09}
    prediction = (
        first["output_u"]["bit_identical"]
        and first["output_v"]["bit_identical"]
        and exposed["u"]["max_abs"] <= thresholds["u"]
        and exposed["v"]["max_abs"] <= thresholds["v"])
    return {
        "status": "PASS",
        "claim_label": "independent with Decision-52 SSH",
        "hf_0_carried_vs_reconstructed": hf_score,
        "first_raw_een_call": first,
        "exposed_stage2": exposed,
        "prediction_R32_P4": "CONFIRMED" if prediction else "REFUTED",
        "plant": plant,
        "worktree": worktree_stamp(),
        "citations": {
            "hf_0": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206",
            "r3f": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286",
            "consumer": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    outcome = sub.add_parser("outcome")
    outcome.add_argument("--parent", type=Path, required=True)
    outcome.add_argument("--arm", type=Path, required=True)
    outcome.add_argument("--plant", choices=("truncate", "velocity"))
    replay = sub.add_parser("ldf-replay")
    replay.add_argument("--deck-root", type=Path, required=True)
    replay.add_argument("--record-root", type=Path, required=True)
    replay.add_argument("--plant", action="store_true")
    hf0 = sub.add_parser("hf0")
    hf0.add_argument("--deck-root", type=Path, required=True)
    hf0.add_argument("--record-root", type=Path, required=True)
    hf0.add_argument("--plant", action="store_true")
    for child in (outcome, replay, hf0):
        child.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "outcome":
            result = evaluate_outcome(args.parent, args.arm, plant=args.plant)
            success = result["status"] == "LANDED"
        elif args.command == "ldf-replay":
            result = capture_ldf_replay(
                args.deck_root, args.record_root, plant=args.plant)
            success = result["status"] == "PASS"
        else:
            result = capture_hf0(
                args.deck_root, args.record_root, plant=args.plant)
            success = result["status"] == "PASS"
    except (GateError, OSError, ValueError, RuntimeError) as exc:
        print(f"GATE REFUSED: {exc}")
        return 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
