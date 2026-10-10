#!/usr/bin/env python3
"""Offline source-order walk of OMT-4's northern-fold tracer fluxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)

PLANTS = ("none", "centred-wall", "source-order", "support", "sufficiency")
TRACE_ORDER = (
    "first_u_raw", "first_v_raw", "first_w_raw", "first_div", "midpoint",
    "average_u_raw", "average_v_raw", "average_w_raw", "explicit_ztra",
    "implicit_ztra", "total_ztra", "base_content", "dt_ztra", "numerator",
    "after_thickness", "paft",
)


class GateError(RuntimeError):
    """The passive fold replay no longer supports its source-order verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _row(left, right, support=None) -> dict[str, object]:
    left = np.asarray(left)
    right = np.asarray(right)
    require(left.shape == right.shape and left.dtype == right.dtype,
            "comparison shape/dtype moved")
    if support is None:
        support = np.ones(left.shape, dtype=bool)
    else:
        support = np.broadcast_to(np.asarray(support, dtype=bool), left.shape)
    unequal = (left.view(np.uint64) != right.view(np.uint64)) & support
    delta = np.abs(left - right)
    return {
        "support": int(np.count_nonzero(support)),
        "unequal": int(np.count_nonzero(unequal)),
        "maximum_absolute": (float(np.max(delta[support]))
                             if np.any(support) else 0.0),
        "bit_exact": not bool(np.any(unequal)),
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "centred-wall":
        report["centred_stages"]["stage1"]["T"]["literal_bit_exact"] = False
    elif plant == "source-order":
        report["stage3"]["first_moved_field"] = "average_v_raw"
    elif plant == "support":
        report["stage3"]["active_fold_support"] = 0
    elif plant == "sufficiency":
        report["statement_sufficiency"] = "QUALIFIED"

    require(report.get("execution") == "offline-pure-jit-cpu-fp64-libm",
            "execution policy moved")
    require(report.get("record_scope") == "admitted OMT-4 kt=1 completed states",
            "passive-state source moved")
    require(report.get("in_executable_observers") == 0,
            "an in-executable observer entered the walk")
    for stage in ("stage1", "stage2"):
        for tracer in ("T", "S"):
            row = report["centred_stages"][stage][tracer]
            require(row["literal_bit_exact"],
                    f"{stage} {tracer}: live centred fold differs from literal")
            require(row["wall_control_unequal"] > 0,
                    f"{stage} {tracer}: wall control is vacuous")

    stage3 = report["stage3"]
    require(stage3["active_fold_support"] > 0,
            "stage-3 fold support is empty")
    require(stage3["first_moved_field"] == "first_v_raw",
            "first moved FCT statement is not the donor-cell V flux")
    require(stage3["first_u_raw"]["bit_exact"],
            "an earlier donor-cell U statement moved")
    require(stage3["first_w_raw"]["bit_exact"],
            "the vertical donor-cell statement moved")
    require(stage3["first_v_raw"]["unequal"] > 0,
            "donor-cell V substitution is inert")
    require(stage3["average_v_raw"]["unequal"] > 0,
            "midpoint donor-cell V substitution is inert")
    require(stage3["off_fold_unequal"] == 0,
            "donor-cell substitution moved a non-fold face")
    require(report.get("statement_sufficiency") == "UNMEASURED_WITH_SPEC",
            "offline attribution manufactured a landing verdict")

    report["predictions"] = {
        "R224-P1": "CONFIRMED",
        "R224-P2": "CONFIRMED",
        "R224-P3": "CONFIRMED",
        "R224-P4": "UNMEASURED_WITH_SPEC",
        "R224-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    report["status"] = "PASS_R224_FIRST_STATEMENT_DONOR_V_FOLD"
    return report


def _full_fluxes(frame: dict[str, np.ndarray], raw) -> tuple[np.ndarray, np.ndarray]:
    """Metric-free h*u/h*v from a passive completed stage state."""

    u_native = np.asarray(frame["u"], dtype=np.float64)
    v_native = np.asarray(frame["v"], dtype=np.float64)
    e3u = np.asarray(raw.e3u_0, dtype=np.float64)
    e3v = np.asarray(raw.e3v_0, dtype=np.float64)
    umask = np.asarray(raw.umask, dtype=np.float64)
    vmask = np.asarray(raw.vmask, dtype=np.float64)
    require(u_native.shape == e3u.shape == umask.shape,
            "U passive/raw-mesh shapes moved")
    require(v_native.shape == e3v.shape == vmask.shape,
            "V passive/raw-mesh shapes moved")
    u = np.zeros((u_native.shape[0], u_native.shape[1] + 1, u_native.shape[2]),
                 dtype=np.float64)
    v = np.zeros((v_native.shape[0] + 1, v_native.shape[1], v_native.shape[2]),
                 dtype=np.float64)
    u[:, 1:] = e3u * umask * u_native
    u[:, 0] = u[:, -1]
    v[1:] = e3v * vmask * v_native
    return u, v


def measure(deck_root: Path, frames_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import advection
    from legoesm.ocean.advection import NEMO_FCT_UP1_TRACE_FIELDS
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        diagnose_w_from_flux_div,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-224 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-224 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "fold replay requires production JIT on CPU")
    require(tuple(NEMO_FCT_UP1_TRACE_FIELDS) == TRACE_ORDER,
            "FCT source-order registry moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    grid = card.recipe.grid
    fold = grid.fold
    require(fold is not None and fold.is_active and fold.pivot_row_stored,
            "OMT-4 is not the admitted stored T-pivot mesh")
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 raw mesh operands are absent")

    frames = {stage: rung0.assemble_frame(frames_root, 1, stage)
              for stage in range(4)}
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    raw_fold_mask = np.asarray(raw.vmask, dtype=bool)[-1]

    centred = {}
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points
    for stage, frame_stage in (("stage1", 0), ("stage2", 1)):
        centred[stage] = {}
        for tracer in ("T", "S"):
            field = jnp.asarray(frames[frame_stage][tracer])
            live = np.asarray(jax.jit(
                lambda x: interp_to_v_points(x, grid, nemo_source_sum=True)
            )(field))
            literal_north = np.asarray(
                (field[-2:-1] + field[-1:])[:, fold.perm_T])
            wall = np.asarray(2.0 * advection.centred2_to_v_points(field))
            literal = _row(live[-1:], literal_north)
            control = _row(wall[-1:], literal_north, raw_fold_mask[None, ...])
            centred[stage][tracer] = {
                "literal_bit_exact": literal["bit_exact"],
                "wall_control_unequal": control["unequal"],
                "wall_control_maximum_absolute": control["maximum_absolute"],
            }

    mf_u_np, mf_v_np = _full_fluxes(frames[2], raw)
    fold_support = raw_fold_mask & (mf_v_np[-1] != 0.0)
    require(np.count_nonzero(fold_support) > 0,
            "passive stage-3 state has no active nonzero fold transport")
    h_base = compute_layer_thickness(
        jnp.asarray(frames[0]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_now = compute_layer_thickness(
        jnp.asarray(frames[2]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_after = compute_layer_thickness(
        jnp.asarray(frames[3]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    mf_u, mf_v = jnp.asarray(mf_u_np), jnp.asarray(mf_v_np)
    w = diagnose_w_from_flux_div(
        divergence_cgrid(mf_u, mf_v, grid), card.recipe.z_coord,
        thickness_weighted=True)
    zero_w = jnp.zeros_like(w)

    original_upwind = pe.upwind_to_v_points

    def run_trace(tracer, before, *, fold_upwind: bool):
        if fold_upwind:
            pe.upwind_to_v_points = (
                lambda value, flux: original_upwind(value, flux, grid=grid))
        else:
            pe.upwind_to_v_points = original_upwind
        try:
            def replay(now_value, before_value):
                return advection.fct_tracer_advection(
                    now_value, mf_u, mf_v, w, h_now, grid, card.dt_s,
                    high_order="centred2", tracer_before=before_value,
                    active_mask=jnp.asarray(active),
                    low_order_predictor="nemo_rk3_two_step",
                    base_thickness=h_base, after_thickness=h_after,
                    implicit_w=zero_w, return_nemo_up1_trace=True,
                )[2]
            return tuple(np.asarray(value) for value in jax.jit(replay)(
                jnp.asarray(tracer), jnp.asarray(before)))
        finally:
            pe.upwind_to_v_points = original_upwind

    trace_rows = {}
    first_moved = None
    off_fold_unequal = 0
    for tracer in ("T", "S"):
        current = run_trace(frames[2][tracer], frames[0][tracer],
                            fold_upwind=False)
        folded = run_trace(frames[2][tracer], frames[0][tracer],
                           fold_upwind=True)
        rows = {name: _row(a, b) for name, a, b in
                zip(TRACE_ORDER, current, folded, strict=True)}
        trace_rows[tracer] = rows
        if first_moved is None:
            first_moved = next((name for name in TRACE_ORDER
                                if rows[name]["unequal"]), None)
        # Both V-face traces may move on the north fold only.  Check the first
        # donor pass explicitly; later cell-centred fields legitimately spread
        # its divergence onto the adjacent T row.
        current_first_v = current[TRACE_ORDER.index("first_v_raw")]
        folded_first_v = folded[TRACE_ORDER.index("first_v_raw")]
        changed = current_first_v.view(np.uint64) != folded_first_v.view(np.uint64)
        off_fold_unequal += int(np.count_nonzero(changed[:-1]))

    stage3 = {
        "active_fold_support": int(np.count_nonzero(fold_support)),
        "first_moved_field": first_moved,
        "off_fold_unequal": off_fold_unequal,
    }
    for name in ("first_u_raw", "first_v_raw", "first_w_raw", "average_v_raw"):
        combined = {
            "support": sum(trace_rows[t][name]["support"] for t in ("T", "S")),
            "unequal": sum(trace_rows[t][name]["unequal"] for t in ("T", "S")),
            "maximum_absolute": max(trace_rows[t][name]["maximum_absolute"]
                                    for t in ("T", "S")),
        }
        combined["bit_exact"] = combined["unequal"] == 0
        stage3[name] = combined

    return {
        "format": "nemo-testcase-l4-orca2-round224-tracer-fold-v1",
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 completed states",
        "in_executable_observers": 0,
        "centred_stages": centred,
        "stage3": stage3,
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
        "worktree": stamp,
        "compiled_citations": {
            "dispatcher": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540",
            "centred": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90:149-160",
            "first_upstream_v": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:495-539",
            "second_upstream_v": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:562-573",
            "high_order_v": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:191-199",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.frames_root is None
                    and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root and args.frames_root and args.expect_commit,
                    "runtime mode requires deck, frames, and commit")
            raw = measure(args.deck_root, args.frames_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, omt4.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R224_FIRST_STATEMENT_DONOR_V_FOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
