#!/usr/bin/env python3
"""Gate the rung-0 seven-array external-mode boundary association."""

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
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round98_coriolis_residual as r98,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round129_substep2_walk as r129,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


PLANTS = ("none", "observer-bit", "post-bit", "registry", "scope-bit")
STATE_FIELDS = ("T", "S", "u", "v", "eta", "uu_b", "vv_b")
POST_FIELDS = (
    ("u", "boundary_post_u", "j001_ua_new", "u"),
    ("v", "boundary_post_v", "j001_va_new", "v"),
    ("depth_u", "boundary_post_depth_u", "j001_hu_e", "u"),
    ("depth_v", "boundary_post_depth_v", "j001_hv_e", "v"),
    ("inverse_u", "boundary_post_inverse_u", "j001_hur_e", "u"),
    ("inverse_v", "boundary_post_inverse_v", "j001_hvr_e", "v"),
    ("eta", "boundary_post_eta", "j001_ssha_e", "t"),
)


class GateError(RuntimeError):
    """The record or causal control cannot support the registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def exact_row(candidate, reference) -> dict:
    """Report bit identity without allowing a magnitude-only zero pass."""

    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    require(candidate.shape == reference.shape, "exact-row shape mismatch")
    unequal = (
        np.ascontiguousarray(candidate).view(np.uint64)
        != np.ascontiguousarray(reference).view(np.uint64)
    )
    locations = np.argwhere(unequal)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "first_unequal_index": (
            None if locations.size == 0 else list(map(int, locations[0]))
        ),
        "maximum_absolute": float(np.max(np.abs(candidate - reference))),
    }


def boundary_scope(pre, post, face: str) -> dict:
    """Require the compact association to touch boundary storage only."""

    pre = np.asarray(pre)
    post = np.asarray(post)
    require(pre.shape == post.shape, "boundary-scope shape mismatch")
    changed = (
        np.ascontiguousarray(pre).view(np.uint64)
        != np.ascontiguousarray(post).view(np.uint64)
    )
    allowed = np.zeros(pre.shape, dtype=bool)
    if face == "u":
        allowed[:, 0] = True
    elif face == "v":
        allowed[0] = True
        allowed[-1] = True
    elif face != "t":
        raise GateError(f"unknown boundary-scope face {face}")
    outside = changed & ~allowed
    return {
        "changed_cells": int(np.count_nonzero(changed)),
        "outside_allowed_cells": int(np.count_nonzero(outside)),
        "first_outside_index": (
            None if not np.any(outside)
            else list(map(int, np.argwhere(outside)[0]))
        ),
    }


def validate_post_registry(registry) -> None:
    """Pin the compiled call's seven-array order."""

    require(tuple(registry) == tuple(row[0] for row in POST_FIELDS),
            "seven-array post-association registry reordered")


def run_known_answer_plant(plant: str) -> None:
    """Exercise one non-degenerate refusal before the expensive JIT run."""

    if plant == "observer-bit":
        reference = np.array([0.25], dtype=np.float64)
        candidate = np.nextafter(reference, np.float64(np.inf))
        require(not exact_row(candidate, reference)["bit_exact"],
                "observer-bit plant stayed green")
        raise GateError("observer-bit plant fired")
    if plant == "post-bit":
        reference = np.array([1.0], dtype=np.float64)
        candidate = np.nextafter(reference, np.float64(np.inf))
        require(exact_row(candidate, reference)["differing_cells"] == 1,
                "post-bit plant stayed green")
        raise GateError("post-bit plant fired")
    if plant == "scope-bit":
        pre = np.zeros((3, 4), dtype=np.float64)
        post = pre.copy()
        post[1, 2] = 1.0
        require(boundary_scope(pre, post, "u")["outside_allowed_cells"] == 1,
                "scope-bit plant stayed green")
        raise GateError("scope-bit plant fired")


def _state_arrays(state) -> dict[str, np.ndarray]:
    arrays = {}
    for name in STATE_FIELDS:
        value = getattr(state, name)
        require(value is not None, f"state field {name} unexpectedly absent")
        arrays[name] = np.asarray(value.data)
    return arrays


def _run(card, state, freshwater, surface, slow, raw_history, *, expose, arm):
    import jax

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=expose,
            expose_barotropic_boundary_association=expose,
            barotropic_slow_forcing_override=slow,
            barotropic_raw_history_override=raw_history,
            barotropic_external_mode_association=arm,
        ),
    )
    return jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))


def measure(
    deck_root: Path,
    frame_root: Path,
    spg_root: Path,
    coefficient_root: Path,
    expect_commit: str,
    *,
    plant: str,
) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-146 measurement worktree is dirty")
    require(
        stamp["commit"].lower() == expect_commit.lower(),
        "round-146 commit stamp mismatch",
    )
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-146 gate requires production JIT on CPU")

    post_registry = [row[0] for row in POST_FIELDS]
    if plant == "registry":
        post_registry[0], post_registry[1] = post_registry[1], post_registry[0]
    validate_post_registry(post_registry)
    run_known_answer_plant(plant)

    oracle, record_census = r97.assemble_record(spg_root)
    oracle_coeff, coefficient_census = r98.assemble_oracle_coefficients(
        coefficient_root)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)
    slow = (
        r97._to_model_u(oracle["i000_zu_frc"]),
        r97._to_model_v(oracle["i000_zv_frc"]),
    )
    raw_history = (
        r97._to_model_u(oracle["i000_ub_e"]),
        r97._to_model_u(oracle["i000_ubb_e"]),
        r97._to_model_v(oracle["i000_vb_e"]),
        r97._to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"],
        oracle["i000_sshbb_e"],
    )

    ordinary = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=False, arm=False)
    observed = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=False)
    arm = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=True)

    observed_state = _state_arrays(observed.state_after)
    ordinary_state = _state_arrays(ordinary)
    passivity = {
        name: exact_row(observed_state[name], ordinary_state[name])
        for name in STATE_FIELDS
    }
    require(all(row["bit_exact"] for row in passivity.values()),
            "WRITE-only observer moved the ordinary production state")

    trace = observed.substeps
    arm_trace = arm.substeps
    require(trace["eta_entry"].shape[0] == 65,
            "production trace does not contain 65 substeps")

    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    coefficient_rows = {
        name: exact_row(np.asarray(trace[name][0]), oracle_coeff[name])
        for name in r98.COEFFICIENTS
    }
    require(all(row["bit_exact"] for row in coefficient_rows.values()),
            "admitted literal EEN coefficients moved")

    post_rows = {}
    scope_rows = {}
    pre_keys = {
        "u": "u_exit", "v": "v_exit",
        "depth_u": "face_depth_u_exit", "depth_v": "face_depth_v_exit",
        "inverse_u": "r1_face_depth_u_exit",
        "inverse_v": "r1_face_depth_v_exit", "eta": "eta_exit",
    }
    for index, (name, trace_name, oracle_name, face) in enumerate(POST_FIELDS):
        candidate_model = np.asarray(trace[trace_name][0])
        candidate = (
            r97._native_u(candidate_model) if face == "u"
            else r97._native_v(candidate_model) if face == "v"
            else candidate_model
        )
        post_rows[name] = exact_row(candidate, oracle[oracle_name])
        post_scope = np.array(candidate_model, copy=True)
        scope_rows[name] = boundary_scope(
            trace[pre_keys[name]][0], post_scope, face)

    area = np.asarray(card.recipe.grid.area)
    baseline_rows = []
    arm_rows = []
    for index in range(2):
        baseline_rows.extend(r129._score_substep(
            trace, oracle, active, area, index, plant="none"))
        arm_rows.extend(r129._score_substep(
            arm_trace, oracle, active, area, index, plant="none"))
    baseline_first = r129.first_nonbit(baseline_rows)
    arm_first = r129.first_nonbit(arm_rows)
    require(baseline_first is not None and arm_first is not None,
            "source-order discriminator unexpectedly has no debt")

    baseline_target = {
        row["boundary"]: row for row in baseline_rows
        if row["substep"] == 2
        and row["boundary"] in ("continuity_du", "after_ssh")
    }
    arm_target = {
        row["boundary"]: row for row in arm_rows
        if row["substep"] == 2
        and row["boundary"] in ("continuity_du", "after_ssh")
    }
    p2 = (
        sum(row["changed_cells"] for row in scope_rows.values()) > 0
        and all(row["outside_allowed_cells"] == 0
                for row in scope_rows.values())
    )
    p3 = all(row["bit_exact"] for row in post_rows.values())
    p4 = all(row["bit_exact"] for row in arm_target.values())
    return {
        "status": "MEASURED_R146_BOUNDARY_ASSOCIATION",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": record_census,
        "coefficient_census": coefficient_census,
        "passivity": passivity,
        "coefficient_rows": coefficient_rows,
        "post_association_rows": post_rows,
        "boundary_scope_rows": scope_rows,
        "baseline_first_non_bit": baseline_first,
        "arm_first_non_bit": arm_first,
        "baseline_target_rows": baseline_target,
        "arm_target_rows": arm_target,
        "predictions": {
            "R146-P1": "CONFIRMED",
            "R146-P2": "CONFIRMED" if p2 else "REFUTED",
            "R146-P3": "CONFIRMED" if p3 else "REFUTED",
            "R146-P4": (
                "CONFIRMED" if p3 and p4
                else "REFUTED" if p3
                else "UNMEASURED_PREREQUISITE_R146-P3"
            ),
        },
        "baseline_rows": baseline_rows,
        "arm_rows": arm_rows,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--coefficient-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.frame_root, args.spg_root,
            args.coefficient_root, args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, rhs_walk.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R146_BOUNDARY_ASSOCIATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
