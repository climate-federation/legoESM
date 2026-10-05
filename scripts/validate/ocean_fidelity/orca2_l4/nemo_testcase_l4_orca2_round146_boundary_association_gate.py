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


PLANTS = ("none", "observer-bit", "post-bit", "registry", "scope-bit",
          "u-fold-sign", "stored-pivot-source", "v-depth-bit",
          "inverse-v-registry", "wrong-entry-frame", "entry-inverse-v-bit",
          "midpoint-v-registry", "midpoint-v-bit")
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
        # Compact U stores one periodic closure column before NEMO's native
        # row.  On a T-pivot mesh the U association also rewrites the native
        # pivot row's right half (lbcnfd.f90:665-681).
        half = (pre.shape[1] - 1) // 2
        allowed[-1, half + 1:] = True
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


def entry_inverse_v_names() -> tuple[str, ...]:
    """NEMO hvr_e consumed at each of the 65 external-substep entries."""

    return ("i000_hvr_e",) + tuple(
        f"j{step:03d}_hvr_e" for step in range(1, 65))


def build_entry_inverse_v_override(oracle, *, plant: str):
    """Build the rank-complete carried-hvr sequence from its named frames."""

    names = list(entry_inverse_v_names())
    if plant == "inverse-v-registry":
        names[0], names[1] = names[1], names[0]
    require(tuple(names) == entry_inverse_v_names(),
            "entry inverse V override registry reordered")
    native = [np.array(oracle[name], copy=True) for name in names]
    if plant == "wrong-entry-frame":
        # i000 and j001 are bit-identical on the admitted record, so using
        # i000 here would be a vacuous plant.  j002 is the first distinct
        # recorded frame; fail closed if that precondition ever changes.
        wrong = np.array(native[2], copy=True)
        require(
            not exact_row(wrong, native[1])["bit_exact"],
            "wrong-entry-frame plant source is not distinct",
        )
        native[1] = wrong
    if plant == "entry-inverse-v-bit":
        candidates = np.argwhere(native[1] != 0.0)
        require(candidates.size > 0,
                "entry-inverse-v-bit plant has no nonzero cell")
        location = tuple(map(int, candidates[0]))
        native[1][location] = np.nextafter(
            native[1][location], np.float64(np.inf))
    return np.stack([r97._to_model_v(value) for value in native])


def midpoint_v_operand_names() -> tuple[str, ...]:
    """Compiled dynspg_ts midpoint V-depth operand order."""

    return (
        "midpoint_ssh", "area_t", "local_area_ssh", "north_area_ssh",
        "reference_depth_v", "reciprocal_area_v", "ssvmask",
    )


def midpoint_v_operand_split(card, state, trace, oracle, masks, *, plant: str):
    """Replay and split dynspg_ts.f90:519,542-545 without a second solver."""

    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.grids.operators_latlon_cgrid import fold_ghost_source_T
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_ssh_avg_apply,
        _nemo_ssh_avg_prep,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import fold_vface_row

    names = list(midpoint_v_operand_names())
    if plant == "midpoint-v-registry":
        names[0], names[1] = names[1], names[0]
    require(tuple(names) == midpoint_v_operand_names(),
            "midpoint V operand registry reordered")

    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    raw = z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no raw NEMO V operands")
    eta = jnp.asarray(trace["eta_mid"][1], dtype=jnp.float64)
    area = jnp.asarray(grid.area, dtype=jnp.float64)
    # The shared helper consumes legoESM's compact faces, while the gate's
    # scoring masks are NEMO-native.  Reuse the certified record converters;
    # do not infer the redundant periodic/southern storage here.
    u_mask = jnp.asarray(r97._to_model_u(
        np.max(np.asarray(raw.umask), axis=-1)), dtype=jnp.float64)
    v_mask = jnp.asarray(r97._to_model_v(
        np.max(np.asarray(raw.vmask), axis=-1)), dtype=jnp.float64)
    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data, dtype=jnp.float64),
        jnp.asarray(state.land_mask.data, dtype=jnp.float64),
        grid, jnp.float64,
    )
    replay = _nemo_ssh_avg_apply(
        eta, u_mask, v_mask, grid, area, prep,
        return_literal_inverse=True, return_ssh_average=True,
    )
    model_depth_v = np.asarray(replay[1])
    ssh_average_v = np.asarray(replay[5])
    traced_depth_v = np.asarray(trace["transport_face_depth_v"][1])
    replay_row = exact_row(model_depth_v, traced_depth_v)
    require(replay_row["bit_exact"],
            "midpoint V replay does not reproduce the production trace")

    oracle_eta = np.asarray(oracle["j002_sshp2_mid"])
    raw_area_t = np.asarray(
        nemo_source_round(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)))
    raw_area_v = np.asarray(
        nemo_source_round(jnp.asarray(raw.e1v) * jnp.asarray(raw.e2v)))
    raw_hv = np.asarray(raw.hv_0)
    raw_ssvmask = np.max(np.asarray(raw.vmask), axis=-1)

    model_area_eta = np.asarray(nemo_source_round(area * eta))
    oracle_area_eta = np.asarray(nemo_source_round(
        jnp.asarray(raw_area_t) * jnp.asarray(oracle_eta)))
    model_north = np.concatenate(
        [model_area_eta[1:], np.asarray(fold_vface_row(
            jnp.asarray(model_area_eta), grid))], axis=0)
    fold = grid.fold
    oracle_north = np.concatenate(
        [oracle_area_eta[1:], np.asarray(nemo_source_round(
            fold_ghost_source_T(jnp.asarray(raw_area_t), fold)[:, fold.perm_T]
            * fold_ghost_source_T(jnp.asarray(oracle_eta), fold)[:, fold.perm_T]
        ))], axis=0)

    candidate = {
        "midpoint_ssh": np.asarray(eta),
        "area_t": np.asarray(area),
        "local_area_ssh": model_area_eta,
        "north_area_ssh": model_north,
        "reference_depth_v": r97._native_v(np.asarray(prep[1])),
        "reciprocal_area_v": r97._native_v(np.asarray(prep[3])),
        "ssvmask": r97._native_v(np.asarray(v_mask)),
    }
    reference = {
        "midpoint_ssh": oracle_eta,
        "area_t": raw_area_t,
        "local_area_ssh": oracle_area_eta,
        "north_area_ssh": oracle_north,
        "reference_depth_v": raw_hv,
        "reciprocal_area_v": np.asarray(
            nemo_source_round(1.0 / jnp.asarray(raw_area_v))),
        "ssvmask": raw_ssvmask,
    }
    if plant == "midpoint-v-bit":
        exact_locations = np.argwhere(
            np.ascontiguousarray(candidate["local_area_ssh"]).view(np.uint64)
            == np.ascontiguousarray(reference["local_area_ssh"]).view(np.uint64)
        )
        require(exact_locations.size > 0,
                "midpoint-v-bit plant has no exact source cell")
        location = tuple(map(int, exact_locations[0]))
        planted = np.array(candidate["local_area_ssh"], copy=True)
        planted[location] = np.nextafter(
            planted[location], np.float64(np.inf))
        candidate["local_area_ssh"] = planted

    rows = {name: exact_row(candidate[name], reference[name]) for name in names}
    if plant == "midpoint-v-bit":
        require(rows["local_area_ssh"]["differing_cells"] == 1,
                "midpoint-v-bit plant stayed green")
        raise GateError("midpoint-v-bit plant fired")

    raw_hv_model = r97._to_model_v(raw_hv)
    reference_depth_arm = np.asarray(nemo_source_round(
        jnp.asarray(raw_hv_model) + jnp.asarray(ssh_average_v)))
    arm_row = exact_row(
        r97._native_v(reference_depth_arm), oracle["j002_hvp2_e"])
    control_row = exact_row(
        r97._native_v(model_depth_v), oracle["j002_hvp2_e"])
    return {
        "operand_rows": rows,
        "replay_passivity": replay_row,
        "control_mid_depth_v": control_row,
        "reference_depth_arm_mid_depth_v": arm_row,
    }


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
    if plant == "midpoint-v-registry":
        names = list(midpoint_v_operand_names())
        names[0], names[1] = names[1], names[0]
        require(tuple(names) != midpoint_v_operand_names(),
                "midpoint-v-registry plant stayed green")
        raise GateError("midpoint-v-registry plant fired")
    if plant == "midpoint-v-bit":
        reference = np.array([1.0], dtype=np.float64)
        candidate = np.nextafter(reference, np.float64(np.inf))
        require(exact_row(candidate, reference)["differing_cells"] == 1,
                "midpoint-v-bit plant stayed green")
        raise GateError("midpoint-v-bit plant fired")


def _state_arrays(state) -> dict[str, np.ndarray]:
    arrays = {}
    for name in STATE_FIELDS:
        value = getattr(state, name)
        require(value is not None, f"state field {name} unexpectedly absent")
        arrays[name] = np.asarray(value.data)
    return arrays


def _run(card, state, freshwater, surface, slow, raw_history, *, expose, arm,
         t_pivot_north_neighbor=False, inverse_v_override=None):
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
            barotropic_t_pivot_north_neighbor=t_pivot_north_neighbor,
            barotropic_substep_inverse_v_override=inverse_v_override,
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
    inverse_v_override = build_entry_inverse_v_override(
        oracle, plant=plant)

    ordinary = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=False, arm=False)
    observed = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=False)
    control = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=True, t_pivot_north_neighbor=False)
    arm = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=True,
        t_pivot_north_neighbor=(plant != "stored-pivot-source"))
    inverse_arm = _run(
        card, state, freshwater, surface, slow, raw_history,
        expose=True, arm=True, t_pivot_north_neighbor=False,
        inverse_v_override=inverse_v_override)

    observed_state = _state_arrays(observed.state_after)
    ordinary_state = _state_arrays(ordinary)
    passivity = {
        name: exact_row(observed_state[name], ordinary_state[name])
        for name in STATE_FIELDS
    }
    require(all(row["bit_exact"] for row in passivity.values()),
            "WRITE-only observer moved the ordinary production state")

    trace = observed.substeps
    control_trace = control.substeps
    arm_trace = arm.substeps
    inverse_arm_trace = inverse_arm.substeps
    v_depth_plant_expected = None
    if plant == "v-depth-bit":
        arm_trace = dict(arm_trace)
        planted = np.array(arm_trace["transport_face_depth_v"], copy=True)
        native = r97._native_v(planted[1])
        oracle_depth = oracle["j002_hvp2_e"]
        unequal = (
            np.ascontiguousarray(native).view(np.uint64)
            != np.ascontiguousarray(oracle_depth).view(np.uint64)
        )
        exact_top = np.flatnonzero(~unequal[-1])
        require(exact_top.size > 0,
                "v-depth-bit plant has no exact northern cell to perturb")
        column = int(exact_top[0])
        v_depth_plant_expected = int(np.count_nonzero(unequal)) + 1
        planted[1, -1, column] = np.nextafter(
            planted[1, -1, column], np.float64(np.inf))
        arm_trace["transport_face_depth_v"] = planted
    require(trace["eta_entry"].shape[0] == 65,
            "production trace does not contain 65 substeps")

    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    midpoint_v_split = midpoint_v_operand_split(
        card, state, control_trace, oracle, masks, plant=plant)
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
        if plant == "u-fold-sign" and name == "u":
            candidate = np.array(candidate, copy=True)
            candidate[-1, candidate.shape[1] // 2:] *= -1.0
        post_rows[name] = exact_row(candidate, oracle[oracle_name])
        post_scope = np.array(candidate_model, copy=True)
        scope_rows[name] = boundary_scope(
            trace[pre_keys[name]][0], post_scope, face)

    area = np.asarray(card.recipe.grid.area)
    baseline_rows = []
    control_rows = []
    arm_rows = []
    inverse_arm_rows = []
    for index in range(2):
        baseline_rows.extend(r129._score_substep(
            trace, oracle, active, area, index, plant="none"))
        control_rows.extend(r129._score_substep(
            control_trace, oracle, active, area, index, plant="none"))
        arm_rows.extend(r129._score_substep(
            arm_trace, oracle, active, area, index, plant="none"))
        inverse_arm_rows.extend(r129._score_substep(
            inverse_arm_trace, oracle, active, area, index, plant="none"))
    baseline_first = r129.first_nonbit(baseline_rows)
    arm_first = r129.first_nonbit(arm_rows)
    require(baseline_first is not None,
            "source-order baseline unexpectedly has no debt")

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
    control_substep2 = {
        row["boundary"]: row for row in control_rows if row["substep"] == 2
    }
    arm_substep2 = {
        row["boundary"]: row for row in arm_rows if row["substep"] == 2
    }
    inverse_arm_substep2 = {
        row["boundary"]: row for row in inverse_arm_rows
        if row["substep"] == 2
    }
    p148_control_census = (
        control_substep2["mid_depth_v"]["operand_differing_cells"] == 30
        and control_substep2["transport_v"]["operand_differing_cells"] == 68
    )
    source_order = [row["boundary"] for row in control_rows
                    if row["substep"] == 2]
    mid_depth_v_index = source_order.index("mid_depth_v")
    p148_control = (
        p148_control_census
        and all(control_substep2[name]["operand_bit_exact"]
                for name in source_order[:mid_depth_v_index])
    )
    p148_depth = arm_substep2["mid_depth_v"]["operand_bit_exact"]
    p148_chain = all(
        arm_substep2[name]["operand_bit_exact"]
        for name in ("transport_v", "continuity_dv", "after_ssh")
    )
    p149_control = (
        control_substep2["entry_inverse_v"]["operand_differing_cells"] == 68
        and control_substep2["entry_inverse_v"]["operand_absolute_max"]
        == 0.03332976059679253
    )
    p149_inverse = inverse_arm_substep2["entry_inverse_v"][
        "operand_bit_exact"]
    p149_chain_retained = (
        inverse_arm_substep2["mid_depth_v"]["operand_differing_cells"] == 30
        and inverse_arm_substep2["transport_v"]["operand_differing_cells"] == 68
        and inverse_arm_substep2["continuity_dv"]["operand_differing_cells"] == 68
        and inverse_arm_substep2["after_ssh"]["operand_differing_cells"] == 68
    )
    p2 = (
        sum(row["changed_cells"] for row in scope_rows.values()) > 0
        and all(row["outside_allowed_cells"] == 0
                for row in scope_rows.values())
    )
    p3 = all(row["bit_exact"] for row in post_rows.values())
    p4 = all(row["bit_exact"] for row in arm_target.values())
    if plant == "u-fold-sign":
        require(not post_rows["u"]["bit_exact"],
                "u-fold-sign plant stayed green")
        raise GateError("u-fold-sign plant fired")
    if plant == "stored-pivot-source":
        require(not p148_depth and p148_control_census,
                "stored-pivot-source plant stayed green")
        raise GateError("stored-pivot-source plant fired")
    if plant == "v-depth-bit":
        require(
            arm_substep2["mid_depth_v"]["operand_differing_cells"]
            == v_depth_plant_expected,
            "v-depth-bit plant stayed green",
        )
        raise GateError("v-depth-bit plant fired")
    if plant == "wrong-entry-frame":
        require(not p149_inverse, "wrong-entry-frame plant stayed green")
        raise GateError("wrong-entry-frame plant fired")
    if plant == "entry-inverse-v-bit":
        require(
            inverse_arm_substep2["entry_inverse_v"]
            ["operand_differing_cells"] == 1,
            "entry-inverse-v-bit plant stayed green",
        )
        raise GateError("entry-inverse-v-bit plant fired")
    midpoint_rows = midpoint_v_split["operand_rows"]
    p150_components = all(
        midpoint_rows[name]["bit_exact"]
        for name in (
            "midpoint_ssh", "area_t", "local_area_ssh", "north_area_ssh",
        )
    )
    p150_reference = (
        midpoint_rows["reference_depth_v"]["differing_cells"] == 30
        and midpoint_rows["reference_depth_v"]["maximum_absolute"] == 899.0
        and midpoint_rows["reciprocal_area_v"]["bit_exact"]
        and midpoint_rows["ssvmask"]["bit_exact"]
    )
    p150_arm = midpoint_v_split[
        "reference_depth_arm_mid_depth_v"]["bit_exact"]
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
        "midpoint_v_operand_split": midpoint_v_split,
        "baseline_first_non_bit": baseline_first,
        "arm_first_non_bit": arm_first,
        "control_first_non_bit": r129.first_nonbit(control_rows),
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
            "R148-P1": "CONFIRMED" if p3 else "REFUTED",
            "R148-P2": "CONFIRMED" if p148_control else "REFUTED",
            "R148-P3": "CONFIRMED" if p148_depth else "REFUTED",
            "R148-P4": "CONFIRMED" if p148_chain else "REFUTED",
            "R149-P1": "CONFIRMED" if p3 else "REFUTED",
            "R149-P2": "CONFIRMED" if p149_control else "REFUTED",
            "R149-P3": (
                "CONFIRMED" if p149_control and p149_inverse
                else "REFUTED" if p149_control
                else "UNMEASURED_PREREQUISITE_R149-P2"),
            "R149-P4": (
                "CONFIRMED" if p149_chain_retained else "REFUTED"),
            "R150-P1": (
                "CONFIRMED" if p3 and p148_control_census
                else "REFUTED"),
            "R150-P2": "CONFIRMED" if p150_components else "REFUTED",
            "R150-P3": "CONFIRMED" if p150_reference else "REFUTED",
            "R150-P4": (
                "CONFIRMED_DEPTH_ONLY" if p150_reference and p150_arm
                else "REFUTED" if p150_reference
                else "UNMEASURED_PREREQUISITE_R150-P3"),
        },
        "r148_control_census_reproduced": p148_control_census,
        "baseline_rows": baseline_rows,
        "control_rows": control_rows,
        "arm_rows": arm_rows,
        "inverse_arm_rows": inverse_arm_rows,
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
