#!/usr/bin/env python3
"""ORCA2 round-14 gate: split the barotropic owner in two.

Round 13 left the ladder's largest kt=1 row -- the end-of-step sea surface,
0.2448 m -- owned by a PLAUSIBLE candidate rather than a named one, and named
the one measurement that separates the two halves of that candidate:
substitute the record's own ``oracle_slow_forcing_kt00000001.bin`` into
legoESM's barotropic solve and re-measure.  If the sea surface closes, the
owner is the depth-mean momentum tendency the solver RECEIVES; if it does not,
the owner is the solver.

This gate runs that substitution AND, because the same record carries the
forcing's own ordered intermediates, walks the forcing side as a boundary
ladder so that a forcing owner is named rather than merely implicated.

NEMO's ordered statements, all from the build that produced the record
(``ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo``):

*  ``stp2d.f90:139-147,162-167`` -- one evaluation of the 3-D right-hand side
   at the BEFORE level: pressure gradient, lateral viscosity, vorticity, then
   the kinetic-energy gradient and vertical advection.
*  ``stp2d.f90:196-200`` -- the vertical average, with the REFERENCE-thickness
   reciprocal ``r1_hu_0``, not a live one.
*  ``stp2d.f90:219`` -- ``dyn_drg_init``'s baroclinic-residual bottom drag.
*  ``stp2d.f90:228-231`` -- the wind, ``r1_rho0 * utauU * r1_hu_0/(1+r3u)``.
*  ``stp2d.f90:232-236`` -- the record's last write; the value after it is what
   ``dyn_spg_ts`` receives at ``stp2d.f90:302-303``.

TWO ORDERS, ONE LADDER.  NEMO applies the drag BEFORE the wind and legoESM
applies the wind before the drag.  The two are additive on the same depth
mean, so this gate compares the INCREMENTS (drag increment, wind increment)
rather than the intermediates.  Comparing ``post_wind`` to ``post_wind``
across those two orders would be a confound, not a result.

HALF A DOMAIN, AND IT IS SAID OUT LOUD.  The record's slow forcing exists for
RANK 0 only, so the substitution covers the 90 longitude columns rank 0 owns
and the other 90 keep legoESM's own forcing.  Sea-surface information travels
at the external gravity-wave speed, so the substituted half is contaminated
inward from both of its periodic edges.  Every substituted-arm number is
reported twice -- over the whole half and over its interior -- with the
gravity-wave margin measured on the card's own metric, and the receipt says
which is which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))
_TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(_TESTCASES) not in sys.path:
    sys.path.insert(0, str(_TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
# EVERY value here is a key of the campaign citation map, is checked to
# identify its line, and is checked to FAIL under a two-line shift, by
# ``tests/ocean/fidelity/test_nemo_testcase_l4_orca2_round14_receipt_citations.py``.
# That test used to read only the RECEIPT, so this dict -- which is copied
# verbatim into every evidence JSON this gate writes -- was unchecked, and it
# carried a wrong line range: ``dynspg_ts.f90:296-300`` is a comment banner
# and the coefficient-setup call, NOT the statement that removes the 2-D
# Coriolis trend.  That is at :320-324, and the review that caught it is the
# reason the test now covers this dict too.
CITATIONS = {
    "rhs_3d_evaluation": f"{_PP}/stp2d.f90:139-147",
    "rhs_3d_advection": f"{_PP}/stp2d.f90:162-166",
    "vertical_average_reference_reciprocal": f"{_PP}/stp2d.f90:196-199",
    "baroclinic_drag": f"{_PP}/stp2d.f90:218-221",
    "wind_forcing": f"{_PP}/stp2d.f90:229-230",
    "handoff_to_the_external_solver": f"{_PP}/stp2d.f90:302-303",
    "external_solver_copies_it": f"{_PP}/dynspg_ts.f90:287-291",
    "external_solver_removes_the_2d_coriolis": f"{_PP}/dynspg_ts.f90:320-324",
    "external_solver_substep_update": f"{_PP}/dynspg_ts.f90:668-671",
    "reference_depth_is_built_from_the_same_thickness":
        f"{_PP}/domain.f90:199",
    "stage_sea_surface_interpolation": f"{_PP}/stprk3_stg.f90:137,152-154",
}

# Round 13's published end-of-step (stage-3) rank-0 sea-surface maximum.  The
# reproduction row must land on it or every comparison below is void.
ROUND13_END_OF_STEP_SSH_M = 2.4484e-01

# The record's rank-0 extent, ``(jpi, jpj, jpk)`` as the writer stamped it.
RANK0_DIMS = (94, 152, 31)
RANK0_COLUMNS = 90


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_u(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[:, 1:, ...]


def _native_v(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[1:, :, ...]


def _rank0(values: np.ndarray) -> np.ndarray:
    return values[:, :RANK0_COLUMNS, ...]


def _source_sum(e3, rhs, mask, reciprocal) -> np.ndarray:
    """Literal left-to-right transcription of the Fortran ``SUM``."""
    product = (e3 * rhs) * mask
    total = np.array(product[..., 0], copy=True)
    for level in range(1, product.shape[-1]):
        total = total + product[..., level]
    return total * reciprocal


def _face_arrays(operands, face: str) -> dict[str, np.ndarray]:
    """legoESM's own ordered slow-forcing intermediates on rank 0's columns."""
    take = _native_u if face == "u" else _native_v
    suffix = "u" if face == "u" else "v"
    out = {
        "e3": _rank0(take(operands[f"h_{suffix}"])),
        "krhs": _rank0(take(operands[f"d{suffix}_dt"])),
        "H": _rank0(take(operands[f"H_{suffix}"])),
        "depth_mean": _rank0(take(operands[f"depth_{suffix}"])),
        "post_wind": _rank0(take(operands[f"post_wind_{suffix}"])),
        "post_drag": _rank0(take(operands[f"post_drag_{suffix}"])),
        "final": _rank0(take(operands[f"pre_external_{suffix}"])),
        "tau": _rank0(take(operands[f"wind_tau_{suffix}"])),
    }
    out["r1_h0"] = 1.0 / out["H"]
    # legoESM order: depth mean -> wind -> drag.
    out["wind_increment"] = out["post_wind"] - out["depth_mean"]
    out["drag_increment"] = out["post_drag"] - out["post_wind"]
    return out


def _oracle_arrays(oracle, face: str) -> dict[str, np.ndarray]:
    suffix = "u" if face == "u" else "v"
    out = {
        "e3": oracle[f"e3{suffix}"],
        "krhs": oracle[f"krhs_{suffix}"],
        "mask": oracle[f"{suffix}mask"],
        "r1_h0": oracle[f"r1_h{suffix}0"],
        "depth_mean": oracle[f"depth_{suffix}"],
        "post_drag": oracle[f"post_drag_{suffix}"],
        "post_wind": oracle[f"post_wind_{suffix}"],
        "tau": oracle[f"{suffix}tau"],
    }
    # NEMO order: depth mean -> drag -> wind.
    out["drag_increment"] = out["post_drag"] - out["depth_mean"]
    out["wind_increment"] = out["post_wind"] - out["post_drag"]
    out["final"] = out["post_wind"]
    return out


def _localize(delta, active) -> dict[str, object]:
    """WHERE a disagreement lives: the fold row, the rim, or the interior.

    A field-wide maximum says nothing about whether a difference is a
    boundary-row artefact or a basin-wide operator error, and on a tripolar
    grid the last row is the fold.  This reports the row histogram rather
    than leaving the reader to assume.
    """
    delta = np.where(np.asarray(active, dtype=bool),
                     np.asarray(delta, dtype=np.float64), 0.0)
    nonzero = delta != 0.0
    rows = nonzero.reshape(delta.shape[0], -1).sum(axis=1)
    total = int(nonzero.sum())
    order = np.argsort(rows)[::-1][:5]
    flat = int(np.argmax(np.abs(delta)))
    return {
        "differing_cells": total,
        "on_the_last_row_the_tripolar_fold": int(rows[-1]),
        "on_the_first_row": int(rows[0]),
        "rows_carrying_any_difference": int((rows > 0).sum()),
        "total_rows": int(delta.shape[0]),
        "five_worst_rows": [[int(r), int(rows[r])] for r in order],
        "argmax_index": [int(i) for i in np.unravel_index(flat, delta.shape)],
        "argmax_value": float(delta.reshape(-1)[flat]),
    }


def compare(candidate, oracle, active) -> dict[str, object]:
    from nemo_testcase_state_ulp_probe import ulp_distance

    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape,
            f"shape mismatch {candidate.shape} {oracle.shape} {active.shape}")
    delta = candidate[active] - oracle[active]
    return {
        "bit_exact": bool(np.array_equal(candidate[active], oracle[active])),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "ulp_max": int(np.max(
            ulp_distance(candidate[active], oracle[active]), initial=0)),
        "differing_cells": int(np.count_nonzero(delta)),
        "scored_cells": int(np.count_nonzero(active)),
    }


def _ssh_row(stage_outputs, oracle_ssh, stage: int) -> dict[str, object]:
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    candidate = ladder._stage_candidate_fields(stage_outputs[stage - 1])["ssh"]
    ny, nx = oracle_ssh.shape
    delta = np.asarray(candidate, dtype=np.float64)[:ny, :nx] - oracle_ssh
    per_column = np.abs(delta).max(axis=0)
    return {
        "stage": stage,
        "max_abs_m": float(np.abs(delta).max()),
        # The last row of a tripolar grid is the FOLD.  Reporting only a
        # field-wide maximum would let a fold-row artefact be read as a
        # basin-wide solver error, which is the difference between two very
        # different next rounds.
        "max_abs_excluding_the_fold_row_m": float(np.abs(delta[:-1]).max()),
        "argmax_column": int(np.argmax(per_column)),
        "unequal_cells": int(np.count_nonzero(delta)),
        "scored_cells": int(delta.size),
        "per_column_max_abs_m": [float(value) for value in per_column],
        "where": _localize(delta, np.ones_like(delta, dtype=bool)),
    }


def _seeded_state(card, entry):
    import jax.numpy as jnp

    state = card.recipe.initial_state
    return state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"], dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(entry["S"], dtype=jnp.float64)),
        eta=state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )


def _inject(full_u, full_v, rank0_u, rank0_v):
    """Replace ONLY rank 0's owned native window; leave every other face."""
    out_u = np.array(full_u, dtype=np.float64, copy=True)
    out_v = np.array(full_v, dtype=np.float64, copy=True)
    require(out_u[:, 1:1 + RANK0_COLUMNS].shape == rank0_u.shape,
            "U injection window does not match the record's rank-0 extent")
    require(out_v[1:, :RANK0_COLUMNS].shape == rank0_v.shape,
            "V injection window does not match the record's rank-0 extent")
    out_u[:, 1:1 + RANK0_COLUMNS] = rank0_u
    out_v[1:, :RANK0_COLUMNS] = rank0_v
    return out_u, out_v


def run(deck_root: Path, root: Path, json_out: Path | None,
        plant: bool = False) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )

    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    slow_path = root / "oracle_slow_forcing_kt00000001.bin"
    require(slow_path.is_file(), f"missing {slow_path}")
    oracle = read_slow_forcing(slow_path, dims=RANK0_DIMS)

    _, card = ladder.card_fields(deck_root)
    config = card.recipe.model_config
    require(config.barotropic.barotropic_solver == "explicit_substep",
            "the ORCA2 card no longer selects the split-explicit solver")
    require(config.coriolis_scheme == "explicit_ab2"
            and config.barotropic_coriolis_split == "live",
            "the ORCA2 card no longer reaches the live-Coriolis split, which "
            "is where the substitution hook lands")
    entry = ladder.assemble_state_fields(root, 1, stage=None)

    # CONTROL, printed: the only operand Decision 52 bridges is the sea
    # surface.  If T/S/u/v were not already bit-identical, seeding them would
    # be a second changed variable and the substitution would not be a
    # one-variable arm.
    state0 = card.recipe.initial_state
    entry_identity = {
        "T": bool(np.array_equal(
            np.asarray(state0.T.data)[..., :30], entry["T"])),
        "S": bool(np.array_equal(
            np.asarray(state0.S.data)[..., :30], entry["S"])),
        "u": bool(np.array_equal(
            np.asarray(state0.u.data)[:, 1:, :30], entry["u"])),
        "v": bool(np.array_equal(
            np.asarray(state0.v.data)[1:, :, :30], entry["v"])),
        "ssh_is_the_decision52_bridge": bool(not np.array_equal(
            np.asarray(state0.eta.data), entry["ssh"])),
    }
    entry_velocity = {
        "max_abs_entry_u_m_s": float(np.abs(entry["u"]).max()),
        "max_abs_entry_v_m_s": float(np.abs(entry["v"]).max()),
    }
    entry_velocity["kt1_is_a_rest_step"] = bool(
        entry_velocity["max_abs_entry_u_m_s"] == 0.0
        and entry_velocity["max_abs_entry_v_m_s"] == 0.0)
    print("CONTROL entry operands other than the sea surface are already "
          f"bit-identical: {entry_identity}")
    print("CONTROL the kt=1 entry velocity, which decides whether the "
          f"velocity-dependent operators are reachable at all: {entry_velocity}")
    require(all(entry_identity[name] for name in ("T", "S", "u", "v")),
            "an entry operand other than the sea surface is not bit-identical; "
            "seeding it would make the substitution a two-variable arm")

    state = _seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)

    def _stage_trace(incoming_override=None):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                slow_forcing_incoming_override=incoming_override),
        iwm_forcing=card.recipe.iwm_forcing,
    )
        return model.step(state, dt=card.dt_s, freshwater=freshwater,
                          surface_forcing=surface)

    # --- 0. the baseline, and round 13's number reproduced -----------------
    print("STEP baseline production step")
    baseline = _stage_trace()
    oracle_ssh = {
        stage: ladder.read_state_frame(
            root / f"oracle_stage_kt00000001_s{stage}.bin", kt=1, stage=stage
        )["ssh"] for stage in (1, 2, 3)
    }
    baseline_rows = {
        stage: _ssh_row(baseline.stage_outputs, oracle_ssh[stage], stage)
        for stage in (1, 2, 3)
    }
    end_of_step = baseline_rows[3]["max_abs_m"]
    reproduction = {
        "round13_published_end_of_step_max_abs_m": ROUND13_END_OF_STEP_SSH_M,
        "measured_end_of_step_max_abs_m": end_of_step,
        "relative_departure": abs(
            end_of_step - ROUND13_END_OF_STEP_SSH_M
        ) / ROUND13_END_OF_STEP_SSH_M,
        "reproduces": bool(abs(end_of_step - ROUND13_END_OF_STEP_SSH_M)
                           < 5.0e-05 * ROUND13_END_OF_STEP_SSH_M),
    }
    require(reproduction["reproduces"],
            "the baseline end-of-step sea surface does not reproduce round "
            f"13's {ROUND13_END_OF_STEP_SSH_M} m (measured {end_of_step})")

    # --- 1. the forcing ladder, on the record's own ordered intermediates --
    model_ops = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True),
        iwm_forcing=card.recipe.iwm_forcing,
    )
    print("STEP slow-forcing operand trace")
    operand_trace = model_ops.step(
        state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
    operands = jax.device_get(operand_trace.slow_forcing_operands)

    # The model's OWN three-dimensional face masks, built by the same helper
    # the WS-RK3 stage program uses, against NEMO's ``umask``/``vmask``.  The
    # two-dimensional ``state.u_mask`` is a surface face mask and would not be
    # the operand NEMO's vertical sum multiplies.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )

    _u_mask3, _v_mask3 = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    card_masks = {
        "u": _rank0(_native_u(np.asarray(_u_mask3, dtype=np.float64)))[
            ..., :30],
        "v": _rank0(_native_v(np.asarray(_v_mask3, dtype=np.float64)))[
            ..., :30],
    }
    ladder_rows: dict[str, dict[str, dict]] = {}
    instrument_rows: dict[str, dict] = {}
    for face in ("u", "v"):
        cand = _face_arrays(operands, face)
        orc = _oracle_arrays(oracle, face)
        active3 = orc["mask"] > 0.0
        active2 = active3[..., 0]
        everywhere = np.ones_like(active3, dtype=bool)
        pairs = {
            "e3": (cand["e3"], orc["e3"], active3),
            "Krhs": (cand["krhs"], orc["krhs"], active3),
            "mask": (card_masks[face], orc["mask"], everywhere),
            "r1_h0": (cand["r1_h0"], orc["r1_h0"], active2),
            "depth_mean": (cand["depth_mean"], orc["depth_mean"], active2),
            "drag_increment": (
                cand["drag_increment"], orc["drag_increment"], active2),
            "wind_increment": (
                cand["wind_increment"], orc["wind_increment"], active2),
            "wind_tau": (cand["tau"], orc["tau"], active2),
            "final": (cand["final"], orc["final"], active2),
        }
        ladder_rows[face] = {}
        for name, (left, right, active) in pairs.items():
            row = compare(left, right, active)
            # WHERE, not just how big.  On a tripolar grid a field-wide
            # maximum cannot tell a fold-row artefact from a basin-wide
            # operator error, and this round found rows of both kinds.
            if not row["bit_exact"]:
                row["where"] = _localize(
                    np.asarray(left, dtype=np.float64)
                    - np.asarray(right, dtype=np.float64), active)
            ladder_rows[face][name] = row
        # INSTRUMENT CONTROL: replay NEMO's own written SUM from NEMO's own
        # written operands.  If this is not bit-exact, the reader or the
        # transcription of the statement is wrong and no row above is
        # quotable.
        instrument_rows[face] = {
            "oracle_operands_replay_the_oracle_depth_mean": compare(
                _source_sum(orc["e3"], orc["krhs"], orc["mask"], orc["r1_h0"]),
                orc["depth_mean"], active2),
            "live_divisor_differs_from_the_reference_divisor": compare(
                cand["r1_h0"], orc["r1_h0"], active2),
            # ONE-VARIABLE ARMS on the record's own written operands: each
            # replays NEMO's own statement with exactly ONE operand family
            # swapped for legoESM's, so the depth-mean disagreement is split
            # between the three-dimensional right-hand side and the metric it
            # is weighted with.  No model run, no other operand moves.
            "only_the_rhs_is_legoesms": compare(
                _source_sum(orc["e3"], cand["krhs"], orc["mask"],
                            orc["r1_h0"]),
                orc["depth_mean"], active2),
            "only_the_metric_is_legoesms": compare(
                _source_sum(cand["e3"], orc["krhs"], card_masks[face],
                            cand["r1_h0"]),
                orc["depth_mean"], active2),
            "both_are_legoesms_replayed_in_nemos_association": compare(
                _source_sum(cand["e3"], cand["krhs"], card_masks[face],
                            cand["r1_h0"]),
                orc["depth_mean"], active2),
            # The mask the vertical sum EFFECTIVELY applies is the support of
            # legoESM's own face thickness, not a separately stored array.
            "effective_weight_support_against_nemos_mask": compare(
                (cand["e3"] > 0.0).astype(np.float64),
                (orc["mask"] > 0.0).astype(np.float64),
                np.ones_like(active3, dtype=bool)),
        }
    print("CONTROL the record's own operands replay its own depth mean: "
          + json.dumps({
              face: instrument_rows[face][
                  "oracle_operands_replay_the_oracle_depth_mean"]["bit_exact"]
              for face in ("u", "v")}))
    for face in ("u", "v"):
        require(instrument_rows[face][
            "oracle_operands_replay_the_oracle_depth_mean"]["bit_exact"],
            f"the {face.upper()} depth-mean replay of NEMO's OWN operands is "
            "not bit-exact: the reader or the statement is wrong")

    order = ("e3", "Krhs", "mask", "r1_h0", "depth_mean",
             "drag_increment", "wind_increment", "final")
    first_boundary = None
    for boundary in order:
        for face in ("u", "v"):
            row = ladder_rows[face][boundary]
            if first_boundary is None and not row["bit_exact"]:
                first_boundary = {"boundary": boundary, "face": face, **row}

    # --- 2. the substitution ----------------------------------------------
    producer = jax.device_get(baseline.slow_forcing_producer)
    own_u = np.asarray(producer["incoming_u"], dtype=np.float64)
    own_v = np.asarray(producer["incoming_v"], dtype=np.float64)

    # CONTROL: the hook itself must be inert when it is fed legoESM's own
    # values.  Without this, any movement below could be the hook.
    print("STEP no-op substitution control")
    noop = _stage_trace((jnp.asarray(own_u), jnp.asarray(own_v)))
    noop_row = _ssh_row(noop.stage_outputs, oracle_ssh[3], 3)
    noop_identical = bool(np.array_equal(
        np.asarray(noop.stage_outputs[2][4]),
        np.asarray(baseline.stage_outputs[2][4])))
    print("CONTROL the substitution hook fed legoESM's own forcing leaves the "
          f"end-of-step sea surface bit-identical: {noop_identical} "
          f"(max {noop_row['max_abs_m']:.6e} m)")
    require(noop_identical,
            "the substitution hook is not inert on legoESM's own operand; no "
            "substituted number below would be interpretable")

    injected_u, injected_v = _inject(
        own_u, own_v, _oracle_arrays(oracle, "u")["final"],
        _oracle_arrays(oracle, "v")["final"])
    print("STEP substituted arm")
    substituted = _stage_trace(
        (jnp.asarray(injected_u), jnp.asarray(injected_v)))
    substituted_producer = jax.device_get(substituted.slow_forcing_producer)
    landed = {
        "u_on_the_substituted_window": bool(np.array_equal(
            np.asarray(substituted_producer["incoming_u"])[
                :, 1:1 + RANK0_COLUMNS],
            injected_u[:, 1:1 + RANK0_COLUMNS])),
        "v_on_the_substituted_window": bool(np.array_equal(
            np.asarray(substituted_producer["incoming_v"])[
                1:, :RANK0_COLUMNS],
            injected_v[1:, :RANK0_COLUMNS])),
        "u_off_the_window_is_legoesms_own": bool(np.array_equal(
            np.asarray(substituted_producer["incoming_u"])[
                :, 1 + RANK0_COLUMNS:],
            own_u[:, 1 + RANK0_COLUMNS:])),
    }
    print(f"CONTROL the substitution landed where it was aimed: {landed}")
    require(all(landed.values()),
            "the injected slow forcing did not reach the production trace")

    substituted_rows = {
        stage: _ssh_row(substituted.stage_outputs, oracle_ssh[stage], stage)
        for stage in (1, 2, 3)
    }

    # THE CAUSAL MEASURE, and the first draft of this gate got it wrong.
    # Comparing the two arms' MAXIMA against the oracle is a difference of
    # maxima, which can stay flat while the field moves elsewhere.  What the
    # substitution actually does is the maximum of the DIFFERENCE between the
    # two arms' own sea surfaces.
    def _end_of_step(trace):
        return np.asarray(ladder._stage_candidate_fields(
            trace.stage_outputs[2])["ssh"], dtype=np.float64)[
                :, :RANK0_COLUMNS]

    base_ssh = _end_of_step(baseline)
    sub_ssh = _end_of_step(substituted)
    noop_ssh = _end_of_step(noop)
    arm_delta = np.abs(sub_ssh - base_ssh)
    arm_to_arm = {
        "max_abs_move_m": float(arm_delta.max()),
        "cells_moved": int(np.count_nonzero(arm_delta)),
        "scored_cells": int(arm_delta.size),
        "argmax_index": [int(i) for i in
                         np.unravel_index(int(np.argmax(arm_delta)),
                                          arm_delta.shape)],
        "per_column_max_abs_m": [float(v) for v in arm_delta.max(axis=0)],
        "the_noop_control_moves_nothing": bool(
            np.array_equal(noop_ssh, base_ssh)),
        "move_over_the_baseline_disagreement": float(
            arm_delta.max() / baseline_rows[3]["max_abs_m"]),
    }
    print("MEASURE substituting NEMO's own slow forcing over rank 0 moves the "
          f"end-of-step sea surface by at most {arm_to_arm['max_abs_move_m']:.6e} m "
          f"on {arm_to_arm['cells_moved']} of {arm_to_arm['scored_cells']} cells; "
          "the baseline disagreement is "
          f"{baseline_rows[3]['max_abs_m']:.6e} m")

    # The gravity-wave margin, measured on the card's own metric rather than
    # assumed: how far the external mode travels in one baroclinic step, and
    # how many rank-0 columns that is at the baseline row that owns the
    # maximum.
    import legoesm.constants as constants
    depth = np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)
    c_ext = float(np.sqrt(float(constants.g) * float(depth.max())))
    reach_m = c_ext * float(card.dt_s)
    dx_t = np.asarray(card.recipe.grid.dx_T, dtype=np.float64)
    row_index = int(np.argmax(
        np.abs(np.asarray(
            ladder._stage_candidate_fields(
                substituted.stage_outputs[2])["ssh"]
        )[:, :RANK0_COLUMNS] - oracle_ssh[3]).max(axis=1)))
    row_dx = dx_t[row_index, :RANK0_COLUMNS]
    columns_in_reach = int(np.searchsorted(np.cumsum(row_dx), reach_m)) + 1
    margin = min(columns_in_reach, (RANK0_COLUMNS - 1) // 2)
    interior = slice(margin, RANK0_COLUMNS - margin)
    interior_profile = np.asarray(
        substituted_rows[3]["per_column_max_abs_m"])[interior]
    baseline_interior = np.asarray(
        baseline_rows[3]["per_column_max_abs_m"])[interior]
    contamination = {
        "external_gravity_wave_speed_m_s": c_ext,
        "reach_in_one_baroclinic_step_m": reach_m,
        "margin_columns": margin,
        "columns_within_reach_at_the_worst_row": columns_in_reach,
        "worst_row_index": row_index,
        "interior_columns": [int(interior.start), int(interior.stop)],
        "substituted_interior_max_abs_m": float(interior_profile.max()),
        "baseline_interior_max_abs_m": float(baseline_interior.max()),
    }

    whole = {
        "baseline_end_of_step_max_abs_m": baseline_rows[3]["max_abs_m"],
        "substituted_end_of_step_max_abs_m": substituted_rows[3]["max_abs_m"],
        "movement_m": abs(substituted_rows[3]["max_abs_m"]
                          - baseline_rows[3]["max_abs_m"]),
        "closed_fraction": 1.0 - (substituted_rows[3]["max_abs_m"]
                                  / baseline_rows[3]["max_abs_m"]),
    }
    interior_summary = {
        "baseline_max_abs_m": contamination["baseline_interior_max_abs_m"],
        "substituted_max_abs_m": contamination[
            "substituted_interior_max_abs_m"],
        "closed_fraction": 1.0 - (
            contamination["substituted_interior_max_abs_m"]
            / contamination["baseline_interior_max_abs_m"]),
    }

    # The verdict is read from how far the substitution MOVES the sea
    # surface, relative to the disagreement it would have to close.
    share = arm_to_arm["move_over_the_baseline_disagreement"]
    verdict = ("THE_FORCING_IS_A_MAJORITY_OWNER" if share >= 0.5 else
               ("THE_FORCING_IS_A_CONTRIBUTOR" if share >= 0.1 else
                "THE_FORCING_IS_NOT_THE_OWNER"))

    # THE PLANT, and the first version of it was VACUOUS.  It planted a
    # one-representable-value change in the candidate face thickness and
    # required that to become the ladder's first non-bit boundary -- which it
    # already was, so the plant could not fail.  What this round's conclusion
    # actually rests on is that the substitution CHANNEL can carry the
    # smallest possible change into the solver: "the forcing barely moves the
    # sea surface" means nothing if the channel is deaf.  So the plant adds
    # ONE unit in the last place to ONE injected value and requires the
    # end-of-step sea surface to change.
    plant_control = {"requested": plant}
    if plant:
        planted_u = np.array(injected_u, dtype=np.float64, copy=True)
        window = planted_u[:, 1:1 + RANK0_COLUMNS]
        index = tuple(np.argwhere(window != 0.0)[0])
        before = window[index]
        window[index] = np.nextafter(before, np.inf)
        planted_u[:, 1:1 + RANK0_COLUMNS] = window
        moved = int(np.count_nonzero(planted_u - injected_u))
        require(moved == 1, "the plant changed more than one value")
        planted = _stage_trace(
            (jnp.asarray(planted_u), jnp.asarray(injected_v)))
        planted_ssh = _end_of_step(planted)
        delta = np.abs(planted_ssh - sub_ssh)
        fired = bool(delta.max() > 0.0)
        plant_control.update({
            "one_value_changed_by_one_unit_in_the_last_place": True,
            "planted_index": [int(i) for i in index],
            "sea_surface_cells_moved": int(np.count_nonzero(delta)),
            "sea_surface_max_abs_move_m": float(delta.max()),
            "fires": fired,
        })
        if fired:
            print("PLANT FIRED: the gate refuses a one-representable-value "
                  "move in the injected slow forcing -- it changes the "
                  f"end-of-step sea surface on {int(np.count_nonzero(delta))} "
                  f"of {delta.size} cells, max {delta.max():.6e} m")
        else:
            print("PLANT DID NOT FIRE: a one-representable-value move in the "
                  "injected slow forcing left the sea surface bit-identical, "
                  "so the substitution channel is deaf and no substituted "
                  "number in this gate is interpretable")
        require(fired, "the one-representable-value plant did not reach the "
                       "end-of-step sea surface")

    result = {
        "gate": "nemo_testcase_l4_orca2_round14_barotropic_owner_gate",
        "kt": 1,
        "label": "given NEMO's entry (the kt=1 recorded state, surface frames "
                 "and slow-forcing frame)",
        "citations": CITATIONS,
        "record": str(slow_path),
        "provenance": worktree_stamp(),
        "entry_operand_controls": entry_identity,
        "round13_reproduction": reproduction,
        "baseline_sea_surface_by_stage": baseline_rows,
        "forcing_ladder": ladder_rows,
        "instrument_controls": instrument_rows,
        "first_non_bit_boundary": first_boundary,
        "plant": plant_control,
        "substitution_controls": {
            "hook_is_inert_on_legoesms_own_operand": noop_identical,
            "injection_landed": landed,
        },
        "substituted_sea_surface_by_stage": substituted_rows,
        "arm_to_arm_sea_surface_movement": arm_to_arm,
        "entry_velocity": entry_velocity,
        "whole_rank0_half": whole,
        "contamination_margin": contamination,
        "interior_only": interior_summary,
        "verdict": verdict,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.json_out,
                     plant=args.plant)
    except GateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("baseline_sea_surface_by_stage",
                       "substituted_sea_surface_by_stage")
    }, indent=2, sort_keys=True))
    if args.plant:
        # A planted control exits NONZERO: 1 when it correctly fired, 2 when
        # it did not, which would mean the channel is deaf and this gate's
        # substituted numbers are not interpretable.
        return 1 if result["plant"].get("fires") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
