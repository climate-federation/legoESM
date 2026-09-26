#!/usr/bin/env python3
"""Round 31: land dyn_ldf's own F thickness; gate dyn_vor's e3f_0vor build.

Two fail-closed subcommands.

``e3f0vor`` is part B.  It scores legoESM's production frozen vertex
thickness against a literal NumPy transcription of the compiled statements
``dynvor.f90:914-919`` (masked four-cell reference average), ``:935``
(F-point north-fold exchange) and ``:937`` (zero substitution by the MESH
reference thickness ``e3f_3d``), one statement at a time, on the card's own
carried mesh.  No acquisition carries ``e3f_0vor`` itself, so the reference
is the transcription, not a NEMO array; the carried ``e3f_0`` that rounds
28-30 routed is scored against it separately, which is what explains their
651 m maximum.

``direct-ldf`` is part A's given-NEMO's-entry control: it scores the
lateral-diffusion tendency before and after the consumer-local reference and
checks that the exposed stage-2 EEN component is untouched.
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

CITATIONS = {
    "ldf_f_curl": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "dynldf_lev.f90:123"),
    "vor_reciprocal": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "dynvor.f90:734-738"),
    "e3f0vor_average": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "dynvor.f90:914-919"),
    "e3f0vor_fold": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "dynvor.f90:935"),
    "e3f0vor_substitute": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "dynvor.f90:937"),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def score(left, right) -> dict:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    require(left.shape == right.shape, f"shape mismatch {left.shape} {right.shape}")
    unequal = ~(left == right)
    count = int(unequal.sum())
    row = {"cells": int(left.size), "unequal": count}
    if count:
        delta = np.abs(left - right)
        flat = int(np.argmax(np.where(unequal, delta, -1.0)))
        index = np.unravel_index(flat, left.shape)
        row["max_abs_diff"] = float(delta[index])
        row["argmax_index"] = [int(value) for value in index]
        row["argmax_left"] = float(left[index])
        row["argmax_right"] = float(right[index])
    return row


def _policy():
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 31 is CPU-only")


def _transcribe(e3t_0, tmask, e3f_3d, grid, nn_e3f_typ, *, plant=None):
    """Literal NumPy transcription of dynvor.f90:914-919, :935, :937.

    Native NEMO indexing (ji east, jj north) maps to legoESM's storage as
    axis 1 east and axis 0 north, so ``ji+1`` is ``roll(-1, axis=1)`` and
    ``jj+1`` is the shift-up-with-zero-fill used everywhere on this card.
    """
    e3t = np.asarray(e3t_0, np.float64)
    msk = np.asarray(tmask, np.float64)

    def east(value):
        return np.roll(value, -1, axis=1)

    def north(value):
        return np.concatenate([value[1:], np.zeros_like(value[:1])], axis=0)

    # :915-918  ( e3t(ji,jj+1)*tmask + e3t(ji+1,jj+1)*tmask )
    #         + ( e3t(ji,jj  )*tmask + e3t(ji+1,jj  )*tmask ) ) * 0.25
    masked = e3t * msk
    masked_n = north(masked)
    total = (masked_n + east(masked_n)) + (masked + east(masked))
    if nn_e3f_typ == 0:
        e3f0vor = total * 0.25
    else:
        wet = (msk + east(msk)) + (north(msk) + east(north(msk)))
        e3f0vor = np.where(wet != 0.0, total / np.where(wet == 0.0, 1.0, wet), 0.0)
    stage_average = e3f0vor.copy()

    # :935  CALL lbc_lnk( 'dynvor', e3f_0vor, 'F', 1._wp ).  BLIND SPOT: the
    # F-origin permutation itself is legoESM's helper on both sides, so this
    # statement scores the ORDER of the exchange, not the permutation, which
    # rounds 21-22 gated separately.
    import jax.numpy as jnp

    from legoesm.ocean.vertical import nemo_t_fold_f_owned

    e3f0vor = np.asarray(
        nemo_t_fold_f_owned(jnp.asarray(e3f0vor), grid), np.float64)
    stage_fold = e3f0vor.copy()

    # :937  WHERE( e3f_0vor == 0 ) e3f_0vor = e3f_3d
    fill = np.asarray(e3f_3d, np.float64)
    if plant == "substitute_operand":
        fill = fill * 2.0
    e3f0vor = np.where(e3f0vor == 0.0, fill, e3f0vor)
    return stage_average, stage_fold, e3f0vor


def capture_e3f0vor(deck_root: Path, *, plant: str | None = None) -> dict:
    import jax.numpy as jnp

    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        nemo_dynvor_e3f_0vor,
        nemo_ldf_reference_e3f,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    _policy()
    _, card = ladder.card_fields(deck_root)
    recipe = card.recipe
    grid = recipe.grid
    z_coord = recipe.z_coord
    state = recipe.initial_state
    config = recipe.model_config
    raw = getattr(z_coord, "nemo_een_barotropic", None)
    require(raw is not None, "ORCA2 card carries no NEMO mesh F operands")
    require(getattr(raw, "e3f_0vor", None) is None,
            "an e3f_0vor field appeared on the card; score against it instead")

    from legoesm.ocean.vertical import compute_layer_thickness

    e3t_0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m)
    tmask = getattr(z_coord, "is_active", None)
    require(tmask is not None, "ORCA2 z-coordinate carries no live T mask")
    e3f_3d = nemo_ldf_reference_e3f(z_coord)

    average, folded, nemo_final = _transcribe(
        e3t_0, tmask, e3f_3d, grid, 0, plant=plant)

    default_final, default_s1, default_s2 = (
        np.asarray(value, np.float64) for value in nemo_dynvor_e3f_0vor(
            e3t_0, tmask, grid=grid, dtype=jnp.float64, return_stages=True))
    repaired_final, repaired_s1, repaired_s2 = (
        np.asarray(value, np.float64) for value in nemo_dynvor_e3f_0vor(
            e3t_0, tmask, grid=grid, dtype=jnp.float64,
            substitute_e3f=e3f_3d, return_stages=True))

    statement_rows = {
        "s1_masked_four_cell_average": score(default_s1, average),
        "s2_after_fold_default_order": score(default_s2, folded),
        "s2_after_fold_repaired_order": score(repaired_s2, folded),
        "s3_final_default_order": score(default_final, nemo_final),
        "s3_final_repaired_order": score(repaired_final, nemo_final),
    }
    require(score(default_s1, repaired_s1)["unequal"] == 0,
            "the two orders disagree at statement 1")
    carried_vs_transcription = score(np.asarray(e3f_3d, np.float64), nemo_final)
    return {
        "status": "CAPTURED",
        "claim_label": "card mesh, no trajectory",
        "citations": {
            key: CITATIONS[key] for key in
            ("e3f0vor_average", "e3f0vor_fold", "e3f0vor_substitute",
             "ldf_f_curl", "vor_reciprocal")},
        "nn_e3f_typ": 0,
        "statements": statement_rows,
        "carried_e3f_0_vs_transcribed_e3f_0vor": carried_vs_transcription,
        "plant": plant,
        "worktree": worktree_stamp(),
    }


def evaluate_e3f0vor(capture: dict) -> dict:
    rows = capture["statements"]
    verdict = "PASS"
    reasons = []
    if rows["s3_final_repaired_order"]["unequal"] != 0:
        verdict = "HELD"
        reasons.append("repaired builder still differs from the transcription")
    if rows["s3_final_default_order"]["unequal"] == 0:
        verdict = "HELD"
        reasons.append("the default builder already matches; no defect found")
    for name in ("s1_masked_four_cell_average", "s2_after_fold_repaired_order"):
        if rows[name]["unequal"] != 0:
            verdict = "HELD"
            reasons.append(f"{name} is not exact")
    return {**capture, "status": verdict, "reasons": reasons}


# Round 28's measured parent digest of the exposed stage-2 EEN component,
# quoted from its receipt.  The lateral-diffusion landing must not move it.
R28_PARENT_EEN_DIGEST = (
    "032cb7d192afb4a60ec5816ab78504ffb96faa5d19d4e83b61c17d1d462247b2")


def capture_direct_ldf(deck_root: Path, record_root: Path,
                       *, plant: str | None = None) -> dict:
    """Part A, given NEMO's own recorded kt=2 entry.

    Three controls: the lateral-diffusion tendency built on the vorticity
    reference versus the mesh reference (is the landing active?); a runtime
    census of which production caller asks for which reference (is it
    consumer-local?); and the exposed stage-2 EEN component against round
    28's measured parent digest (is the vorticity operator untouched?).
    """
    import hashlib

    from legoesm.ocean import vertical
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round27_consumer_gate as r27,
    )

    _policy()
    original = vertical.nemo_qco_live_vorticity_e3f_cgrid

    vorticity_reference = r27._ldf_inputs(deck_root, record_root, 2)[0]["base"]

    def with_mesh_reference(*args, **kwargs):
        z_coord = args[1]
        kwargs["reference_e3f"] = vertical.nemo_ldf_reference_e3f(z_coord)
        return original(*args, **kwargs)

    vertical.nemo_qco_live_vorticity_e3f_cgrid = with_mesh_reference
    try:
        mesh_reference = r27._ldf_inputs(deck_root, record_root, 2)[0]["base"]
    finally:
        vertical.nemo_qco_live_vorticity_e3f_cgrid = original

    census: list[dict] = []

    def observed(*args, **kwargs):
        import sys as _sys
        census.append({
            "caller": _sys._getframe(1).f_code.co_name,
            "reference_given": kwargs.get("reference_e3f") is not None,
        })
        return original(*args, **kwargs)

    vertical.nemo_qco_live_vorticity_e3f_cgrid = observed
    try:
        exposed_u, exposed_v = r27._stage2_vorticity(deck_root, record_root)
    finally:
        vertical.nemo_qco_live_vorticity_e3f_cgrid = original

    digest = hashlib.sha256()
    for value in (exposed_u, exposed_v):
        arr = np.ascontiguousarray(np.asarray(value, np.float64))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    een_digest = digest.hexdigest()
    if plant == "een_digest":
        een_digest = "0" * 64

    return {
        "status": "CAPTURED",
        "claim_label": "given NEMO's entry",
        "citations": {"ldf_f_curl": CITATIONS["ldf_f_curl"],
                      "vor_reciprocal": CITATIONS["vor_reciprocal"]},
        "ldf_tendency_vorticity_vs_mesh_reference": {
            "u": score(vorticity_reference[0], mesh_reference[0]),
            "v": score(vorticity_reference[1], mesh_reference[1]),
        },
        "builder_call_census": {
            "total_calls": len(census),
            "callers_with_consumer_local_reference": sorted(
                {row["caller"] for row in census if row["reference_given"]}),
            "callers_on_the_vorticity_reference": sorted(
                {row["caller"] for row in census
                 if not row["reference_given"]}),
        },
        "exposed_stage2_een_digest": een_digest,
        "round28_parent_een_digest": R28_PARENT_EEN_DIGEST,
        "plant": plant,
        "worktree": worktree_stamp(),
    }


def evaluate_direct_ldf(capture: dict) -> dict:
    reasons = []
    verdict = "PASS"
    if capture["exposed_stage2_een_digest"] != capture[
            "round28_parent_een_digest"]:
        verdict = "HELD"
        reasons.append("the exposed stage-2 EEN component moved")
    if not capture["builder_call_census"][
            "callers_with_consumer_local_reference"]:
        verdict = "HELD"
        reasons.append("no production caller asked for the mesh reference")
    rows = capture["ldf_tendency_vorticity_vs_mesh_reference"]
    if rows["u"]["unequal"] == 0 and rows["v"]["unequal"] == 0:
        verdict = "HELD"
        reasons.append("the reference swap is inert; the arm proves nothing")
    return {**capture, "status": verdict, "reasons": reasons}


def capture_shared_cards(*, plant: str | None = None) -> dict:
    """Which shared cards the landed statement moves, and by how much.

    A card whose mesh reference ``e3f_0`` is bit-equal to its own frozen
    vorticity array cannot move: the landing changes only which of the two
    the lateral-diffusion curl multiplies.  This is the cheap, decisive
    precondition for the GYRE-unchanged claim; the trajectory gate confirms
    it.
    """
    import jax.numpy as jnp

    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipes
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_dynvor_e3f_0vor,
        nemo_ldf_reference_e3f,
    )

    _policy()
    rows = {}
    builders = {
        "GYRE-zco": recipes.build_gyre_zco_card,
        "LOCK_EXCHANGE-zco": recipes.build_lock_exchange_zco_card,
        "OVERFLOW-zps": recipes.build_overflow_zps_card,
    }
    for name, build in builders.items():
        card = build()
        recipe = card.recipe
        config = recipe.model_config
        executes = (
            getattr(config, "lateral_viscosity_operator", None) == "nemo_div_curl"
            and getattr(config, "lateral_viscosity_e3_weighting", "off")
            == "nemo_e3")
        row = {"executes_the_changed_statement": bool(executes)}
        if executes:
            z_coord = recipe.z_coord
            state = recipe.initial_state
            e3t_0 = compute_layer_thickness(
                jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
                min_water_column_m=config.min_water_column_m)
            tmask = getattr(z_coord, "is_active", None)
            if tmask is None:
                tmask = jnp.broadcast_to(
                    state.land_mask.data[..., None], e3t_0.shape)
            vorticity = nemo_dynvor_e3f_0vor(
                e3t_0, tmask, grid=recipe.grid, dtype=jnp.float64)
            mesh = nemo_ldf_reference_e3f(z_coord)
            if plant == "card_reference":
                # Perturb EVERY cell, so the plant is guaranteed to reach a
                # vertex whose masked coefficient is non-zero; a single-cell
                # plant under land would be refused for the wrong reason.
                mesh = jnp.asarray(mesh) + 1.0
            row["mesh_reference_vs_vorticity_reference"] = score(
                mesh, vorticity)
            # dynldf_lev.f90:123 multiplies the reference by ahmf, which
            # ldfdyn.f90:387-393 has already multiplied by the F mask.  Where
            # that product is exactly zero the reference cannot reach the
            # tendency at ANY step, so the card is unmoved for every day, not
            # only for the ten scored ones.
            differing = np.asarray(mesh, np.float64) != np.asarray(
                vorticity, np.float64)
            ahmf = getattr(z_coord, "nemo_ldf_ahmf", None)
            if ahmf is None:
                from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                    nemo_lateral_viscosity_coefficients,
                )
                half_um = (config.lateral_viscosity.A_h
                           / (recipe.grid.radius * recipe.grid.dlon))
                ahmf = nemo_lateral_viscosity_coefficients(
                    recipe.grid, half_um)[1]
            ahmf = np.asarray(ahmf, np.float64)
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                compute_vertex_mask,
            )
            import jax as _jax
            vertex3 = np.asarray(_jax.vmap(
                lambda m2: compute_vertex_mask(m2, grid=recipe.grid),
                in_axes=-1, out_axes=-1)(
                    jnp.asarray(tmask) * state.land_mask.data[..., None]),
                np.float64)
            surface = np.asarray(compute_vertex_mask(
                state.land_mask.data, grid=recipe.grid), np.float64)
            vertex3 = vertex3 * surface[..., None]
            if ahmf.ndim == 1:
                ahmf = ahmf[:, None, None]
            coefficient = np.abs(np.broadcast_to(
                ahmf, vertex3.shape) * vertex3)
            # The reference lives on the (n_lat+1, n_lon+1) vertex layout the
            # operator reads; the native comparison above is A2D, so map the
            # differing flags the same way the builder's tail does.
            with_south = np.concatenate([differing[:1], differing], axis=0)
            differing_vertex = np.concatenate(
                [with_south[:, -1:], with_south], axis=1)
            row["differing_reference_cells"] = int(differing.sum())
            row["max_coefficient_at_differing_cells"] = float(
                coefficient[differing_vertex].max(initial=0.0))
        rows[name] = row
    return {
        "status": "CAPTURED",
        "claim_label": "card mesh, no trajectory",
        "cards": rows,
        "plant": plant,
        "worktree": worktree_stamp(),
    }


def evaluate_shared_cards(capture: dict) -> dict:
    reasons = []
    verdict = "PASS"
    for name, row in capture["cards"].items():
        if not row["executes_the_changed_statement"]:
            continue
        unequal = row["mesh_reference_vs_vorticity_reference"]["unequal"]
        coefficient = row.get("max_coefficient_at_differing_cells")
        if unequal and coefficient != 0.0:
            verdict = "MOVES"
            reasons.append(
                f"{name} moves: {unequal} unequal reference cells reach a "
                f"non-zero coefficient ({coefficient})")
    return {**capture, "status": verdict, "reasons": reasons}


# Round 28's measured parent kt=10 stage-3 velocity maxima, quoted from its
# receipt.  The landing must cut them, not merely change them.
R28_PARENT_KT10_STAGE3 = {"u": 15.365503106245665, "v": 42.669598831454074}


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text())


def _row(document: dict, kt: int, checkpoint: str, field: str) -> dict:
    for item in document["candidate_trajectory"]["checkpoints"]:
        if item["kt"] == kt and item["checkpoint"] == checkpoint:
            return item["rows"][field]
    raise GateError(f"missing score kt={kt}:{checkpoint}:{field}")


def evaluate_outcome(parent_path: Path, arm_path: Path, cards_path: Path,
                     direct_path: Path, *, plant: str | None = None) -> dict:
    import copy

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round25_outcome_gate as r25,
    )

    parent = _read(parent_path)
    arm = _read(arm_path)
    cards = _read(cards_path)
    direct = _read(direct_path)
    require(parent["status"] == "LADDER_MEASURED", "parent is not measured")
    require(arm["status"] == "LADDER_MEASURED", "arm is not measured")
    require(len(parent["candidate_trajectory"]["checkpoints"]) == 40,
            "parent is not the ten-step/40-checkpoint ladder")

    if plant == "arm_refusal":
        arm = copy.deepcopy(arm)
        arm["candidate_trajectory"]["checkpoints"] = (
            arm["candidate_trajectory"]["checkpoints"][:12])
    if plant == "worse_velocity":
        arm = copy.deepcopy(arm)
        for field in ("u", "v"):
            _row(arm, 10, "stage3", field)["max_abs"] = 1.0e3

    comparison = r25._compare(parent, arm, "ldf_reference")
    velocities = {
        field: {
            "parent": float(_row(parent, 10, "stage3", field)["max_abs"]),
            "arm": float(_row(arm, 10, "stage3", field)["max_abs"]),
            "round28_parent": R28_PARENT_KT10_STAGE3[field],
        }
        for field in ("u", "v")
    }
    moved = comparison["moved_row_count"]
    reasons = []
    verdict = "LANDED"
    if comparison["checkpoint_count"] != 40:
        verdict = "HELD"
        reasons.append("the arm did not complete kt=1..10")
    if comparison["at_bar_rows_left"]:
        verdict = "HELD"
        reasons.append("a formerly bit-identical row left the bar")
    if not comparison["first_non_bit_statement_unchanged"]:
        verdict = "HELD"
        reasons.append("the first non-bit statement changed")
    for field, row in velocities.items():
        if row["arm"] >= row["parent"]:
            verdict = "HELD"
            reasons.append(f"kt=10 stage-3 {field} maximum did not fall")
    if cards["status"] != "PASS":
        verdict = "HELD"
        reasons.append("a shared card moves")
    if direct["status"] != "PASS":
        verdict = "HELD"
        reasons.append("the given-entry control did not pass")
    if not moved:
        verdict = "HELD"
        reasons.append("the arm is inert")

    predictions = {
        "R31-P2": direct["exposed_stage2_een_digest"] == direct[
            "round28_parent_een_digest"],
        "R31-P3": (comparison["checkpoint_count"] == 40 and all(
            row["arm"] < row["parent"] / 10.0
            for row in velocities.values())),
        "R31-P5": cards["status"] == "PASS",
    }
    return {
        "status": verdict,
        "reasons": reasons,
        "claim_label": "independent with Decision-52 SSH",
        "parent_commit": parent["worktree"]["commit"],
        "arm_commit": arm["worktree"]["commit"],
        "kt10_stage3_velocity_maxima": velocities,
        "ladder": {key: value for key, value in comparison.items()
                   if key != "moved_rows"},
        "moved_rows": comparison["moved_rows"],
        "shared_cards_status": cards["status"],
        "given_entry_status": direct["status"],
        "predictions": {key: "CONFIRMED" if value else "REFUTED"
                        for key, value in predictions.items()},
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("e3f0vor")
    cap.add_argument("--deck-root", type=Path, required=True)
    cap.add_argument("--json-out", type=Path, required=True)
    cap.add_argument("--plant", choices=("substitute_operand",))
    direct = sub.add_parser("direct-ldf")
    direct.add_argument("--deck-root", type=Path, required=True)
    direct.add_argument("--record-root", type=Path, required=True)
    direct.add_argument("--json-out", type=Path, required=True)
    direct.add_argument("--plant", choices=("een_digest",))
    cards = sub.add_parser("shared-cards")
    cards.add_argument("--json-out", type=Path, required=True)
    cards.add_argument("--plant", choices=("card_reference",))
    outcome = sub.add_parser("outcome")
    outcome.add_argument("--parent", type=Path, required=True)
    outcome.add_argument("--arm", type=Path, required=True)
    outcome.add_argument("--shared-cards", type=Path, required=True)
    outcome.add_argument("--direct-ldf", type=Path, required=True)
    outcome.add_argument("--json-out", type=Path, required=True)
    outcome.add_argument("--plant", choices=("arm_refusal", "worse_velocity"))
    args = parser.parse_args()
    try:
        if args.command == "e3f0vor":
            result = evaluate_e3f0vor(
                capture_e3f0vor(args.deck_root, plant=args.plant))
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True))
            print(json.dumps(result["statements"], indent=1, sort_keys=True))
            print(json.dumps(
                {"carried_e3f_0_vs_transcribed_e3f_0vor":
                 result["carried_e3f_0_vs_transcribed_e3f_0vor"]},
                indent=1, sort_keys=True))
            print(result["status"], result["reasons"])
            return 0 if result["status"] == "PASS" else 2
        if args.command == "outcome":
            result = evaluate_outcome(
                args.parent, args.arm, args.shared_cards, args.direct_ldf,
                plant=args.plant)
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True))
            print(json.dumps(result["kt10_stage3_velocity_maxima"], indent=1,
                             sort_keys=True))
            print(json.dumps(result["ladder"], indent=1, sort_keys=True))
            print(json.dumps(result["predictions"], indent=1, sort_keys=True))
            print(result["status"], result["reasons"])
            return 0 if result["status"] == "LANDED" else 2
        if args.command == "shared-cards":
            result = evaluate_shared_cards(
                capture_shared_cards(plant=args.plant))
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True))
            print(json.dumps(result["cards"], indent=1, sort_keys=True))
            print(result["status"], result["reasons"])
            return 0 if result["status"] == "PASS" else 2
        if args.command == "direct-ldf":
            result = evaluate_direct_ldf(capture_direct_ldf(
                args.deck_root, args.record_root, plant=args.plant))
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True))
            print(json.dumps(result["builder_call_census"], indent=1,
                             sort_keys=True))
            print("een digest", result["exposed_stage2_een_digest"])
            print(result["status"], result["reasons"])
            return 0 if result["status"] == "PASS" else 2
    except GateError as exc:
        print(f"GATE REFUSED: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
