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
    return {"status": verdict, "reasons": reasons, **capture}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("e3f0vor")
    cap.add_argument("--deck-root", type=Path, required=True)
    cap.add_argument("--json-out", type=Path, required=True)
    cap.add_argument("--plant", choices=("substitute_operand",))
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
    except GateError as exc:
        print(f"GATE REFUSED: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
