#!/usr/bin/env python3
"""Walk rung-0 EEN stored-product signed zeros in compiled source order."""

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
    nemo_testcase_l4_orca2_round107_een_u_operand_walk as r107,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round109_een_per_level_walk as r109,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round112_een_south_ff_walk as r112,
)


PLANTS = ("none", "oracle-bit", "candidate-bit", "scope-route")
SOURCE_ORDER = r109.SOURCE_ORDER


class GateError(RuntimeError):
    """The recorded product walk cannot support its registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def measure(deck_root: Path, frame_root: Path, step_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-115 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-115 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-115 walk requires production JIT on CPU")

    card_scope = r112.resolved_card_scope(deck_root)
    if plant == "scope-route":
        card_scope["GYRE-zco"]["executes_southern_een_ff_association"] = True
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in card_scope.items()
    }
    require(observed_scope == r112.EXPECTED_CARD_SCOPE,
            "resolved card scope moved")

    oracle, census = r109.assemble_record(step_root, "none")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried operands")
    e3t0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t0, card.recipe.z_coord.is_active, grid=card.recipe.grid,
        dtype=jnp.float64, substitute_e3f=raw.e3f_0,
    )
    source_z = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor),
    )

    def product_arms(value):
        _, parts = r107.literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True, literal_bottom_loop=True,
            south_ff_copy=True,
            south_e3f0_fill=jnp.asarray(raw.e3f_0, dtype=jnp.float64),
            south_mask_zero=True,
        )
        literal_product = nemo_source_round(
            parts["source_e3u"] * parts["source_e3v"]
            * jnp.asarray(raw.vmask, dtype=jnp.float64)
            * parts["zpvo_u_nw"]
        )
        return parts, literal_product

    parts, literal_product = jax.device_get(jax.jit(product_arms)(
        jnp.asarray(state.eta.data, dtype=jnp.float64)))
    mbku = np.asarray(parts["mbku"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= mbku[..., None]

    def recurrence(term, bottom):
        acc = jnp.zeros(term.shape[:2], dtype=term.dtype)
        before = jnp.zeros_like(term)
        after = jnp.zeros_like(term)
        for jk in range(term.shape[-1]):
            before = before.at[..., jk].set(acc)
            updated = nemo_source_round(acc + term[..., jk])
            acc = jnp.where(jk < bottom, updated, acc)
            after = after.at[..., jk].set(acc)
        return before, after

    baseline_before, baseline_after = jax.device_get(jax.jit(recurrence)(
        jnp.asarray(parts["term_u_nw"]), jnp.asarray(parts["mbku"])))
    candidate_before, candidate_after = jax.device_get(jax.jit(recurrence)(
        jnp.asarray(literal_product), jnp.asarray(parts["mbku"])))
    prefix = {
        "mbku": mbku,
        "zpvo_nw": np.asarray(parts["zpvo_u_nw"]),
        "e3u_live": np.asarray(parts["source_e3u"]),
        "e3v_live": np.asarray(parts["source_e3v"]),
        "neighbor_mask": np.asarray(raw.vmask, dtype=np.float64),
    }
    baseline = {
        **prefix,
        "term_nw": np.asarray(parts["term_u_nw"]),
        "acc_before": np.asarray(baseline_before),
        "acc_after": np.asarray(baseline_after),
    }
    candidate = {
        **prefix,
        "term_nw": np.asarray(literal_product),
        "acc_before": np.asarray(candidate_before),
        "acc_after": np.asarray(candidate_after),
    }
    for fields in (baseline, candidate):
        for name in SOURCE_ORDER[1:]:
            fields[name] = np.where(executed, fields[name], np.float64(0.0))

    if plant == "oracle-bit":
        changed = np.array(oracle["term_nw"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        oracle = dict(oracle, term_nw=changed)
    if plant == "candidate-bit":
        changed = np.array(candidate["term_nw"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        candidate = dict(candidate, term_nw=changed)

    baseline_scores = {
        name: r109._score(baseline[name], oracle[name]) for name in SOURCE_ORDER
    }
    candidate_scores = {
        name: r109._score(candidate[name], oracle[name]) for name in SOURCE_ORDER
    }
    baseline_first = next(
        (name for name in SOURCE_ORDER if baseline_scores[name]["bit_unequal"]), None)
    candidate_first = next(
        (name for name in SOURCE_ORDER if candidate_scores[name]["bit_unequal"]), None)
    require(baseline_first == "term_nw",
            "baseline first boundary moved: "
            f"{baseline_first} {baseline_scores.get(baseline_first)}")
    require(baseline_scores["term_nw"] == {
        "bit_unequal": 2,
        "magnitude_unequal": 0,
        "signed_zero_only": 2,
        "first_bit_unequal_j_i_k": baseline_scores["term_nw"]["first_bit_unequal_j_i_k"],
        "bit_unequal_j_values": baseline_scores["term_nw"]["bit_unequal_j_values"],
        "bit_unequal_k_values": baseline_scores["term_nw"]["bit_unequal_k_values"],
    }, "baseline stored-product census moved")
    if plant != "none":
        raise GateError(f"{plant} plant fired")

    return {
        "status": "MEASURED_R115_EEN_PRODUCT_WALK",
        "claim_label": "given NEMO's recorded entry",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "card_scope": card_scope,
        "source_order": list(SOURCE_ORDER),
        "baseline_first_non_bit_item": baseline_first,
        "candidate_first_non_bit_item": candidate_first,
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "shapes": {name: list(value.shape) for name, value in candidate.items()},
        "dtypes": {name: str(value.dtype) for name, value in candidate.items()},
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.step_root,
                         args.expect_commit, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
