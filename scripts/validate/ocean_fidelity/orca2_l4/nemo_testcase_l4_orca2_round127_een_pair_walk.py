#!/usr/bin/env python3
"""Resume the northern V EEN walk after the quotient operands close."""

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
    nemo_testcase_l4_orca2_round109_een_per_level_walk as r109,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round112_een_south_ff_walk as r112,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round115_een_product_walk as r115,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round119_een_v_recurrence_walk as r119,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round121_een_v_fraction_walk as r121,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round122_een_north_ff_walk as r122,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round123_een_ne_e3f_walk as r123,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round124_een_nw_e3f_walk as r124,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round125_een_ne_mask_walk as r125,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round126_een_nw_mask_walk as r126,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition import (
    check_record as record_gate,
)


PATHS = ("nw", "ne")
SOURCE_ORDER = r119.SOURCE_ORDER
PLANTS = ("none", "oracle-bit", "candidate-bit", "scope-route")
EXPECTED_FIRST = {"nw": "e3u", "ne": "e3u"}
EXPECTED_PREFIX = {
    "nw": {"mbkv": (0, 0), "zpvo": (0, 0), "e3v": (0, 0),
           "e3u": (95, 95), "mask": (1270, 1270)},
    "ne": {"mbkv": (0, 0), "zpvo": (0, 0), "e3v": (0, 0),
           "e3u": (91, 91), "mask": (1283, 1283)},
}


class GateError(RuntimeError):
    """The admitted records cannot support the source-ordered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def literal_term(face, neighbor, neighbor_mask, quotient):
    """Replay the compiled four-factor EEN recurrence addend."""
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    return b(b(b(face * neighbor) * neighbor_mask) * quotient)


def _value_bits(value: np.ndarray, location: tuple[int, int, int]) -> dict:
    scalar = np.asarray(value[location], dtype=np.float64)
    return {"value": float(scalar), "bits": f"0x{scalar.view(np.uint64):016x}"}


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            source_root: Path, expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.operators_latlon_cgrid import fold_perm_f
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_een_north_f
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-127 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-127 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-127 walk requires production JIT on CPU")

    admission = record_gate.run(record_root, source_root, "none")
    require(admission["status"] == "PASS_R120_EEN_V_FRACTION_ADMISSION",
            "round-120 record is not admitted")
    fraction_oracle, fraction_census = r121.assemble_record(record_root, "none")
    recurrence_oracle, recurrence_census = r119.assemble_record(record_root, "none")
    if plant == "oracle-bit":
        recurrence_oracle = dict(recurrence_oracle)
        changed = np.array(recurrence_oracle["e3u_nw"], copy=True)
        changed.view(np.uint64)[-1, 30, 0] ^= np.uint64(1)
        recurrence_oracle["e3u_nw"] = changed

    scope = r112.resolved_card_scope(deck_root)
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in scope.items()
    }
    if plant == "scope-route":
        observed_scope["GYRE-zco"] = True
    require(observed_scope == r112.EXPECTED_CARD_SCOPE,
            "resolved literal-EEN card scope moved")

    card = r122.rung0.build_rung0_card(deck_root)
    r122.rung0.validate_rung0_card(card)
    state = r122.rung0.bridge_entry(
        card, r122.rung0.assemble_frame(frame_root, 1, 0)
    )
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
    _, parts = jax.device_get(jax.jit(
        lambda value: r122.r107.literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True, literal_bottom_loop=True,
            return_fraction_operands=True, south_ff_copy=True,
            south_e3f0_fill=jnp.asarray(raw.e3f_0, dtype=jnp.float64),
            south_mask_zero=True,
        )
    )(jnp.asarray(state.eta.data, dtype=jnp.float64)))

    bottom = np.asarray(parts["mbkv"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= bottom[..., None]
    ff = np.asarray(parts["een_ff"], dtype=np.float64)
    north_ff = np.asarray(_nemo_een_north_f(
        jnp.asarray(ff), card.recipe.grid
    ))[-1]
    fractions = r122._candidate_fields(parts, executed, north_ff)
    fold_perm = np.asarray(fold_perm_f(card.recipe.grid.fold), dtype=np.int64)
    north_e3f = np.asarray(parts["een_e3f0"], dtype=np.float64)[-3, fold_perm]
    north_mask = np.asarray(parts["een_fmask"], dtype=np.float64)[-3, fold_perm]
    fractions["ne"] = r123._replace_ne_north_e3f(
        fractions["ne"], north_e3f, executed)
    fractions["ne"] = r125._replace_ne_north_mask(
        fractions["ne"], north_mask, executed)
    fractions["nw"] = r124._replace_nw_north_e3f(
        fractions["nw"], north_e3f, executed)
    fractions["nw"] = r126._replace_nw_north_mask(
        fractions["nw"], north_mask, executed)

    rows = {}
    for path in PATHS:
        fraction_scores = r126._scores(fractions[path], fraction_oracle, path)
        require(all(row["bit_unequal"] == 0 for row in fraction_scores.values()),
                f"{path}: a quotient operand reopened before the recurrence")
        face = np.asarray(parts["source_e3v"], dtype=np.float64)
        neighbor = np.asarray(parts[f"neighbor_e3u_v_{path}"], dtype=np.float64)
        neighbor_mask = np.asarray(
            parts[f"neighbor_mask_v_{path}"], dtype=np.float64)
        quotient = np.asarray(fractions[path]["sum"], dtype=np.float64)
        term = np.asarray(jax.device_get(jax.jit(literal_term)(
            jnp.asarray(face), jnp.asarray(neighbor),
            jnp.asarray(neighbor_mask), jnp.asarray(quotient))))
        before, after = jax.device_get(jax.jit(
            lambda value, mbk: r115.replay_recurrence(
                value, mbk, ieee_zero_add=True))(
                    jnp.asarray(term), jnp.asarray(bottom)))
        candidate = {
            "mbkv": bottom,
            "zpvo": quotient,
            "e3v": face,
            "e3u": neighbor,
            "mask": neighbor_mask,
            "term": term,
            "before": np.asarray(before),
            "after": np.asarray(after),
        }
        reference = {"mbkv": recurrence_oracle["mbkv"]}
        for name in SOURCE_ORDER[1:]:
            candidate[name] = np.where(
                executed, candidate[name], np.float64(0.0))
            reference[name] = recurrence_oracle[f"{name}_{path}"]
        if plant == "candidate-bit" and path == "ne":
            candidate["zpvo"] = np.array(candidate["zpvo"], copy=True)
            candidate["zpvo"].view(np.uint64)[-1, 30, 0] ^= np.uint64(1)
        scores = {name: r109._score(candidate[name], reference[name])
                  for name in SOURCE_ORDER}
        first = next((name for name in SOURCE_ORDER
                      if scores[name]["bit_unequal"]), None)
        require(first == EXPECTED_FIRST[path],
                f"{path}: first unequal item moved: {first}")
        for name, expected in EXPECTED_PREFIX[path].items():
            observed = (scores[name]["bit_unequal"],
                        scores[name]["magnitude_unequal"])
            require(observed == expected,
                    f"{path}: {name} census moved: {observed}")
        require(scores[first]["bit_unequal_j_values"] == [147],
                f"{path}: first boundary escaped the northern fold")
        locations = np.argwhere(
            np.ascontiguousarray(candidate[first]).view(np.uint64)
            != np.ascontiguousarray(reference[first]).view(np.uint64))
        samples = []
        for raw_location in locations[:5]:
            location = tuple(map(int, raw_location))
            samples.append({
                "j_i_k": list(location),
                "oracle": _value_bits(reference[first], location),
                "candidate": _value_bits(candidate[first], location),
            })
        rows[path] = {
            "first_non_bit_item": first,
            "scores": scores,
            "first_boundary_samples": samples,
        }

    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R127_EEN_PAIR_PREFIX",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "fraction_record_census": fraction_census,
        "recurrence_record_census": recurrence_census,
        "card_scope": observed_scope,
        "source_order": list(SOURCE_ORDER),
        "paths": rows,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.record_root,
                         args.source_root, args.expect_commit, args.plant)
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
