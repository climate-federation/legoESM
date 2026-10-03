#!/usr/bin/env python3
"""Walk the northern U-grid operands in the literal EEN V recurrence."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round127_een_pair_walk as r127,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition import (
    check_record as record_gate,
)


PATHS = ("nw", "ne")
SOURCE_ORDER = r119.SOURCE_ORDER
ARMS = ("baseline", "thickness-only", "mask-only", "combined")
PLANTS = ("none", "oracle-bit", "candidate-bit", "wrong-row",
          "wrong-permutation", "scope-route")
EXPECTED_BASELINE = {
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


def northern_u_row(field: np.ndarray, perm_u: np.ndarray, *, source_offset: int = -2,
                   permutation: np.ndarray | None = None) -> np.ndarray:
    """Apply NEMO's scalar-sign U-grid T-pivot association to one row."""
    field = np.asarray(field)
    perm = perm_u if permutation is None else np.asarray(permutation)
    require(field.shape[1] == perm.shape[0], "U-fold permutation width moved")
    return np.asarray(field[source_offset, perm])


def replace_northern_neighbor(field: np.ndarray, north_row: np.ndarray,
                              path: str) -> np.ndarray:
    """Replace only the recorded ``jj+1`` U operand for NW or NE."""
    require(path in PATHS, f"unknown northern path {path}")
    out = np.array(field, copy=True)
    out[-1] = np.roll(north_row, 1, axis=0) if path == "nw" else north_row
    return out


def _source_rows(parts, fractions, path, bottom, executed, *, neighbor, mask):
    import jax
    import jax.numpy as jnp

    quotient = np.asarray(fractions[path]["sum"], dtype=np.float64)
    face = np.asarray(parts["source_e3v"], dtype=np.float64)
    term = np.asarray(jax.device_get(jax.jit(r127.literal_term)(
        jnp.asarray(face), jnp.asarray(neighbor), jnp.asarray(mask),
        jnp.asarray(quotient))))
    before, after = jax.device_get(jax.jit(
        lambda value, mbk: r115.replay_recurrence(
            value, mbk, ieee_zero_add=True))(
                jnp.asarray(term), jnp.asarray(bottom)))
    return {
        "mbkv": np.asarray(bottom, dtype=np.float64),
        "zpvo": quotient,
        "e3v": face,
        "e3u": np.asarray(neighbor, dtype=np.float64),
        "mask": np.asarray(mask, dtype=np.float64),
        "term": term,
        "before": np.asarray(before),
        "after": np.asarray(after),
    }


def _score_rows(candidate, reference, executed):
    scores = {}
    for name in SOURCE_ORDER:
        value = np.asarray(candidate[name], dtype=np.float64)
        oracle = np.asarray(reference[name], dtype=np.float64)
        if name != "mbkv":
            value = np.where(executed, value, np.float64(0.0))
            oracle = np.where(executed, oracle, np.float64(0.0))
        scores[name] = r109._score(value, oracle)
    return scores


def _first(scores):
    return next((name for name in SOURCE_ORDER
                 if scores[name]["bit_unequal"]), None)


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            source_root: Path, expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.operators_latlon_cgrid import fold_perm_f, fold_perm_u
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_een_north_f
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-128 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-128 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-128 walk requires production JIT on CPU")

    admission = record_gate.run(record_root, source_root, "none")
    require(admission["status"] == "PASS_R120_EEN_V_FRACTION_ADMISSION",
            "round-120 record is not admitted")
    fraction_oracle, fraction_census = r121.assemble_record(record_root, "none")
    recurrence_oracle, recurrence_census = r119.assemble_record(record_root, "none")
    if plant == "oracle-bit":
        recurrence_oracle = dict(recurrence_oracle)
        changed = np.array(recurrence_oracle["e3u_nw"], copy=True)
        changed.view(np.uint64)[-1, 30, 3] ^= np.uint64(1)
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
        card, r122.rung0.assemble_frame(frame_root, 1, 0))
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried operands")
    e3t0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t0, card.recipe.z_coord.is_active, grid=card.recipe.grid,
        dtype=jnp.float64, substitute_e3f=raw.e3f_0)
    source_z = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor))
    _, parts = jax.device_get(jax.jit(
        lambda value: r122.r107.literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True, literal_bottom_loop=True,
            return_fraction_operands=True, south_ff_copy=True,
            south_e3f0_fill=jnp.asarray(raw.e3f_0, dtype=jnp.float64),
            south_mask_zero=True))(
                jnp.asarray(state.eta.data, dtype=jnp.float64)))

    bottom = np.asarray(parts["mbkv"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= bottom[..., None]
    ff = np.asarray(parts["een_ff"], dtype=np.float64)
    north_ff = np.asarray(_nemo_een_north_f(
        jnp.asarray(ff), card.recipe.grid))[-1]
    fractions = r122._candidate_fields(parts, executed, north_ff)
    perm_f = np.asarray(fold_perm_f(card.recipe.grid.fold), dtype=np.int64)
    north_e3f = np.asarray(parts["een_e3f0"], dtype=np.float64)[-3, perm_f]
    north_fmask = np.asarray(parts["een_fmask"], dtype=np.float64)[-3, perm_f]
    fractions["ne"] = r123._replace_ne_north_e3f(
        fractions["ne"], north_e3f, executed)
    fractions["ne"] = r125._replace_ne_north_mask(
        fractions["ne"], north_fmask, executed)
    fractions["nw"] = r124._replace_nw_north_e3f(
        fractions["nw"], north_e3f, executed)
    fractions["nw"] = r126._replace_nw_north_mask(
        fractions["nw"], north_fmask, executed)
    for path in PATHS:
        fraction_scores = r126._scores(fractions[path], fraction_oracle, path)
        require(all(row["bit_unequal"] == 0
                    for row in fraction_scores.values()),
                f"{path}: a quotient operand reopened before the U walk")

    perm_u = np.asarray(fold_perm_u(card.recipe.grid.fold), dtype=np.int64)
    source_offset = -3 if plant == "wrong-row" else -2
    association_perm = perm_f if plant == "wrong-permutation" else perm_u
    north_e3u = northern_u_row(
        np.asarray(parts["source_e3u"]), perm_u,
        source_offset=source_offset, permutation=association_perm)
    north_umask = northern_u_row(
        np.asarray(raw.umask, dtype=np.float64), perm_u,
        source_offset=source_offset, permutation=association_perm)

    rows = {}
    for path in PATHS:
        baseline_e3u = np.asarray(parts[f"neighbor_e3u_v_{path}"], dtype=np.float64)
        baseline_mask = np.asarray(parts[f"neighbor_mask_v_{path}"], dtype=np.float64)
        associated_e3u = replace_northern_neighbor(baseline_e3u, north_e3u, path)
        associated_mask = replace_northern_neighbor(baseline_mask, north_umask, path)
        reference = {"mbkv": recurrence_oracle["mbkv"]}
        reference.update({name: recurrence_oracle[f"{name}_{path}"]
                          for name in SOURCE_ORDER[1:]})
        arm_values = {
            "baseline": (baseline_e3u, baseline_mask),
            "thickness-only": (associated_e3u, baseline_mask),
            "mask-only": (baseline_e3u, associated_mask),
            "combined": (associated_e3u, associated_mask),
        }
        arm_rows = {}
        for arm, (neighbor, mask) in arm_values.items():
            candidate = _source_rows(
                parts, fractions, path, bottom, executed,
                neighbor=neighbor, mask=mask)
            if plant == "candidate-bit" and path == "ne" and arm == "combined":
                candidate["zpvo"] = np.array(candidate["zpvo"], copy=True)
                candidate["zpvo"].view(np.uint64)[-1, 30, 0] ^= np.uint64(1)
            scores = _score_rows(candidate, reference, executed)
            arm_rows[arm] = {
                "first_non_bit_item": _first(scores),
                "scores": scores,
            }
        for name, expected in EXPECTED_BASELINE[path].items():
            observed = tuple(arm_rows["baseline"]["scores"][name][key]
                             for key in ("bit_unequal", "magnitude_unequal"))
            require(observed == expected,
                    f"{path}: baseline {name} census moved: {observed}")
        rows[path] = arm_rows

    if plant == "oracle-bit":
        require(rows["nw"]["combined"]["scores"]["e3u"]["bit_unequal"] != 0,
                "oracle-bit plant stayed green")
    elif plant == "candidate-bit":
        require(rows["ne"]["combined"]["first_non_bit_item"] == "zpvo",
                "candidate-bit plant stayed green")
    elif plant in ("wrong-row", "wrong-permutation"):
        require(any(rows[path]["combined"]["scores"][name]["magnitude_unequal"]
                    for path in PATHS for name in ("e3u", "mask")),
                f"{plant} plant stayed green")
    if plant != "none":
        raise GateError(f"{plant} plant fired")

    combined_exact = {
        path: all(rows[path]["combined"]["scores"][name]["bit_unequal"] == 0
                  for name in SOURCE_ORDER)
        for path in PATHS
    }
    return {
        "status": "MEASURED_R128_EEN_U_FOLD_ASSOCIATION",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "association": {
            "source_offset": source_offset,
            "permutation": "U-origin",
            "sign": 1,
        },
        "admission": admission,
        "fraction_record_census": fraction_census,
        "recurrence_record_census": recurrence_census,
        "card_scope": observed_scope,
        "source_order": list(SOURCE_ORDER),
        "combined_all_rows_exact": combined_exact,
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
