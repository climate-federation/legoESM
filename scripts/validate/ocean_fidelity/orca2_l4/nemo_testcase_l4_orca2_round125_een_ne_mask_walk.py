#!/usr/bin/env python3
"""Gate northeast EEN's northern frozen-mask association."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition import (
    check_record as record_gate,
)


PLANTS = (
    "none", "oracle-bit", "candidate-bit", "wrong-row",
    "northwest-isolation", "scope-route",
)
EXPECTED_BEFORE = {
    "1_e3f0": (0, 0),
    "1_r3f": (0, 0),
    "1_mask": (1160, 1160),
    "1_denom": (0, 0),
    "1_frac": (0, 0),
    "partial": (0, 0),
    "sum": (0, 0),
}
EXPECTED_AFTER = {name: (0, 0) for name in EXPECTED_BEFORE}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _replace_ne_north_mask(
    fields: dict[str, np.ndarray], north_mask: np.ndarray,
    executed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replace only NE fraction 1's northern mask and its descendants."""
    result = {name: np.array(value, copy=True) for name, value in fields.items()}
    mask = np.array(result["1_mask"], copy=True)
    mask[-1] = np.asarray(north_mask, dtype=np.float64)
    result["1_mask"] = np.where(executed, mask, np.float64(0.0))

    one_plus = np.add(
        np.float64(1.0), np.multiply(result["1_r3f"], result["1_mask"])
    )
    denom = np.multiply(result["1_e3f0"], one_plus)
    result["1_denom"] = np.where(executed, denom, np.float64(0.0))
    fraction = np.zeros_like(denom)
    np.divide(result["1_ff"], denom, out=fraction, where=executed)
    result["1_frac"] = fraction
    result["partial"] = np.where(
        executed, np.add(fraction, result["2_frac"]), np.float64(0.0)
    )
    result["sum"] = np.where(
        executed, np.add(result["partial"], result["3_frac"]), np.float64(0.0)
    )
    return result


def _scores(candidate: dict[str, np.ndarray], oracle: dict[str, np.ndarray],
            path: str) -> dict[str, dict]:
    return {
        name: r109._score(candidate[name], oracle[f"{path}_{name}"])
        for name in r121.SOURCE_ORDER
    }


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
    require(stamp["clean"], "round-125 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-125 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-125 walk requires production JIT on CPU")

    admission = record_gate.run(record_root, source_root, "none")
    require(admission["status"] == "PASS_R120_EEN_V_FRACTION_ADMISSION",
            "round-120 record is not admitted")
    oracle, census = r121.assemble_record(record_root, "none")
    if plant == "oracle-bit":
        oracle = dict(oracle)
        changed = np.array(oracle["ne_1_mask"], copy=True)
        changed.view(np.uint64)[-1, 29, 0] ^= np.uint64(1)
        oracle["ne_1_mask"] = changed

    inherited_scope = r112.resolved_card_scope(deck_root)
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in inherited_scope.items()
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
    require(r109._score(bottom, oracle["mbkv"])["bit_unequal"] == 0,
            "mbkv moved before the northern mask association")
    executed = np.arange(1, 31)[None, None, :] <= bottom[..., None]
    ff = np.asarray(parts["een_ff"], dtype=np.float64)
    north_ff = np.asarray(_nemo_een_north_f(
        jnp.asarray(ff), card.recipe.grid
    ))[-1]
    baseline = r122._candidate_fields(parts, executed, north_ff)
    fold_perm = np.asarray(fold_perm_f(card.recipe.grid.fold), dtype=np.int64)
    source_e3f = np.asarray(parts["een_e3f0"], dtype=np.float64)[-3, fold_perm]
    candidate_ne = r123._replace_ne_north_e3f(
        baseline["ne"], source_e3f, executed
    )
    candidate_nw = r124._replace_nw_north_e3f(
        baseline["nw"], source_e3f, executed
    )

    source_row = -2 if plant == "wrong-row" else -3
    north_mask = np.asarray(parts["een_fmask"], dtype=np.float64)[
        source_row, fold_perm
    ]
    before_ne = {name: np.array(value, copy=True)
                 for name, value in candidate_ne.items()}
    candidate_ne = _replace_ne_north_mask(candidate_ne, north_mask, executed)
    if plant == "candidate-bit":
        candidate_ne["1_mask"].view(np.uint64)[-1, 29, 0] ^= np.uint64(1)
    if plant == "northwest-isolation":
        candidate_nw["3_mask"].view(np.uint64)[-1, 30, 0] ^= np.uint64(1)

    before = _scores(before_ne, oracle, "ne")
    after = _scores(candidate_ne, oracle, "ne")
    for name, expected in EXPECTED_BEFORE.items():
        observed = (before[name]["bit_unequal"], before[name]["magnitude_unequal"])
        require(observed == expected, f"NE before census moved at {name}: {observed}")
    for name, expected in EXPECTED_AFTER.items():
        observed = (after[name]["bit_unequal"], after[name]["magnitude_unequal"])
        require(observed == expected, f"NE after census moved at {name}: {observed}")
    first = next((name for name in r121.SOURCE_ORDER
                  if after[name]["bit_unequal"]), None)
    require(first is None, f"NE remains non-bit after mask arm: {first}")

    nw_scores = _scores(candidate_nw, oracle, "nw")
    for name in ("3_e3f0", "3_denom", "3_frac", "sum"):
        require(nw_scores[name]["bit_unequal"] == 0,
                f"northwest isolation moved at {name}")
    require((nw_scores["3_mask"]["bit_unequal"],
             nw_scores["3_mask"]["magnitude_unequal"]) == (1154, 1154),
            "northwest isolated mask census moved")

    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R125_EEN_NE_MASK_ASSOCIATION",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "record_census": census,
        "card_scope": observed_scope,
        "ne": {
            "first_after_arm": first,
            "association": {
                "target_j": 147,
                "source_j": 145,
                "source_i_first_last": [int(fold_perm[0]), int(fold_perm[-1])],
                "sign": 1,
            },
            "before": {name: before[name] for name in EXPECTED_BEFORE},
            "after": {name: after[name] for name in EXPECTED_AFTER},
        },
        "nw_isolation": {
            name: nw_scores[name]
            for name in ("3_e3f0", "3_mask", "3_denom", "3_frac", "sum")
        },
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
