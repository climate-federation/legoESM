#!/usr/bin/env python3
"""Gate ORCA2's source-exact southern ff_f association in the EEN triad."""

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
    nemo_testcase_l4_orca2_round111_een_fraction_walk as r111,
)

PLANTS = ("none", "oracle-bit", "candidate-bit", "scope-route")
EXPECTED_CARD_SCOPE = {
    "DINO-nemo_dino_kamm": True,
    "DINO-nemo_dino_kamm_mlf": True,
    "ORCA2-zps": True,
    "GYRE-zco": False,
    "LOCK_EXCHANGE-zco": False,
    "OVERFLOW-zps": False,
    "VORTEX-zco": True,
    "VORTEX_VEC-zco": True,
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def resolved_card_scope(deck_root: Path) -> dict[str, dict[str, object]]:
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        nemo_faithful_dino_domain,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        build_lock_exchange_zco_card,
        build_orca2_zps_card,
        build_overflow_zps_card,
        build_vortex_zco_card,
    )

    cards = {
        "ORCA2-zps": build_orca2_zps_card(deck_root),
        "GYRE-zco": build_gyre_zco_card(),
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card(),
        "OVERFLOW-zps": build_overflow_zps_card(),
        "VORTEX-zco": build_vortex_zco_card(),
        "VORTEX_VEC-zco": build_vortex_zco_card("vector"),
    }
    rows = {}
    for name, card in cards.items():
        cfg = card.recipe.model_config
        has_operands = card.recipe.z_coord.nemo_een_barotropic is not None
        rows[name] = {
            "barotropic_coriolis": cfg.barotropic.barotropic_coriolis,
            "coefficient_evaluation": (
                cfg.barotropic.barotropic_een_coefficient_evaluation),
            "has_literal_operands": has_operands,
            "executes_southern_een_ff_association": bool(
                cfg.barotropic.barotropic_coriolis == "een_metric"
                and cfg.barotropic.barotropic_een_coefficient_evaluation
                == "nemo_literal"
                and has_operands),
        }
    dino_domain = nemo_faithful_dino_domain()
    dino_has_operands = dino_domain.z_coord.nemo_een_barotropic is not None
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        cfg = dino_config_for_recipe(recipe)
        rows[f"DINO-{recipe}"] = {
            "barotropic_coriolis": cfg.barotropic_coriolis,
            "coefficient_evaluation": (
                cfg.barotropic_een_coefficient_evaluation),
            "has_literal_operands": dino_has_operands,
            "executes_southern_een_ff_association": bool(
                cfg.barotropic_coriolis == "een_metric"
                and cfg.barotropic_een_coefficient_evaluation
                == "nemo_literal"
                and dino_has_operands),
        }
    return rows


def measure(deck_root: Path, frame_root: Path, fraction_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-112 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-112 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-112 walk requires production JIT on CPU")

    card_scope = resolved_card_scope(deck_root)
    if plant == "scope-route":
        card_scope["GYRE-zco"]["executes_southern_een_ff_association"] = True
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in card_scope.items()
    }
    require(observed_scope == EXPECTED_CARD_SCOPE,
            "resolved card scope moved")

    oracle, census = r111.assemble_record(fraction_root, "none")
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

    def replay(copy_south: bool):
        _, parts = jax.device_get(jax.jit(
            lambda value: r107.literal_accumulators(
                value, source_z, jnp.float64, grid=card.recipe.grid,
                source_face_thickness=True, literal_bottom_loop=True,
                return_fraction_operands=True, south_ff_copy=copy_south),
        )(jnp.asarray(state.eta.data, dtype=jnp.float64)))
        return parts

    baseline_parts = replay(False)
    candidate_parts = replay(True)
    mbku = np.asarray(candidate_parts["mbku"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= mbku[..., None]

    def fields(parts: dict, copy_south_ff: bool) -> dict[str, np.ndarray]:
        ff = np.asarray(parts["een_ff"])
        e3f0 = np.asarray(parts["een_e3f0"])
        r3f = np.asarray(parts["een_r3f"])
        mask = np.asarray(parts["een_fmask"])
        denom = np.asarray(parts["een_denom"])
        frac = np.asarray(parts["een_q"])

        def south(value):
            return np.roll(value, 1, axis=0)

        south_ff = np.concatenate([ff[:1], ff[:-1]], axis=0) if copy_south_ff else south(ff)
        south_denom = south(denom)
        south_frac = nemo_source_round(
            south_ff[..., None] / south_denom)
        west_frac = np.roll(frac, 1, axis=1)
        center_frac = frac
        rows = {
            "west_ff": np.broadcast_to(np.roll(ff, 1, axis=1)[..., None], executed.shape),
            "west_e3f0": np.roll(e3f0, 1, axis=1),
            "west_r3f": np.broadcast_to(np.roll(r3f, 1, axis=1)[..., None], executed.shape),
            "west_mask": np.roll(mask, 1, axis=1),
            "west_denom": np.roll(denom, 1, axis=1),
            "frac_west": west_frac,
            "center_ff": np.broadcast_to(ff[..., None], executed.shape),
            "center_e3f0": e3f0,
            "center_r3f": np.broadcast_to(r3f[..., None], executed.shape),
            "center_mask": mask,
            "center_denom": denom,
            "frac_center": center_frac,
            "south_ff": np.broadcast_to(south_ff[..., None], executed.shape),
            "south_e3f0": south(e3f0),
            "south_r3f": np.broadcast_to(south(r3f)[..., None], executed.shape),
            "south_mask": south(mask),
            "south_denom": south_denom,
            "frac_south": south_frac,
        }
        rows["sum_west_center"] = nemo_source_round(west_frac + center_frac)
        rows["sum_all"] = nemo_source_round(rows["sum_west_center"] + south_frac)
        return {name: np.where(executed, value, np.float64(0.0))
                for name, value in rows.items()}

    baseline = fields(baseline_parts, False)
    candidate = fields(candidate_parts, True)
    if plant == "oracle-bit":
        changed = np.array(oracle["south_ff"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        oracle = dict(oracle, south_ff=changed)
    if plant == "candidate-bit":
        changed = np.array(candidate["south_ff"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        candidate = dict(candidate, south_ff=changed)

    baseline_scores = {name: r109._score(baseline[name], oracle[name])
                       for name in r111.SOURCE_ORDER}
    candidate_scores = {name: r109._score(candidate[name], oracle[name])
                        for name in r111.SOURCE_ORDER}
    baseline_first = next(name for name in r111.SOURCE_ORDER
                          if baseline_scores[name]["bit_unequal"])
    candidate_first = next(name for name in r111.SOURCE_ORDER
                           if candidate_scores[name]["bit_unequal"])
    require(baseline_first == "south_ff", "baseline first boundary moved")
    require(baseline_scores["south_ff"]["magnitude_unequal"] == 180,
            "baseline south_ff census moved")
    require(candidate_scores["south_ff"]["bit_unequal"] == 0,
            "source-copy south_ff is not bit-exact")
    require(candidate_first == "south_e3f0",
            "candidate did not stop at the next registered operand")
    require(candidate_scores["south_e3f0"]["bit_unequal"] == 7,
            "next south_e3f0 census moved")
    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "PASS_R112_EEN_SOUTH_FF_WALK",
        "claim_label": "given NEMO's recorded entry",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "card_scope": card_scope,
        "baseline_first_non_bit_item": baseline_first,
        "candidate_first_non_bit_item": candidate_first,
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "dtypes": {
            "state_eta": str(np.asarray(state.eta.data).dtype),
            "geometry_e3t0": str(np.asarray(e3t0).dtype),
            "een_ff": str(np.asarray(candidate_parts["een_ff"]).dtype),
        },
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--fraction-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.fraction_root,
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
