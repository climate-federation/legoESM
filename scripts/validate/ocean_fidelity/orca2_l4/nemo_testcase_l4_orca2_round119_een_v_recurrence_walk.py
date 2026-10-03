#!/usr/bin/env python3
"""Walk the rung-0 V EEN recurrences in compiled source order."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round115_een_product_walk as r115,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round118_een_v_recurrence_acquisition import (
    check_record as record_gate,
)


LABELS = record_gate.LABELS
SOURCE_ORDER = ("mbkv", "zpvo", "e3v", "e3u", "mask", "term", "before", "after")
PLANTS = ("none", "oracle-bit", "candidate-bit", "scope-route")


class GateError(RuntimeError):
    """The admitted V recurrence walk cannot support its claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def assemble_record(root: Path, plant: str) -> tuple[dict[str, np.ndarray], dict]:
    assembled = {
        name: np.zeros((148, 180, 30), dtype=np.float64)
        for name in record_gate.FIELDS[:-1]
    }
    assembled["mbkv"] = np.zeros((148, 180), dtype=np.float64)
    coverage = np.zeros((148, 180), dtype=np.int8)
    rows = []
    for rank in range(2):
        path = root / f"oracle_r118_een_v_rank{rank:04d}_kt00000001.bin"
        row = record_gate.read_record(path)
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        coverage[j0:j1, i0:i1] += 1
        groups = row["groups"]
        assembled["mbkv"][j0:j1, i0:i1] = groups["mbkv"][..., 0].T
        for name in record_gate.FIELDS[:-1]:
            values = groups[name]
            require(bool(np.all(values[..., 30] == 0.0)),
                    f"{path.name}: {name} writes NEMO's dummy level")
            if plant == "oracle-bit" and rank == 0 and name == "zpvo_nw":
                values = np.array(values, copy=True)
                values.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
            assembled[name][j0:j1, i0:i1] = values[..., :30].transpose(1, 0, 2)
        rows.append({"rank": rank, "sha256": row["sha256"], "owned": row["owned"]})
    require(bool(np.all(coverage == 1)), "rank slabs do not cover the domain once")
    return assembled, {"records": rows, "coverage": "exactly-once"}


def _value_bits(value: np.ndarray, location: tuple[int, int, int]) -> dict:
    scalar = np.asarray(value[location], dtype=np.float64)
    return {"value": float(scalar), "bits": f"0x{scalar.view(np.uint64):016x}"}


def measure(deck_root: Path, frame_root: Path, recurrence_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-119 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-119 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-119 walk requires production JIT on CPU")

    card_scope = r112.resolved_card_scope(deck_root)
    if plant == "scope-route":
        card_scope["GYRE-zco"]["executes_southern_een_ff_association"] = True
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in card_scope.items()
    }
    require(observed_scope == r112.EXPECTED_CARD_SCOPE, "resolved card scope moved")

    oracle, census = assemble_record(recurrence_root, plant)
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
    _, parts = jax.device_get(jax.jit(
        lambda value: r107.literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True, literal_bottom_loop=True,
            south_ff_copy=True,
            south_e3f0_fill=jnp.asarray(raw.e3f_0, dtype=jnp.float64),
            south_mask_zero=True,
        ),
    )(jnp.asarray(state.eta.data, dtype=jnp.float64)))
    bottom = np.asarray(parts["mbkv"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= bottom[..., None]

    source_e3v = np.asarray(parts["source_e3v"])

    rows = {}
    for label in LABELS:
        baseline_term = np.asarray(parts[f"term_v_{label}"])
        baseline_before, baseline_after = jax.device_get(jax.jit(
            lambda term, mbk: r115.replay_recurrence(
                term, mbk, ieee_zero_add=False))(
                    jnp.asarray(baseline_term), jnp.asarray(bottom)))
        candidate_before, candidate_after = jax.device_get(jax.jit(
            lambda term, mbk: r115.replay_recurrence(
                term, mbk, ieee_zero_add=True))(
                    jnp.asarray(baseline_term), jnp.asarray(bottom)))
        baseline = {
            "mbkv": bottom,
            "zpvo": np.asarray(parts[f"zpvo_v_{label}"]),
            "e3v": source_e3v,
            "e3u": np.asarray(parts[f"neighbor_e3u_v_{label}"]),
            "mask": np.asarray(parts[f"neighbor_mask_v_{label}"], dtype=np.float64),
            "term": baseline_term,
            "before": np.asarray(baseline_before),
            "after": np.asarray(baseline_after),
        }
        candidate = dict(baseline)
        candidate["before"] = np.asarray(candidate_before)
        candidate["after"] = np.asarray(candidate_after)
        reference = {"mbkv": oracle["mbkv"]}
        for short in SOURCE_ORDER[1:]:
            reference[short] = oracle[f"{short}_{label}"]
            baseline[short] = np.where(executed, baseline[short], np.float64(0.0))
            candidate[short] = np.where(executed, candidate[short], np.float64(0.0))
        if plant == "candidate-bit" and label == "se":
            candidate["after"] = np.array(candidate["after"], copy=True)
            candidate["after"].view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        baseline_scores = {
            name: r109._score(baseline[name], reference[name]) for name in SOURCE_ORDER
        }
        candidate_scores = {
            name: r109._score(candidate[name], reference[name]) for name in SOURCE_ORDER
        }
        movement = {
            name: r109._score(candidate[name], baseline[name])
            for name in ("before", "after")
        }
        first = next((name for name in SOURCE_ORDER
                      if baseline_scores[name]["bit_unequal"]), None)
        candidate_first = next((name for name in SOURCE_ORDER
                                if candidate_scores[name]["bit_unequal"]), None)
        samples = []
        if first is not None and first != "mbkv":
            locations = np.argwhere(
                np.ascontiguousarray(baseline[first]).view(np.uint64)
                != np.ascontiguousarray(reference[first]).view(np.uint64))[:5]
            for raw_location in locations:
                location = tuple(map(int, raw_location))
                samples.append({
                    "j_i_k": list(location),
                    "oracle": _value_bits(reference[first], location),
                    "baseline": _value_bits(baseline[first], location),
                    "candidate": _value_bits(candidate[first], location),
                })
        rows[label] = {
            "baseline_first_non_bit_item": first,
            "candidate_first_non_bit_item": candidate_first,
            "baseline_scores": baseline_scores,
            "candidate_scores": candidate_scores,
            "candidate_vs_baseline": movement,
            "first_boundary_samples": samples,
            "shapes": {name: list(candidate[name].shape) for name in SOURCE_ORDER},
            "dtypes": {name: str(candidate[name].dtype) for name in SOURCE_ORDER},
        }

    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R119_EEN_V_RECURRENCES_EXPLORATORY",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "candidate_arm": "host-ieee-zero-add-only",
        "record_census": census,
        "card_scope": card_scope,
        "source_order": list(SOURCE_ORDER),
        "recurrences": rows,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--recurrence-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.recurrence_root,
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
