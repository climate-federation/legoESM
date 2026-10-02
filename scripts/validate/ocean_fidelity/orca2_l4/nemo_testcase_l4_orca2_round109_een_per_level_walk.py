#!/usr/bin/env python3
"""Compare admitted rung-0 EEN per-level operands in compiled source order."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round107_een_step_acquisition import (
    check_record as record_gate,
)


PLANTS = ("none", "oracle-bit", "model-bit")
SOURCE_ORDER = ("zpvo_nw", "e3u_live", "e3v_live", "neighbor_mask", "term_nw")


class GateError(RuntimeError):
    """The admitted operand walk cannot support its registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.ascontiguousarray(candidate, dtype=np.float64)
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    require(candidate.shape == reference.shape,
            f"operand score shape moved: {candidate.shape} != {reference.shape}")
    unequal = candidate.view(np.uint64) != reference.view(np.uint64)
    magnitude = candidate != reference
    signed_zero = unequal & ~magnitude
    locations = np.argwhere(unequal)
    return {
        "bit_unequal": int(np.count_nonzero(unequal)),
        "magnitude_unequal": int(np.count_nonzero(magnitude)),
        "signed_zero_only": int(np.count_nonzero(signed_zero)),
        "first_bit_unequal_j_i_k": (
            None if locations.size == 0 else list(map(int, locations[0]))
        ),
    }


def assemble_record(root: Path, plant: str) -> tuple[dict[str, np.ndarray], dict]:
    assembled = {
        name: np.zeros((148, 180, 30), dtype=np.float64)
        for name in record_gate.FIELDS[:-1]
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    recurrence_bits = 0
    rows = []
    for rank in range(2):
        path = root / f"oracle_r107_een_step_rank{rank:04d}_kt00000001.bin"
        row = record_gate.read_record(path)
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        coverage[j0:j1, i0:i1] += 1
        groups = row["groups"]
        bottom = groups["mbku"][..., 0].astype(np.int64)
        require(bool(np.all(bottom <= 30)), "NEMO mbku enters its dummy jpk level")
        executed = np.arange(1, 32)[None, None, :] <= bottom[..., None]
        recurrence_bits += int(np.count_nonzero(
            groups["acc_after"][executed].view(np.uint64)
            != (groups["acc_before"] + groups["term_nw"])[executed].view(np.uint64)
        ))
        for name in assembled:
            values = groups[name]
            require(bool(np.all(values[..., 30] == 0.0)),
                    f"{name} writes NEMO's dummy jpk level")
            if plant == "oracle-bit" and rank == 0 and name == "zpvo_nw":
                values = np.array(values, copy=True)
                values.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
            assembled[name][j0:j1, i0:i1, :] = values[..., :30].transpose(1, 0, 2)
        rows.append({"rank": rank, "sha256": row["sha256"], "owned": row["owned"]})
    require(bool(np.all(coverage == 1)), "rank slabs do not cover the domain exactly once")
    require(recurrence_bits == 0, "recorded acc_after != acc_before + term_nw")
    return assembled, {"records": rows, "coverage": "exactly-once",
                       "recurrence_bit_unequal": recurrence_bits}


def measure(deck_root: Path, frame_root: Path, step_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-109 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-109 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-109 walk requires production JIT on CPU")

    oracle, census = assemble_record(step_root, plant)
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
            source_face_thickness=True, literal_bottom_loop=True),
    )(jnp.asarray(state.eta.data, dtype=jnp.float64)))
    model = {
        "zpvo_nw": np.asarray(parts["zpvo_u_nw"]),
        "e3u_live": np.asarray(parts["source_e3u"]),
        "e3v_live": np.asarray(parts["source_e3v"]),
        "neighbor_mask": np.asarray(raw.vmask, dtype=np.float64),
        "term_nw": np.asarray(parts["term_u_nw"]),
    }
    if plant == "model-bit":
        changed = np.array(model["zpvo_nw"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        model["zpvo_nw"] = changed
    scores = {name: _score(model[name], oracle[name]) for name in SOURCE_ORDER}
    first = next((name for name in SOURCE_ORDER if scores[name]["bit_unequal"]), None)
    require(first is not None, "all recorded operands unexpectedly bit-exact")
    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R109_EEN_PER_LEVEL_WALK",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "source_order": list(SOURCE_ORDER),
        "scores": scores,
        "first_non_bit_operand": first,
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
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
