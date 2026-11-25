#!/usr/bin/env python3
"""Walk the northern V EEN fractions and operands in compiled source order."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition import (
    check_record as record_gate,
)


PATHS = ("ne", "nw")
COMPONENTS = ("1", "2", "3")
OPERANDS = ("ff", "e3f0", "r3f", "mask", "denom", "frac")
SOURCE_ORDER = tuple(
    name
    for component in COMPONENTS
    for name in (
        f"{component}_ff", f"{component}_e3f0", f"{component}_r3f",
        f"{component}_mask", f"{component}_denom", f"{component}_frac",
        *(('partial',) if component == "2" else ()),
    )
) + ("sum",)
SHIFTS = {
    "ne": ((0, -1), (0, 0), (1, 0)),
    "nw": ((0, 0), (1, 0), (1, -1)),
}
EXPECTED_FIRST_COMPONENT = {"ne": "1", "nw": "3"}
EXPECTED_FIRST = {"ne": "1_ff", "nw": "3_ff"}
EXPECTED_FINAL = {"ne": (1431, 1431), "nw": (1431, 1431)}
EXPECTED_SCORES = {
    path: {name: (0, 0) for name in SOURCE_ORDER} for path in PATHS
}
EXPECTED_SCORES["ne"].update({
    "1_ff": (1431, 1431),
    "1_e3f0": (514, 514),
    "1_mask": (1160, 1160),
    "1_denom": (514, 514),
    "1_frac": (1431, 1431),
    "partial": (1431, 1431),
    "sum": (1431, 1431),
})
EXPECTED_SCORES["nw"].update({
    "3_ff": (1431, 1431),
    "3_e3f0": (521, 521),
    "3_mask": (1154, 1154),
    "3_denom": (521, 521),
    "3_frac": (1431, 1431),
    "sum": (1431, 1431),
})
PLANTS = ("none", "oracle-bit", "candidate-bit", "rank-seam", "scope-route")


class GateError(RuntimeError):
    """The admitted fraction record cannot support its source-ordered claim."""


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
        path = root / f"oracle_r120_een_v_fraction_rank{rank:04d}_kt00000001.bin"
        row = record_gate.read_record(path)
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        coverage[j0:j1, i0:i1] += 1
        assembled["mbkv"][j0:j1, i0:i1] = row["groups"]["mbkv"][..., 0].T
        for name in record_gate.FIELDS[:-1]:
            values = row["groups"][name]
            require(bool(np.all(values[..., 30] == 0.0)),
                    f"{path.name}: {name} writes NEMO's dummy level")
            if plant == "oracle-bit" and rank == 0 and name == "ne_1_ff":
                values = np.array(values, copy=True)
                values.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
            assembled[name][j0:j1, i0:i1] = values[..., :30].transpose(1, 0, 2)
        rows.append({"rank": rank, "sha256": row["sha256"], "owned": row["owned"]})
    require(bool(np.all(coverage == 1)), "rank slabs do not cover the domain once")
    return assembled, {"coverage": "exactly-once", "records": rows}


def _shift(value: np.ndarray, di: int, dj: int) -> np.ndarray:
    shifted = np.roll(value, di, axis=1) if di else value
    return np.roll(shifted, dj, axis=0) if dj else shifted


def _as_levels(value: np.ndarray) -> np.ndarray:
    return np.broadcast_to(value[..., None], (*value.shape, 30)) if value.ndim == 2 else value


def _value_bits(value: np.ndarray, location: tuple[int, int, int]) -> dict:
    scalar = np.asarray(value[location], dtype=np.float64)
    return {"value": float(scalar), "bits": f"0x{scalar.view(np.uint64):016x}"}


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            source_root: Path, expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-121 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-121 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-121 walk requires production JIT on CPU")

    admission = record_gate.run(record_root, source_root, "none")
    require(admission["status"] == "PASS_R120_EEN_V_FRACTION_ADMISSION",
            "round-120 record is not admitted")
    oracle, census = assemble_record(record_root, plant)

    card_scope = r112.resolved_card_scope(deck_root)
    if plant == "scope-route":
        card_scope["GYRE-zco"]["executes_southern_een_ff_association"] = True
    observed_scope = {
        name: bool(row["executes_southern_een_ff_association"])
        for name, row in card_scope.items()
    }
    require(observed_scope == r112.EXPECTED_CARD_SCOPE, "resolved card scope moved")

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
            return_fraction_operands=True, south_ff_copy=True,
            south_e3f0_fill=jnp.asarray(raw.e3f_0, dtype=jnp.float64),
            south_mask_zero=True,
        ),
    )(jnp.asarray(state.eta.data, dtype=jnp.float64)))

    bottom = np.asarray(parts["mbkv"], dtype=np.float64)
    bottom_score = r109._score(bottom, oracle["mbkv"])
    require(bottom_score["bit_unequal"] == 0, "mbkv moved before the fractions")
    executed = np.arange(1, 31)[None, None, :] <= bottom[..., None]
    bases = {
        "ff": np.asarray(parts["een_ff"], dtype=np.float64),
        "e3f0": np.asarray(parts["een_e3f0"], dtype=np.float64),
        "r3f": np.asarray(parts["een_r3f"], dtype=np.float64),
        "mask": np.asarray(parts["een_fmask"], dtype=np.float64),
        "denom": np.asarray(parts["een_denom"], dtype=np.float64),
        "frac": np.asarray(parts["een_q"], dtype=np.float64),
    }

    paths = {}
    for path in PATHS:
        candidate: dict[str, np.ndarray] = {}
        fractions = []
        for component, (di, dj) in zip(COMPONENTS, SHIFTS[path], strict=True):
            for operand in OPERANDS:
                value = _as_levels(_shift(bases[operand], di, dj))
                candidate[f"{component}_{operand}"] = np.where(
                    executed, value, np.float64(0.0))
            fractions.append(candidate[f"{component}_frac"])
        candidate["partial"] = np.where(
            executed, fractions[0] + fractions[1], np.float64(0.0))
        candidate["sum"] = np.where(
            executed, candidate["partial"] + fractions[2], np.float64(0.0))
        if plant == "candidate-bit" and path == "nw":
            candidate["sum"] = np.array(candidate["sum"], copy=True)
            candidate["sum"].view(np.uint64)[0, 0, 0] ^= np.uint64(1)

        reference = {
            name: oracle[f"{path}_{name}"] for name in SOURCE_ORDER
        }
        scores = {name: r109._score(candidate[name], reference[name])
                  for name in SOURCE_ORDER}
        first = next((name for name in SOURCE_ORDER
                      if scores[name]["bit_unequal"]), None)
        require(first is not None, f"{path}: all recorded fractions unexpectedly exact")
        require(first == EXPECTED_FIRST[path],
                f"{path}: first unequal item moved: {first}")
        require(scores["sum"]["bit_unequal"] == EXPECTED_FINAL[path][0]
                and scores["sum"]["magnitude_unequal"] == EXPECTED_FINAL[path][1],
                f"{path}: completed zpvo census moved")
        require(scores[first]["bit_unequal_j_values"] == [147],
                f"{path}: first boundary escaped the northern fold")
        for name in SOURCE_ORDER:
            observed = (scores[name]["bit_unequal"], scores[name]["magnitude_unequal"])
            require(observed == EXPECTED_SCORES[path][name],
                    f"{path}: {name} census moved: {observed}")
        locations = np.argwhere(
            np.ascontiguousarray(candidate[first]).view(np.uint64)
            != np.ascontiguousarray(reference[first]).view(np.uint64))
        if plant == "rank-seam" and path == "ne":
            locations = locations[locations[:, 1] < 90]
        i_values = np.unique(locations[:, 1])
        support_by_rank = {
            "rank0_i_0_89": int(np.count_nonzero(locations[:, 1] < 90)),
            "rank1_i_90_179": int(np.count_nonzero(locations[:, 1] >= 90)),
        }
        require(all(value > 0 for value in support_by_rank.values()),
                f"{path}: first boundary is confined to one rank")
        require(i_values.size > 2 and bool(np.all(np.diff(i_values) == 1)),
                f"{path}: first boundary is rank-edge-only or discontinuous")
        samples = []
        for raw_location in locations[:5]:
            location = tuple(map(int, raw_location))
            samples.append({
                "j_i_k": list(location),
                "oracle": _value_bits(reference[first], location),
                "candidate": _value_bits(candidate[first], location),
            })
        paths[path] = {
            "first_non_bit_item": first,
            "scores": scores,
            "first_boundary_i_values": list(map(int, i_values)),
            "first_boundary_support_by_rank": support_by_rank,
            "first_boundary_samples": samples,
        }

    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R121_EEN_V_FRACTIONS",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "record_census": census,
        "card_scope": card_scope,
        "source_order": list(SOURCE_ORDER),
        "paths": paths,
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
