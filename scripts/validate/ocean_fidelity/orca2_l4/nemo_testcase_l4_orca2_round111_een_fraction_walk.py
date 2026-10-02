#!/usr/bin/env python3
"""Compare admitted rung-0 EEN fractions and operands in source order."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round110_een_fraction_acquisition import (
    check_record as record_gate,
)


PLANTS = ("none", "oracle-bit", "model-bit")
LABELS = ("west", "center", "south")
SOURCE_ORDER = tuple(
    name
    for label in LABELS
    for name in (
        f"{label}_ff", f"{label}_e3f0", f"{label}_r3f", f"{label}_mask",
        f"{label}_denom", f"frac_{label}",
    )
) + ("sum_west_center", "sum_all")


class GateError(RuntimeError):
    """The admitted fraction walk cannot support its registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def assemble_record(root: Path, plant: str) -> tuple[dict[str, np.ndarray], dict]:
    assembled = {
        name: np.zeros((148, 180, 30), dtype=np.float64)
        for name in record_gate.FIELDS[:-1]
    }
    assembled["mbku"] = np.zeros((148, 180), dtype=np.float64)
    coverage = np.zeros((148, 180), dtype=np.int8)
    rows = []
    for rank in range(2):
        path = root / f"oracle_r110_een_fraction_rank{rank:04d}_kt00000001.bin"
        row = record_gate.read_record(path)
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148), "rank slab latitude placement moved")
        coverage[j0:j1, i0:i1] += 1
        groups = row["groups"]
        bottom = groups["mbku"][..., 0].astype(np.int64)
        require(bool(np.all((bottom >= 1) & (bottom <= 30))),
                "NEMO mbku enters its dummy level or is invalid")
        assembled["mbku"][j0:j1, i0:i1] = bottom.T
        for name in record_gate.FIELDS[:-1]:
            values = np.array(groups[name][..., :30], copy=True)
            if plant == "oracle-bit" and rank == 0 and name == "west_ff":
                values.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
            assembled[name][j0:j1, i0:i1, :] = values.transpose(1, 0, 2)
        rows.append({"rank": rank, "sha256": row["sha256"], "owned": row["owned"]})
    require(bool(np.all(coverage == 1)), "rank slabs do not cover the domain exactly once")
    return assembled, {"coverage": "exactly-once", "records": rows}


def measure(deck_root: Path, frame_root: Path, fraction_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-111 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-111 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-111 walk requires production JIT on CPU")

    oracle, census = assemble_record(fraction_root, plant)
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
            return_fraction_operands=True),
    )(jnp.asarray(state.eta.data, dtype=jnp.float64)))

    mbku = np.asarray(parts["mbku"], dtype=np.float64)
    executed = np.arange(1, 31)[None, None, :] <= mbku[..., None]

    def shift(value: np.ndarray, di: int = 0, dj: int = 0) -> np.ndarray:
        out = np.roll(value, di, axis=1) if di else value
        return np.roll(out, dj, axis=0) if dj else out

    base = {
        "ff": np.broadcast_to(np.asarray(parts["een_ff"])[..., None], executed.shape),
        "e3f0": np.asarray(parts["een_e3f0"]),
        "r3f": np.broadcast_to(np.asarray(parts["een_r3f"])[..., None], executed.shape),
        "mask": np.asarray(parts["een_fmask"]),
        "denom": np.asarray(parts["een_denom"]),
        "frac": np.asarray(parts["een_q"]),
    }
    associations = {"west": (1, 0), "center": (0, 0), "south": (0, 1)}
    model: dict[str, np.ndarray] = {}
    for label, (di, dj) in associations.items():
        for name, values in base.items():
            key = f"frac_{label}" if name == "frac" else f"{label}_{name}"
            model[key] = shift(values, di, dj)
    model["sum_west_center"] = nemo_source_round(
        model["frac_west"] + model["frac_center"])
    model["sum_all"] = nemo_source_round(
        model["sum_west_center"] + model["frac_south"])
    for name in SOURCE_ORDER:
        model[name] = np.where(executed, model[name], np.float64(0.0))
    if plant == "model-bit":
        changed = np.array(model["west_ff"], copy=True)
        changed.view(np.uint64)[0, 0, 0] ^= np.uint64(1)
        model["west_ff"] = changed

    scores = {name: r109._score(model[name], oracle[name]) for name in SOURCE_ORDER}
    first = next((name for name in SOURCE_ORDER if scores[name]["bit_unequal"]), None)
    require(first is not None, "all recorded EEN fractions unexpectedly bit-exact")
    if plant != "none":
        raise GateError(f"{plant} plant fired")
    return {
        "status": "MEASURED_R111_EEN_FRACTION_WALK",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "dtypes": {
            "state_eta": str(np.asarray(state.eta.data).dtype),
            "geometry_e3t0": str(np.asarray(e3t0).dtype),
            "een_ff": str(np.asarray(parts["een_ff"]).dtype),
            "een_e3f0": str(np.asarray(parts["een_e3f0"]).dtype),
        },
        "record_census": census,
        "source_order": list(SOURCE_ORDER),
        "scores": scores,
        "first_non_bit_item": first,
        "fraction_scores": {label: scores[f"frac_{label}"] for label in LABELS},
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
