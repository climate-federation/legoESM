#!/usr/bin/env python3
"""Gate ORCA2's compact U cyclic closure and T-pivot V fold bit for bit."""

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
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round156_association_growth_gate as r156,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round157_exchange_split_gate as r157,
)


STEPS = 65
PLANTS = (
    "none", "u-rank-source", "u-pivot-sign", "v-source",
    "v-permutation", "v-sign", "composition", "selector",
)
U_HALO_PAIRS = (
    ("rank0-west-outer", 0, 0, 1, 90),
    ("rank0-west-inner", 0, 1, 1, 91),
    ("rank0-east-inner", 0, 92, 1, 2),
    ("rank0-east-outer", 0, 93, 1, 3),
    ("rank1-west-outer", 1, 0, 0, 90),
    ("rank1-west-inner", 1, 1, 0, 91),
    ("rank1-east-inner", 1, 92, 0, 2),
    ("rank1-east-outer", 1, 93, 0, 3),
)


class GateError(RuntimeError):
    """The admitted record does not support the exact halo claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _pair(candidate, reference) -> dict[str, object]:
    try:
        return r156.exact_pair_row(candidate, reference)
    except r156.GateError as error:
        raise GateError(str(error)) from error


def validate_selector(value: str) -> None:
    require(value in ("", "u_cyclic", "v_fold"),
            f"unknown exact-halo selector {value!r}")


def _frame_name(step: int, face: str) -> str:
    require(1 <= step <= STEPS, f"substep outside 1..{STEPS}: {step}")
    require(face in ("u", "v"), f"unknown face {face!r}")
    return f"j{step:03d}_{'ua' if face == 'u' else 'va'}_new"


def _load_rank_frames(spg_root: Path) -> tuple[list[dict], list[dict]]:
    """Read only post-association U/V frames after the admitted reader passes."""

    metadata_rows: list[dict] = []
    frames: list[dict] = []
    for rank in (0, 1):
        path = spg_root / f"oracle_r95_spg_rank{rank:04d}_kt00000001.bin"
        metadata, values = r97._payload(path)
        require(metadata["rank"] == rank, f"{path.name}: rank moved")
        require(metadata["shape"][:2] == [94, 152],
                f"{path.name}: local shape moved")
        require(metadata["owned"] == [3, 3, 92, 150],
                f"{path.name}: owned slab moved")
        require(metadata["origin"] == [1 + 90 * rank, 1],
                f"{path.name}: origin moved")
        require(metadata["icycle"] == STEPS,
                f"{path.name}: substep count moved")
        kept = {
            _frame_name(step, face): np.array(
                values[_frame_name(step, face)], copy=True)
            for step in range(1, STEPS + 1)
            for face in ("u", "v")
        }
        metadata_rows.append({
            key: metadata[key] for key in (
                "rank", "sha256", "bytes", "shape", "owned", "origin",
                "icycle",
            )
        })
        frames.append(kept)
    return metadata_rows, frames


def _plant_one(value: np.ndarray) -> np.ndarray:
    planted = np.array(value, copy=True)
    nonzero = np.argwhere(planted != 0.0)
    location = tuple(map(int, nonzero[0] if nonzero.size else np.zeros(
        planted.ndim, dtype=int)))
    planted[location] = np.nextafter(
        planted[location], np.float64(np.inf))
    return planted


def _u_rank_halo_rows(frames: list[dict], *, plant: str) -> dict[str, object]:
    rows = []
    for step in range(1, STEPS + 1):
        name = _frame_name(step, "u")
        values = [frames[rank][name] for rank in (0, 1)]
        for pair_index, (label, target_rank, target_i,
                         source_rank, source_i) in enumerate(U_HALO_PAIRS):
            target = values[target_rank][target_i, 2:150]
            source = values[source_rank][source_i, 2:150]
            if plant == "u-rank-source" and step == 1 and pair_index == 0:
                source = _plant_one(source)
            row = _pair(target, source)
            rows.append({"substep": step, "pair": label, **row})
    first_nonbit = next((row for row in rows if not row["bit_exact"]), None)
    require(first_nonbit is None,
            f"two-rank U halo ordering differs: {first_nonbit}")
    return {
        "comparisons": len(rows),
        "bit_exact_comparisons": sum(row["bit_exact"] for row in rows),
        "maximum_absolute": max(row["maximum_absolute"] for row in rows),
        "first_non_bit": first_nonbit,
    }


def _oracle_compact_rows(frames: list[dict], step: int) -> dict[str, np.ndarray]:
    """Map rank-local post-call fields into the compact closure/fold rows."""

    u_name = _frame_name(step, "u")
    v_name = _frame_name(step, "v")
    rank0_u, rank1_u = (frames[rank][u_name] for rank in (0, 1))
    rank0_v, rank1_v = (frames[rank][v_name] for rank in (0, 1))
    u_native = np.concatenate(
        [rank0_u[2:92, 2:150].T, rank1_u[2:92, 2:150].T], axis=1)
    v_native = np.concatenate(
        [rank0_v[2:92, 2:150].T, rank1_v[2:92, 2:150].T], axis=1)
    return {
        "u_native": u_native,
        "v_native": v_native,
        # Inner west halo of rank 0 is the compact periodic U closure.
        "u_closure": np.array(rank0_u[1, 2:150], copy=True),
        # Local j=149/150 (zero based 148/149) are global rows 147/148.
        "v_source": np.concatenate(
            [rank0_v[2:92, 148], rank1_v[2:92, 148]]),
        "v_target": np.concatenate(
            [rank0_v[2:92, 149], rank1_v[2:92, 149]]),
    }


def _operand_walk(trace, frames: list[dict], grid, *, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.grids.operators_latlon_cgrid import fold_perm_u
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_external_mode_boundary_association,
    )

    fold = grid.fold
    require(fold is not None and fold.is_active
            and bool(fold.pivot_row_stored),
            "round-158 requires the stored T-pivot row")
    expected_v_perm = np.concatenate(
        [np.array([0], dtype=np.int32),
         np.arange(179, 0, -1, dtype=np.int32)])
    v_perm = np.asarray(fold.perm_v)
    if plant == "v-permutation":
        v_perm = np.roll(v_perm, 1)
    permutation_row = _pair(
        v_perm.astype(np.int64), expected_v_perm.astype(np.int64))
    require(permutation_row["bit_exact"],
            "card V fold permutation differs from compiled T-pivot order")
    require(np.array_equal(np.asarray(fold_perm_u(fold)),
                           np.arange(179, -1, -1, dtype=np.int32)),
            "card U fold permutation differs from compiled T-pivot order")

    u_nonpivot_rows = []
    u_pivot_cyclic_rows = []
    u_pivot_final_rows = []
    v_source_rows = []
    v_formula_rows = []
    v_final_rows = []
    u_change_rows = []
    v_change_rows = []
    keys = (
        "u_exit", "v_exit", "face_depth_u_exit", "face_depth_v_exit",
        "r1_face_depth_u_exit", "r1_face_depth_v_exit", "eta_exit",
    )
    for index in range(STEPS):
        trace_operands = tuple(jnp.asarray(trace[key][index]) for key in keys)
        full = _nemo_external_mode_boundary_association(
            *trace_operands, grid, component="")
        trace_u_cyclic = _nemo_external_mode_boundary_association(
            *trace_operands, grid, component="u_cyclic")
        u_both = _nemo_external_mode_boundary_association(
            *trace_u_cyclic, grid, component="u_fold")
        trace_v_fold = _nemo_external_mode_boundary_association(
            *trace_operands, grid, component="v_fold")
        oracle = _oracle_compact_rows(frames, index + 1)

        oracle_u = np.concatenate(
            [oracle["u_closure"][:, None], oracle["u_native"]], axis=1)
        oracle_v = np.concatenate(
            [np.zeros_like(oracle["v_native"][:1]), oracle["v_native"]],
            axis=0)
        zero_u = np.zeros_like(oracle_u)
        zero_v = np.zeros_like(oracle_v)
        zero_t = np.zeros_like(oracle["u_native"])
        oracle_operands = tuple(map(jnp.asarray, (
            oracle_u, oracle_v, zero_u, zero_v, zero_u, zero_v, zero_t)))

        # Known-answer restoration: perturb only the target, retain NEMO's
        # recorded source, and require the selected operation to restore it.
        planted_u = np.array(oracle_u, copy=True)
        planted_u[28, 0] = np.nextafter(
            planted_u[28, 0], np.float64(np.inf))
        cyclic_operands = (jnp.asarray(planted_u), *oracle_operands[1:])
        oracle_u_cyclic = _nemo_external_mode_boundary_association(
            *cyclic_operands, grid, component="u_cyclic")[0]
        planted_u_pivot = np.array(oracle_u, copy=True)
        planted_u_pivot[-1, 0] = -planted_u_pivot[-1, 0]
        fold_operands = (jnp.asarray(planted_u_pivot), *oracle_operands[1:])
        oracle_u_fold = _nemo_external_mode_boundary_association(
            *fold_operands, grid, component="u_fold")[0]
        planted_v = np.array(oracle_v, copy=True)
        planted_v[-1, 135] = np.nextafter(
            planted_v[-1, 135], np.float64(np.inf))
        v_operands = (oracle_operands[0], jnp.asarray(planted_v),
                      *oracle_operands[2:])
        oracle_v_fold = _nemo_external_mode_boundary_association(
            *v_operands, grid, component="v_fold")[1]

        u_cyclic_value = np.asarray(jax.device_get(oracle_u_cyclic))
        u_final = np.asarray(jax.device_get(oracle_u_fold))
        if plant == "u-pivot-sign" and index == 0:
            u_final = np.array(u_final, copy=True)
            u_final[-1, 0] = -u_final[-1, 0]
        u_nonpivot_rows.append(_pair(
            u_cyclic_value[:-1, 0], oracle["u_closure"][:-1]))
        trace_cyclic = np.asarray(jax.device_get(trace_u_cyclic[0]))
        u_pivot_cyclic_rows.append(_pair(
            trace_cyclic[-1:, 0], oracle["u_closure"][-1:]))
        u_pivot_final_rows.append(_pair(
            u_final[-1:, 0], oracle["u_closure"][-1:]))
        u_change_rows.append(_pair(
            trace_cyclic, np.asarray(trace["u_exit"][index])))

        source = np.asarray(oracle["v_source"])
        if plant == "v-source" and index == 0:
            source = _plant_one(source)
        v_source_rows.append(_pair(source, oracle["v_source"]))
        sign = np.float64(1.0 if plant == "v-sign" else fold.vector_sign_v)
        formula = sign * source[v_perm]
        v_formula_rows.append(_pair(formula, oracle["v_target"]))
        v_final = np.asarray(jax.device_get(oracle_v_fold))[-1]
        v_final_rows.append(_pair(v_final, oracle["v_target"]))
        v_change_rows.append(_pair(
            np.asarray(jax.device_get(trace_v_fold[1])),
            np.asarray(trace["v_exit"][index])))

        # The sequential split must remain the exact complete helper image.
        require(_pair(np.asarray(jax.device_get(u_both[0])),
                      np.asarray(jax.device_get(full[0])))["bit_exact"],
                f"U split composition differs at substep {index + 1}")

    for label, rows in (
        ("U cyclic non-pivot closure", u_nonpivot_rows),
        ("U pivot closure after fold", u_pivot_final_rows),
        ("V fold source", v_source_rows),
        ("V fold formula", v_formula_rows),
        ("V fold result", v_final_rows),
    ):
        first = next((index + 1 for index, row in enumerate(rows)
                      if not row["bit_exact"]), None)
        require(first is None, f"{label} differs at substep {first}")

    u_first = u_change_rows[0]
    v_first = v_change_rows[0]
    require(u_first["differing_cells"] == 24,
            "round-157 U cyclic non-vacuity census moved")
    require(v_first["differing_cells"] == 180,
            "round-157 V fold non-vacuity census moved")
    return {
        "u_cyclic_nonpivot": {
            "rows": len(u_nonpivot_rows),
            "bit_exact_rows": sum(row["bit_exact"] for row in u_nonpivot_rows),
        },
        "u_cyclic_pivot_before_fold": {
            "rows": len(u_pivot_cyclic_rows),
            "bit_exact_rows": sum(
                row["bit_exact"] for row in u_pivot_cyclic_rows),
            "first_non_bit": next((
                {"substep": index + 1, **row}
                for index, row in enumerate(u_pivot_cyclic_rows)
                if not row["bit_exact"]), None),
        },
        "u_pivot_after_fold": {
            "rows": len(u_pivot_final_rows),
            "bit_exact_rows": sum(
                row["bit_exact"] for row in u_pivot_final_rows),
        },
        "v_source": {
            "rows": len(v_source_rows),
            "bit_exact_rows": sum(row["bit_exact"] for row in v_source_rows),
        },
        "v_permutation": permutation_row,
        "v_sign": float(fold.vector_sign_v),
        "v_formula": {
            "rows": len(v_formula_rows),
            "bit_exact_rows": sum(row["bit_exact"] for row in v_formula_rows),
        },
        "v_result": {
            "rows": len(v_final_rows),
            "bit_exact_rows": sum(row["bit_exact"] for row in v_final_rows),
        },
        "substep1_nonvacuity": {
            "u_cyclic": u_first,
            "v_fold": v_first,
        },
    }


def measure(deck_root: Path, record_root: Path, spg_root: Path,
            expect_commit: str, *, plant: str) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    if plant == "selector":
        validate_selector("plausible")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-158 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-158 commit stamp mismatch")
    r157._setup_policy()
    metadata, frames = _load_rank_frames(spg_root)
    rank_order = _u_rank_halo_rows(frames, plant=plant)

    (card, reference_depth, _entry, initial_state,
     freshwater, surface) = r156._setup(deck_root, record_root)
    ordinary = r157._run(
        card, initial_state, reference_depth, freshwater, surface)
    observed = r157._run(
        card, initial_state, reference_depth, freshwater, surface,
        expose=True)
    passivity = r157._compare_states(observed, ordinary)
    require(all(row["bit_exact"] for row in passivity.values()),
            "association observer moved production")
    production_stage = r157._run(
        card, initial_state, reference_depth, freshwater, surface,
        stage=1, expose=True)
    require(production_stage.substeps["eta_entry"].shape[0] == STEPS,
            "production trace does not contain 65 substeps")
    composition = r157._composition_rows(
        production_stage.substeps, card.recipe.grid,
        plant="composition" if plant == "composition" else "none")
    operands = _operand_walk(
        production_stage.substeps, frames, card.recipe.grid, plant=plant)
    pivot_before = operands["u_cyclic_pivot_before_fold"]
    return {
        "status": "PASS_R158_EXACT_HALO_OPERANDS",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record": metadata,
        "u_two_rank_order": rank_order,
        "operand_walk": operands,
        "passivity": passivity,
        "composition": composition,
        "predictions": {
            "R158-P1": "CONFIRMED",
            "R158-P2": (
                "CONFIRMED" if pivot_before["bit_exact_rows"] == STEPS
                else "REFUTED_PIVOT_OVERWRITTEN_LATER"),
            "R158-P3": "CONFIRMED",
            "R158-P4": "CONFIRMED",
            "R158-P5": "UNMEASURED_PENDING_LADDER",
        },
        "choices": {"asked": [], "unasked": []},
        "worktree": stamp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = measure(
            args.deck_root, args.record_root, args.spg_root,
            args.expect_commit, plant=args.plant)
    except (GateError, r97.GateError, r156.GateError, r157.GateError,
            OSError, ValueError) as error:
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
