#!/usr/bin/env python3
"""Replay round 333's live stage-3 tracer boundaries against NEMO.

The production L8 helper already returns every written ``Krhs`` boundary, but
its caller deliberately discards that diagnostic tuple.  This gate observes
the tuple with an ordered JAX debug callback, then proves that the observed
step is bit-identical to a separately compiled ordinary production step.  It
also observes the ordinary implicit solve's returned T/S before the stage-end
north-fold exchange.  No formula is reimplemented here.
"""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round333_tracer_gate as record_gate,
)


class GateError(RuntimeError):
    """The replay is incomplete or does not preserve production execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


BOUNDARIES = (
    "zero",
    "after_advection",
    "after_sbc",
    "after_qsr",
    "after_ldf",
    "after_damping",
    "finish",
    "post_boundary_finish",
)


def _bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    return left.shape == right.shape and bool(
        np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _score(candidate: np.ndarray, oracle: np.ndarray, wet: np.ndarray) -> dict[str, object]:
    require(candidate.shape == oracle.shape == wet.shape, "score shapes differ")
    candidate = np.ascontiguousarray(candidate)
    oracle = np.ascontiguousarray(oracle)
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    value_unequal = candidate != oracle
    delta = candidate - oracle
    wet_delta = delta[wet]
    return {
        "bit_unequal_wet": int(np.count_nonzero(unequal & wet)),
        "value_unequal_wet": int(np.count_nonzero(value_unequal & wet)),
        "max_abs_wet": float(np.max(np.abs(wet_delta))),
        "rms_wet": float(np.sqrt(np.mean(wet_delta * wet_delta))),
        "wet_count": int(np.count_nonzero(wet)),
    }


def _target_score(
    candidate: np.ndarray, oracle: np.ndarray, targets: np.ndarray
) -> dict[str, object]:
    require(candidate.shape == oracle.shape == targets.shape, "target shapes differ")
    require(bool(np.any(targets)), "target set is empty")
    candidate = np.ascontiguousarray(candidate)
    oracle = np.ascontiguousarray(oracle)
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    value_unequal = candidate != oracle
    delta = candidate - oracle
    return {
        "targets": int(np.count_nonzero(targets)),
        "bit_unequal": int(np.count_nonzero(unequal & targets)),
        "value_unequal": int(np.count_nonzero(value_unequal & targets)),
        "max_abs": float(np.max(np.abs(delta[targets]))),
        "rms": float(np.sqrt(np.mean(delta[targets] * delta[targets]))),
    }


def _tree_bit_identical(left, right) -> tuple[bool, int]:
    import jax

    leaves_left, tree_left = jax.tree_util.tree_flatten(left)
    leaves_right, tree_right = jax.tree_util.tree_flatten(right)
    require(tree_left == tree_right, "ordinary and observed pytrees differ")
    changed = 0
    for lhs, rhs in zip(leaves_left, leaves_right, strict=True):
        if not _bits_equal(np.asarray(lhs), np.asarray(rhs)):
            changed += 1
    return changed == 0, changed


def _global_record(fields: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([record_gate._owned_xyz(value) for value in fields], axis=1)


def _record_boundaries(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    records = [record_gate.read_stage3(root / record_gate.record_name(rank)) for rank in (0, 1)]
    result: dict[str, dict[str, np.ndarray]] = {}
    for boundary in BOUNDARIES[:-1]:
        if boundary == "finish":
            stem = "Kaa"
        else:
            stem = boundary
        result[boundary] = {
            tracer: _global_record([record["fields"][f"{stem}_{tracer}"] for record in records])
            for tracer in ("T", "S")
        }
    wet = _global_record(
        [record_gate._full_wet_mask(root, rank).astype(np.float64) for rank in (0, 1)]
    ).astype(bool)
    return result, wet


def _surface(card, deck_root: Path, root: Path, kt: int):
    fields = rung10.assemble_surface_fields(root, kt)
    freshwater, surface = rung10._surface_forcings(card, deck_root, fields, kt)
    return freshwater, surface, rung10.damping_step_kwargs(card, deck_root, kt)


def measure(
    label: str, deck_root: Path, record_root: Path, expect_commit: str
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(
        stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
        "round-333 replay requires its clean committed tree",
    )
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
        and bool(jax.config.jax_enable_x64),
        "fp64/libm is not active",
    )
    require(
        jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
        "replay requires production JIT on CPU",
    )
    require(label in ("given_nemo_entry", "independent"), "unknown label")

    producer = (record_root / "producer_commit.txt").read_text(encoding="ascii").strip()
    admission = record_gate.run_gate(record_root, expect_commit=producer)
    require(
        admission["status"] == "PASS_R333_TRACER_ACQUISITION",
        "round-333 tracer record is not admitted",
    )

    _, shipped = rung10.card_fields(deck_root)
    card = r329.selected_card(shipped, arm="recordice", rung="rung10")
    state0 = card.recipe.initial_state
    if label == "given_nemo_entry":
        entry = rung10.assemble_state_fields(record_root, 1, stage=None)
        state0 = state0._replace(
            eta=state0.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64))
        )

    model1 = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
    )
    freshwater1, surface1, damping1 = _surface(card, deck_root, record_root, 1)
    state1 = model1.step(
        state0,
        dt=card.dt_s,
        freshwater=freshwater1,
        surface_forcing=surface1,
        **damping1,
    )
    state1 = jax.tree.map(lambda value: value.block_until_ready(), state1)

    freshwater2, surface2, damping2 = _surface(card, deck_root, record_root, 2)
    ordinary = model1.step(
        state1,
        dt=card.dt_s,
        freshwater=freshwater2,
        surface_forcing=surface2,
        **damping2,
    )
    ordinary = jax.tree.map(lambda value: value.block_until_ready(), ordinary)

    captured_boundaries: list[tuple[np.ndarray, ...]] = []
    captured_ldf_terms: list[tuple[np.ndarray, np.ndarray]] = []
    captured_finish: list[tuple[np.ndarray, np.ndarray]] = []
    original_walk = model_module._nemo_stage3_l8_krhs_walk
    original_solve = LatLonCGridOceanModel._apply_implicit_vertical_mixing

    def store_boundaries(*values) -> None:
        captured_boundaries.append(tuple(np.array(value, copy=True) for value in values))

    def store_finish(temperature, salinity) -> None:
        captured_finish.append((np.array(temperature, copy=True), np.array(salinity, copy=True)))

    def store_ldf_terms(temperature, salinity) -> None:
        captured_ldf_terms.append(
            (np.array(temperature, copy=True), np.array(salinity, copy=True)))

    def observed_walk(adv_t, adv_s, *term_groups):
        result = original_walk(adv_t, adv_s, *term_groups)
        jax.debug.callback(
            store_boundaries,
            *(value for pair in result[2] for value in pair),
            ordered=True,
        )
        jax.debug.callback(
            store_ldf_terms,
            term_groups[2][0][0], term_groups[2][1][0],
            ordered=True,
        )
        return result

    def observed_solve(self, *args, **kwargs):
        result = original_solve(self, *args, **kwargs)
        solved = result[0] if isinstance(result, tuple) else result
        jax.debug.callback(store_finish, solved.T.data, solved.S.data, ordered=True)
        return result

    model_module._nemo_stage3_l8_krhs_walk = observed_walk
    LatLonCGridOceanModel._apply_implicit_vertical_mixing = observed_solve
    try:
        observed_model = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing,
        )
        observed = observed_model.step(
            state1,
            dt=card.dt_s,
            freshwater=freshwater2,
            surface_forcing=surface2,
            **damping2,
        )
        observed = jax.tree.map(lambda value: value.block_until_ready(), observed)
    finally:
        model_module._nemo_stage3_l8_krhs_walk = original_walk
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = original_solve

    passive, changed_leaves = _tree_bit_identical(ordinary, observed)
    require(passive, f"observer moved {changed_leaves} prognostic leaves")
    require(len(captured_boundaries) == 1, "expected one boundary callback")
    require(len(captured_ldf_terms) == 1, "expected one LDF-term callback")
    require(len(captured_finish) == 1, "expected one finish callback")

    candidate: dict[str, dict[str, np.ndarray]] = {
        "zero": {
            "T": np.zeros_like(captured_boundaries[0][0]),
            "S": np.zeros_like(captured_boundaries[0][1]),
        }
    }
    callback_names = (
        "after_advection",
        "after_sbc",
        "after_qsr",
        "after_ldf",
        "after_bbc",
        "after_bbl",
        "after_damping",
    )
    for index, name in enumerate(callback_names):
        candidate[name] = {
            "T": captured_boundaries[0][2 * index],
            "S": captured_boundaries[0][2 * index + 1],
        }
    candidate["finish"] = {
        "T": captured_finish[0][0],
        "S": captured_finish[0][1],
    }
    candidate["post_boundary_finish"] = {
        "T": np.asarray(observed.T.data),
        "S": np.asarray(observed.S.data),
    }

    oracle, wet = _record_boundaries(record_root)
    oracle_stage3 = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3
    )
    oracle["post_boundary_finish"] = {
        "T": oracle_stage3["T"],
        "S": oracle_stage3["S"],
    }

    ldf_final_update_arms = {}
    for tracer_index, tracer in enumerate(("T", "S")):
        before = candidate["after_qsr"][tracer][..., :30]
        term = captured_ldf_terms[0][tracer_index][..., :30]
        direct = np.asarray(before + term, dtype=np.float64)
        increment = np.asarray(direct - before, dtype=np.float64)
        add_sub_add = np.asarray(before + increment, dtype=np.float64)
        ldf_final_update_arms[tracer] = {
            "direct_add": _score(direct, oracle["after_ldf"][tracer], wet),
            "nemo_add_sub_add": _score(
                add_sub_add, oracle["after_ldf"][tracer], wet),
            "term_vs_recorded_increment": _score(
                term,
                np.asarray(
                    oracle["after_ldf"][tracer]
                    - oracle["after_qsr"][tracer],
                    dtype=np.float64,
                ),
                wet,
            ),
        }

    targets = {}
    for tracer in ("T", "S"):
        rank0_candidate = candidate["post_boundary_finish"][tracer][:, :90, :30]
        target = rank0_candidate != oracle_stage3[tracer]
        require(
            int(np.count_nonzero(target)) == (18 if tracer == "T" else 13),
            f"{tracer}: registered target census moved",
        )
        full = np.zeros_like(wet)
        full[:, :90] = target
        targets[tracer] = full

    ldf_target_walk = {}
    for tracer_index, tracer in enumerate(("T", "S")):
        target = targets[tracer]
        before = candidate["after_qsr"][tracer][..., :30]
        term = captured_ldf_terms[0][tracer_index][..., :30]
        reconstructed = np.asarray(before + term, dtype=np.float64)
        live = candidate["after_ldf"][tracer][..., :30]
        post_terms = candidate["after_damping"][tracer][..., :30]
        solved = candidate["finish"][tracer][..., :30]
        ldf_target_walk[tracer] = {
            "krhs_before_ldf": _target_score(
                before, oracle["after_qsr"][tracer], target),
            "ldf_term_add": _target_score(
                reconstructed, oracle["after_ldf"][tracer], target),
            "live_accumulator_after_ldf": _target_score(
                live, oracle["after_ldf"][tracer], target),
            "live_vs_reconstructed": _target_score(live, reconstructed, target),
            "post_bbc_bbl_dmp": _target_score(
                post_terms, oracle["after_damping"][tracer], target),
            "implicit_vertical_solve_return": _target_score(
                solved, oracle["finish"][tracer], target),
        }

    rows: dict[str, dict[str, object]] = {}
    first = None
    for boundary in BOUNDARIES:
        rows[boundary] = {}
        for tracer in ("T", "S"):
            if boundary == "post_boundary_finish":
                cand = candidate[boundary][tracer][:, :90, :30]
                ref = oracle[boundary][tracer]
                wet_here = wet[:, :90]
                target_here = targets[tracer][:, :90]
            else:
                cand = candidate[boundary][tracer][..., :30]
                ref = oracle[boundary][tracer]
                wet_here = wet
                target_here = targets[tracer]
            rows[boundary][tracer] = {
                "everywhere_wet": _score(cand, ref, wet_here),
                "registered_targets": _target_score(cand, ref, target_here),
            }
            if first is None and rows[boundary][tracer]["everywhere_wet"]["value_unequal_wet"]:
                first = {"boundary": boundary, "tracer": tracer}

    require(first is not None, "source-order replay found no wet departure")
    return {
        "format": "nemo-testcase-l4-orca2-round333-tracer-replay-v1",
        "status": "PASS_R333_TRACER_REPLAY",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_admission": admission["status"],
        "observer_passivity": {
            "bit_identical": passive,
            "changed_leaves": changed_leaves,
            "callback_copies": 1,
        },
        "registered_target_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in targets.items()
        },
        "ldf_target_walk": ldf_target_walk,
        "first_wet_value_boundary": first,
        "ldf_final_update_arms": ldf_final_update_arms,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("given_nemo_entry", "independent"), required=True)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.label, args.deck_root, args.record_root, args.expect_commit)
    except (GateError, record_gate.GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
