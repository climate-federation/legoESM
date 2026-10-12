#!/usr/bin/env python3
"""Split OMT-4's fold-local ``dyn_drg_init`` V correction offline."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)
from nemo_testcase_l2_gyre_round14_advmean import read_ordered  # noqa: E402
from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing  # noqa: E402

LABELS = ("independent", "given_nemo_entry")
FLOOR = np.float64(2.0e-10)
OPERAND_ORDER = (
    "stored_reciprocal", "live_reciprocal", "drag_coefficient",
    "bottom_velocity", "barotropic_velocity", "bottom_residual",
    "drag_increment", "post_drag",
)
PLANTS = (
    "none", "label-coverage", "production-route", "record-admission",
    "self-replay", "owner-support", "source-order", "coefficient-owner",
)


class GateError(RuntimeError):
    """The record, replay, source order, or frozen owner moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :90, ...]


def _bottom_value(values: np.ndarray, indicator: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    indicator = np.asarray(indicator, dtype=np.float64)
    require(values.shape == indicator.shape and values.ndim == 3,
            "bottom-value inputs moved")
    require(np.all(np.sum(indicator, axis=-1) <= 1.0),
            "bottom indicator selects multiple levels")
    return np.sum(values * indicator, axis=-1)


def _drag_chain(depth_mean, inverse_depth, coefficient,
                bottom_velocity, barotropic_velocity) -> dict[str, np.ndarray]:
    """Replay dynspg_ts.f90:1443-1463 in written scalar association."""

    depth_mean = np.asarray(depth_mean, dtype=np.float64)
    inverse_depth = np.asarray(inverse_depth, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    bottom_velocity = np.asarray(bottom_velocity, dtype=np.float64)
    barotropic_velocity = np.asarray(barotropic_velocity, dtype=np.float64)
    residual = bottom_velocity - barotropic_velocity
    increment = (inverse_depth * coefficient) * residual
    post_drag = depth_mean + increment
    return {
        "bottom_residual": residual,
        "drag_increment": increment,
        "post_drag": post_drag,
    }


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-250 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-250 measurement requires production JIT on CPU")
    return stamp


def measure(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_vface
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_bottom_drag_rate_faces,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    admission = r236._admit(twin_a, twin_b)
    slow = read_slow_forcing(
        twin_a / "oracle_slow_forcing_kt00000001.bin",
        dims=(94, 152, 31),
    )
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2,
    )

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True),
    )
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True),
    )
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        plain = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        traced = jax.device_get(traced_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live operand trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, plain)
    require(all(passivity.values()), "live operand trace moved completed state")
    producer = traced.slow_forcing_producer

    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(jnp.float64)
    h_v = min_cell_to_vface(h_k, card.recipe.grid)
    depth_v = jnp.maximum(jnp.sum(h_v, axis=-1), jnp.float64(1.0e-10))
    _, rate_v, _, indicator_v = nemo_bottom_drag_rate_faces(
        state.u.data, state.v.data, h_k, card.recipe.z_coord,
        card.recipe.model_config, card.recipe.grid,
    )
    bottom_v = jnp.sum(state.v.data * indicator_v, axis=-1)
    barotropic_v = jnp.sum(state.v.data * h_v, axis=-1) / depth_v
    candidate_arrays = jax.device_get((
        depth_v, rate_v, indicator_v, bottom_v, barotropic_v,
    ))
    depth_v, rate_v, indicator_v, bottom_v, barotropic_v = map(
        np.asarray, candidate_arrays)

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 card lacks resolved raw NEMO operands")
    raw_hv = np.asarray(raw.hv_0, dtype=np.float64)[:, :90]
    raw_wet = (raw_hv > 0.0).astype(np.float64)
    stored_reciprocal = raw_wet / (raw_hv + 1.0 - raw_wet)

    oracle_v = np.asarray(entry["v"], dtype=np.float64)[:, :90, :]
    oracle_mask = np.asarray(raw.vmask, dtype=np.float64)[:, :90, :30]
    oracle_indicator = np.zeros_like(oracle_mask)
    counts = np.sum(oracle_mask != 0.0, axis=-1, dtype=np.int64)
    levels = np.maximum(counts - 1, 0)
    np.put_along_axis(
        oracle_indicator, levels[..., None],
        (counts > 0)[..., None].astype(np.float64), axis=-1,
    )

    candidate = {
        "stored_reciprocal": stored_reciprocal,
        "live_reciprocal": _native_v(1.0 / depth_v),
        # legoESM stores +Cd|U|; NEMO stores the signed negative coefficient.
        "drag_coefficient": -_native_v(rate_v),
        "bottom_velocity": _native_v(bottom_v),
        "barotropic_velocity": _native_v(barotropic_v),
        "post_drag": _native_v(producer["post_drag_v"]),
    }
    # Recover it without a new observer: the already-certified production
    # depth mean is the passive RHS/thickness replay used by round 236.  On
    # the 35-face claim support the atomic fold unit selects these raw D114
    # operands; off-support values cannot enter any verdict below.
    candidate["depth_mean"] = r236._candidate_depth(
        np.asarray(raw.e3v_0, np.float64)[:, :90, :30],
        _native_v(producer["rhs_v"]), oracle_mask, stored_reciprocal,
    )
    oracle = {
        "stored_reciprocal": np.asarray(slow["r1_hv0"], np.float64),
        "live_reciprocal": np.asarray(slow["r1_hv"], np.float64),
        "drag_coefficient": np.asarray(slow["cd_v"], np.float64),
        "bottom_velocity": _bottom_value(oracle_v, oracle_indicator),
        "barotropic_velocity": np.asarray(ordered["v_entry"][0], np.float64),
        "depth_mean": np.asarray(slow["depth_v"], np.float64),
        "post_drag": np.asarray(slow["post_drag_v"], np.float64),
    }
    candidate_chain = _drag_chain(
        candidate["depth_mean"], candidate["live_reciprocal"],
        candidate["drag_coefficient"], candidate["bottom_velocity"],
        candidate["barotropic_velocity"],
    )
    oracle_chain = _drag_chain(
        oracle["depth_mean"], oracle["live_reciprocal"],
        oracle["drag_coefficient"], oracle["bottom_velocity"],
        oracle["barotropic_velocity"],
    )
    candidate.update(candidate_chain)
    oracle.update(oracle_chain)

    owner = candidate["post_drag"] != oracle["post_drag"]
    require(int(np.count_nonzero(owner)) == 35,
            f"post-drag owner support moved: {np.count_nonzero(owner)} != 35")
    require(int(np.count_nonzero(owner[:-3])) == 0,
            "post-drag owner escaped the fold band")
    rows = {name: r236._row(candidate[name], oracle[name], owner)
            for name in OPERAND_ORDER}
    self_replay = {
        "candidate": r236._row(
            candidate_chain["post_drag"], candidate["post_drag"], owner),
        "oracle": r236._row(
            oracle_chain["post_drag"], oracle["post_drag"], owner),
    }

    accumulated = {
        name: candidate[name] for name in (
            "live_reciprocal", "drag_coefficient", "bottom_velocity",
            "barotropic_velocity")
    }
    substitutions = {}
    for name in (
        "live_reciprocal", "drag_coefficient", "bottom_velocity",
        "barotropic_velocity",
    ):
        accumulated[name] = oracle[name]
        replay = _drag_chain(
            candidate["depth_mean"], accumulated["live_reciprocal"],
            accumulated["drag_coefficient"], accumulated["bottom_velocity"],
            accumulated["barotropic_velocity"],
        )
        substitutions[f"through_{name}"] = r236._row(
            replay["post_drag"], oracle["post_drag"], owner)

    first = next((name for name in OPERAND_ORDER
                  if not rows[name]["bit_exact"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round250-drag-split-v1",
        "status": "PASS_R250_DRAG_SPLIT",
        "label": label,
        "operand_route": "production-d114-resolved",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_admission": admission,
        "passivity": passivity,
        "source_order": list(OPERAND_ORDER),
        "owner_support": r249._score(
            candidate["post_drag"], oracle["post_drag"]),
        "rows": rows,
        "first_unequal": first,
        "self_replay": self_replay,
        "substitutions": substitutions,
        "compiled_source": {
            "coefficient": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1420-1440",
            "residual_and_addition": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1443-1463",
            "rate": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90:215-249",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    if plant == "production-route":
        by_label["independent"]["operand_route"] = "retired-route"
    elif plant == "record-admission":
        by_label["independent"]["record_admission"][0]["status"] = "MOVED"
    elif plant == "self-replay":
        by_label["independent"]["self_replay"]["oracle"]["bit_exact"] = False
    elif plant == "owner-support":
        by_label["independent"]["owner_support"]["unequal"] = 34
    elif plant == "source-order":
        by_label["independent"]["source_order"][0:2] = reversed(
            by_label["independent"]["source_order"][0:2])
    elif plant == "coefficient-owner":
        by_label["independent"]["rows"]["drag_coefficient"]["bit_exact"] = True

    summary = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R250_DRAG_SPLIT",
                f"{label}: measurement status moved")
        require(report["operand_route"] == "production-d114-resolved",
                f"{label}: production operand route moved")
        require(all(report["passivity"].values()),
                f"{label}: passive trace moved bits")
        require(all(row["status"] == "EXACT_DEFINED_BYTES"
                    for row in report["record_admission"]),
                f"{label}: inherited record admission moved")
        require(tuple(report["source_order"]) == OPERAND_ORDER,
                f"{label}: source order moved")
        require(report["owner_support"]["unequal"] == 35,
                f"{label}: 35-face owner support moved")
        require(report["self_replay"]["candidate"]["at_floor"]
                and report["self_replay"]["oracle"]["bit_exact"],
                f"{label}: source statement does not replay itself")
        rows = report["rows"]
        require(rows["stored_reciprocal"]["bit_exact"]
                and rows["live_reciprocal"]["bit_exact"],
                f"{label}: reciprocal precedes the coefficient")
        require(not rows["drag_coefficient"]["bit_exact"],
                f"{label}: drag coefficient is not the first debt")
        require(report["first_unequal"] == "drag_coefficient",
                f"{label}: first unequal operand moved")
        coefficient = report["substitutions"]["through_drag_coefficient"]
        require(coefficient["at_floor"],
                f"{label}: coefficient substitution does not close post-drag")
        summary[label] = {
            "first_unequal": report["first_unequal"],
            "rows": rows,
            "coefficient_substitution": coefficient,
        }
    require(summary["independent"] == summary["given_nemo_entry"],
            "claim labels disagree on drag split")
    return {
        "format": "nemo-testcase-l4-orca2-round250-classification-v1",
        "status": "CONFIRMED_R250_DRAG_COEFFICIENT_OWNER",
        "floor": float(FLOOR),
        "source_order": list(OPERAND_ORDER),
        "labels": summary,
        "predictions": {
            "R250-P1": "CONFIRMED",
            "R250-P2": "CONFIRMED",
            "R250-P3": "PENDING_ATOMIC_SCORE",
            "R250-P4": "PENDING",
            "R250-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), required=True)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--reports", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.twin_a,
                         args.twin_b, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            require(args.plant == "none", "plants apply only to classification")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit,
            )
        else:
            require(args.reports is not None, "classification reports missing")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.reports
            ], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r236.GateError, r249.GateError, rung0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
