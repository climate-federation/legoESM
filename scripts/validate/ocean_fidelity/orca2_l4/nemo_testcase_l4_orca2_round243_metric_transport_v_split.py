#!/usr/bin/env python3
"""Split OMT-4 substep-6 ``zhV=e1v*va_e*zhvp2_e`` offline."""

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
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as r229,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
)
from nemo_testcase_l2_gyre_round14_advmean import (  # noqa: E402
    read_advmean,
    read_ordered,
)

FLOOR = np.float64(2.0e-10)
SUBSTEP = 6
OPERAND_ORDER = ("e1v", "va_e", "zhvp2_e")
CLASSIFY_PLANTS = (
    "none", "label-coverage", "source-order", "operand-ulp",
    "replay-closure", "passivity",
)


class GateError(RuntimeError):
    """The admitted record cannot support the frozen round-243 split."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, np.float64)).view(np.uint64)


def _score(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    require(left.shape == right.shape, f"shape mismatch {left.shape} != {right.shape}")
    unequal = _bits(left) != _bits(right)
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.zeros_like(left)
    np.subtract(left, right, out=delta, where=finite)
    absolute = np.abs(delta)
    if np.any(finite):
        flat = int(np.argmax(np.where(finite, absolute, -np.inf)))
        argmax = list(map(int, np.unravel_index(flat, left.shape)))
        maximum = float(absolute[tuple(argmax)])
    else:
        argmax = None
        maximum = None
    nonfinite = np.argwhere(~finite)
    return {
        "support": int(left.size),
        "unequal": int(np.count_nonzero(unequal)),
        "over_floor": int(np.count_nonzero(finite & (absolute > FLOOR))),
        "signed_zero_only": int(np.count_nonzero(
            unequal & (left == 0.0) & (right == 0.0))),
        "max_abs": maximum,
        "argmax": argmax,
        "first_nonfinite": (
            list(map(int, nonfinite[0])) if nonfinite.size else None),
    }


def _regional(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    require(left.shape == right.shape == (148, 90),
            f"expected rank-0 V slab, got {left.shape} / {right.shape}")
    return {
        "complete": _score(left, right),
        "interior": _score(left[:-3], right[:-3]),
        "fold_band": _score(left[-3:], right[-3:]),
    }


def _product(values: dict[str, np.ndarray]) -> np.ndarray:
    """Replay compiled dynspg_ts.f90:535 in its written association."""
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    first = nemo_source_round(
        nemo_source_round(jnp.asarray(values["e1v"]))
        * nemo_source_round(jnp.asarray(values["va_e"])))
    return np.asarray(nemo_source_round(
        nemo_source_round(first)
        * nemo_source_round(jnp.asarray(values["zhvp2_e"]))))


def _one_ulp_control() -> dict[str, object]:
    base = np.asarray([[1.0]], np.float64)
    moved = np.nextafter(base, np.asarray([[np.inf]], np.float64))
    return _score(base, moved)


def _measure(
    deck_root: Path,
    frame_root: Path,
    twin_a: Path,
    twin_b: Path,
    label: str,
    expect_commit: str,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-243 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-243 measurement requires production JIT on CPU")

    admission = r235._admit_streams(twin_a, twin_b)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    producer_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            barotropic_atomic_fold_unit=True))
    producer_trace = jax.device_get(producer_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(producer_trace, _NEMOWSLiveOperandTrace),
            "slow-forcing producer trace return type moved")
    producer = producer_trace.slow_forcing_producer
    raw_vmask = r229._raw_vmask(card)
    slow_override = (
        jnp.asarray(producer["final_u"]),
        nemo_source_round(
            jnp.asarray(producer["incoming_v"])
            - nemo_source_round(jnp.asarray(producer["coriolis_v"]) * raw_vmask)),
    )

    def run(expose: bool):
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=expose,
            expose_barotropic_boundary_association=expose,
            barotropic_slow_forcing_override=slow_override,
            barotropic_atomic_fold_unit=True,
        )
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    ordinary = run(False)
    observed = run(True)
    require(isinstance(observed, _NEMOWSBoundaryAssociationTrace),
            "barotropic trace return type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary)
    require(all(passivity.values()), "barotropic observer moved completed state")

    index = SUBSTEP - 1
    trace = observed.substeps
    candidate = {
        "e1v": r235._native(card.recipe.grid.dx_v, "v"),
        "va_e": r235._native(trace["transport_velocity_v"][index], "v"),
        "zhvp2_e": r235._native(trace["transport_face_depth_v"][index], "v"),
    }
    reference = {
        # e1v is static; the source-complete two-row stream records it.
        "e1v": np.asarray(ordered["metric_e1v"][0], np.float64),
        "va_e": np.asarray(advmean["velocity_v"][index], np.float64),
        "zhvp2_e": np.asarray(advmean["face_depth_v"][index], np.float64),
    }
    oracle = np.asarray(advmean["metric_v"][index], np.float64)
    observed_product = r235._native(trace["transport_metric_v"][index], "v")
    require(all(value.shape == (148, 90) for value in (*candidate.values(),
                                                       *reference.values(), oracle)),
            "operand slab shape moved")

    operand_rows = {
        name: _regional(candidate[name], reference[name]) for name in OPERAND_ORDER
    }
    candidate_product = _product(candidate)
    cumulative = dict(candidate)
    cumulative_rows = {}
    for name in OPERAND_ORDER:
        cumulative[name] = reference[name]
        cumulative_rows[name] = _regional(_product(cumulative), oracle)

    first_nonbit = next((name for name in OPERAND_ORDER
                         if operand_rows[name]["complete"]["unequal"]), None)
    first_over_floor = next((name for name in OPERAND_ORDER
                             if operand_rows[name]["complete"]["over_floor"]), None)
    first_interior_closure = next((name for name in OPERAND_ORDER
                                   if cumulative_rows[name]["interior"]["over_floor"] == 0),
                                  None)
    return {
        "format": "nemo-testcase-l4-orca2-round243-metric-v-split-v1",
        "status": "PASS_R243_METRIC_V_OPERAND_SPLIT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "source": {
            "compiled_file": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90",
            "lines": "530-536",
            "statement": "zhV=e1v*va_e*zhvp2_e",
        },
        "substep": SUBSTEP,
        "operand_order": list(OPERAND_ORDER),
        "record_admission": admission,
        "passivity": passivity,
        "instantiated": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in candidate.items()
        },
        "operand_rows": operand_rows,
        "candidate_product_vs_oracle": _regional(candidate_product, oracle),
        "candidate_replay_vs_production_trace": _regional(
            candidate_product, observed_product),
        "cumulative_substitution": cumulative_rows,
        "first_nonbit_operand": first_nonbit,
        "first_over_floor_operand": first_over_floor,
        "first_interior_at_floor_after": first_interior_closure,
        "one_ulp_control": _one_ulp_control(),
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict:
    require(plant in CLASSIFY_PLANTS, f"unknown classify plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "both claim labels are required")
    if plant == "source-order":
        by_label["independent"]["operand_order"][:2] = reversed(
            by_label["independent"]["operand_order"][:2])
    elif plant == "operand-ulp":
        by_label["independent"]["one_ulp_control"]["unequal"] = 0
    elif plant == "replay-closure":
        by_label["independent"]["cumulative_substitution"]["zhvp2_e"][
            "complete"]["unequal"] = 1
    elif plant == "passivity":
        by_label["independent"]["passivity"]["eta"] = False

    for label, report in by_label.items():
        require(report["status"] == "PASS_R243_METRIC_V_OPERAND_SPLIT",
                f"{label}: measurement status moved")
        require(tuple(report["operand_order"]) == OPERAND_ORDER,
                f"{label}: source operand order moved")
        require(all(report["passivity"].values()),
                f"{label}: passive observer moved the state")
        require(report["one_ulp_control"]["unequal"] == 1
                and report["one_ulp_control"]["max_abs"] > 0.0,
                f"{label}: one-ULP control did not fire")
        require(report["candidate_replay_vs_production_trace"]["complete"][
                    "unequal"] == 0,
                f"{label}: offline product does not reproduce production trace")
        require(report["cumulative_substitution"]["zhvp2_e"]["complete"][
                    "unequal"] == 0,
                f"{label}: all-recorded operand replay is not bit-exact")
        require(all(row["dtype"] == "float64"
                    for row in report["instantiated"].values()),
                f"{label}: operand dtype moved")

    summary = {}
    for label, report in by_label.items():
        summary[label] = {
            "first_nonbit_operand": report["first_nonbit_operand"],
            "first_over_floor_operand": report["first_over_floor_operand"],
            "first_interior_at_floor_after": report["first_interior_at_floor_after"],
            "operand_rows": report["operand_rows"],
            "candidate_product_vs_oracle": report["candidate_product_vs_oracle"],
            "cumulative_substitution": report["cumulative_substitution"],
        }
    identity = lambda row: (
        row["first_nonbit_operand"], row["first_over_floor_operand"],
        row["first_interior_at_floor_after"])
    require(identity(summary["independent"]) == identity(summary["given_nemo_entry"]),
            "claim labels disagree on operand attribution")

    first = summary["independent"]
    predictions = {
        "R243-P1": "CONFIRMED_PASSIVE_ADMITTED_FP64_BOTH_LABELS",
        "R243-P2": ("CONFIRMED_E1V_BIT_EXACT"
                    if first["operand_rows"]["e1v"]["complete"]["unequal"] == 0
                    else "REFUTED_E1V_NONBIT"),
        "R243-P3": ("CONFIRMED_VA_E_FIRST_NONBIT"
                    if first["first_nonbit_operand"] == "va_e"
                    else f"REFUTED_FIRST_{first['first_nonbit_operand']}"),
        "R243-P4": ("CONFIRMED_CLOSES_BY_VA_E"
                    if first["first_interior_at_floor_after"] == "va_e"
                    else f"REFUTED_CLOSES_BY_{first['first_interior_at_floor_after']}"),
        "R243-P5": "MEASUREMENT_ONLY_NO_PACKAGE_CHANGE",
        "R243-P6": "CONFIRMED" if plant == "none" else "PLANT",
    }
    return {
        "format": "nemo-testcase-l4-orca2-round243-classification-v1",
        "status": "HELD_R243_METRIC_V_OPERANDS_NAMED",
        "source": by_label["independent"]["source"],
        "substep": SUBSTEP,
        "operand_order": list(OPERAND_ORDER),
        "rows": summary,
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--classify-plant", choices=CLASSIFY_PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            reports = [json.loads(path.read_text(encoding="utf-8"))
                       for path in args.classify]
            result = classify(reports, plant=args.classify_plant)
        else:
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)),
                "measurement arguments are incomplete")
            result = _measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        planted = args.classify is not None and args.classify_plant != "none"
        print(f"STATUS {'PLANT-FIRED' if planted else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
