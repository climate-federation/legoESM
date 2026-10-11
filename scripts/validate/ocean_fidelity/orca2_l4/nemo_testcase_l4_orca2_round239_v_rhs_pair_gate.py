#!/usr/bin/env python3
"""Gate round 239's Ve_rhs pair and source-order V-RHS attribution."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as r179,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
)

PLANTS = ("none", "pair-effect", "label-coverage", "source-order", "first-operator")
ORDER = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")


class GateError(RuntimeError):
    """The frozen pair/attribution evidence no longer supports its verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits_equal(left, right) -> bool:
    left = np.ascontiguousarray(np.asarray(left, np.float64)).view(np.uint64)
    right = np.ascontiguousarray(np.asarray(right, np.float64)).view(np.uint64)
    return bool(np.array_equal(left, right))


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "pair-effect":
        report["pair_completed_rhs"]["independent"]["unequal"] = 0
    elif plant == "label-coverage":
        report["pair_completed_rhs"].pop("given_nemo_entry")
    elif plant == "source-order":
        report["operator_table"]["source_order"][0:2] = reversed(
            report["operator_table"]["source_order"][0:2])
    elif plant == "first-operator":
        report["operator_table"]["first_operator"]["boundary"] = "after_ldf"

    pair = report["pair_completed_rhs"]
    require(set(pair) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")
    for label, row in pair.items():
        require(row["support"] == 1050, f"{label}: owner support moved")
        require(row["unequal"] == 668,
                f"{label}: Ve_rhs pair unexpectedly changed completed V RHS")
        require(row["pair_on_equals_pair_off"],
                f"{label}: downstream pair changed its upstream completed V RHS")
        require(all(row["pair_on_passivity"].values()),
                f"{label}: pair-on trace moved completed state")
        require(all(row["pair_off_passivity"].values()),
                f"{label}: pair-off trace moved completed state")
    require(pair["independent"]["max_abs"] == pair["given_nemo_entry"]["max_abs"],
            "claim labels disagree on completed V RHS")

    table = report["operator_table"]
    require(tuple(table["source_order"]) == ORDER, "compiled source order moved")
    require(table["target"] == {"cells": 35, "row": 147, "all_on_one_row": True},
            "35-face fold target moved")
    first = next(
        ({"boundary": name, **table["operator_rows"][name]}
         for name in ORDER if not table["operator_rows"][name]["at_floor"]),
        None,
    )
    require(first == table["first_operator"], "first operator selector moved")
    require(first is not None and first["boundary"] == "after_hpg",
            "HPG is no longer the first non-bit V-RHS accumulator")
    require(table["candidate_calibration"]["same_over_floor_set"],
            "offline operator replay does not reproduce the live debt set")
    require(table["candidate_calibration"]["all_faces_at_floor"],
            "offline operator replay does not close its live endpoint")
    report["predictions"] = {
        "R239-P1": "CONFIRMED_ADMITTED_PASSIVE_BOTH_LABELS",
        "R239-P2": "REFUTED_PAIR_LEAVES_668_OF_1050_RHS_SLOTS_UNEQUAL",
        "R239-P3": "NOT_ACTIVATED_PREREQUISITE_R239-P2",
        "R239-P4": "REFUTED_PREREQUISITE_R239-P2",
        "R239-P5": "NOT_ACTIVATED_PREREQUISITE_R239-P4",
        "R239-P6": "PENDING_DOCUMENTARY_GATES",
    }
    report["status"] = "HELD_R239_FIRST_V_RHS_OPERATOR_HPG"
    return report


def _pair_row(card, state, freshwater, surface, oracle_rhs) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    plain_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    off_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    seed = jax.device_get(off_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(seed, _NEMOWSLiveOperandTrace), "seed trace type moved")
    producer = seed.slow_forcing_producer
    raw_mask = r236.r229._raw_vmask(card)
    b = nemo_source_round
    slow_override = (
        jnp.asarray(producer["final_u"]),
        b(jnp.asarray(producer["incoming_v"])
          - b(jnp.asarray(producer["coriolis_v"]) * raw_mask)),
    )
    on_plain_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            barotropic_slow_forcing_override=slow_override,
            barotropic_atomic_fold_unit=True))
    on_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            barotropic_slow_forcing_override=slow_override,
            barotropic_atomic_fold_unit=True))
    plain = jax.device_get(plain_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    off = seed
    on_plain = jax.device_get(on_plain_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    on = jax.device_get(on_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(off, _NEMOWSLiveOperandTrace)
            and isinstance(on, _NEMOWSLiveOperandTrace), "trace type moved")
    off_rhs = r236._native_v(off.slow_forcing_producer["rhs_v"])
    on_rhs = r236._native_v(on.slow_forcing_producer["rhs_v"])
    oracle_rhs = np.asarray(oracle_rhs, np.float64)
    owner = on_rhs != oracle_rhs
    owner[-3:-1] = False
    owner[:-3] = False
    # The registered support is the 35 fold-halo faces times 30 levels.
    surface_owner = np.any(on_rhs != oracle_rhs, axis=-1)
    require(int(np.count_nonzero(surface_owner)) == 35, "fold-face support moved")
    support = np.broadcast_to(surface_owner[..., None], on_rhs.shape)
    delta = np.abs(on_rhs - oracle_rhs)
    return {
        "support": int(np.count_nonzero(support)),
        "unequal": int(np.count_nonzero((on_rhs != oracle_rhs) & support)),
        "max_abs": float(np.max(delta[support], initial=0.0)),
        "argmax": [int(v) for v in np.unravel_index(
            int(np.argmax(np.where(support, delta, -np.inf))), delta.shape)],
        "pair_on_equals_pair_off": _bits_equal(on_rhs, off_rhs),
        "pair_off_passivity": passive._ordinary_state_equal(off.state_after, plain),
        "pair_on_passivity": passive._ordinary_state_equal(on.state_after, on_plain),
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == args.expect_commit.lower(),
            "round-239 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-239 measurement requires production JIT on CPU")

    slow_reports = {
        "independent": json.loads(args.independent_slow.read_text()),
        "given_nemo_entry": json.loads(args.given_slow.read_text()),
    }
    for label, report in slow_reports.items():
        require(report["label"] == label and report["status"] == "PASS_R236_SLOW_V_SPLIT",
                f"{label}: inherited slow split did not pass")
        require(report["rows"]["completed_v_rhs"]["unequal"] == 668,
                f"{label}: inherited completed V RHS moved")

    card = omt4.build_omt4_card(args.omt4_root)
    omt4.validate_omt4_card(args.omt4_root, card)
    entry = rung0.assemble_frame(args.omt4_root, 1, 0)
    states = {
        "independent": card.recipe.initial_state,
        "given_nemo_entry": rung0.bridge_entry(card, entry),
    }
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    oracle_slow = r236.read_slow_forcing(
        args.omt4_root / "oracle_slow_forcing_kt00000001.bin",
        dims=(94, 152, 31))
    pair = {
        label: _pair_row(card, state, freshwater, surface, oracle_slow["krhs_v"])
        for label, state in states.items()
    }

    original_classify = r179.classify
    try:
        r179.classify = lambda report, plant="none": report
        table = r179.measure(
            args.rung0_root, args.rhs_root, args.slow_root,
            args.static_root, args.spg_root, args.expect_commit)
    finally:
        r179.classify = original_classify

    report = {
        "format": "nemo-testcase-l4-orca2-round239-v-rhs-pair-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "pair_completed_rhs": pair,
        "operator_table": {
            key: table[key] for key in (
                "source_order", "target", "operator_rows", "first_operator",
                "candidate_calibration", "trace_passivity", "record_census",
                "record_control", "endpoint_ulp_control", "compiled_source")
        },
        "compiled_source": {
            "depth_average": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-200",
            "copy_pair": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-328",
            "operator_order": table["compiled_source"]["operator_order"],
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--omt4-root", type=Path)
    parser.add_argument("--rung0-root", type=Path)
    parser.add_argument("--rhs-root", type=Path)
    parser.add_argument("--slow-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--independent-slow", type=Path)
    parser.add_argument("--given-slow", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            required = (args.omt4_root, args.rung0_root, args.rhs_root,
                        args.slow_root, args.static_root, args.spg_root,
                        args.independent_slow, args.given_slow, args.expect_commit)
            require(all(value is not None for value in required),
                    "measurement arguments incomplete")
            result = measure(args)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            r179.GateError, rung0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
