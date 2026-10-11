#!/usr/bin/env python3
"""Audit OMT-4's in-loop external-mode fold association at substeps 1..6."""

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
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as r229,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

SUBSTEPS = tuple(range(1, 7))
EXIT_ORDER = ("eta_exit", "u_exit", "v_exit")
POINT_TYPE = {"eta_exit": "T", "u_exit": "U", "v_exit": "V"}
SIGN = {"eta_exit": 1.0, "u_exit": -1.0, "v_exit": -1.0}
PAIR_ORDER = ("v_mid", "face_depth_v_mid", "metric_transport_v")
CLASSIFY_PLANTS = (
    "none", "label-coverage", "substep-order", "fold-sign", "passivity",
    "pair-census", "one-bit",
)


class GateError(RuntimeError):
    """The audit record, trace, or fail-closed classification moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _one_bit_control() -> dict[str, object]:
    left = np.asarray([1.0], np.float64)
    right = np.nextafter(left, np.inf)
    return r228._difference(left, right)


def _native_full(values, point_type: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if point_type == "U":
        return r97._native_u(values)
    if point_type == "V":
        return r97._native_v(values)
    require(point_type == "T", f"unknown point type {point_type!r}")
    return values


def _regional(candidate, oracle, support=None) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape,
            f"regional shape moved: {candidate.shape} != {oracle.shape}")
    require(candidate.shape == (148, 90),
            f"regional slab moved: {candidate.shape}")
    if support is None:
        support = np.ones_like(candidate, dtype=bool)
    support = np.asarray(support, dtype=bool)
    require(support.shape == candidate.shape and np.any(support),
            "regional support is empty or has the wrong shape")
    return {
        "complete": r228._difference(candidate[support], oracle[support]),
        "interior": r228._difference(candidate[:-3][support[:-3]],
                                      oracle[:-3][support[:-3]]),
        "fold_band": r228._difference(candidate[-3:][support[-3:]],
                                       oracle[-3:][support[-3:]]),
    }


def _trace_run(deck_root: Path, frame_root: Path, label: str):
    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

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
            - nemo_source_round(
                jnp.asarray(producer["coriolis_v"]) * raw_vmask)),
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
    return card, observed.substeps, passivity


def measure(
    deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
    label: str, expect_commit: str,
) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(label in ("independent", "given_nemo_entry"),
            f"unknown label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-244 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-244 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-244 audit requires production JIT on CPU")

    admission = r235._admit_streams(twin_a, twin_b)
    oracle_exit = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    oracle_pair = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    card, trace, passivity = _trace_run(deck_root, frame_root, label)
    fold = card.recipe.grid.fold
    support_v = np.asarray(oracle_pair["r1_e1v"] != 0.0, dtype=bool)

    rows = []
    for substep in SUBSTEPS:
        index = substep - 1
        exits = {}
        for name in EXIT_ORDER:
            point = POINT_TYPE[name]
            candidate_full = _native_full(trace[name][index], point)
            candidate = candidate_full[:, :90]
            oracle = np.asarray(oracle_exit[name][index], np.float64)
            exits[name] = {
                "candidate_fold_identity": r228.fold_residual(
                    candidate_full, point, fold, SIGN[name]),
                "candidate_vs_nemo": _regional(candidate, oracle),
            }
        pair_values = {
            "v_mid": (
                _native_full(trace["transport_velocity_v"][index], "V")[:, :90],
                np.asarray(oracle_pair["velocity_v"][index], np.float64)),
            "face_depth_v_mid": (
                _native_full(trace["transport_face_depth_v"][index], "V")[:, :90],
                np.asarray(oracle_pair["face_depth_v"][index], np.float64)),
            "metric_transport_v": (
                _native_full(trace["transport_metric_v"][index], "V")[:, :90],
                np.asarray(oracle_pair["metric_v"][index], np.float64)),
        }
        pair = {
            name: _regional(*pair_values[name], support=support_v)
            for name in PAIR_ORDER
        }
        rows.append({"substep": substep, "exits": exits, "pair": pair})

    return {
        "format": "nemo-testcase-l4-orca2-round244-inloop-fold-v1",
        "status": "PASS_R244_INLOOP_FOLD_AUDIT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "source": {
            "compiled_file": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90",
            "lines": "721-739",
            "statement": "one in-loop lbc_lnk for ua/va/hu/hv/hur/hvr/ssha",
        },
        "record_admission": admission,
        "passivity": passivity,
        "substeps": list(SUBSTEPS),
        "exit_order": list(EXIT_ORDER),
        "pair_order": list(PAIR_ORDER),
        "rows": rows,
        "one_bit_control": _one_bit_control(),
    }


def _first_fold_violation(report: dict) -> dict | None:
    for row in report["rows"]:
        for name in EXIT_ORDER:
            result = row["exits"][name]["candidate_fold_identity"]
            if result["unequal"]:
                return {"substep": row["substep"], "field": name, **result}
    return None


def classify(reports: list[dict], *, plant: str = "none") -> dict[str, object]:
    require(plant in CLASSIFY_PLANTS, f"unknown classify plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "both claim labels are required")
    independent = by_label["independent"]
    if plant == "substep-order":
        independent["substeps"][0] = 2
    elif plant == "fold-sign":
        independent["rows"][0]["exits"]["v_exit"][
            "candidate_fold_identity"]["sign"] = 1.0
    elif plant == "passivity":
        independent["passivity"]["eta"] = False
    elif plant == "pair-census":
        independent["rows"][5]["pair"]["v_mid"]["complete"]["unequal"] = 0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summary = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R244_INLOOP_FOLD_AUDIT",
                f"{label}: measurement status moved")
        require(tuple(report["substeps"]) == SUBSTEPS,
                f"{label}: substep order moved")
        require(tuple(report["exit_order"]) == EXIT_ORDER,
                f"{label}: exit order moved")
        require(tuple(report["pair_order"]) == PAIR_ORDER,
                f"{label}: pair order moved")
        require(all(report["passivity"].values()),
                f"{label}: passive observer moved state")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        for row in report["rows"]:
            for name in EXIT_ORDER:
                require(row["exits"][name]["candidate_fold_identity"]["sign"]
                        == SIGN[name], f"{label}: {name} fold sign moved")
        pair6 = report["rows"][5]["pair"]
        require(pair6["v_mid"]["complete"]["unequal"] == 129,
                f"{label}: round-243 va_e census moved")
        require(pair6["face_depth_v_mid"]["complete"]["unequal"] == 120,
                f"{label}: round-243 zhvp2_e census moved")
        summary[label] = {
            "first_fold_violation": _first_fold_violation(report),
            "rows": report["rows"],
        }

    require(summary["independent"]["first_fold_violation"]
            == summary["given_nemo_entry"]["first_fold_violation"],
            "claim labels disagree on first fold violation")
    first = summary["independent"]["first_fold_violation"]
    return {
        "format": "nemo-testcase-l4-orca2-round244-classification-v1",
        "status": "HELD_R244_INLOOP_FOLD_BOUNDARY_NAMED",
        "source": independent["source"],
        "first_fold_violation": first,
        "rows": summary,
        "predictions": {
            "R244-P1": "CONFIRMED_PASSIVE_ADMITTED_FP64_BOTH_LABELS",
            "R244-P2": "SOURCE_PROVEN_RANK0_RECORD_SUPPORT_ONLY",
            "R244-P3": ("REFUTED_NO_FOLD_VIOLATION" if first is None
                         else f"CONFIRMED_FIRST_{first['field']}_S{first['substep']}"),
            "R244-P4": "MEASURED_PAIR_CENSUS_REPRODUCED",
            "R244-P5": "MEASUREMENT_ONLY_PENDING_ATOMIC_SCORE",
            "R244-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
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
            result = measure(
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
