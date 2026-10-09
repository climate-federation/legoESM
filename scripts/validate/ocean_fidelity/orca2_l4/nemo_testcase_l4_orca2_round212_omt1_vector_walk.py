#!/usr/bin/env python3
"""Walk OMT-1's kt=1 vector-form split-explicit solve offline."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as r15,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round205_omt0_substep_walk as r205,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round197_vector_v_update as r197,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_ordered

PLANTS = (
    "none", "record-header", "twin-ulp", "source-order", "passivity",
    "terminal-ulp", "slow-v-replay", "vector-mask-replay",
)


class GateError(RuntimeError):
    """The admitted OMT-1 source-ordered replay moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first(rows: list[dict[str, object]], key: str) -> dict[str, object] | None:
    return next((row for row in rows if not bool(row[key])), None)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "record-header":
        report["record_admission"]["streams"][0]["header_valid"] = False
    elif plant == "twin-ulp":
        report["record_admission"]["streams"][0]["defined_twin_status"] = "DIFF"
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "passivity":
        report["offline_replay_passivity"]["ssh"] = False
    elif plant == "terminal-ulp":
        report["terminal_ulp_control"]["bit_exact"] = True
    elif plant == "slow-v-replay":
        report["slow_v_arm"]["input"]["comparison_bit_exact"] = False
    elif plant == "vector-mask-replay":
        report["vector_v_split"]["record_replay_vs_target"]["bit_exact"] = False

    require(report["claim_label"] == "independent OMT-1", "claim label moved")
    require(report["record_admission"]["stream_count"] == 2,
            "stream census moved")
    require(all(row["header_valid"]
                for row in report["record_admission"]["streams"]),
            "record header plant or malformed record")
    require(all(row["defined_twin_status"] == "EXACT_DEFINED_BYTES"
                for row in report["record_admission"]["streams"]),
            "barotropic twin payload differs")
    require(tuple(report["source_order"]) == r205.ENTRY_ORDER,
            "entry source order moved")
    require(all(report["offline_replay_passivity"].values()),
            "offline trace does not reproduce the untraced pure solver")
    require(len(report["substep_table"]) == 65, "substep table is incomplete")
    require(report["slow_v_arm"]["input"]["comparison_bit_exact"],
            "recorded slow-V substitution did not install exactly")
    require(len(report["slow_v_arm"]["substep_table"]) == 65,
            "recorded slow-V replay is incomplete")
    split = report["vector_v_split"]
    require(split["input_order"] == list(r197.INPUT_ORDER),
            "vector-V input order moved")
    require(split["candidate_replay_vs_passive"]["bit_exact"],
            "candidate vector-V replay does not reproduce the passive trace")
    require(split["record_replay_vs_target"]["bit_exact"],
            "all-recorded vector-V replay does not reproduce NEMO")
    require(report["first_nonbit"] == _first(
        report["source_rows"], "comparison_bit_exact"),
        "first non-bit selector moved")
    require(report["first_over_floor"] == _first(
        report["source_rows"], "at_floor"),
        "first-over-floor selector moved")
    require(report["terminal_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "terminal one-ULP control did not fire")
    report["status"] = "PASS_R212_OMT1_VECTOR_WALK"
    return report


def measure(deck_root: Path, twin_a: Path, twin_b: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower().startswith(
        expect_commit.lower()),
        "round-212 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-212 replay requires production JIT on CPU")

    admission = r205._admit_streams(twin_a, twin_b)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    card = omt1.build_omt1_card(deck_root)
    omt1.validate_omt1_card(card)
    cfg = card.recipe.model_config
    resolved = {
        "dtype": str(card.recipe.initial_state.eta.data.dtype),
        "outer_integrator": cfg.outer_integrator,
        "momentum_time_integrator": cfg.momentum_time_integrator,
        "momentum_advection": cfg.momentum_advection,
        "momentum_flux_scheme": cfg.momentum_flux_scheme,
        "vertical_momentum_scheme": cfg.vertical_momentum_scheme,
        "barotropic_solver": cfg.barotropic.barotropic_solver,
        "n_substeps": cfg.barotropic.n_barotropic_substeps,
        "time_filter": cfg.barotropic.barotropic_time_filter,
        "continuity": cfg.barotropic.barotropic_continuity_evaluation,
        "transport": cfg.barotropic.barotropic_transport_accumulation_evaluation,
        "face_depth": cfg.barotropic.barotropic_face_depth,
        "pgf": cfg.barotropic.barotropic_pgf_evaluation,
        "coriolis": cfg.barotropic.barotropic_coriolis,
        "drag": bool(cfg.barotropic_drag_substep),
    }
    require(resolved == {
        "dtype": "float64", "outer_integrator": "forward_euler",
        "momentum_time_integrator": "rk3_ws",
        "momentum_advection": "vector_invariant",
        "momentum_flux_scheme": "upwind",
        "vertical_momentum_scheme": "nemo_advective",
        "barotropic_solver": "explicit_substep", "n_substeps": 65,
        "time_filter": "nemo_ab3am4", "continuity": "nemo_literal",
        "transport": "nemo_literal", "face_depth": "nemo_ssh_avg",
        "pgf": "nemo_literal", "coriolis": "een_metric", "drag": False,
    }, f"resolved OMT-1 identity moved: {resolved}")

    state = card.recipe.initial_state
    freshwater, surface = rung0_ladder._zero_forcing(state.eta.data.shape)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    model.prime_step_caches(state)
    slow_u, slow_v = r205._slow_forcing(model, card, state, surface)
    zero_eta = np.zeros(state.eta.data.shape, dtype=np.float64)
    dt = np.float64(card.dt_s / 65)
    require(np.array_equal(np.asarray([dt]), np.asarray([ordered["dt"]])),
            f"external substep dt moved: {dt} != {ordered['dt']}")

    def solve(trace: bool, slow_v_override=None):
        selected_v = slow_v if slow_v_override is None else slow_v_override
        return jax.device_get(jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, dt, 65, card.recipe.grid, card.recipe.z_coord, cfg,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
            )
        ))(state, jnp.asarray(zero_eta), jnp.asarray(slow_u),
           jnp.asarray(selected_v)))

    live = solve(False)
    traced = solve(True)
    passivity = r205._terminal_rows(live, traced)
    require(all(passivity.values()), f"offline trace is not passive: {passivity}")
    trace = traced[2]
    masks = r15._masks(card)
    candidate_ordered = r15._candidate_ordered(card, trace)
    entry_candidates = {
        "continuity_forcing": candidate_ordered["continuity_forcing"][0],
        "slow_u": r205._native_u(slow_u), "slow_v": r205._native_v(slow_v),
        "eta_entry": candidate_ordered["eta_entry"][0],
        "u_entry": candidate_ordered["u_entry"][0],
        "v_entry": candidate_ordered["v_entry"][0],
        "eta_history_b": candidate_ordered["eta_history_b"][0],
        "eta_history_bb": candidate_ordered["eta_history_bb"][0],
        "u_history_b": candidate_ordered["u_history_b"][0],
        "u_history_bb": candidate_ordered["u_history_bb"][0],
        "v_history_b": candidate_ordered["v_history_b"][0],
        "v_history_bb": candidate_ordered["v_history_bb"][0],
    }
    face = {name: ("u" if "_u" in name or name.startswith("u_") else
                   "v" if "_v" in name or name.startswith("v_") else "t")
            for name in r205.ENTRY_ORDER}
    entry_rows = [{"name": name, **r205._score(
        entry_candidates[name],
        ordered["continuity_forcing"][0] if name == "continuity_forcing"
        else ordered[name][0], masks[face[name]])}
        for name in r205.ENTRY_ORDER]
    substep_rows, substep_table = r205._summary_table(trace, substeps, masks)
    source_rows = entry_rows + substep_rows
    first_nonbit = _first(source_rows, "comparison_bit_exact")
    first_over = _first(source_rows, "at_floor")
    recorded_slow_v = np.asarray(slow_v).copy()
    recorded_slow_v[1:149, :90] = ordered["slow_v"][0]
    slow_v_arm = solve(True, slow_v_override=recorded_slow_v)
    slow_v_rows, slow_v_table = r205._summary_table(
        slow_v_arm[2], substeps, masks)
    index = 0
    raw_mask = card.recipe.z_coord.nemo_een_barotropic
    require(raw_mask is not None, "OMT-1 card has no raw NEMO mask bundle")
    candidate_inputs = {
        "vn_e": r205._native_v(trace["v_entry"][index]),
        "rDt_e": dt,
        "zv_spg": r205._native_v(trace["pgf_v"][index]),
        "zv_trd": r205._native_v(trace["trd_v"][index]),
        "zv_frc": r205._native_v(trace["slow_v"][index]),
        "ssvmask": r205._native_v(np.asarray(state.v_mask.data)),
    }
    reference_inputs = {
        "vn_e": np.asarray(substeps["v_entry"][index]),
        "rDt_e": np.float64(ordered["dt"]),
        "zv_spg": np.asarray(substeps["pgf_v"][index]),
        "zv_trd": np.asarray(substeps["trd_v"][index]),
        "zv_frc": np.asarray(substeps["slow_v"][index]),
        "ssvmask": np.max(np.asarray(raw_mask.vmask, dtype=np.float64), axis=-1),
    }
    target_v = np.asarray(substeps["v_exit"][index])
    candidate_terms = r197._literal_terms(candidate_inputs)
    reference_terms = r197._literal_terms(reference_inputs)
    candidate_post = r197._associate_v(candidate_terms["raw_va_e"], card)
    reference_post = r197._associate_v(reference_terms["raw_va_e"], card)
    passive_post = r205._native_v(trace["v_exit"][index])
    operand_rows = {
        name: r197._row(candidate_inputs[name], reference_inputs[name])
        for name in r197.INPUT_ORDER
    }
    cumulative_rows = {}
    accumulated = dict(candidate_inputs)
    for name in r197.INPUT_ORDER:
        accumulated[name] = reference_inputs[name]
        terms = r197._literal_terms(accumulated)
        cumulative_rows[name] = r197._row(
            r197._associate_v(terms["raw_va_e"], card), target_v)
    one = np.asarray([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    raw = {
        "format": "nemo-testcase-l4-orca2-round212-omt1-vector-walk-v1",
        "claim_label": "independent OMT-1",
        "execution": "offline-production-pure-jit-cpu-fp64-x64-libm",
        "floor": float(r205.FLOOR), "worktree": stamp,
        "record_admission": admission, "resolved": resolved,
        "source_order": list(r205.ENTRY_ORDER), "entry_rows": entry_rows,
        "entry_first_nonbit": _first(entry_rows, "comparison_bit_exact"),
        "offline_replay_passivity": passivity,
        "substep_source_order": list(r205.SUBSTEP_ORDER),
        "source_rows": source_rows,
        "substep_rows": substep_rows, "substep_table": substep_table,
        "first_nonbit": first_nonbit, "first_over_floor": first_over,
        "slow_v_arm": {
            "source_statement": "dynspg_ts.f90:289,320-324",
            "input": r205._score(r205._native_v(recorded_slow_v),
                                  ordered["slow_v"][0], masks["v"]),
            "terminal": r205._terminal_rows(live, slow_v_arm),
            "first_nonbit": _first(slow_v_rows, "comparison_bit_exact"),
            "first_over_floor": _first(slow_v_rows, "at_floor"),
            "substep_table": slow_v_table,
        },
        "vector_v_split": {
            "source_statement": "dynspg_ts.f90:674-678",
            "input_order": list(r197.INPUT_ORDER),
            "operand_rows": operand_rows,
            "candidate_replay_vs_passive": r197._row(
                candidate_post, passive_post),
            "record_replay_vs_target": r197._row(reference_post, target_v),
            "cumulative_substitution_post_v": cumulative_rows,
        },
        "terminal_ulp_control": {
            "bit_exact": bool(np.array_equal(one, next_one)),
            "differing_cells": int(np.count_nonzero(one != next_one)),
        },
        "compiled_source": {
            "forcing": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:286-324",
            "entry": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:339-381",
            "substeps": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:445-813",
            "vector_update": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:655-679",
        },
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.twin_a, args.twin_b,
                         args.expect_commit)), "measurement arguments missing")
            result = measure(args.deck_root, args.twin_a, args.twin_b,
                             args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            r205.GateError, omt1.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R212_OMT1_VECTOR_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
