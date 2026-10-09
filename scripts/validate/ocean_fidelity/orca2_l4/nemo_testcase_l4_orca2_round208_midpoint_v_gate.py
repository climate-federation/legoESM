#!/usr/bin/env python3
"""Split OMT-0 substep-2 midpoint V without observing the executable."""

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
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round205_omt0_substep_walk as r205,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from nemo_testcase_l2_gyre_round14_advmean import read_ordered

PLANTS = (
    "none", "record-header", "forward-coefficients", "candidate-rotation",
    "all-recorded-midpoint", "update-closure",
)
FORWARD = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)


class GateError(RuntimeError):
    """The admitted record, source order, or offline closure moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:149, :90]


def _row(candidate, reference, mask) -> dict[str, object]:
    return r205._score(candidate, reference, mask)


def _midpoint(coefficients, now, before, before_before) -> np.ndarray:
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_midpoint_extrapolation,
    )

    return np.asarray(nemo_literal_midpoint_extrapolation(
        jnp.asarray(coefficients), jnp.asarray(now), jnp.asarray(before),
        jnp.asarray(before_before)))


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "record-header":
        report["record_admission"]["streams"][0]["header_valid"] = False
    elif plant == "forward-coefficients":
        report["coefficients"]["candidate"][1] = 1.0
    elif plant == "candidate-rotation":
        report["history_rotation"]["candidate"]["comparison_bit_exact"] = False
    elif plant == "all-recorded-midpoint":
        report["midpoint_replay"]["all_recorded"]["comparison_bit_exact"] = False
    elif plant == "update-closure":
        report["substep1_flux_form_update"]["status"] = "AT_BAR"

    require(report["claim_label"] == "independent OMT-0", "claim label moved")
    require(all(row["header_valid"] for row in
                report["record_admission"]["streams"]),
            "record header plant or malformed record")
    require(all(row["defined_twin_status"] == "EXACT_DEFINED_BYTES" for row in
                report["record_admission"]["streams"]),
            "ordered twin payload differs")
    require(np.array_equal(report["coefficients"]["candidate"], FORWARD),
            "candidate substep-2 coefficients are not Forward")
    require(np.array_equal(report["coefficients"]["oracle"], FORWARD),
            "oracle substep-2 coefficients are not Forward")
    require(all(report["offline_trace_passivity"].values()),
            "private trace changed a terminal value")
    require(report["history_rotation"]["candidate"]["comparison_bit_exact"],
            "candidate va_e -> vn_e rotation changed bits")
    require(report["history_rotation"]["oracle"]["comparison_bit_exact"],
            "oracle va_e -> vn_e rotation changed bits")
    require(report["midpoint_replay"]["candidate_written_association"][
                "comparison_bit_exact"],
            "offline helper does not reproduce candidate va_e")
    require(report["midpoint_replay"]["candidate_midpoint_equals_now"][
                "comparison_bit_exact"],
            "Forward midpoint is not the current vn_e")
    require(report["midpoint_replay"]["all_recorded"]["comparison_bit_exact"],
            "recorded midpoint operands do not close va_e bit-exactly")
    require(report["midpoint_replay"]["first_effective_input"] == "v_entry",
            "first effective midpoint input moved")
    require(report["substep1_flux_form_update"]["status"] ==
            "UNMEASURED_WITH_SPEC",
            "unrecorded flux-form operands were treated as a closure")
    require(tuple(report["substep1_flux_form_update"]["missing_oracle_operands"])
            == ("hv_e", "zhv_bck", "hv_0_times_1_plus_r3v_Kmm"),
            "flux-form missing-operand specification moved")
    report["status"] = "PASS_R208_MIDPOINT_V_SPLIT"
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
        expect_commit.lower()), "round-208 measurement requires its clean commit")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-208 replay requires production JIT on CPU")

    admission = r205._admit_streams(twin_a, twin_b)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    card = omt0.build_omt0_card(deck_root)
    omt0.validate_omt0_card(card)
    cfg, state = card.recipe.model_config, card.recipe.initial_state
    freshwater, surface = rung0_ladder._zero_forcing(state.eta.data.shape)
    del freshwater
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    model.prime_step_caches(state)
    slow_u, slow_v = r205._slow_forcing(model, card, state, surface)
    zero_eta = np.zeros(state.eta.data.shape, dtype=np.float64)
    dt = np.float64(card.dt_s / 65)

    def solve(trace: bool):
        return jax.device_get(jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, dt, 65, card.recipe.grid, card.recipe.z_coord, cfg,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
            )
        ))(state, jnp.asarray(zero_eta), jnp.asarray(slow_u), jnp.asarray(slow_v)))

    live, traced = solve(False), solve(True)
    passivity = r205._terminal_rows(live, traced)
    trace = traced[2]
    masks = r15._masks(card)
    index = 1
    candidate_coeff = np.asarray([
        trace["mid_weight_1"][index], trace["mid_weight_2"][index],
        trace["mid_weight_3"][index]], dtype=np.float64)
    oracle_coeff = np.asarray(ordered["weights"][index, :3], dtype=np.float64)
    candidate = {
        "v_entry": _native_v(trace["v_entry"][index]),
        "v_history_b": _native_v(trace["v_history_b"][index]),
        "v_history_bb": _native_v(trace["v_history_bb"][index]),
    }
    oracle = {
        "v_entry": np.asarray(ordered["v_entry"][index]),
        "v_history_b": np.asarray(ordered["v_history_b"][index]),
        "v_history_bb": np.asarray(ordered["v_history_bb"][index]),
    }
    candidate_mid = _native_v(trace["v_mid"][index])
    oracle_mid = np.asarray(ordered["v_mid"][index])
    candidate_replay = _midpoint(candidate_coeff, *candidate.values())
    recorded_now = _midpoint(candidate_coeff, oracle["v_entry"],
                             candidate["v_history_b"],
                             candidate["v_history_bb"])
    all_recorded = _midpoint(oracle_coeff, *oracle.values())
    input_rows = {name: _row(candidate[name], oracle[name], masks["v"])
                  for name in candidate}
    midpoint = {
        "source_statement": "dynspg_ts.f90:461-484",
        "input_order": list(candidate),
        "input_rows": input_rows,
        "candidate_output": _row(candidate_mid, oracle_mid, masks["v"]),
        "candidate_written_association": _row(
            candidate_replay, candidate_mid, masks["v"]),
        "candidate_midpoint_equals_now": _row(
            candidate_mid, candidate["v_entry"], masks["v"]),
        "recorded_now_only": _row(recorded_now, oracle_mid, masks["v"]),
        "all_recorded": _row(all_recorded, oracle_mid, masks["v"]),
        "first_effective_input": next(
            name for name in candidate if not input_rows[name]["comparison_bit_exact"]),
    }
    candidate_rotation = _row(
        _native_v(trace["v_entry"][1]), _native_v(trace["v_exit"][0]), masks["v"])
    oracle_rotation = _row(
        ordered["v_entry"][1], ordered["v_exit"][0], masks["v"])
    raw = {
        "format": "nemo-testcase-l4-orca2-round208-midpoint-v1",
        "claim_label": "independent OMT-0",
        "execution": "offline-production-pure-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "dtype": str(state.eta.data.dtype),
        "record_admission": admission,
        "coefficients": {
            "candidate": candidate_coeff.tolist(),
            "oracle": oracle_coeff.tolist(),
        },
        "offline_trace_passivity": passivity,
        "history_rotation": {
            "source_statement": "dynspg_ts.f90:787-789",
            "candidate": candidate_rotation,
            "oracle": oracle_rotation,
        },
        "midpoint_replay": midpoint,
        "substep1_flux_form_update": {
            "source_statement": "dynspg_ts.f90:681-702",
            "status": "UNMEASURED_WITH_SPEC",
            "missing_oracle_operands": [
                "hv_e", "zhv_bck", "hv_0_times_1_plus_r3v_Kmm"],
            "required_number": (
                "substep-1 va_e from the source-written flux-form expression "
                "given all eight recorded NEMO operands"),
            "available_recorded_operands": [
                "vn_e", "zv_spg", "zhvp2_e", "zv_trd", "zv_frc",
                "z1_hv", "va_e"],
        },
        "compiled_source": {
            "coefficients_and_midpoint": (
                "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:461-484"),
            "flux_form_update": (
                "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:681-702"),
            "history_rotation": (
                "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:781-789"),
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
    except (OSError, ValueError, KeyError, StopIteration, TypeError, GateError,
            r205.GateError, omt0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R208_MIDPOINT_V_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
