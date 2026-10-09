#!/usr/bin/env python3
"""Split rung-0 substep-3 ``zhV`` across its three compiled operands."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round146_boundary_association_gate as r146,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)

SUBSTEP = 3
PLANTS = ("none", "registry-order", "operand-bit", "missing-stream")


class GateError(RuntimeError):
    """The record, trace, or source-ordered classification moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate a frozen measurement and exercise its fail-closed controls."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "registry-order":
        report["operand_order"][0], report["operand_order"][1] = (
            report["operand_order"][1], report["operand_order"][0])
    elif plant == "operand-bit":
        report["split"]["operand_rows"]["e1v"]["bit_exact"] = False
        report["split"]["operand_rows"]["e1v"]["differing_cells"] = 1
    elif plant == "missing-stream":
        report["record_fields"]["j003_va_ext"] = False

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("record_census") == {
        "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
    }, "record census moved")
    require(report.get("operand_order") == ["e1v", "va_e", "zhvp2_e"],
            "V metric-transport operand registry reordered")
    require(report.get("record_fields") == {
        "j003_va_ext": True,
        "j003_hvp2_e": True,
        "j003_zhV": True,
    }, "substep-3 operand stream is absent")
    require(all(report["passivity"].values()),
            "substep trace is not passive")
    require(report.get("prerequisite_transport_exact") == [True, True, False],
            "round-194 substep-1/2/3 prerequisite moved")

    split = report["split"]
    operand_rows = split["operand_rows"]
    require(operand_rows["e1v"]["bit_exact"],
            "static e1v is the first non-bit operand")
    require(not split["rows"]["unmasked_transport_v"]["bit_exact"],
            "substep-3 completed zhV unexpectedly became bit-exact")

    ordered_boundaries = (
        ("va_e", operand_rows["va_e"]),
        ("first_product", split["rows"]["first_product"]),
        ("zhvp2_e", operand_rows["zhvp2_e"]),
        ("completed_zhV", split["rows"]["unmasked_transport_v"]),
    )
    first_nonbit = next(
        (name for name, row in ordered_boundaries if not row["bit_exact"]),
        None,
    )
    require(first_nonbit is not None,
            "source-ordered split did not reproduce the transport debt")
    report["first_nonbit_boundary"] = first_nonbit

    single = split["single_substitution_transport_v"]
    closing_single = next(
        (name for name in report["operand_order"] if single[name]["bit_exact"]),
        None,
    )
    cumulative = split["cumulative_substitution_transport_v"]
    closing_cumulative = next(
        (name for name in report["operand_order"]
         if cumulative[name]["bit_exact"]),
        None,
    )
    report["closing_single_substitution"] = closing_single
    report["closing_cumulative_boundary"] = closing_cumulative
    report["prediction_ledger"] = {
        "R195-P1": "CONFIRMED_RECORD_SUFFICIENT",
        "R195-P2": "CONFIRMED_E1V_EXACT",
        "R195-P3": (
            "CONFIRMED_VA_E_FIRST" if first_nonbit == "va_e"
            else f"REFUTED_FIRST_{first_nonbit.upper()}"),
        "R195-P4": (
            f"CONFIRMED_SINGLE_{closing_single.upper()}"
            if closing_single else "REFUTED_NO_SINGLE_SUBSTITUTION_CLOSES"),
        "R195-P5": "CONFIRMED_CONTROLS_BIND",
    }
    report["status"] = "PASS_R195_SUBSTEP3_OPERAND_SPLIT"
    return report


def measurement_context(deck_root: Path, frame_root: Path, spg_root: Path,
                        expect_commit: str) -> dict[str, object]:
    """Return the admitted passive trace shared by later offline splits."""

    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-195 operand gate requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "operand split requires production JIT on CPU")

    oracle, census = r97.assemble_record(spg_root)
    require(len(census["records"]) == 2
            and all(row["icycle"] == 65 for row in census["records"]),
            "round-96 record is not rank-complete at 65 substeps")
    record_fields = {
        name: name in oracle
        for name in ("j003_va_ext", "j003_hvp2_e", "j003_zhV")
    }
    require(all(record_fields.values()),
            "round-96 record lacks a substep-3 operand stream")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    frame = rung0.assemble_frame(frame_root, 1, 0)
    candidate = rung0.candidate_fields(state)
    for name in ("T", "S", "u", "v", "ssh"):
        require(np.asarray(candidate[name]).shape == np.asarray(frame[name]).shape,
                f"independent entry {name} shape moved")

    forcing = (
        np.asarray(oracle["i000_ssh_frc"]),
        r178._to_model_u(oracle["i000_zu_frc"]),
        r178._to_model_v(oracle["i000_zv_frc"]),
    )
    dt = float(oracle["i000_entry_sc"][0])
    count = int(oracle["i000_entry_sc"][2])
    reference_depth = rung0.ladder.build_reference_depth_override(card)

    def solve(trace: bool):
        return jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, dt, count, card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
                _nemo_reference_face_depth_test_override=reference_depth,
                _nemo_unmasked_v_transport_test_override=True,
                _nemo_materialize_v_transport_test_override=True,
                _nemo_external_mode_association_test_override=True,
            )
        ))(state, *forcing)

    live_state, live_transport = jax.device_get(solve(False))
    traced_state, traced_transport, trace = jax.device_get(solve(True))
    passivity = {
        "ssh": bool(np.array_equal(live_state.eta.data, traced_state.eta.data)),
        "u": bool(np.array_equal(live_state.u.data, traced_state.u.data)),
        "v": bool(np.array_equal(live_state.v.data, traced_state.v.data)),
        "u_barotropic": bool(np.array_equal(
            live_state.uu_b.data, traced_state.uu_b.data)),
        "v_barotropic": bool(np.array_equal(
            live_state.vv_b.data, traced_state.vv_b.data)),
        "transport_u": bool(np.array_equal(live_transport[0], traced_transport[0])),
        "transport_v": bool(np.array_equal(live_transport[1], traced_transport[1])),
    }

    return {
        "stamp": stamp,
        "oracle": oracle,
        "census": census,
        "record_fields": record_fields,
        "card": card,
        "state": state,
        "trace": trace,
        "passivity": passivity,
    }


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    """Run the admitted passive trace and split only substep 3 offline."""

    context = measurement_context(
        deck_root, frame_root, spg_root, expect_commit)
    stamp = context["stamp"]
    oracle = context["oracle"]
    record_fields = context["record_fields"]
    card = context["card"]
    state = context["state"]
    trace = context["trace"]
    passivity = context["passivity"]
    prerequisite = [
        r146.exact_row(
            r178._native_v(trace["transport_metric_v"][index]),
            oracle[f"j{index + 1:03d}_zhV"])["bit_exact"]
        for index in range(SUBSTEP)
    ]
    split = r146.transport_v_operand_split(
        card, state, trace, oracle, plant="none", substep=SUBSTEP)
    return classify({
        "format": "nemo-testcase-l4-orca2-round195-transport-operands-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": record_fields,
        "operand_order": list(r146.transport_v_operand_names()),
        "passivity": passivity,
        "prerequisite_transport_exact": prerequisite,
        "split": split,
        "compiled_sources": [
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:502-511",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:535-546",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:564-570",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-619",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit)), "runtime inputs are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r146.GateError, rung0.GateError, r97.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R195_SUBSTEP3_OPERAND_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
