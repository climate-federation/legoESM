#!/usr/bin/env python3
"""Replay kt=1's external mode offline under the complete private unit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

FLOOR = np.float64(2.0e-10)
GROWTH_SHA256 = "a25b92b470ba0c00cd4d49778204c9c38ea832aa3a93df857f88ca97087a2a29"
PLANTS = (
    "none", "growth-pin", "record-bit", "passivity", "source-order",
    "first-selector", "arm-identity",
)
COMPILED_SOURCE = {
    "ssh_forcing": "stp2d.f90:203-219",
    "u_forcing": "stp2d.f90:203-219",
    "v_forcing": "stp2d.f90:203-219",
    "u_entry": "dynspg_ts.f90:447-469",
    "v_entry": "dynspg_ts.f90:447-469",
    "ssh_entry": "dynspg_ts.f90:447-469",
    "u_history_b": "dynspg_ts.f90:447-469",
    "u_history_bb": "dynspg_ts.f90:447-469",
    "v_history_b": "dynspg_ts.f90:447-469",
    "v_history_bb": "dynspg_ts.f90:447-469",
    "ssh_history_b": "dynspg_ts.f90:447-469",
    "ssh_history_bb": "dynspg_ts.f90:447-469",
    "u_mid": "dynspg_ts.f90:502-508",
    "v_mid": "dynspg_ts.f90:509-511",
    "ssh_mid": "dynspg_ts.f90:513-519",
    "depth_u_mid": "dynspg_ts.f90:535-540",
    "depth_v_mid": "dynspg_ts.f90:542-545",
    "transport_u": "dynspg_ts.f90:564-567",
    "transport_v": "dynspg_ts.f90:568-570",
    "ssh_after": "dynspg_ts.f90:580-591",
    "transport_sum_u": "dynspg_ts.f90:600-608",
    "transport_sum_v": "dynspg_ts.f90:600-608",
    "face_ssh_u": "dynspg_ts.f90:630-637",
    "face_ssh_v": "dynspg_ts.f90:630-637",
    "ssh_back": "dynspg_ts.f90:642-650",
    "pgf_u": "dynspg_ts.f90:652-656",
    "pgf_v": "dynspg_ts.f90:652-656",
    "coriolis_u": "dynspg_ts.f90:663-666",
    "coriolis_v": "dynspg_ts.f90:663-666",
    "trend_u": "dynspg_ts.f90:680-702",
    "trend_v": "dynspg_ts.f90:680-702",
    "u_exit": "dynspg_ts.f90:704-728",
    "v_exit": "dynspg_ts.f90:704-728",
    "depth_u_exit": "dynspg_ts.f90:761-764",
    "depth_v_exit": "dynspg_ts.f90:765-766",
    "inverse_u_exit": "dynspg_ts.f90:761-766",
    "inverse_v_exit": "dynspg_ts.f90:761-766",
    "transport_mean_u": "dynspg_ts.f90:600-608",
    "transport_mean_v": "dynspg_ts.f90:600-608",
    "external_mean_u": "dynspg_ts.f90:800-842",
    "external_mean_v": "dynspg_ts.f90:800-842",
    "external_mean_ssh": "dynspg_ts.f90:800-842",
}


class GateError(RuntimeError):
    """The offline replay or one of its frozen prerequisites moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_id(row: dict[str, object]) -> str:
    if "substep" in row:
        return f"{int(row['substep']):03d}:{row['name']}"
    return str(row["name"])


def _expected_order() -> list[str]:
    order = list(r178.ENTRY_ORDER)
    for substep in range(1, 66):
        order.extend(f"{substep:03d}:{name}" for name in r178.SUBSTEP_ORDER)
    order.extend(r178.EXIT_ORDER)
    return order


def _first_over_floor(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next((copy.deepcopy(row) for row in rows if not row["at_floor"]), None)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "growth-pin":
        report["growth_report"]["sha256"] = "0" * 64
    elif plant == "record-bit":
        report["record_control"]["differing_cells"] = 0
    elif plant == "passivity":
        report["observer_passivity"]["ssh"] = False
    elif plant == "source-order":
        report["source_rows"][0], report["source_rows"][1] = (
            report["source_rows"][1], report["source_rows"][0])
    elif plant == "first-selector":
        report["first_over_floor"]["name"] = "planted"
    elif plant == "arm-identity":
        report["private_arm"]["raw_reference_depth"] = False

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report["growth_report"] == {
        "sha256": GROWTH_SHA256, "first_kt": 1, "first_stage": 1,
    }, "growth-boundary pin moved")
    require(report["record_census"]["coverage"] == "exactly-once",
            "rank placement moved")
    require(report["record_control"] == {
        "bit_exact": False, "differing_cells": 1,
    }, "record one-bit control did not fire")
    require(all(report["observer_passivity"].values()),
            "offline trace changes the solver")
    require(report["private_arm"] == {
        "slow_forcing": "recorded_source_exact",
        "external_mode_association": True,
        "raw_reference_depth": True,
        "unmasked_v_transport": True,
        "materialize_v_transport": True,
    }, "complete private arm moved")
    require(all(row["bit_exact"] for row in report["independent_entry"].values()),
            "independent entry moved")
    rows = report["source_rows"]
    require(isinstance(rows, list) and len(rows) >= 2,
            "offline source rows are incomplete")
    order = [_row_id(row) for row in rows]
    require(order == _expected_order()[:len(order)],
            "offline source order moved")
    require(all(row["at_floor"] for row in rows[:-1]),
            "replay crossed an earlier debt")
    first = _first_over_floor(rows)
    require(first == report["first_over_floor"],
            "first-debt selector moved")
    require(first is not None and first == rows[-1],
            "replay did not stop at a first debt")
    source = COMPILED_SOURCE.get(str(first["name"]))
    require(source is not None, "first debt has no compiled-source citation")
    split = report["accumulation_split"]
    require(first["name"] == "transport_sum_v" and first["substep"] == 2,
            "frozen accumulator split reached a different boundary")
    require(split["previous_sum"]["comparison_bit_exact"],
            "substep-2 incoming V accumulator moved")
    require(split["completed_transport"]["comparison_bit_exact"],
            "substep-2 completed V transport moved")
    require(split["weight"]["comparison_bit_exact"],
            "substep-2 transport weight moved")
    require(split["candidate"]["full_domain_differing_cells"] == 68,
            "candidate V accumulator census moved")
    require(split["unmasked_reciprocal_replay"]["comparison_bit_exact"],
            "unmasked reciprocal replay did not close the V accumulator")
    report["first_statement"] = {
        "name": first["name"], "substep": first.get("substep"),
        "compiled_source": (
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/" + source),
    }
    report["prediction_ledger"] = {
        "R193-P1": "CONFIRMED",
        "R193-P2": "CONFIRMED",
        "R193-P3": "CONFIRMED",
        "R193-P4": "REFUTED_EXTERNAL_MODE_OWNS_FIRST_DEBT",
        "R193-P5": "CONFIRMED_ROUND96_RECORD_SUFFICIENT",
        "R193-P6": "CONFIRMED_MEASUREMENT_ONLY",
        "R193-P7": "CONFIRMED_UNMASKED_RECIPROCAL_CLOSES",
    }
    report["status"] = "HELD_R193_FIRST_EXTERNAL_DEBT"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            growth_report: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-193 replay requires its clean committed candidate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "external replay requires production JIT on CPU")
    require(growth_report.is_file() and sha256(growth_report) == GROWTH_SHA256,
            "round-193 growth report content moved")
    growth = json.loads(growth_report.read_text(encoding="utf-8"))
    require((growth["first_growth"]["kt"], growth["first_growth"]["stage"])
            == (1, 1), "growth report no longer selects kt=1 stage 1")

    oracle, census = r97.assemble_record(spg_root)
    require(census["records"][0]["icycle"] == 65, "substep count moved")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    frame = rung0.assemble_frame(frame_root, 1, 0)
    candidate = rung0.candidate_fields(state)
    entry_masks = {
        "T": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "S": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "u": np.asarray(masks["u"], dtype=bool),
        "v": np.asarray(masks["v"], dtype=bool),
        "ssh": active["t"],
    }
    entry = {
        name: r178._score(candidate[name], frame[name], entry_masks[name])
        for name in ("T", "S", "u", "v", "ssh")
    }
    require(all(row["bit_exact"] for row in entry.values()),
            "independent entry is not active-domain exact")

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
    passive = {
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
    require(all(passive.values()), "offline trace changes the solver")
    observed = SimpleNamespace(
        slow_forcing=forcing, substeps=trace,
        state_after_barotropic=traced_state,
        transport_average=traced_transport,
    )
    rows = r178._source_rows(observed, oracle, active)
    first = _first_over_floor(rows)
    b = nemo_source_round
    previous_sum = r178._native_v(trace["transport_sum_v_exit"][0])
    completed_transport = r178._native_v(trace["transport_metric_v"][1])
    weight = np.asarray(trace["transport_weight"][1], dtype=np.float64)
    e1v = r178._native_v(np.asarray(card.recipe.grid.dx_v, dtype=np.float64))
    reciprocal = np.divide(
        np.float64(1.0), e1v,
        out=np.zeros_like(e1v), where=e1v != np.float64(0.0))
    unmasked_replay = jax.device_get(b(
        b(previous_sum) + b(
            b(weight * b(completed_transport)) * b(reciprocal))))
    weight_oracle = np.asarray(
        oracle["j002_sum_coef"], dtype=np.float64).reshape(-1)[1:2]
    weight_candidate = np.asarray(weight, dtype=np.float64).reshape(1)
    accumulation_split = {
        "previous_sum": r178._score(
            previous_sum, oracle["j001_vn_adv"], active["v"],
            complete_domain=True),
        "completed_transport": r178._score(
            completed_transport, oracle["j002_zhV"], active["v"],
            complete_domain=True),
        "weight": r178._score(
            weight_candidate, weight_oracle, np.ones((1,), dtype=bool),
            complete_domain=True),
        "candidate": r178._score(
            r178._native_v(trace["transport_sum_v_exit"][1]),
            oracle["j002_vn_adv"], active["v"], complete_domain=True),
        "unmasked_reciprocal_replay": r178._score(
            unmasked_replay, oracle["j002_vn_adv"], active["v"],
            complete_domain=True),
    }
    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    record_control = rhs_walk.score(next_one, one, np.ones_like(one, dtype=bool))
    return classify({
        "format": "nemo-testcase-l4-orca2-round193-external-replay-v1",
        "claim_label": "independent hierarchy rung 0",
        "growth_report": {"sha256": sha256(growth_report),
                          "first_kt": 1, "first_stage": 1},
        "record_census": {"coverage": "exactly-once",
                          "records": len(census["records"])},
        "record_control": {
            "bit_exact": record_control["bit_exact"],
            "differing_cells": record_control["differing_cells"],
        },
        "independent_entry": entry,
        "observer_passivity": passive,
        "private_arm": {
            "slow_forcing": "recorded_source_exact",
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "source_rows": rows,
        "source_order": [_row_id(row) for row in rows],
        "first_over_floor": first,
        "accumulation_split": accumulation_split,
        "worktree": stamp,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--growth-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.growth_report, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.growth_report, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(
                json.loads(args.report_in.read_text(encoding="utf-8")), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r97.GateError, r178.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS HELD_R193_FIRST_EXTERNAL_DEBT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
