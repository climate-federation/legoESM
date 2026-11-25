#!/usr/bin/env python3
"""Split OMT-4's stage-1 vector-V update at matched program points."""

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
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
    nemo_testcase_l4_orca2_round176_stage1_offline_replay as r176,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)

FLOOR = np.float64(2.0e-10)
OPERAND_ORDER = ("entry_v", "rhs_v", "rDt", "vmask", "written_order")
PLANTS = (
    "none", "passivity", "self-replay", "source-order", "owner-prefix",
    "one-bit",
)


class GateError(RuntimeError):
    """A frozen stage-update predicate or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    require(values.shape[:2] == (149, 180),
            f"redundant V shape moved: {values.shape}")
    return values[1:]


def _score(left, right, support=None) -> dict[str, object]:
    return r281._exact_score(
        np.asarray(left, np.float64), np.asarray(right, np.float64), support)


def _update(entry, rhs, dt_stage, mask):
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        rk3_stage_velocity_update,
    )

    return np.asarray(jax.device_get(rk3_stage_velocity_update(
        jnp.asarray(entry), jnp.asarray(rhs), np.float64(dt_stage),
        jnp.asarray(mask), vector_form=True)), np.float64)


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-283 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 283 requires production JIT on CPU")

    admission = r281.admission_gate.run(record_root, baseline_root)
    oracle = r281._assemble_values(record_root)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    require(not card.recipe.model_config.rk3_fold_exchange_top_v_row,
            "Decision 125 top-V selector is unexpectedly enabled")
    state = card.recipe.initial_state
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))

    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False, live=True))
    ordinary_after = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSLiveOperandTrace),
            "live operand trace type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)

    entry = _native_v(state.v.data)
    live_rhs = _native_v(observed.stage_rhs[0][1])
    live_raw = _native_v(observed.stage_raw_velocities[0][1])
    require(entry.shape == live_rhs.shape == live_raw.shape == (148, 180, 30),
            "candidate stage-1 operand shape moved")

    candidate_ops = r170._candidate_reference_operands(card, state)
    live_mask = np.asarray(candidate_ops["mask_v"][..., :-1], np.float64)
    require(live_mask.shape == live_raw.shape, "candidate stage mask shape moved")

    raw_mesh = card.recipe.z_coord.nemo_een_barotropic
    require(raw_mesh is not None, "OMT-4 lacks raw NEMO mesh operands")
    # The resolved raw mask and stage-entry V carry NEMO's jpkm1 levels
    # already; the self-describing Krhs/Kaa record carries jpk and needs its
    # final land level removed.
    nemo_mask = np.asarray(raw_mesh.vmask, np.float64)
    nemo_entry = np.asarray(rung0.assemble_frame(record_root, 1, 0)["v"],
                            np.float64)
    nemo_rhs = np.asarray(oracle["rhs_v"], np.float64)[..., :-1]
    nemo_raw = np.asarray(oracle["pre_v"], np.float64)[..., :-1]
    active = nemo_mask != 0.0
    oracle_shapes = {
        "live_raw": live_raw.shape,
        "nemo_mask": nemo_mask.shape,
        "nemo_entry": nemo_entry.shape,
        "nemo_rhs": nemo_rhs.shape,
        "nemo_raw": nemo_raw.shape,
        "active": active.shape,
    }
    require(all(value.shape == live_raw.shape for value in (
        nemo_mask, nemo_entry, nemo_rhs, nemo_raw, active)),
        f"oracle stage-1 operand shape moved: {oracle_shapes}")

    candidate_dt = np.float64(card.dt_s) / np.float64(3.0)
    nemo_dt = np.float64(10800.0) / np.float64(3.0)
    self_replay = _update(entry, live_rhs, candidate_dt, live_mask)

    prefixes = {
        "candidate": self_replay,
        "entry_v": _update(nemo_entry, live_rhs, candidate_dt, live_mask),
        "rhs_v": _update(nemo_entry, nemo_rhs, candidate_dt, live_mask),
        "rDt": _update(nemo_entry, nemo_rhs, nemo_dt, live_mask),
        "vmask": _update(nemo_entry, nemo_rhs, nemo_dt, nemo_mask),
    }
    prefix_rows = {
        name: _score(value, nemo_raw, active)
        for name, value in prefixes.items()
    }
    operand_rows = {
        "entry_v": _score(entry, nemo_entry, active),
        "rhs_v": _score(live_rhs, nemo_rhs, active),
        "rDt": _score(np.asarray([candidate_dt]), np.asarray([nemo_dt])),
        "vmask": _score(live_mask, nemo_mask),
        "written_order": prefix_rows["vmask"],
    }
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round283-stage1-v-split-v1",
        "status": "PASS_R283_MEASUREMENT",
        "claim_label": "independent_omt4",
        "worktree": stamp,
        "admission": admission,
        "execution": "production-jit-cpu-fp64-libm; offline pure update replay",
        "source_order": list(OPERAND_ORDER),
        "compiled_source": {
            "rDt": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:136",
            "update": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480",
        },
        "passivity": passivity,
        "self_replay": _score(self_replay, live_raw),
        "candidate_raw_vs_nemo": _score(live_raw, nemo_raw, active),
        "operand_rows": operand_rows,
        "prefix_rows": prefix_rows,
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
        "dt_seconds": {"candidate": float(candidate_dt), "nemo": float(nemo_dt)},
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "self-replay":
        report["self_replay"]["unequal"] = 1
    elif plant == "source-order":
        report["source_order"][1:3] = reversed(report["source_order"][1:3])
    elif plant == "owner-prefix":
        planted_owner = next(
            name for name in OPERAND_ORDER
            if report["operand_rows"][name]["max_abs"] > FLOOR)
        report["prefix_rows"][planted_owner] = report["prefix_rows"]["candidate"]
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R283_MEASUREMENT", "measurement status moved")
    require(report["claim_label"] == "independent_omt4", "claim label moved")
    require(report["admission"]["status"] == "PASS_R280_OMT4_STAGE1_ADMISSION",
            "record admission moved")
    require(tuple(report["source_order"]) == OPERAND_ORDER, "source order moved")
    require(all(report["passivity"].values()), "live trace moved completed state")
    require(report["self_replay"]["unequal"] == 0,
            "candidate operand replay did not reproduce raw stage V")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    rows = report["operand_rows"]
    entry_exact = rows["entry_v"]["unequal"] == 0
    first = next((name for name in OPERAND_ORDER
                  if rows[name]["max_abs"] > FLOOR), None)
    require(first is not None, "no valued stage-update operand boundary")
    prefix = report["prefix_rows"]
    rhs_toward = prefix["rhs_v"]["rms"] < prefix["candidate"]["rms"]
    require(first in prefix and prefix[first] != prefix["candidate"],
            f"{first} prefix arm is vacuous")
    report["first_valued_operand"] = first
    report["prediction_ledger"] = {
        "R283-P1": "CONFIRMED",
        "R283-P2": "CONFIRMED" if entry_exact else "REFUTED",
        "R283-P3": "CONFIRMED" if first == "rhs_v" else "REFUTED",
        "R283-P4": "CONFIRMED" if rhs_toward else "REFUTED",
        "R283-P5": "CONFIRMED",
    }
    report["first_nonbit_statement"] = {
        "operand": first,
        "nemo_source": report["compiled_source"][
            "rDt" if first == "rDt" else "update"],
        "status": "HELD_OPERAND_SPLIT",
    }
    report["status"] = "HELD_FIRST_VALUED_STAGE1_V_OPERAND"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.record_root, args.baseline_root,
                             args.expect_commit)
        else:
            require(args.report_in is not None, "classification report missing")
            result = classify(json.loads(args.report_in.read_text(encoding="utf-8")),
                              plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, r281.GateError, r281.admission_gate.Refusal,
            r281.admission_gate.schema.Refusal, omt4.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
