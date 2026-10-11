#!/usr/bin/env python3
"""Measure and classify round 241's literal-HPG compact-mask arm."""

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
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
)

PLANTS = ("none", "arm-effect", "rhs-closure", "off-support", "label-coverage")
ATOMIC_PLANTS = ("none", "salinity", "temperature", "completion")


class GateError(RuntimeError):
    """The frozen compact-mask attribution no longer holds."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits_equal(left, right) -> bool:
    left = np.ascontiguousarray(np.asarray(left, np.float64)).view(np.uint64)
    right = np.ascontiguousarray(np.asarray(right, np.float64)).view(np.uint64)
    return bool(np.array_equal(left, right))


def classify_rhs(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "arm-effect":
        report["rows"]["independent"]["changed_slots"] = 0
    elif plant == "rhs-closure":
        report["rows"]["independent"]["arm_unequal"] = 1
    elif plant == "off-support":
        report["rows"]["independent"]["off_support_changed"] = 1
    elif plant == "label-coverage":
        report["rows"].pop("given_nemo_entry")

    rows = report["rows"]
    require(set(rows) == {"independent", "given_nemo_entry"},
            "both claim labels are required")
    for label, row in rows.items():
        require(row["baseline_unequal"] == 668,
                f"{label}: frozen completed-RHS baseline moved")
        require(row["arm_unequal"] == 0,
                f"{label}: literal-HPG arm did not close completed V RHS")
        require(row["changed_slots"] > 0,
                f"{label}: arm was vacuous")
        require(row["off_support_changed"] == 0,
                f"{label}: arm changed a slot outside the raw-vs-compact mask support")
        require(row["arm_passivity"], f"{label}: trace observer moved the arm state")
    require(rows["independent"]["changed_slots"]
            == rows["given_nemo_entry"]["changed_slots"],
            "claim labels disagree on the arm support")
    report["status"] = "PASS_R241_LITERAL_HPG_RHS_CLOSED"
    report["predictions"] = {
        "R241-P1": "CONFIRMED_COMPLETED_V_RHS_668_TO_0_BOTH_LABELS",
        "R241-P2": "PENDING_ATOMIC_TRAJECTORY",
        "R241-P3": "PENDING_FULL_GATES",
        "R241-P4": "CONFIRMED" if plant == "none" else "PLANT",
        "R241-P5": "PENDING_DOCUMENTARY_GATES",
    }
    return report


def _find_row(ladder: dict[str, object], kt: int, checkpoint: str,
              field: str) -> dict[str, object]:
    rows = [row for row in ladder["rows"] if (
        row["kt"], row["checkpoint"], row["field"])
        == (kt, checkpoint, field)]
    require(len(rows) == 1, f"missing unique {kt}/{checkpoint}/{field} row")
    return rows[0]


def classify_atomic(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in ATOMIC_PLANTS, f"unknown atomic plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "salinity":
        report["ladders"]["independent"]["kt1_stage1"]["S_max_abs"] = 1.0
    elif plant == "temperature":
        report["ladders"]["independent"]["kt1_stage1"]["T_max_abs"] = 1.0
    elif plant == "completion":
        report["ladders"]["independent"]["last_completed_kt"] = 7

    ladders = report["ladders"]
    require(set(ladders) == {"independent", "given_nemo_entry"},
            "atomic run requires both claim labels")
    for label, ladder in ladders.items():
        require(ladder["last_completed_kt"] >= 8,
                f"{label}: atomic unit did not complete kt=8")
        require(ladder["kt1_stage1"]["S_max_abs"] <= 0.0012,
                f"{label}: kt1 stage1 salinity did not fall below preregistered limit")
        require(ladder["kt1_stage1"]["T_max_abs"] <= 0.0014,
                f"{label}: kt1 stage1 temperature did not fall below preregistered limit")
    report["status"] = "PASS_R241_ATOMIC_UNIT_THROUGH_KT8"
    report["predictions"] = {
        "R241-P1": "CONFIRMED_IN_SEPARATE_RHS_REPORT",
        "R241-P2": "CONFIRMED_ATOMIC_UNIT_COMPLETES_KT8",
        "R241-P3": "PENDING_FULL_DECISION96_GATES",
        "R241-P4": "CONFIRMED" if plant == "none" else "PLANT",
        "R241-P5": "PENDING_DOCUMENTARY_GATES",
    }
    return report


def _policy_and_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-241 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-241 measurement requires production JIT on CPU")
    return stamp


def _rhs_row(card, state, freshwater, surface, oracle_rhs) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    common = dict(expose_live_stage_operands=True, barotropic_atomic_fold_unit=True)
    base_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**common))
    arm_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            **common, preserve_literal_hpg_vmask=True))
    arm_plain_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            barotropic_atomic_fold_unit=True,
            preserve_literal_hpg_vmask=True))
    baseline = jax.device_get(base_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    arm = jax.device_get(arm_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    arm_plain = jax.device_get(arm_plain_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(baseline, _NEMOWSLiveOperandTrace)
            and isinstance(arm, _NEMOWSLiveOperandTrace), "trace type moved")

    baseline_rhs = r236._native_v(baseline.slow_forcing_producer["rhs_v"])
    arm_rhs = r236._native_v(arm.slow_forcing_producer["rhs_v"])
    oracle_rhs = np.asarray(oracle_rhs, np.float64)
    require(baseline_rhs.shape == arm_rhs.shape == oracle_rhs.shape,
            "completed V-RHS shapes moved")

    legacy = r170._candidate_reference_operands(card, state)
    compact_global = np.asarray(legacy["mask_v"])
    raw_live = np.asarray(card.recipe.z_coord.nemo_een_barotropic.vmask)
    require(raw_live.shape == (
        compact_global.shape[0], compact_global.shape[1],
        compact_global.shape[2] - 1),
            "raw/native V-mask shape relation moved")
    compact = compact_global[:, :baseline_rhs.shape[1], :baseline_rhs.shape[2]]
    raw = raw_live[:, :baseline_rhs.shape[1], :baseline_rhs.shape[2]]
    require(raw.shape == compact.shape == baseline_rhs.shape,
            "rank-0 native mask mapping moved")
    support = (raw != 0.0) & (compact == 0.0)
    changed = arm_rhs != baseline_rhs
    off_support = changed & ~support
    baseline_delta = np.abs(baseline_rhs - oracle_rhs)
    arm_delta = np.abs(arm_rhs - oracle_rhs)
    return {
        "support_slots": int(np.count_nonzero(support)),
        "changed_slots": int(np.count_nonzero(changed)),
        "off_support_changed": int(np.count_nonzero(off_support)),
        "baseline_unequal": int(np.count_nonzero(baseline_rhs != oracle_rhs)),
        "baseline_max_abs": float(np.max(baseline_delta, initial=0.0)),
        "arm_unequal": int(np.count_nonzero(arm_rhs != oracle_rhs)),
        "arm_max_abs": float(np.max(arm_delta, initial=0.0)),
        "arm_passivity": all(passive._ordinary_state_equal(
            arm.state_after, arm_plain).values()),
        "arm_changes_bits": not _bits_equal(arm_rhs, baseline_rhs),
        "dtype": str(np.asarray(arm_rhs).dtype),
    }


def measure_rhs(args: argparse.Namespace) -> dict[str, object]:
    stamp = _policy_and_stamp(args.expect_commit)
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
    rows = {
        label: _rhs_row(card, state, freshwater, surface, oracle_slow["krhs_v"])
        for label, state in states.items()
    }
    return classify_rhs({
        "format": "nemo-testcase-l4-orca2-round241-rhs-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "geometry_route": "nemo_qco_resolved_mesh_operands",
        "worktree": stamp,
        "rows": rows,
        "compiled_source": {
            "hpg_write": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427",
            "depth_average": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215",
            "candidate_boundary": "ocean_pe_latlon_cgrid.py:5982-5996",
        },
    })


def _run_atomic_ladder(card, record_root: Path, state, label: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    reference_depth = rung0.ladder.build_reference_depth_override(card)

    def hooks(stage: int = 0):
        return _NEMOWSRK3TestHooks(
            expose_momentum_stage=stage,
            expose_tracer_stage=stage,
            barotropic_external_mode_association=True,
            barotropic_reference_face_depth_override=reference_depth,
            barotropic_unmasked_v_transport=True,
            barotropic_materialize_v_transport=True,
            barotropic_atomic_fold_unit=True,
            preserve_literal_hpg_vmask=True,
        )

    stage_models = tuple(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks(stage)) for stage in (1, 2))
    final_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks())
    checkpoints = []
    for kt in range(1, 11):
        entry = rung0.assemble_frame(record_root, kt, 0)
        checkpoints.append(omt1._checkpoint(
            kt, "entry", rung0.candidate_fields(state), entry))
        stage_states = tuple(jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
            for model in stage_models)
        state_after = jax.device_get(final_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        for stage, stage_state in enumerate((*stage_states, state_after), start=1):
            checkpoints.append(omt1._checkpoint(
                kt, f"stage{stage}", rung0.candidate_fields(stage_state),
                rung0.assemble_frame(record_root, kt, stage)))
        state = state_after
        print(f"PROGRESS round241 {label} kt={kt}", file=sys.stderr, flush=True)
    rows = [
        {"kt": checkpoint["kt"], "checkpoint": checkpoint["checkpoint"],
         "field": field, **checkpoint["rows"][field]}
        for checkpoint in checkpoints for field in omt1.FIELDS
    ]
    return {
        "label": label,
        "last_completed_kt": 10,
        "checkpoint_count": len(checkpoints),
        "row_count": len(rows),
        "rows": rows,
    }


def measure_atomic(args: argparse.Namespace) -> dict[str, object]:
    stamp = _policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.omt4_root)
    omt4.validate_omt4_card(args.omt4_root, card)
    entry = rung0.assemble_frame(args.omt4_root, 1, 0)
    states = {
        "independent": card.recipe.initial_state,
        "given_nemo_entry": rung0.bridge_entry(card, entry),
    }
    raw_ladders = {
        label: _run_atomic_ladder(card, args.omt4_root, state, label)
        for label, state in states.items()
    }
    ladders = {}
    for label, ladder in raw_ladders.items():
        t_row = _find_row(ladder, 1, "stage1", "T")
        s_row = _find_row(ladder, 1, "stage1", "S")
        ladders[label] = {
            **ladder,
            "kt1_stage1": {
                "T_max_abs": t_row["max_abs"],
                "S_max_abs": s_row["max_abs"],
            },
        }
    return classify_atomic({
        "format": "nemo-testcase-l4-orca2-round241-atomic-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "ladders": ladders,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("rhs", "atomic", "classify", "classify-atomic"),
        default="classify")
    parser.add_argument("--omt4-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode in ("rhs", "atomic"):
            require(args.omt4_root is not None and args.expect_commit,
                    "measurement arguments incomplete")
            result = measure_rhs(args) if args.mode == "rhs" else measure_atomic(args)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            payload = json.loads(args.report_in.read_text())
            result = (classify_rhs(payload, plant=args.plant)
                      if args.mode == "classify"
                      else classify_atomic(payload, plant=args.plant))
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            rung0.GateError, r236.GateError) as error:
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
