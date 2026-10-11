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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
)

PLANTS = ("none", "arm-effect", "rhs-closure", "off-support", "label-coverage")


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
    compact = np.asarray(legacy["mask_v"])
    raw = r236.r229._raw_vmask(card)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rhs", "classify"), default="classify")
    parser.add_argument("--omt4-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "rhs":
            require(args.omt4_root is not None and args.expect_commit,
                    "rhs measurement arguments incomplete")
            result = measure_rhs(args)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify_rhs(json.loads(args.report_in.read_text()), plant=args.plant)
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
