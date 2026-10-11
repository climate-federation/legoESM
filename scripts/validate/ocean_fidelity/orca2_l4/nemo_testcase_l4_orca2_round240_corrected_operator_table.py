#!/usr/bin/env python3
"""Rebuild the fold-row operator table through D114 production geometry."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round93_rhs_walk as r93,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round172_passive_rhs_replay as r172,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as r179,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round181_hpg_component_walk as r181,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_round84_rhs_walk as r84,
)

ORDER = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")
PLANTS = ("none", "source-order", "hpg", "first-owner", "calibration", "route")


class GateError(RuntimeError):
    """The corrected production-geometry operator table does not bind."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first(rows: dict[str, dict[str, object]]) -> dict[str, object] | None:
    for name in ORDER:
        if not rows[name]["at_floor"]:
            return {"boundary": name, **rows[name]}
    return None


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "source-order":
        report["source_order"][0:2] = reversed(report["source_order"][0:2])
    elif plant == "hpg":
        report["operator_rows"]["after_hpg"]["at_floor"] = False
    elif plant == "first-owner":
        report["first_operator"] = {"boundary": "after_hpg"}
    elif plant == "calibration":
        report["candidate_calibration"]["all_faces_at_floor"] = False
    elif plant == "route":
        report["geometry_route"] = "nemo_qco_card_mesh_operands"

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(tuple(report["source_order"]) == ORDER, "source order moved")
    require(report["geometry_route"] == "nemo_qco_resolved_mesh_operands",
            "operator table bypasses D114 production geometry")
    require(report["target"] == {"cells": 68, "row": 147},
            "fold-row target moved")
    require(report["operator_rows"]["after_hpg"]["bit_exact"],
            "direct round-240 HPG exoneration did not carry into the table")
    require(report["candidate_calibration"]["all_faces_at_floor"],
            "offline after-ZAD endpoint does not reproduce the live endpoint")
    require(report["candidate_calibration"]["same_over_floor_set"],
            "offline and live endpoint debt sets differ")
    require(all(report["trace_passivity"].values()), "passive trace moved state")
    first = _first(report["operator_rows"])
    require(first == report["first_operator"], "first operator selector moved")
    require(first is not None and first["boundary"] != "after_hpg",
            "corrected table did not move ownership downstream of exact HPG")
    report["status"] = f"HELD_R240_FIRST_OPERATOR_{first['boundary'].upper()}"
    return report


def measure(deck_root: Path, rhs_root: Path, static_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_face_masks_3d
    from legoesm.ocean.vertical import compute_layer_thickness

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-240 corrected table requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "corrected table requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    freshwater, surface = r93._forcing(state.eta.data.shape)
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    traced = jax.device_get(traced_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    passivity = r172._state_rows(traced.state_after, ordinary)

    offline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    offline.prime_step_caches(state)
    _total, _diagnostics, parts = jax.device_get(jax.jit(
        lambda source_state, forcing: r172._stage1_tendency(
            offline, source_state, forcing, card.dt_s, components=True)
    )(state, surface))
    accumulated = jax.device_get(jax.jit(r84.source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))

    oracle_rhs, _ = r93.assemble_rhs(rhs_root)
    oracle_static, _ = r170.assemble_record(static_root)
    oracle_e3v = np.asarray(oracle_static["e3v"])
    oracle_vmask = np.asarray(oracle_static["vmask"])
    oracle_r1 = np.asarray(oracle_static["r1_hv0"])

    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(state.u.data.dtype)
    umask, vmask = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    prod = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, umask.astype(state.u.data.dtype),
        vmask.astype(state.v.data.dtype), card.recipe.grid,
        z_coord=card.recipe.z_coord, include_reciprocals=True)
    prod_e3v = r179._with_jpk_zero(np.asarray(prod[1])[1:])
    prod_vmask = oracle_vmask
    prod_r1 = np.asarray(prod[5])[1:]

    target = np.zeros(oracle_vmask.shape[:2], dtype=bool)
    target[-1] = np.any(oracle_vmask[-1, :, :30] != 0.0, axis=-1)
    oracle_depths = {
        name: r179._depth(r179._with_jpk_zero(oracle_rhs[f"{name}_v"]),
                          oracle_e3v, oracle_vmask, oracle_r1)
        for name in ORDER
    }
    candidate_depths = {
        name: r179._depth(
            r179._with_jpk_zero(r83.native_v(accumulated[f"{name}_v"])),
            prod_e3v, prod_vmask, prod_r1)
        for name in ORDER
    }
    rows = {name: r181._score(candidate_depths[name], oracle_depths[name], target)
            for name in ORDER}

    live_final = r83.native_v(traced.slow_forcing_producer["post_wind_v"])
    offline_final = candidate_depths["after_zad"]
    live_delta = np.abs(offline_final - live_final)
    oracle_final = oracle_depths["after_zad"]
    live_set = np.abs(live_final - oracle_final) > r179.FLOOR
    offline_set = np.abs(offline_final - oracle_final) > r179.FLOOR
    calibration = {
        "all_faces_at_floor": bool(np.max(live_delta) <= r179.FLOOR),
        "absolute_max": float(np.max(live_delta)),
        "same_over_floor_set": bool(np.array_equal(live_set, offline_set)),
        "symmetric_difference_cells": int(np.count_nonzero(live_set != offline_set)),
    }
    report = {
        "format": "nemo-testcase-l4-orca2-round240-corrected-operator-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "source_order": list(ORDER),
        "geometry_route": "nemo_qco_resolved_mesh_operands",
        "target": {"cells": int(np.count_nonzero(target)), "row": 147},
        "trace_passivity": passivity,
        "candidate_calibration": calibration,
        "operator_rows": rows,
        "first_operator": _first(rows),
        "compiled_source": {
            "operator_order": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175",
            "depth_average": "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215",
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--rhs-root", type=Path)
    parser.add_argument("--static-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.rhs_root, args.static_root,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.rhs_root, args.static_root,
                             args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError,
            rung0.GateError, r93.GateError, r170.GateError) as error:
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
