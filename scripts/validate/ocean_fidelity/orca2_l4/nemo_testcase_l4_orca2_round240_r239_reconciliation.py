#!/usr/bin/env python3
"""Reconcile round 239's HPG attribution against D114 production geometry."""

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
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as r42,
)
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
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as r179,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round181_hpg_component_walk as r181,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)

PLANTS = ("none", "raw-hpg", "stale-effect", "production-closure", "route")


class GateError(RuntimeError):
    """The two measurements do not reconcile at one upstream quantity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "raw-hpg":
        report["raw_hpg"]["bit_exact"] = False
    elif plant == "stale-effect":
        report["depth_endpoints"]["legacy_diagnostic"]["bit_exact"] = True
    elif plant == "production-closure":
        report["depth_endpoints"]["production"]["bit_exact"] = False
    elif plant == "route":
        report["routes"]["production"] = report["routes"]["legacy_diagnostic"]

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(report["target"] == {"cells": 68, "row": 147, "wet_levels": 1319},
            "registered fold target moved")
    require(report["raw_hpg"]["bit_exact"],
            "current literal raw HPG is not exact")
    require(not report["depth_endpoints"]["legacy_diagnostic"]["bit_exact"],
            "legacy diagnostic route no longer reproduces the false debt")
    require(report["depth_endpoints"]["legacy_diagnostic"]["differing_cells"] == 68,
            "legacy diagnostic debt set moved")
    require(report["depth_endpoints"]["production"]["bit_exact"],
            "D114 production route does not close the HPG depth endpoint")
    require(report["depth_endpoints"]["production"]["at_floor"],
            "D114 production route left the HPG endpoint floor")
    require(report["routes"] == {
        "legacy_diagnostic": "nemo_qco_card_mesh_operands",
        "production": "nemo_qco_resolved_mesh_operands",
    }, "geometry routes are not discriminated")
    require(any(not row["bit_exact"] for row in
                report["geometry_route_rows"].values()),
            "the two geometry routes are indistinguishable")

    report["status"] = "PASS_R240_R239_STALE_GEOMETRY_DIAGNOSTIC"
    report["reconciliation"] = (
        "round239 HPG attribution retracted: its offline depth-average helper "
        "bypassed the D114 resolved raw-face geometry route"
    )
    return report


def measure(
    deck_root: Path,
    rhs_root: Path,
    static_root: Path,
    expect_commit: str,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_face_masks_3d
    from legoesm.ocean.vertical import compute_layer_thickness

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-240 reconciliation requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "reconciliation requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    model.prime_step_caches(state)
    inputs = r42._candidate_inputs(model, state)
    literal = r42._literal_from_inputs(
        inputs, card.recipe.model_config.g, grid=card.recipe.grid)
    raw_hpg = r179._with_jpk_zero(r83.native_v(literal["sum_v"]))

    oracle_rhs, _ = r93.assemble_rhs(rhs_root)
    oracle_static, _ = r170.assemble_record(static_root)
    oracle_hpg = r179._with_jpk_zero(oracle_rhs["after_hpg_v"])
    oracle_e3v = np.asarray(oracle_static["e3v"])
    oracle_vmask = np.asarray(oracle_static["vmask"])
    oracle_r1 = np.asarray(oracle_static["r1_hv0"])
    oracle_depth = r179._depth(oracle_hpg, oracle_e3v, oracle_vmask, oracle_r1)

    target = np.zeros(oracle_vmask.shape[:2], dtype=bool)
    target[-1] = np.any(oracle_vmask[-1, :, :30] != 0.0, axis=-1)
    contributing = np.broadcast_to(target[..., None], oracle_hpg.shape) & (
        oracle_vmask != 0.0)

    legacy = r170._candidate_reference_operands(card, state)
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

    legacy_depth = r179._depth(
        raw_hpg, np.asarray(legacy["e3_v"]),
        np.asarray(legacy["mask_v"]), np.asarray(legacy["r1_h0_v"]))
    production_depth = r179._depth(
        raw_hpg, prod_e3v, prod_vmask, prod_r1)

    report = {
        "format": "nemo-testcase-l4-orca2-round240-reconciliation-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "target": {"cells": int(np.count_nonzero(target)), "row": 147,
                   "wet_levels": int(np.count_nonzero(contributing))},
        "raw_hpg": r181._score(raw_hpg, oracle_hpg, contributing),
        "geometry_route_rows": {
            "e3v": r181._score(np.asarray(legacy["e3_v"]), prod_e3v,
                                contributing),
            "vmask": r181._score(np.asarray(legacy["mask_v"]), prod_vmask,
                                  np.broadcast_to(target[..., None],
                                                  prod_vmask.shape)),
            "r1_hv0": r181._score(np.asarray(legacy["r1_h0_v"]), prod_r1,
                                    target),
        },
        "depth_endpoints": {
            "legacy_diagnostic": r181._score(legacy_depth, oracle_depth, target),
            "production": r181._score(production_depth, oracle_depth, target),
        },
        "routes": {
            "legacy_diagnostic": "nemo_qco_card_mesh_operands",
            "production": "nemo_qco_resolved_mesh_operands",
        },
        "source": {
            "production_route": "ocean_model_latlon_cgrid.py:1820-1840",
            "legacy_diagnostic_route": (
                "nemo_testcase_l4_orca2_round170_slow_producer_walk.py:139-166"
            ),
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
            result = measure(
                args.deck_root, args.rhs_root, args.static_root, args.expect_commit)
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
