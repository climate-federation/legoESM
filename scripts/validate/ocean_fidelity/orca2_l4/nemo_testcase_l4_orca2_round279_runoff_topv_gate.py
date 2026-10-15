#!/usr/bin/env python3
"""Verify ORCA2 runoff depth and localise the held top-V exchange growth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.ocean.freshwater import resolve_runoff_spread_arg
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as round249,
    nemo_testcase_l4_orca2_round256_fold_origin_gate as round256,
)

FIELDS = ("T", "S", "u", "v", "ssh")
PLANTS = ("none", "runoff-depth", "fold-row")


class GateError(RuntimeError):
    """Round-279's frozen card or boundary predictions moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(actual, expected) -> dict[str, object]:
    a = np.asarray(actual, np.float64)
    b = np.asarray(expected, np.float64)
    require(a.shape == b.shape, f"shape moved: {a.shape} != {b.shape}")
    bits = a.view(np.uint64) != b.view(np.uint64)
    delta = np.abs(a - b)
    flat = int(np.nanargmax(delta)) if delta.size else 0
    return {
        "shape": list(a.shape),
        "unequal": int(np.count_nonzero(bits)),
        "max_abs": float(np.nanmax(delta)) if delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "argmax": list(np.unravel_index(flat, delta.shape)) if delta.size else [],
    }


def _cards(deck_root: Path):
    return {
        "rung10": build_orca2_zps_card(deck_root),
        "rung0": rung0.build_rung0_card(deck_root),
        "omt1": omt1.build_omt1_card(deck_root),
        "omt2": omt2.build_omt2_card(deck_root),
        "omt3": omt3.build_omt3_card(deck_root),
        "omt4": omt4.build_omt4_card(deck_root),
    }


def _runoff_audit(deck_root: Path, *, plant: str) -> dict[str, object]:
    rows = {}
    for name, card in _cards(deck_root).items():
        cfg = card.recipe.model_config
        spread = cfg.runoff_depth_spread_m
        if plant == "runoff-depth" and name == "rung10":
            spread = 0.0
        resolved = resolve_runoff_spread_arg(
            cfg._replace(runoff_depth_spread_m=spread))
        rows[name] = {
            "runoff_depth_spread_m": float(spread),
            "runoff_depth_spread_map": (
                None if cfg.runoff_depth_spread_map is None else "array"),
            "resolved": None if resolved is None else (
                float(resolved) if np.ndim(resolved) == 0 else "array"),
        }
    require(rows["rung10"] == {
        "runoff_depth_spread_m": 150.0,
        "runoff_depth_spread_map": None,
        "resolved": 150.0,
    }, f"shipped ORCA2 runoff depth moved: {rows['rung10']}")
    for name in ("rung0", "omt1", "omt2", "omt3", "omt4"):
        require(rows[name] == {
            "runoff_depth_spread_m": 0.0,
            "runoff_depth_spread_map": None,
            "resolved": None,
        }, f"{name} runoff depth moved: {rows[name]}")
    return {
        "cards": rows,
        "oracle": {
            "ln_rnf_depth": False,
            "ln_rnf_depth_ini": False,
            "resolved": "surface level only",
        },
        "verdict": "CONFIRMED_CARD_MISMATCH",
    }


def _stage_model(card, stage: int, override=None):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )

    hooks = _NEMOWSRK3TestHooks(
        expose_momentum_stage=stage,
        expose_tracer_stage=stage,
        stage_entry_override=override,
    )
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _boundary_rows(off, on) -> dict[str, dict[str, object]]:
    off_fields = rung0.candidate_fields(off)
    on_fields = rung0.candidate_fields(on)
    return {name: _score(on_fields[name], off_fields[name]) for name in FIELDS}


def _top_v_audit(deck_root: Path, record_root: Path, *, plant: str):
    import jax

    base = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, base)
    cfg = base.recipe.model_config
    require(not cfg.rk3_fold_exchange_top_v_row,
            "top-V exchange unexpectedly enabled on the production OMT-4 card")
    arm = base._replace(recipe=base.recipe._replace(
        model_config=cfg._replace(rk3_fold_exchange_top_v_row=True)))
    freshwater, surface = rung0_ladder._zero_forcing((148, 180))
    state = base.recipe.initial_state

    outputs = {}
    for label, card in (("off", base), ("on", arm)):
        outputs[label] = []
        for stage in (1, 2):
            outputs[label].append(jax.device_get(_stage_model(card, stage).step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)))

    rows = {
        f"stage{stage}": _boundary_rows(
            outputs["off"][stage - 1], outputs["on"][stage - 1])
        for stage in (1, 2)
    }
    if plant == "fold-row":
        rows["stage1"]["T"]["unequal"] = 1
    require(all(rows["stage1"][name]["unequal"] == 0
                for name in ("T", "S", "u", "ssh")),
            f"top-V statement moved another stage-1 field: {rows['stage1']}")
    require(rows["stage1"]["v"]["unequal"] == 5400,
            f"top-V direct support moved: {rows['stage1']['v']}")
    require(rows["stage1"]["v"]["argmax"][0] == 147,
            "top-V direct argmax left the northern fold row")
    require(rows["stage2"]["T"]["unequal"] > 0
            and rows["stage2"]["S"]["unequal"] > 0,
            "the first downstream tracer boundary disappeared")
    require(rows["stage2"]["T"]["argmax"][0] == 147
            and rows["stage2"]["S"]["argmax"][0] == 147,
            "the first downstream tracer argmax left the fold row")

    oracle1 = rung0.assemble_frame(record_root, 1, 1)
    arm1 = rung0.candidate_fields(outputs["on"][0])
    v_mask = round249._native(
        "v", np.asarray(outputs["on"][0].v_mask.data, np.float64))
    perm = np.asarray(arm.recipe.grid.fold.perm_v)
    sign = float(arm.recipe.grid.fold.vector_sign_v)
    source_mask = v_mask[-2, perm]
    if source_mask.ndim == 1:
        source_mask = source_mask[:, None]
    candidate_fold = sign * arm1["v"][-2, perm] * source_mask
    oracle_fold = sign * oracle1["v"][-2, perm]
    source = {
        "candidate_source_vs_nemo": _score(
            arm1["v"][-2], oracle1["v"][-2]),
        "candidate_top_replays_its_source": _score(
            arm1["v"][-1], candidate_fold),
        "nemo_top_replays_its_source": _score(
            oracle1["v"][-1], oracle_fold),
        "candidate_top_vs_nemo": _score(
            arm1["v"][-1], oracle1["v"][-1]),
    }
    require(source["candidate_top_replays_its_source"]["unequal"] == 0,
            "candidate top-V copy does not replay its written source")
    require(source["nemo_top_replays_its_source"]["unequal"] == 0,
            "NEMO top-V row is not the compiled fold image")
    require(source["candidate_source_vs_nemo"]["unequal"] > 0,
            "the top-V source row unexpectedly became bit-exact")

    # Frozen missing-T/S-partner hypothesis: apply the same source-derived
    # post-stage association used by round 256, then replay stage 2 at the
    # private pointer-swap seam.  It is a diagnostic, not production physics.
    stage1 = outputs["on"][0]
    associated = round256._associated_stage_values(base, (
        np.asarray(stage1.u.data), np.asarray(stage1.v.data),
        np.asarray(stage1.T.data), np.asarray(stage1.S.data)))
    override = (2, *associated, np.asarray(stage1.eta.data))
    corrected = jax.device_get(_stage_model(arm, 2, override).step(
        state, arm.dt_s, freshwater=freshwater, surface_forcing=surface))
    oracle2 = rung0.assemble_frame(record_root, 1, 2)
    raw2 = rung0.candidate_fields(outputs["on"][1])
    corrected2 = rung0.candidate_fields(corrected)
    association = {
        name: {
            "raw_vs_nemo": _score(raw2[name], oracle2[name]),
            "associated_vs_nemo": _score(corrected2[name], oracle2[name]),
        } for name in FIELDS
    }
    closes = all(
        association[name]["associated_vs_nemo"]["max_abs"]
        <= association[name]["raw_vs_nemo"]["max_abs"]
        for name in ("T", "S"))
    return {
        "label": "independent",
        "direct_and_first_downstream": rows,
        "top_v_source_operand": source,
        "post_stage_ts_association_control": association,
        "preregistered_ts_partner": "CONFIRMED" if closes else "REFUTED",
        "owner": (
            "the completed stage-1 V source row is already non-bit before "
            "stprk3_stg.f90:795-796 copies it onto the pivot row"
        ),
    }


def run(deck_root: Path, record_root: Path, expect_commit: str,
        *, plant: str = "none") -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(plant in PLANTS, f"unknown plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-279 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu", "round 279 is CPU-only")
    return {
        "format": "nemo-testcase-l4-orca2-round279-v1",
        "status": "PASS_R279_RUNOFF_AND_TOPV_DIAGNOSIS",
        "worktree": stamp,
        "runoff": _runoff_audit(deck_root, plant=plant),
        "top_v": _top_v_audit(deck_root, record_root, plant=plant),
        "compiled_sources": {
            "runoff_branch": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:259-283",
            "post_stage_exchange": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:775-798",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.expect_commit,
                     plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print("STATUS PASS_R279_RUNOFF_AND_TOPV_DIAGNOSIS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
