#!/usr/bin/env python3
"""Gate round 247's resolved stage mask and complete atomic vector unit."""

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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round241_compact_mask_atomic_gate as r241,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round246_atomic_pair_gate as r246,
)

LABELS = r246.LABELS
PLANTS = (
    "none", "label-coverage", "direct-closure", "completion", "vmask",
    "zfv", "correction-input",
)


class GateError(RuntimeError):
    """The frozen round-247 atomic candidate no longer qualifies."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "label-coverage":
        report["operands"].pop("given_nemo_entry")
    elif plant == "direct-closure":
        report["direct"]["independent"]["rows"][0]["pair"][
            "metric_transport_v"]["complete"]["unequal"] = 1
    elif plant == "completion":
        report["ladders"]["independent"]["last_completed_kt"] = 7
    elif plant == "vmask":
        report["operands"]["independent"]["operand_rows"]["vmask"][
            "unequal"] = 1
    elif plant == "zfv":
        report["operands"]["independent"]["zFv_signature"]["unequal"] = 1
    elif plant == "correction-input":
        report["operands"]["independent"]["correction_input_rows"][
            "vn_adv"]["unequal"] = 1

    base = r246.classify(report)
    require(set(report["operands"]) == set(LABELS),
            "both claim labels are required")
    for label, operands in report["operands"].items():
        for name, row in operands["operand_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: stage operand {name} is non-bit")
        for name, row in operands["correction_input_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: correction input {name} is non-bit")
        require(operands["zFv_signature"]["unequal"] == 0,
                f"{label}: completed zFv is non-bit")

    base["format"] = "nemo-testcase-l4-orca2-round247-resolved-stage-mask-v1"
    base["status"] = "PASS_R247_RESOLVED_STAGE_MASK_ATOMIC_KT10"
    base["predictions"] = {
        "R247-P1": "CONFIRMED_STAGE_MASK_AND_ZFV_EXACT_BOTH_LABELS",
        "R247-P2": "CONFIRMED_DIRECT_0_0_0_BOTH_LABELS",
        "R247-P3": "CONFIRMED_OMT4_COMPLETES_KT10_BOTH_LABELS",
        "R247-P4": "PENDING_FULL_DECISION96_GATES",
        "R247-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    return base


def measure_direct(args: argparse.Namespace) -> dict[str, object]:
    return r246.measure_direct(args)


def measure_atomic(args: argparse.Namespace) -> dict[str, object]:
    return r246.measure_atomic(args)


def measure_operands(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    with r246._resolved_candidate_stage_geometry(card):
        with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True,
        ):
            return r232.measure(
                args.deck_root, args.frame_root, args.record_root,
                args.label, args.expect_commit, resolved_stage_masks=True)


def _live_w_row(card, state) -> dict[str, object]:
    """Rebuild the guarded entry geometry from a completed passive state."""

    if isinstance(state, dict):
        eta = np.asarray(state["ssh"], np.float64)
        depth = np.asarray(
            card.recipe.initial_state.H_bathy.data, np.float64)
        fields = {
            name: np.asarray(state["ssh" if name == "eta" else name], np.float64)
            for name in ("T", "S", "u", "v", "eta")
        }
    else:
        eta = np.asarray(state.eta.data, np.float64)
        depth = np.asarray(state.H_bathy.data, np.float64)
        fields = {
            name: np.asarray(getattr(state, name).data, np.float64)
            for name in ("T", "S", "u", "v", "eta")
        }
    raw = np.asarray(card.recipe.z_coord.nemo_e3w_0, np.float64)[..., 1:]
    safe_depth = np.where(depth > 0.0, depth, 1.0)
    reciprocal = np.float64(1.0) / safe_depth
    r3t = np.where(depth > 0.0, eta * reciprocal, 0.0)
    stretch = np.maximum(np.float64(1.0) + r3t, np.float64(1.0e-6))
    e3w = raw * stretch[..., None]
    bad = ~(np.isfinite(e3w) & (e3w > 0.0))
    indices = np.argwhere(bad)
    index = tuple(map(int, indices[0])) if indices.size else None
    result: dict[str, object] = {
        "invalid_e3w_count": int(np.count_nonzero(bad)),
        "first_invalid_jik": list(index) if index is not None else None,
        "field_nonfinite_counts": {
            name: int(np.count_nonzero(~np.isfinite(value)))
            for name, value in fields.items()
        },
    }
    if index is not None:
        j, i, k = index
        result["first_invalid"] = {
            "ssh_m": float(eta[j, i]),
            "ht0_m": float(depth[j, i]),
            "r1_ht0_per_m": float(reciprocal[j, i]),
            "r3t": float(r3t[j, i]),
            "stretch": float(stretch[j, i]),
            "raw_e3w0_m": float(raw[j, i, k]),
            "live_e3w_m": float(e3w[j, i, k]),
        }
    return result


def measure_boundary(args: argparse.Namespace) -> dict[str, object]:
    """Locate kt=5's refusal from completed kt=4 states, offline only."""

    import jax

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    entry = rung0.assemble_frame(args.frame_root, 1, 0)
    state = (card.recipe.initial_state if args.label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = r241.omt0.rung0_ladder._zero_forcing((148, 180))
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
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        for _ in range(3):
            state = jax.device_get(final_model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        kt4_entry = state
        kt4_stages = tuple(jax.device_get(model.step(
            kt4_entry, card.dt_s, freshwater=freshwater,
            surface_forcing=surface)) for model in stage_models)
        kt4_final = jax.device_get(final_model.step(
            kt4_entry, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    candidate = {
        name: _live_w_row(card, stage_state)
        for name, stage_state in zip(
            ("entry", "stage1", "stage2", "stage3"),
            (kt4_entry, *kt4_stages, kt4_final), strict=True)
    }
    oracle = {
        name: _live_w_row(card, rung0.assemble_frame(args.frame_root, 4, stage))
        for stage, name in enumerate(("entry", "stage1", "stage2", "stage3"))
    }
    first_bad = next(
        (name for name in ("entry", "stage1", "stage2", "stage3")
         if candidate[name]["invalid_e3w_count"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round247-live-w-boundary-v1",
        "status": "PASS_R247_LIVE_W_BOUNDARY_LOCATED",
        "label": args.label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "observer": "offline completed-stage reconstruction",
        "candidate_kt4": candidate,
        "nemo_kt4": oracle,
        "first_bad_completed_boundary": first_bad,
        "kt5_guard": "raw-mesh e3w_int must contain only finite values > 0",
        "worktree": stamp,
    }


def _load(paths: list[Path]) -> dict[str, dict]:
    return {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in zip(LABELS, paths, strict=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("direct", "atomic", "operands", "boundary", "classify"),
        default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--direct-in", nargs=2, type=Path)
    parser.add_argument("--atomic-in", nargs=2, type=Path)
    parser.add_argument("--operands-in", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "direct":
            require(all((args.deck_root, args.frame_root, args.twin_a,
                         args.twin_b, args.label, args.expect_commit)),
                    "direct measurement arguments are incomplete")
            result = measure_direct(args)
        elif args.mode == "atomic":
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "atomic measurement arguments are incomplete")
            result = measure_atomic(args)
        elif args.mode == "operands":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)),
                    "operand measurement arguments are incomplete")
            result = measure_operands(args)
        elif args.mode == "boundary":
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "boundary measurement arguments are incomplete")
            result = measure_boundary(args)
        else:
            require(all((args.direct_in, args.atomic_in, args.operands_in)),
                    "classification needs all six section reports")
            result = classify({
                "format": "nemo-testcase-l4-orca2-round247-input-v1",
                "execution": "production-jit-cpu-fp64-x64-libm",
                "direct": _load(args.direct_in),
                "ladders": _load(args.atomic_in),
                "operands": _load(args.operands_in),
            }, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r232.GateError, r241.GateError, r245.GateError, r246.GateError,
            rung0.GateError) as error:
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
