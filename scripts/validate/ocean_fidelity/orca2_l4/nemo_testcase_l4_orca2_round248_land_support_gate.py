#!/usr/bin/env python3
"""Locate round 247's kt=5 failure at NEMO's mask-safe reciprocals."""

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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round241_compact_mask_atomic_gate as r241,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
)

LABELS = ("independent", "given_nemo_entry")
PLANTS = (
    "none", "label-coverage", "full-support", "finite-entry",
    "source-order", "writer",
)


class GateError(RuntimeError):
    """The round-248 land-support record moved from its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _field_row(values) -> dict[str, object]:
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        absolute = np.where(finite, np.abs(values), -np.inf)
        index = tuple(map(int, np.unravel_index(np.argmax(absolute), values.shape)))
        maximum = float(absolute[index])
    else:
        index, maximum = None, None
    bad = np.argwhere(~finite)
    return {
        "shape": list(values.shape),
        "nonfinite": int(np.count_nonzero(~finite)),
        "max_abs_finite": maximum,
        "argmax_abs_finite": None if index is None else list(index),
        "first_nonfinite": None if bad.size == 0 else list(map(int, bad[0])),
    }


def _mask_row(candidate, oracle) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    indices = np.argwhere(unequal)
    return {
        "support": int(candidate.size),
        "unequal": int(np.count_nonzero(unequal)),
        "first_unequal": None if indices.size == 0 else list(map(int, indices[0])),
        "candidate_wet_oracle_dry": int(np.count_nonzero((candidate != 0) & (oracle == 0))),
        "candidate_dry_oracle_wet": int(np.count_nonzero((candidate == 0) & (oracle != 0))),
    }


def _reciprocal(depth, wet) -> np.ndarray:
    depth = np.asarray(depth, np.float64)
    wet = np.asarray(wet, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return wet / (depth + np.float64(1.0) - wet)


def _live_reciprocal(eta, depth, wet, area_t, area_face, axis) -> np.ndarray:
    """Replay domain.f90:213-214 then domqco.f90:213-218."""

    eta = np.asarray(eta, np.float64)
    weighted = np.asarray(area_t, np.float64) * eta
    neighbour = np.roll(weighted, -1, axis=axis)
    reference = _reciprocal(depth, wet)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.float64(0.5) * (weighted + neighbour) * reference
        ratio = ratio / np.asarray(area_face, np.float64)
        return reference / (np.float64(1.0) + ratio)


def _candidate_kt5_entry(deck_root: Path, frame_root: Path, label: str):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = r241.omt0.rung0_ladder._zero_forcing((148, 180))
    reference_depth = rung0.ladder.build_reference_depth_override(card)
    hooks = _NEMOWSRK3TestHooks(
        barotropic_external_mode_association=True,
        barotropic_reference_face_depth_override=reference_depth,
        barotropic_unmasked_v_transport=True,
        barotropic_materialize_v_transport=True,
        barotropic_atomic_fold_unit=True,
        preserve_literal_hpg_vmask=True,
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        for _ in range(4):
            state = jax.device_get(model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    return card, state


def measure(args: argparse.Namespace) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_resolved_mesh_operands,
    )

    require(args.label in LABELS, f"bad label {args.label!r}")
    stamp = r241._policy_and_stamp(args.expect_commit)
    card, state = _candidate_kt5_entry(args.deck_root, args.frame_root, args.label)
    fields = {
        name: _field_row(np.asarray(getattr(state, name).data, np.float64))
        for name in ("T", "S", "u", "v", "eta", "uu_b", "vv_b")
    }
    require(all(row["nonfinite"] == 0 for row in fields.values()),
            "kt=5 entry is not finite")

    eta = jnp.asarray(state.eta.data)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(eta), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    compact_u, compact_v = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    ops = jax.device_get(nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid,
        compact_u.astype(eta.dtype), compact_v.astype(eta.dtype),
        eta.dtype, h_ref.shape[-1]))

    hu0 = np.asarray(ops.hu_0, np.float64)
    hv0 = np.asarray(ops.hv_0, np.float64)
    candidate_u = (hu0 > 0.0).astype(np.float64)
    candidate_v = (hv0 > 0.0).astype(np.float64)
    oracle_u = np.max(np.asarray(ops.umask3, np.float64), axis=-1)
    oracle_v = np.max(np.asarray(ops.vmask3, np.float64), axis=-1)
    rows = {
        "u_mask_source": _mask_row(candidate_u, oracle_u),
        "v_mask_source": _mask_row(candidate_v, oracle_v),
    }
    ref_candidate_u = _reciprocal(hu0, candidate_u)
    ref_candidate_v = _reciprocal(hv0, candidate_v)
    ref_oracle_u = _reciprocal(hu0, oracle_u)
    ref_oracle_v = _reciprocal(hv0, oracle_v)
    area_t = np.asarray(ops.area_t, np.float64)
    entry_eta = np.asarray(state.eta.data, np.float64)
    live_candidate_u = _live_reciprocal(
        entry_eta, hu0, candidate_u, area_t, ops.area_u, 1)
    live_candidate_v = _live_reciprocal(
        entry_eta, hv0, candidate_v, area_t, ops.area_v, 0)
    live_oracle_u = _live_reciprocal(
        entry_eta, hu0, oracle_u, area_t, ops.area_u, 1)
    live_oracle_v = _live_reciprocal(
        entry_eta, hv0, oracle_v, area_t, ops.area_v, 0)
    for name, values in (
        ("hu_0", hu0), ("hv_0", hv0),
        ("candidate_r1_hu_0", ref_candidate_u),
        ("candidate_r1_hv_0", ref_candidate_v),
        ("oracle_r1_hu_0", ref_oracle_u),
        ("oracle_r1_hv_0", ref_oracle_v),
        ("candidate_live_r1_hu", live_candidate_u),
        ("candidate_live_r1_hv", live_candidate_v),
        ("oracle_live_r1_hu", live_oracle_u),
        ("oracle_live_r1_hv", live_oracle_v),
    ):
        rows[name] = _field_row(values)

    native_j, native_i = 0, 49
    point = {
        "compact_ji": [1, 49],
        "native_v_ji": [native_j, native_i],
        "hv_0_m": float(hv0[native_j, native_i]),
        "candidate_wet_v": float(candidate_v[native_j, native_i]),
        "resolved_ssvmask": float(oracle_v[native_j, native_i]),
        "candidate_r1_hv_0": float(ref_candidate_v[native_j, native_i]),
        "oracle_r1_hv_0": float(ref_oracle_v[native_j, native_i]),
        "candidate_live_r1_hv": float(live_candidate_v[native_j, native_i]),
        "oracle_live_r1_hv": float(live_oracle_v[native_j, native_i]),
        "entry_ssh_m": float(entry_eta[1, 49]),
    }
    return {
        "format": "nemo-testcase-l4-orca2-round248-land-support-v1",
        "status": "PASS_R248_LAND_SUPPORT_MEASURED",
        "label": args.label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "observer": "offline from ordinary completed kt=5 entry",
        "worktree": stamp,
        "entry_fields": fields,
        "rows": rows,
        "guard_cell": point,
        "source_order": [
            "hu_0/hv_0", "ssumask/ssvmask", "r1_hu_0/r1_hv_0",
            "live r1_hu/r1_hv", "zub/zvb", "stage transport",
            "continuity/W", "stage state",
        ],
    }


def classify(reports: dict[str, dict], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop("given_nemo_entry")
    elif plant == "full-support":
        reports["independent"]["rows"]["v_mask_source"]["support"] -= 1
    elif plant == "finite-entry":
        reports["independent"]["entry_fields"]["eta"]["nonfinite"] = 1
    elif plant == "source-order":
        reports["independent"]["source_order"][0] = "ssvmask"
    elif plant == "writer":
        reports["independent"]["rows"]["candidate_r1_hv_0"]["nonfinite"] = 1

    require(set(reports) == set(LABELS), "both claim labels are required")
    summary = {}
    for label, report in reports.items():
        require(report["status"] == "PASS_R248_LAND_SUPPORT_MEASURED",
                f"{label}: measurement status moved")
        require(report["observer"] == "offline from ordinary completed kt=5 entry",
                f"{label}: observer is not passive/offline")
        require(all(row["nonfinite"] == 0
                    for row in report["entry_fields"].values()),
                f"{label}: kt=5 entry moved non-finite")
        require(report["source_order"][:4] == [
            "hu_0/hv_0", "ssumask/ssvmask", "r1_hu_0/r1_hv_0",
            "live r1_hu/r1_hv",
        ], f"{label}: source order moved")
        for face in ("u", "v"):
            row = report["rows"][f"{face}_mask_source"]
            require(row["support"] == 148 * 180,
                    f"{label}: {face} full-support census moved")
        reciprocal_rows = [
            report["rows"][name]
            for name in (
                "candidate_r1_hu_0", "candidate_r1_hv_0",
                "candidate_live_r1_hu", "candidate_live_r1_hv",
            )
        ]
        mismatch = sum(report["rows"][name]["unequal"] for name in (
            "u_mask_source", "v_mask_source"))
        nonfinite = sum(row["nonfinite"] for row in reciprocal_rows)
        require(nonfinite == 0,
                f"{label}: a reciprocal is the first non-finite writer")
        summary[label] = {
            "mask_source_unequal": mismatch,
            "reciprocal_nonfinite": nonfinite,
            "guard_cell": report["guard_cell"],
            "entry_fields": report["entry_fields"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round248-classification-v1",
        "status": "HELD_R248_RECIPROCAL_OWNER_REFUTED",
        "summary": summary,
        "predictions": {
            "R248-P1": "REFUTED_RECIPROCALS_FINITE_BOTH_LABELS",
            "R248-P2": "OPEN_FIRST_WRITER_AFTER_LIVE_RECIPROCAL",
            "R248-P3": "NOT_ACTIVATED_PREREQUISITE_R248-P2",
            "R248-P4": "NOT_ACTIVATED_PREREQUISITE_R248-P3",
            "R248-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
        "open": (
            "The mask-source hypothesis is not the non-finite writer; continue "
            "offline from zub/zvb and the exact stage transport."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--inputs", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)), "measurement arguments are incomplete")
            result = measure(args)
        else:
            require(args.inputs is not None, "classification inputs are missing")
            reports = {
                label: json.loads(path.read_text(encoding="utf-8"))
                for label, path in zip(LABELS, args.inputs, strict=True)
            }
            result = classify(reports, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError) as error:
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
