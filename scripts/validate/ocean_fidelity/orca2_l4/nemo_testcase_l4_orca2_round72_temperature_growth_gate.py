#!/usr/bin/env python3
"""Rank independent ORCA2 temperature error by exact interval, level, and basin."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round43_tracer_handoff_gate as handoff,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as month,
)

CHECKPOINTS = (0, 10, 240)
INTERVALS = ("0->10", "10->240")
REGION_VARIABLES = {
    "atlantic": "atlmsk",
    "pacific": "pacmsk",
    "indian": "indmsk",
}
SOURCE_ORDER = ("entry", "after_advection", "after_sbc", "qco_rk")
PLANTS = (
    "none",
    "restart-time-level",
    "dry-inclusion",
    "region-overlap",
    "interval-order",
    "temperature-boundary",
    "terminal-ulp",
)
ROUND71_T = {
    "bit_identical": False,
    "unequal": 438484,
    "count": 799200,
    "max_abs": 21.637832697714707,
    "mean_abs_over_unequal": 0.09897196744800131,
    "rms": 0.17564842520962015,
    "first_unequal_index": [0, 0, 0],
}


class GateError(RuntimeError):
    """A round-72 frozen predicate or instrument invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _temperature_restart(root: Path, step: int, card) -> tuple[np.ndarray, dict]:
    slabs = []
    coordinates = []
    files = []
    for rank in (0, 1):
        path = root / f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
        require(path.is_file(), f"missing step-{step} restart: {path}")
        with Dataset(path) as dataset:
            require(float(np.asarray(dataset["kt"][:])) == float(step),
                    f"{path.name}: kt is not {step}")
            slabs.append(month._restart_xyz(dataset["tn"], "tn"))
            coordinates.append((
                np.asarray(dataset["nav_lat"][:], dtype=np.float32),
                np.asarray(dataset["nav_lon"][:], dtype=np.float32),
            ))
        files.append({"path": str(path), "sha256": month.sha256(path)})
    temperature = np.concatenate(slabs, axis=1)
    lat = np.concatenate([row[0] for row in coordinates], axis=1)
    lon = np.concatenate([row[1] for row in coordinates], axis=1)
    expected_lat = np.asarray(
        np.rad2deg(np.asarray(card.recipe.grid.lat_T)), dtype=np.float32)
    expected_lon = np.asarray(
        np.rad2deg(np.asarray(card.recipe.grid.lon_T)), dtype=np.float32)
    require(np.array_equal(lat.view(np.uint32), expected_lat.view(np.uint32)),
            f"step-{step} restart latitude orientation differs")
    require(np.array_equal(lon.view(np.uint32), expected_lon.view(np.uint32)),
            f"step-{step} restart longitude orientation differs")
    return temperature, {"step": step, "files": files, "orientation_bit_exact": True}


def _regions(path: Path, wet: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    require(path.is_file(), f"missing supplied ORCA2 subbasin file: {path}")
    raw = {}
    with Dataset(path) as dataset:
        for name, variable in REGION_VARIABLES.items():
            values = np.asarray(dataset[variable][:], dtype=np.float64)
            require(values.shape == wet.shape[:2],
                    f"{variable}: shape {values.shape} != {wet.shape[:2]}")
            require(np.isin(values, (0.0, 1.0)).all(),
                    f"{variable}: mask is not exactly binary")
            raw[name] = values == 1.0
    overlap = sum(mask.astype(np.int8) for mask in raw.values())
    wet2 = np.any(wet, axis=-1)
    require(np.all(overlap[wet2] <= 1), "supplied basin masks overlap on wet cells")
    regions = {name: mask & wet2 for name, mask in raw.items()}
    regions["other"] = wet2 & (overlap == 0)
    coverage = sum(mask.astype(np.int8) for mask in regions.values())
    require(np.all(coverage[wet2] == 1), "basin regions do not partition wet cells")
    return regions, {
        "path": str(path),
        "sha256": month.sha256(path),
        "variables": REGION_VARIABLES,
        "binary": True,
        "overlap_wet_cells": 0,
        "uncovered_wet_cells_after_other": 0,
        "partition_exact": True,
        "column_counts": {name: int(mask.sum()) for name, mask in regions.items()},
    }


def _weighted_row(actual: np.ndarray, expected: np.ndarray,
                  weights: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == weights.shape == mask.shape,
            "weighted score shapes differ")
    require(mask.any(), "weighted score support is empty")
    a, e, w = actual[mask], expected[mask], weights[mask]
    require(np.isfinite(a).all() and np.isfinite(e).all(),
            "weighted score contains a non-finite value")
    require(np.isfinite(w).all() and np.all(w > 0.0),
            "weighted score has a non-positive weight")
    residual = a - e
    unequal = np.ascontiguousarray(a).view(np.uint64) != np.ascontiguousarray(e).view(np.uint64)
    sse = float(np.sum(w * residual * residual, dtype=np.float64))
    weight = float(np.sum(w, dtype=np.float64))
    maximum_index = int(np.argmax(np.abs(residual)))
    return {
        "bit_identical": not bool(unequal.any()),
        "unequal": int(unequal.sum()),
        "count": int(a.size),
        "max_abs": float(np.max(np.abs(residual))),
        "weighted_rms": float(np.sqrt(sse / weight)),
        "weighted_squared_error": sse,
        "weight_m3": weight,
        "support_max_flat_index": maximum_index,
    }


def _bin_growth(actual: dict[int, np.ndarray], expected: dict[int, np.ndarray],
                weights: np.ndarray, bins: dict[str, np.ndarray]) -> dict[str, object]:
    rows = {}
    for interval, start, end in (("0->10", 0, 10), ("10->240", 10, 240)):
        interval_rows = []
        for name, mask in bins.items():
            start_row = _weighted_row(actual[start], expected[start], weights, mask)
            end_row = _weighted_row(actual[end], expected[end], weights, mask)
            interval_rows.append({
                "name": name,
                "delta_weighted_squared_error": (
                    end_row["weighted_squared_error"]
                    - start_row["weighted_squared_error"]),
                "start": start_row,
                "end": end_row,
            })
        interval_rows.sort(
            key=lambda row: (-float(row["delta_weighted_squared_error"]), row["name"]))
        rows[interval] = interval_rows
    total = []
    for name, mask in bins.items():
        start_row = _weighted_row(actual[0], expected[0], weights, mask)
        end_row = _weighted_row(actual[240], expected[240], weights, mask)
        total.append({
            "name": name,
            "delta_weighted_squared_error": (
                end_row["weighted_squared_error"]
                - start_row["weighted_squared_error"]),
            "start": start_row,
            "end": end_row,
        })
    total.sort(key=lambda row: (-float(row["delta_weighted_squared_error"]), row["name"]))
    return {"intervals": rows, "total_0->240": total}


def _source_walk(card, deck_root: Path, record_root: Path) -> dict[str, object]:
    import jax.numpy as jnp

    tracer = handoff.phase2l.read_tracer(record_root / handoff.phase2l.TRACER_RECORD)
    oracle_entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    oracle_stage = ladder.read_state_frame(
        record_root / "oracle_stage_kt00000001_s1.bin", kt=1, stage=1)
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(card, deck_root, surface_fields, 1)
    state = card.recipe.initial_state
    mask = handoff._support_masks(card)["T"]
    actual = {"entry": np.asarray(state.T.data)[:, :90, :30]}
    expected = {
        "entry": np.asarray(oracle_entry["T"])[:, :90, :30],
        "after_advection": np.asarray(tracer["after_advection_T"])[..., :30],
        "after_sbc": np.asarray(tracer["after_sbc_T"])[..., :30],
        "qco_rk": np.asarray(oracle_stage["T"]),
    }
    for exposure, name in (("after_advection", "after_advection"),
                           ("after_sbc", "after_sbc"),
                           ("stage1", "qco_rk")):
        out = handoff._run(
            card, state, freshwater, surface, endpoint=None, exposure=exposure)
        actual[name] = np.asarray(out.T.data)[:, :90, :30]
    rows = {name: handoff.bit_score(actual[name], expected[name], mask)
            for name in SOURCE_ORDER}
    first = next((name for name in SOURCE_ORDER if not rows[name]["bit_exact"]), None)
    require(first is not None, "independent kt=1 temperature stayed bit-exact")
    return {
        "order": list(SOURCE_ORDER),
        "rows": rows,
        "first_non_bit_boundary": first,
        "compiled_citations": {
            "after_advection": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:633-643"),
            "after_sbc": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:645-651"),
            "qco_rk": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:670-680"),
        },
    }


def _closure(rows: list[dict], expected: float, label: str) -> dict[str, object]:
    observed = float(sum(float(row["end"]["weighted_squared_error"]) for row in rows))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(expected), 1.0)
    require(abs(observed - expected) <= tolerance,
            f"{label} decomposition does not close: {observed} != {expected}")
    return {"observed": observed, "expected": expected,
            "absolute_error": abs(observed - expected), "tolerance": tolerance}


def run_growth(deck_root: Path, surface_root: Path, month_root: Path,
               restart_ledger: Path, ten_step_root: Path,
               ten_step_reference: Path, subbasins: Path) -> dict[str, object]:
    snapshots = {}

    def observe(step, state):
        snapshots[int(step)] = np.asarray(state.T.data)[..., :30].copy()

    month_report = month.run_month(
        deck_root, surface_root, month_root, restart_ledger,
        ten_step_root, ten_step_reference, temperature_observer=observe)
    require(set(snapshots) == {10, 240}, "month observer missed a frozen checkpoint")
    require(month_report["terminal"]["rows"]["T"] == ROUND71_T,
            "round-71 terminal all-cell temperature row did not reproduce")

    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(ten_step_root, 1, stage=None)["T"]
    restart10, restart10_meta = _temperature_restart(ten_step_root, 10, card)
    restart240, restart240_meta = _temperature_restart(month_root, 240, card)
    stage10 = ladder.read_state_frame(
        ten_step_root / "oracle_stage_kt00000010_s3.bin", kt=10, stage=3)["T"]
    wet = np.asarray(card.recipe.z_coord.is_active, dtype=bool)[..., :30]
    calibration = handoff.bit_score(
        restart10[:, :90], stage10, wet[:, :90])
    require(calibration["bit_exact"],
            "step-10 restart differs from its stage-3 frame on wet rank-0 cells")

    weights = (
        np.asarray(card.recipe.grid.area_T, dtype=np.float64)[..., None]
        * np.asarray(card.recipe.z_coord.dz_ref, dtype=np.float64)[None, None, :30]
        * wet
    )
    actual = {0: np.asarray(card.recipe.initial_state.T.data)[..., :30], **snapshots}
    expected = {0: entry, 10: restart10, 240: restart240}
    global_rows = {
        str(step): _weighted_row(actual[step], expected[step], weights, wet)
        for step in CHECKPOINTS
    }
    intervals = []
    for name, start, end in (("0->10", 0, 10), ("10->240", 10, 240)):
        intervals.append({
            "name": name,
            "delta_weighted_squared_error": (
                global_rows[str(end)]["weighted_squared_error"]
                - global_rows[str(start)]["weighted_squared_error"]),
            "start_weighted_rms": global_rows[str(start)]["weighted_rms"],
            "end_weighted_rms": global_rows[str(end)]["weighted_rms"],
        })
    intervals.sort(key=lambda row: -float(row["delta_weighted_squared_error"]))

    depth_bins = {
        f"k={level}": wet & (np.arange(30)[None, None, :] == level)
        for level in range(30)
    }
    regions2, region_meta = _regions(subbasins, wet)
    region_bins = {name: wet & mask[..., None] for name, mask in regions2.items()}
    depth = _bin_growth(actual, expected, weights, depth_bins)
    regions = _bin_growth(actual, expected, weights, region_bins)
    terminal_sse = global_rows["240"]["weighted_squared_error"]
    closure = {
        "depth": _closure(depth["total_0->240"], terminal_sse, "depth"),
        "region": _closure(regions["total_0->240"], terminal_sse, "region"),
    }

    dry = ~wet
    dry_row = ladder.score(actual[240][dry], expected[240][dry])
    residual = np.abs(actual[240] - expected[240])
    wet_peak = np.where(wet, residual, -np.inf)
    peak = tuple(int(value) for value in np.unravel_index(np.argmax(wet_peak), wet.shape))
    source_walk = _source_walk(card, deck_root, ten_step_root)
    return {
        "format": "nemo-testcase-l4-orca2-round72-temperature-growth-v1",
        "claim_label": "independent",
        "execution": month_report["execution"],
        "steps_completed": month_report["steps_completed"],
        "initial_mode": month_report["initial_mode"],
        "decision52_bridge": month_report["decision52_bridge"],
        "unmeasured_features": month_report["unmeasured_features"],
        "weighting": {
            "wet_only": True,
            "formula": "area_T * dz_ref * is_active",
            "dtype": "float64",
        },
        "restart_calibration": calibration,
        "restarts": {"10": restart10_meta, "240": restart240_meta},
        "round71_terminal_temperature_reproduced": True,
        "round71_terminal_temperature": month_report["terminal"]["rows"]["T"],
        "global_checkpoints": global_rows,
        "interval_ranking": intervals,
        "depth_growth": depth,
        "region_growth": regions,
        "region_partition": region_meta,
        "decomposition_closure": closure,
        "terminal_dry_cells": dry_row,
        "terminal_wet_peak": {
            "index": list(peak),
            "max_abs": float(residual[peak]),
            "latitude_deg": float(np.rad2deg(np.asarray(card.recipe.grid.lat_T))[peak[:2]]),
            "longitude_deg": float(np.rad2deg(np.asarray(card.recipe.grid.lon_T))[peak[:2]]),
            "depth_m": float(np.asarray(card.recipe.z_coord.z_full_ref)[peak[2]]),
        },
        "source_walk": source_walk,
        "worktree": month_report["worktree"],
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "restart-time-level":
        report["restart_calibration"]["bit_exact"] = False
    elif plant == "dry-inclusion":
        report["weighting"]["wet_only"] = False
    elif plant == "region-overlap":
        report["region_partition"]["overlap_wet_cells"] = 1
    elif plant == "interval-order":
        report["interval_ranking"][0]["name"] = "0->240"
    elif plant == "temperature-boundary":
        report["source_walk"]["rows"]["entry"]["bit_exact"] = False
    elif plant == "terminal-ulp":
        report["round71_terminal_temperature"]["max_abs"] = math.nextafter(
            float(report["round71_terminal_temperature"]["max_abs"]), math.inf)

    require(report.get("claim_label") == "independent", "claim is not independent")
    require(report.get("initial_mode") == "card_own_state"
            and report.get("decision52_bridge") is None,
            "growth run did not use the card's own state")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(int(report.get("steps_completed", -1)) == 240, "month did not complete")
    require(tuple(report.get("unmeasured_features", ())) == ladder.EXPECTED_UNMEASURED,
            "sea-ice/unmeasured registry changed")
    require(report["weighting"] == {
        "wet_only": True,
        "formula": "area_T * dz_ref * is_active",
        "dtype": "float64",
    }, "wet/volume weighting changed")
    require(report["restart_calibration"].get("bit_exact") is True,
            "step-10 restart time level is not calibrated")
    require(report.get("round71_terminal_temperature_reproduced") is True
            and report.get("round71_terminal_temperature") == ROUND71_T,
            "round-71 terminal temperature row did not reproduce")
    require(list(report["global_checkpoints"]) == ["0", "10", "240"],
            "checkpoint registry changed")
    require({row["name"] for row in report["interval_ranking"]} == set(INTERVALS),
            "interval registry changed")
    require(report["region_partition"].get("partition_exact") is True
            and int(report["region_partition"].get("overlap_wet_cells", -1)) == 0
            and int(report["region_partition"].get(
                "uncovered_wet_cells_after_other", -1)) == 0,
            "supplied basin partition is not exact")
    require(list(report["source_walk"]["rows"]) == list(SOURCE_ORDER),
            "source boundary order changed")
    require(report["source_walk"]["rows"]["entry"].get("bit_exact") is True,
            "independent kt=1 entry temperature is not bit-exact")
    first = report["source_walk"].get("first_non_bit_boundary")
    require(first in SOURCE_ORDER[1:], "no post-entry non-bit temperature boundary")
    require(report["source_walk"]["rows"][first].get("bit_exact") is False,
            "named first boundary is exact")
    for name in ("depth", "region"):
        row = report["decomposition_closure"][name]
        require(float(row["absolute_error"]) <= float(row["tolerance"]),
                f"{name} decomposition closure failed")

    interval_owner = report["interval_ranking"][0]["name"]
    depth_owner = report["depth_growth"]["total_0->240"][0]["name"]
    region_owner = report["region_growth"]["total_0->240"][0]["name"]
    predictions = {
        "interval_10->240_largest": {
            "status": "CONFIRMED" if interval_owner == "10->240" else "REFUTED",
            "observed": interval_owner,
        },
        "surface_level_largest": {
            "status": "CONFIRMED" if depth_owner == "k=0" else "REFUTED",
            "observed": depth_owner,
        },
        "pacific_largest": {
            "status": "CONFIRMED" if region_owner == "pacific" else "REFUTED",
            "observed": region_owner,
        },
        "after_advection_first": {
            "status": "CONFIRMED" if first == "after_advection" else "REFUTED",
            "observed": first,
        },
    }
    return {**report, "status": "PASS_TEMPERATURE_GROWTH_RANKING",
            "prediction_ledger": predictions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--month-root", type=Path)
    parser.add_argument("--restart-ledger", type=Path)
    parser.add_argument("--ten-step-root", type=Path)
    parser.add_argument("--ten-step-reference", type=Path)
    parser.add_argument("--subbasins", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime plants require --classify-json")
            inputs = (args.deck_root, args.surface_root, args.month_root,
                      args.restart_ledger, args.ten_step_root,
                      args.ten_step_reference, args.subbasins)
            require(all(inputs), "run mode requires every record input")
            raw = run_growth(*inputs)
        report = classify(raw, plant=args.plant)
    except (GateError, month.GateError, ladder.GateError, handoff.GateError,
            KeyError, OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_TEMPERATURE_GROWTH_RANKING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
