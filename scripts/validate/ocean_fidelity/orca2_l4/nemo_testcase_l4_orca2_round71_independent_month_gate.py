#!/usr/bin/env python3
"""Run and classify Decision 52's independent 240-step ORCA2 ocean month."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from netCDF4 import Dataset

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round69_month_surface_gate as surface_gate,
)

STEPS = 240
FIELD_ORDER = ladder.FIELD_ORDER
FIELD_UNITS = {"T": "degC", "S": "g/kg", "u": "m/s", "v": "m/s", "ssh": "m"}
EXPECTED_TEN_STEP_SHA256 = "297fb05b535e119e5a0c812c7778658230d3d94cacfcb4bd960b5cce9cc7efce"
RESTART_DIMS_3D = ("time_counter", "nav_lev", "y", "x")
RESTART_DIMS_2D = ("time_counter", "y", "x")
PLANTS = (
    "none",
    "initial-mode",
    "surface-frame-drop",
    "ten-step-metric",
    "restart-orientation",
    "terminal-ulp",
    "terminal-nonfinite",
)


class GateError(RuntimeError):
    """The independent month violated a frozen round-71 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_ledger(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing SHA-256 ledger: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        require(len(parts) == 2 and len(parts[0]) == 64, f"malformed SHA-256 row: {line!r}")
        name = Path(parts[1].lstrip("* ")).name
        require(name not in rows, f"duplicate SHA-256 name: {name}")
        rows[name] = parts[0]
    return rows


def assemble_surface(root: Path, kt: int) -> dict[str, np.ndarray]:
    """Read and join the two self-describing owned surface slabs."""
    slabs = [
        surface_gate.read_surface(root / surface_gate._record_name(kt, rank), kt=kt, rank=rank)[
            "fields"
        ]
        for rank in (0, 1)
    ]
    require(slabs[0].keys() == slabs[1].keys(), f"kt={kt}: surface slab registry differs")
    assembled = {
        name: np.concatenate([slabs[0][name], slabs[1][name]], axis=1) for name in slabs[0]
    }
    require(
        all(values.shape[:2] == (148, 180) for values in assembled.values()),
        f"kt={kt}: assembled surface shape changed",
    )
    return assembled


def validate_surface_schema_calibration(
    fields: dict[str, np.ndarray], old_fields: dict[str, np.ndarray], *, kt: int
) -> tuple[int, list[str]]:
    """Compare every production operand while registering legacy-only streams."""
    require(
        tuple(fields) == surface_gate.FIELDS,
        f"kt={kt}: month surface field registry differs from the admitted schema",
    )
    missing = set(fields) - set(old_fields)
    require(not missing, f"kt={kt}: old surface frame misses {sorted(missing)}")
    for name in fields:
        require(
            np.array_equal(
                np.ascontiguousarray(fields[name]).view(np.uint64),
                np.ascontiguousarray(old_fields[name]).view(np.uint64),
            ),
            f"kt={kt}: old/new surface payload differs for {name}",
        )
    return len(fields), sorted(set(old_fields) - set(fields))


def validate_chlorophyll_clock(root: Path) -> dict[str, object]:
    """Bind the reconstructed month clock to NEMO's resolved record centres."""
    path = root / "ocean.output"
    require(path.is_file(), f"missing resolved NEMO log: {path}")
    source = path.read_text()
    anchors = {
        "kt1": (
            r"fld_read: var CHLA kt =\s+1 \(\s*0\.0625 days\).*"
            r"records b/a:\s+0012/\s+0001 \(days\s+-15\.5000/\s+15\.5000\)"
        ),
        "kt240": (
            r"fld_read: var CHLA kt =\s+240 \(\s*29\.9375 days\).*"
            r"records b/a:\s+0001/\s+0002 \(days\s+15\.5000/\s+45\.0000\)"
        ),
    }
    for name, pattern in anchors.items():
        require(
            re.search(pattern, source) is not None,
            f"resolved chlorophyll clock anchor changed: {name}",
        )
    return {
        "status": "RESOLVED_CENTRES_MATCH",
        "path": str(path),
        "sha256": sha256(path),
        "midpoint_days": {"kt1": 0.0625, "kt240": 29.9375},
        "record_centres_days": [-15.5, 15.5, 45.0],
    }


def _restart_xyz(variable, name: str) -> np.ndarray:
    require(
        variable.dimensions == RESTART_DIMS_3D,
        f"{name}: axes {variable.dimensions} != {RESTART_DIMS_3D}",
    )
    require(variable.dtype == np.dtype("float64"), f"{name}: not float64")
    variable.set_auto_maskandscale(False)
    values = np.asarray(variable[0], dtype=np.float64).transpose(1, 2, 0)
    require(values.shape == (148, 90, 31), f"{name}: shape {values.shape}")
    return values[..., :30]


def _restart_xy(variable, name: str) -> np.ndarray:
    require(
        variable.dimensions == RESTART_DIMS_2D,
        f"{name}: axes {variable.dimensions} != {RESTART_DIMS_2D}",
    )
    require(variable.dtype == np.dtype("float64"), f"{name}: not float64")
    variable.set_auto_maskandscale(False)
    values = np.asarray(variable[0], dtype=np.float64)
    require(values.shape == (148, 90), f"{name}: shape {values.shape}")
    return values


def read_terminal_restart(
    root: Path,
    ledger_path: Path,
    card,
    *,
    plant: str = "none",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assemble NEMO's two owned restart slabs and bind their orientation."""
    ledger = _sha_ledger(ledger_path)
    slabs: list[dict[str, np.ndarray]] = []
    coordinates: list[tuple[np.ndarray, np.ndarray]] = []
    files = []
    for rank in (0, 1):
        path = root / f"ORCA2_00000240_restart_{rank:04d}.nc"
        require(path.is_file(), f"missing terminal restart: {path}")
        observed_sha = sha256(path)
        if plant == "terminal-ulp" and rank == 0:
            observed_sha = "0" * 64
        require(
            ledger.get(path.name) == observed_sha, f"terminal restart digest changed: {path.name}"
        )
        with Dataset(path) as dataset:
            require(
                float(np.asarray(dataset["kt"][:])) == float(STEPS),
                f"{path.name}: kt is not {STEPS}",
            )
            slab = {
                "T": _restart_xyz(dataset["tn"], "tn"),
                "S": _restart_xyz(dataset["sn"], "sn"),
                "u": _restart_xyz(dataset["un"], "un"),
                "v": _restart_xyz(dataset["vn"], "vn"),
                "ssh": _restart_xy(dataset["sshn"], "sshn"),
            }
            coordinates.append(
                (
                    np.asarray(dataset["nav_lat"][:], dtype=np.float32),
                    np.asarray(dataset["nav_lon"][:], dtype=np.float32),
                )
            )
        require(
            all(np.isfinite(values).all() for values in slab.values()),
            f"{path.name}: non-finite terminal payload",
        )
        slabs.append(slab)
        files.append({"path": str(path), "sha256": observed_sha})

    fields = {
        name: np.concatenate([slabs[0][name], slabs[1][name]], axis=1) for name in FIELD_ORDER
    }
    lat = np.concatenate([coordinates[0][0], coordinates[1][0]], axis=1)
    lon = np.concatenate([coordinates[0][1], coordinates[1][1]], axis=1)
    if plant == "restart-orientation":
        lon = lon[:, ::-1]
    expected_lat = np.asarray(np.rad2deg(np.asarray(card.recipe.grid.lat_T)), dtype=np.float32)
    expected_lon = np.asarray(np.rad2deg(np.asarray(card.recipe.grid.lon_T)), dtype=np.float32)
    require(
        np.array_equal(lat.view(np.uint32), expected_lat.view(np.uint32)),
        "terminal restart latitude orientation differs from the card",
    )
    require(
        np.array_equal(lon.view(np.uint32), expected_lon.view(np.uint32)),
        "terminal restart longitude orientation differs from the card",
    )
    return fields, {
        "status": "BIT_EXACT_ORIENTATION",
        "shape": [148, 180],
        "files": files,
        "latitude_bit_identical": True,
        "longitude_bit_identical": True,
    }


def _reference_stage3(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    require(path.is_file(), f"missing ten-step reference: {path}")
    require(
        sha256(path) == EXPECTED_TEN_STEP_SHA256, "ten-step independent reference digest changed"
    )
    document = json.loads(path.read_text())
    require(document.get("status") == "LADDER_MEASURED", "ten-step reference is not measured")
    trajectory = document["candidate_trajectory"]
    rows = [
        row
        for row in trajectory["checkpoints"]
        if int(row["kt"]) == 10 and row["checkpoint"] == "stage3"
    ]
    require(len(rows) == 1, "ten-step reference has no unique kt10 stage3 row")
    return rows[0], trajectory["executed_initial_state_vs_nemo"]


def _finite_score(row: dict[str, object], label: str) -> None:
    count = int(row["count"])
    unequal = int(row["unequal"])
    require(count > 0 and 0 <= unequal <= count, f"{label}: invalid census")
    for key in ("max_abs", "mean_abs_over_unequal", "rms"):
        require(math.isfinite(float(row[key])), f"{label}: non-finite {key}")
    require(bool(row["bit_identical"]) == (unequal == 0), f"{label}: bit flag/count disagreement")


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    """Apply instrument predicates and classify the frozen scientific ones."""
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "initial-mode":
        report["initial_mode"] = "decision52_ssh_bridge"
    elif plant == "surface-frame-drop":
        report["surface_frames_consumed"] = 479
    elif plant == "ten-step-metric":
        report["ten_step_calibration"]["exact"] = False
    elif plant == "restart-orientation":
        report["terminal_restart"]["longitude_bit_identical"] = False
    elif plant == "terminal-ulp":
        report["terminal_restart"]["files"][0]["sha256"] = "0" * 64
    elif plant == "terminal-nonfinite":
        report["terminal"]["rows"]["T"]["max_abs"] = float("inf")

    require(report.get("claim_label") == "independent", "month claim is not independent")
    require(
        report.get("initial_mode") == "card_own_state" and report.get("decision52_bridge") is None,
        "month did not start from the card's own state",
    )
    require(
        report.get("execution") == "production-jit-cpu-fp64-x64-libm", "execution policy changed"
    )
    require(int(report.get("steps_completed", -1)) == STEPS, "month did not complete 240 steps")
    require(
        int(report.get("surface_frames_consumed", -1)) == 2 * STEPS,
        "month did not consume 480 rank-step surface frames",
    )
    require(
        tuple(report.get("unmeasured_features", ())) == ladder.EXPECTED_UNMEASURED,
        "unmeasured/sea-ice registry changed",
    )
    require(
        bool(report["ten_step_calibration"].get("exact")), "ten-step instrument calibration differs"
    )
    require(
        report["chlorophyll_clock"].get("status") == "RESOLVED_CENTRES_MATCH",
        "resolved chlorophyll clock was not validated",
    )
    require(
        report["chlorophyll_clock_retraction"].get("round66_metrics_reproduced") is False,
        "corrected chlorophyll clock unexpectedly reproduced round 66",
    )
    require(
        report["terminal_restart"].get("status") == "BIT_EXACT_ORIENTATION"
        and report["terminal_restart"].get("latitude_bit_identical") is True
        and report["terminal_restart"].get("longitude_bit_identical") is True,
        "terminal restart orientation is not exact",
    )
    for row in report["terminal_restart"]["files"]:
        require(row["sha256"] != "0" * 64, "terminal restart digest plant fired")

    rows = report["terminal"]["rows"]
    require(tuple(rows) == FIELD_ORDER, "terminal field registry changed")
    for name in FIELD_ORDER:
        _finite_score(rows[name], f"terminal {name}")
    require(
        all(np.isfinite(float(rows[name]["max_abs"])) for name in FIELD_ORDER),
        "terminal score is non-finite",
    )

    max_order = [row["field"] for row in report["terminal"]["ranking_by_max_abs"]]
    rms_order = [row["field"] for row in report["terminal"]["ranking_by_rms"]]
    require(
        set(max_order) == set(FIELD_ORDER) and set(rms_order) == set(FIELD_ORDER),
        "terminal ranking is incomplete",
    )
    predictions = {
        "all_five_non_bit": {
            "status": "CONFIRMED"
            if all(not rows[name]["bit_identical"] for name in FIELD_ORDER)
            else "REFUTED",
            "observed_bit_identical": [name for name in FIELD_ORDER if rows[name]["bit_identical"]],
        },
        "temperature_largest_max_abs": {
            "status": "CONFIRMED" if max_order[0] == "T" else "REFUTED",
            "observed_largest": max_order[0],
        },
        "original_round66_metric_reproduction": {
            "status": "REFUTED"
            if not report["chlorophyll_clock_retraction"]["round66_metrics_reproduced"]
            else "CONFIRMED"
        },
    }
    return {**report, "status": "PASS_INDEPENDENT_MONTH_RANKING", "prediction_ledger": predictions}


def run_month(
    deck_root: Path,
    surface_root: Path,
    month_root: Path,
    restart_ledger: Path,
    ten_step_root: Path,
    ten_step_reference: Path,
) -> dict[str, object]:
    """Advance the production ORCA2 card and return terminal score evidence."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    _, card = ladder.card_fields(deck_root)
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64/libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(
        tuple(card.unmeasured_features) == ladder.EXPECTED_UNMEASURED,
        "card unmeasured feature registry changed",
    )

    reference_stage3, reference_initial = _reference_stage3(ten_step_reference)
    chlorophyll_clock = validate_chlorophyll_clock(surface_root)
    oracle_entry = ladder.assemble_state_fields(ten_step_root, 1, stage=None)
    state = card.recipe.initial_state
    calibration_state = card.recipe.initial_state
    initial = ladder.compare_fields(ladder._candidate_fields(state), oracle_entry)
    require(
        initial == reference_initial,
        "independent initial state differs from the round-66 reference",
    )
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    started = time.time()
    ten_step_actual = None
    schema_calibration = None
    surface_schema_comparisons = 0
    old_only_surface_fields: list[str] | None = None
    consumed = 0
    for kt in range(1, STEPS + 1):
        fields = assemble_surface(surface_root, kt)
        consumed += 2
        freshwater, surface = ladder._surface_forcings(card, deck_root, fields, kt)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
        if kt <= 10:
            old_fields = ladder.assemble_surface_fields(ten_step_root, kt)
            comparisons, old_only = validate_surface_schema_calibration(
                fields, old_fields, kt=kt
            )
            surface_schema_comparisons += comparisons
            if old_only_surface_fields is None:
                old_only_surface_fields = old_only
            require(
                old_only == old_only_surface_fields,
                f"kt={kt}: legacy-only surface registry changed",
            )
            old_freshwater, old_surface = ladder._surface_forcings(card, deck_root, old_fields, kt)
            calibration_state = model.step(
                calibration_state,
                dt=card.dt_s,
                freshwater=old_freshwater,
                surface_forcing=old_surface,
            )
        if kt == 10:
            schema_calibration = ladder.compare_fields(
                ladder._candidate_fields(state),
                ladder._candidate_fields(calibration_state),
            )
            require(
                schema_calibration["first_non_bit_field"] is None,
                "new surface schema changes the corrected ten-step state",
            )
            oracle_stage3 = ladder.read_state_frame(
                ten_step_root / "oracle_stage_kt00000010_s3.bin",
                kt=10,
                stage=3,
            )
            ten_step_actual = {
                "kt": 10,
                "checkpoint": "stage3",
                **ladder.compare_fields(
                    ladder._rank0_fields(ladder._candidate_fields(state)),
                    oracle_stage3,
                ),
            }
        if kt % 40 == 0:
            print(
                f"MONTH_PROGRESS step={kt}/{STEPS} wall_s={time.time() - started:.1f}", flush=True
            )

    require(
        ten_step_actual is not None and schema_calibration is not None,
        "ten-step calibration was not executed",
    )
    calibration_exact = schema_calibration["first_non_bit_field"] is None
    round66_reproduced = ten_step_actual == reference_stage3
    require(
        not round66_reproduced,
        "source-corrected chlorophyll clock unexpectedly reproduced round 66",
    )
    candidate = ladder._candidate_fields(state)
    require(
        all(values.dtype == np.dtype(np.float64) for values in candidate.values()),
        "terminal candidate state is not fp64",
    )
    require(
        all(np.isfinite(values).all() for values in candidate.values()),
        "terminal candidate state contains non-finite values",
    )
    oracle, restart = read_terminal_restart(month_root, restart_ledger, card)
    comparison = ladder.compare_fields(candidate, oracle)
    rows = comparison["rows"]
    by_max = sorted(
        ({"field": name, "units": FIELD_UNITS[name], **rows[name]} for name in FIELD_ORDER),
        key=lambda row: (-float(row["max_abs"]), FIELD_ORDER.index(row["field"])),
    )
    by_rms = sorted(
        ({"field": name, "units": FIELD_UNITS[name], **rows[name]} for name in FIELD_ORDER),
        key=lambda row: (-float(row["rms"]), FIELD_ORDER.index(row["field"])),
    )
    return {
        "format": "nemo-testcase-l4-orca2-round71-independent-month-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed": STEPS,
        "dt_s": float(card.dt_s),
        "wall_seconds": time.time() - started,
        "surface_frames_consumed": consumed,
        "unmeasured_features": list(card.unmeasured_features),
        "initial_state_vs_nemo": initial,
        "chlorophyll_clock": chlorophyll_clock,
        "ten_step_calibration": {
            "exact": calibration_exact,
            "surface_schema_field_comparisons": surface_schema_comparisons,
            "legacy_only_surface_fields": old_only_surface_fields,
            "new_vs_old_schema": schema_calibration,
        },
        "chlorophyll_clock_retraction": {
            "round66_metrics_reproduced": round66_reproduced,
            "reference_path": str(ten_step_reference),
            "reference_sha256": sha256(ten_step_reference),
            "round66_stage3": reference_stage3,
            "corrected_stage3": ten_step_actual,
        },
        "terminal_restart": restart,
        "terminal": {
            "rows": rows,
            "ranking_by_max_abs": by_max,
            "ranking_by_rms": by_rms,
        },
        "compiled_citations": {
            "forcing_interpolation": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246"
            ),
            "restart_fields": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184"
            ),
            "restart_call": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-271"
            ),
        },
        "worktree": worktree_stamp(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--month-root", type=Path)
    parser.add_argument("--restart-ledger", type=Path)
    parser.add_argument("--ten-step-root", type=Path)
    parser.add_argument("--ten-step-reference", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(
                not any(
                    (
                        args.deck_root,
                        args.surface_root,
                        args.month_root,
                        args.restart_ledger,
                        args.ten_step_root,
                        args.ten_step_reference,
                    )
                ),
                "--classify-json cannot be combined with run inputs",
            )
            raw = json.loads(args.classify_json.read_text())
        else:
            require(
                args.plant == "none", "runtime plants use --classify-json after one measured run"
            )
            require(
                all(
                    (
                        args.deck_root,
                        args.surface_root,
                        args.month_root,
                        args.restart_ledger,
                        args.ten_step_root,
                        args.ten_step_reference,
                    )
                ),
                "run mode requires every record input",
            )
            raw = run_month(
                args.deck_root,
                args.surface_root,
                args.month_root,
                args.restart_ledger,
                args.ten_step_root,
                args.ten_step_reference,
            )
        report = classify(raw, plant=args.plant)
    except (
        GateError,
        ladder.GateError,
        surface_gate.GateError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_INDEPENDENT_MONTH_RANKING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
