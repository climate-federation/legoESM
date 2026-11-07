#!/usr/bin/env python3
"""Score the 2026-09-19 GYRE/DINO year-equivalence evidence.

This is a read-only comparator.  It never advances either model and never
invokes NEMO.  Every field is converted to fp64, differenced on NEMO's wet
mask, and reported both as wet RMS and as wet RMS divided by NEMO's spatial
standard deviation on the same mask and day.

GYRE uses the full NEMO wet population.  DINO follows the campaign's canonical
``twin_nemo_ts_maps.py`` population: the explicit one-ring ``[1:-1,1:-1]``
map interior.  The same crop is extended here to u/v, after dropping legoESM's
redundant west/south C-grid face exactly as ``day_gap_table.py`` does.

The operator's GYRE ``year_owners`` reference has daily restarts through day
30 only.  The independently archived ``year_fromrest`` record supplies the
monthly continuation; this comparator requires their shared day-30 restart to
be byte-identical before admitting that continuation.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TARGET_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
FIELDS = ("T", "S", "u", "v", "SSH")
GYRE_STEPS_PER_DAY = 6
DINO_STEPS_PER_DAY = 32
ONE_RING = (slice(1, -1), slice(1, -1))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_stamp() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).splitlines()
    require(not dirty, f"scorer checkout has dirty tracked files: {dirty}")
    return {"git_sha": head, "branch": branch, "dirty_tracked_files": 0}


def score_field(candidate: np.ndarray, reference: np.ndarray,
                wet: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    require(candidate.shape == reference.shape == wet.shape,
            f"shape mismatch {candidate.shape} / {reference.shape} / {wet.shape}")
    require(bool(wet.any()), "empty wet population")
    require(np.isfinite(candidate[wet]).all(), "non-finite candidate wet value")
    require(np.isfinite(reference[wet]).all(), "non-finite reference wet value")
    difference = candidate[wet] - reference[wet]
    rms = float(np.sqrt(np.mean(difference * difference)))
    nemo_std = float(np.std(reference[wet], ddof=0))
    require(nemo_std > 0.0, "NEMO spatial standard deviation is zero")
    return {
        "absolute_rms": rms,
        "nemo_spatial_std": nemo_std,
        "normalized_rms_over_nemo_std": rms / nemo_std,
        "wet_cells": int(wet.sum()),
    }


def planted_violation_self_test() -> dict[str, Any]:
    reference = np.asarray([0.0, 8.0, 2.0], dtype=np.float64)
    wet = np.asarray([True, False, True])
    baseline = score_field(reference.copy(), reference, wet)
    planted = reference.copy()
    planted[0] += 0.25
    moved = score_field(planted, reference, wet)
    dry = reference.copy()
    dry[1] += 99.0
    excluded = score_field(dry, reference, wet)
    require(baseline["absolute_rms"] == 0.0, "control baseline is not exact")
    require(np.isclose(moved["absolute_rms"], 0.25 / np.sqrt(2.0),
                       rtol=0.0, atol=1.0e-16),
            "wet-cell plant did not move RMS by the analytic amount")
    require(excluded == baseline, "dry-cell plant leaked into the score")
    return {
        "passed": True,
        "wet_cell_plant": 0.25,
        "expected_planted_rms": float(0.25 / np.sqrt(2.0)),
        "observed_planted_rms": moved["absolute_rms"],
        "dry_cell_excluded": True,
    }


def load_mesh_masks(path: Path, nlev: int) -> dict[str, np.ndarray]:
    import netCDF4

    with netCDF4.Dataset(path) as handle:
        masks = {}
        for key, variable in (("T", "tmask"), ("S", "tmask"),
                              ("u", "umask"), ("v", "vmask")):
            value = handle.variables[variable]
            require(value.dimensions ==
                    ("time_counter", "nav_lev", "y", "x"),
                    f"{path}:{variable} axes changed: {value.dimensions}")
            masks[key] = np.asarray(value[0], dtype=bool).transpose(1, 2, 0)[
                ..., :nlev]
    masks["SSH"] = masks["T"][..., 0]
    return masks


def gyre_restart_path(root: Path, day: int) -> Path:
    step = day * GYRE_STEPS_PER_DAY
    matches = sorted((root / "nemo_seed0").glob(
        f"*_{step:08d}_restart.nc"))
    require(len(matches) == 1,
            f"GYRE day {day}: expected one restart under {root}, got {matches}")
    return matches[0]


def load_gyre_reference(path: Path, nlev: int) -> dict[str, np.ndarray]:
    import netCDF4

    with netCDF4.Dataset(path) as handle:
        def xyz(name: str) -> np.ndarray:
            variable = handle.variables[name]
            require(variable.dimensions ==
                    ("time_counter", "nav_lev", "y", "x"),
                    f"{path}:{name} axes changed: {variable.dimensions}")
            return np.asarray(variable[0], dtype=np.float64).transpose(
                1, 2, 0)[..., :nlev]

        ssh = handle.variables["sshn"]
        require(ssh.dimensions == ("time_counter", "y", "x"),
                f"{path}:sshn axes changed: {ssh.dimensions}")
        return {"T": xyz("tn"), "S": xyz("sn"), "u": xyz("un"),
                "v": xyz("vn"),
                "SSH": np.asarray(ssh[0], dtype=np.float64)}


def load_gyre_candidate(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing GYRE snapshot {path}")
    with np.load(path, allow_pickle=False) as handle:
        require(all(str(np.asarray(handle[key]).dtype) == "float64"
                    for key in ("T", "S", "u", "v", "ssh")),
                f"GYRE snapshot is not fp64: {path}")
        return {"T": np.asarray(handle["T"], dtype=np.float64),
                "S": np.asarray(handle["S"], dtype=np.float64),
                "u": np.asarray(handle["u"], dtype=np.float64),
                "v": np.asarray(handle["v"], dtype=np.float64),
                "SSH": np.asarray(handle["ssh"], dtype=np.float64)}


def load_rebuilder():
    path = ROOT / "scripts/validate/ocean_fidelity/rebuild_nemo_restart.py"
    spec = importlib.util.spec_from_file_location("_year_equiv_rebuild", path)
    require(spec is not None and spec.loader is not None,
            f"cannot import restart rebuilder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.rebuild, path


def dino_restart_pattern(root: Path, day: int) -> str:
    return str(root / f"DINO_{day * DINO_STEPS_PER_DAY:08d}_restart*.nc")


def load_dino_reference(rebuild, root: Path,
                        day: int) -> tuple[dict[str, np.ndarray], list[Path]]:
    pattern = dino_restart_pattern(root, day)
    paths = [Path(item).resolve() for item in sorted(glob.glob(pattern))]
    require(bool(paths), f"DINO day {day}: no restart matches {pattern}")
    raw = rebuild(pattern, ["tn", "sn", "un", "vn", "sshn"])

    def xyz(name: str) -> np.ndarray:
        return np.nan_to_num(np.moveaxis(np.asarray(raw[name], dtype=np.float64),
                                         0, -1))

    return ({"T": xyz("tn"), "S": xyz("sn"), "u": xyz("un"),
             "v": xyz("vn"),
             "SSH": np.nan_to_num(np.asarray(raw["sshn"], dtype=np.float64))},
            paths)


def index_dino_snapshots(run_dir: Path) -> dict[int, Path]:
    paths = sorted((run_dir / "snapshots").glob("snapshot_*.npz"))
    require(bool(paths), f"no DINO snapshots under {run_dir}")
    indexed: dict[int, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            day_value = float(np.asarray(handle["time_days"]))
        day = int(round(day_value))
        require(abs(day_value - day) < 1.0e-9,
                f"non-integral snapshot day {day_value}: {path}")
        require(day not in indexed, f"duplicate DINO snapshot day {day}")
        indexed[day] = path.resolve()
    return indexed


def load_dino_candidate(path: Path, tmask_surface: np.ndarray
                        ) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        raw = {key: np.asarray(handle[key])
               for key in ("T", "S", "u", "v", "eta", "land_mask")}
    require(all(str(value.dtype) == "float64" for value in raw.values()),
            f"DINO snapshot is not all fp64: {path}")
    require(np.array_equal(raw["land_mask"] > 0.5, tmask_surface),
            f"DINO/NEMO surface masks differ: {path}")
    # legoESM stores one redundant face on the west/south side.
    return {"T": raw["T"].astype(np.float64),
            "S": raw["S"].astype(np.float64),
            "u": raw["u"][:, 1:, :].astype(np.float64),
            "v": raw["v"][1:, :, :].astype(np.float64),
            "SSH": raw["eta"].astype(np.float64)}


def crop_dino(value: np.ndarray) -> np.ndarray:
    return np.asarray(value)[ONE_RING[0], ONE_RING[1], ...]


def score_pair(candidate: dict[str, np.ndarray],
               reference: dict[str, np.ndarray],
               masks: dict[str, np.ndarray], *, crop: bool) -> dict[str, Any]:
    out = {}
    for field in FIELDS:
        left, right, wet = candidate[field], reference[field], masks[field]
        if crop:
            left, right, wet = crop_dino(left), crop_dino(right), crop_dino(wet)
        out[field] = score_field(left, right, wet)
    return out


def growth_fit(rows: dict[str, Any]) -> dict[str, Any]:
    def fit(days: list[int]) -> dict[str, Any]:
        x = np.log(np.asarray(days, dtype=np.float64))
        y = np.log(np.asarray(
            [rows[str(day)]["fields"]["T"]["absolute_rms"] for day in days],
            dtype=np.float64))
        slope, intercept = np.polyfit(x, y, 1)
        predicted = intercept + slope * x
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return {"days": days, "exponent_a": float(slope),
                "intercept_log": float(intercept),
                "r_squared_log_space": 1.0 - ss_res / ss_tot}

    all_days = list(TARGET_DAYS)
    late_days = [180, 240, 300, 360]
    values = [rows[str(day)]["fields"]["T"]["absolute_rms"]
              for day in all_days]
    peak_index = int(np.argmax(values))
    return {"all_registered_days": fit(all_days),
            "late_days_180_360": fit(late_days),
            "day360_over_day180": values[-1] / values[4],
            "day360_over_peak": values[-1] / values[peak_index],
            "peak_day": all_days[peak_index],
            "peak_rms": values[peak_index]}


def daily_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) >= 2, "daily diagnostic needs at least two days")
    ratios = []
    decreases = []
    for previous, current in zip(rows[:-1], rows[1:]):
        require(current["day"] == previous["day"] + 1,
                "daily ratio requested on non-consecutive days")
        ratio = current["T_rms"] / previous["T_rms"]
        ratios.append({"from_day": previous["day"], "to_day": current["day"],
                       "ratio": ratio})
        if current["T_rms"] < previous["T_rms"]:
            decreases.append({"from_day": previous["day"],
                              "to_day": current["day"],
                              "from_rms": previous["T_rms"],
                              "to_rms": current["T_rms"]})
    largest = max(ratios, key=lambda row: row["ratio"])
    return {"rows": rows, "largest_consecutive_day_ratio": largest,
            "decrease_count": len(decreases),
            "first_decrease": decreases[0] if decreases else None,
            "monotone_non_decreasing": not decreases}


def parse_hash_manifest(path: Path) -> dict[Path, str]:
    out = {}
    for raw in path.read_text().splitlines():
        digest, filename = raw.split(maxsplit=1)
        out[Path(filename).resolve()] = digest
    return out


def aggregate_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gyre-lego-root", type=Path)
    parser.add_argument("--gyre-early-nemo-root", type=Path)
    parser.add_argument("--gyre-year-nemo-root", type=Path)
    parser.add_argument("--gyre-mesh", type=Path)
    parser.add_argument("--dino-twin-dir", type=Path)
    parser.add_argument("--dino-standalone-dir", type=Path)
    parser.add_argument("--dino-early-nemo-root", type=Path)
    parser.add_argument("--dino-traj-root", type=Path)
    parser.add_argument("--dino-continuation-root", type=Path)
    parser.add_argument("--dino-day360-root", type=Path)
    parser.add_argument("--dino-mesh", type=Path)
    parser.add_argument("--dino-twin-hash-manifest", type=Path)
    parser.add_argument("--standalone-prior-day360-json", type=Path)
    parser.add_argument("--gyre-noise-floor-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(planted_violation_self_test(), indent=2,
                         sort_keys=True))
        return 0
    required = [name for name, value in vars(args).items()
                if name != "self_test" and value is None]
    if required:
        parser.error("required unless --self-test: " + ", ".join(required))

    controls: dict[str, Any] = {
        "planted_violation": planted_violation_self_test()}
    scorer_stamp = git_stamp()
    rebuild, rebuild_path = load_rebuilder()

    # GYRE: admit the monthly continuation only after the shared endpoint is
    # proven byte-identical to the requested daily record.
    gyre_early_day30 = gyre_restart_path(args.gyre_early_nemo_root, 30)
    gyre_year_day30 = gyre_restart_path(args.gyre_year_nemo_root, 30)
    gyre_early_hash = sha256(gyre_early_day30)
    gyre_year_hash = sha256(gyre_year_day30)
    require(gyre_early_hash == gyre_year_hash,
            "GYRE day-30 references are not byte-identical")
    controls["gyre_reference_continuity"] = {
        "passed": True, "day": 30,
        "requested_daily_path": str(gyre_early_day30),
        "continuation_path": str(gyre_year_day30),
        "shared_sha256": gyre_early_hash,
    }
    gyre_masks = load_mesh_masks(args.gyre_mesh, 30)
    gyre_rows: dict[str, Any] = {}
    gyre_snapshot_hashes: dict[str, str] = {}
    gyre_reference_hashes: dict[str, str] = {}
    for day in TARGET_DAYS:
        snapshot = (args.gyre_lego_root / "lego_seed0_year" /
                    f"day{day:03d}.npz").resolve()
        reference_path = gyre_restart_path(
            args.gyre_early_nemo_root if day == 30 else args.gyre_year_nemo_root,
            day).resolve()
        fields = score_pair(load_gyre_candidate(snapshot),
                            load_gyre_reference(reference_path, 30),
                            gyre_masks, crop=False)
        gyre_snapshot_hashes[str(day)] = sha256(snapshot)
        gyre_reference_hashes[str(day)] = sha256(reference_path)
        gyre_rows[str(day)] = {
            "classification": "MEASURED", "day": day, "fields": fields,
            "candidate_path": str(snapshot),
            "candidate_sha256": gyre_snapshot_hashes[str(day)],
            "nemo_reference_path": str(reference_path),
            "nemo_reference_sha256": gyre_reference_hashes[str(day)],
        }

    gyre_daily_rows = []
    for day in range(1, 31):
        snapshot = (args.gyre_lego_root / "lego_seed0_year" /
                    f"day{day:03d}.npz").resolve()
        reference_path = gyre_restart_path(args.gyre_early_nemo_root, day)
        candidate = load_gyre_candidate(snapshot)
        reference = load_gyre_reference(reference_path, 30)
        gyre_daily_rows.append({
            "day": day,
            "T_rms": score_field(candidate["T"], reference["T"],
                                 gyre_masks["T"])["absolute_rms"],
            "candidate_sha256": sha256(snapshot),
            "nemo_reference_sha256": sha256(reference_path),
        })

    # DINO references.  RUN_VERDICT360_M0 is the continuation after day 180;
    # its shared d180 state and its d360 state must equal the named canonical
    # RUN_TRAJ / RUN_FROMREST_Y1 fields bit for bit.
    dino_masks = load_mesh_masks(args.dino_mesh, 36)
    traj180, traj180_paths = load_dino_reference(
        rebuild, args.dino_traj_root, 180)
    continuation180, continuation180_paths = load_dino_reference(
        rebuild, args.dino_continuation_root, 180)
    y1360, y1360_paths = load_dino_reference(
        rebuild, args.dino_day360_root, 360)
    continuation360, continuation360_paths = load_dino_reference(
        rebuild, args.dino_continuation_root, 360)
    dino_continuity = {}
    for label, left, right in (
            ("day180_RUN_TRAJ_vs_continuation", traj180, continuation180),
            ("day360_RUN_FROMREST_Y1_vs_continuation", y1360,
             continuation360)):
        exact = {field: bool(np.array_equal(left[field], right[field]))
                 for field in FIELDS}
        require(all(exact.values()), f"DINO reference continuity failed: {label}")
        dino_continuity[label] = exact
    controls["dino_reference_continuity"] = {
        "passed": True, "field_bit_identity": dino_continuity,
        "day180_RUN_TRAJ_files_aggregate_sha256": aggregate_hash(traj180_paths),
        "day180_continuation_files_aggregate_sha256": aggregate_hash(
            continuation180_paths),
        "day360_RUN_FROMREST_Y1_files_aggregate_sha256": aggregate_hash(
            y1360_paths),
        "day360_continuation_files_aggregate_sha256": aggregate_hash(
            continuation360_paths),
    }

    twin_index = index_dino_snapshots(args.dino_twin_dir)
    standalone_index = index_dino_snapshots(args.dino_standalone_dir)
    for day in (*TARGET_DAYS, 1, 2, 3, 4, 5):
        require(day in twin_index, f"DINO twin missing day {day}")
    for day in TARGET_DAYS:
        require(day in standalone_index, f"DINO standalone missing day {day}")

    hash_manifest = parse_hash_manifest(args.dino_twin_hash_manifest)
    checked_manifest = {}
    for day in sorted(set(TARGET_DAYS) | {1, 2, 3, 4, 5}):
        snapshot = twin_index[day]
        require(snapshot in hash_manifest,
                f"DINO twin hash manifest omits {snapshot}")
        actual = sha256(snapshot)
        require(actual == hash_manifest[snapshot],
                f"DINO twin hash mismatch: {snapshot}")
        checked_manifest[str(day)] = actual
    controls["dino_twin_hash_manifest"] = {
        "passed": True, "manifest": str(args.dino_twin_hash_manifest),
        "manifest_sha256": sha256(args.dino_twin_hash_manifest),
        "selected_snapshot_sha256": checked_manifest,
    }

    dino_twin_rows: dict[str, Any] = {}
    dino_standalone_rows: dict[str, Any] = {}
    dino_reference_provenance: dict[str, Any] = {}
    for day in TARGET_DAYS:
        if day <= 180:
            reference_root = args.dino_traj_root
        elif day < 360:
            reference_root = args.dino_continuation_root
        else:
            reference_root = args.dino_day360_root
        if day == 180:
            reference, reference_paths = traj180, traj180_paths
        elif day == 360:
            reference, reference_paths = y1360, y1360_paths
        else:
            reference, reference_paths = load_dino_reference(
                rebuild, reference_root, day)
        dino_reference_provenance[str(day)] = {
            "root": str(reference_root),
            "paths": [str(path) for path in reference_paths],
            "aggregate_sha256": aggregate_hash(reference_paths),
        }
        for label, index, destination, classification in (
                ("DINO_TWIN", twin_index, dino_twin_rows, "MEASURED"),
                ("DINO_STANDALONE", standalone_index, dino_standalone_rows,
                 "REPRODUCED-FROM-ARTIFACT")):
            snapshot = index[day]
            candidate = load_dino_candidate(snapshot, dino_masks["SSH"])
            destination[str(day)] = {
                "classification": classification, "day": day,
                "fields": score_pair(candidate, reference, dino_masks,
                                     crop=True),
                "candidate_path": str(snapshot),
                "candidate_sha256": sha256(snapshot),
                "nemo_reference": dino_reference_provenance[str(day)],
                "population": "DINO canonical one-ring [1:-1,1:-1] wet mask",
                "case": label,
            }

    dino_daily_rows = []
    for day in range(1, 6):
        reference, paths = load_dino_reference(
            rebuild, args.dino_early_nemo_root, day)
        snapshot = twin_index[day]
        candidate = load_dino_candidate(snapshot, dino_masks["SSH"])
        row = score_pair(candidate, reference, dino_masks, crop=True)
        dino_daily_rows.append({
            "day": day, "T_rms": row["T"]["absolute_rms"],
            "candidate_sha256": sha256(snapshot),
            "nemo_reference_files_aggregate_sha256": aggregate_hash(paths),
        })

    prior = json.loads(args.standalone_prior_day360_json.read_text())
    prior_t = float(prior["statistics"]["T3D"]["rms_difference"])
    reproduced_t = dino_standalone_rows["360"]["fields"]["T"]["absolute_rms"]
    require(abs(prior_t - reproduced_t) <= 5.0e-15,
            f"standalone day360 did not reproduce prior scorer: "
            f"{reproduced_t} vs {prior_t}")
    controls["standalone_prior_day360_reproduction"] = {
        "passed": True,
        "prior_json": str(args.standalone_prior_day360_json),
        "prior_json_sha256": sha256(args.standalone_prior_day360_json),
        "prior_T3D_rms": prior_t,
        "reproduced_T3D_rms": reproduced_t,
        "absolute_difference": abs(prior_t - reproduced_t),
    }

    gyre_floor_source = json.loads(args.gyre_noise_floor_json.read_text())
    gyre_nemo_t_floor = {
        str(day): float(gyre_floor_source["rows"]["T3D"]["days"][str(day)]
                        ["spread_nemo"])
        for day in TARGET_DAYS
    }

    gyre_manifest = args.gyre_lego_root / "lego_seed0_year/manifest.json"
    twin_metadata = args.dino_twin_dir / "run_metadata.json"
    standalone_metadata = args.dino_standalone_dir / "run_metadata.json"
    result = {
        "schema": "year-equivalence-gyre-vs-dino-2026-09-19-v1",
        "date": "2026-09-19",
        "metric": {
            "absolute": "fp64 wet RMS(candidate - NEMO)",
            "normalized": "absolute / population std(NEMO), ddof=0",
            "GYRE_population": "full NEMO tmask/umask/vmask wet cells",
            "DINO_population": ("canonical twin_nemo_ts_maps.py one-ring "
                                "[1:-1,1:-1] wet cells; extended to u/v"),
        },
        "target_days": list(TARGET_DAYS),
        "cases": {
            "GYRE": {"classification": "MEASURED", "rows": gyre_rows,
                     "growth": growth_fit(gyre_rows)},
            "DINO_TWIN": {"classification": "MEASURED",
                          "rows": dino_twin_rows,
                          "growth": growth_fit(dino_twin_rows)},
            "DINO_STANDALONE": {
                "classification": "REPRODUCED-FROM-ARTIFACT; not rerun",
                "rows": dino_standalone_rows,
                "growth": growth_fit(dino_standalone_rows)},
        },
        "daily_diagnostics": {
            "GYRE": {**daily_diagnostics(gyre_daily_rows),
                     "coverage": "exact consecutive NEMO restarts, days 1-30"},
            "DINO_TWIN": {**daily_diagnostics(dino_daily_rows),
                          "coverage": ("exact consecutive NEMO restarts, days "
                                       "1-5; no later daily NEMO restarts exist")},
        },
        "noise_floors": {
            "GYRE_T3D_NEMO_vs_NEMO_max_pairwise_rms": gyre_nemo_t_floor,
            "GYRE_source": str(args.gyre_noise_floor_json),
            "GYRE_source_sha256": sha256(args.gyre_noise_floor_json),
            "DINO_T3D_NEMO_vs_NEMO": "UNMEASURED in the named state artifacts",
            "DINO_note": ("Existing DINO ensemble floors are different "
                          "statistics and are not relabelled as T3D state RMS."),
        },
        "controls": controls,
        "provenance": {
            "scorer_repo": scorer_stamp,
            "scorer_path": str(Path(__file__).resolve()),
            "scorer_sha256": sha256(Path(__file__).resolve()),
            "command": [sys.executable, str(Path(__file__).resolve()),
                        *(argv if argv is not None else sys.argv[1:])],
            "restart_rebuilder": str(rebuild_path),
            "restart_rebuilder_sha256": sha256(rebuild_path),
            "gyre_manifest": str(gyre_manifest),
            "gyre_manifest_sha256": sha256(gyre_manifest),
            "gyre_manifest_record": json.loads(gyre_manifest.read_text()),
            "dino_twin_metadata": str(twin_metadata),
            "dino_twin_metadata_sha256": sha256(twin_metadata),
            "dino_twin_metadata_record": json.loads(twin_metadata.read_text()),
            "dino_standalone_metadata": str(standalone_metadata),
            "dino_standalone_metadata_sha256": sha256(standalone_metadata),
            "dino_standalone_metadata_record": json.loads(
                standalone_metadata.read_text()),
            "dino_reference_by_day": dino_reference_provenance,
            "gyre_mesh": str(args.gyre_mesh),
            "gyre_mesh_sha256": sha256(args.gyre_mesh),
            "dino_mesh": str(args.dino_mesh),
            "dino_mesh_sha256": sha256(args.dino_mesh),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} sha256={sha256(args.output)}")
    for case, record in result["cases"].items():
        print(case, "day360 T rms",
              record["rows"]["360"]["fields"]["T"]["absolute_rms"])
    print("GYRE P4", result["daily_diagnostics"]["GYRE"]
          ["largest_consecutive_day_ratio"])
    print("DINO P4", result["daily_diagnostics"]["DINO_TWIN"]
          ["largest_consecutive_day_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
