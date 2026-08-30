#!/usr/bin/env python3
"""Produce provenance-bound twin-vs-NEMO SSH/T/S maps and statistics.

The DINO climate arms and rebuilt NEMO restart are full ``(j,i)=(199,52)``
arrays. This map diagnostic deliberately scores the common one-ring interior
``[1:-1,1:-1] -> (197,50)`` used by the review-inbox draft. The crop is
explicit and asserted here; it is *not* presented as NEMO file-halo removal.
``packages/ocean/legoesm/ocean/fidelity/nemo_io.py:3-18`` is the governing
convention: DINO 5.x files are canonical halo-free arrays and 3-D NEMO fields
move from ``(k,j,i)`` to ``(j,i,k)`` without vertical reversal. Full-frame
gates remain the authority for boundary-ring claims.

Example, from the repository root::

  python scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py \
    --arm-npz /tmp/dino-climate-rebattery-01a04e34/arms/climate_a.npz \
    --day 360 --nemo-kt 17280 \
    --output-dir /tmp/dino-twin-nemo-ts-maps-day360

The JSON sidecar contains every printed statistic and provenance stamp.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/legoesm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
DEFAULT_NEMO_RUN = NEMO_ROOT / "cfgs/DINO/RUN_VERDICT360_M0"
DEFAULT_MESH = NEMO_ROOT / "cfgs/DINO/RUN_TRAJ/mesh_mask.nc"
FULL_HORIZONTAL_SHAPE = (199, 52)
MAP_HORIZONTAL_SHAPE = (197, 50)
ONE_RING = (slice(1, -1), slice(1, -1))
ALIGNMENT_CITATION = (
    "packages/ocean/legoesm/ocean/fidelity/nemo_io.py:3-18; "
    "DINO 5.x inputs are canonical halo-free (199,52), while this map-only "
    "diagnostic explicitly crops [1:-1,1:-1] to the registered (197,50) "
    "review frame"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    require(key in data.files, f"arm is missing required stamp {key}")
    value = np.asarray(data[key])
    require(value.shape == (), f"arm stamp {key} must be scalar, got {value.shape}")
    return value.item()


def git_stamp() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).splitlines()
    require(not dirty, "tracked comparator checkout must be clean")
    return {"git_sha": head, "dirty_tracked_files": 0}


def nemo_git_stamp() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(NEMO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(NEMO_ROOT), "status", "--porcelain",
             "--untracked-files=no"], text=True).splitlines()
        return {"git_sha": head, "dirty_tracked_files": len(dirty)}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_sha": "UNAVAILABLE", "dirty_tracked_files": "UNAVAILABLE"}


def load_rebuilder():
    path = ROOT / "scripts/validate/ocean_fidelity/rebuild_nemo_restart.py"
    spec = importlib.util.spec_from_file_location("_ts_map_restart_rebuild", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.rebuild


def crop_one_ring(name: str, array: np.ndarray) -> np.ndarray:
    """Assert the full frame and return the registered map-only interior."""
    value = np.asarray(array)
    require(value.ndim in (2, 3), f"{name}: expected 2-D/3-D, got {value.shape}")
    require(value.shape[:2] == FULL_HORIZONTAL_SHAPE,
            f"{name}: expected full horizontal {FULL_HORIZONTAL_SHAPE}, "
            f"got {value.shape[:2]}")
    cropped = value[ONE_RING[0], ONE_RING[1], ...]
    require(cropped.shape[:2] == MAP_HORIZONTAL_SHAPE,
            f"{name}: one-ring crop produced {cropped.shape[:2]}, expected "
            f"{MAP_HORIZONTAL_SHAPE}")
    return cropped


def error_stats(candidate: np.ndarray, reference: np.ndarray,
                wet: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    wet = np.asarray(wet, dtype=bool)
    require(candidate.shape == reference.shape == wet.shape,
            f"stats shape mismatch: {candidate.shape}, {reference.shape}, {wet.shape}")
    require(bool(np.any(wet)), "wet mask is empty")
    require(np.isfinite(candidate[wet]).all() and np.isfinite(reference[wet]).all(),
            "non-finite scored wet value")
    difference = candidate[wet].astype(np.float64) - reference[wet].astype(np.float64)
    return {
        "wet_count": int(wet.sum()),
        "total_count": int(wet.size),
        "wet_fraction": float(wet.mean()),
        "mean_difference": float(np.mean(difference)),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "max_abs_difference": float(np.max(np.abs(difference))),
    }


def planted_violation_self_test() -> dict[str, Any]:
    """Perturb one wet cell and prove the scored statistics move."""
    reference = np.zeros((3, 4, 2), dtype=np.float64)
    candidate = reference.copy()
    wet = np.zeros_like(reference, dtype=bool)
    wet[1, 2, 1] = True
    wet[2, 1, 0] = True
    before = error_stats(candidate, reference, wet)
    candidate[1, 2, 1] = 0.25
    after = error_stats(candidate, reference, wet)
    require(before["max_abs_difference"] == 0.0,
            "planted control baseline is not exact")
    require(after["max_abs_difference"] == 0.25,
            "one-wet-cell plant was not recovered")
    require(after["rms_difference"] > before["rms_difference"],
            "one-wet-cell plant did not move RMS")
    dry_candidate = reference.copy()
    dry_candidate[0, 0, 0] = 9.0
    dry = error_stats(dry_candidate, reference, wet)
    require(dry == before, "dry-cell plant leaked into wet statistics")
    return {
        "passed": True,
        "planted_index_jik": [1, 2, 1],
        "planted_value": 0.25,
        "before": before,
        "after": after,
        "dry_cell_excluded": True,
    }


def locate_abs_argmax(candidate: np.ndarray, reference: np.ndarray,
                      wet: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      depth: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    wet = np.asarray(wet, dtype=bool)
    require(candidate.ndim == 3 and candidate.shape == reference.shape == wet.shape,
            "3-D argmax fields/mask are not aligned")
    require(lat.shape == lon.shape == candidate.shape[:2],
            "argmax latitude/longitude are not aligned")
    require(depth.shape in (candidate.shape, (candidate.shape[-1],)),
            f"argmax depth shape {depth.shape} is neither 3-D nor 1-D")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    score = np.where(wet, np.abs(delta), -np.inf)
    require(np.isfinite(score).any(), "argmax wet population is empty")
    j, i, k = (int(v) for v in np.unravel_index(np.argmax(score), score.shape))
    depth_m = float(depth[j, i, k] if depth.ndim == 3 else depth[k])
    return {
        "index_jik_map_zero_based": [j, i, k],
        "index_jik_full_zero_based": [j + 1, i + 1, k],
        "latitude_deg_north": float(lat[j, i]),
        "longitude_deg_east": float(lon[j, i]),
        "depth_m": depth_m,
        "candidate_value": float(candidate[j, i, k]),
        "nemo_value": float(reference[j, i, k]),
        "difference": float(delta[j, i, k]),
        "abs_difference": float(abs(delta[j, i, k])),
    }


def plot_row(axes, candidate: np.ndarray, reference: np.ndarray,
             wet: np.ndarray, lon: np.ndarray, lat: np.ndarray,
             name: str, unit: str, cmap: str) -> dict[str, float | int]:
    stats = error_stats(candidate, reference, wet)
    c = np.where(wet, candidate, np.nan)
    r = np.where(wet, reference, np.nan)
    delta = c - r
    vmin, vmax = np.nanpercentile(r, (1.0, 99.0))
    dlim = max(float(np.nanmax(np.abs(delta))), 1.0e-30)
    panels = (
        (axes[0], c, f"legoESM {name}", dict(cmap=cmap, vmin=vmin, vmax=vmax)),
        (axes[1], r, f"NEMO {name}", dict(cmap=cmap, vmin=vmin, vmax=vmax)),
        (axes[2], delta,
         (f"difference max|d|={stats['max_abs_difference']:.3e} "
          f"rms={stats['rms_difference']:.3e} {unit}"),
         dict(cmap="RdBu_r", vmin=-dlim, vmax=dlim)),
    )
    for axis, field, title, kwargs in panels:
        image = axis.pcolormesh(lon, lat, field, shading="auto", **kwargs)
        plt.colorbar(image, ax=axis, shrink=0.85)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("longitude [deg E]")
        axis.set_ylabel("latitude [deg N]")
    return stats


def save_figures(output_dir: Path, day: int, fields: dict[str, np.ndarray],
                 masks: dict[str, np.ndarray], lon: np.ndarray, lat: np.ndarray,
                 depths: np.ndarray) -> dict[str, str]:
    surface_path = output_dir / f"surface_day{day}.png"
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    plot_row(axes[0], fields["SSH_arm"], fields["SSH_nemo"], masks["surface"],
             lon, lat, f"SSH day {day}", "m", "viridis")
    plot_row(axes[1], fields["T_arm"][..., 0], fields["T_nemo"][..., 0],
             masks["surface"], lon, lat, "SST", "degC", "turbo")
    plot_row(axes[2], fields["S_arm"][..., 0], fields["S_nemo"][..., 0],
             masks["surface"], lon, lat, "SSS", "psu", "viridis")
    fig.suptitle(f"DINO twin vs NEMO 5.0.2 — day {day} one-ring map interior")
    fig.savefig(surface_path, dpi=140)
    plt.close(fig)

    target_depths = (300.0, 1000.0)
    levels = [int(np.argmin(np.abs(depths - target))) for target in target_depths]
    subsurface_path = output_dir / f"subsurface_day{day}.png"
    fig, axes = plt.subplots(4, 3, figsize=(13, 15), constrained_layout=True)
    rows = (
        ("T", levels[0], "degC", "turbo"),
        ("T", levels[1], "degC", "turbo"),
        ("S", levels[0], "psu", "viridis"),
        ("S", levels[1], "psu", "viridis"),
    )
    for row, (name, level, unit, cmap) in enumerate(rows):
        plot_row(
            axes[row], fields[f"{name}_arm"][..., level],
            fields[f"{name}_nemo"][..., level], masks["three_d"][..., level],
            lon, lat, f"{name} at {depths[level]:.1f} m", unit, cmap)
    fig.suptitle(f"DINO twin vs NEMO — day {day} subsurface one-ring interior")
    fig.savefig(subsurface_path, dpi=140)
    plt.close(fig)
    return {surface_path.name: sha256(surface_path),
            subsurface_path.name: sha256(subsurface_path)}


def print_report(result: dict[str, Any]) -> None:
    provenance = result["provenance"]
    print(f"PROVENANCE git_sha={provenance['comparator_repo']['git_sha']} "
          f"dirty_tracked_files={provenance['comparator_repo']['dirty_tracked_files']}")
    print(f"PROVENANCE arm={provenance['arm_path']} "
          f"sha256={provenance['input_sha256'][provenance['arm_path']]}")
    print(f"PROVENANCE mesh={provenance['mesh_path']} "
          f"sha256={provenance['input_sha256'][provenance['mesh_path']]}")
    print(f"PROVENANCE restart_pattern={provenance['restart_pattern']} "
          f"tiles={len(provenance['restart_paths'])}")
    for path in provenance["restart_paths"]:
        print(f"PROVENANCE restart={path} "
              f"sha256={provenance['input_sha256'][path]}")
    print(f"ALIGNMENT {result['alignment']['citation']}")
    print("field  wet/total   wet_fraction arm_dtype nemo_dtype "
          "max_abs rms mean_difference")
    for name, row in result["statistics"].items():
        print(f"{name:<5} {row['wet_count']:>6}/{row['total_count']:<6} "
              f"{row['wet_fraction']:.6f} {row['arm_dtype']:<9} "
              f"{row['nemo_dtype']:<10} {row['max_abs_difference']:.9e} "
              f"{row['rms_difference']:.9e} {row['mean_difference']:.9e}")
    peak = result["three_dimensional_temperature_argmax"]
    print("3D_T_ARGMAX "
          f"map_jik={tuple(peak['index_jik_map_zero_based'])} "
          f"full_jik={tuple(peak['index_jik_full_zero_based'])} "
          f"lat={peak['latitude_deg_north']:.9f} "
          f"lon={peak['longitude_deg_east']:.9f} "
          f"depth_m={peak['depth_m']:.9f} "
          f"difference_C={peak['difference']:.12e} "
          f"abs_C={peak['abs_difference']:.12e}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm_path = args.arm_npz.resolve()
    output_dir = args.output_dir.resolve()
    nemo_run = args.nemo_run.resolve()
    mesh_path = args.mesh_mask.resolve()
    require(arm_path.is_file(), f"missing arm {arm_path}")
    require(mesh_path.is_file(), f"missing mesh {mesh_path}")
    require(nemo_run.is_dir(), f"missing NEMO run directory {nemo_run}")
    require(not output_dir.exists() or not any(output_dir.iterdir()),
            f"output directory is not fresh/empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = git_stamp()
    restart_pattern = str(nemo_run / f"DINO_{args.nemo_kt:08d}_restart*.nc")
    restart_paths = [Path(path).resolve() for path in sorted(glob.glob(restart_pattern))]
    require(restart_paths, f"no restart tiles match {restart_pattern}")
    rebuild = load_rebuilder()
    raw = rebuild(restart_pattern, ["tn", "sn", "sshn"])
    require(set(raw) == {"tn", "sn", "sshn"},
            f"incomplete rebuilt comparator: {sorted(raw)}")
    nemo_t_full = np.moveaxis(np.asarray(raw["tn"]), 0, -1)
    nemo_s_full = np.moveaxis(np.asarray(raw["sn"]), 0, -1)
    nemo_ssh_full = np.asarray(raw["sshn"])
    require(np.isfinite(nemo_t_full).all() and np.isfinite(nemo_s_full).all()
            and np.isfinite(nemo_ssh_full).all(), "stitched NEMO state is incomplete")

    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask

    mesh_full = read_nemo_mesh_mask(str(mesh_path), nn_hls=0)
    require(mesh_full.tmask.shape[:2] == FULL_HORIZONTAL_SHAPE,
            f"mesh full shape is {mesh_full.tmask.shape[:2]}, expected "
            f"{FULL_HORIZONTAL_SHAPE}")
    with np.load(arm_path, allow_pickle=False) as arm:
        keys = {name: f"{name}3d_day{args.day}" for name in ("T", "S", "eta")}
        for key in keys.values():
            require(key in arm.files, f"arm is missing requested day field {key}")
        require(bool(scalar(arm, "stable")), "arm producer marked unstable")
        require(int(scalar(arm, "producer_dirty_tracked_files")) == 0,
                "arm records a dirty producer")
        require(str(scalar(arm, "control_dtype")) == "float64",
                "arm control is not fp64")
        arm_t_full = np.asarray(arm[keys["T"]])
        arm_s_full = np.asarray(arm[keys["S"]])
        arm_ssh_full = np.asarray(arm[keys["eta"]])
        arm_land_full = np.asarray(arm["land_mask"])
        arm_stamps = {
            "producer_git_sha": str(scalar(arm, "producer_git_sha")),
            "producer_dirty_tracked_files": int(
                scalar(arm, "producer_dirty_tracked_files")),
            "codex_session_id": str(scalar(arm, "codex_session_id")),
            "control_dtype": str(scalar(arm, "control_dtype")),
            "field_dtypes": {key: str(np.asarray(arm[key]).dtype)
                             for key in keys.values()},
        }
    for name, value in (("arm T", arm_t_full), ("arm S", arm_s_full),
                        ("NEMO T", nemo_t_full), ("NEMO S", nemo_s_full)):
        require(value.shape == (199, 52, 36),
                f"{name}: expected (199,52,36), got {value.shape}")
    for name, value in (("arm SSH", arm_ssh_full),
                        ("NEMO SSH", nemo_ssh_full),
                        ("arm land mask", arm_land_full)):
        require(value.shape == FULL_HORIZONTAL_SHAPE,
                f"{name}: expected {FULL_HORIZONTAL_SHAPE}, got {value.shape}")
    require(all(dtype == "float64" for dtype in arm_stamps["field_dtypes"].values()),
            f"arm scored fields are not all fp64: {arm_stamps['field_dtypes']}")

    fields = {
        "T_arm": crop_one_ring("arm T", arm_t_full),
        "S_arm": crop_one_ring("arm S", arm_s_full),
        "SSH_arm": crop_one_ring("arm SSH", arm_ssh_full),
        "T_nemo": crop_one_ring("NEMO T", nemo_t_full),
        "S_nemo": crop_one_ring("NEMO S", nemo_s_full),
        "SSH_nemo": crop_one_ring("NEMO SSH", nemo_ssh_full),
    }
    tmask = crop_one_ring("mesh tmask", np.asarray(mesh_full.tmask)) > 0.5
    lat = crop_one_ring("mesh latitude", np.asarray(mesh_full.gphit))
    lon = crop_one_ring("mesh longitude", np.asarray(mesh_full.glamt))
    arm_land = crop_one_ring("arm land mask", arm_land_full) > 0.5
    require(np.array_equal(arm_land, tmask[..., 0]),
            "arm and NEMO surface wet masks differ on map interior")
    masks = {"surface": tmask[..., 0], "three_d": tmask}
    depth = (crop_one_ring("mesh gdept_0", np.asarray(mesh_full.gdept_0))
             if mesh_full.gdept_0 is not None
             else np.asarray(mesh_full.gdept_1d))

    stats: dict[str, dict[str, Any]] = {}
    rows = {
        "SSH": (fields["SSH_arm"], fields["SSH_nemo"], masks["surface"]),
        "SST": (fields["T_arm"][..., 0], fields["T_nemo"][..., 0],
                masks["surface"]),
        "SSS": (fields["S_arm"][..., 0], fields["S_nemo"][..., 0],
                masks["surface"]),
        "T3D": (fields["T_arm"], fields["T_nemo"], masks["three_d"]),
        "S3D": (fields["S_arm"], fields["S_nemo"], masks["three_d"]),
    }
    for name, (candidate, reference, wet) in rows.items():
        stats[name] = {
            **error_stats(candidate, reference, wet),
            "arm_dtype": str(candidate.dtype),
            "nemo_dtype": str(reference.dtype),
        }
    argmax = locate_abs_argmax(
        fields["T_arm"], fields["T_nemo"], masks["three_d"], lat, lon, depth)
    controls = {"one_wet_cell_plant": planted_violation_self_test()}
    figure_hashes = save_figures(
        output_dir, args.day, fields, masks, lon, lat,
        np.asarray(mesh_full.gdept_1d))

    input_paths = [arm_path, mesh_path, *restart_paths]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    script_path = Path(__file__).resolve()
    result = {
        "schema": "dino-twin-nemo-ts-maps-v1",
        "day": args.day,
        "nemo_kt": args.nemo_kt,
        "alignment": {
            "full_horizontal_shape": list(FULL_HORIZONTAL_SHAPE),
            "one_ring_slice": "[1:-1,1:-1]",
            "map_horizontal_shape": list(MAP_HORIZONTAL_SHAPE),
            "citation": ALIGNMENT_CITATION,
            "scope": "map-only interior; no full-frame boundary-ring claim",
        },
        "statistics": stats,
        "wet_mask_statistics": {
            "surface_wet": int(masks["surface"].sum()),
            "surface_total": int(masks["surface"].size),
            "three_dimensional_wet": int(masks["three_d"].sum()),
            "three_dimensional_total": int(masks["three_d"].size),
        },
        "three_dimensional_temperature_argmax": argmax,
        "controls": controls,
        "provenance": {
            "comparator_repo": repo,
            "nemo_repo": nemo_git_stamp(),
            "script_path": str(script_path),
            "script_sha256": sha256(script_path),
            "arm_path": str(arm_path),
            "mesh_path": str(mesh_path),
            "nemo_run": str(nemo_run),
            "restart_pattern": restart_pattern,
            "restart_paths": [str(path) for path in restart_paths],
            "input_sha256": input_hashes,
            "arm_stamps": arm_stamps,
            "numpy_version": np.__version__,
            "scored_dtypes": {name: {"arm": row["arm_dtype"],
                                      "nemo": row["nemo_dtype"]}
                              for name, row in stats.items()},
        },
        "artifacts_sha256": figure_hashes,
    }
    output_json = output_dir / f"twin_nemo_ts_maps_day{args.day}_kt{args.nemo_kt}.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print_report(result)
    print(f"OUTPUT json={output_json} sha256={sha256(output_json)}")
    for name, digest in figure_hashes.items():
        print(f"OUTPUT figure={output_dir / name} sha256={digest}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-npz", type=Path,
                        help="current faithful twin artifact (.npz)")
    parser.add_argument("--day", type=int, help="snapshot day key in the arm")
    parser.add_argument("--nemo-kt", type=int,
                        help="NEMO restart timestep, e.g. 17280 for day 360")
    parser.add_argument("--output-dir", type=Path,
                        help="fresh directory for PNGs and JSON sidecar")
    parser.add_argument("--nemo-run", type=Path, default=DEFAULT_NEMO_RUN)
    parser.add_argument("--mesh-mask", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--self-test", action="store_true",
                        help="run the planted wet-cell violation without inputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(planted_violation_self_test(), indent=2, sort_keys=True))
        return 0
    missing = [name for name in ("arm_npz", "day", "nemo_kt", "output_dir")
               if getattr(args, name) is None]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    require(args.day >= 0 and args.nemo_kt >= 0, "day and NEMO kt must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
