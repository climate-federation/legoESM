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

Two producers can be scored, through ONE scoring path (``load_candidate``
picks the reader; everything after it -- masking, statistics, controls,
figures -- is shared, so the two can never be scored differently):

* ``--arm-npz`` : a climate-battery arm artifact, with its own fp64 stamps.
* ``--run-dino-dir`` : a ``scripts/run/run_dino.py`` output directory, scored
  straight from its ``snapshots/``.  This needs the run to be on NEMO's own
  ``(199, 52)`` frame (``--nemo-faithful-grid``); it replaces the scratch
  adapter that used to embed a 48x195 standalone run into the middle of NEMO's
  domain, which the frame fix made unnecessary.  ``run_dino.py`` leaves
  legoESM's precision policy at its float32 default, so the snapshot's SOURCE
  dtypes are stamped and printed and the arrays are upcast for the arithmetic
  -- read any residual against a float32 representation floor (~1.2e-7
  relative) when the stamp says float32.

Examples, from the repository root::

  python scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py \
    --arm-npz /tmp/dino-climate-rebattery-01a04e34/arms/climate_a.npz \
    --day 360 --nemo-kt 17280 \
    --output-dir /tmp/dino-twin-nemo-ts-maps-day360

  python scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py \
    --run-dino-dir /data/abyssal/.../lego_trueframe \
    --day 360 --nemo-kt 11520 --nemo-run <NEMO>/cfgs/DINO/RUN_FROMREST_Y1 \
    --output-dir /data/abyssal/.../maps_trueframe

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

import numpy as np


def _set_script_env():
    """Process-wide defaults for RUNNING this comparator as a script, and the
    Matplotlib handle it returns.

    Kept out of import so the unit tests -- and any other importer -- do not
    inherit a CPU-only JAX, a hidden GPU or a redirected Matplotlib cache
    (review comment: importing a module must not reconfigure the process).
    Matplotlib is imported HERE rather than at module scope because importing
    it before ``MPLCONFIGDIR`` is set fails outright on a node whose home is
    read-only ("No usable temporary directory"), which is exactly the headless
    case this function exists to configure.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/legoesm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

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
        axis.figure.colorbar(image, ax=axis, shrink=0.85)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("longitude [deg E]")
        axis.set_ylabel("latitude [deg N]")
    return stats


def save_figures(output_dir: Path, day: int, fields: dict[str, np.ndarray],
                 masks: dict[str, np.ndarray], lon: np.ndarray, lat: np.ndarray,
                 depths: np.ndarray) -> dict[str, str]:
    surface_path = output_dir / f"surface_day{day}.png"
    plt = _set_script_env()
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
    plt = _set_script_env()
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
    _st = provenance["arm_stamps"]
    print(f"PROVENANCE producer={_st.get('producer', 'climate_arm_npz')} "
          f"source_dtypes={_st['field_dtypes']} "
          f"control_dtype={_st['control_dtype']}")
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


def load_candidate(args: argparse.Namespace):
    """The candidate state to score, from EITHER producer.

    Returns ``(T, S, ssh, land_mask, stamps)`` with ``T``/``S`` at
    ``(199, 52, 36)``, ``ssh``/``land_mask`` at ``(199, 52)``, and every scored
    array float64.  Which producer supplied them is recorded in ``stamps``;
    everything downstream -- masking, statistics, controls, figures -- is the
    SAME code for both, so the two producers can never be scored differently.
    """
    if args.run_dino_dir is not None:
        return load_run_dino_snapshot(args.run_dino_dir.resolve(), args.day)
    return load_arm_npz(args.arm_npz.resolve(), args.day)


def load_arm_npz(arm_path: Path, day: int):
    """The climate-battery arm producer (``*.npz`` with its own stamps)."""
    with np.load(arm_path, allow_pickle=False) as arm:
        keys = {name: f"{name}3d_day{day}" for name in ("T", "S", "eta")}
        for key in keys.values():
            require(key in arm.files, f"arm is missing requested day field {key}")
        require(bool(scalar(arm, "stable")), "arm producer marked unstable")
        require(int(scalar(arm, "producer_dirty_tracked_files")) == 0,
                "arm records a dirty producer")
        require(str(scalar(arm, "control_dtype")) == "float64",
                "arm control is not fp64")
        field_dtypes = {key: str(np.asarray(arm[key]).dtype)
                        for key in keys.values()}
        require(all(dtype == "float64" for dtype in field_dtypes.values()),
                f"arm scored fields are not all fp64: {field_dtypes}")
        return (np.asarray(arm[keys["T"]]), np.asarray(arm[keys["S"]]),
                np.asarray(arm[keys["eta"]]), np.asarray(arm["land_mask"]),
                {"producer": "climate_arm_npz",
                 "producer_path": str(arm_path),
                 "producer_git_sha": str(scalar(arm, "producer_git_sha")),
                 "producer_dirty_tracked_files": int(
                     scalar(arm, "producer_dirty_tracked_files")),
                 "codex_session_id": str(scalar(arm, "codex_session_id")),
                 "control_dtype": str(scalar(arm, "control_dtype")),
                 "field_dtypes": field_dtypes})


def load_run_dino_snapshot(run_dir: Path, day: int):
    """A ``scripts/run/run_dino.py`` output directory, scored directly.

    ``run_dino.py`` writes ``snapshots/snapshot_NNNNN.npz`` carrying ``T``,
    ``S``, ``eta``, ``land_mask`` and ``time_days``; the snapshot whose
    ``time_days`` matches ``day`` is the one scored.  On the NEMO-faithful
    grid that state is ALREADY NEMO's full ``(199, 52)`` frame, so there is no
    embedding step -- the earlier scratch adapter existed only because the
    standalone path built a 48x195 grid and had to be pasted into the middle
    of NEMO's domain, which is exactly the defect this frame fix removed.

    Two honest conversions, both stamped rather than silent:

    * the vertical axis is zero-PADDED up to NEMO's 36 levels when legoESM's
      ladder carries fewer.  NEMO's deepest level is permanently dry
      (``tmask[...,35]`` is empty everywhere), so the padding is only ever
      scored against dry cells, which the wet mask removes.
    * the arrays are UPCAST to float64 for the arithmetic.  That is a change
      of accumulator, not of information: ``run_dino.py`` leaves legoESM's
      precision policy at its default, which stores float32 even under
      JAX_ENABLE_X64 -- so the SOURCE dtypes go into the JSON under
      ``field_dtypes`` and are printed, and a float32 source carries a ~1.2e-7
      relative representation floor that any residual must be read against.
      (The arm producer sets fp64 and is required to; this producer is
      reported as it is, not asserted to be something it is not.)
    """
    require(run_dir.is_dir(), f"missing run directory {run_dir}")
    paths = sorted((run_dir / "snapshots").glob("snapshot_*.npz"))
    require(bool(paths), f"no snapshots under {run_dir / 'snapshots'}")
    chosen, chosen_days, times = None, None, []
    for path in paths:
        with np.load(path, allow_pickle=False) as snap:
            t_days = float(np.asarray(snap["time_days"]))
        times.append(t_days)
        if abs(t_days - day) < 0.5:
            require(chosen is None,
                    f"two snapshots claim day {day}: {chosen} and {path}")
            chosen, chosen_days = path, t_days
    require(chosen is not None,
            f"no snapshot at day {day} in {run_dir}; have days {times}")

    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    with np.load(chosen, allow_pickle=False) as snap:
        raw = {k: np.asarray(snap[k]) for k in ("T", "S", "eta", "land_mask")}
    field_dtypes = {k: str(v.dtype) for k, v in raw.items()}
    for name in ("T", "S"):
        value = raw[name]
        require(value.ndim == 3 and value.shape[:2] == FULL_HORIZONTAL_SHAPE,
                f"snapshot {name}: expected {FULL_HORIZONTAL_SHAPE}+levels, "
                f"got {value.shape}. A run on a frame other than NEMO's "
                "cannot be scored here -- use --nemo-faithful-grid.")
        require(value.shape[2] <= 36,
                f"snapshot {name} has {value.shape[2]} levels, more than "
                "NEMO's 36")
    for name in ("eta", "land_mask"):
        require(raw[name].shape == FULL_HORIZONTAL_SHAPE,
                f"snapshot {name}: expected {FULL_HORIZONTAL_SHAPE}, got "
                f"{raw[name].shape}")

    def to36(value: np.ndarray) -> np.ndarray:
        out = np.zeros((*FULL_HORIZONTAL_SHAPE, 36), dtype=np.float64)
        out[..., :value.shape[2]] = value.astype(np.float64)
        return out

    # The arm producer carries a `stable` stamp; a run_dino snapshot does not,
    # so stability is CHECKED here instead of trusted -- finite everywhere wet,
    # and a temperature inside a range no DINO run can leave while healthy.
    # (`run_dino.py` writes no git sha into run_metadata.json, so producer_git_sha
    # can be "unknown"; producer_sha256 pins the exact bytes scored regardless.)
    wet = raw["land_mask"] > 0.5
    for name in ("T", "S"):
        require(np.isfinite(raw[name][wet]).all(),
                f"snapshot {name} is not finite on wet columns -- the run "
                "was unstable and must not be scored")
    require(np.isfinite(raw["eta"][wet]).all(), "snapshot eta is not finite")
    t_wet = raw["T"][wet]
    require(float(t_wet.min()) > -5.0 and float(t_wet.max()) < 45.0,
            f"snapshot T spans [{float(t_wet.min()):.2f}, "
            f"{float(t_wet.max()):.2f}] degC on wet cells -- outside any "
            "healthy DINO state; the run was unstable and must not be scored")
    return (to36(raw["T"]), to36(raw["S"]),
            raw["eta"].astype(np.float64),
            raw["land_mask"].astype(np.float64),
            {"producer": "run_dino_snapshot",
             "producer_path": str(chosen),
             "producer_sha256": sha256(chosen),
             "producer_git_sha": str(meta.get("git_sha", "unknown")),
             "producer_run_metadata": str(meta_path) if meta else "absent",
             # The snapshot's OWN clock, not the requested day. --day selects
             # within +/-0.5 d, and the NEMO side is pinned by --nemo-kt, so
             # recording the request here would hide a half-day offset between
             # the two states being differenced.
             "snapshot_time_days": float(chosen_days),
             "requested_day": float(day),
             "snapshot_day_offset_days": float(chosen_days - day),
             # The dtype the SCORING arithmetic runs in.  This loader upcasts
             # every field, so reporting the snapshot's float32 source here
             # would suggest the comparison itself was float32; the source
             # dtypes are reported separately in field_dtypes.
             "control_dtype": "float64",
             "field_dtypes": field_dtypes,
             "upcast_to_float64_for_scoring": True,
             "vertical_zero_pad_to_36_levels": int(36 - raw["T"].shape[2])})


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    nemo_run = args.nemo_run.resolve()
    mesh_path = args.mesh_mask.resolve()
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
    # WHICH TIME LEVEL (oracle-fidelity Rule 1d).  NEMO's MLF restart carries
    # both, and comparing against the wrong one silently substitutes
    # |T_now - T_before| for "error".  "now" (tn/sn/sshn, the default) is the
    # state AFTER the step the restart is stamped with, and is what a day-N
    # comparison wants.  "before" (tb/sb/sshb, the Kbb level) is the ONLY
    # correct reference for a DAY-0 comparison against a from-rest kt=1
    # record: the Euler first step (l_1st_euler, istate.f90:114-115) advances
    # Kmm/Kaa only, so Kbb is still the untouched initial condition while
    # tn/sn/sshn have already moved (sshn reaches 1.17e-1 m by kt=1).
    level = {"now": ("tn", "sn", "sshn"),
             "before": ("tb", "sb", "sshb")}[args.nemo_time_level]
    raw = rebuild(restart_pattern, list(level))
    require(set(raw) == set(level),
            f"incomplete rebuilt comparator: {sorted(raw)}")
    if args.day == 0 and args.nemo_time_level == "now":
        print("WARNING day 0 is being scored against the NOW level. If this "
              "restart is a from-rest kt=1 record, the initial condition is "
              "the BEFORE level (tb/sb/sshb) and the now level has already "
              "taken the Euler step -- its sshn reaches 1.17e-1 m where the "
              "initial ssh is exactly 0. Pass --nemo-time-level before.")
    nemo_t_full = np.moveaxis(np.asarray(raw[level[0]]), 0, -1)
    nemo_s_full = np.moveaxis(np.asarray(raw[level[1]]), 0, -1)
    nemo_ssh_full = np.asarray(raw[level[2]])
    require(np.isfinite(nemo_t_full).all() and np.isfinite(nemo_s_full).all()
            and np.isfinite(nemo_ssh_full).all(), "stitched NEMO state is incomplete")

    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask

    mesh_full = read_nemo_mesh_mask(str(mesh_path), nn_hls=0)
    require(mesh_full.tmask.shape[:2] == FULL_HORIZONTAL_SHAPE,
            f"mesh full shape is {mesh_full.tmask.shape[:2]}, expected "
            f"{FULL_HORIZONTAL_SHAPE}")
    arm_t_full, arm_s_full, arm_ssh_full, arm_land_full, arm_stamps = (
        load_candidate(args))
    for name, value in (("arm T", arm_t_full), ("arm S", arm_s_full),
                        ("NEMO T", nemo_t_full), ("NEMO S", nemo_s_full)):
        require(value.shape == (199, 52, 36),
                f"{name}: expected (199,52,36), got {value.shape}")
    for name, value in (("arm SSH", arm_ssh_full),
                        ("NEMO SSH", nemo_ssh_full),
                        ("arm land mask", arm_land_full)):
        require(value.shape == FULL_HORIZONTAL_SHAPE,
                f"{name}: expected {FULL_HORIZONTAL_SHAPE}, got {value.shape}")

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

    input_paths = [Path(arm_stamps["producer_path"]), mesh_path,
                   *restart_paths]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    script_path = Path(__file__).resolve()
    result = {
        "schema": "dino-twin-nemo-ts-maps-v1",
        "day": args.day,
        "nemo_kt": args.nemo_kt,
        "nemo_time_level": args.nemo_time_level,
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
            "arm_path": arm_stamps["producer_path"],
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
    parser.add_argument("--run-dino-dir", type=Path,
                        help="score a scripts/run/run_dino.py output directory "
                             "directly (its snapshots/ must be on NEMO's "
                             "(199,52) frame, i.e. --nemo-faithful-grid); "
                             "mutually exclusive with --arm-npz")
    parser.add_argument("--day", type=int, help="snapshot day key in the arm")
    parser.add_argument("--nemo-kt", type=int,
                        help="NEMO restart timestep, e.g. 17280 for day 360")
    parser.add_argument("--output-dir", type=Path,
                        help="fresh directory for PNGs and JSON sidecar")
    parser.add_argument("--nemo-run", type=Path, default=DEFAULT_NEMO_RUN)
    parser.add_argument("--mesh-mask", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--nemo-time-level", choices=("now", "before"),
                        default="now",
                        help="which MLF level of the NEMO restart to score "
                             "against: 'now' (tn/sn/sshn, default) is the "
                             "state after the stamped step; 'before' "
                             "(tb/sb/sshb) is the Kbb level, and is the only "
                             "correct reference for a day-0 comparison "
                             "against a from-rest kt=1 record")
    parser.add_argument("--self-test", action="store_true",
                        help="run the planted wet-cell violation without inputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    _set_script_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(planted_violation_self_test(), indent=2, sort_keys=True))
        return 0
    if (args.arm_npz is None) == (args.run_dino_dir is None):
        parser.error("give exactly one of --arm-npz / --run-dino-dir")
    missing = [name for name in ("day", "nemo_kt", "output_dir")
               if getattr(args, name) is None]
    if missing:
        parser.error("required unless --self-test: " + ", ".join(missing))
    require(args.day >= 0 and args.nemo_kt >= 0, "day and NEMO kt must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
