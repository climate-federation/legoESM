#!/usr/bin/env python
"""Run hydrostatic + non-hydrostatic atmosphere matrices with unified outputs.

This test-side orchestrator runs available benchmark scripts and reorganizes
artifacts under:
  results/atmosphere/<dycore>/<grid>/<case>/

Coverage (available in current workspace):
- Hydrostatic:
  - Held-Suarez FV cube (gray, rrtmgp)
  - Held-Suarez FV lat-lon (gray, rrtmgp)
  - Held-Suarez spectral (gray, rrtmgp)
  - AMIP FV cube (gray, rrtmgp)
  - AMIP spectral (gray, rrtmgp)
  - RCE fixed uniform SST (gray)
  - RCE slab land (gray)
- Non-hydrostatic:
  - DCMIP TC1/TC2a/TC3 FV cube (gray, rrtmgp)
  - DCMIP TC1 spectral (no radiation-coupled benchmark path)
- Icosahedral:
  - Shallow-water Williamson TC2/TC5/TC6 (native + lat-lon pixel snapshots)

Artifacts per case (standardized when available):
- field_snapshots.png
- vertical_profiles.png / vertical_profiles.csv
- integrated_timeseries.png / integrated_timeseries.csv
- conservation_timeseries.png / conservation_timeseries.csv
- zonal_cross_sections.png / meridional_cross_sections.png (where available)
- snapshot_times.txt
- results.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Preset:
    name: str
    cube_resolution: int
    latlon_nlat: int
    latlon_nlon: int
    spectral_truncation: int
    hydro_levels: int
    nh_levels: int


PRESETS = {
    "low": Preset(
        name="low",
        cube_resolution=8,
        latlon_nlat=24,
        latlon_nlon=48,
        spectral_truncation=10,
        hydro_levels=10,
        nh_levels=10,
    ),
    "medium": Preset(
        name="medium",
        cube_resolution=12,
        latlon_nlat=36,
        latlon_nlon=72,
        spectral_truncation=15,
        hydro_levels=15,
        nh_levels=15,
    ),
    "atm2deg": Preset(
        name="atm2deg",
        cube_resolution=32,
        latlon_nlat=90,
        latlon_nlon=180,
        spectral_truncation=63,
        hydro_levels=20,
        nh_levels=20,
    ),
}


CASE_MAP = {
    "01_hs_fv_cube": ("hydrostatic", "cube_sphere", "held_suarez_finite_volume"),
    "02_hs_fv_latlon": ("hydrostatic", "latlon", "held_suarez_finite_volume"),
    "03_hs_spectral": ("hydrostatic", "gaussian_latlon", "held_suarez_spectral"),
    "04_dcmip_tc1_fv_cube": ("nonhydrostatic", "cube_sphere", "dcmip_tc1_finite_volume"),
    "05_dcmip_tc2a_fv_cube": ("nonhydrostatic", "cube_sphere", "dcmip_tc2a_finite_volume"),
    "06_dcmip_tc3_fv_cube": ("nonhydrostatic", "cube_sphere", "dcmip_tc3_finite_volume"),
}


def _run(cmd: list[str], cwd: Path) -> tuple[int, float]:
    env = os.environ.copy()
    root = str(cwd.resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    env.setdefault("JAX_ENABLE_X64", "True")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return proc.returncode, time.time() - t0


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_csv_columns(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        return headers, np.zeros((0, len(headers)), dtype=np.float64)
    data = np.zeros((len(rows), len(headers)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, h in enumerate(headers):
            try:
                data[i, j] = float(row.get(h, "nan"))
            except Exception:
                data[i, j] = np.nan
    return headers, data


def _derive_conservation_from_mean(case_dir: Path) -> bool:
    mean_csv = case_dir / "integrated_timeseries.csv"
    if not mean_csv.exists():
        mean_csv = case_dir / "mean_timeseries.csv"
    if not mean_csv.exists():
        return False

    headers, data = _load_csv_columns(mean_csv)
    if data.size == 0:
        return False

    idx = {k: i for i, k in enumerate(headers)}
    if "step" in idx:
        step = data[:, idx["step"]]
    else:
        step = np.arange(data.shape[0], dtype=np.float64)
    if "time_days" in idx:
        t_days = data[:, idx["time_days"]]
    elif "time_seconds" in idx:
        t_days = data[:, idx["time_seconds"]] / 86400.0
    elif "days" in idx:
        t_days = data[:, idx["days"]]
    else:
        t_days = np.arange(data.shape[0], dtype=np.float64)

    mass_candidates = ("mean_p_s", "dry_mass_ps", "dry_mass", "mean_rho_prime_3d", "mean_T_3d")
    energy_candidates = ("energy_column", "mean_wind_3d", "mean_abs_w_3d", "mean_T_3d")
    mass_key = next((k for k in mass_candidates if k in idx), None)
    energy_key = next((k for k in energy_candidates if k in idx), None)
    if mass_key is None and energy_key is None:
        return False

    mass = data[:, idx[mass_key]] if mass_key is not None else np.ones_like(t_days)
    energy = data[:, idx[energy_key]] if energy_key is not None else np.ones_like(t_days)
    mass0 = max(abs(float(mass[0])), 1.0e-30)
    energy0 = max(abs(float(energy[0])), 1.0e-30)
    mrel = (mass - mass[0]) / mass0
    erel = (energy - energy[0]) / energy0

    out_csv = case_dir / "conservation_timeseries.csv"
    with out_csv.open("w") as f:
        f.write("step,time_days,mass_proxy,energy_proxy,mass_rel,energy_rel\n")
        for i in range(step.size):
            f.write(
                f"{int(step[i])},{t_days[i]:.8f},{mass[i]:.12e},{energy[i]:.12e},"
                f"{mrel[i]:.12e},{erel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    axes[0].plot(t_days, mrel, lw=1.8, color="tab:blue")
    axes[0].axhline(0.0, color="0.3", lw=0.8, ls="--")
    axes[0].set_ylabel("Mass proxy drift")
    axes[0].set_title(mass_key or "mass")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_days, erel, lw=1.8, color="tab:red")
    axes[1].axhline(0.0, color="0.3", lw=0.8, ls="--")
    axes[1].set_ylabel("Energy proxy drift")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title(energy_key or "energy")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_snapshot_times(case_dir: Path, days: list[float]) -> None:
    with (case_dir / "snapshot_times.txt").open("w") as f:
        for d in sorted(set(float(x) for x in days)):
            f.write(f"day {d:.2f}\n")


def _write_placeholder_plot(path: Path, title: str, message: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_placeholder_csv(path: Path, header: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header.rstrip() + "\n")


def _harmonize_output_names(case_dir: Path) -> None:
    # Snapshot conventions
    if not (case_dir / "field_snapshots_native.png").exists():
        _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_native.png")
    if not (case_dir / "field_snapshots_latlon_pixels.png").exists():
        _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_latlon_pixels.png")
    _write_placeholder_plot(
        case_dir / "field_snapshots_native.png",
        "Native Grid Snapshots",
        "Snapshot fields unavailable for this short/incomplete run.",
    )
    _write_placeholder_plot(
        case_dir / "field_snapshots_latlon_pixels.png",
        "Lat-Lon Pixel Snapshots",
        "Snapshot fields unavailable for this short/incomplete run.",
    )
    _copy_if_exists(case_dir / "field_snapshots_latlon_pixels.png", case_dir / "field_snapshots.png")

    # Mean time series conventions
    _copy_if_exists(case_dir / "mean_timeseries.csv", case_dir / "integrated_timeseries.csv")
    _copy_if_exists(case_dir / "mean_timeseries.png", case_dir / "integrated_timeseries.png")
    _copy_if_exists(case_dir / "integrated_timeseries.csv", case_dir / "mean_timeseries.csv")
    _copy_if_exists(case_dir / "integrated_timeseries.png", case_dir / "mean_timeseries.png")
    _write_placeholder_csv(case_dir / "mean_timeseries.csv", "time_days")
    _write_placeholder_csv(case_dir / "integrated_timeseries.csv", "time_days")
    _write_placeholder_plot(
        case_dir / "mean_timeseries.png",
        "Mean Conserved Variables",
        "Mean time-series unavailable for this short/incomplete run.",
    )
    _copy_if_exists(case_dir / "mean_timeseries.png", case_dir / "integrated_timeseries.png")

    # Vertical profile conventions
    _copy_if_exists(case_dir / "vertical_profiles.csv", case_dir / "mean_profiles.csv")
    _copy_if_exists(case_dir / "vertical_profiles.png", case_dir / "mean_profiles.png")
    _copy_if_exists(case_dir / "mean_profiles.csv", case_dir / "vertical_profiles.csv")
    _copy_if_exists(case_dir / "mean_profiles.png", case_dir / "vertical_profiles.png")
    _copy_if_exists(case_dir / "vertical_profiles.png", case_dir / "vertical_profile_evolution.png")
    _write_placeholder_csv(case_dir / "vertical_profiles.csv", "time_days,sigma,profile_value")
    _write_placeholder_csv(case_dir / "mean_profiles.csv", "time_days,sigma,profile_value")

    # Cross-section conventions
    _copy_if_exists(case_dir / "zonal_cross_sections.png", case_dir / "lat_vertical_snapshots.png")
    _copy_if_exists(case_dir / "meridional_cross_sections.png", case_dir / "lon_vertical_snapshots.png")
    _copy_if_exists(case_dir / "lat_vertical_snapshots.png", case_dir / "zonal_cross_sections.png")
    _copy_if_exists(case_dir / "lon_vertical_snapshots.png", case_dir / "meridional_cross_sections.png")

    # Fill missing-but-required diagnostic views with explicit placeholders.
    _write_placeholder_plot(
        case_dir / "lat_vertical_snapshots.png",
        "Latitude-Vertical Snapshots",
        "Not available for this case output in current model path.",
    )
    _write_placeholder_plot(
        case_dir / "lon_vertical_snapshots.png",
        "Longitude-Vertical Snapshots",
        "Not available for this case output in current model path.",
    )
    _write_placeholder_plot(
        case_dir / "vertical_profile_evolution.png",
        "Vertical Profile Evolution",
        "Not available for this case output in current model path.",
    )
    _write_placeholder_csv(
        case_dir / "conservation_timeseries.csv",
        "step,time_days,mass_proxy,energy_proxy,mass_rel,energy_rel",
    )
    _write_placeholder_plot(
        case_dir / "conservation_timeseries.png",
        "Conservation Diagnostics",
        "Conservation diagnostics unavailable for this short/incomplete run.",
    )


def _standardize_case_artifacts(case_dir: Path) -> None:
    _harmonize_output_names(case_dir)
    if not (case_dir / "conservation_timeseries.csv").exists():
        _derive_conservation_from_mean(case_dir)


def _build_amip_artifacts(case_dir: Path, default_snapshot_days: list[float]) -> None:
    if not _copy_if_exists(case_dir / "amip_snapshots.png", case_dir / "field_snapshots.png"):
        _copy_if_exists(case_dir / "amip_final_state.png", case_dir / "field_snapshots.png")
    if not _copy_if_exists(case_dir / "amip_snapshots_native.png", case_dir / "field_snapshots_native.png"):
        _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_native.png")
    if not _copy_if_exists(
        case_dir / "amip_snapshots_latlon_pixels.png",
        case_dir / "field_snapshots_latlon_pixels.png",
    ):
        _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_latlon_pixels.png")
    _copy_if_exists(case_dir / "amip_profiles.png", case_dir / "vertical_profiles.png")
    _copy_if_exists(case_dir / "amip_timeseries.png", case_dir / "mean_timeseries.png")
    _copy_if_exists(case_dir / "amip_profiles.png", case_dir / "mean_profiles.png")
    _copy_if_exists(case_dir / "amip_timeseries.png", case_dir / "integrated_timeseries.png")

    npz_path = case_dir / "timeseries.npz"
    if not npz_path.exists():
        _write_snapshot_times(case_dir, default_snapshot_days)
        _harmonize_output_names(case_dir)
        return

    data = np.load(npz_path)
    keys = list(data.keys())
    if "days" in data:
        t_days = np.asarray(data["days"], dtype=float)
    elif "times" in data:
        t_days = np.asarray(data["times"], dtype=float)
    else:
        t_days = np.arange(len(np.asarray(data[keys[0]])), dtype=float)

    one_d = {}
    for k in keys:
        arr = np.asarray(data[k])
        if arr.ndim == 1 and arr.size == t_days.size:
            one_d[k] = arr.astype(float)

    if t_days.size == 0:
        _write_snapshot_times(case_dir, default_snapshot_days)
        _harmonize_output_names(case_dir)
        return

    mean_csv = case_dir / "mean_timeseries.csv"
    with mean_csv.open("w") as f:
        cols = ["time_days"] + sorted(one_d.keys())
        f.write(",".join(cols) + "\n")
        for i in range(t_days.size):
            vals = [f"{t_days[i]:.8f}"] + [f"{one_d[k][i]:.12e}" for k in sorted(one_d.keys())]
            f.write(",".join(vals) + "\n")

    mass = None
    for key in ("dry_mass_ps", "dry_mass"):
        if key in one_d:
            mass = one_d[key]
            break
    energy = one_d.get("energy_column")
    if mass is not None and mass.size > 0:
        mass0 = max(abs(float(mass[0])), 1.0e-30)
        mrel = (mass - mass[0]) / mass0
    else:
        mrel = np.zeros_like(t_days)
    if energy is not None and energy.size > 0:
        e0 = max(abs(float(energy[0])), 1.0e-30)
        erel = (energy - energy[0]) / e0
    else:
        erel = np.full_like(t_days, np.nan)

    with (case_dir / "conservation_timeseries.csv").open("w") as f:
        f.write("step,time_days,mass_proxy,energy_proxy,mass_rel,energy_rel\n")
        for i in range(t_days.size):
            mass_i = float(mass[i]) if mass is not None else float("nan")
            energy_i = float(energy[i]) if energy is not None else float("nan")
            f.write(
                f"{i},{t_days[i]:.8f},{mass_i:.12e},{energy_i:.12e},{mrel[i]:.12e},{erel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    axes[0].plot(t_days, mrel, lw=1.8, color="tab:blue")
    axes[0].axhline(0.0, color="0.3", lw=0.8, ls="--")
    axes[0].set_ylabel("Mass drift (rel.)")
    axes[0].set_title("Dry-mass proxy conservation")
    axes[0].grid(True, alpha=0.3)
    if np.all(np.isnan(erel)):
        axes[1].text(0.5, 0.5, "Energy proxy unavailable", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        axes[1].plot(t_days, erel, lw=1.8, color="tab:red")
        axes[1].axhline(0.0, color="0.3", lw=0.8, ls="--")
        axes[1].set_ylabel("Energy drift (rel.)")
        axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (days)")
    fig.tight_layout()
    fig.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    prof_csv = case_dir / "vertical_profiles.csv"
    sigma = np.asarray(data["sigma"]) if "sigma" in data else None
    prof_t = np.asarray(data["profiles_T"]) if "profiles_T" in data else np.array([])
    prof_q = np.asarray(data["profiles_qv"]) if "profiles_qv" in data else np.array([])
    if sigma is not None and prof_t.ndim == 2 and prof_t.shape[0] == t_days.size:
        with prof_csv.open("w") as f:
            f.write("time_days,sigma,T_profile,qv_profile_gkg\n")
            for ti in range(t_days.size):
                for k in range(sigma.size):
                    qval = np.nan
                    if prof_q.ndim == 2 and prof_q.shape == prof_t.shape:
                        qval = prof_q[ti, k]
                    f.write(f"{t_days[ti]:.8f},{sigma[k]:.8e},{prof_t[ti, k]:.8e},{qval:.8e}\n")

    _write_snapshot_times(case_dir, default_snapshot_days)
    _harmonize_output_names(case_dir)


def _build_rce_artifacts(case_dir: Path, default_snapshot_days: list[float]) -> None:
    if not _copy_if_exists(case_dir / "rce_snapshots.png", case_dir / "field_snapshots.png"):
        _copy_if_exists(case_dir / "rce_final_state.png", case_dir / "field_snapshots.png")
    if not _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_native.png"):
        _copy_if_exists(case_dir / "rce_final_state.png", case_dir / "field_snapshots_native.png")
    if not _copy_if_exists(case_dir / "field_snapshots.png", case_dir / "field_snapshots_latlon_pixels.png"):
        _copy_if_exists(case_dir / "rce_final_state.png", case_dir / "field_snapshots_latlon_pixels.png")

    _copy_if_exists(case_dir / "rce_profiles.png", case_dir / "vertical_profiles.png")
    _copy_if_exists(case_dir / "rce_profiles.png", case_dir / "mean_profiles.png")
    _copy_if_exists(case_dir / "rce_profiles.png", case_dir / "vertical_profile_evolution.png")

    _copy_if_exists(case_dir / "rce_timeseries.png", case_dir / "mean_timeseries.png")
    _copy_if_exists(case_dir / "rce_timeseries.png", case_dir / "integrated_timeseries.png")

    _write_snapshot_times(case_dir, default_snapshot_days)
    _harmonize_output_names(case_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hydro/NH benchmark matrix with standardized outputs.")
    parser.add_argument("--preset", type=str, default="atm2deg", choices=sorted(PRESETS.keys()))
    parser.add_argument("--output", type=Path, default=Path("results/atmosphere"))
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--schemes", type=str, default="gray,rrtmgp")
    parser.add_argument("--solver", type=str, default="ssp45")
    parser.add_argument("--hs-days", type=float, default=0.5)
    parser.add_argument("--tc1-hours", type=float, default=0.2)
    parser.add_argument("--tc23-minutes", type=float, default=6.0)
    parser.add_argument("--mean-every", type=int, default=10)
    parser.add_argument("--amip-days", type=int, default=10)
    parser.add_argument("--amip-dt", type=float, default=600.0)
    parser.add_argument("--amip-spec-dt", type=float, default=900.0)
    parser.add_argument("--rce-days", type=int, default=30)
    parser.add_argument("--rce-diag-days", type=int, default=5)
    parser.add_argument("--rce-dt", type=float, default=0.0, help="RCE dt override; 0 uses script auto dt.")
    parser.add_argument("--icosa-cases", type=str, default="tc2,tc5,tc6")
    parser.add_argument("--icosa-mesh-levels", type=str, default="5")
    parser.add_argument("--icosa-days-tc2", type=float, default=5.0)
    parser.add_argument("--icosa-days-tc5", type=float, default=5.0)
    parser.add_argument("--icosa-days-tc6", type=float, default=5.0)
    parser.add_argument("--icosa-save-every", type=int, default=20)
    parser.add_argument("--skip-hs-dcmip", action="store_true")
    parser.add_argument("--skip-amip", action="store_true")
    parser.add_argument("--skip-rce", action="store_true")
    parser.add_argument("--skip-nh-spectral", action="store_true")
    parser.add_argument("--skip-icosahedral-sw", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_exec = args.python if args.python.is_absolute() else (repo_root / args.python)
    output_root = (repo_root / args.output).resolve()
    raw_root = output_root / "_raw_hydro_nh_matrix"
    raw_root.mkdir(parents=True, exist_ok=True)

    preset = PRESETS[args.preset]
    schemes = []
    for s in [x.strip() for x in args.schemes.split(",") if x.strip()]:
        schemes.append("rrtmgp" if s == "rrtmg" else s)
    records: list[dict[str, object]] = []

    # 1) HS + DCMIP radiation matrix (hydro + NH FV)
    if not args.skip_hs_dcmip:
        matrix_out = raw_root / "hs_dcmip_matrix"
        cmd_matrix = [
            str(python_exec),
            "scripts/run_hs_dcmip_radiation_matrix.py",
            "--output",
            str(matrix_out),
            "--schemes",
            ",".join(schemes),
            "--presets",
            preset.name,
            "--hs-days",
            str(args.hs_days),
            "--tc1-hours",
            str(args.tc1_hours),
            "--tc23-minutes",
            str(args.tc23_minutes),
            "--mean-every",
            str(args.mean_every),
            "--solver",
            args.solver,
        ]
        rc, wall = _run(cmd_matrix, repo_root)
        records.append(
            {
                "component": "hs_dcmip_matrix",
                "status": "PASS" if rc == 0 else "FAIL",
                "wall_time_s": wall,
                "output": str(matrix_out),
            },
        )

        if rc == 0:
            for scheme in schemes:
                for src_name, (dycore, grid, case) in CASE_MAP.items():
                    src = matrix_out / scheme / preset.name / src_name
                    if not src.exists():
                        records.append(
                            {
                                "component": f"{scheme}/{src_name}",
                                "status": "MISSING",
                                "wall_time_s": 0.0,
                                "output": "",
                            },
                        )
                        continue
                    dst = output_root / dycore / grid / case / f"{scheme}_{preset.name}"
                    _copy_tree(src, dst)
                    _standardize_case_artifacts(dst)
                    records.append(
                        {
                            "component": f"{scheme}/{src_name}",
                            "status": "PASS",
                            "wall_time_s": 0.0,
                            "output": str(dst),
                        },
                    )

    # 2) AMIP FV + spectral (gray + rrtmgp)
    if not args.skip_amip:
        amip_snapshot_days = [d for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300] if d <= args.amip_days]
        if args.amip_days not in amip_snapshot_days:
            amip_snapshot_days.append(float(args.amip_days))

        for rad in ("gray", "rrtmgp"):
            amip_raw = raw_root / f"amip_cube_fv_{rad}_{preset.name}"
            cmd_amip = [
                str(python_exec),
                "scripts/run_amip.py",
                "--dataset",
                "analytical",
                "--vertical-coord",
                "sigma",
                "--days",
                str(args.amip_days),
                "--resolution",
                str(preset.cube_resolution),
                "--nlev",
                str(preset.hydro_levels),
                "--dt",
                str(args.amip_dt),
                "--diag-days",
                "1",
                "--discretization",
                "finite_volume",
                "--radiation",
                rad,
                "--output",
                str(amip_raw),
            ]
            rc, wall = _run(cmd_amip, repo_root)
            status = "PASS" if rc == 0 else "FAIL"
            records.append(
                {
                    "component": f"amip_cube_fv_{rad}",
                    "status": status,
                    "wall_time_s": wall,
                    "output": str(amip_raw),
                },
            )
            if rc == 0:
                dst = output_root / "hydrostatic" / "cube_sphere" / "amip_finite_volume" / f"{rad}_{preset.name}"
                _copy_tree(amip_raw, dst)
                _build_amip_artifacts(dst, amip_snapshot_days)

        for rad in ("gray", "rrtmgp"):
            amip_spec_raw = raw_root / f"amip_spectral_{rad}_{preset.name}"
            cmd_amip_spec = [
                str(python_exec),
                "scripts/run_amip_spectral.py",
                "--dataset",
                "analytical",
                "--vertical-coord",
                "sigma",
                "--days",
                str(args.amip_days),
                "--truncation",
                str(preset.spectral_truncation),
                "--nlev",
                str(preset.hydro_levels),
                "--dt",
                str(args.amip_spec_dt),
                "--diag-days",
                "1",
                "--radiation",
                rad,
                "--output",
                str(amip_spec_raw),
            ]
            rc, wall = _run(cmd_amip_spec, repo_root)
            status = "PASS" if rc == 0 else "FAIL"
            records.append(
                {
                    "component": f"amip_spectral_{rad}",
                    "status": status,
                    "wall_time_s": wall,
                    "output": str(amip_spec_raw),
                },
            )
            if rc == 0:
                dst = output_root / "hydrostatic" / "gaussian_latlon" / "amip_spectral" / f"{rad}_{preset.name}"
                _copy_tree(amip_spec_raw, dst)
                _build_amip_artifacts(dst, amip_snapshot_days)

    # 3) RCE fixed-SST ocean + RCE land (gray)
    if not args.skip_rce:
        rce_snapshot_days = [d for d in [10, 30, 60, 100, 200, 300] if d <= args.rce_days]
        if args.rce_days not in rce_snapshot_days:
            rce_snapshot_days.append(float(args.rce_days))

        rce_cases = [
            ("rce_fixed_uniform_sst", "scripts/run_rce_slab_ocean.py", raw_root / f"rce_uniform_sst_gray_{preset.name}"),
            ("rce_slab_land", "scripts/run_rce_slab_land.py", raw_root / f"rce_slab_land_gray_{preset.name}"),
        ]
        for case_name, script_path, rce_raw in rce_cases:
            cmd_rce = [
                str(python_exec),
                script_path,
                "--days",
                str(args.rce_days),
                "--resolution",
                str(preset.cube_resolution),
                "--nlev",
                str(preset.hydro_levels),
                "--diag-days",
                str(max(1, args.rce_diag_days)),
                "--output",
                str(rce_raw),
            ]
            if args.rce_dt > 0.0:
                cmd_rce.extend(["--dt", str(args.rce_dt)])
            rc, wall = _run(cmd_rce, repo_root)
            status = "PASS" if rc == 0 else "FAIL"
            records.append(
                {
                    "component": f"{case_name}_gray",
                    "status": status,
                    "wall_time_s": wall,
                    "output": str(rce_raw),
                },
            )
            if rc == 0:
                dst = output_root / "hydrostatic" / "cube_sphere" / case_name / f"gray_{preset.name}"
                _copy_tree(rce_raw, dst)
                _build_rce_artifacts(dst, rce_snapshot_days)

    # 4) NH spectral TC1 (available benchmark spectral NH path)
    if not args.skip_nh_spectral:
        nh_spec_raw = raw_root / f"nh_spectral_tc1_{preset.name}"
        cmd_nh_spec = [
            str(python_exec),
            "scripts/run_atmosphere_25deg_ssp45_full.py",
            "--only",
            "nh_spec_tc1",
            "--output",
            str(nh_spec_raw),
            "--spectral-truncation",
            str(preset.spectral_truncation),
            "--nh-levels",
            str(preset.nh_levels),
            "--mean-every",
            str(args.mean_every),
        ]
        rc, wall = _run(cmd_nh_spec, repo_root)
        status = "PASS" if rc == 0 else "FAIL"
        records.append(
            {
                "component": "nh_spectral_tc1",
                "status": status,
                "wall_time_s": wall,
                "output": str(nh_spec_raw),
            },
        )
        if rc == 0:
            produced = [p for p in nh_spec_raw.iterdir() if p.is_dir() and (p / "field_snapshots.png").exists()]
            if produced:
                dst = output_root / "nonhydrostatic" / "gaussian_latlon" / "dcmip_tc1_spectral" / f"default_{preset.name}"
                _copy_tree(produced[0], dst)
                _standardize_case_artifacts(dst)

    # 5) Icosahedral shallow-water Williamson runs (MPAS-style Voronoi mesh)
    if not args.skip_icosahedral_sw:
        sw_icosa_raw = raw_root / f"shallow_water_icosahedral_{preset.name}"
        cmd_sw_icosa = [
            str(python_exec),
            "tests/atmosphere/shallow_water/run_williamson_mpas_cases.py",
            "--output",
            str(sw_icosa_raw),
            "--mesh-levels",
            args.icosa_mesh_levels,
            "--latlon-nlon",
            str(max(180, preset.latlon_nlon)),
            "--latlon-nlat",
            str(max(90, preset.latlon_nlat)),
            "--cases",
            args.icosa_cases,
            "--days-tc2",
            str(args.icosa_days_tc2),
            "--days-tc5",
            str(args.icosa_days_tc5),
            "--days-tc6",
            str(args.icosa_days_tc6),
            "--save-every",
            str(max(1, args.icosa_save_every)),
        ]
        rc, wall = _run(cmd_sw_icosa, repo_root)
        records.append(
            {
                "component": "shallow_water_icosahedral",
                "status": "PASS" if rc == 0 else "FAIL",
                "wall_time_s": wall,
                "output": str(sw_icosa_raw),
            },
        )
        if rc == 0 and sw_icosa_raw.exists():
            for case_dir in sorted([p for p in sw_icosa_raw.iterdir() if p.is_dir() and p.name.startswith("tc")]):
                case_tag = case_dir.name.split("_", 1)[0]
                dst = output_root / "shallow_water" / "icosahedral" / f"williamson_{case_tag}" / f"default_{preset.name}"
                _copy_tree(case_dir, dst)
                _standardize_case_artifacts(dst)

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "preset": preset.__dict__,
            "schemes": schemes,
            "solver": args.solver,
            "hs_days": args.hs_days,
            "tc1_hours": args.tc1_hours,
            "tc23_minutes": args.tc23_minutes,
            "amip_days": args.amip_days,
            "amip_dt": args.amip_dt,
            "amip_spec_dt": args.amip_spec_dt,
            "rce_days": args.rce_days,
            "rce_diag_days": args.rce_diag_days,
            "rce_dt": args.rce_dt,
            "mean_every": args.mean_every,
            "icosa_cases": args.icosa_cases,
            "icosa_mesh_levels": args.icosa_mesh_levels,
            "icosa_days_tc2": args.icosa_days_tc2,
            "icosa_days_tc5": args.icosa_days_tc5,
            "icosa_days_tc6": args.icosa_days_tc6,
            "icosa_save_every": args.icosa_save_every,
        },
        "results": records,
    }
    with (output_root / "hydro_nh_matrix_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    n_pass = sum(1 for r in records if r["status"] == "PASS")
    n_fail = sum(1 for r in records if r["status"] == "FAIL")
    n_skip = sum(1 for r in records if r["status"] == "SKIPPED")
    n_missing = sum(1 for r in records if r["status"] == "MISSING")
    print(f"Summary written to {output_root / 'hydro_nh_matrix_summary.json'}")
    print(f"PASS={n_pass} FAIL={n_fail} SKIPPED={n_skip} MISSING={n_missing} TOTAL={len(records)}")


if __name__ == "__main__":
    main()
