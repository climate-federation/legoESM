#!/usr/bin/env python
"""Ordered atmosphere dycore progression suite with unified artifacts.

This orchestrator prepares and runs a consistent dycore validation ladder:
1) Shallow-water (spectral lat-lon, FV lat-lon, FV cube-sphere)
2) Hydrostatic core (FV cube, FV lat-lon, spectral)
3) Non-hydrostatic core (FV cube, spectral, harder FV cube cases)

Outputs are written in one location under tests/validation/results with a
standardized per-case artifact schema:
- field_snapshots.png
- snapshot_times.txt
- mean_timeseries.csv / mean_timeseries.png
- conservation_timeseries.csv / conservation_timeseries.png
- mean_profiles.csv / mean_profiles.png (or a placeholder for SW cases)
"""

from __future__ import annotations

import argparse
import csv
import json
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
class SuiteCase:
    rank: int
    case_id: str
    title: str
    source_kind: str  # shallow | atmosphere
    source_key: str   # shallow subdir key or atmosphere --only key


SW_SUBCASE_MAP = {
    "01_latlon_spectral_williamson2": "01_sw_spectral_latlon",
    "02_latlon_fv_williamson2": "02_sw_fv_latlon",
    "03_cubesphere_fv_williamson2": "03_sw_fv_cubesphere",
}


ORDERED_CASES: list[SuiteCase] = [
    SuiteCase(1, "01_sw_spectral_latlon", "SW Spectral Williamson-2 (lat-lon)", "shallow", "01_latlon_spectral_williamson2"),
    SuiteCase(2, "02_sw_fv_latlon", "SW FV Williamson-2 (lat-lon remap)", "shallow", "02_latlon_fv_williamson2"),
    SuiteCase(3, "03_sw_fv_cubesphere", "SW FV Williamson-2 (cube-sphere)", "shallow", "03_cubesphere_fv_williamson2"),
    SuiteCase(4, "04_hydro_fv_cube_held_suarez", "Hydro FV Held-Suarez (cube-sphere)", "atmosphere", "hydro_fv_hs"),
    SuiteCase(5, "05_hydro_fv_latlon_held_suarez", "Hydro FV Held-Suarez (lat-lon)", "atmosphere", "hydro_latlon_hs"),
    SuiteCase(6, "06_hydro_spectral_held_suarez", "Hydro Spectral Held-Suarez (gaussian lat-lon)", "atmosphere", "hydro_spec_hs"),
    SuiteCase(7, "07_hydro_fv_cube_baroclinic", "Hydro FV Baroclinic Wave (cube-sphere)", "atmosphere", "hydro_fv_bw"),
    SuiteCase(8, "08_hydro_spectral_baroclinic", "Hydro Spectral Baroclinic Wave (gaussian lat-lon)", "atmosphere", "hydro_spec_bw"),
    SuiteCase(9, "09_nh_fv_cube_tc1", "NH FV DCMIP TC1 (cube-sphere)", "atmosphere", "nh_fv_tc1"),
    SuiteCase(10, "10_nh_spectral_tc1", "NH Spectral DCMIP TC1 (gaussian lat-lon)", "atmosphere", "nh_spec_tc1"),
    SuiteCase(11, "11_nh_fv_cube_tc2a", "NH FV DCMIP TC2a (cube-sphere)", "atmosphere", "nh_fv_tc2a"),
    SuiteCase(12, "12_nh_fv_cube_tc3", "NH FV DCMIP TC3 (cube-sphere)", "atmosphere", "nh_fv_tc3"),
]


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
        return headers, np.zeros((0, len(headers)))
    data = np.zeros((len(rows), len(headers)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, h in enumerate(headers):
            try:
                data[i, j] = float(row.get(h, "nan"))
            except Exception:
                data[i, j] = np.nan
    return headers, data


def _derive_conservation_from_mean(case_dir: Path) -> None:
    mean_csv = case_dir / "mean_timeseries.csv"
    if not mean_csv.exists():
        return

    headers, data = _load_csv_columns(mean_csv)
    if data.size == 0:
        return

    idx = {k: i for i, k in enumerate(headers)}
    if "step" not in idx:
        step = np.arange(data.shape[0], dtype=np.float64)
    else:
        step = data[:, idx["step"]]
    if "time_days" in idx:
        t_days = data[:, idx["time_days"]]
    elif "time_seconds" in idx:
        t_days = data[:, idx["time_seconds"]] / 86400.0
    else:
        t_days = np.arange(data.shape[0], dtype=np.float64)

    mass_candidates = [
        "mean_p_s",
        "mean_height",
        "mean_q1_3d",
        "mean_rho_prime_3d",
        "mean_T_3d",
    ]
    energy_candidates = [
        "mean_wind_3d",
        "mean_wind_speed",
        "mean_T_3d",
        "mean_abs_w_3d",
    ]

    mass_key = next((k for k in mass_candidates if k in idx), None)
    energy_key = next((k for k in energy_candidates if k in idx), None)
    if mass_key is None and energy_key is None:
        return

    mass = data[:, idx[mass_key]] if mass_key is not None else np.ones_like(t_days)
    energy = data[:, idx[energy_key]] if energy_key is not None else np.ones_like(t_days)
    mass_rel = (mass - mass[0]) / max(abs(mass[0]), 1.0e-30)
    energy_rel = (energy - energy[0]) / max(abs(energy[0]), 1.0e-30)

    out_csv = case_dir / "conservation_timeseries.csv"
    with out_csv.open("w") as f:
        f.write("step,time_days,mass_proxy,energy_proxy,mass_rel,energy_rel\n")
        for i in range(step.size):
            f.write(
                f"{step[i]:.0f},{t_days[i]:.8f},"
                f"{mass[i]:.12e},{energy[i]:.12e},"
                f"{mass_rel[i]:.12e},{energy_rel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    axes[0].plot(t_days, mass_rel, lw=1.8, color="tab:blue")
    axes[0].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[0].set_ylabel("Mass proxy drift")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(f"{mass_key or 'mass'} proxy conservation")

    axes[1].plot(t_days, energy_rel, lw=1.8, color="tab:red")
    axes[1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[1].set_ylabel("Energy proxy drift")
    axes[1].set_xlabel("Time (days)")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title(f"{energy_key or 'energy'} proxy conservation")

    fig.tight_layout()
    fig.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ensure_standard_artifacts(case_dir: Path) -> dict[str, bool]:
    copied = {
        "field_snapshots.png": (case_dir / "field_snapshots.png").exists(),
        "snapshot_times.txt": (case_dir / "snapshot_times.txt").exists(),
        "mean_timeseries.csv": False,
        "mean_timeseries.png": False,
        "conservation_timeseries.csv": False,
        "conservation_timeseries.png": False,
        "mean_profiles.csv": False,
        "mean_profiles.png": False,
    }

    copied["mean_timeseries.csv"] = _copy_if_exists(
        case_dir / "integrated_timeseries.csv",
        case_dir / "mean_timeseries.csv",
    ) or _copy_if_exists(case_dir / "mean_fields_timeseries.csv", case_dir / "mean_timeseries.csv")
    copied["mean_timeseries.png"] = _copy_if_exists(
        case_dir / "integrated_timeseries.png",
        case_dir / "mean_timeseries.png",
    ) or _copy_if_exists(case_dir / "mean_fields_timeseries.png", case_dir / "mean_timeseries.png")

    copied["conservation_timeseries.csv"] = _copy_if_exists(
        case_dir / "conservation_timeseries.csv",
        case_dir / "conservation_timeseries.csv",
    )
    copied["conservation_timeseries.png"] = _copy_if_exists(
        case_dir / "conservation_timeseries.png",
        case_dir / "conservation_timeseries.png",
    )
    if not copied["conservation_timeseries.csv"] or not copied["conservation_timeseries.png"]:
        _derive_conservation_from_mean(case_dir)
        copied["conservation_timeseries.csv"] = (case_dir / "conservation_timeseries.csv").exists()
        copied["conservation_timeseries.png"] = (case_dir / "conservation_timeseries.png").exists()

    copied["mean_profiles.csv"] = _copy_if_exists(
        case_dir / "vertical_profiles.csv",
        case_dir / "mean_profiles.csv",
    ) or _copy_if_exists(case_dir / "slab_timeseries.csv", case_dir / "mean_profiles.csv")
    copied["mean_profiles.png"] = _copy_if_exists(
        case_dir / "vertical_profiles.png",
        case_dir / "mean_profiles.png",
    ) or _copy_if_exists(case_dir / "slab_timeseries.png", case_dir / "mean_profiles.png")

    if not copied["mean_profiles.csv"]:
        with (case_dir / "mean_profiles.csv").open("w") as f:
            f.write("note\nno_vertical_profile_available_for_this_case\n")
        copied["mean_profiles.csv"] = True
    if not copied["mean_profiles.png"]:
        fig, ax = plt.subplots(figsize=(6.4, 2.2))
        ax.text(0.5, 0.5, "No vertical profile available for this case", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(case_dir / "mean_profiles.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        copied["mean_profiles.png"] = True

    with (case_dir / "standardized_manifest.json").open("w") as f:
        json.dump(copied, f, indent=2)
    return copied


def _run(cmd: list[str], cwd: Path) -> tuple[str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    wall = time.time() - t0
    return ("PASS" if proc.returncode == 0 else "FAIL"), wall


def _copy_case_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ordered dycore progression suite with unified artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/validation/results/dycore_progression_suite"),
        help="Unified output root under tests/",
    )
    parser.add_argument("--cube-resolution", type=int, default=36)
    parser.add_argument("--latlon-nlat", type=int, default=72)
    parser.add_argument("--latlon-nlon", type=int, default=144)
    parser.add_argument("--spectral-truncation", type=int, default=42)
    parser.add_argument("--hydro-levels", type=int, default=20)
    parser.add_argument("--nh-levels", type=int, default=20)
    parser.add_argument("--solver", type=str, default="ssp45")
    parser.add_argument("--mean-every", type=int, default=20)
    parser.add_argument("--sw-days", type=float, default=5.0)
    parser.add_argument(
        "--projection",
        type=str,
        default="platecarree",
        choices=("platecarree", "robinson"),
        help="Map projection for all case snapshots.",
    )
    parser.add_argument("--coastlines", action="store_true", help="Draw coastlines on maps.")
    parser.add_argument("--skip-run", action="store_true", help="Only write the plan/manifest without executing.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Run only the first N ordered cases (0 = all). Useful for smoke checks.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(".venv/bin/python"),
        help="Python executable to run child scripts.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_exec = args.python if args.python.is_absolute() else (repo_root / args.python)
    python_exec = python_exec.expanduser()
    out_root = (repo_root / args.output).resolve()
    raw_root = out_root / "_raw"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    suite_records: list[dict[str, object]] = []
    selected_cases = ORDERED_CASES if args.max_cases <= 0 else ORDERED_CASES[: args.max_cases]
    selected_ids = {c.case_id for c in selected_cases}

    # Step 1: shallow-water multi-grid run (produces three standardized cases).
    needs_shallow = any(c.source_kind == "shallow" for c in selected_cases)
    if needs_shallow:
        sw_raw_out = raw_root / "sw_compare"
        sw_cmd = [
            str(python_exec),
            "scripts/run_shallow_water_latlon_cube_compare.py",
            "--output",
            str(sw_raw_out),
            "--days",
            str(args.sw_days),
            "--cube-resolution",
            str(args.cube_resolution),
            "--spectral-truncation",
            str(args.spectral_truncation),
            "--projection",
            args.projection,
        ]
        if args.coastlines:
            sw_cmd.append("--coastlines")

        if args.skip_run:
            sw_status, sw_wall = "SKIPPED", 0.0
        else:
            sw_status, sw_wall = _run(sw_cmd, repo_root)

        shallow_pairs = [(k, v) for k, v in SW_SUBCASE_MAP.items() if v in selected_ids]
        if sw_status == "PASS" or args.skip_run:
            for src_name, case_id in shallow_pairs:
                src = sw_raw_out / src_name
                dst = out_root / case_id
                if src.exists():
                    _copy_case_tree(src, dst)
                    artifacts = _ensure_standard_artifacts(dst)
                    status = "PASS" if not args.skip_run else "SKIPPED"
                    notes = "from shallow-water compare suite"
                else:
                    artifacts = {}
                    status = "FAIL" if not args.skip_run else "SKIPPED"
                    notes = f"missing source subdir: {src_name}"
                suite_records.append(
                    {
                        "case_id": case_id,
                        "title": next(c.title for c in ORDERED_CASES if c.case_id == case_id),
                        "source": src_name,
                        "status": status,
                        "wall_time_s": sw_wall,
                        "artifacts": artifacts,
                        "notes": notes,
                    },
                )
        else:
            for src_name, case_id in shallow_pairs:
                suite_records.append(
                    {
                        "case_id": case_id,
                        "title": next(c.title for c in ORDERED_CASES if c.case_id == case_id),
                        "source": src_name,
                        "status": "FAIL",
                        "wall_time_s": sw_wall,
                        "artifacts": {},
                        "notes": "shallow-water aggregate run failed",
                    },
                )

    # Step 2+: ordered hydro/NH progression from atmosphere suite (one case at a time).
    atm_cases = [c for c in selected_cases if c.source_kind == "atmosphere"]
    for case in atm_cases:
        run_out = raw_root / case.case_id
        run_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python_exec),
            "scripts/run_atmosphere_25deg_ssp45_full.py",
            "--output",
            str(run_out),
            "--cube-resolution",
            str(args.cube_resolution),
            "--latlon-nlat",
            str(args.latlon_nlat),
            "--latlon-nlon",
            str(args.latlon_nlon),
            "--spectral-truncation",
            str(args.spectral_truncation),
            "--hydro-levels",
            str(args.hydro_levels),
            "--nh-levels",
            str(args.nh_levels),
            "--solver",
            args.solver,
            "--mean-every",
            str(args.mean_every),
            "--projection",
            args.projection,
            "--only",
            case.source_key,
        ]
        if args.coastlines:
            cmd.append("--coastlines")

        if args.skip_run:
            status, wall = "SKIPPED", 0.0
        else:
            status, wall = _run(cmd, repo_root)

        produced = sorted(
            p for p in run_out.iterdir() if p.is_dir() and (p / "field_snapshots.png").exists()
        )
        if status == "PASS" and produced:
            src_case = produced[0]
            dst_case = out_root / case.case_id
            _copy_case_tree(src_case, dst_case)
            artifacts = _ensure_standard_artifacts(dst_case)
            notes = f"source_case_dir={src_case.name}"
        elif status == "SKIPPED":
            artifacts = {}
            notes = "execution skipped"
        else:
            artifacts = {}
            notes = "case run failed or no case directory produced"

        suite_records.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "source": case.source_key,
                "status": status,
                "wall_time_s": wall,
                "artifacts": artifacts,
                "notes": notes,
            },
        )

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "cube_resolution": args.cube_resolution,
            "latlon_nlat": args.latlon_nlat,
            "latlon_nlon": args.latlon_nlon,
            "spectral_truncation": args.spectral_truncation,
            "hydro_levels": args.hydro_levels,
            "nh_levels": args.nh_levels,
            "solver": args.solver,
            "mean_every": args.mean_every,
            "sw_days": args.sw_days,
            "projection": args.projection,
            "coastlines": bool(args.coastlines),
            "skip_run": bool(args.skip_run),
        },
        "ordered_cases": [
            {"rank": c.rank, "case_id": c.case_id, "title": c.title, "source_kind": c.source_kind}
            for c in selected_cases
        ],
        "results": suite_records,
        "notes": [
            "NH FV lat-lon full-benchmark case is not currently included (no dedicated full test-case runner).",
            "Conservation diagnostics are copied when available; otherwise a proxy timeseries is derived from mean fields.",
        ],
    }
    with (out_root / "suite_manifest.json").open("w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Dycore Progression Suite",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Ordered Complexity Ladder",
        "",
    ]
    for c in ORDERED_CASES:
        lines.append(f"{c.rank}. `{c.case_id}`: {c.title}")
    lines.extend(
        [
            "",
            "## Standardized Artifacts Per Case",
            "",
            "- `field_snapshots.png`",
            "- `snapshot_times.txt`",
            "- `mean_timeseries.csv` / `mean_timeseries.png`",
            "- `conservation_timeseries.csv` / `conservation_timeseries.png`",
            "- `mean_profiles.csv` / `mean_profiles.png`",
            "",
            "## Case Status",
            "",
            "| Case | Status | Wall (s) | Notes |",
            "|---|---|---:|---|",
        ],
    )
    for r in suite_records:
        lines.append(
            f"| {r['case_id']} | {r['status']} | {float(r.get('wall_time_s', 0.0)):.2f} | {r.get('notes', '')} |",
        )

    with (out_root / "SUITE_PLAN.md").open("w") as f:
        f.write("\n".join(lines) + "\n")

    n_pass = sum(1 for r in suite_records if r["status"] == "PASS")
    n_fail = sum(1 for r in suite_records if r["status"] == "FAIL")
    n_skip = sum(1 for r in suite_records if r["status"] == "SKIPPED")
    print(f"Suite prepared at: {out_root}")
    print(f"PASS={n_pass} FAIL={n_fail} SKIPPED={n_skip} TOTAL={len(suite_records)}")


if __name__ == "__main__":
    main()
