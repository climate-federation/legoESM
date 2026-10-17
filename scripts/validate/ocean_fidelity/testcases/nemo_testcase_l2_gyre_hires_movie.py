#!/usr/bin/env python3
"""Render common-scale NEMO/legoESM GYRE movies for years 1 and 5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from nemo_testcase_l2_gyre_hires_climate import (  # noqa: E402
    DailyFrame,
    cell_velocity,
    geometry,
    legoesm_npz_provider,
    nemo_ioipsl_provider,
    sha256,
)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402


def movie_fields(frame: DailyFrame, geom: dict, grid) -> dict[str, np.ndarray]:
    """Return the four registered panels using shared C-grid diagnostics."""
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre

    u_native = np.where(geom["u_mask"], frame.u, 0.0)
    v_native = np.where(geom["v_mask"], frame.v, 0.0)
    u = np.concatenate([np.zeros_like(u_native[:, :1]), u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
    zeta = np.asarray(relative_vorticity_cell_centre(u, v, grid))
    u_c, v_c = cell_velocity(frame.u, frame.v, geom)
    wet = geom["wet"]
    return {
        "relative vorticity / f": np.where(wet, zeta / geom["f"], np.nan),
        "SSH (m)": np.where(wet, frame.ssh, np.nan),
        "surface speed (m/s)": np.where(wet, np.sqrt(u_c * u_c + v_c * v_c), np.nan),
        "SST (C)": np.where(wet, frame.sst, np.nan),
    }


def common_scales(providers, geom: dict, grid) -> dict[str, tuple[float, float]]:
    ranges: dict[str, list[float]] = {}
    wet = np.asarray(geom["wet"], dtype=bool)
    require(wet.ndim == 2 and np.any(wet), "movie wet mask is empty or invalid")
    for provider in providers:
        for frame in provider.iterate():
            for name, value in movie_fields(frame, geom, grid).items():
                require(value.shape == wet.shape, f"{name}: movie field shape differs")
                require(np.all(np.isfinite(value[wet])),
                        f"{name}: non-finite wet movie pixel")
                finite = value[wet]
                if name not in ranges:
                    ranges[name] = [float(np.min(finite)), float(np.max(finite))]
                else:
                    ranges[name][0] = min(ranges[name][0], float(np.min(finite)))
                    ranges[name][1] = max(ranges[name][1], float(np.max(finite)))
    for name in ("relative vorticity / f", "SSH (m)"):
        limit = max(abs(ranges[name][0]), abs(ranges[name][1]))
        ranges[name] = [-limit, limit]
    ranges["surface speed (m/s)"][0] = 0.0
    return {name: (value[0], value[1]) for name, value in ranges.items()}


def render_frame(
    frames: tuple[DailyFrame, DailyFrame], names: tuple[str, str],
    scales: dict[str, tuple[float, float]], geom: dict, grid,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(frames[0].day == frames[1].day, "movie model days differ")
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.4), dpi=100,
                             constrained_layout=True)
    for row, (frame, model) in enumerate(zip(frames, names, strict=True)):
        fields = movie_fields(frame, geom, grid)
        for col, (name, value) in enumerate(fields.items()):
            vmin, vmax = scales[name]
            image = axes[row, col].imshow(
                value, origin="lower", vmin=vmin, vmax=vmax,
                cmap="RdBu_r" if vmin < 0.0 else "viridis",
            )
            axes[row, col].set_title(f"{model}: {name}", fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(image, ax=axes[row, col], shrink=0.72)
    fig.suptitle(f"GYRE r4_smag daily mean, day {frames[0].day}")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def render_year(
    nemo_root: Path, lego_run_label: str, output: Path, *, year: int, fps: int,
    registry_path: Path,
) -> dict:
    import imageio_ffmpeg
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_gyre_zco_card,
    )

    require(year in (1, 5), "only preregistered years 1 and 5 may be rendered")
    start = 1 + (year - 1) * 365
    providers = (
        nemo_ioipsl_provider(nemo_root, 365, start_day=start),
        legoesm_npz_provider(
            lego_run_label, 365, start_day=start, registry_path=registry_path,
        ),
    )
    geom = geometry()
    grid = build_gyre_zco_card(GYRE_R4_SMAG_NAMELIST).recipe.grid
    scales = common_scales(providers, geom, grid)
    writer = imageio_ffmpeg.write_frames(
        str(output), (1280, 640), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=16,
    )
    writer.send(None)
    count = 0
    try:
        for left, right in zip(providers[0].iterate(), providers[1].iterate(), strict=True):
            image = render_frame((left, right), (providers[0].name, providers[1].name),
                                 scales, geom, grid)
            require(image.shape == (640, 1280, 3) and image.dtype == np.uint8,
                    "rendered frame format differs")
            writer.send(np.ascontiguousarray(image))
            count += 1
    finally:
        writer.close()
    require(count == 365, "movie did not receive 365 daily frames")
    return {
        "year": year,
        "window_days": [start, start + 364],
        "frames": count,
        "fps": fps,
        "scales": scales,
        "provenance": {provider.name: provider.provenance for provider in providers},
        "sha256": sha256(output),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-run-label", required=True)
    parser.add_argument(
        "--run-registry", type=Path,
        default=HERE / "gyre_hires_registered_runs.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", choices=(1, 5), default=(1, 5))
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        require(args.fps > 0, "fps must be positive")
        args.output.mkdir(parents=True, exist_ok=False)
        reports = []
        for year in args.years:
            reports.append(render_year(
                args.nemo_root, args.legoesm_run_label,
                args.output / f"gyre_r4_smag_year{year}.mp4", year=year, fps=args.fps,
                registry_path=args.run_registry,
            ))
        report = {"schema": "gyre-r4-smag-side-by-side-movies-v1",
                  "models": ["NEMO", "legoESM"],
                  "panels": ["relative vorticity / f", "SSH", "surface speed", "SST"],
                  "daily_mean_frames": reports}
        (args.output / "movie_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
