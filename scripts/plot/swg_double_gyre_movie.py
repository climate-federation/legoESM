#!/usr/bin/env python3
"""Render matched-sampling NEMO/legoESM SWG_DG fields side by side."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKOUT_TOOLS = (
    REPO_ROOT / "scripts/validate/ocean_fidelity/testcases")
sys.path.insert(0, str(CHECKOUT_TOOLS))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402

CHECKOUT_IMPORT = bootstrap_checkout(__file__, enforce_x64=__name__ == "__main__")
CLIMATE_PATH = (
    REPO_ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l1_swg_dg_climate.py")
FRAME_SIZE = (1280, 720)


def _load_climate():
    spec = importlib.util.spec_from_file_location("swg_dg_climate_movie", CLIMATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load climate provider {CLIMATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CLIMATE_PATH.parent))
    spec.loader.exec_module(module)
    return module


def derive_fields(frame, card) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-centred relative-vorticity/f, SSH, and speed fields."""
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import (
        relative_vorticity_cell_centre,
        velocity_to_cell_centre,
    )

    state = card.recipe.initial_state
    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    umask = np.asarray(state.u_mask.data[:, 1:], dtype=bool)
    vmask = np.asarray(state.v_mask.data[1:, :], dtype=bool)
    u_native = np.where(umask, frame.u, 0.0)
    v_native = np.where(vmask, frame.v, 0.0)
    u = jnp.asarray(np.concatenate([np.zeros_like(u_native[:, :1]), u_native], axis=1))
    v = jnp.asarray(np.concatenate([np.zeros_like(v_native[:1, :]), v_native], axis=0))
    u_c, v_c = velocity_to_cell_centre(u, v)
    zeta = relative_vorticity_cell_centre(u, v, card.recipe.grid)
    coriolis = np.asarray(card.recipe.grid.f_T)
    relative = np.asarray(zeta) / coriolis
    speed = np.hypot(np.asarray(u_c), np.asarray(v_c))
    return (
        np.where(tmask, relative, np.nan),
        np.where(tmask, frame.ssh, np.nan),
        np.where(tmask, speed, np.nan),
    )


def _shared_limits(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]):
    limits = []
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        joined = np.concatenate([first[np.isfinite(first)], second[np.isfinite(second)]])
        if not joined.size:
            raise RuntimeError("movie field contains no finite wet values")
        if index < 2:
            bound = float(np.max(np.abs(joined)))
            limits.append((-bound if bound else -1.0, bound if bound else 1.0))
        else:
            upper = float(np.max(joined))
            limits.append((0.0, upper if upper else 1.0))
    return limits


def render_frame(
        nemo_frame, lego_frame, card, *, absolute_day: int,
) -> np.ndarray:
    """Render one common-scale RGB frame for the two matched model states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nemo = derive_fields(nemo_frame, card)
    lego = derive_fields(lego_frame, card)
    limits = _shared_limits(nemo, lego)
    titles = ("relative vorticity / f", "sea-surface height (m)", "speed (m/s)")
    cmaps = ("RdBu_r", "RdBu_r", "magma")
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), dpi=100)
    for row, (label, fields) in enumerate((("NEMO", nemo), ("legoESM", lego))):
        for column, field in enumerate(fields):
            low, high = limits[column]
            image = axes[row, column].imshow(
                field, origin="lower", cmap=cmaps[column],
                vmin=low, vmax=high, interpolation="nearest")
            axes[row, column].set_title(f"{label}: {titles[column]}")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.suptitle(f"SWG double gyre, matched daily means, day {absolute_day}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.canvas.draw()
    rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return rgb


def write_movie(
        nemo_root: Path, legoesm_root: Path, output: Path, *,
        start_day: int, days: int, fps: int,
) -> None:
    """Stream matched daily means through imageio-ffmpeg without loading a year."""
    import imageio_ffmpeg

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("movie output must end in .mp4")
    if start_day <= 0 or days <= 0 or fps <= 0:
        raise ValueError("start-day, days, and fps must be positive")
    # Python 3.13 correctly warns that forking after JAX starts worker threads
    # can deadlock.  Start ffmpeg first, with the renderer's fixed pixel size,
    # and publish atomically only after all validated frames have been written.
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    if temporary.exists():
        raise FileExistsError(
            f"stale partial movie exists; inspect or remove {temporary}")
    writer = imageio_ffmpeg.write_frames(
        temporary, FRAME_SIZE, fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7)
    writer.send(None)
    count = 0
    try:
        from legoesm.ocean.fidelity.nemo_testcase_recipe import (
            SWG_MADDISON_NAMELIST,
            build_swg_card,
        )

        climate = _load_climate()
        providers = (
            climate.nemo_ioipsl_provider(
                nemo_root, days, start_day=start_day),
            climate.legoesm_npz_provider(
                legoesm_root, days, start_day=start_day),
        )
        if providers[0].sampling != providers[1].sampling:
            raise RuntimeError(
                f"sampling mismatch: NEMO={providers[0].sampling}, "
                f"legoESM={providers[1].sampling}")
        if providers[0].sampling != "daily_mean":
            raise RuntimeError(
                "side-by-side movie requires matched daily means")
        card = build_swg_card(SWG_MADDISON_NAMELIST)
        frames = zip(
            providers[0].iterate(), providers[1].iterate(), strict=True)
        first_nemo, first_lego = next(frames)
        first = render_frame(
            first_nemo, first_lego, card, absolute_day=start_day)
        if first.shape[:2][::-1] != FRAME_SIZE:
            raise RuntimeError(
                f"renderer returned {first.shape[:2][::-1]}, expected "
                f"{FRAME_SIZE}")
        writer.send(first)
        count = 1
        for count, (nemo_frame, lego_frame) in enumerate(frames, 2):
            writer.send(render_frame(
                nemo_frame, lego_frame, card,
                absolute_day=start_day + count - 1))
    finally:
        writer.close()
    if count != days:
        raise RuntimeError(f"movie provider yielded {count} frames, expected {days}")
    temporary.replace(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        write_movie(
            args.nemo_root, args.legoesm_root, args.output,
            start_day=args.start_day, days=args.days, fps=args.fps)
    except Exception as exc:  # noqa: BLE001 - fail-closed plotting boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"SWG_DG_MOVIE COMPLETE frames={args.days} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
