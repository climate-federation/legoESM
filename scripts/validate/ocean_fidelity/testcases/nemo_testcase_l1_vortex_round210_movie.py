#!/usr/bin/env python3
"""Movie of the round-210 VORTEX_VEC-zco 100-day comparison (legoESM vs NEMO).

Plotting only -- no model run, no option change. Reuses, rather than
re-derives, the round-210 scorer's exact readers and masks:

* ``load_nemo`` / ``load_lego`` / ``CARDS`` from
  ``nemo_testcase_l1_vortex_round210_100day_comparison.py`` -- the same
  NEMO-restart axis contract (``time_counter, nav_lev, y, x`` ->
  ``transpose(1, 2, 0)``) and the same daily legoESM snapshot reader.
* ``expected_masks`` from ``nemo_testcase_phase3_trajectory_gate.py`` -- the
  same 61x61-of-63x63 wet mask the certified kt-ladder and round 210 both
  score against.

See the round-210 receipt:
``docs/ocean/fidelity/testcases/nemo_testcases_l1_vortex_round210_100day_comparison_receipt.md``.

Vector card (``VORTEX_VEC-zco``) only, per this task's order. Three panels
per frame: NEMO ssh+quiver, legoESM ssh+quiver (shared colour scale), and
their difference (its own colour scale, FIXED across all frames at the
day-100 max |dssh|).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_l1_vortex_round210_100day_comparison import (  # noqa: E402
    CARDS, N_DAYS, load_lego, load_nemo,
)
from nemo_testcase_phase3_trajectory_gate import expected_masks  # noqa: E402

# ROUND 216: the same three panels for any card the round-210 scorer knows,
# selected by --card, with the seamount's bathymetry drawn over every panel.
# The vector flat card stays the default, so the round-210 invocation is
# unchanged.
OUT_DIR = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round210")
TAG = "vec"
LEGO_DIR = OUT_DIR / "lego_vec"
CASE, NEMO_DIR = CARDS["vec"]
BATHY_KM = None          # (X, Y, depth_m) when the card has real topography
QUIVER_STRIDE = 3
DPI = 100
FIGSIZE = (14.0, 4.6)
FPS = 10
MONTAGE_DAYS = (1, 30, 60, 100)


def _rms(diff: np.ndarray, mask: np.ndarray) -> float:
    # Copied verbatim from the round-210 scorer's `_rms` (same formula,
    # same mask convention) -- not re-derived.
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def _max(diff: np.ndarray, mask: np.ndarray) -> float:
    # Copied verbatim from the round-210 scorer's `_max`.
    return float(np.max(np.abs(diff[mask])))


def _masked(arr2d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(arr2d, dtype=float)
    out[~mask] = np.nan
    return out


def _build_card():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    return build_nemo_testcase_card(CASE)


def _load_day(day: int, nlev: int) -> tuple[dict, dict]:
    return load_lego(LEGO_DIR, day), load_nemo(NEMO_DIR, day, nlev)


def _render_panels(fig, axes, day, lego_d, nemo_d, masks, X, Y, u_mask,
                    v_mask, fixed_dssh, qscale):
    wet2d = masks["ssh"]
    sl = slice(None, None, QUIVER_STRIDE)

    ssh_n = _masked(nemo_d["ssh"], wet2d)
    ssh_l = _masked(lego_d["ssh"], wet2d)
    vmax = float(np.nanmax(np.abs(np.stack([ssh_n, ssh_l]))))

    un = np.ma.masked_invalid(_masked(nemo_d["u"][:, :, 0], u_mask))
    vn = np.ma.masked_invalid(_masked(nemo_d["v"][:, :, 0], v_mask))
    ul = np.ma.masked_invalid(_masked(lego_d["u"][:, :, 0], u_mask))
    vl = np.ma.masked_invalid(_masked(lego_d["v"][:, :, 0], v_mask))

    diff = lego_d["ssh"] - nemo_d["ssh"]
    frame_max = _max(diff, wet2d)
    T_rms = _rms(lego_d["T"] - nemo_d["T"], masks["T"])
    diff_plot = _masked(diff, wet2d)

    titles = ("NEMO", "legoESM",
              f"legoESM - NEMO\nmax|d(ssh)|={frame_max:.2e} m  T_rms={T_rms:.2e} K")
    for ax in axes:
        ax.cla()

    im0 = axes[0].pcolormesh(X, Y, ssh_n, cmap="RdBu_r", vmin=-vmax,
                              vmax=vmax, shading="nearest")
    axes[0].quiver(X[sl, sl], Y[sl, sl], un[sl, sl], vn[sl, sl],
                    color="k", scale=qscale, width=0.0035)
    im1 = axes[1].pcolormesh(X, Y, ssh_l, cmap="RdBu_r", vmin=-vmax,
                              vmax=vmax, shading="nearest")
    axes[1].quiver(X[sl, sl], Y[sl, sl], ul[sl, sl], vl[sl, sl],
                    color="k", scale=qscale, width=0.0035)
    im2 = axes[2].pcolormesh(X, Y, diff_plot, cmap="RdBu_r",
                              vmin=-fixed_dssh, vmax=fixed_dssh,
                              shading="nearest")
    if BATHY_KM is not None:
        # The card's OWN resolved bathymetry, drawn identically on all three
        # panels so the seamount's position is readable in the difference.
        bx, by, depth = BATHY_KM
        for ax in axes:
            ax.contour(bx, by, depth, levels=8, colors="k",
                        linewidths=0.5, alpha=0.55)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [km]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [km]")
    fig.suptitle(f"{CASE}: legoESM vs NEMO, day {day}/100")
    return im0, im1, im2


def main(argv=None) -> int:
    global TAG, CASE, NEMO_DIR, LEGO_DIR, OUT_DIR, BATHY_KM
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", default="vec",
                     help="card tag known to the round-210 scorer")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if args.card not in CARDS:
        raise SystemExit(f"REFUSE: unknown card tag {args.card!r}; "
                         f"expected from {sorted(CARDS)}")
    TAG = args.card
    CASE, NEMO_DIR = CARDS[TAG]
    OUT_DIR = args.out
    LEGO_DIR = OUT_DIR / f"lego_{TAG}"
    card = _build_card()
    nlev = card.recipe.z_coord.n_levels
    masks = expected_masks(card)
    for name, arr in masks.items():
        print(f"mask[{name}] shape={arr.shape} wet={int(arr.sum())}/{arr.size}")

    dx_km = float(np.asarray(card.recipe.grid.dx_T)[0, 0]) / 1000.0
    dy_km = float(np.asarray(card.recipe.grid.dy_T)[0, 0]) / 1000.0
    ny, nx = masks["ssh"].shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx_km,
                        (np.arange(ny) + 0.5) * dy_km)
    u_mask, v_mask = masks["u"][:, :, 0], masks["v"][:, :, 0]
    depth = np.asarray(card.recipe.initial_state.H_bathy.data,
                        dtype=np.float64)
    if float(depth.max() - depth.min()) > 1.0:
        BATHY_KM = (X, Y, np.where(masks["ssh"], depth, np.nan))
        print(f"bathymetry contours ON: depth {depth.min():.1f}-"
              f"{depth.max():.1f} m")
    else:
        print("bathymetry contours OFF: this card is flat-bottomed")

    lego1, nemo1 = _load_day(1, nlev)
    print(f"day001: lego ssh {lego1['ssh'].shape} u {lego1['u'].shape}  "
          f"nemo ssh {nemo1['ssh'].shape} u {nemo1['u'].shape}")
    # Fixed quiver scale (data units per arrow-length unit) from day-1's
    # surface speed, so arrow length visibly shrinks as the vortex spins
    # down instead of each frame re-autoscaling to hide that decay.
    qscale = 10.0 * float(np.nanmax(np.abs(nemo1["u"][:, :, 0])) + 1e-30)

    lego100, nemo100 = _load_day(N_DAYS, nlev)
    print(f"day100: lego ssh {lego100['ssh'].shape}  nemo ssh {nemo100['ssh'].shape}")
    fixed_dssh = _max(lego100["ssh"] - nemo100["ssh"], masks["ssh"])
    print(f"FIXED day-100 max|d(ssh)| ({CASE}) = {fixed_dssh:.6e} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    cbar_lr = cbar_diff = None
    frames = []
    for day in range(1, N_DAYS + 1):
        lego_d, nemo_d = _load_day(day, nlev)
        im0, im1, im2 = _render_panels(fig, axes, day, lego_d, nemo_d, masks,
                                        X, Y, u_mask, v_mask, fixed_dssh,
                                        qscale)
        if cbar_lr is None:
            cbar_lr = fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=0.85,
                                    label="ssh [m]")
            cbar_diff = fig.colorbar(im2, ax=axes[2], shrink=0.85,
                                      label="ssh diff [m]")
        else:
            cbar_lr.update_normal(im1)
            cbar_diff.update_normal(im2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        if day % 20 == 0 or day == 1:
            print(f"rendered day {day}/{N_DAYS}")
    plt.close(fig)

    mp4_path = OUT_DIR / f"vortex_{TAG}_100d.mp4"
    imageio.mimsave(mp4_path, frames, fps=FPS, codec="libx264", quality=8)
    print(f"WROTE {mp4_path} ({mp4_path.stat().st_size / 1e6:.2f} MB, "
          f"{len(frames)} frames)")

    gif_path = OUT_DIR / f"vortex_{TAG}_100d.gif"
    from PIL import Image
    pil_frames = [Image.fromarray(f).convert(
        "P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=int(1000 / FPS), loop=0, optimize=True)
    gif_mb = gif_path.stat().st_size / 1e6
    print(f"WROTE {gif_path} ({gif_mb:.2f} MB, {len(frames)} frames)")
    if gif_mb > 25:
        print("WARNING: GIF exceeds the 25 MB budget")

    # ---- montage PNG: days 1, 30, 60, 100 x 3 panels ----
    fig2, axes2 = plt.subplots(3, len(MONTAGE_DAYS),
                                figsize=(4.2 * len(MONTAGE_DAYS), 11), dpi=110)
    for col, day in enumerate(MONTAGE_DAYS):
        lego_d, nemo_d = _load_day(day, nlev)
        _render_panels(fig2, axes2[:, col], day, lego_d, nemo_d, masks, X, Y,
                       u_mask, v_mask, fixed_dssh, qscale)
        for row in range(3):
            fig2.colorbar(axes2[row, col].collections[0], ax=axes2[row, col],
                           shrink=0.8)
    fig2.suptitle(f"{CASE}: legoESM vs NEMO at days {MONTAGE_DAYS}")
    fig2.tight_layout()
    png_path = OUT_DIR / f"vortex_{TAG}_frames.png"
    fig2.savefig(png_path)
    plt.close(fig2)
    print(f"WROTE {png_path} ({png_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
