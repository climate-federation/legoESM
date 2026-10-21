#!/usr/bin/env python3
"""Eight-panel movie of the rigid-lid SWG double gyre: fine truth, truth filtered to the coarse deck, the no-closure
control and five trained-closure free runs, all as relative vorticity / f0 on the same model days.

Pattern from ``session_2026-10-14_scratch/swg_dg_movie.py`` (FuncAnimation + imageio-ffmpeg + libx264); fields are the
scorer's own: day files read under their pinned digests, truth restricted by ``truth_restrictor("symmetric")`` from
``nemo_testcase_l1_swg_online_evaluation.py``, vorticity by ``restricted_relative_vorticity`` (C-grid curl, 4-point
averaged to T points).  All runs start from rest, so a calendar day is comparable statistically, not phase for phase.

usage (CPU only):  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/plot/plot_swg_online_movie.py OUT_DIR
      [--first-day 3651 --last-day 4015 --step 2 --fps 15]
Writes OUT_DIR/{swg_online_comparison.mp4, ..._midframe.png, ..._timemean.png} and caches vorticity in OUT_DIR/_cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py"
P = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
R21 = P / "swgonl_rounds/round21_neuralgcm"
SCORES = P / "swgonl_rounds/round19_limits/arm_e_scores/scores"
TRUTH = P / "swgdg_rounds/legoesm_runs/dg_rl_gpu_7300d_truth"
# No-closure control on the closure runs' own deck (dg50-rl; the nu80 control differs only in rn_Lv_m 80 vs 10).
NO_CLOSURE = P / "swgonl_rounds/round18_rigidlid_scores/gpu_controls/swgonl_control_nu10_rl_gpu"
# (key, label, run root, score file label); order = panels 4-8 of the figure
RUNS = (
    ("ng8", "NeuralGCM-style det., 8 d (best transport & map, too energetic)",
     R21 / "rerun_seed13/swgonl_exp9_rigidlid_neuralgcm/seed20261013/free_runs/w08", "exp9long_s13rerun_w08"),
    ("ng16", "NeuralGCM-style det., 16 d (most balanced)",
     R21 / "swgonl_exp9_rigidlid_neuralgcm/seed20261015/free_runs/w16", "exp9long_s15_w16"),
    ("es8", "Energy score + filter, 8 d (exact energy, transport fails)",
     R21 / "swgonl_exp10_rigidlid_energy_score_neuralgcm_m8/seed20261015/free_runs/w08", "exp10_esneuralgcm_s15_w08"),
    ("pz60", "Perezhogin-style stats, 60 d",
     P / "swgonl_rounds/round19_limits/arm_e_free_runs/transport_eke_all_T60_step50", "transport_eke_all_T60_step50"),
    ("ng32", "NeuralGCM-style det., 32 d (good spectrum, damped)",
     R21 / "swgonl_exp9_rigidlid_neuralgcm/seed20261014/free_runs/w32", "exp9long_s14_w32"),
)
SCORE_KEYS = ("mean_face_transport_rmse_m3_s", "wet_eke_map_nrmse", "basin_eke_ratio")
BARS = "bars: transport < 0.93e6 m3/s | EKE map < 0.20 | basin EKE 0.93-1.07"


def frame_days(first: int, last: int, step: int) -> list[int]:
    if not (1 <= first <= last and step >= 1):
        raise ValueError(f"bad day range {first}..{last} step {step}")
    return list(range(first, last + 1, step))


def score_line(medians: dict) -> str:
    t, e, b = (medians[k] for k in SCORE_KEYS)
    return f"transport {t / 1e6:.2f}e6 | EKE map {e:.2f} | basin EKE {b:.2f}"


def symmetric_limit(arrays, quantile: float) -> float:
    """One symmetric colour limit: the ``quantile`` of |value| pooled over every array (NaN ignored)."""
    pooled = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrays])
    return float(np.quantile(pooled, quantile))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_snapshot(root: Path, day: int, pins: dict, snapshot_label: str):
    """(ssh, u, v) end-of-day snapshot of one day file, read under its pinned digest."""
    name = f"day{day:04d}.npz"
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != pins.get(name):
        raise ValueError(f"{root.name}/{name}: digest differs from the manifest pin")
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        if int(z["day"]) != day:
            raise ValueError(f"{name}: embedded day differs")
        if "snapshot_sampling" in z.files and str(z["snapshot_sampling"]) != snapshot_label:
            raise ValueError(f"{name}: snapshot label {z['snapshot_sampling']!r} is not the driver's")
        return tuple(np.asarray(z[k], dtype=np.float64) for k in ("ssh_m", "u_m_s", "v_m_s"))


def vorticity_series(root: Path, days, geometry, f0: float, snapshot_label: str, restrict=None) -> np.ndarray:
    """(day, y, x) float32 relative vorticity / f0 at T points, NaN on land; ``restrict`` coarse-grains first."""
    from legoesm.ocean.fidelity.swg_coarse_graining import restricted_relative_vorticity

    pins = json.loads((root / "manifest.json").read_text())["daily_sha256"]
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    out = np.full((len(days),) + wet.shape, np.nan, dtype=np.float32)
    for k, day in enumerate(days):
        ssh, u, v = read_snapshot(root, day, pins, snapshot_label)
        if restrict is not None:
            c = restrict({"T": ssh, "U": u, "V": v})
            u, v = c["U"], c["V"]
        zeta = restricted_relative_vorticity({"U": u, "V": v}, geometry) / f0
        out[k][wet] = zeta[wet]
        if (k + 1) % 50 == 0:
            print(f"  {root.name}: {k + 1}/{len(days)}", flush=True)
    return out


def figure(titles, fields, limits, *, plt, suptitle, caption, cbar_labels):
    """2 x 4 panels; panel 0 has its own colour limit, panels 1-7 share ``limits[1]``.  Returns fig, images, suptitle."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    images = []
    for i, (title, field) in enumerate(zip(titles, fields)):
        r, c = divmod(i, 4)
        ax = fig.add_axes([0.03 + 0.245 * c, 0.565 - 0.4 * r, 0.193, 0.343])  # 3.7 in square panels
        lim = limits[0] if i == 0 else limits[1]
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest"
                       if field.shape[0] < 200 else "antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, linespacing=1.25)
        images.append(im)
    cax0 = fig.add_axes([0.04, 0.11, 0.17, 0.013])
    cax1 = fig.add_axes([0.32, 0.11, 0.45, 0.013])
    for cax, im, lab in ((cax0, images[0], cbar_labels[0]), (cax1, images[1], cbar_labels[1])):
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    st = fig.suptitle(suptitle, fontsize=13, y=0.985)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.8, linespacing=1.35)
    return fig, images, st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--first-day", type=int, default=3651)
    ap.add_argument("--last-day", type=int, default=4015)
    ap.add_argument("--step", type=int, default=2, help="days between movie frames (the time mean uses every day)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quantile", type=float, default=0.995, help="colour limit = this |value| quantile of the filtered truth")
    ap.add_argument("--truth-label", default="HR truth (6.25 km native)")
    args = ap.parse_args(argv)

    import jax

    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX_ENABLE_X64=1 required")
    import matplotlib

    matplotlib.use("Agg")
    import imageio_ffmpeg
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST, SWG_MADDISON_NAMELIST
    from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry
    from legoesm.training.swg_online import SNAPSHOT_LABEL

    ev = _load(EVAL)
    out = args.out_dir
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    # --- provenance guards: the run shown is the run scored, the truth is the frozen one, one deck f0 everywhere
    if _sha(TRUTH / "manifest.json") != ev.RIGID_LID_MANIFEST_SHA256["truth"]:
        raise SystemExit("truth manifest is not the frozen rigid-lid truth")
    f0s = {json.loads((r / "manifest.json").read_text())["namelist"]["rn_f0_s"]
           for r in [TRUTH, NO_CLOSURE] + [run[2] for run in RUNS]}
    if len(f0s) != 1:
        raise SystemExit(f"runs disagree on f0: {f0s}")
    f0 = float(f0s.pop())
    gate = ev.frozen_gate("symmetric", "rigid_lid")
    scores = {}
    for key, _, root, label in RUNS:
        rec = json.loads((SCORES / f"{label}.json").read_text())
        if Path(rec["root"]) != root or rec["manifest_sha256"] != _sha(root / "manifest.json"):
            raise SystemExit(f"score file {label} does not belong to {root}")
        scores[key] = score_line(rec["window_medians"])
    ctl = {k: float(np.median(gate["controls"]["nu10"][k])) for k in SCORE_KEYS}
    ctl_line = score_line(ctl)

    days = frame_days(args.first_day, args.last_day, 1)
    movie_idx = [d - args.first_day for d in frame_days(args.first_day, args.last_day, args.step)]
    fine_geom = cgrid_geometry(SWG_MADDISON_NAMELIST)
    coarse_geom = cgrid_geometry(SWG_DG50_NAMELIST)
    restrict = ev.truth_restrictor("symmetric")

    sources = [("truth", TRUTH, fine_geom, None), ("truth50", TRUTH, coarse_geom, restrict),
               ("noclosure", NO_CLOSURE, coarse_geom, None)] + [(k, r, coarse_geom, None) for k, _, r, _ in RUNS]
    zeta = {}
    for key, root, geom, restr in sources:
        f = cache / f"zeta_{key}_{days[0]}_{days[-1]}.npy"
        if f.exists():
            zeta[key] = np.load(f)
        else:
            print("computing", key, flush=True)
            zeta[key] = vorticity_series(root, days, geom, f0, SNAPSHOT_LABEL, restr)
            np.save(f, zeta[key])
    keys = [s[0] for s in sources]

    print("per-panel zeta/f0 over all days, wet cells: min / max / 99.5% |.|")
    for key, _, geom, _ in sources:
        a = zeta[key]
        wet = np.asarray(geom["mask"]["T"], dtype=bool)
        if not np.isfinite(a[:, wet]).all():
            raise SystemExit(f"{key}: non-finite vorticity inside the wet domain")
        if not np.nanmax(np.abs(a)) > 0:
            raise SystemExit(f"{key}: blank panel")
        print(f"  {key:10s} {np.nanmin(a): .4f} {np.nanmax(a): .4f} {np.nanquantile(np.abs(a[::10]), 0.995):.4f}", flush=True)

    # crop every panel to its wet rectangle (the same 4000 km square)
    def crop(a, geom):
        jj, ii = np.where(np.asarray(geom["mask"]["T"], dtype=bool))
        return a[..., jj.min():jj.max() + 1, ii.min():ii.max() + 1]

    geom_of = {s[0]: s[2] for s in sources}
    zc = {k: crop(zeta[k], geom_of[k]) for k in keys}
    mean = {k: np.nanmean(zc[k], axis=0) for k in keys}

    lim_movie = (symmetric_limit([zc["truth"][::8]], args.quantile), symmetric_limit([zc["truth50"][::8]], args.quantile))
    lim_mean = (symmetric_limit([mean["truth"]], args.quantile), symmetric_limit([mean["truth50"]], args.quantile))
    print(f"colour limits (zeta/f0): movie native {lim_movie[0]:.3f}, movie coarse (panels 2-8) {lim_movie[1]:.3f};"
          f" mean native {lim_mean[0]:.3f}, mean coarse {lim_mean[1]:.4f}", flush=True)

    titles = [f"{args.truth_label}\nown colour scale\n",
              "HR truth filtered to 50 km\n(symmetric restrictor: what the scores compare to)\n",
              "No closure (50 km, rigid lid)\nscores of the frozen CPU control of this deck:\n" + ctl_line]
    for key, label, _, _ in RUNS:
        wrapped = textwrap.wrap(label, 52)
        titles.append("\n".join(wrapped + [""] * (2 - len(wrapped))) + "\n" + scores[key])
    span = f"days {days[0]}-{days[-1]}"
    def caption(lim):
        return (BARS + "   (scores: median over the scored two-year windows, days 366-6935)\n"
                "All runs start from rest on the same deck: the calendar day is comparable statistically, not phase for phase. "
                f"Panels 2-8 share one colour scale (+/-{lim[1]:.3g} f0); panel 1 has its own (+/-{lim[0]:.3g} f0).")

    # --- time-mean figure
    labs = ("zeta / f0, native (own scale)", "zeta / f0, panels 2-8 (shared scale)")
    fig, _, _ = figure(titles, [mean[k] for k in keys], lim_mean, plt=plt,
                       suptitle=f"1-year time-mean relative vorticity / f0, {span} (365 daily snapshots)",
                       caption=caption(lim_mean), cbar_labels=labs)
    fig.savefig(out / "swg_online_comparison_timemean.png")
    plt.close(fig)

    # --- movie + mid frame
    k0 = movie_idx[0]
    fig, images, st = figure(titles, [zc[k][k0] for k in keys], lim_movie, plt=plt,
                             suptitle="", caption=caption(lim_movie), cbar_labels=labs)

    def update(n):
        k = movie_idx[n]
        for im, key in zip(images, keys):
            im.set_data(zc[key][k])
        st.set_text(f"Rigid-lid double gyre, relative vorticity / f0 -- model day {days[k]}  (frame {n + 1}/{len(movie_idx)})")
        return images

    update(0)
    mid = len(movie_idx) // 2
    update(mid)
    fig.savefig(out / "swg_online_comparison_midframe.png")
    anim = animation.FuncAnimation(fig, update, frames=len(movie_idx), blit=False)
    writer = animation.FFMpegWriter(fps=args.fps, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium"])
    anim.save(out / "swg_online_comparison.mp4", writer=writer, dpi=100)
    print("wrote", out / "swg_online_comparison.mp4", flush=True)


if __name__ == "__main__":
    main()
