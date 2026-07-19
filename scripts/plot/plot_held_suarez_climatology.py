#!/usr/bin/env python
"""Held-Suarez climatology + jet metrics from a dry-HS run.

Consumes the ``.npz`` written by ``scripts/run/run_dry_held_suarez_cube.py``
(and siblings) and produces the comparison against Held & Suarez (1994):

  (a) TIME-MEAN zonal-mean zonal wind, latitude x sigma  -> HS94 Fig. 1
  (b) TIME-MEAN zonal-mean temperature,  latitude x sigma -> HS94 Fig. 2
  (c) eddy kinetic energy vs time -- the spin-up curve, which shows whether
      baroclinic eddies developed at all and when the flow reached
      statistical equilibrium (so the reader can judge the averaging window)

TIME MEAN, NOT SNAPSHOT.  HS94's published figures are means over ~1000 days
after spin-up; a single instantaneous zonal mean of a turbulent flow is a
different quantity and will not match.  This script REFUSES to plot panels
(a)/(b) unless the run carries ``uz_timemean`` (i.e. was run with
``--mean-from-day``), rather than silently falling back to the final snapshot.

HS94 reference for the jet: ~28 m/s near 45 deg latitude at sigma ~ 0.25.
The repo's own matrix gate uses a floor of 20 m/s
(``run_atmosphere_test_matrix._HELD_SUAREZ_MIN_JET_MS``).

Run:  python scripts/plot/plot_held_suarez_climatology.py \
          --npz <run.npz> [--out results/paper_figs] [--name fig_hs]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Held & Suarez (1994) benchmark jet: peak eastward zonal-mean zonal wind.
HS94_JET_MS = 28.0
HS94_JET_LAT_DEG = 45.0
HS94_JET_SIGMA = 0.25
# The repo's existing pass floor (run_atmosphere_test_matrix).
JET_FLOOR_MS = 20.0

INK, MUTED = "#1a1a1a", "#666666"


def load(npz_path):
    d = np.load(npz_path)
    if "uz_timemean" not in d.files:
        raise SystemExit(
            f"{npz_path} has no 'uz_timemean' — the run was not given "
            "--mean-from-day, so only an instantaneous snapshot exists. "
            "HS94 Figs 1-2 are TIME MEANS; re-run with --mean-from-day.")
    return d


def jet_metrics(d):
    """Peak time-mean eastward jet: value, latitude, sigma, and HS94 deltas."""
    uz = np.asarray(d["uz_timemean"])
    lat = np.asarray(d["lat_bins"])
    sig = np.asarray(d["sigma_full"])
    jk = np.unravel_index(np.nanargmax(uz), uz.shape)
    u_max = float(uz[jk])
    return {
        "jet_u_ms": u_max,
        "jet_lat_deg": float(lat[jk[0]]),
        "jet_sigma": float(sig[jk[1]]),
        "hs94_u_ms": HS94_JET_MS,
        "hs94_lat_deg": HS94_JET_LAT_DEG,
        "hs94_sigma": HS94_JET_SIGMA,
        "u_ratio_vs_hs94": u_max / HS94_JET_MS,
        "passes_repo_floor": bool(u_max >= JET_FLOOR_MS),
        "n_mean_samples": int(d["n_mean_samples"]),
        "mean_from_day": float(d["mean_from_day"]),
    }


def perf_metrics(d):
    """Wall-clock cost of the run — the laptop-portability claim."""
    out = {}
    for key in ("sypd", "wall_loop_s", "jit_s", "dt_seconds", "days",
                "n_cells", "nlev"):
        if key in d.files:
            out[key] = float(d[key])
    if "wall_loop_s" in out and out["wall_loop_s"] > 0:
        out["wall_hours"] = out["wall_loop_s"] / 3600.0
        if "n_cells" in out and "nlev" in out and "dt_seconds" in out:
            steps = out["days"] * 86400.0 / out["dt_seconds"]
            cells = out["n_cells"] * out["nlev"]
            out["mcells_per_s"] = cells * steps / out["wall_loop_s"] / 1e6
    return out


def _style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8.5, color="#999999")


def make_figure(d, out, name="fig_hs_climatology"):
    import matplotlib.pyplot as plt

    lat = np.asarray(d["lat_bins"])
    sig = np.asarray(d["sigma_full"])
    uz = np.asarray(d["uz_timemean"])
    tz = np.asarray(d["Tz_timemean"]) if "Tz_timemean" in d.files else None
    eke = np.asarray(d["eke_t"])

    n_panels = 3 if tz is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.3 * n_panels, 3.9))

    # (a) zonal-mean zonal wind — HS94 Fig 1. Diverging map centred on zero
    # (easterlies vs westerlies is a POLARITY, so a diverging ramp with a
    # neutral midpoint is the correct encoding).
    ax = axes[0]
    vmax = float(np.nanmax(np.abs(uz))) or 1.0
    cf = ax.contourf(lat, sig, uz.T, levels=np.linspace(-vmax, vmax, 21),
                     cmap="RdBu_r")
    ax.contour(lat, sig, uz.T, levels=[0], colors="#666666", linewidths=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("latitude (deg)", fontsize=9.5, color=INK)
    ax.set_ylabel("sigma", fontsize=9.5, color=INK)
    ax.set_title("(a) time-mean zonal wind [m/s]", fontsize=10.5, color=INK,
                 loc="left")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    m = jet_metrics(d)
    ax.plot(m["jet_lat_deg"], m["jet_sigma"], "k+", markersize=9,
            markeredgewidth=1.4)
    ax.annotate(f"peak {m['jet_u_ms']:.1f} m/s\n"
                f"HS94 ≈ {HS94_JET_MS:.0f} m/s @ {HS94_JET_LAT_DEG:.0f}°",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=7.5,
                color=MUTED, va="bottom")
    _style(ax)

    # (b) zonal-mean temperature — HS94 Fig 2. Sequential (magnitude).
    if tz is not None:
        ax = axes[1]
        cf = ax.contourf(lat, sig, tz.T, levels=18, cmap="viridis")
        ax.invert_yaxis()
        ax.set_xlabel("latitude (deg)", fontsize=9.5, color=INK)
        ax.set_title("(b) time-mean temperature [K]", fontsize=10.5,
                     color=INK, loc="left")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        _style(ax)

    # (c) eddy KE spin-up — shows equilibrium was reached before averaging.
    ax = axes[-1]
    if eke.size:
        ax.plot(eke[:, 0], eke[:, 1], "-", color="#2c6fbb", linewidth=1.6)
        ax.set_yscale("log")
        if "mean_from_day" in d.files:
            ax.axvline(float(d["mean_from_day"]), color="#b5562a",
                       linestyle="--", linewidth=1.2)
            ax.annotate("averaging starts", xy=(float(d["mean_from_day"]), 1),
                        xycoords=("data", "axes fraction"), xytext=(4, -12),
                        textcoords="offset points", fontsize=7.5,
                        color="#b5562a", va="top")
    ax.set_xlabel("day", fontsize=9.5, color=INK)
    ax.set_ylabel("eddy KE [m²/s²]", fontsize=9.5, color=INK)
    ax.set_title("(c) baroclinic eddy spin-up", fontsize=10.5, color=INK,
                 loc="left")
    ax.grid(True, which="both", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)

    p = perf_metrics(d)
    sub = (f"C{int(round((p.get('n_cells', 0) / 6) ** 0.5))} · "
           f"L{int(p.get('nlev', 0))} · dt={p.get('dt_seconds', 0):.0f}s · "
           f"{int(p.get('days', 0))} days")
    if "sypd" in p:
        sub += (f"  —  {p['sypd']:.0f} SYPD, "
                f"{p.get('wall_hours', 0):.2f} h wall (laptop CPU)")
    fig.suptitle(f"Dry Held-Suarez climatology — {sub}", fontsize=11,
                 color=INK, y=1.03)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default="results/paper_figs")
    p.add_argument("--name", default="fig_hs_climatology")
    p.add_argument("--json", default=None, help="write metrics JSON here")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    d = load(a.npz)
    metrics = {"jet": jet_metrics(d), "performance": perf_metrics(d)}
    written = make_figure(d, a.out, a.name)

    j, perf = metrics["jet"], metrics["performance"]
    print(f"HS94 jet comparison  (time mean of {j['n_mean_samples']} daily "
          f"samples from day {j['mean_from_day']:.0f})")
    print(f"  peak zonal-mean u : {j['jet_u_ms']:.1f} m/s  "
          f"(HS94 ≈ {HS94_JET_MS:.0f})  -> {j['u_ratio_vs_hs94']:.2f}× HS94")
    print(f"  latitude          : {j['jet_lat_deg']:.0f}°  "
          f"(HS94 ≈ {HS94_JET_LAT_DEG:.0f}°)")
    print(f"  sigma             : {j['jet_sigma']:.3f}  "
          f"(HS94 ≈ {HS94_JET_SIGMA:.2f})")
    print(f"  repo floor ({JET_FLOOR_MS:.0f} m/s): "
          f"{'PASS' if j['passes_repo_floor'] else 'FAIL'}")
    if perf:
        print(f"performance: {perf.get('sypd', 0):.1f} SYPD, "
              f"{perf.get('wall_hours', 0):.2f} h wall, "
              f"{perf.get('mcells_per_s', 0):.0f} Mcells/s")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"wrote {a.json}")
    for path in written:
        print(f"wrote {path}")
    return metrics


if __name__ == "__main__":
    main()
