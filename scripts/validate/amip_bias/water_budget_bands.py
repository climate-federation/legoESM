#!/usr/bin/env python3
"""Storage-corrected band water budget: P - E + dW/dt = implied convergence.

A regional P-E difference alone cannot say whether a moist bias is a steady
excess convergence or the atmosphere still filling up: a column that is
accumulating water shows the same P-E signature as one exporting less.  This
reads total atmospheric water W (all six water tracers, mass-weighted over the
column) from the run's OWN checkpoints, prints its time series per band so the
steadiness of the analysis window is visible rather than assumed, and reports
the storage tendency next to the published surface fluxes.

Sign convention: W [kg/m2] is a column store; dW/dt > 0 means the band is
GAINING water.  P and E are positive-downward/upward surface fluxes as CMOR
publishes them (pr = rain to the surface, evspsbl = evaporation from it), so
the column budget is  dW/dt = C + E - P  with C the horizontal convergence,
i.e.  C = P - E + dW/dt.  All three printed in mm/day.

Usage: water_budget_bands.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import os
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402

from legoesm import constants  # noqa: E402

SEC_PER_DAY = 86400.0
_WATER = ("trc_q_v", "trc_q_c", "trc_q_i", "trc_q_r", "trc_q_s", "trc_q_g")

# Bands PARTITION the globe so a convergence excess in one strip has to show as
# a divergence excess in another: a band list with gaps hides the source, which
# is exactly what a tropics-only table did.
BANDS = {"SH extratrop 90-30S": (-90.0, -30.0), "SH trades 30-10S": (-30.0, -10.0),
         "ITCZ 10S-10N": (-10.0, 10.0), "NH trades 10-30N": (10.0, 30.0),
         "NH extratrop 30-90N": (30.0, 90.0), "global": (-90.0, 90.0)}


def column_water(ck):
    """Total atmospheric water [kg/m2] per column from one checkpoint."""
    z = np.load(ck, allow_pickle=True)
    missing = [k for k in _WATER if k not in z]
    if missing:
        raise SystemExit(f"FATAL: {ck} lacks water tracers {missing}")
    vg = np.asarray(z["meta_vgrid"])
    ps = np.asarray(z["p_s"], dtype=np.float64)
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    dp = p_half[:, 1:] - p_half[:, :-1]
    q = sum(np.asarray(z[k], dtype=np.float64) for k in _WATER)
    w = (q * dp).sum(axis=1) / constants.g
    if not np.all(np.isfinite(w)):
        raise SystemExit(f"FATAL: non-finite column water in {ck}")
    return float(np.asarray(z["day"])), w


def _flux_fields(run):
    """Model and reference (GPCP pr / ERA5 evspsbl) surface water fluxes on the
    model grid, time-meaned over the SAME calendar months the run published."""
    out = {}
    for var in ("pr", "evspsbl"):
        d = rb._load_model(run, var)
        if d is None:
            raise SystemExit(f"FATAL: {run} publishes no {var}")
        mlat, mlon = np.asarray(d.lat), np.asarray(d.lon)
        months = rb._month_labels(d)
        ref = rb._ref_clim(var, months, mlat, mlon)
        if ref is None:
            raise SystemExit(f"FATAL: no reference climatology for {var}")
        out[var] = (np.asarray(d[var]).mean(axis=0), np.asarray(ref), mlat, mlon)
    return out


def _band(field, mlat, mlon, lo, hi):
    return rb.region_mean(field, mlat, mlon, (lo, hi, 0, 360)) * SEC_PER_DAY


def _report(run):
    rundir = f"{rb.ROOT}/{run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, _lon, area = mesh_coords(exp)
    cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
    if len(cks) < 2:
        raise SystemExit(f"FATAL: {run} has {len(cks)} checkpoints, need >= 2")
    days, cols = zip(*(column_water(c) for c in cks))

    print(f"\n=== {run}: column water store W [kg/m2], area-weighted ===")
    print(f"{'band':<22}" + "".join(f"{f'd{d:.0f}':>9}" for d in days))
    wser = {}
    for name, (lo, hi) in BANDS.items():
        w = area * ((lat >= lo) & (lat <= hi))
        s = [float((c * w).sum() / w.sum()) for c in cols]
        wser[name] = s
        print(f"{name:<22}" + "".join(f"{v:9.3f}" for v in s))

    dt = days[-1] - days[0]
    ff = _flux_fields(run)
    print(f"\n=== {run}: band water budget, days {days[0]:.0f}-{days[-1]:.0f} "
          f"[mm/day].  C = P - E + dW/dt = implied horizontal convergence ===")
    print(f"{'band':<22}{'P':>8}{'Pobs':>8}{'E':>8}{'Eobs':>8}"
          f"{'dW/dt':>8}{'C':>8}{'Cobs':>8}{'dC':>8}")
    for name, (lo, hi) in BANDS.items():
        pm, pr_, mlat, mlon = ff["pr"]
        em, er, elat, elon = ff["evspsbl"]
        p, po = _band(pm, mlat, mlon, lo, hi), _band(pr_, mlat, mlon, lo, hi)
        e, eo = _band(em, elat, elon, lo, hi), _band(er, elat, elon, lo, hi)
        dw = (wser[name][-1] - wser[name][0]) / dt
        c, co = p - e + dw, po - eo
        print(f"{name:<22}{p:8.3f}{po:8.3f}{e:8.3f}{eo:8.3f}"
              f"{dw:8.3f}{c:8.3f}{co:8.3f}{c - co:8.3f}")
    print("P against GPCP, E against ERA5, both over the calendar months the "
          "run published; dW/dt over the checkpoint span, which is NOT the "
          "same window.  Cobs carries no storage term: a climatology is steady "
          "by construction, the model need not be -- read dW/dt first.")


def _plot(runs, out):
    """Zonal-mean P and E against the references, plus the column-water store's
    time series -- the two panels that decide whether a band bias is a steady
    redistribution or an unconverged drift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, run in enumerate(runs):
        ff = _flux_fields(run)
        for k, var in enumerate(("pr", "evspsbl")):
            mv, rv, mlat, mlon = ff[var]
            ax[k].plot(mlat, mv.mean(axis=1) * SEC_PER_DAY, label=run)
            if i == 0:
                ref = "GPCP" if var == "pr" else "ERA5"
                ax[k].plot(mlat, rv.mean(axis=1) * SEC_PER_DAY, "k--", label=ref)
        rundir = f"{rb.ROOT}/{run}"
        exp = json.load(open(f"{rundir}/experiment_config.json"))
        lat, _lon, area = mesh_coords(exp)
        cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
        days, cols = zip(*(column_water(c) for c in cks))
        for name, (lo, hi) in BANDS.items():
            if name == "global":
                continue
            w = area * ((lat >= lo) & (lat <= hi))
            ax[2].plot(days, [float((c * w).sum() / w.sum()) for c in cols],
                       ls=("-" if i == 0 else "--"),
                       label=f"{name} [{run}]" if name == "ITCZ 10S-10N" else None)
    for k, t in enumerate(("zonal-mean precipitation", "zonal-mean evaporation")):
        ax[k].set_title(t)
        ax[k].set_xlabel("latitude")
        ax[k].set_ylabel("mm/day")
        ax[k].legend(fontsize=7)
        ax[k].grid(alpha=0.3)
    ax[2].set_title("column water store by band (solid/dashed = run)")
    ax[2].set_xlabel("day")
    ax[2].set_ylabel("W [kg/m2]")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    for r in args:
        _report(r)
    if "--plot" in sys.argv:
        _plot(args, os.environ.get("WB_PLOT_OUT", "water_budget_bands.png"))
