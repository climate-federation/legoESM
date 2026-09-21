#!/usr/bin/env python3
"""Regional reduction of a per-column process ledger (``budget_ledger_columns.npz``).

The global ledger table hides WHERE a term acts. This reduces the per-column
rows to area-weighted means over the tropics split by surface (fractional
``sftlf`` from the run's own published fx file, nearest-neighbour onto the
MPAS mesh), in kg/m2/day. It also prints the checkpoint's instantaneous
convective rain source (``physstate_conv_precip``, the post-sub-cloud-
evaporation survivor that enters the ``q_r`` tracer) so the sedimentation
rows can be read against the rain the convection scheme actually hands over.

Usage: ledger_regional.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402

SEC_PER_DAY = 86400.0
BOXES = {"ITCZ 10S-10N": (-10.0, 10.0), "trades 10-30N": (10.0, 30.0),
         "trades 10-30S": (-30.0, -10.0), "NH midlat 30-60N": (30.0, 60.0),
         "SO stormtrack 60-30S": (-60.0, -30.0), "global": (-90.0, 90.0)}


def _sftlf_on_mesh(run, lat, lon):
    # A short branch run has not published a monthly stream yet, so its land
    # fraction has to be named EXPLICITLY from the parent it branched off.
    # Falling back to some other run's mask silently would be exactly the
    # hidden choice that makes two tables incomparable.
    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf -- name the run "
                         f"whose land mask to use with --sftlf-from <run>")
    d = xr.open_dataset(fs[0])
    frac = np.asarray(d["sftlf"]) / 100.0
    glat, glon = np.asarray(d.lat), np.asarray(d.lon) % 360.0
    i = np.abs(glat[None, :] - lat[:, None]).argmin(axis=1)
    j = np.abs(((glon[None, :] - lon[:, None] + 180.0) % 360.0) - 180.0).argmin(axis=1)
    return frac[i, j]


def _report(run, sftlf_run=None):
    rundir = f"{rb.ROOT}/{run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    d = np.load(f"{rundir}/budget_ledger_columns.npz", allow_pickle=True)
    rates = np.asarray(d["ledger_rates"])[:, :, 0] * SEC_PER_DAY   # water, kg/m2/day
    procs = [str(p) for p in d["processes"]]
    rows = {}
    if not np.all(np.isfinite(rates)):
        raise SystemExit(f"FATAL: non-finite ledger rates in {run}")
    fl = _sftlf_on_mesh(sftlf_run or run, lat, lon)
    cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
    ck = np.load(cks[-1], allow_pickle=True)
    pconv = np.asarray(ck["physstate_conv_precip"]) * SEC_PER_DAY if "physstate_conv_precip" in ck else None
    print(f"\n=== {run}: per-column ledger, day {float(d['day']):.1f}, {int(d['n_steps'])} steps "
          f"[kg/m2/day, area-weighted] ===")
    hdr = f"{'region':<26}" + "".join(f"{p[:12]:>13}" for p in procs) + f"{'conv_src':>13}"
    print(hdr)
    for name, (lo, hi) in BOXES.items():
        box = (lat >= lo) & (lat <= hi)
        for surf, wsurf in (("all", np.ones_like(fl)), ("ocean", 1.0 - fl), ("land", fl)):
            w = area * box * wsurf
            if w.sum() <= 0.0:
                continue
            row = (rates * w[:, None]).sum(0) / w.sum()
            pc = (pconv * w).sum() / w.sum() if pconv is not None else np.nan
            rows[name + " " + surf] = row
            print(f"{name + ' ' + surf:<26}" + "".join(f"{v:13.3f}" for v in row) + f"{pc:13.3f}")
    print("conv_src = checkpoint-instant survivor convective rain source entering q_r "
          "(after IFS downdraft + sub-cloud evaporation).")
    return procs, rows


def layers_figure(runs, sftlf_run, out, regions=("ITCZ 10S-10N ocean", "trades 10-30N ocean",
                                                "trades 10-30S ocean")):
    """Grouped bars of the ledger rows for several runs that each carry the
    ledger on a different sigma band (the run name is the layer label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tabs = {r: _report(r, sftlf_run) for r in runs}
    procs = tabs[runs[0]][0]
    keep = [i for i, p in enumerate(procs) if p in ("turbulence", "convection", "microphysics", "dynamics")]
    fig, ax = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4), sharey=True)
    x = np.arange(len(keep)); wdt = 0.8 / len(runs)
    for a, reg in zip(np.atleast_1d(ax), regions):
        for j, r in enumerate(runs):
            a.bar(x + (j - (len(runs) - 1) / 2) * wdt, tabs[r][1][reg][keep], wdt,
                  label=r.split("_")[-1])
        a.set_xticks(x); a.set_xticklabels([procs[i] for i in keep]); a.axhline(0, color="k", lw=0.8)
        a.set_title(reg); a.grid(axis="y")
    np.atleast_1d(ax)[0].set_ylabel("kg/m2/day"); np.atleast_1d(ax)[0].legend(title="ledger band")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"figure {out}")


def profile(run, sftlf_run=None, dlat=5.0, out=None):
    """Latitude profile of the ledger rows, plus the implied meridional flux.

    Bands hide WHERE a transport deficit sits: a Hadley cell that is too narrow
    and a storm track whose eddies are too weak both show up as "the
    extratropics import too little".  The zero crossings of the transport
    profile separate them, and its cumulative integral from the pole is the
    net meridional water flux, which is the quantity a reanalysis publishes
    directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rundir = f"{rb.ROOT}/{run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, _lon, area = mesh_coords(exp)
    d = np.load(f"{rundir}/budget_ledger_columns.npz", allow_pickle=True)
    rates = np.asarray(d["ledger_rates"])[:, :, 0] * SEC_PER_DAY
    procs = [str(p) for p in d["processes"]]

    edges = np.arange(-90.0, 90.0 + dlat, dlat)
    mid = 0.5 * (edges[:-1] + edges[1:])
    keep = ("turbulence", "convection", "microphysics", "dynamics")
    prof = {k: np.full(mid.size, np.nan) for k in keep}
    wsum = np.zeros(mid.size)
    for i in range(mid.size):
        m = (lat >= edges[i]) & (lat < edges[i + 1])
        w = area * m
        wsum[i] = w.sum()
        if wsum[i] <= 0:
            continue
        for k in keep:
            prof[k][i] = float((rates[:, procs.index(k)] * w).sum() / wsum[i])
    if not np.all(np.isfinite(prof["dynamics"])):
        raise SystemExit("FATAL: empty latitude bin -- widen dlat")

    # Net northward flux across each edge [kg/m/s-equivalent, reported as the
    # cumulative column source from the south pole]: a band that gains water
    # must be fed across its southern edge by everything south of it.
    flux = np.cumsum(prof["dynamics"] * wsum)

    # Reference transport: the convergence implied by observed precipitation
    # minus observed evaporation, on the SAME latitude bins.  Its zero
    # crossings are where the real circulation turns over, so a model whose
    # crossings sit equatorward of these has a too-narrow overturning rather
    # than weak eddies.
    import water_budget_bands as wbb
    ff = wbb._flux_fields(run)
    pm, pr_, mlat, _mlon = ff["pr"]
    _em, er, _elat, _elon = ff["evspsbl"]
    obs = np.full(mid.size, np.nan)
    for i in range(mid.size):
        band = (mlat >= edges[i]) & (mlat < edges[i + 1])
        if not band.any():
            continue
        w = np.cos(np.deg2rad(mlat[band]))[:, None] * np.ones((1, pr_.shape[1]))
        obs[i] = float(((pr_ - er)[band] * w).sum() / w.sum()) * SEC_PER_DAY

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for k in keep:
        ax[0].plot(mid, prof[k], label=k)
    ax[0].plot(mid, obs, "k--", lw=2, label="observed transport")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("latitude")
    ax[0].set_ylabel("kg/m2/day")
    ax[0].set_title(f"{run}: ledger rows by latitude")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(mid, flux / 1e12)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("latitude")
    ax[1].set_ylabel("cumulative transport [1e12 kg/day]")
    ax[1].set_title("implied northward water flux")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = out or f"ledger_profile_{run}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"{'lat':>6}" + "".join(f"{k[:9]:>11}" for k in keep))
    for i in range(mid.size):
        if abs(mid[i]) <= 70:
            print(f"{mid[i]:6.1f}" + "".join(f"{prof[k][i]:11.3f}" for k in keep))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    argv = sys.argv[1:]
    src = None
    if "--sftlf-from" in argv:
        i = argv.index("--sftlf-from")
        src = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    prof = "--profile" in argv
    argv = [a for a in argv if a != "--profile"]
    if "--layers" in argv:
        i = argv.index("--layers")
        out = argv[i + 1]
        layers_figure(argv[:i] + argv[i + 2:], src, out)
        raise SystemExit(0)
    for r in argv:
        if prof:
            profile(r, sftlf_run=src,
                    out=f"{rb.ROOT}/ledger_profile_{r}.png")
        else:
            _report(r, sftlf_run=src)
