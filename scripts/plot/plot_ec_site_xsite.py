#!/usr/bin/env python
"""Cross-site EC performance summary, grouped by plant-functional type (PFT).

Scans a directory of ``run_ec_site.py`` outputs (``<SITE>_ec_<mode>.nc`` with
``{gpp,le,h}_{mod,obs}`` + ``valid``/``score_valid`` and a ``pft`` attr), computes
GPP / LE / H skill (Pearson r, RMSE, bias) per site, and plots:

* per-site r for each flux, sorted and coloured by PFT (where the model is
  strong vs weak across plant types), and
* PFT-aggregated skill (mean r per PFT per flux),

so cross-site / cross-PFT performance gaps are visible at a glance.  Also writes
a ``xsite_skill.csv`` table.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_xsite.py \
        --in-dir diagnostics/ec_site_xsite --out-dir diagnostics/ec_site_xsite
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

_FLUXES = [("gpp", "GPP"), ("le", "LE"), ("h", "H"), ("ef", "EF")]
_UNIT = {"GPP": r"$\mu$mol m$^{-2}$s$^{-1}$", "LE": "W m$^{-2}$",
         "H": "W m$^{-2}$", "EF": "-"}
_EF_FLOOR = 50.0   # W/m2: form EF only where LE+H exceeds this (daytime)
_METRICS = {"r": "Pearson $r$", "rmse": "RMSE"}


def _set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.axisbelow": True, "font.family": "DejaVu Sans",
        "mathtext.default": "regular",
    })


def _pair(ds, key):
    """Return (model, obs) arrays for a flux key; EF derived from LE & H."""
    if key == "ef":
        lem, leo = ds["le_mod"].values, ds["le_obs"].values
        hm, ho = ds["h_mod"].values, ds["h_obs"].values
        am, ao = lem + hm, leo + ho
        m = np.where(am > _EF_FLOOR, lem / am, np.nan)
        o = np.where(ao > _EF_FLOOR, leo / ao, np.nan)
        return np.clip(m, -0.5, 1.5), np.clip(o, -0.5, 1.5)
    return ds[f"{key}_mod"].values, ds[f"{key}_obs"].values


def _skill(ds, key):
    base = (ds["score_valid"].values if "score_valid" in ds
            else ds["valid"].values).astype(bool)
    m, o = _pair(ds, key)
    mask = base & np.isfinite(m) & np.isfinite(o)
    if mask.sum() < 10 or np.std(m[mask]) == 0 or np.std(o[mask]) == 0:
        return dict(n=int(mask.sum()), r=np.nan, rmse=np.nan, bias=np.nan)
    mm, oo = m[mask], o[mask]
    return dict(n=int(mask.sum()), r=float(np.corrcoef(mm, oo)[0, 1]),
                rmse=float(np.sqrt(np.mean((mm - oo) ** 2))),
                bias=float(np.mean(mm - oo)))


def collect(in_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(in_dir, "*_ec_*.nc"))):
        ds = xr.open_dataset(path)
        site = str(ds.attrs.get("site", os.path.basename(path))).replace(
            "_driver_v2.nc", "").replace("_driver_v2_gapfree.nc", "")
        pft = str(ds.attrs.get("pft", "?"))
        rec = dict(site=site, pft=pft)
        for key, label in _FLUXES:
            need = ["le_mod", "h_mod"] if key == "ef" else [f"{key}_mod"]
            if any(v not in ds for v in need):
                continue
            s = _skill(ds, key)
            rec[f"{label}_r"] = s["r"]
            rec[f"{label}_rmse"] = s["rmse"]
            rec[f"{label}_bias"] = s["bias"]
            rec[f"{label}_n"] = s["n"]
        rows.append(rec)
    return rows


def _ax_metric(ax, metric, label):
    ax.set_ylabel(f"{label}  {_METRICS[metric]}"
                  + (f"\n({_UNIT[label]})" if metric == "rmse" else ""))
    if metric == "r":
        ax.axhline(0.8, color="0.3", ls="--", lw=0.8)
        ax.axhline(0.0, color="0.6", lw=0.5)
        ax.set_ylim(-0.45, 1.0)


def plot(in_dir, out_dir, mode_label="diagnostic"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _set_pub_style()

    os.makedirs(out_dir, exist_ok=True)
    rows = collect(in_dir)
    if not rows:
        print(f"no outputs found in {in_dir}"); return

    csv = os.path.join(out_dir, "xsite_skill.csv")
    cols = ["site", "pft"] + [f"{lab}_{m}" for _, lab in _FLUXES
                              for m in ("r", "rmse", "bias", "n")]
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  table -> {csv}")

    pfts = sorted({r["pft"] for r in rows})
    cmap = {p: cm.tab10(i % 10) for i, p in enumerate(pfts)}
    rows.sort(key=lambda r: (r["pft"], -(r.get("GPP_r", -9)
              if np.isfinite(r.get("GPP_r", np.nan)) else -9)))
    sites = [r["site"] for r in rows]
    colors = [cmap[r["pft"]] for r in rows]
    x = np.arange(len(sites))

    for metric in ("r", "rmse"):
        # ---- per-site, one panel per flux ----
        fig, axes = plt.subplots(len(_FLUXES), 1,
                                 figsize=(max(9, 0.4 * len(sites)), 11), sharex=True)
        for ax, (key, label) in zip(axes, _FLUXES):
            vals = [r.get(f"{label}_{metric}", np.nan) for r in rows]
            ax.bar(x, vals, color=colors, edgecolor="0.3", linewidth=0.3)
            _ax_metric(ax, metric, label)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(sites, rotation=90)
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[p]) for p in pfts]
        axes[0].legend(handles, pfts, ncol=min(len(pfts), 7),
                       loc="upper center", title="PFT", fontsize=7.5)
        axes[0].set_title(f"Cross-site EC {_METRICS[metric]} "
                          f"({len(sites)} sites, {mode_label})")
        p = os.path.join(out_dir, f"xsite_{metric}_by_site.png")
        fig.savefig(p); plt.close(fig); print(f"  plot -> {p}")

        # ---- PFT-aggregated, one panel per flux ----
        fig, axes = plt.subplots(1, len(_FLUXES), figsize=(15, 3.4))
        for ax, (key, label) in zip(axes, _FLUXES):
            means = [np.nanmean([r.get(f"{label}_{metric}", np.nan)
                                 for r in rows if r["pft"] == p]) for p in pfts]
            ax.bar(range(len(pfts)), means,
                   color=[cmap[p] for p in pfts], edgecolor="0.3", linewidth=0.3)
            ax.set_xticks(range(len(pfts)))
            ax.set_xticklabels(pfts, rotation=45, ha="right")
            ax.set_title(label)
            ax.set_ylabel(_METRICS[metric] + (f" ({_UNIT[label]})"
                          if metric == "rmse" else ""))
            if metric == "r":
                ax.axhline(0.8, color="0.3", ls="--", lw=0.8); ax.set_ylim(-0.45, 1.0)
        fig.suptitle(f"Mean EC {_METRICS[metric]} by PFT ({mode_label})", y=1.04)
        p = os.path.join(out_dir, f"xsite_{metric}_by_pft.png")
        fig.savefig(p); plt.close(fig); print(f"  plot -> {p}")

    print(f"\n=== cross-site ({len(sites)} sites, {mode_label}) ===")
    print(f"  {'site':11s} {'pft':5s}  GPP(r/rmse) LE(r/rmse) H(r/rmse)")
    for r in rows:
        print(f"  {r['site']:11s} {r['pft']:5s}  "
              + "  ".join(f"{r.get(l+'_r', np.nan):.2f}/{r.get(l+'_rmse', np.nan):.2g}"
                         for _, l in _FLUXES[:3]))


def compare(diag_dir, prog_dir, out_dir):
    """Overlay diagnostic vs prognostic skill for sites present in BOTH dirs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_pub_style()

    os.makedirs(out_dir, exist_ok=True)
    diag = {r["site"]: r for r in collect(diag_dir)}
    prog = {r["site"]: r for r in collect(prog_dir)}
    common = sorted(set(diag) & set(prog), key=lambda s: (diag[s]["pft"], s))
    if not common:
        print("no sites common to both modes"); return
    pft = {s: diag[s]["pft"] for s in common}
    cD, cP = "#0072B2", "#D55E00"

    for metric in ("r", "rmse"):
        fig, axes = plt.subplots(len(_FLUXES), 1,
                                 figsize=(max(9, 0.55 * len(common)), 12), sharex=True)
        x = np.arange(len(common)); w = 0.4
        for ax, (key, label) in zip(axes, _FLUXES):
            dv = [diag[s].get(f"{label}_{metric}", np.nan) for s in common]
            pv = [prog[s].get(f"{label}_{metric}", np.nan) for s in common]
            ax.bar(x - w/2, dv, w, label="diagnostic", color=cD,
                   edgecolor="0.3", linewidth=0.3)
            ax.bar(x + w/2, pv, w, label="prognostic (nudged)", color=cP,
                   edgecolor="0.3", linewidth=0.3)
            _ax_metric(ax, metric, label)
            if key == "gpp":
                ax.legend(loc="lower left", ncol=2)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{s}  ({pft[s]})" for s in common], rotation=90)
        axes[0].set_title(f"Diagnostic vs prognostic (nudged) — {_METRICS[metric]} "
                          f"by site/PFT")
        p = os.path.join(out_dir, f"xsite_diag_vs_prog_{metric}.png")
        fig.savefig(p); plt.close(fig); print(f"  plot -> {p}")

    print(f"\n=== diag vs prog (nudged), {len(common)} common sites — r [RMSE] ===")
    for s in common:
        cells = "  ".join(
            f"{l} {diag[s].get(l+'_r', np.nan):.2f}/{prog[s].get(l+'_r', np.nan):.2f}"
            for _, l in _FLUXES)
        print(f"  {s:11s} {pft[s]:5s} | {cells}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="diagnostics/ec_site_xsite")
    ap.add_argument("--out-dir", default="diagnostics/ec_site_xsite")
    ap.add_argument("--prog-dir", default=None,
                    help="if given, also overlay diagnostic vs prognostic skill")
    args = ap.parse_args()
    plot(args.in_dir, args.out_dir)
    if args.prog_dir:
        compare(args.in_dir, args.prog_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
