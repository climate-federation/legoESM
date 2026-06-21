#!/usr/bin/env python
"""One-off map plotter for the combined full-realistic CMIP coupled run.

Reads the CMOR monthly-mean NetCDF written by ``run_coupled.py`` and renders a
6-panel global-map realism check (tas, pr, tos, siconc, OLR, planetary albedo)
plus a zonal-mean near-surface-air-temperature line against a crude observed
reference.  Raw model fields only (no saturation/EOS re-derivation); the lone
physical constant is ``constants.T_freeze`` for the K->degC axis.

Usage::

    python scripts/tmp/_plot_cmip_fullreal.py \
        --run results/cmip6_coupled/full_realistic_90d \
        --out results/cmip6_coupled/_probes/full_realistic_90d_maps.png
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from legoesm import constants  # noqa: E402


def _load(base: str, table_var: str) -> xr.DataArray:
    var = table_var.split("/")[-1]
    fs = glob.glob(f"{base}/cmor/{table_var}_*.nc")
    if not fs:
        fs = glob.glob(f"{base}/cmor/**/{var}_*.nc", recursive=True)
    if not fs:
        raise FileNotFoundError(table_var)
    ds = xr.open_dataset(fs[0])
    da = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
    return da.mean("time") if "time" in da.dims else da


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tas = _load(args.run, "Amon/tas")
    pr = _load(args.run, "Amon/pr") * 86400.0  # kg/m2/s -> mm/day
    tos = _load(args.run, "Omon/tos")
    sic = _load(args.run, "SImon/siconc")
    olr = _load(args.run, "Amon/rlut")
    rsut = _load(args.run, "Amon/rsut")
    rsdt = _load(args.run, "Amon/rsdt")
    albedo = 100.0 * rsut / np.maximum(rsdt, 1.0)

    lat = tas["lat"].values
    lon = tas["lon"].values
    Tf = float(constants.T_freeze)

    panels = [
        ("tas (degC)", tas - Tf, "RdBu_r", -40, 40),
        ("pr (mm/day)", pr, "YlGnBu", 0, 10),
        ("tos (degC)", tos - Tf, "RdBu_r", -2, 32),
        ("siconc (%)", sic, "Blues_r", 0, 100),
        ("OLR rlut (W/m2)", olr, "magma", 150, 300),
        ("planetary albedo (%)", albedo, "viridis", 0, 70),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, (title, fld, cmap, vmin, vmax) in zip(axes.flat[:6], panels):
        im = ax.pcolormesh(lon, lat, np.asarray(fld), cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # Zonal-mean tas vs crude observed reference.
    axz = axes.flat[6]
    zmean = (tas - Tf).mean("lon").values
    # Crude observed zonal-mean near-surface T (degC): ~27 eq -> ~-25 pole.
    obs = 27.0 - 52.0 * np.sin(np.deg2rad(np.abs(lat))) ** 1.4
    axz.plot(lat, zmean, "b-o", ms=3, label="model")
    axz.plot(lat, obs, "k--", label="obs (crude)")
    axz.set_title(f"zonal-mean tas  (global {float(tas.mean())-Tf:.1f}degC, "
                  f"Earth ~14)")
    axz.set_xlabel("lat"); axz.set_ylabel("degC"); axz.legend(fontsize=8)
    axz.grid(alpha=0.3)

    # Summary text panel.
    axt = axes.flat[7]; axt.axis("off")
    R_TOA = float((rsdt - rsut - olr).mean())
    txt = (
        "COMBINED full-realistic 90d\n"
        "(thin-cirrus conv cloud + multilayer\n"
        " Richards land + snow-albedo)\n\n"
        f"planetary albedo : {float(albedo.mean()):.1f} %  (Earth ~30)\n"
        f"OLR rlut         : {float(olr.mean()):.1f} W/m2 (~240)\n"
        f"rsut             : {float(rsut.mean()):.1f} W/m2 (~100)\n"
        f"R_TOA (uw map)   : {R_TOA:+.1f} W/m2 (~0)\n"
        f"tas global       : {float(tas.mean())-Tf:.1f} degC (~14)\n"
        f"tos global       : {float(tos.mean())-Tf:.1f} degC (~18)\n"
        f"siconc global    : {float(sic.mean()):.1f} %  (~6)\n"
        f"pr global        : {float(pr.mean()):.2f} mm/d (~2.7)\n\n"
        "Earth-like albedo/OLR; residual\n"
        "cold = incomplete slab spin-up\n"
        "(R_TOA>0 area-wtd => warming back)"
    )
    axt.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10)

    axes.flat[8].axis("off")
    fig.suptitle("legoESM coupled CMIP — combined full-realistic run (90 days)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
