#!/usr/bin/env python
"""Global maps of the science-grade land-carbon initial condition (finidat).

Loads ``results/global_carbon_ic/global_carbon_ic.npz`` (per-cell 8-pool
CarbonState + lat/lon/land_mask + permafrost phi on the CLM5 1.9x2.5 grid) and
renders global maps of the derived pools -- SOC (soil organic carbon = the three
SOM pools), live biomass (labile+foliage+root+wood), and the perennial-frost
index phi -- plus the zonal-mean SOC vs the observed ~9.5 kgC/m2 line, so the
ARCTIC-localised deficit (the residual) is visible.

Compute-light (numpy load + matplotlib), but run via sbatch on a compute node
per the login-node policy.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

_G_PER_KG = 1000.0
_SOM_POOLS = ("C_som_active", "C_som_slow", "C_som_passive")
_LIVE_POOLS = ("C_lab", "C_fol", "C_root", "C_wood")


def _grid(field, land_mask, n_lat, n_lon):
    """Per-cell (ncell,) row-major -> (n_lat, n_lon), ocean/no-veg -> NaN."""
    f = np.where(land_mask, field, np.nan)
    return f.reshape(n_lat, n_lon)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finidat",
                   default="results/global_carbon_ic/global_carbon_ic.npz")
    p.add_argument("--out", default="results/global_carbon_ic/carbon_ic_maps.png")
    args = p.parse_args(argv)

    z = np.load(args.finidat, allow_pickle=True)
    lat = np.asarray(z["lat"], float)
    lon = np.asarray(z["lon"], float)
    lm = np.asarray(z["land_mask"]).astype(bool)
    lat1d = np.unique(lat)
    lon1d = np.unique(lon)
    n_lat, n_lon = lat1d.size, lon1d.size

    soc = sum(np.asarray(z[k], float) for k in _SOM_POOLS) / _G_PER_KG   # kgC/m2
    bio = sum(np.asarray(z[k], float) for k in _LIVE_POOLS) / _G_PER_KG  # kgC/m2
    phi = np.asarray(z["soil_frozen_fraction"], float)

    # lon 0..360 -> -180..180 for a centred map (roll the columns).
    lon_plot = np.where(lon1d > 180.0, lon1d - 360.0, lon1d)
    order = np.argsort(lon_plot)
    lon_plot = lon_plot[order]

    def field2d(a):
        g = _grid(a, lm, n_lat, n_lon)
        return g[:, order]

    fig = plt.figure(figsize=(13, 11))
    panels = [
        ("SOC (soil organic C)  [kgC/m$^2$]", field2d(soc), "YlOrBr", 0, 25),
        ("Live biomass  [kgC/m$^2$]", field2d(bio), "YlGn", 0, 25),
        ("Permafrost index $\\phi$ (annual frozen fraction)", field2d(phi),
         "Blues", 0, 1),
    ]
    for i, (title, data, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.3")
        ax.set_global()
        m = ax.pcolormesh(lon_plot, lat1d, data, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), shading="auto")
        fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
        ax.set_title(title, fontsize=11)

    # Zonal-mean SOC vs the observed ~9.5 kgC/m2 target (the arctic deficit).
    ax = fig.add_subplot(2, 2, 4)
    soc2d = _grid(soc, lm, n_lat, n_lon)
    with np.errstate(invalid="ignore"):
        zonal = np.nanmean(soc2d, axis=1)
    ax.plot(zonal, lat1d, color="#b35806", lw=2, label="model SOC")
    ax.axvline(9.5, color="0.4", ls="--", lw=1, label="obs ~9.5")
    ax.set_xlabel("zonal-mean SOC [kgC/m$^2$]")
    ax.set_ylabel("latitude")
    ax.set_ylim(-60, 85)
    ax.axhspan(60, 85, color="#c6dbef", alpha=0.5, label="Arctic (deficit)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Zonal-mean SOC: arctic deficit vs obs", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("Science-grade global land-carbon IC (real ERA5 climate, "
                 "drift 0.025 %/yr)", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"# wrote {args.out}")

    # console summary
    m = lm
    w = np.cos(np.deg2rad(lat)) * m
    print(f"# area-wt SOC {np.average(soc, weights=w):.2f} | biomass "
          f"{np.average(bio, weights=w):.2f} kgC/m2  (obs SOC ~9.5)")


if __name__ == "__main__":
    main()
