#!/usr/bin/env python
"""Per-cell / zonal / per-biome validation of the model global SOC map against a
gridded SOC OBSERVATION (built by ``scripts/data/build_soilgrids_soc.py``).

The CUE-fixed carbon IC has only been checked against a single global-mean SOC
(~11 kgC/m2 obs vs 4.12 model land-mean).  A single mean cannot localise the bias.
This validator loads the model SOC map
(``results/global_carbon_ic_cuefix2/global_carbon_ic.npz`` -- per cell
``(C_som_active + C_som_slow + C_som_passive) / 1000`` [kgC/m2]) and the obs on the
SAME grid, then quantifies WHERE the model departs from obs:

* a global AREA-weighted (cos-latitude) bias, RMSE and spatial correlation over land;
* a ZONAL-MEAN profile (model vs obs vs latitude) -- reveals a high-latitude vs
  tropical concentration of the bias;
* a per-BIOME / per-dominant-PFT bias table (biomes via the shared
  :data:`legoesm.land.carbon.realism_ranges.PFT_BIOME`);
* model / obs / (model - obs) maps + the zonal profile as one PNG.

The verdict (uniform low?  high-latitude peat cap?  tropics low?) tells a sibling
permafrost-carbon fix whether high-latitude SOC is the right lever.

Constants: cos-latitude area weights for the means; exact spherical cell area
(``legoesm.constants.R_earth``) for the total-stock PgC reporting.

Login-node policy: importing ``legoesm`` pulls JAX -> run on a COMPUTE node.  The
pure statistic / zonal / biome helpers are offline-unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ===========================================================================
# Grid reshaping + area weights
# ===========================================================================
def reshape_flat_to_grid(flat, lat_flat, lon_flat, *, fill=np.nan):
    """Scatter a flattened per-cell field onto a ``(nlat, nlon)`` grid.

    Robust to the flattening order: places each cell by its (lat, lon) value via
    ``searchsorted`` on the UNIQUE ascending centres, so no C/F-order assumption.
    Returns ``(field2d, lat, lon)`` with lat/lon ascending.
    """
    lat_u = np.unique(np.asarray(lat_flat, dtype=np.float64))
    lon_u = np.unique(np.asarray(lon_flat, dtype=np.float64))
    ilat = np.searchsorted(lat_u, np.asarray(lat_flat, dtype=np.float64))
    ilon = np.searchsorted(lon_u, np.asarray(lon_flat, dtype=np.float64))
    flat = np.asarray(flat)
    out = np.full((lat_u.size, lon_u.size), fill, dtype=np.result_type(flat, float)
                  if fill is np.nan else flat.dtype)
    out[ilat, ilon] = flat
    return out, lat_u, lon_u


def cos_lat_weight(lat):
    """cos(latitude) area weight (proportional to cell area for uniform d-lat),
    clipped >= 0 so the poles carry zero weight.  ``lat`` 1-D [deg] -> 1-D weight."""
    return np.clip(np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64))), 0.0, None)


def cell_area_m2(lat, lon):
    """Exact spherical cell areas ``(nlat, nlon)`` [m2] using ``constants.R_earth``.

    ``A = R^2 * dlon * (sin(lat_north_edge) - sin(lat_south_edge))`` per cell, from
    the cell EDGES (midpoints of the centres, extrapolated at the ends).  Used only
    for the total-stock PgC reporting (the weighted means use cos-latitude).
    """
    from legoesm import constants

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    def _edges(c):
        mid = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])

    lat_e = np.clip(_edges(lat), -90.0, 90.0)
    lon_e = _edges(lon)
    dsin = np.abs(np.sin(np.deg2rad(lat_e[1:])) - np.sin(np.deg2rad(lat_e[:-1])))  # (nlat,)
    dlon = np.abs(np.deg2rad(lon_e[1:] - lon_e[:-1]))                              # (nlon,)
    r2 = float(constants.R_earth) ** 2
    return r2 * dsin[:, None] * dlon[None, :]


# ===========================================================================
# Weighted statistics (offline-unit-tested)
# ===========================================================================
def weighted_stats(model, obs, weight, mask):
    """Area-weighted bias / RMSE / spatial correlation over ``mask`` cells.

    All inputs share one shape.  ``bias = <model - obs>_w`` (positive => model high),
    ``rmse = sqrt(<(model-obs)^2>_w)``, ``corr`` = weighted Pearson correlation.
    Returns a dict incl. weighted model/obs means and the valid-cell count.
    """
    model = np.asarray(model, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(model) & np.isfinite(obs) & (w > 0)
    n = int(m.sum())
    if n == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, corr=np.nan,
                    model_mean=np.nan, obs_mean=np.nan, wsum=0.0)
    wm = w[m]
    x = model[m]
    y = obs[m]
    wsum = float(wm.sum())
    xbar = float((wm * x).sum() / wsum)
    ybar = float((wm * y).sum() / wsum)
    diff = x - y
    bias = float((wm * diff).sum() / wsum)
    rmse = float(np.sqrt((wm * diff ** 2).sum() / wsum))
    cov = float((wm * (x - xbar) * (y - ybar)).sum() / wsum)
    vx = float((wm * (x - xbar) ** 2).sum() / wsum)
    vy = float((wm * (y - ybar) ** 2).sum() / wsum)
    corr = cov / np.sqrt(vx * vy) if (vx > 0 and vy > 0) else np.nan
    return dict(n=n, bias=bias, rmse=rmse, corr=float(corr),
                model_mean=xbar, obs_mean=ybar, wsum=wsum)


def zonal_profile(model2d, obs2d, weight2d, mask2d, lat):
    """Per-latitude area-weighted model & obs zonal means (over valid lon cells).

    Returns ``(lat, model_zonal, obs_zonal, bias_zonal)`` -- 1-D over latitude; a
    band with no valid cell is NaN.  Reveals whether the bias concentrates poleward.
    """
    model2d = np.asarray(model2d, dtype=np.float64)
    obs2d = np.asarray(obs2d, dtype=np.float64)
    w = np.asarray(weight2d, dtype=np.float64)
    m = np.asarray(mask2d, bool) & np.isfinite(model2d) & np.isfinite(obs2d) & (w > 0)
    ww = np.where(m, w, 0.0)
    den = ww.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mz = np.where(den > 0, np.nansum(np.where(m, model2d, 0.0) * ww, axis=1) / den, np.nan)
        oz = np.where(den > 0, np.nansum(np.where(m, obs2d, 0.0) * ww, axis=1) / den, np.nan)
    return np.asarray(lat, dtype=np.float64), mz, oz, mz - oz


def per_biome_bias(model2d, obs2d, weight2d, mask2d, dompft2d, pft_names):
    """Area-weighted per-biome model/obs SOC + bias, keyed by dominant-PFT biome.

    Dominant PFT index -> PFT name -> biome via the shared ``PFT_BIOME`` table
    (``None`` biome, e.g. bare_soil, is bucketed as ``"bare/none"``).  Returns a list
    of dict rows sorted by descending |bias|, each with the Jobbagy SOC range for
    context.  A pure grouping over the SAME cells the global stats use.
    """
    from legoesm.land.carbon.realism_ranges import (
        LITERATURE_BIOME_RANGES, PFT_BIOME)

    model2d = np.asarray(model2d, dtype=np.float64)
    obs2d = np.asarray(obs2d, dtype=np.float64)
    w = np.asarray(weight2d, dtype=np.float64)
    dompft = np.asarray(dompft2d)
    m = np.asarray(mask2d, bool) & np.isfinite(model2d) & np.isfinite(obs2d) & (w > 0)
    names = list(pft_names)

    def _biome(idx):
        if idx < 0 or idx >= len(names):
            return "bare/none"
        return PFT_BIOME.get(str(names[idx])) or "bare/none"

    biome_of = np.vectorize(_biome, otypes=[object])(dompft)
    rows = []
    for biome in sorted(set(biome_of[m].tolist())):
        sel = m & (biome_of == biome)
        if not sel.any():
            continue
        st = weighted_stats(model2d, obs2d, w, sel)
        soc_range = LITERATURE_BIOME_RANGES.get(biome, {}).get("soc")
        rows.append(dict(biome=biome, n=st["n"], model_mean=st["model_mean"],
                         obs_mean=st["obs_mean"], bias=st["bias"],
                         jobbagy_soc_0_100=list(soc_range) if soc_range else None))
    rows.sort(key=lambda r: -abs(r["bias"]) if np.isfinite(r["bias"]) else 0.0)
    return rows


# ===========================================================================
# Load + align + assess
# ===========================================================================
_POOL_FIELDS = ("C_som_active", "C_som_slow", "C_som_passive")


def load_model_soc(ic_npz):
    """Model SOC map [kgC/m2] on its grid + land mask + dominant-PFT + geometry.

    SOC = sum of the SOM pools / 1000 (gC/m2 -> kgC/m2).  Returns
    ``(soc2d, lat, lon, land2d, dompft2d, pft_names)`` all on the ascending IC grid.
    """
    d = np.load(ic_npz, allow_pickle=True)
    soc_flat = np.zeros_like(np.asarray(d[_POOL_FIELDS[0]], dtype=np.float64))
    for f in _POOL_FIELDS:
        soc_flat = soc_flat + np.asarray(d[f], dtype=np.float64)
    soc_flat = soc_flat / 1000.0
    lat_flat = np.asarray(d["lat"], dtype=np.float64)
    lon_flat = np.asarray(d["lon"], dtype=np.float64)
    soc2d, lat, lon = reshape_flat_to_grid(soc_flat, lat_flat, lon_flat)
    land2d, _, _ = reshape_flat_to_grid(
        np.asarray(d["land_mask"], dtype=bool), lat_flat, lon_flat, fill=False)
    dompft2d, _, _ = reshape_flat_to_grid(
        np.asarray(d["dominant_pft"], dtype=np.int64), lat_flat, lon_flat, fill=-1)
    pft_names = [str(x) for x in np.asarray(d["pft_names"])]
    return soc2d, lat, lon, land2d.astype(bool), dompft2d.astype(np.int64), pft_names


def align_obs(obs_nc, lat, lon):
    """Load the obs NetCDF ``soc`` [kgC/m2] and put it on the model ``(lat, lon)``.

    Same grid -> used directly; a different grid -> conservatively regridded.  A
    UNIT-mismatch guard rejects an obs whose cos-lat mean is implausible for kgC/m2
    (e.g. a gC/m2 field 1000x too large), so a units bug fails loudly not silently.
    """
    import xarray as xr

    ds = xr.open_dataset(obs_nc)
    try:
        if "soc" not in ds:
            raise SystemExit(f"obs {obs_nc} has no 'soc' variable ({list(ds.data_vars)}).")
        soc = np.asarray(ds["soc"].values, dtype=np.float64)
        olat = np.asarray(ds["lat"].values, dtype=np.float64)
        olon = np.asarray(ds["lon"].values, dtype=np.float64)
        units = str(ds["soc"].attrs.get("units", "")).strip()
    finally:
        ds.close()

    same = (olat.shape == lat.shape and olon.shape == lon.shape
            and np.allclose(olat, lat) and np.allclose(olon, lon))
    if not same:
        from legoesm.grids.regridding import conservative_regrid_latlon
        soc = conservative_regrid_latlon(soc, olat, olon, lat, lon)

    finite = np.isfinite(soc)
    if finite.any():
        w = cos_lat_weight(lat)[:, None] * np.ones_like(soc)
        wmean = float(np.nansum(np.where(finite, soc, 0.0) * w)
                      / np.nansum(np.where(finite, w, 0.0)))
        if not (0.05 < wmean < 200.0):
            raise SystemExit(
                f"obs SOC cos-lat mean {wmean:.4g} implausible for kgC/m2 "
                f"(units attr {units!r}); suspected unit mismatch (gC/m2 vs kgC/m2).")
    return soc


def total_pgc(field2d, area2d, mask2d):
    """Area-integrated stock [PgC] over ``mask`` (1 PgC = 1e12 kgC)."""
    f = np.asarray(field2d, dtype=np.float64)
    m = np.asarray(mask2d, bool) & np.isfinite(f)
    return float(np.nansum(np.where(m, f, 0.0) * np.asarray(area2d, dtype=np.float64) * m) / 1.0e12)


def assess(ic_npz, obs_nc):
    """Full comparison -> a dict of global stats, zonal profile, per-biome rows,
    totals, and the 2-D fields for plotting."""
    model2d, lat, lon, land2d, dompft2d, pft_names = load_model_soc(ic_npz)
    obs2d = align_obs(obs_nc, lat, lon)
    w1d = cos_lat_weight(lat)
    w2d = w1d[:, None] * np.ones_like(model2d)
    mask2d = land2d & np.isfinite(model2d) & np.isfinite(obs2d)

    glob = weighted_stats(model2d, obs2d, w2d, mask2d)
    zlat, mz, oz, bz = zonal_profile(model2d, obs2d, w2d, mask2d, lat)
    biome_rows = per_biome_bias(model2d, obs2d, w2d, mask2d, dompft2d, pft_names)

    area2d = cell_area_m2(lat, lon)
    result = dict(
        ic_npz=os.path.abspath(ic_npz), obs_nc=os.path.abspath(obs_nc),
        n_land=int(mask2d.sum()), **{f"global_{k}": v for k, v in glob.items()},
        model_total_PgC=total_pgc(model2d, area2d, mask2d),
        obs_total_PgC=total_pgc(obs2d, area2d, mask2d),
        zonal=dict(lat=zlat.tolist(), model=mz.tolist(), obs=oz.tolist(), bias=bz.tolist()),
        per_biome=biome_rows,
        _fields=dict(model2d=model2d, obs2d=obs2d, lat=lat, lon=lon,
                     mask2d=mask2d, zlat=zlat, mz=mz, oz=oz),
    )
    return result


# ===========================================================================
# Plot
# ===========================================================================
def plot_soc_comparison(result, out_png):
    """model / obs / (model-obs) maps + zonal profile -> one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = result["_fields"]
    lat, lon = f["lat"], f["lon"]
    model = np.where(f["mask2d"], f["model2d"], np.nan)
    obs = np.where(f["mask2d"], f["obs2d"], np.nan)
    diff = model - obs
    vmax = float(np.nanpercentile(np.concatenate([model[np.isfinite(model)],
                                                  obs[np.isfinite(obs)]]), 98)) if \
        np.isfinite(model).any() else 30.0
    dmax = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)) if \
        np.isfinite(diff).any() else 10.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for ax, fld, title, cmap, vlo, vhi in (
            (axes[0, 0], model, "model SOC [kgC/m2]", "viridis", 0, vmax),
            (axes[0, 1], obs, "obs SOC [kgC/m2]", "viridis", 0, vmax),
            (axes[1, 0], diff, "model - obs [kgC/m2]", "RdBu_r", -dmax, dmax)):
        im = ax.pcolormesh(lon, lat, fld, cmap=cmap, vmin=vlo, vmax=vhi, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.8)

    axz = axes[1, 1]
    axz.plot(f["mz"], f["zlat"], "-", color="C3", label="model")
    axz.plot(f["oz"], f["zlat"], "-", color="C0", label="obs")
    axz.axvline(0.0, color="k", lw=0.5)
    axz.fill_betweenx(f["zlat"], f["mz"], f["oz"], color="grey", alpha=0.3)
    axz.set_title("zonal-mean SOC (cos-lat weighted)")
    axz.set_xlabel("SOC [kgC/m2]"); axz.set_ylabel("lat"); axz.legend()
    axz.grid(alpha=0.3)

    g = result
    fig.suptitle(
        f"SOC model vs obs  |  area-wt bias {g['global_bias']:+.2f} kgC/m2, "
        f"RMSE {g['global_rmse']:.2f}, corr {g['global_corr']:.2f}  |  "
        f"model {g['model_total_PgC']:.0f} vs obs {g['obs_total_PgC']:.0f} PgC")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _print_report(result):
    g = result
    print(f"\n# ===== SOC model vs obs ({os.path.basename(g['obs_nc'])}) =====")
    print(f"# land cells: {g['n_land']}")
    print(f"# area-weighted (cos-lat) global: bias {g['global_bias']:+.3f} kgC/m2  "
          f"(model {g['global_model_mean']:.3f} - obs {g['global_obs_mean']:.3f})  "
          f"RMSE {g['global_rmse']:.3f}  corr {g['global_corr']:.3f}")
    print(f"# total stock: model {g['model_total_PgC']:.1f} PgC  vs  obs {g['obs_total_PgC']:.1f} PgC")
    print("# per-biome (|bias| desc):")
    print(f"#   {'biome':<18} {'n':>5} {'model':>8} {'obs':>8} {'bias':>8}  jobbagy_0-100")
    for r in g["per_biome"]:
        jb = r["jobbagy_soc_0_100"]
        print(f"#   {r['biome']:<18} {r['n']:>5} {r['model_mean']:>8.2f} "
              f"{r['obs_mean']:>8.2f} {r['bias']:>+8.2f}  {jb if jb else ''}")
    z = g["zonal"]
    lat = np.asarray(z["lat"]); bz = np.asarray(z["bias"])
    if np.isfinite(bz).any():
        k = int(np.nanargmax(np.abs(bz)))
        print(f"# zonal peak |bias| at lat {lat[k]:+.0f}: model {z['model'][k]:.1f} "
              f"vs obs {z['obs'][k]:.1f} ({bz[k]:+.1f} kgC/m2)")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-npz", default="results/global_carbon_ic_cuefix2/global_carbon_ic.npz")
    p.add_argument("--obs", default="results/soc_obs/soc_obs.nc",
                   help="obs NetCDF from build_soilgrids_soc.py (dims lat,lon; var 'soc')")
    p.add_argument("--out-png", default="results/soc_obs/soc_vs_gridded.png")
    p.add_argument("--out-json", default="results/soc_obs/soc_vs_gridded.json")
    return p


def _jsonable(result):
    r = {k: v for k, v in result.items() if k != "_fields"}
    return r


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = assess(args.ic_npz, args.obs)
    _print_report(result)
    png = plot_soc_comparison(result, args.out_png)
    print(f"# wrote plot {png}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(_jsonable(result), fh, indent=2, default=float)
    print(f"# wrote metrics {args.out_json}")
    return result


if __name__ == "__main__":
    main()
