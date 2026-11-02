"""Monthly MARBL surface deposition forcing on the legoESM grid, from CESM/POP input data.

Sources (CESM inputdata, POP gx1v6 grid, 12 monthly means):
  dust_flux  <- dst79gnx_gx1v6_090416.nc                              DSTSF  [kg/m2/s]
  iron_flux  <- solFe_scenario4_current_gx1v6_8gmol_cesm1_93_20161114.nc  DSTSF  [kg Fe/m2/s] (soluble Fe)
  nox_flux   <- ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc             NOy_deposition [kg N/m2/s]
  nhy_flux   <- same file                                              NHx_deposition [kg N/m2/s]
Output units (as expected by the JAX MARBL driver): dust kg/m2/s, Fe and N mmol/m2/s.
Sea-ice fraction is not written (the coupling then uses ice where SST < -1 C).

Regridding: every POP T cell is assigned to the model cell containing its centre and
averaged with its area (TAREA) as weight; model cells receiving no POP cell are filled
from the nearest filled cell.  Global ocean totals are printed as a check.

Usage:
  python scripts/build_marbl_forcing.py --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg.npz \
      --out /work/bb1596/b384036/diffESM/data/marbl_forcing_1deg.nc
"""
import argparse
from pathlib import Path

import numpy as np

CESM = "/work/bb1346/cesm/inputdata/ocn/pop/gx1v6/forcing"
SOURCES = {   # name: (file, variable, factor to model units, unit label for totals, total factor)
    "dust_flux": ("dst79gnx_gx1v6_090416.nc", "DSTSF", 1.0, "Tg dust/yr", 1e-9),          # kg -> Tg
    "iron_flux": ("solFe_scenario4_current_gx1v6_8gmol_cesm1_93_20161114.nc", "DSTSF",
                  1e6 / 55.845, "Gmol Fe/yr", 1e-3 * 1e-9),                               # kg Fe -> mmol
    "nox_flux":  ("ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc", "NOy_deposition",
                  1e6 / 14.007, "Tg N/yr", 1e-3 * 14.007 * 1e-12),                         # kg N -> mmol N
    "nhy_flux":  ("ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc", "NHx_deposition",
                  1e6 / 14.007, "Tg N/yr", 1e-3 * 14.007 * 1e-12),
}
SEC_PER_YEAR = 365.0 * 86400.0


def model_grid(path):
    g = np.load(path)
    lat, lon, mask = np.asarray(g["lat"], float), np.asarray(g["lon"], float), g["mask"] > 0.5
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    return lat, lon, mask


def cell_area_deg(latd, lond):
    lat, lon = np.deg2rad(latd), np.deg2rad(lond)
    dlat, dlon = np.abs(np.gradient(lat)), np.abs(np.gradient(lon))
    band = np.sin(np.minimum(lat + dlat / 2, np.pi / 2)) - np.sin(np.maximum(lat - dlat / 2, -np.pi / 2))
    return 6.371e6 ** 2 * band[:, None] * dlon[None, :]


def bin_to_grid(src, src_lat, src_lon, w, latd, lond):
    """Area-weighted mean of source cells by target cell; empty cells -> nearest filled."""
    from scipy import ndimage
    nlat, nlon = latd.size, lond.size
    lat_edges = np.concatenate([[-90.0], 0.5 * (latd[1:] + latd[:-1]), [90.0]])
    dlon = float(np.median(np.diff(lond)))
    ii = np.clip(np.searchsorted(lat_edges, src_lat, side="right") - 1, 0, nlat - 1)
    jj = np.floor(np.mod(src_lon - (lond[0] - dlon / 2), 360.0) / dlon).astype(int) % nlon
    ok = np.isfinite(src) & np.isfinite(w) & (w > 0)
    idx = (ii * nlon + jj)[ok]
    num = np.bincount(idx, weights=(src * w)[ok], minlength=nlat * nlon)
    den = np.bincount(idx, weights=w[ok], minlength=nlat * nlon)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan).reshape(nlat, nlon)
    bad = ~np.isfinite(out)
    if bad.any():
        k = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        out = out[tuple(k)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cesm-dir", default=CESM)
    a = ap.parse_args()
    import xarray as xr

    latd, lond, mask = model_grid(a.grid)
    area = cell_area_deg(latd, lond)
    fields = {}
    for name, (fn, var, fac, ulab, tfac) in SOURCES.items():
        ds = xr.open_dataset(Path(a.cesm_dir) / fn, decode_times=False)
        tlat = np.asarray(ds["TLAT"].values, float)
        tlon = np.asarray(ds["TLONG"].values, float)
        tarea = np.asarray(ds["TAREA"].values, float) * 1e-4           # cm2 -> m2
        raw = np.asarray(ds[var].values, float)                         # (12, Y, X)
        fill = ds[var].attrs.get("_FillValue", ds[var].encoding.get("_FillValue"))
        if fill is not None:
            raw = np.where(raw == fill, np.nan, raw)
        raw = np.where(np.abs(raw) > 1e10, np.nan, raw)
        out = np.stack([bin_to_grid(raw[m].ravel(), tlat.ravel(), tlon.ravel(), tarea.ravel(), latd, lond)
                        for m in range(raw.shape[0])]) * fac
        fields[name] = out
        src_tot = np.nanmean([np.nansum(raw[m] * tarea) for m in range(raw.shape[0])]) * fac * SEC_PER_YEAR * tfac
        mod_tot = np.mean([np.sum(out[m] * area * mask) for m in range(out.shape[0])]) * SEC_PER_YEAR * tfac
        print(f"{name:10s} <- {fn}:{var}  months {raw.shape[0]}  "
              f"global total {src_tot:9.3g} {ulab} (POP grid, all cells)  -> {mod_tot:9.3g} {ulab} (model ocean)")

    ds_out = xr.Dataset(
        {k: (("month", "lat", "lon"), v.astype(np.float64)) for k, v in fields.items()},
        coords={"month": np.arange(1, 13), "lat": latd, "lon": lond},
        attrs={"description": "MARBL monthly surface deposition for legoESM, regridded from CESM POP gx1v6 "
                              "input data (dust: dst79gnx; soluble Fe: solFe scenario4 8 Gmol; "
                              "N: ndep_ocn 1850 NOy/NHx)",
               "units": "dust_flux kg/m2/s; iron_flux mmol Fe/m2/s; nox_flux, nhy_flux mmol N/m2/s"},
    )
    ds_out.to_netcdf(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
