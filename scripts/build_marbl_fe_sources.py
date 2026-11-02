"""Iron forcing for MARBL on the legoESM grid, version 2.

1. Soluble Fe deposition without the 45 S step.
   The CESM solFe "scenario4" file has a 5-6 fold lower Fe solubility south of 45 S, applied as a
   step between two grid rows.  Here the soluble fraction r = iron_flux / dust_flux is
   interpolated (in log r, along each longitude) between 52.5 S and 37.5 S, so the transition is
   smooth over 15 degrees; outside that band nothing changes.  The field is then rescaled so the
   global total equals that of the original file.  Other variables are copied unchanged.

2. Sediment and hydrothermal Fe sources (CESM2 inputdata, POP gx1v6):
     fesedfluxTot_gx1v6_cesm2_2018_c180618.nc   FESEDFLUXIN (z, y, x)
     feventflux_gx1v6_5gmol_cesm1_97_2017.nc    FESEDFLUXIN (z, y, x)
   in umol m-2 d-1 (the CESM2/POP convention; the vent file is named for its 5 Gmol/yr total,
   which the printed total confirms).  Each POP column is integrated in depth and put into the
   deepest wet cell of the model column that contains its centre (area-weighted with TAREA);
   flux falling on model land goes to the nearest model ocean column.  Totals are preserved.
   Output in mmol m-2 s-1 (per unit horizontal area), dims (lat, lon, lev).

Usage:
  python scripts/build_marbl_fe_sources.py \
     --forcing /work/bb1596/b384036/diffESM/data/marbl_forcing_1deg.nc \
     --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg_mc.npz \
     --restart <a restart_dayNNNNNN.npz on the bathy_1deg_etopo_mc mask> \
     --out-forcing /work/bb1596/b384036/diffESM/data/marbl_forcing_1deg_v2.nc \
     --out-sources /work/bb1596/b384036/diffESM/data/marbl_fe_sources_1deg_mc.nc
"""
import argparse

import numpy as np

CESM = "/work/bb1346/cesm/inputdata/ocn/pop/gx1v6/forcing"
SED = f"{CESM}/fesedfluxTot_gx1v6_cesm2_2018_c180618.nc"
VENT = f"{CESM}/feventflux_gx1v6_5gmol_cesm1_97_2017.nc"
POPGRID = f"{CESM}/dst79gnx_gx1v6_090416.nc"          # same gx1v6 grid; carries TLAT, TLONG, TAREA
SEC_PER_YEAR = 365.0 * 86400.0


def cell_area(latd, nlon):
    R = 6.371e6
    e = np.deg2rad(np.concatenate([[-90.0], 0.5 * (latd[1:] + latd[:-1]), [90.0]]))
    return (R ** 2 * (np.sin(e[1:]) - np.sin(e[:-1])) * 2 * np.pi / nlon)[:, None] * np.ones((1, nlon))


def smooth_solubility(iron, dust, latd, lo=-52.5, hi=-37.5):
    """iron, dust (12, nlat, nlon); log-linear interpolation of iron/dust between rows lo and hi."""
    out = iron.copy()
    i0, i1 = int(np.argmin(np.abs(latd - lo))), int(np.argmin(np.abs(latd - hi)))
    w = (latd[i0:i1 + 1] - latd[i0]) / (latd[i1] - latd[i0])                  # 0 at lo, 1 at hi
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dust > 0, iron / dust, np.nan)
    lr0, lr1 = np.log(r[:, i0, :]), np.log(r[:, i1, :])                       # (12, nlon)
    ok = np.isfinite(lr0) & np.isfinite(lr1)
    lr = lr0[:, None, :] + (lr1 - lr0)[:, None, :] * w[None, :, None]         # (12, band, nlon)
    new_band = np.where(ok[:, None, :] & (dust[:, i0:i1 + 1] > 0), dust[:, i0:i1 + 1] * np.exp(lr), iron[:, i0:i1 + 1])
    out[:, i0:i1 + 1] = new_band
    return out


def pop_to_bottom_cells(flux_pop_2d, tlat, tlon, tarea, latd, lond, ocean, kbot):
    """Area-weighted sum of a POP column-integrated flux [umol m-2 d-1] into model cells; returns
    (flux at the model bottom cell [mmol m-2 s-1] as (nlat, nlon), total in Gmol/yr before, after)."""
    from scipy import ndimage
    nlat, nlon = latd.size, lond.size
    lat_e = np.concatenate([[-90.0], 0.5 * (latd[1:] + latd[:-1]), [90.0]])
    dl = float(np.median(np.diff(lond)))
    ok = np.isfinite(flux_pop_2d) & (flux_pop_2d > 0) & np.isfinite(tarea)
    ii = np.clip(np.searchsorted(lat_e, tlat[ok], side="right") - 1, 0, nlat - 1)
    jj = (np.floor(np.mod(tlon[ok] - (lond[0] - dl / 2), 360.0) / dl).astype(int)) % nlon
    mol = flux_pop_2d[ok] * tarea[ok]                                          # umol/d per POP cell
    tot = np.bincount(ii * nlon + jj, weights=mol, minlength=nlat * nlon).reshape(nlat, nlon)
    total_before = mol.sum() * 365.0 * 1e-15                                   # Gmol/yr
    on_land = (tot > 0) & ~ocean
    if on_land.any():                                                          # move to the nearest ocean column
        idx = ndimage.distance_transform_edt(~ocean, return_distances=False, return_indices=True)
        ti, tj = idx[0][on_land], idx[1][on_land]
        np.add.at(tot, (ti, tj), tot[on_land]); tot[on_land] = 0.0
    area = cell_area(latd, nlon)
    flux = tot / area * 1e-3 / 86400.0                                         # umol/d -> mmol/m2/s
    total_after = np.sum(flux * area) * SEC_PER_YEAR * 1e-12                   # mmol/s -> Gmol/yr
    return flux, total_before, total_after, int(on_land.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--out-forcing", required=True)
    ap.add_argument("--out-sources", required=True)
    a = ap.parse_args()
    import xarray as xr

    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z, zh = np.abs(np.asarray(g["z"], float)), np.abs(np.asarray(g["z_half"], float))
    r = np.load(a.restart)
    ocean, H = r["land_mask"] > 0.5, r["H_bathy"]
    wet = ocean[..., None] & (H[..., None] > zh[:-1][None, None, :] + 1e-6)
    kbot = np.maximum(wet.sum(-1) - 1, 0)
    area = cell_area(lat, lon.size)

    # 1. deposition: smooth the solubility step
    fo = xr.open_dataset(a.forcing).load()
    iron, dust = np.asarray(fo["iron_flux"].values, float), np.asarray(fo["dust_flux"].values, float)
    tot = lambda f: np.mean([np.sum(f[m] * area * ocean) for m in range(f.shape[0])]) * SEC_PER_YEAR * 1e-12  # noqa: E731
    new = smooth_solubility(iron, dust, lat)
    new *= tot(iron) / tot(new)
    zm = lambda f: np.nanmean(np.where(ocean, f.mean(0), np.nan), axis=1)       # noqa: E731
    zo, zn = zm(iron), zm(new)
    print(f"soluble Fe deposition: total {tot(iron):.3f} Gmol/yr (original) -> {tot(new):.3f} Gmol/yr (smoothed)")
    print("  zonal mean [mmol m-2 s-1], original -> smoothed:")
    for la in (-55.5, -50.5, -46.5, -45.5, -44.5, -43.5, -40.5, -35.5):
        i = int(np.argmin(np.abs(lat - la)))
        print(f"   {lat[i]:6.1f}  {zo[i]:.3e} -> {zn[i]:.3e}")
    fo["iron_flux"] = (fo["iron_flux"].dims, new.astype(fo["iron_flux"].dtype))
    fo.attrs["description"] = fo.attrs.get("description", "") + \
        "; iron_flux: solubility step at 45S smoothed (log-linear in iron/dust, 52.5S-37.5S), total kept (build_marbl_fe_sources.py)"
    fo.to_netcdf(a.out_forcing)
    print(f"wrote {a.out_forcing}\n")

    # 2. sediment + vent sources
    pg = xr.open_dataset(POPGRID, decode_times=False)
    tlat, tlon = np.asarray(pg["TLAT"].values, float), np.asarray(pg["TLONG"].values, float)
    tarea = np.asarray(pg["TAREA"].values, float) * 1e-4                      # cm2 -> m2
    out = {}
    for name, path in (("fe_sed", SED), ("fe_vent", VENT)):
        d = xr.open_dataset(path, decode_times=False)["FESEDFLUXIN"]
        v = np.asarray(d.values, float)
        v = np.where(np.isfinite(v) & (np.abs(v) < 1e10), v, 0.0)
        col = v.sum(axis=0)                                                    # column integral [umol m-2 d-1]
        flux2d, tb, ta, nland = pop_to_bottom_cells(col, tlat, tlon, tarea, lat, lon, ocean, kbot)
        f3 = np.zeros(wet.shape)
        ii, jj = np.nonzero(flux2d > 0)
        f3[ii, jj, kbot[ii, jj]] = flux2d[ii, jj]
        out[name] = f3
        print(f"{name:7s} <- {path.split('/')[-1]}: POP levels {v.shape[0]}, units attr {d.attrs.get('units')!r}; "
              f"total {tb:.3f} Gmol/yr on the POP grid -> {ta:.3f} Gmol/yr on the model grid "
              f"({nland} model land cells moved to the nearest ocean column); "
              f"{int((flux2d > 0).sum())} model columns with a source")
    ds = xr.Dataset({k: (("lat", "lon", "lev"), v.astype(np.float64)) for k, v in out.items()},
                    coords={"lat": lat, "lon": lon, "lev": z})
    ds.attrs.update(units="mmol Fe m-2 s-1 (per horizontal area), in the deepest wet cell of each column",
                    description="CESM2 sediment (fesedfluxTot) and hydrothermal (feventflux) Fe sources, "
                                "column-integrated and put at the model sea floor (build_marbl_fe_sources.py)",
                    mask_from=str(a.restart))
    ds.to_netcdf(a.out_sources)
    print(f"wrote {a.out_sources}")


if __name__ == "__main__":
    main()
