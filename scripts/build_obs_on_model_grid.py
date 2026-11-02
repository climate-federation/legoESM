"""Calibration / validation data on the legoESM 1-degree model grid (one call builds all products).

Surface products -> monthly climatologies (12, lat, lon), written to <out>_surface_monthly.nc
3-D products     -> annual means (lat, lon, lev at the model layer centres), <out>_3d_annual.nc
CESM2 (MARBL in CESM, CMIP6 OMIP2) -> monthly climatologies of the surface fields, for the
"same BGC model in another ocean model" comparison, written into the surface file as CESM2_*.

All fields are on the model land mask (taken from a model restart: land mask and sea-floor
depth); model ocean cells without data stay NaN in the surface file (no filling: the
evaluation must not compare against invented values), and are filled from the nearest value
at the same depth in the 3-D file (as for WOA in build_woa_on_model_grid.py).

Units after conversion (same as the model diagnostics):
  chl mg m-3 | NPP, export g C m-2 yr-1 | fgco2 mol C m-2 yr-1 (+ into the ocean) | spco2 uatm
  dissic, talk, no3, o2 mmol m-3 | pH | mlotst m

Every product's global mean / integral is printed, as a check of the units.

Usage:
  python scripts/build_obs_on_model_grid.py \
     --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg_mc.npz \
     --restart <a restart_dayNNNNNN.npz of the run> \
     --out /work/bb1596/b384036/diffESM/data/obs_1deg_mc
  (optional: --mld FILE VAR for a mixed-layer-depth climatology, --skip NAME ... to leave products out)
"""
import argparse
import glob
from pathlib import Path

import numpy as np

OBS = "/work/bd0854/DATA/ESMValTool2/OBS/Tier2"
CESM = "/pool/data/CMIP6/data/OMIP/NCAR/CESM2/omip2/r1i1p1f1/Omon"
# OMIP2 time axis: model years 1-366 = 6 cycles of JRA55-do 1958-2018; years 337-366 = 1989-2018 of the last cycle
SEC_PER_YEAR = 365.0 * 86400.0
M_C = 12.011

# name: (file glob, variable, kind, years (first, last) or None = all)
SURFACE = {
    "chl_ESACCI":       (f"{OBS}/ESACCI-OC/OBS6_ESACCI-OC_sat_fv5.0_Omon_chl_*.nc", "chl", "chl", (1998, 2020)),
    "NPP_VGPM":         (f"{OBS}/Eppley-VGPM-MODIS/OBS_Eppley-VGPM-MODIS_sat_R2018_Omon_intpp_*.nc", "intpp", "carbonflux", (2003, 2018)),
    "fgco2_Landschuetzer2016": (f"{OBS}/Landschuetzer2016/OBS_Landschuetzer2016_clim_v2016_Omon_fgco2_*.nc", "fgco2", "fgco2", None),
    "spco2_Landschuetzer2016": (f"{OBS}/Landschuetzer2016/OBS_Landschuetzer2016_clim_v2016_Omon_spco2_*.nc", "spco2", "spco2", None),
    "fgco2_OceanSODA":  (f"{OBS}/OceanSODA-ETHZ/OBS6_OceanSODA-ETHZ_reanaly_v2023_Omon_fgco2_*.nc", "fgco2", "fgco2", (1990, 2019)),
    "spco2_OceanSODA":  (f"{OBS}/OceanSODA-ETHZ/OBS6_OceanSODA-ETHZ_reanaly_v2023_Omon_spco2_*.nc", "spco2", "spco2", (1990, 2019)),
    "dissic_MOBO":      (f"{OBS}/MOBO-DIC2004-2019/OBS6_MOBO-DIC2004-2019_reanaly_2.3_Omon_dissic_*.nc", "dissic", "conc", None),
    "CESM2_chl":        (f"{CESM}/chl/gn/*/chl_*.nc", "chl", "chl", (337, 366)),
    "CESM2_NPP":        (f"{CESM}/intpp/gn/*/intpp_*.nc", "intpp", "carbonflux", (337, 366)),
    "CESM2_export":     (f"{CESM}/epc100/gn/*/epc100_*.nc", "epc100", "carbonflux", (337, 366)),
    "CESM2_fgco2":      (f"{CESM}/fgco2/gn/*/fgco2_*.nc", "fgco2", "fgco2", (337, 366)),
    "CESM2_no3":        (f"{CESM}/no3/gn/*/no3_*.nc", "no3", "conc", (337, 366)),
    "CESM2_o2":         (f"{CESM}/o2/gn/*/o2_*.nc", "o2", "conc", (337, 366)),
}
THREE_D = {
    "DIC_GLODAP": (f"{OBS}/GLODAP/OBS6_GLODAP_clim_v2.2016b_Oyr_dissic_*.nc", "dissic", "conc"),
    "ALK_GLODAP": (f"{OBS}/GLODAP/OBS6_GLODAP_clim_v2.2016b_Oyr_talk_*.nc", "talk", "conc"),
    "pH_GLODAP":  (f"{OBS}/GLODAP/OBS6_GLODAP_clim_v2.2016b_Oyr_ph_*.nc", "ph", "none"),
}


# ------------------------------------------------------------------ units
def convert(kind, a, units):
    u = (units or "").replace(" ", "").lower()
    if kind == "chl":                                    # kg m-3 -> mg m-3
        return a * 1e6 if ("kg" in u or np.nanmedian(a) < 1e-3) else a
    if kind == "carbonflux":                             # mol C m-2 s-1 or mg C m-2 d-1 -> g C m-2 yr-1
        if "mol" in u and "s-1" in u.replace("/s", "s-1"):
            return a * M_C * SEC_PER_YEAR
        if "mg" in u and ("d-1" in u or "day" in u or "/d" in u):
            return a * 1e-3 * 365.0
        return a * M_C * SEC_PER_YEAR if np.nanmax(np.abs(a)) < 1e-5 else a
    if kind == "fgco2":                                  # kg C m-2 s-1 -> mol C m-2 yr-1
        return a / (M_C * 1e-3) * SEC_PER_YEAR if ("kg" in u or np.nanmax(np.abs(a)) < 1e-5) else a
    if kind == "spco2":                                  # Pa -> uatm
        return a / 0.101325 if (u == "pa" or np.nanmedian(a) > 1000) else a
    if kind == "conc":                                   # mol m-3 -> mmol m-3
        if ("molm-3" in u or "mol/m3" in u) and not any(p in u for p in ("mmol", "umol", "µmol")):
            return a * 1e3
        if "umol/kg" in u or "µmol/kg" in u or "umolkg-1" in u:
            return a * 1.025
        return a * 1e3 if np.nanmedian(a) < 0.1 else a
    return a


# ------------------------------------------------------------------ regridding
def _dims(da):
    latn = next((d for d in da.dims if d.lower().startswith(("lat", "nlat", "j", "y"))), None)
    lonn = next((d for d in da.dims if d.lower().startswith(("lon", "nlon", "i", "x"))), None)
    return latn, lonn


def regular_to_model(da, latd, lond):
    """(..., lat, lon) on a regular grid -> model grid; block mean first if the source is finer."""
    import xarray as xr
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    da = da.sortby(latn)
    da = da.assign_coords({lonn: np.mod(da[lonn].values, 360.0)}).sortby(lonn)
    f = int(round(abs(float(np.diff(latd[:2])[0])) / abs(float(np.diff(da[latn].values[:2])[0]))))
    if f >= 2:
        da = da.coarsen({latn: f, lonn: f}, boundary="trim").mean()
    first, last = da.isel({lonn: [0]}), da.isel({lonn: [-1]})
    da = xr.concat([last.assign_coords({lonn: last[lonn].values - 360.0}), da,
                    first.assign_coords({lonn: first[lonn].values + 360.0})], dim=lonn)
    lq = np.mod(lond, 360.0); o = np.argsort(lq)
    out = da.interp({latn: latd, lonn: lq[o]}).values
    return out[..., np.argsort(o)]


def curvilinear_to_model(a, lat2, lon2, latd, lond):
    """(n, ny, nx) on a curvilinear grid (e.g. CESM gx1) -> model grid: cos(lat)-weighted mean of the
    source cells whose centre falls in each model cell; cells with no source cell stay NaN."""
    nlat, nlon = latd.size, lond.size
    lat_e = np.concatenate([[-90.0], 0.5 * (latd[1:] + latd[:-1]), [90.0]])
    dl = float(np.median(np.diff(lond)))
    ii = np.clip(np.searchsorted(lat_e, lat2.ravel(), side="right") - 1, 0, nlat - 1)
    jj = (np.floor(np.mod(lon2.ravel() - (lond[0] - dl / 2), 360.0) / dl).astype(int)) % nlon
    idx = ii * nlon + jj
    w0 = np.cos(np.deg2rad(lat2.ravel()))
    out = np.full((a.shape[0], nlat, nlon), np.nan)
    for t in range(a.shape[0]):
        v = a[t].ravel(); ok = np.isfinite(v)
        num = np.bincount(idx[ok], weights=(v * w0)[ok], minlength=nlat * nlon)
        den = np.bincount(idx[ok], weights=w0[ok], minlength=nlat * nlon)
        out[t] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).reshape(nlat, nlon)
    return out


def monthly_climatology(files, var, years, latd, lond):
    """Monthly climatology (12, lat, lon) of the surface field; returns (field, units, info)."""
    import xarray as xr
    ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True) if len(files) > 1 \
        else xr.open_dataset(files[0], use_cftime=True)
    da = ds[var]
    units = da.attrs.get("units", "")
    for d in list(da.dims):                                       # surface level
        if d.lower() in ("lev", "depth", "olevel", "z", "z_t") or d.lower().startswith("depth"):
            da = da.isel({d: 0})
    tdim = next(d for d in da.dims if d.lower().startswith("time"))
    yrs = np.array([t.year for t in da[tdim].values])
    if years is not None:
        sel = (yrs >= years[0]) & (yrs <= years[1])
        if sel.any():
            da = da.isel({tdim: np.nonzero(sel)[0]}); yrs = yrs[sel]
    info = f"{yrs.min()}-{yrs.max()}" if yrs.size else "?"
    if da.sizes[tdim] > 12:
        da = da.groupby(f"{tdim}.month").mean(tdim)
        mdim = "month"
    else:
        mdim = tdim
    if "lat" in ds and ds["lat"].ndim == 2:                      # curvilinear (CESM gn)
        lat2, lon2 = np.asarray(ds["lat"].values, float), np.asarray(ds["lon"].values, float)
        a = np.asarray(da.transpose(mdim, ...).values, float)
        out = curvilinear_to_model(a, lat2, lon2, latd, lond)
    else:
        latn = next(d for d in da.dims if d.lower().startswith("lat"))
        lonn = next(d for d in da.dims if d.lower().startswith("lon"))
        out = regular_to_model(da.transpose(mdim, latn, lonn), latd, lond)
    if out.shape[0] == 1:
        out = np.repeat(out, 12, axis=0)
    return np.asarray(out, float), units, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>_surface_monthly.nc and <out>_3d_annual.nc")
    ap.add_argument("--mld", nargs=2, metavar=("FILE", "VAR"), default=None, help="mixed-layer-depth climatology")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    import xarray as xr
    from build_woa_on_model_grid import fill_nearest, load_woa

    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z, zh = np.abs(np.asarray(g["z"], float)), np.abs(np.asarray(g["z_half"], float))
    r = np.load(a.restart)
    ocean, H = r["land_mask"] > 0.5, r["H_bathy"]
    wet = ocean[..., None] & (H[..., None] > zh[:-1][None, None, :] + 1e-6)
    R = 6.371e6
    area = R ** 2 * np.deg2rad(1.0) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    vol = area[..., None] * np.diff(zh)[None, None, :] * wet

    surf = dict(SURFACE)
    if a.mld:
        surf["mlotst_obs"] = (a.mld[0], a.mld[1], "none", None)
    out_s = {}
    for name, (pat, var, kind, years) in surf.items():
        if name in a.skip:
            continue
        files = sorted(glob.glob(pat))
        if not files:
            print(f"{name:26s} MISSING  {pat}"); continue
        try:
            f, units, info = monthly_climatology(files, var, years, lat, lon)
        except Exception as e:                                    # keep going: one bad product must not stop the rest
            print(f"{name:26s} FAILED   {type(e).__name__}: {e}"); continue
        f = np.where(ocean[None], convert(kind, f, units), np.nan)
        if kind == "fgco2":                                       # sign: + into the ocean; the equatorial Pacific outgasses
            eq = (np.abs(lat) <= 5)[:, None] & ((lon >= 180) & (lon <= 270))[None, :]
            if np.nanmean(np.nanmean(f, axis=0)[eq]) > 0:
                f = -f; print(f"{name:26s} sign flipped (file has + = outgassing)")
        out_s[name] = f
        ann = np.nanmean(f, axis=0); cov = np.isfinite(ann) & ocean
        if kind in ("carbonflux",):
            tot = f"global {np.nansum(ann * area * cov) * 1e-15:7.2f} Pg C/yr (covered area only)"
        elif kind == "fgco2":
            tot = f"global {np.nansum(ann * area * cov) * M_C * 1e-15:+7.2f} Pg C/yr (covered area only)"
        else:
            tot = f"area mean {np.nansum(ann * area * cov) / np.sum(area * cov):9.4g}"
        print(f"{name:26s} {len(files):3d} file(s), years {info:9s} [{units}] -> {tot}, "
              f"ocean covered {100 * cov.sum() / ocean.sum():5.1f} %")
    if out_s:
        ds = xr.Dataset({k: (("month", "lat", "lon"), v.astype(np.float32)) for k, v in out_s.items()},
                        coords={"month": np.arange(1, 13), "lat": lat, "lon": lon})
        ds.attrs.update(description="observations and CESM2-OMIP2 monthly climatologies on the legoESM 1-degree grid "
                                    "(build_obs_on_model_grid.py); units: chl mg m-3, NPP/export g C m-2 yr-1, "
                                    "fgco2 mol C m-2 yr-1 (+ into ocean), spco2 uatm, dissic/no3/o2 mmol m-3, mlotst m; "
                                    "no filling (NaN = no data)", mask_from=str(a.restart))
        p = f"{a.out}_surface_monthly.nc"; ds.to_netcdf(p); print(f"wrote {p}\n")

    out_3 = {}
    for name, (pat, var, kind) in THREE_D.items():
        if name in a.skip:
            continue
        files = sorted(glob.glob(pat))
        if not files:
            print(f"{name:12s} MISSING  {pat}"); continue
        f, units, (z0, z1, nz) = load_woa(files[0], var, lat, lon, z)
        f = convert(kind, f, units)
        f, filled = fill_nearest(f, wet)
        out_3[name] = f
        print(f"{name:12s} <- {Path(files[0]).name} [{units}], depths {z0:.0f}-{z1:.0f} m ({nz}): "
              f"surface mean {np.nanmean(f[..., 0][wet[..., 0]]):8.4g}, volume mean {np.nansum(np.nan_to_num(f) * vol) / vol.sum():8.4g}, "
              f"filled {100 * filled.sum() / wet.sum():.1f} %")
    if out_3:
        ds = xr.Dataset({k: (("lat", "lon", "lev"), v.astype(np.float32)) for k, v in out_3.items()},
                        coords={"lat": lat, "lon": lon, "lev": z})
        ds.attrs.update(description="3-D annual observations on the legoESM 1-degree grid (build_obs_on_model_grid.py); "
                                    "DIC/ALK mmol m-3; cells without data filled from the nearest value at the same depth",
                        mask_from=str(a.restart))
        p = f"{a.out}_3d_annual.nc"; ds.to_netcdf(p); print(f"wrote {p}")


if __name__ == "__main__":
    main()
