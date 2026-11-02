"""Build the MARBL initial state on the legoESM model grid.

Sources:
  NO3, PO4, SiO3, O2   WOA18 annual climatology
  DIC, ALK             GLODAPv2.2016b mapped climatology (DIC_ALT_CO2 = DIC, ALK_ALT_CO2 = ALK)
  Fe                   analytic profile, 0.2 nM surface -> 0.6 nM deep
  NH4, Lig, DOM, plankton, zooplankton: analytic depth profiles (as in the 1-D column)

Each source field is interpolated bilinearly to the model lat/lon, gaps (coasts,
marginal seas, below the source's deepest level) are filled from the nearest
valid point on the same level, then interpolated linearly to the model levels.
Units: mmol/m^3 (ALK meq/m^3, Chl mg/m^3).  Land is 0.

Usage:
  # 1. write the model grid (same grid/bathymetry options as the run):
  python scripts/run_omip.py <grid/bathymetry options> --forcing-mode jra55_do_tropical \\
      --jra55-cache <cache> --biogeo marbl --marbl-write-grid marbl_grid_0p5.npz
  # 2. build the initial state:
  python scripts/build_marbl_ic.py --grid marbl_grid_0p5.npz --out marbl_ic_0p5.nc
"""
import argparse
from pathlib import Path

import numpy as np

OBS = "/work/bd0854/DATA/ESMValTool2/OBS/Tier2"
SOURCES = {
    "NO3":  (f"{OBS}/WOA/OBS6_WOA_clim_2018_Oyr_no3_200001-200012.nc", "no3"),
    "PO4":  (f"{OBS}/WOA/OBS6_WOA_clim_2018_Oyr_po4_200001-200012.nc", "po4"),
    "SiO3": (f"{OBS}/WOA/OBS6_WOA_clim_2018_Oyr_si_200001-200012.nc", "si"),
    "O2":   (f"{OBS}/WOA/OBS6_WOA_clim_2018_Oyr_o2_200001-200012.nc", "o2"),
    "DIC":  (f"{OBS}/GLODAP/OBS6_GLODAP_clim_v2.2016b_Oyr_dissic_200001-200012.nc", "dissic"),
    "ALK":  (f"{OBS}/GLODAP/OBS6_GLODAP_clim_v2.2016b_Oyr_talk_200001-200012.nc", "talk"),
}
TRACER_NAMES = ("PO4", "NO3", "SiO3", "NH4", "Fe", "Lig", "O2", "DIC", "DIC_ALT_CO2", "ALK",
                "ALK_ALT_CO2", "DOC", "DON", "DOP", "DOPr", "DONr", "DOCr", "zooC", "spChl",
                "spC", "spP", "spFe", "spCaCO3", "diatChl", "diatC", "diatP", "diatFe", "diatSi",
                "diazChl", "diazC", "diazP", "diazFe")
PLAUSIBLE = {"NO3": (0, 60), "PO4": (0, 4), "SiO3": (0, 200), "O2": (0, 450),
             "DIC": (1700, 2500), "ALK": (1900, 2600)}


def to_mmol(v, units):
    u = units.replace(" ", "").lower()
    if "mmol" in u:
        return v
    if "umol/kg" in u or "µmol/kg" in u or "umolkg-1" in u or "µmolkg-1" in u:
        return v * 1.025
    if "umol" in u or "µmol" in u:
        return v * 1e-3 if ("m-3" in u or "/m3" in u) else v
    if "ml/l" in u or "mll-1" in u:
        return v * 44.66
    if "mol" in u:
        return v * 1e3
    return v * 1e3 if np.nanmax(v) < 10 else v


def fill_nearest(a):
    """Fill NaNs of a 2-D field from the nearest valid point."""
    from scipy import ndimage
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def load_source(path, var, lat_t, lon_t, z_t):
    import xarray as xr
    da = xr.open_dataset(path, decode_times=False)[var]
    tdim = [d for d in da.dims if d.lower().startswith("time")]
    if tdim:
        da = da.mean(tdim[0])
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    zn = next(d for d in da.dims if d not in (latn, lonn))
    units = da.attrs.get("units", "")
    da = da.transpose(zn, latn, lonn).sortby(latn)
    # Longitudes to [0, 360) (GLODAP runs 20.5..379.5, others -180..180 or 0..360),
    # then pad one column on each side so the interpolation is periodic.
    da = da.assign_coords({lonn: np.mod(da[lonn].values, 360.0)}).sortby(lonn)
    first, last = da.isel({lonn: [0]}), da.isel({lonn: [-1]})
    da = xr.concat([last.assign_coords({lonn: last[lonn].values - 360.0}), da,
                    first.assign_coords({lonn: first[lonn].values + 360.0})], dim=lonn)

    lon_q = np.mod(lon_t, 360.0)
    order = np.argsort(lon_q)
    hor = da.interp({latn: lat_t, lonn: lon_q[order]}, method="linear").values   # (nz_src, nlat, nlon)
    hor = hor[:, :, np.argsort(order)]
    z_src = np.abs(np.asarray(da[zn].values, float))
    if da[zn].attrs.get("units", "").lower() in ("cm", "centimeters"):
        z_src = z_src / 100.0

    for k in range(hor.shape[0]):                       # horizontal gap filling per level
        if np.isfinite(hor[k]).any():
            hor[k] = fill_nearest(hor[k])
        elif k > 0:
            hor[k] = hor[k - 1]
    from scipy.interpolate import interp1d
    f = interp1d(z_src, hor, axis=0, bounds_error=False, fill_value=(hor[0], hor[-1]))
    out = np.moveaxis(f(z_t), 0, -1)                     # (nlat, nlon, nlev)
    for k in range(out.shape[-1]):                       # any residual gaps on the model levels
        if not np.isfinite(out[..., k]).all() and np.isfinite(out[..., k]).any():
            out[..., k] = fill_nearest(out[..., k])
    return to_mmol(out, units), units


def analytic(z):
    step = lambda zc, w: 0.5 * (1 + np.tanh((z - zc) / w))   # noqa: E731
    ez = lambda s: np.exp(-z / s)                               # noqa: E731
    spC, diatC, diazC = 0.5 * ez(100.0), 0.1 * ez(100.0), 0.05 * ez(80.0)
    DOC, DOCr = 20.0 * ez(150.0) + 1.0, np.full_like(z, 42.0)
    return dict(
        NH4=np.full_like(z, 0.01), Fe=2.0e-4 + 4.0e-4 * step(500.0, 300.0),
        Lig=np.full_like(z, 1.5e-3),
        DOC=DOC, DON=DOC / 12.0, DOP=DOC / 200.0,
        DOCr=DOCr, DONr=DOCr / 16.0, DOPr=DOCr / 1000.0,
        zooC=0.3 * ez(100.0),
        spC=spC, spChl=spC * 12.0 / 60.0, spP=spC / 117.0, spFe=spC * 2.0e-5, spCaCO3=spC * 0.07,
        diatC=diatC, diatChl=diatC * 12.0 / 50.0, diatP=diatC / 117.0,
        diatFe=diatC * 2.0e-5, diatSi=diatC * 0.13,
        diazC=diazC, diazChl=diazC * 12.0 / 60.0, diazP=diazC / 300.0, diazFe=diazC * 4.0e-5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help=".npz from run_omip.py --marbl-write-grid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = np.load(a.grid)
    lat, lon, z, mask = g["lat"], g["lon"], g["z"], g["mask"] > 0.5
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:          # model grid stores radians
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    print(f"lat {lat.min():.2f} .. {lat.max():.2f} deg, lon {lon.min():.2f} .. {lon.max():.2f} deg")
    shape = (lat.size, lon.size, z.size)
    print(f"model grid {shape}, levels {z[0]:.1f} .. {z[-1]:.0f} m, ocean fraction {mask.mean():.2f}")

    fields = {}
    for name, (path, var) in SOURCES.items():
        arr, units = load_source(path, var, lat, lon, z)
        fields[name] = np.maximum(arr, 0.0)
        wet = fields[name][mask]
        n_bad = int((~np.isfinite(wet)).sum())
        if n_bad:
            raise SystemExit(f"{name}: {n_bad} ocean points are still NaN after gap filling -- not writing")
        lo, hi = PLAUSIBLE[name]
        flag = "" if lo <= np.median(wet) <= hi else "   <-- CHECK UNITS"
        print(f"  {name:5s} from {Path(path).name} [{units}]: ocean min {wet.min():.3g}, "
              f"median {np.median(wet):.4g}, max {wet.max():.4g} mmol/m3{flag}")
    fields["DIC_ALT_CO2"] = fields["DIC"].copy()
    fields["ALK_ALT_CO2"] = fields["ALK"].copy()
    for k, prof in analytic(z).items():
        fields[k] = np.broadcast_to(prof, shape).copy()

    import xarray as xr
    m3 = mask[..., None]
    ds = xr.Dataset(
        {k: (("lat", "lon", "lev"), np.where(m3, fields[k], 0.0).astype(np.float32)) for k in TRACER_NAMES},
        coords={"lat": lat, "lon": lon, "lev": z},
        attrs={"description": "MARBL initial state for legoESM: WOA18 NO3/PO4/SiO3/O2, "
                              "GLODAPv2.2016b DIC/ALK, analytic Fe/DOM/plankton",
               "units": "mmol/m3 (ALK meq/m3, Chl mg/m3)"},
    )
    ds.to_netcdf(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
