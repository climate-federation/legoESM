"""Does the production ocean bulk-flux law reproduce ERA5's El Nino flux anomaly?

Offline: ERA5 surface inputs (10 m wind, 2 m T/dewpoint, SST, p_s) for Dec 1997
and Dec 1991-2000 go through the MPAS production deck's ocean flux law
(``large_yeager_cesm`` via ``compute_surface_fluxes``, sea-water surface
humidity as the CLUBB integration builds it).  The Dec-1997 minus climatology
anomaly of latent heat flux and zonal stress is scored against ERA5's own
fluxes, and the latent-heat anomaly is split into a wind-speed term and a
thermodynamic (humidity-difference) term by swapping inputs through the same
function (paired by day/hour; residual = interaction).

  fetch: python scripts/validate/enso_bulk_flux_anomaly.py fetch --out DIR
  score: JAX_ENABLE_X64=1 python scripts/validate/enso_bulk_flux_anomaly.py score --out DIR

Detail choices (votes author/GLM/codex): z_bot=10 m with 2 m T,q (a/a/a);
climatology includes 1997 (a/b/a); 1 deg (a/a/a); 6-hourly (a/a/a, after
transfer cost); per-sample fluxes (a/a/a); swap split (a/a/a); ratio + pattern
correlation (a/a/a); ERA5 mean-rate fluxes (a/a/a).
"""
import argparse
import os
import subprocess

import numpy as np

_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_IN = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
       "2m_dewpoint_temperature", "sea_surface_temperature", "surface_pressure"]
_OUT = ["mean_surface_latent_heat_flux", "mean_eastward_turbulent_surface_stress",
        "mean_surface_sensible_heat_flux"]
_YEARS = range(1991, 2001)
# 30N..30S, 120E..280E on the 0.25-deg ARCO grid (lat descending), 1-deg stride
_LAT, _LON = slice(240, 481, 4), slice(480, 1121, 4)
_NINO34 = (-5.0, 5.0, 190.0, 240.0)


def fetch(out):
    import gcsfs
    import pandas as pd
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO), chunks={})
    sub = ds.isel(latitude=_LAT, longitude=_LON)
    for y in _YEARS:
        f = os.path.join(out, f"era5_dec{y}.npz")
        if os.path.exists(f):
            continue
        t = pd.date_range(f"{y}-12-01", f"{y}-12-31 18:00", freq="6h")
        # ERA5 mean rates at hour h cover (h-1, h]: take h+1 so the flux window
        # starts at the instantaneous input time and never reaches November.
        si = sub[_IN].sel(time=t).compute(scheduler="threads", num_workers=16)
        so = sub[_OUT].sel(time=t + pd.Timedelta("1h")).compute(
            scheduler="threads", num_workers=16)
        np.savez(f, time=t.values.astype("datetime64[s]").astype("int64"),
                 lat=sub.latitude.values, lon=sub.longitude.values,
                 **{v: si[v].values for v in _IN}, **{v: so[v].values for v in _OUT})
        print(f"# saved {f}", flush=True)
    st = sub[["land_sea_mask"]].isel(time=0).compute()
    np.savez(os.path.join(out, "static.npz"), lsm=st.land_sea_mask.values,
             lat=sub.latitude.values, lon=sub.longitude.values)


def model_fluxes(u, v, t2, td2, sst, ps):
    """(LHF up [W/m2], zonal stress on the surface [Pa]) from the production law."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
    from legoesm.core.bulk_flux import ocean_surface_q_sat
    from legoesm.thermo import (mixing_ratio_to_specific_humidity, moist_air_density,
                                saturation_specific_humidity)
    cfg = SurfaceLayerConfig(bulk_scheme="large_yeager_cesm", ocean_q_sfc_saline=True)
    q2 = saturation_specific_humidity(td2, ps)  # dewpoint -> specific humidity
    q_sfc = constants.q_sat_saline_fraction * mixing_ratio_to_specific_humidity(
        ocean_surface_q_sat(sst, ps, thermo_convention=cfg.thermo_convention,
                            bulk_scheme=cfg.bulk_scheme))
    rho = moist_air_density(t2, ps, q2)
    # production hands the law a surface-referenced temperature (T + g z / c_p);
    # ERA5's air temperature is at 2 m, so the shift is 2 m (~0.02 K)
    theta = t2 + constants.g * 2.0 / constants.c_pd
    tau_x, _, _, lhf, _ = compute_surface_fluxes(
        u, v, theta, q2, sst, q_sfc, rho, cfg, z_ref=cfg.z_ref, t_bot=t2)
    return np.asarray(lhf), -np.asarray(tau_x)  # on-atmosphere -> on-surface


def check_inputs(d, ocean, lat, lon):
    """Fatal unless every year has the same 124 Dec samples, grid, and finite ocean data."""
    t0 = d[1997]["time"] - np.datetime64("1997-12-01", "s").astype("int64")
    for y, f in d.items():
        assert f["time"].shape == (124,), (y, f["time"].shape)
        assert np.array_equal(f["time"] - np.datetime64(f"{y}-12-01", "s").astype("int64"), t0), y
        assert np.array_equal(f["lat"], lat) and np.array_equal(f["lon"], lon), y
        for v in _IN + _OUT:
            bad = ~np.isfinite(f[v][:, ocean])
            assert not bad.any(), f"{y} {v}: {int(bad.sum())} non-finite ocean values"


def _wmean(x, w):
    m = np.isfinite(x) & (w > 0)
    return float((x[m] * w[m]).sum() / w[m].sum())


def _wcorr(a, b, w):
    m = np.isfinite(a) & np.isfinite(b) & (w > 0)
    a, b, w = a[m], b[m], w[m]
    a, b = a - (a * w).sum() / w.sum(), b - (b * w).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def _wslope(y, x, w):
    """Area-weighted regression slope of y on x (sign-robust amplitude ratio)."""
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    x, y = x - (x * w).sum() / w.sum(), y - (y * w).sum() / w.sum()
    return float((w * x * y).sum() / (w * x * x).sum())


def score(out):
    st = np.load(os.path.join(out, "static.npz"))
    lat, lon = st["lat"], st["lon"]
    d = {y: np.load(os.path.join(out, f"era5_dec{y}.npz")) for y in _YEARS}
    ocean = (st["lsm"] < 0.5) & np.all(np.isfinite(d[1997]["sea_surface_temperature"]), 0)
    check_inputs(d, ocean, lat, lon)
    w = np.where(ocean, np.cos(np.deg2rad(lat))[:, None], 0.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    w34 = np.where((la >= _NINO34[0]) & (la <= _NINO34[1]) & (lo >= _NINO34[2])
                   & (lo <= _NINO34[3]), w, 0.0)

    def run(wind_y, thermo_y):
        a, b = d[wind_y], d[thermo_y]
        sst = np.where(ocean, b["sea_surface_temperature"], 300.0)  # land masked later
        lhf, tau = model_fluxes(a["10m_u_component_of_wind"], a["10m_v_component_of_wind"],
                                b["2m_temperature"], b["2m_dewpoint_temperature"],
                                sst, b["surface_pressure"])
        return lhf.mean(0), tau.mean(0)

    base = {y: run(y, y) for y in _YEARS}
    clim_l = np.mean([base[y][0] for y in _YEARS], 0)
    clim_t = np.mean([base[y][1] for y in _YEARS], 0)
    sw = [run(1997, y) for y in _YEARS]
    st_ = [run(y, 1997) for y in _YEARS]
    wind_l = np.mean([r[0] for r in sw], 0) - clim_l
    thermo_l = np.mean([r[0] for r in st_], 0) - clim_l
    wind_t = np.mean([r[1] for r in sw], 0) - clim_t
    thermo_t = np.mean([r[1] for r in st_], 0) - clim_t
    m_l, m_t = base[1997][0] - clim_l, base[1997][1] - clim_t
    e_l = {y: -d[y]["mean_surface_latent_heat_flux"].mean(0) for y in _YEARS}  # ERA5 + down
    e_t = {y: d[y]["mean_eastward_turbulent_surface_stress"].mean(0) for y in _YEARS}
    ec_l, ec_t = np.mean(list(e_l.values()), 0), np.mean(list(e_t.values()), 0)
    a_l, a_t = e_l[1997] - ec_l, e_t[1997] - ec_t

    print(f"# window: Dec 1-31 6-hourly; clim Dec {_YEARS[0]}-{_YEARS[-1]} incl 1997; "
          f"ocean cells {int(ocean.sum())}; 1 deg")
    print(f"clim LHF  model {_wmean(clim_l, w):7.2f}  ERA5 {_wmean(ec_l, w):7.2f} W/m2 (wide)")
    print(f"clim taux model {_wmean(clim_t, w):7.4f}  ERA5 {_wmean(ec_t, w):7.4f} Pa (wide)")
    for name, ww in (("nino34", w34), ("wide", w)):
        print(f"[{name}] LHF anom model {_wmean(m_l, ww):7.2f} ERA5 {_wmean(a_l, ww):7.2f} "
              f"ratio {_wmean(m_l, ww) / _wmean(a_l, ww):6.3f} | wind {_wmean(wind_l, ww):7.2f} "
              f"thermo {_wmean(thermo_l, ww):7.2f} resid {_wmean(m_l - wind_l - thermo_l, ww):6.2f}")
        print(f"[{name}] taux anom model {_wmean(m_t, ww):8.4f} ERA5 {_wmean(a_t, ww):8.4f} "
              f"ratio {_wmean(m_t, ww) / _wmean(a_t, ww):6.3f} | wind {_wmean(wind_t, ww):8.4f} "
              f"thermo {_wmean(thermo_t, ww):8.4f}")
        print(f"[{name}] slope model-on-ERA5 anom LHF {_wslope(m_l, a_l, ww):6.3f} "
              f"taux {_wslope(m_t, a_t, ww):6.3f}")
    print(f"[wide] pattern corr LHF {_wcorr(m_l, a_l, w):.3f}  taux {_wcorr(m_t, a_t, w):.3f}")
    print(f"[wide] clim pattern corr LHF {_wcorr(clim_l, ec_l, w):.3f}  "
          f"taux {_wcorr(clim_t, ec_t, w):.3f}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    print(f"# git {sha}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["fetch", "score"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fetch(a.out) if a.cmd == "fetch" else score(a.out)
