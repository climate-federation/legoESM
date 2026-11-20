"""Does the production ocean bulk-flux law reproduce ERA5's El Nino latent heat anomaly?

Offline: ERA5 surface inputs (10 m wind, 2 m T/dewpoint, SST, p_s) for Dec 1997
and Dec 1991-2000 go through the MPAS production deck's ocean flux law
(``large_yeager_cesm`` via ``compute_surface_fluxes``, sea-water surface
humidity as the CLUBB integration builds it).  The Dec-1997 minus climatology
anomaly of latent heat flux is scored against ERA5's own flux, and split into a
wind-speed term and a thermodynamic (humidity-difference) term by swapping
inputs through the same function (paired by day/hour; residual = interaction).
The source store carries no surface stress, so the zonal stress anomaly is the
bulk formula on ERA5 winds only (no ERA5 stress to score against).

Source: WeatherBench2 ERA5, 1.5 deg conservative, 6-hourly (``_WB2``).  Its
``mean_surface_latent_heat_flux`` at stamp t is the mean over the hour BEFORE t,
so each December uses stamps Dec 1 06 UTC .. Jan 1 00 UTC (124 samples, every
flux window inside December) for inputs and fluxes alike; the inputs are
instantaneous at t, half an hour after the flux window's centre.

  fetch (login node, single process): python scripts/validate/enso_bulk_flux_anomaly.py fetch --out DIR
  score: JAX_ENABLE_X64=1 python scripts/validate/enso_bulk_flux_anomaly.py score --out DIR

Detail choices (votes author/GLM/codex): z_bot=10 m with 2 m T,q (a/a/a);
climatology includes 1997 (a/b/a); 6-hourly (a/a/a, after transfer cost);
per-sample fluxes (a/a/a); swap split (a/a/a); ratio + pattern correlation
(a/a/a); ERA5 mean-rate fluxes (a/a/a).  1.5 deg WB2 store: user choice.
"""
import argparse
import hashlib
import os
import subprocess

import numpy as np

_WB2 = ("gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
_IN = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
       "2m_dewpoint_temperature", "sea_surface_temperature", "surface_pressure"]
_OUT = ["mean_surface_latent_heat_flux"]
_YEARS = range(1991, 2001)
_BOX = dict(latitude=slice(-30.0, 30.0), longitude=slice(120.0, 280.0))
_NINO34 = (-5.0, 5.0, 190.0, 240.0)
_MAX_BYTES = 1_000_000_000  # user cap on the login-node transfer


def stamps(y):
    import pandas as pd
    return pd.date_range(f"{y}-12-01 06:00", f"{y + 1}-01-01 00:00", freq="6h")


def fetch(out):
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_WB2), chunks=None)
    # whole-globe chunks: count the chunks touched and refuse past the cap
    idx = [ds.indexes["time"].get_indexer(stamps(y)) for y in _YEARS]
    assert all((i >= 0).all() for i in idx), "a December stamp is missing from the store"
    chunks = sorted({int(c) for i in idx for c in i // 8})
    est = sum(fs.info(f"{_WB2}/{v}/{c}.0.0")["size"] for v in _IN + _OUT for c in chunks)
    print(f"# transfer {est / 1e6:.0f} MB ({len(chunks)} time chunks per variable)")
    assert est < _MAX_BYTES, est
    for y in _YEARS:
        f = os.path.join(out, f"era5_dec{y}.npz")
        if os.path.exists(f):
            continue
        sel = ds[_IN + _OUT].sel(time=stamps(y)).sel(**_BOX).load()
        sel = sel.transpose("time", "latitude", "longitude")
        assert sel.sizes["time"] == 124 and sel.sizes["latitude"] == 41 \
            and sel.sizes["longitude"] == 107, sel.sizes  # 1.5 deg, 30S-30N, 120E-279E
        np.savez(f, time=stamps(y).values.astype("datetime64[s]").astype("int64"),
                 lat=sel.latitude.values, lon=sel.longitude.values,
                 **{v: sel[v].values for v in _IN + _OUT})
        print(f"# saved {f}", flush=True)
    st = ds["land_sea_mask"].sel(**_BOX).transpose("latitude", "longitude").load()
    np.savez(os.path.join(out, "static.npz"), lsm=st.values,
             lat=st.latitude.values, lon=st.longitude.values)


def _provenance(out):
    h = hashlib.sha256()
    for name in ["static.npz"] + [f"era5_dec{y}.npz" for y in _YEARS]:
        with open(os.path.join(out, name), "rb") as fh:
            h.update(fh.read())
    return f"# source {_WB2}; sha256(static+10 Decembers) {h.hexdigest()[:16]}"


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
    t0 = np.arange(6, 745, 6) * 3600  # Dec 1 06 .. Jan 1 00, seconds after Dec 1 00
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
    ocean = st["lsm"] < 0.5
    check_inputs(d, ocean, lat, lon)  # any non-finite SST on an ocean cell is fatal
    from legoesm import constants
    sst_min = min(float(d[y]["sea_surface_temperature"][:, ocean].min()) for y in _YEARS)
    assert sst_min > constants.T_freeze, f"ocean SST {sst_min} K: sea ice in the box"
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
    ec_l = np.mean(list(e_l.values()), 0)
    a_l = e_l[1997] - ec_l
    assert 50.0 < _wmean(ec_l, w) < 250.0, "ERA5 LHF sign/units: ocean clim must be ~100 W/m2 up"

    print(f"# window: Dec 1 06 - Jan 1 00 UTC 6-hourly (flux = hour before stamp); clim Dec "
          f"{_YEARS[0]}-{_YEARS[-1]} incl 1997; ocean cells {int(ocean.sum())}; 1.5 deg")
    print(_provenance(out))
    print(f"clim LHF  model {_wmean(clim_l, w):7.2f}  ERA5 {_wmean(ec_l, w):7.2f} W/m2 (wide)")
    print(f"clim taux model {_wmean(clim_t, w):7.4f} Pa (wide; no ERA5 stress in source)")
    for name, ww in (("nino34", w34), ("wide", w)):
        print(f"[{name}] LHF anom model {_wmean(m_l, ww):7.2f} ERA5 {_wmean(a_l, ww):7.2f} "
              f"ratio {_wmean(m_l, ww) / _wmean(a_l, ww):6.3f} | wind {_wmean(wind_l, ww):7.2f} "
              f"thermo {_wmean(thermo_l, ww):7.2f} resid {_wmean(m_l - wind_l - thermo_l, ww):6.2f}")
        print(f"[{name}] taux anom model only {_wmean(m_t, ww):8.4f} Pa | wind "
              f"{_wmean(wind_t, ww):8.4f} thermo {_wmean(thermo_t, ww):8.4f}")
        print(f"[{name}] slope model-on-ERA5 anom LHF {_wslope(m_l, a_l, ww):6.3f}")
    print(f"[wide] pattern corr anom LHF {_wcorr(m_l, a_l, w):.3f}; "
          f"clim LHF {_wcorr(clim_l, ec_l, w):.3f}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    print(f"# git {sha}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["fetch", "score"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fetch(a.out) if a.cmd == "fetch" else score(a.out)
