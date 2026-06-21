"""Global offline land validation against ERA5 (slab / multilayer+Richards).

Drives the legoESM land model on a coarse (~2deg) global land mask with REAL ERA5
surface forcing (a monthly climatology streamed from Google ARCO-ERA5), free-
running for a few years, and validates the simulated surface temperature against
ERA5 ``skin_temperature``.  Both surface schemes are covered:

* ``--scheme slab``       — 1-layer bucket slab (``land.slab_land.step_land``)
* ``--scheme multilayer`` — 8-layer soil-thermal + Richards soil-moisture column
                            (``land.multilayer_land.step_multilayer_land``)

Forcing + land mask + elevation + the validation target all come from ERA5
(``2m_temperature``, dewpoint, ``surface_solar/thermal_radiation_downwards``,
precip, ``surface_pressure``, 10 m wind, ``land_sea_mask``,
``geopotential_at_surface``, ``skin_temperature``) so no fabricated geography.

NOTE: a monthly-mean climatology has no diurnal cycle, so the equilibrium surface
temperature carries a known warm bias (Jensen inequality on the sigma*T^4 term)
that is largest over hot/dry land; it is a property of the offline forcing, not
the land model, and vanishes under a coupled (diurnally-resolved) atmosphere.

Requires the ``data`` extra (gcsfs) for the fetch.  Run::

  PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/validate_land_era5.py \
      --scheme multilayer --years 3 --plot

The fetched climatology is cached (``--cache``) so re-runs skip the network.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# ARCO-ERA5 (Google public, anonymous) — analysis-ready, all surface variables.
_ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_FORCING_VARS = ("2m_temperature", "2m_dewpoint_temperature",
                 "surface_solar_radiation_downwards",
                 "surface_thermal_radiation_downwards",
                 "total_precipitation", "surface_pressure",
                 "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "skin_temperature")
_SCHEMES = ("slab", "multilayer")
_G = 9.80665


# --------------------------------------------------------------------------- #
# ERA5 fetch (network; cached)                                                #
# --------------------------------------------------------------------------- #
def fetch_era5_climatology(cache: str, stride: int = 8,
                           years=(2019, 2020), days=(6, 16, 26),
                           hours=(0, 6, 12, 18)) -> dict:
    """Stream a coarse ERA5 monthly surface climatology; cache to ``cache`` (npz)."""
    if os.path.exists(cache):
        return dict(np.load(cache))
    try:
        import xarray as xr
        import gcsfs
        import pandas as pd
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError("ERA5 fetch needs the 'data' extra (gcsfs) + pandas: "
                          f"pip install -e '.[data]'  (missing: {e.name})") from e
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO_ERA5), chunks=None)
    sub = ds.isel(latitude=slice(0, 721, stride), longitude=slice(0, 1440, stride))
    times = [pd.Timestamp(y, m, d, h) for y in years for m in range(1, 13)
             for d in days for h in hours]
    clim = sub[list(_FORCING_VARS)].sel(time=times).load().groupby("time.month").mean("time")
    out = {v: np.asarray(clim[v].values) for v in _FORCING_VARS}
    out["ssrd_wm2"] = out.pop("surface_solar_radiation_downwards") / 3600.0
    out["strd_wm2"] = out.pop("surface_thermal_radiation_downwards") / 3600.0
    out["precip_kgms"] = out.pop("total_precipitation") * 1000.0 / 3600.0
    st = sub[["geopotential_at_surface", "land_sea_mask"]].sel(
        time=pd.Timestamp(years[-1], 1, 1, 0)).load()
    out["elev_m"] = np.asarray(st["geopotential_at_surface"].values) / _G
    out["lsm"] = np.asarray(st["land_sea_mask"].values)
    out["lat"] = np.asarray(sub.latitude.values)
    out["lon"] = np.asarray(sub.longitude.values)
    np.savez(cache, **out)
    return out


# --------------------------------------------------------------------------- #
# Forcing (pure; no network — unit-testable)                                  #
# --------------------------------------------------------------------------- #
def build_monthly_forcing(clim: dict, cols):
    """List of 12 ``AtmToSurface`` (one per month) on the given column indices.

    ``cols`` selects flattened (lat*lon) cells to run; all fields become (ncol,).
    """
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.core.coupling_fields import AtmToSurface

    def C(name):
        return jnp.asarray(clim[name].reshape(12, -1)[:, cols], dtype=jnp.float64)
    t2m = C("2m_temperature"); d2m = C("2m_dewpoint_temperature")
    ssrd = C("ssrd_wm2"); strd = C("strd_wm2")
    prcp = jnp.maximum(C("precip_kgms"), 0.0); sp = C("surface_pressure")
    u10 = C("10m_u_component_of_wind"); v10 = C("10m_v_component_of_wind")
    ncol = len(cols)
    z = lambda v: jnp.full((ncol,), v)
    out = []
    for m in range(12):
        T, Td, psfc = t2m[m], d2m[m], sp[m]
        q = saturation_mixing_ratio(Td, psfc)
        precip = prcp[m]
        out.append(AtmToSurface(
            sw_down=ssrd[m], lw_down=strd[m], precip_total=precip,
            precip_snow=jnp.where(T < constants.T_freeze, precip, 0.0),
            T_lowest=T, q_lowest=q, u_lowest=u10[m], v_lowest=v10[m],
            p_lowest=0.99 * psfc, p_surface=psfc,
            rho_lowest=psfc / (constants.R_d * T), cos_zenith=z(0.5),
            co2_ppmv=z(412.0), has_radiation=z(1.0), has_precipitation=z(1.0)))
    return out, np.asarray(t2m[0])


def validation_metrics(model_T, skt_cols):
    """Bias / RMSE of model monthly T_sfc vs ERA5 monthly skin temperature."""
    err = model_T - skt_cols
    ann = model_T.mean(0) - skt_cols.mean(0)
    return dict(bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())),
                annual_bias=float(ann.mean()),
                annual_rmse=float(np.sqrt((ann ** 2).mean())))


# --------------------------------------------------------------------------- #
# Run (scheme dispatch)                                                        #
# --------------------------------------------------------------------------- #
def run_land(scheme: str, clim: dict, years: int = 3, dt: float = 3600.0,
             land_only: bool = True):
    """Free-run the chosen land scheme and return (monthly_T_sfc, skt, cols, extra)."""
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}.")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    lat1, lon1 = clim["lat"], clim["lon"]
    nlat, nlon = lat1.size, lon1.size
    land2 = clim["lsm"].reshape(nlat, nlon) > 0.5
    cols = np.where(land2.ravel())[0] if land_only else np.arange(nlat * nlon)
    latc = jnp.asarray(np.deg2rad(np.broadcast_to(lat1[:, None], (nlat, nlon)))
                       .ravel()[cols])
    skt = clim["skin_temperature"].reshape(12, -1)[:, cols]
    forc, t2m0 = build_monthly_forcing(clim, cols)
    steps_per_month = int(round(30.4 * 86400 / dt))
    onehots = [tuple(1.0 if m == k else 0.0 for k in range(12)) for m in range(12)]

    if scheme == "slab":
        monthly_T, extra = _run_slab(forc, onehots, latc, t2m0, len(cols),
                                     steps_per_month, years)
    else:
        monthly_T, extra = _run_multilayer(forc, onehots, latc, t2m0, len(cols),
                                            steps_per_month, years)
    return monthly_T, skt, cols, extra


def _run_slab(forc, onehots, latc, t2m0, ncol, spm, years):
    import jax, jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.land import LandState
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land
    cfg = LandConfig(snow_albedo_feedback=True)

    def state(a):
        F = lambda d, u: Field(d, name="x", dims=("col",), units=u)
        return LandState(T_soil=F(a[0], "K"), W_bucket=F(a[1], "kg/m2"),
                         snow_depth=F(a[2], "kg/m2"), snow_age=F(a[3], "s"))

    @jax.jit
    def run_month(a, oh):
        f = jax.tree.map(lambda *xs: sum(o * x for o, x in zip(oh, xs)), *forc)

        def body(c, _):
            ar, ts = c
            ns, r, _ = step_land(state(ar), f, cfg, 1.0, 3600.0, lat=latc, doy=15.0)
            return ((ns.T_soil.data, ns.W_bucket.data, ns.snow_depth.data,
                     ns.snow_age.data), ts + r.T_sfc), None
        (a, ts), _ = jax.lax.scan(body, (a, jnp.zeros((ncol,))), None, length=spm)
        return a, ts / spm

    a = (jnp.asarray(t2m0, jnp.float64), jnp.full((ncol,), 75.0),
         jnp.zeros((ncol,)), jnp.zeros((ncol,)))
    mT = None
    for _ in range(years):
        cols_T = []
        for m in range(12):
            a, tm = run_month(a, onehots[m]); cols_T.append(tm)
        mT = np.stack([np.asarray(x) for x in cols_T])
    return mT, {}


def _run_multilayer(forc, onehots, latc, t2m0, ncol, spm, years):
    import jax, jax.numpy as jnp
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.multilayer_land import (step_multilayer_land,
                                              init_multilayer_land_state)
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0, growth_factor=1.5),
        bulk_scheme="constant", snow_albedo_feedback=True)

    @jax.jit
    def run_month(st, oh):
        f = jax.tree.map(lambda *xs: sum(o * x for o, x in zip(oh, xs)), *forc)

        def body(c, _):
            s, ts = c
            s2, r, _ = step_multilayer_land(s, f, cfg, 1.0, 3600.0, lat=latc, doy=15.0)
            return (s2, ts + r.T_sfc), None
        (st, ts), _ = jax.lax.scan(body, (st, jnp.zeros((ncol,))), None, length=spm)
        return st, ts / spm

    st = init_multilayer_land_state(ncol, cfg, T_init=280.0)
    st = st._replace(T_soil=jnp.broadcast_to(
        jnp.asarray(t2m0, jnp.float64)[:, None], st.T_soil.shape))
    st = jax.tree.map(lambda x: x.astype(jnp.float64), st)
    mT = None
    for _ in range(years):
        cols_T = []
        for m in range(12):
            st, tm = run_month(st, onehots[m]); cols_T.append(tm)
        mT = np.stack([np.asarray(x) for x in cols_T])
    return mT, {"theta_soil": np.asarray(st.theta_soil)}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", choices=_SCHEMES, default="multilayer")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--stride", type=int, default=8, help="0.25deg ERA5 stride (~2deg=8)")
    p.add_argument("--cache", default="/tmp/era5_land_clim.npz")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    clim = fetch_era5_climatology(args.cache, stride=args.stride)
    nland = int((clim["lsm"] > 0.5).sum())
    print(f"# ERA5 land validation: scheme={args.scheme} years={args.years} "
          f"land={nland} cells  elev<= {clim['elev_m'].max():.0f} m")
    mT, skt, cols, extra = run_land(args.scheme, clim, years=args.years)
    M = validation_metrics(mT, skt)
    print(f"# T_sfc vs ERA5 skin_temperature (land): bias={M['bias']:+.2f} K  "
          f"RMSE={M['rmse']:.2f} K  annual bias={M['annual_bias']:+.2f} K  "
          f"RMSE={M['annual_rmse']:.2f} K")
    if "theta_soil" in extra:
        th = extra["theta_soil"]
        print(f"# Richards theta: top {th[:,0].min():.2f}-{th[:,0].max():.2f}  "
              f"col-mean {th.mean():.2f}")
    if args.plot:
        _plot(clim, mT, skt, cols, args.scheme)


def _plot(clim, mT, skt, cols, scheme):  # pragma: no cover - I/O
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat1, lon1 = clim["lat"], clim["lon"]; nlat, nlon = lat1.size, lon1.size
    Tm = np.full(nlat * nlon, np.nan); Tm[cols] = mT.mean(0) - 273.15
    Ts = np.full(nlat * nlon, np.nan); Ts[cols] = skt.mean(0) - 273.15
    Tm = Tm.reshape(nlat, nlon); Ts = Ts.reshape(nlat, nlon)
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
    for a, d, t, cm, vlo, vhi in [
        (ax[0], Tm, f"{scheme} model T (degC)", "RdYlBu_r", -40, 35),
        (ax[1], Ts, "ERA5 skin T (degC)", "RdYlBu_r", -40, 35),
        (ax[2], Tm - Ts, "bias (K)", "RdBu_r", -10, 10)]:
        im = a.pcolormesh(lon1, lat1, d, cmap=cm, shading="auto", vmin=vlo, vmax=vhi)
        a.set_title(t); plt.colorbar(im, ax=a)
    out = f"/tmp/land_era5_{scheme}.png"
    plt.tight_layout(); plt.savefig(out, dpi=95)
    print(f"# saved {out}")


if __name__ == "__main__":
    main()
