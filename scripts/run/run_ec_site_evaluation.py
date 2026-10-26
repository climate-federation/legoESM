"""Offline EC-site (eddy-covariance flux tower) evaluation of the two-leaf canopy.

Runs the prognostic multilayer land model with the two-leaf canopy surface scheme
at a single FLUXNET site and writes one NetCDF with the modelled and observed
fluxes that ``scripts/plot/plot_ec_site_combined.py`` reads:

    modelled : gpp_mod, le_mod (+ le_canopy/le_soil partition), h_mod, ustar_mod,
               theta_prof (soil-moisture profile)
    observed : gpp_obs, le_obs, h_obs, ustar_obs, swc_obs, plus the energy-balance
               closure-corrected le_obs_corr / h_obs_corr (band edge in the figures)
    a ``valid`` mask (all-inputs-observed steps), a ``reverted`` mask (steps the
    NaN guard rolled back, so a model blow-up is distinguishable from missing
    observations), and the site attrs.

Per-site physics (tower height, rooting/column depth) come from the consolidated
``EC_SITE_PHYSICS`` table in ``run_ec_site.py``; soil hydraulics come from the
per-site texture table. Both are the reproducible defaults — no hand-set
environment is needed.

Usage
-----
    python scripts/run/run_ec_site_evaluation.py --site DE-Hai --out DE-Hai.nc \
        --driver-dir /path/to/sitelevel/nc --year-lo 2010 --year-hi 2013

The driver NetCDF is the DifferBESS ``<SITE>_driver_v2[_gapfree|_etcorr].nc``
half-hourly forcing (SW/LW down, T, q, wind, precip, CO2, plus per-step LAI /
Vcmax and the observed fluxes). See docs/land/ec_site_evaluation_runbook.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr

from legoesm import constants
from legoesm.land.boundary_data.ec_site import read_ec_site_driver
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land_with_diagnostics)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_ec_site import (  # noqa: E402
    _build_land_config, _texture_lookup, _DEFAULT_TEXTURE_CSV, ec_site_physics)

_GC_PER_UMOL_CO2 = 12.0e-6   # gC per umol CO2 (canopy GPP is gC/m2/s; obs is umol)
# Physical-plausibility bounds for the FLUXNET closure-corrected observations, used
# to reject fill sentinels (e.g. -9999) and gross outliers before they can leak into
# the closure-uncertainty band.  Generous FLUXNET daily-flux ranges, not tunables.
_ET_RANGE_MM_DAY = (-50.0, 100.0)     # daily evapotranspiration [mm/day]
_H_RANGE_W_M2 = (-1000.0, 1500.0)     # sensible heat flux [W/m2]


def _find_driver(driver_dir: str, site: str) -> str:
    """Prefer the pre-gapfilled driver; fall back to etcorr / base v2 (the reader
    gap-fills internally for sites without a ``*_gapfree.nc``)."""
    for suffix in ("_driver_v2_gapfree.nc", "_driver_v2_etcorr.nc", "_driver_v2.nc"):
        p = os.path.join(driver_dir, site + suffix)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no {site}_driver_v2[_gapfree|_etcorr].nc in {driver_dir!r}")


def run_evaluation(site: str, out_path: str, driver_dir: str,
                   year_lo: int = 0, year_hi: int = 9999,
                   stress_b0: bool = False) -> str:
    """Run the site and write the evaluation NetCDF; returns the output path."""
    nc = _find_driver(driver_dir, site)
    t = pd.DatetimeIndex(xr.open_dataset(nc)["time"].values)
    # The prognostic scan advances the soil state one fixed dt per row, so the
    # driver time axis must be strictly increasing; a non-monotonic axis would both
    # break the contiguous-slice year filter below and desynchronise the soil state
    # from real time.  Validate the axis itself (before the heavy reader), not just
    # the filtered positions.
    if not np.all(np.diff(t.values.astype("i8")) > 0):
        raise ValueError(f"{site}: driver time axis is not strictly increasing")
    d = read_ec_site_driver(nc)
    yr = t.year.to_numpy()
    idx = np.nonzero((yr >= year_lo) & (yr <= year_hi))[0]
    if idx.size == 0:
        raise ValueError(f"{site}: no steps in [{year_lo}, {year_hi}]")
    # With a strictly-increasing axis the year-filtered rows are contiguous; assert
    # it (defense in depth) so the slice below never straddles a gap.
    if not np.all(np.diff(idx) == 1):
        raise ValueError(
            f"{site}: year window [{year_lo}, {year_hi}] selects a non-contiguous "
            "span of the time axis; refusing to run")
    sl = slice(int(idx[0]), int(idx[-1] + 1))
    t = t[sl]
    n = len(t)

    phys = ec_site_physics(site)
    tex = _texture_lookup(site, _DEFAULT_TEXTURE_CSV)
    # stress_b0=False keeps the Ball-Berry cuticular intercept unstressed (the
    # validated default: a live canopy sustains a baseline transpiration; a
    # dormant/deciduous canopy self-limits via LAI -> 0).
    cc = TwoLeafCanopyConfig(max_iters=30, stress_b0=stress_b0)
    land = _build_land_config(
        cc, soil="auto", bottom_bc="free_drainage", depth_m=phys["soil_depth_m"],
        root_depth=phys["root_depth"], z_ref=phys["z_ref"], texture=tex)

    grid = make_soil_grid(land.soil_grid)
    znode = np.asarray(grid.z_node)

    Ts = np.asarray(d.T_soil_top).ravel()[sl]
    th = np.asarray(d.theta_soil).ravel()[sl]
    tr, tsat = float(land.hydraulics.theta_r), float(land.hydraulics.theta_sat)
    T_init = float(Ts[np.isfinite(Ts)][0])
    theta_init = float(np.clip(th[np.isfinite(th)][0], tr + 1e-3, tsat - 1e-3))
    state0 = init_multilayer_land_state(
        1, land, T_init=T_init, theta_init=theta_init,
        TgC_init=T_init - constants.T_freeze)

    def _nonfinite(s):
        return jnp.any(jnp.stack([~jnp.all(jnp.isfinite(getattr(s, k)))
                                  for k in s._asdict() if getattr(s, k) is not None]))

    def step(state, xs):
        f, p, doy = xs
        ns, _r, _c, out = step_multilayer_land_with_diagnostics(
            state, f, land, 1.0, d.dt_s, lat=None, carbon_state=None,
            doy=doy, land_params=p)
        rev = _nonfinite(ns)                     # revert to last good state on NaN
        safe = jax.tree_util.tree_map(lambda a, b: jnp.where(rev, b, a), ns, state)
        m = lambda v: jnp.where(rev, jnp.array(jnp.nan), v)
        ustar = jnp.sqrt(jnp.sqrt(out.tau_x ** 2 + out.tau_y ** 2)
                         / jnp.maximum(f.rho_lowest, 1e-3))
        _nan = jnp.full_like(out.lhflx, jnp.nan)
        lec = out.LE_canopy if out.LE_canopy is not None else _nan
        les = out.LE_soil if out.LE_soil is not None else _nan
        # Mask the emitted soil profile too (not just the fluxes): a reverted step
        # otherwise emits the frozen previous profile, which the plotter would read
        # as a real flat value.  ``reverted`` records where the model blew up so a
        # failure is distinguishable from merely-missing observations.
        emit = (m(out.gpp), m(out.lhflx), m(lec), m(les), m(out.shflx),
                m(ustar), m(safe.theta_soil[0]), rev.astype(jnp.float32))
        return safe, emit

    xs = (jax.tree_util.tree_map(lambda a: a[sl], d.forcing),
          jax.tree_util.tree_map(lambda a: a[sl], d.canopy_params), d.doy[sl])
    _f, out = jax.jit(lambda s, x: jax.lax.scan(step, s, x))(state0, xs)
    gpp, le, lec, les, h, ustar, theta_p, reverted = [np.asarray(a) for a in out]
    gpp = gpp.ravel() / _GC_PER_UMOL_CO2                       # gC/m2/s -> umol/m2/s

    obs = {k: np.asarray(d.obs[k]).ravel()[sl] for k in d.obs}
    # Energy-balance-closure-corrected turbulent fluxes (FLUXNET ET_CORR/H_CORR);
    # ET_CORR is mm/day -> W/m2 via L_v/86400.  Mask only clearly-corrupt DAYTIME
    # closure ratios (ET>1) so nighttime (ratio undefined at ET~0) is not masked.
    raw = xr.open_dataset(nc)
    def rawslice(name):
        return (np.asarray(raw[name].values).ravel()[sl]
                if name in raw.data_vars else np.full(n, np.nan))
    def _inrange(a, bounds):
        lo, hi = bounds
        return np.where(np.isfinite(a) & (a >= lo) & (a <= hi), a, np.nan)
    # Range-mask FIRST so fill sentinels (e.g. -9999) and gross outliers become NaN
    # and can never leak into the band; the nighttime ET~0 ratio is left alone.
    etc = _inrange(rawslice("ET_CORR"), _ET_RANGE_MM_DAY)
    et = _inrange(rawslice("ET"), _ET_RANGE_MM_DAY)
    hcorr = _inrange(rawslice("H_CORR"), _H_RANGE_W_M2)
    ratio = np.where(et > 1.0, etc / np.maximum(et, 1e-6), np.nan)
    bad = np.isfinite(ratio) & ((ratio < 0.3) | (ratio > 3.0))
    le_obs_corr = np.where(bad, np.nan, etc) * constants.L_v / 86400.0
    h_obs_corr = np.where(bad, np.nan, hcorr)

    ds = xr.Dataset(
        {"gpp_mod": ("time", gpp), "le_mod": ("time", le.ravel()),
         "le_canopy": ("time", lec.ravel()), "le_soil": ("time", les.ravel()),
         "h_mod": ("time", h.ravel()), "ustar_mod": ("time", ustar.ravel()),
         "theta_prof": (("time", "z"), theta_p),
         "gpp_obs": ("time", obs.get("gpp_umol", np.full(n, np.nan))),
         "le_obs": ("time", obs.get("le_wm2", np.full(n, np.nan))),
         "le_obs_corr": ("time", le_obs_corr),
         "h_obs": ("time", obs.get("h_wm2", np.full(n, np.nan))),
         "h_obs_corr": ("time", h_obs_corr),
         "ustar_obs": ("time", obs.get("ustar", np.full(n, np.nan))),
         "swc_obs": ("time", np.asarray(d.theta_soil).ravel()[sl]),
         "lai": ("time", np.asarray(d.canopy_params.LAI).ravel()[sl]),
         "reverted": ("time", reverted.ravel()),
         "valid": ("time", np.asarray(d.valid).ravel()[sl].astype(float))},
        coords={"time": t.values, "z": znode})
    ds.attrs.update(site=site, dt_s=float(d.dt_s), pft=str(getattr(d, "pft", "?")),
                    theta_sat=tsat, theta_r=tr,
                    theta_fc=float(land.theta_fc), theta_wp=float(land.theta_wp))
    # Lossless zlib: keeps the checked-in example outputs small without changing values.
    ds.to_netcdf(out_path, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="FLUXNET site ID, e.g. DE-Hai")
    ap.add_argument("--out", required=True, help="output NetCDF path")
    ap.add_argument("--driver-dir", required=True,
                    help="directory holding <SITE>_driver_v2[_gapfree|_etcorr].nc")
    ap.add_argument("--year-lo", type=int, default=0, help="first calendar year")
    ap.add_argument("--year-hi", type=int, default=9999, help="last calendar year")
    ap.add_argument("--stress-b0", action="store_true",
                    help="legacy: also stress the Ball-Berry cuticular intercept")
    args = ap.parse_args()
    p = run_evaluation(args.site, args.out, args.driver_dir,
                       args.year_lo, args.year_hi, stress_b0=args.stress_b0)
    ds = xr.open_dataset(p)
    print(f"saved {p}  n={ds.dims['time']}  site={ds.attrs['site']} pft={ds.attrs['pft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
