#!/usr/bin/env python3
"""Where the ocean evaporation deficit lives: wind, humidity gradient, or coefficient.

Bulk evaporation is LH = rho * L_v * C_E * |U| * (q_sfc - q_air), so the ratio
of the model's flux to a reference's factors into a wind ratio, a humidity
difference ratio, and a residual that is the transfer coefficient (plus
density, plus everything the factorisation hides).  Codex named this as the
deciding measurement: does the air state explain the deficit, and which
variable?

Reference is MERRA2, which publishes the flux, the 10 m wind, the 2 m humidity
and the skin temperature, so every factor comes from ONE product.  Model side
is the CMOR 1000 hPa level (the lowest full level, ~135 m, where it is above
ground) and the run's own prescribed SST with the sea-water factor.

Heights are matched on the model side (review finding, codex + GLM): wind to
10 m with z0 = 2e-4 m, humidity difference to 2 m with z0q = 1e-4 m, with
Businger-Dyer profiles for a neutral and a typical unstable (L = -150 m)
surface layer; both are printed as "neutral/unstable" since the true
stability is not diagnosed here.  The 1000 hPa level is clamped to the
lowest full level wherever it is below it (the CMOR interpolator clips), so
no surface-pressure mask is applied: every ocean column carries the ~135 m
state.  The residual of the factorisation is NOT printed as a coefficient:
it absorbs covariance, gustiness and MERRA2's own internal-state flux and
can neither convict nor exonerate the exchange coefficient.

Usage: evap_deficit_decomposition.py <run> [fig.png]
The optional figure shows the ocean surface-pressure difference to MERRA2
(map) and the ocean zonal means, since the ps printout is what limits the
kept sample.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from surface_humidity_deficit import _prescribed_sst  # noqa: E402

MERRA = "/work/bd1179/b309141/climateeval_input/reanalysis_MERRA2/mon"
Z_MODEL_M, Z0_M, Z0Q_M, L_UNSTABLE_M = 135.0, 2e-4, 1e-4, -150.0


def _psi(z, L):
    """Businger-Dyer unstable stability functions (psi_m, psi_h); 0 when neutral."""
    if L is None:
        return 0.0, 0.0
    x = (1.0 - 16.0 * z / L) ** 0.25
    psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
    return psi_m, 2 * np.log((1 + x * x) / 2)


def height_factors(L=None):
    """(wind 135 m -> 10 m, dq 135 m -> 2 m) profile factors."""
    w = (np.log(10.0 / Z0_M) - _psi(10.0, L)[0]) / (np.log(Z_MODEL_M / Z0_M) - _psi(Z_MODEL_M, L)[0])
    d = (np.log(2.0 / Z0Q_M) - _psi(2.0, L)[1]) / (np.log(Z_MODEL_M / Z0Q_M) - _psi(Z_MODEL_M, L)[1])
    return w, d


BANDS = {"tropical ocean 20S-20N": (-20, 20, 0, 360),
         "trades 10-30N": (10, 30, 0, 360),
         "trades 10-30S": (-30, -10, 0, 360),
         "global": (-90, 90, 0, 360)}


def merra(var, months, mlat, mlon):
    """MERRA2 climatology on the model grid without dask: one file per year,
    per-file monthly means accumulated with equal year weights."""
    fs = sorted(glob.glob(f"{MERRA}/{var}/*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: no MERRA2 {var}")
    acc, n = None, 0
    for f in fs:
        v = xr.open_dataset(f)[var]
        if int(str(np.asarray(v.time)[0])[:4]) < rb.REF_MIN_YEAR:
            continue
        sel = v.groupby("time.month").mean("time")
        have = set(np.asarray(sel["month"]).tolist())
        got = [m for m in months if m in have]
        if not got:
            continue
        a = np.asarray(sel.sel(month=got).mean("month"), dtype=np.float64)
        acc = a if acc is None else acc + a
        n += 1
    if n == 0:
        raise SystemExit(f"FATAL: no MERRA2 {var} years >= {rb.REF_MIN_YEAR}")
    d0 = xr.open_dataset(fs[0])
    rlat = np.asarray(d0["lat"], dtype=np.float64)
    rlon = np.asarray(d0["lon"], dtype=np.float64) % 360.0
    arr = acc / n
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return np.asarray(rb.bin_to_model(arr, rlat, rlon, mlat, mlon, label=var))


def _ps_figure(run, ps, ps_o, sea, lat, lon, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.where(sea, (ps - ps_o) / 100.0, np.nan)
    w = np.cos(np.deg2rad(lat))[:, None] * sea
    zm = lambda f: np.nansum(np.where(sea, f, 0.0) * w, 1) / np.maximum(w.sum(1), 1e-12)  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].pcolormesh(lon, lat, d, cmap="RdBu_r", vmin=-12, vmax=12, shading="auto")
    fig.colorbar(im, ax=ax[0], label="ps model - MERRA2 [hPa]")
    ax[0].set_title(f"{run}: ocean surface pressure minus MERRA2")
    ax[1].plot(zm(ps) / 100.0, lat, label=run)
    ax[1].plot(zm(ps_o) / 100.0, lat, "k--", label="MERRA2")
    ax[1].set_xlabel("ocean zonal-mean ps [hPa]"); ax[1].set_ylabel("lat"); ax[1].legend(); ax[1].grid()
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"figure {out}")


def main(run, fig=None):
    from legoesm import constants
    from legoesm.thermo import saturation_specific_humidity

    du, dv = rb._load_model(run, "ua"), rb._load_model(run, "va")
    dq = rb._load_model(run, "hus")
    lat, lon = np.asarray(du.lat), np.asarray(du.lon)
    plev = np.asarray(du["plev"], dtype=np.float64)
    k = int(np.argmin(np.abs(plev - 100000.0)))
    months = rb._month_labels(du)
    U_m = np.hypot(np.asarray(du["ua"]).mean(0)[k], np.asarray(dv["va"]).mean(0)[k])
    q_m = np.asarray(dq["hus"]).mean(0)[k]
    ps = np.asarray(rb._load_model(run, "ps")["ps"]).mean(0)
    sst = _prescribed_sst(run, months, lat, lon)
    qs_m = np.asarray(saturation_specific_humidity(sst, ps)) * constants.q_sat_saline_fraction
    lh_m = np.asarray(rb._load_model(run, "hfls")["hfls"]).mean(0)

    U_o = np.hypot(merra("uas", months, lat, lon), merra("vas", months, lat, lon))
    q_o = merra("huss", months, lat, lon)
    ts_o = merra("ts", months, lat, lon)
    ps_o = merra("ps", months, lat, lon)
    qs_o = np.asarray(saturation_specific_humidity(ts_o, ps_o)) * constants.q_sat_saline_fraction
    lh_o = merra("hfls", months, lat, lon)

    fs = sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: {run} publishes no sftlf")
    d = xr.open_dataset(fs[0])
    fl = np.asarray(rb.bin_to_model(np.asarray(d["sftlf"], dtype=np.float64) / 100.0,
                                    np.asarray(d.lat), np.asarray(d.lon) % 360.0,
                                    lat, lon, label="sftlf"))
    ocean = (fl < 0.01) & np.isfinite(sst) & np.isfinite(lh_o)

    dq_m, dq_o = qs_m - q_m, qs_o - q_o
    t = np.asarray(du.time)
    sea = (fl < 0.01) & np.isfinite(sst)
    if fig:
        _ps_figure(run, ps, ps_o, sea, lat, lon, fig)
    (wn, dn), (wu, du_) = height_factors(None), height_factors(L_UNSTABLE_M)
    print(f"{run}: model level plev[{k}] = {plev[k]:.0f} Pa (clamped to the lowest full "
          f"level where below it); window {str(t[0])[:10]} to {str(t[-1])[:10]}, months "
          f"{months}; ocean mask sftlf<1%: {100 * ocean.mean():.1f}% of columns; "
          f"height factors neutral/unstable wind {wn:.2f}/{wu:.2f}, dq {dn:.2f}/{du_:.2f}")
    print(f"{'band':<24}{'LH':>6}{'U raw':>7}{'U 10m n/u':>11}{'dq raw':>8}{'dq 2m n/u':>11}"
          f"{'q_air m':>8}{'q_air M2':>9}{'q_sfc m':>8}{'q_sfc M2':>9}")
    for name, box in BANDS.items():
        r = lambda f: rb.region_mean(f, lat, lon, box, valid=ocean)  # noqa: E731
        lh, u, dqr = r(lh_m) / r(lh_o), r(U_m) / r(U_o), r(dq_m) / r(dq_o)
        print(f"{name:<24}{lh:6.2f}{u:7.2f}{u * wn:6.2f}/{u * wu:4.2f}{dqr:8.2f}"
              f"{dqr * dn:6.2f}/{dqr * du_:4.2f}{1e3 * r(q_m):8.2f}{1e3 * r(q_o):9.2f}"
              f"{1e3 * r(qs_m):8.2f}{1e3 * r(qs_o):9.2f}")
    allm = np.isfinite(ps_o)
    print(f"global all-surface mean ps on the shared grid: model "
          f"{rb.region_mean(ps, lat, lon, BANDS['global'], valid=allm) / 100:.1f} hPa, "
          f"MERRA2 {rb.region_mean(ps_o, lat, lon, BANDS['global'], valid=allm) / 100:.1f} hPa")
    print("ratios = model / MERRA2 of cos-lat band means; q in g/kg; q_sfc = 0.98 q_sat(SST, ps).")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
