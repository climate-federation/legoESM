#!/usr/bin/env python3
"""Boundary-layer theta_e vs free-troposphere saturation theta_e, model and ERA5.

A free troposphere colder than ERA5 over prescribed SST either inherits a
low boundary-layer moist entropy (cool/dry boundary layer) or sits further
below the moist adiabat than ERA5's (convection/entrainment holds it cold).
Per month and region this prints, for model and ERA5 on the model's CMOR grid:

  theta_e at 925 hPa        (boundary-layer parcel)
  theta_e* at 500 hPa       (saturation value, i.e. the free-troposphere temperature)
  gap = theta_e*500 - theta_e925

theta_e uses the proxy theta * exp(L_v q / (c_pd T)) (q specific humidity), as
legoesm.atmosphere.dynamics.crm.rce_diagnostics.pseudo_equivalent_potential_temperature
documents; the same formula on both sides, so the model-minus-ERA5 DIFFERENCES are
the result, not the absolute values.  Saturation from legoesm.thermo.

Usage: theta_e_gap.py ERA5_TA ERA5_HUS MODEL_TA MODEL_HUS MODEL_SFTLF
"""
import sys

import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from t2m_window_metrics import bin_to_model


def theta_e(T, q, p):
    return T * (constants.p_ref / p) ** constants.kappa * np.exp(constants.L_v * q / (constants.c_pd * T))


def main(e5t, e5q, mt, mq, lff):
    ta, hus = xr.open_dataset(mt)["ta"], xr.open_dataset(mq)["hus"]
    lf = xr.open_dataset(lff)["sftlf"].squeeze(drop=True).values
    lat, lon = ta["lat"].values, ta["lon"].values
    E5T, E5Q = xr.open_dataset(e5t)["ta"], xr.open_dataset(e5q)["hus"]
    LA = lat[:, None] * np.ones((1, lon.size))
    ocean = lf < 50.0
    regions = {"tropics ocean": ocean & (np.abs(LA) <= 20), "45-70N ocean": ocean & (LA >= 45) & (LA <= 70)}
    w = np.cos(np.deg2rad(LA))

    def e5_on_model(da, t, p):
        x = da.sel(time=str(t)[:7]).squeeze("time", drop=True).sel(plev=p, method="nearest")
        if abs(float(x.plev) - p) > 1:
            raise SystemExit(f"ERA5 level {p} missing")
        v = x.transpose("lat", "lon").values
        ok = np.isfinite(v)
        val = bin_to_model(np.where(ok, v, 0.0), x["lat"].values, x["lon"].values, lat, lon)
        cov = bin_to_model(ok.astype(float), x["lat"].values, x["lon"].values, lat, lon)
        return np.where(cov > 0.999, val, np.nan)

    print("month  region          | thE925 model ERA5 diff | thE*500 model ERA5 diff | gap model ERA5")
    for t in ta["time"].values:
        f = {}
        for src in ("m", "e"):
            T9 = ta.sel(time=t).sel(plev=92500.0).values if src == "m" else e5_on_model(E5T, t, 92500.0)
            q9 = hus.sel(time=t).sel(plev=92500.0).values if src == "m" else e5_on_model(E5Q, t, 92500.0)
            T5 = ta.sel(time=t).sel(plev=50000.0).values if src == "m" else e5_on_model(E5T, t, 50000.0)
            q5s = np.asarray(saturation_specific_humidity(T5, 50000.0), dtype=np.float64)
            f[src] = (theta_e(T9, q9, 92500.0), theta_e(T5, q5s, 50000.0))
        for name, msk in regions.items():
            g = msk & np.all([np.isfinite(a) for s in f.values() for a in s], axis=0)
            mean = lambda x: float((x[g] * w[g]).sum() / w[g].sum())
            b_m, b_e = mean(f["m"][0]), mean(f["e"][0])
            s_m, s_e = mean(f["m"][1]), mean(f["e"][1])
            print(f"{str(t)[:7]}  {name:14s}  | {b_m:6.1f} {b_e:6.1f} {b_m - b_e:+5.1f} | "
                  f"{s_m:6.1f} {s_e:6.1f} {s_m - s_e:+5.1f} | {s_m - b_m:+5.1f} {s_e - b_e:+5.1f}  n{int(g.sum())}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
