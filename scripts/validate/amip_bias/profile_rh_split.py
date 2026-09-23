#!/usr/bin/env python3
"""Regional T/q profile bias, and an EXACT split of the RH error into its
temperature and humidity parts.

The global-mean scorecard cannot tell a broad free-tropospheric cold bias from
an upper-tropospheric moist bias, and the two have different fixes. Both raise
relative humidity, and an RH-thresholding cloud scheme (sundqvist) turns either
into excess cloud cover — so attributing the cover error means attributing RH.

Two tables per run:

1. **Profile bias.** Area-weighted ``ta`` bias and ``hus`` ratio vs ERA5, level
   by level, per region.

2. **RH split.** With ``RH_swapT = w_model / w_sat(T_ERA5)``,

       RH_model - RH_ERA5 = [RH_model - RH_swapT] + [RH_swapT - RH_ERA5]
                                  dT part                  dq part

   an identity, not a regression: the first term is the RH excess the
   temperature bias alone creates (saturation falls ~7 %/K, so a cold column is
   more saturated at unchanged humidity), the second what the humidity bias
   alone creates. They sum exactly, per cell and after the area mean.

NO COVER IS RECONSTRUCTED. An earlier version of this probe ran the sundqvist
operator on monthly-mean RH to predict ``clt``; that is not a valid estimator
and it was retracted before any number from it was used. On ERA5's OWN monthly
means it returns 9 % cover where ERA5 observes 63 %, because the scheme is
strongly nonlinear near ``rh_crit`` and a monthly mean rarely exceeds it.
Attributing the COVER error needs sub-monthly (T, q); this probe attributes the
RH INPUT, which is exact at monthly resolution.

CONVENTIONS AND MASKS, stated not hidden:

* ``saturation_mixing_ratio`` (the model's own curve, the one
  ``compute_cloud_properties`` calls) returns a MIXING RATIO. The model's CMOR
  ``hus`` is its ``q_v`` tracer, already a mixing ratio; ERA5's ``hus`` is a
  true specific humidity and is converted. Skipping that biases the reference
  RH low by ~2 %, the same size as the signal at 850 hPa.
* CMOR ``plev19`` does not mask levels below the terrain — it clamps, so
  1000 hPa over Antarctica is a copy of the lowest model level. Masked against
  the MINIMUM-over-months surface pressure of BOTH sides (the interpolation
  clamps per sample, so the time mean is not conservative), and the masked
  fraction is printed next to every row. Unmasked, this reported a global
  1000 hPa cold bias of -3.9 K where the true value is -1.3 K.
* Levels above the model lid (p < 10000 Pa) are dropped: those plev19 rows were
  byte-copies of the top model level before the CMOR clamp fix.
* ERA5 is interpolated in log-p onto the model's plev axis and binned onto the
  model grid with the SAME area-weighted reduction ``regional_bias.py`` uses.

Usage:  profile_and_cover_attribution.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pathlib
import sys

import numpy as np
import xarray as xr

ROOT = os.environ.get(
    "LEGOESM_AMIP_RUNS", "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
ERA5 = "/work/bd1179/b309141/climateeval_input/reanalysis_ERA5/mon"
REF_MIN_YEAR = 1979
P_FLOOR = 10000.0  # Pa; drop plev rows above the model lid

_SPEC = importlib.util.spec_from_file_location(
    "rb", str(pathlib.Path(__file__).with_name("regional_bias.py")))
rb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rb)

PROFILE_REGIONS = ("ITCZ 10S-10N", "trades 10-30N", "trades 10-30S",
                   "Sc Peru", "NH midlat", "poles 60-90", "GLOBAL")


def _era5_on_model(var, months, plev, mlat, mlon):
    """ERA5 monthly climatology of ``months`` -> (plev, mlat, mlon)."""
    fs = sorted(glob.glob(f"{ERA5}/{var}/*.nc"))
    d = xr.open_dataset(fs[0])
    v = d[var].sel(time=slice(f"{REF_MIN_YEAR}-01-01", None))
    clim = v.groupby("time.month").mean("time").sel(month=months).mean("month").load()
    ep = np.asarray(clim["plev"], dtype=np.float64)
    arr = np.asarray(clim.transpose("plev", "lat", "lon"), dtype=np.float64)
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    binned = np.stack([rb.bin_to_model(arr[k], rlat, rlon, mlat, mlon,
                                       label=f"ERA5 {var}")
                       for k in range(ep.size)])
    # log-p interpolation onto the model plev axis (ERA5 plev descends).
    order = np.argsort(-ep)
    lep = np.log(ep[order])
    out = np.empty((plev.size, mlat.size, mlon.size))
    for j in range(mlat.size):
        for i in range(mlon.size):
            out[:, j, i] = np.interp(np.log(plev)[::-1], lep[::-1],
                                     binned[order, j, i][::-1])[::-1]
    return out


def _rh(T, w_v, plev):
    """Relative humidity from the MODEL's own saturation curve.

    Uses ``legoesm.thermo.saturation_mixing_ratio`` — the same function
    ``compute_cloud_properties`` calls — so the RH reported here is the
    quantity the cloud scheme actually thresholds, not a re-derived Magnus fit.

    ``w_v`` must be a MIXING RATIO (m_v / m_dry), matching what that function
    returns (``eps * e_sat / (p - e_sat)``).  The model's CMOR ``hus`` is
    written straight from its ``q_v`` tracer, which is already a mixing ratio;
    ERA5's ``hus`` is a true SPECIFIC humidity and must be converted by the
    caller (``_to_mixing_ratio``).  Dividing a specific humidity by a
    saturation mixing ratio biases RH low by a factor (1-q) — only ~2 % in the
    tropical boundary layer, but the temperature/humidity split below resolves
    differences of exactly that size.
    """
    import jax.numpy as jnp

    from legoesm.thermo import saturation_mixing_ratio

    p = jnp.broadcast_to(jnp.asarray(plev)[:, None, None], T.shape)
    w_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T), p))
    return w_v / np.maximum(w_sat, 1.0e-10)


def _to_mixing_ratio(q):
    """Specific humidity [kg/kg] -> mixing ratio m_v/m_dry."""
    return q / np.maximum(1.0 - q, 1.0e-10)


def _era5_surface_pressure(months):
    """ERA5 ``ps`` climatology of ``months`` as (arr, rlat, rlon), unbinned."""
    fs = sorted(glob.glob(f"{ERA5}/ps/*.nc"))
    d = xr.open_dataset(fs[0])
    v = d["ps"].sel(time=slice(f"{REF_MIN_YEAR}-01-01", None))
    clim = v.groupby("time.month").mean("time").sel(month=months).mean("month").load()
    return (np.asarray(clim.transpose("lat", "lon"), dtype=np.float64),
            np.asarray(clim["lat"], dtype=np.float64),
            np.asarray(clim["lon"], dtype=np.float64) % 360.0)


FIG = None


def _profile_figure(run, plev, below, Tm, Te, qm, qe, mlat, mlon):
    """q model/ERA5 ratio and T bias per region against pressure (lowest 6 levels
    emphasised: that is where the surface-layer/cloud-layer split lives)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for reg in PROFILE_REGIONS[:4]:
        box = rb.REGIONS[reg]
        qr = [rb.region_mean(qm[k], mlat, mlon, box, ~below[k])
              / rb.region_mean(qe[k], mlat, mlon, box, ~below[k]) for k in range(plev.size)]
        dT = [rb.region_mean(Tm[k] - Te[k], mlat, mlon, box, ~below[k]) for k in range(plev.size)]
        ax[0].plot(qr, plev / 100, "o-", label=reg)
        ax[1].plot(dT, plev / 100, "o-", label=reg)
    ax[0].axvline(1, color="k", lw=0.8); ax[1].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("q model / ERA5"); ax[1].set_xlabel("T model - ERA5 [K]")
    ax[0].set_ylabel("p [hPa]"); ax[0].set_ylim(1000, 300); ax[0].set_xlim(0.5, 3)
    ax[0].legend(); ax[0].grid(); ax[1].grid()
    fig.suptitle(f"{run}: humidity ratio and temperature bias vs ERA5")
    fig.tight_layout(); fig.savefig(FIG, dpi=110)
    print(f"figure {FIG}")


def main(runs):
    for run in runs:
        mt = rb._load_model(run, "ta")
        mq = rb._load_model(run, "hus")
        if mt is None or mq is None:
            print(f"{run}: no ta/hus CMOR output")
            continue
        months = rb._month_labels(mt)
        mlat = np.asarray(mt.lat, dtype=np.float64)
        mlon = np.asarray(mt.lon, dtype=np.float64) % 360.0
        plev_all = np.asarray(mt.plev, dtype=np.float64)
        keep = plev_all >= P_FLOOR
        plev = plev_all[keep]
        Tm = np.asarray(mt["ta"]).mean(axis=0)[keep]
        qm = np.asarray(mq["hus"]).mean(axis=0)[keep]      # model: mixing ratio
        Te = _era5_on_model("ta", months, plev, mlat, mlon)
        qe = _to_mixing_ratio(_era5_on_model("hus", months, plev, mlat, mlon))

        # BELOW-GROUND MASK.  CMOR plev19 does not mask levels under terrain --
        # _interp_to_plev19 clamps its weight, so 1000/925 hPa over Antarctica
        # and the Tibetan plateau are verbatim copies of the lowest model
        # level.  Left in, they are a plausible-looking number that is not a
        # measurement: they are what produced an apparent grid-mean RH > 1.0 in
        # the polar cap.  Mask any (level, cell) where the pressure exceeds the
        # surface pressure on EITHER side, and drop those cells from the area
        # weights rather than averaging them in.
        # MINIMUM over months, not the mean: the plev interpolation clamps per
        # SAMPLE, so a cell whose surface pressure dips below a level in any
        # single month contributes clamped values to that month's mean.  Using
        # the time-mean ps would let that survive.  The monthly minimum is the
        # most conservative estimate available from monthly output; closing it
        # exactly would need the mask applied inside the CMOR writer.
        ps_m = np.asarray(rb._load_model(run, "ps")["ps"]).min(axis=0)
        ps_e = np.min([rb.bin_to_model(*_era5_surface_pressure([m]), mlat, mlon,
                                       label="ERA5 ps")
                       for m in sorted(set(months))], axis=0)
        below = (plev[:, None, None] > np.minimum(ps_m, ps_e)[None, :, :])
        for f in (Tm, qm, Te, qe):
            f[below] = np.nan
        frac_masked = below.mean(axis=(1, 2))

        print(f"\n=== {run}: profile bias vs ERA5 "
              f"(months {sorted(set(months))}, {plev.size} levels) ===")
        print(f"{'p[hPa]':>7}" + "".join(f"{r.split()[0][:7]:>16}"
                                         for r in PROFILE_REGIONS))
        print(f"{'':>7}{'%bg':>5}" + "".join(f"{'dT   q/qe':>16}"
                                             for _ in PROFILE_REGIONS))
        for k, p in enumerate(plev):
            ok = ~below[k]
            cells = []
            for reg in PROFILE_REGIONS:
                box = rb.REGIONS[reg]
                rm = lambda f: rb.region_mean(f, mlat, mlon, box, ok)  # noqa: E731
                cells.append(f"{rm(Tm[k] - Te[k]):8.2f}"
                             f"{rm(qm[k]) / rm(qe[k]):8.2f}")
            print(f"{p / 100:7.0f}{100 * frac_masked[k]:5.0f}"
                  + "".join(f"{c:>16}" for c in cells))
        print("%bg = percent of cells masked as below ground at that level.")
        if FIG:
            _profile_figure(run, plev, below, Tm, Te, qm, qe, mlat, mlon)

        # --- RH decomposition ---------------------------------------------
        # EXACT and additive: RH_m - RH_e = (RH_m - RH_swapT) + (RH_swapT - RH_e)
        # with RH_swapT = q_model / q_sat(T_ERA5).  First term = what the
        # temperature bias alone does to RH (q_sat falls ~7 %/K, so a cold
        # column is more saturated at unchanged humidity); second = what the
        # humidity bias alone does.
        RH_m = _rh(Tm, qm, plev)
        RH_e = _rh(Te, qe, plev)
        RH_swapT = _rh(Te, qm, plev)

        print(f"\n=== {run}: RH bias split into temperature and humidity "
              "parts (scheme thresholds at rh_crit) ===")
        print(f"{'p[hPa]':>7}" + "".join(f"{r.split()[0][:7]:>26}"
                                         for r in PROFILE_REGIONS))
        print(f"{'':>7}" + "".join(f"{'RHmdl  RHera  dT_prt dq_prt':>26}"
                                   for _ in PROFILE_REGIONS))
        for k, p in enumerate(plev):
            ok = ~below[k]
            cells = []
            for reg in PROFILE_REGIONS:
                box = rb.REGIONS[reg]
                rm = lambda f: rb.region_mean(f[k], mlat, mlon, box, ok)  # noqa: E731
                m, e, s = rm(RH_m), rm(RH_e), rm(RH_swapT)
                cells.append(f"{m:7.2f}{e:7.2f}{m - s:7.2f}{s - e:7.2f}")
            print(f"{p / 100:7.0f}" + "".join(f"{c:>26}" for c in cells))
        print("\ndT_prt = RH(model T, model q) - RH(ERA5 T, model q): RH excess "
              "the cold bias alone creates.")
        print("dq_prt = RH(ERA5 T, model q) - RH(ERA5 T, ERA5 q): RH excess the "
              "humidity bias alone creates.  The two sum EXACTLY to the total.")

        mc = rb._load_model(run, "clt")
        if mc is not None:
            A = np.asarray(mc["clt"]).mean(axis=0)
            D = rb._ref_clim("clt", months, mlat, mlon)
            print(f"\n=== {run}: published clt [%] ===")
            print(f"{'region':<16}{'model':>8}{'ERA5':>8}{'diff':>8}")
            for reg in PROFILE_REGIONS:
                box = rb.REGIONS[reg]
                a = rb._region_mean(A, mlat, mlon, box)
                d = rb._region_mean(D, mlat, mlon, box)
                print(f"{reg:<16}{a:8.1f}{d:8.1f}{a - d:8.1f}")
        print("\nNOTE: cover is NOT reconstructed here.  Running the sundqvist "
              "operator on MONTHLY-MEAN RH is not a valid estimator of "
              "monthly-mean cover -- on ERA5's own monthly means it returned "
              "9 % where ERA5 observes 63 %, because the scheme is strongly "
              "nonlinear near rh_crit and monthly means rarely exceed it.  "
              "Attributing the COVER error needs sub-monthly (T, q); this "
              "table attributes the RH INPUT instead, which is exact.")


if __name__ == "__main__":
    _a = sys.argv[1:]
    if "--fig" in _a:
        FIG = _a[_a.index("--fig") + 1]
        _a = [x for i, x in enumerate(_a) if x != "--fig" and (i == 0 or _a[i - 1] != "--fig")]
    main(_a or ["ref1979"])
