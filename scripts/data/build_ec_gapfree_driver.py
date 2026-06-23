#!/usr/bin/env python
"""Build a *gap-free meteorological* EC-site driver for prognostic offline runs.

The DifferBESS ``<SITE>_driver_v2.nc`` keeps only **genuinely-observed** met
(its preprocessing masks every gap-filled sample to missing, because its goal
was *calibration*).  That is fine for the diagnostic mode (``jax.vmap`` isolates
each timestep), but the **prognostic** soil mode integrates the soil state
forward with ``lax.scan`` — a single missing/NaN forcing step would poison the
carried state for the rest of the run.  Prognostic runs therefore need a
continuous forcing series.

This producer reconstructs the continuous met by **reusing the FLUXNET2015
``_F_MDS`` gap-filled variables** that the calibration preprocessing discarded:

* For every substituted field the FULLSET ``_F_MDS`` column is **byte-identical**
  to ``driver_v2`` on every observed step (verified ``maxdiff == 0``) — it is the
  *same* series, only with the MDS gap-fill retained instead of masked out.  So
  the substitution changes nothing on observed steps; it only fills the holes.
* Any residual gap that survives even the MDS fill is closed here:
    - **short** gaps (``< LONG_GAP_HOURS``) → linear time interpolation;
    - **long** gaps (``>= LONG_GAP_HOURS``) → a multi-year **month-of-year x
      hour-of-day diurnal climatology** built from the observed series.  This is
      the sub-daily analogue of the DifferBESS HANTS gapfill: HANTS as configured
      there (``freq=5`` harmonics/yr) targets *daily* LAI and cannot resolve the
      24 h cycle on hourly met, and it explicitly re-NaNs leading/trailing gaps —
      so for a multi-day hourly gap the diurnal climatology is both the faithful
      reconstruction and actually gap-free.

CO2 is intentionally **not** substituted: ``driver_v2``'s ``CO2`` is already a
gap-free site+global blend (distinct from raw ``CO2_F_MDS``), so it is copied
through unchanged.  Everything non-meteorological (LAI, albedos, CI, SZA, site
params, observed fluxes) is copied through verbatim, so the existing
``legoesm.land.boundary_data.ec_site`` reader consumes the output unchanged.

Provenance is preserved: a per-field ``<VAR>_filled`` int8 flag (1 where the
value is *not* an original measured observation) and a global ``met_any_filled``
flag are written, so validation can still restrict to genuinely-observed steps.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/data/build_ec_gapfree_driver.py \
        --driver-nc <DifferBESS>/data/sitelevel/nc/US-MMS_driver_v2.nc \
        --fullset   <THERMAL_SITES>/AMF_US-MMS_FLUXNET_FULLSET_HR_1999-2020_3-5.csv \
        --out       <DifferBESS>/data/sitelevel/nc/US-MMS_driver_v2_gapfree.nc \
        --plot-dir  diagnostics/ec_gapfree
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from legoesm.land.boundary_data.ec_site import is_observed

# FLUXNET2015 missing-data sentinel.
_FLUXNET_MISSING = -9990.0          # values <= this are missing (-9999)
# Gap-length boundary [hours]: at/above this a gap is "long" and filled with the
# diurnal-seasonal climatology rather than linear interpolation.
LONG_GAP_HOURS = 24
# Minimum finite observed-overlap steps required to trust the value/unit check
# between the driver and the FULLSET column before substituting.
_MIN_OVERLAP = 100

# driver_v2 variable  <-  FULLSET _F_MDS column.  Units already match (verified
# maxdiff==0 on observed overlap), so no unit conversion is applied here; the
# ec_site reader performs the physical conversions downstream.
_SUBSTITUTIONS = {
    "TA": "TA_F_MDS",          # air temperature [deg C]
    "VPD": "VPD_F_MDS",        # vapour pressure deficit [hPa]
    "SW_IN": "SW_IN_F_MDS",    # incoming shortwave [W m-2]
    "LW_IN": "LW_IN_F_MDS",    # incoming longwave [W m-2]
    "PA": "PA_F",              # surface pressure [kPa]
    "TS": "TS_F_MDS_1",        # soil temperature (top) [deg C]
    "SWC": "SWC_F_MDS_1",      # soil water content (top) [%]
}
# Atmospheric forcing — used per-step in BOTH diagnostic and prognostic mode, so
# a filled value here means that step's forcing is synthetic (-> met_atm_filled).
_ATM_FORCING = ("TA", "VPD", "SW_IN", "LW_IN", "PA")
# Soil fields — used per-step (prescribed) in DIAGNOSTIC mode, but only for the
# INITIAL state in prognostic mode.  Flagged separately (-> soil_filled) so the
# consumer can treat them mode-appropriately.
_SOIL_FIELDS = ("TS", "SWC")


def _clean_fluxnet(series: np.ndarray) -> np.ndarray:
    """FLUXNET sentinel (-9999) -> NaN, as float64."""
    a = np.asarray(series, dtype=np.float64)
    return np.where(a <= _FLUXNET_MISSING, np.nan, a)


def _gap_runs(is_missing: np.ndarray):
    """Yield (start, stop) half-open index ranges of consecutive missing runs."""
    n = is_missing.shape[0]
    i = 0
    while i < n:
        if is_missing[i]:
            j = i
            while j < n and is_missing[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def fill_gaps(values: np.ndarray, months: np.ndarray, slots: np.ndarray,
              n_slots: int, dt_seconds: float,
              long_gap_hours: int = LONG_GAP_HOURS):
    """Fill NaNs: long/boundary gaps by month x slot-of-day climatology, interior
    short gaps by linear interpolation.

    Returns (filled_values, n_long_filled, n_short_filled).  ``slots`` is the
    timestep-of-day index (0 .. n_slots-1), so a half-hourly driver keeps :00 and
    :30 as DISTINCT climatology bins (binning by integer hour would average them
    and flatten the sub-hourly diurnal phase around sunrise/sunset).

    The long/short boundary is timestep-aware: ``long_gap_hours`` is converted to
    a SAMPLE count using ``dt_seconds``.  Gaps touching the record start/end are
    filled by climatology regardless of length, because two-sided linear
    interpolation has no anchor there (back/forward-filling from the only
    available side is neither interpolation nor a defensible reconstruction).
    """
    out = values.astype(np.float64).copy()
    observed = np.isfinite(values)
    if observed.all():
        return out, 0, 0

    long_gap_samples = max(1, int(round(long_gap_hours * 3600.0 / dt_seconds)))
    n = values.shape[0]

    # Observation-based climatology hierarchy (all from OBSERVED samples; never
    # positional, so a boundary gap can never be filled from a future value):
    #   1. month-of-year x slot-of-day  (diurnal + seasonal)
    #   2. slot-of-day across all months (diurnal only)   -- empty (m,s) cell
    #   3. global observed mean                            -- empty slot cell
    clim_ms = np.full((13, n_slots), np.nan)
    clim_s = np.full(n_slots, np.nan)
    for s in range(n_slots):
        sel_s = observed & (slots == s)
        if sel_s.any():
            clim_s[s] = np.nanmean(values[sel_s])
        for m in range(1, 13):
            sel = sel_s & (months == m)
            if sel.any():
                clim_ms[m, s] = np.nanmean(values[sel])
    clim_global = float(np.nanmean(values[observed]))  # observed.any() guaranteed

    def _clim(m, s):
        v = clim_ms[m, s]
        if not np.isfinite(v):
            v = clim_s[s]
        if not np.isfinite(v):
            v = clim_global
        return v

    n_long = n_short = 0
    for start, stop in _gap_runs(~observed):
        run = stop - start
        touches_boundary = (start == 0) or (stop == n)
        if run >= long_gap_samples or touches_boundary:
            for k in range(start, stop):
                out[k] = _clim(months[k], slots[k])
            n_long += run
        else:
            n_short += run  # interior short gap -> linear pass below

    # Interior short gaps now have finite values on BOTH sides, so plain linear
    # interpolation fills them with no edge extrapolation.
    out = pd.Series(out).interpolate(method="linear").to_numpy()
    if not np.isfinite(out).all():                # defensive: must not happen
        raise ValueError("fill_gaps left residual NaN (no observed data?)")
    return out, n_long, n_short


def build(driver_nc: str, fullset_csv: str, out_nc: str) -> xr.Dataset:
    ds = xr.open_dataset(driver_nc).load()
    n = ds.sizes["time"]
    times = pd.DatetimeIndex(ds["time"].values)
    months = times.month.to_numpy()

    cols = (["TIMESTAMP_START", "TIMESTAMP_END"] + list(_SUBSTITUTIONS.values()))
    fl = pd.read_csv(fullset_csv, usecols=cols)
    if len(fl) != n:
        raise ValueError(
            f"row mismatch: FULLSET has {len(fl)} rows, driver has {n} steps.")

    # Require EXACT 1:1 timestamp alignment (not just equal row counts): the
    # FULLSET interval CENTRE (START + (END-START)/2) must equal the driver time
    # coordinate step-for-step.  This catches a shifted / reordered / wrong-period
    # file that would otherwise be assigned positionally and corrupt chronology.
    def _ts(col):
        return pd.to_datetime(fl[col].astype("int64").astype(str),
                              format="%Y%m%d%H%M")
    centre = (_ts("TIMESTAMP_START")
              + (_ts("TIMESTAMP_END") - _ts("TIMESTAMP_START")) / 2)
    drv = times.to_numpy().astype("datetime64[m]")
    if not np.array_equal(centre.to_numpy().astype("datetime64[m]"), drv):
        bad = int(np.argmax(centre.to_numpy().astype("datetime64[m]") != drv))
        raise ValueError(
            "FULLSET timestamps do not align 1:1 with the driver time axis "
            f"(first mismatch at index {bad}: FULLSET centre "
            f"{centre.iloc[bad]} vs driver {times[bad]}).")

    # Uniform driver timestep (needed for timestep-aware gap classification).
    dts = (np.diff(times.to_numpy()).astype("timedelta64[s]").astype(float))
    if dts.size and not np.allclose(dts, dts[0]):
        raise ValueError("driver time axis is not uniformly spaced")
    dt_seconds = float(dts[0]) if dts.size else 3600.0

    # Timestep-of-day slot index (keeps :00/:30 distinct for sub-hourly data).
    n_slots = max(1, int(round(86400.0 / dt_seconds)))
    secs_of_day = times.hour.to_numpy() * 3600 + times.minute.to_numpy() * 60
    slots = (np.floor(secs_of_day / dt_seconds).astype(int)) % n_slots

    print(f"driver={os.path.basename(driver_nc)}  steps={n}  dt={dt_seconds:.0f}s  "
          f"n_slots={n_slots}  {times[0]} -> {times[-1]}")
    report = []
    atm_filled = np.zeros(n, dtype=bool)    # TA/VPD/SW_IN/LW_IN/PA (per-step forcing)
    soil_filled = np.zeros(n, dtype=bool)   # TS/SWC (prescribed diag / prognostic IC)
    for dv, fv in _SUBSTITUTIONS.items():
        if dv not in ds.data_vars:
            raise KeyError(f"{dv!r} not in driver (expected to substitute)")
        orig = np.asarray(ds[dv].values, dtype=np.float64).ravel()
        new = _clean_fluxnet(fl[fv].to_numpy())

        # "Observed" is defined the SAME way the reader decides validity: finite
        # AND within the physical range (a finite out-of-range sentinel/extreme is
        # NOT an observation).  This keeps provenance flags consistent with how
        # the reader will actually treat the written values.
        orig_obs = is_observed(dv, orig)

        # Sanity: identical on the steps both have (same series, MDS retained).
        # Require a MEANINGFUL overlap — with timestamps already aligned 1:1, a
        # near-empty overlap means the value/unit check is vacuous, so refuse
        # rather than silently substitute (no-overlap must NOT pass as maxdiff=0).
        both = orig_obs & np.isfinite(new)
        n_overlap = int(both.sum())
        if n_overlap < _MIN_OVERLAP:
            raise ValueError(
                f"{dv}<-{fv}: only {n_overlap} finite overlap steps "
                f"(< {_MIN_OVERLAP}); cannot verify units/alignment — refusing.")
        maxdiff = float(np.abs(orig[both] - new[both]).max())
        if maxdiff > 1e-6:
            raise ValueError(
                f"{dv}<-{fv}: maxdiff={maxdiff:.4g} on observed overlap; "
                "units/alignment do not match — refusing to substitute.")

        filled_values, n_long, n_short = fill_gaps(
            new, months, slots, n_slots, dt_seconds)
        # NEVER overwrite a genuine driver observation: keep orig where OBSERVED
        # (finite AND in-range), use the (MDS/climatology) fill elsewhere.  This
        # makes the filled flag exact (not-observed <=> value came from a fill)
        # and prevents a finite-but-invalid orig from being kept+mislabelled.
        result = np.where(orig_obs, orig, filled_values)
        if not np.isfinite(result).all():
            raise ValueError(f"{dv}: residual NaN after fill")

        # filled flag: 1 where the value is NOT a genuine measured observation.
        filled_flag = (~orig_obs).astype(np.int8)
        ds[dv] = (ds[dv].dims, result.reshape(ds[dv].shape))
        ds[f"{dv}_filled"] = (("time",), filled_flag)
        ds[f"{dv}_filled"].attrs = dict(
            long_name=f"{dv} not originally measured (MDS or climatology fill)",
            flag_values="0=measured 1=filled")
        if dv in _ATM_FORCING:
            atm_filled |= filled_flag.astype(bool)
        elif dv in _SOIL_FIELDS:
            soil_filled |= filled_flag.astype(bool)
        report.append((dv, fv, int(filled_flag.sum()), n_long, n_short, maxdiff))

    # Atmospheric-forcing provenance (per-step forcing in both modes).
    ds["met_atm_filled"] = (("time",), atm_filled.astype(np.int8))
    ds["met_atm_filled"].attrs = dict(
        long_name="any atmospheric forcing (TA/VPD/SW_IN/LW_IN/PA) filled",
        flag_values="0=all measured 1=>=1 filled")
    # Soil provenance (per-step in diagnostic; initial-state only in prognostic).
    ds["soil_filled"] = (("time",), soil_filled.astype(np.int8))
    ds["soil_filled"].attrs = dict(
        long_name="any soil field (TS/SWC) filled",
        flag_values="0=all measured 1=>=1 filled")
    # Back-compat aggregate (atmospheric OR soil).
    ds["met_any_filled"] = (("time",), (atm_filled | soil_filled).astype(np.int8))
    ds["met_any_filled"].attrs = dict(
        long_name="any substituted field (atm or soil) filled",
        flag_values="0=all measured 1=>=1 filled")

    ds.attrs["gapfree_source"] = os.path.basename(fullset_csv)
    ds.attrs["gapfree_method"] = (
        "FLUXNET _F_MDS substitution; long gaps (>=%dh) by month-x-slot "
        "diurnal climatology, interior short gaps linear" % LONG_GAP_HOURS)

    print("\n=== substitution report (var <- fullset: filled / long / short / maxdiff) ===")
    for dv, fv, nf, nl, ns, md in report:
        print(f"  {dv:6s}<-{fv:14s} filled={nf:7d} ({100*nf/n:5.2f}%)  "
              f"long={nl:5d}  short={ns:5d}  maxdiff={md:.1e}")
    print(f"  met_atm_filled: {int(atm_filled.sum())} ({100*atm_filled.mean():.2f}%)"
          f"   soil_filled: {int(soil_filled.sum())} ({100*soil_filled.mean():.2f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(out_nc)), exist_ok=True)
    ds.to_netcdf(out_nc)
    print(f"\n  -> {out_nc}")
    return ds


def plot(ds: xr.Dataset, plot_dir: str, site_tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    times = pd.DatetimeIndex(ds["time"].values)
    fields = ["TA", "VPD", "SW_IN", "LW_IN", "PA", "TS", "SWC", "P", "WS", "CO2"]
    fields = [f for f in fields if f in ds.data_vars]

    # ---- overview: daily means of every met field, filled span shaded ----
    s = pd.Series(0, index=times)
    fig, axes = plt.subplots(len(fields), 1, figsize=(13, 2.0 * len(fields)),
                             sharex=True)
    for ax, f in zip(np.atleast_1d(axes), fields):
        v = pd.Series(np.asarray(ds[f].values).ravel(), index=times)
        daily = v.resample("1D").mean()
        ax.plot(daily.index, daily.values, lw=0.5, color="tab:blue")
        if f"{f}_filled" in ds.data_vars:
            fr = pd.Series(np.asarray(ds[f"{f}_filled"].values).ravel(),
                           index=times).resample("1D").max()
            for d in fr.index[fr.values > 0]:
                ax.axvspan(d, d + pd.Timedelta("1D"), color="red", alpha=0.15, lw=0)
        u = ds[f].attrs.get("unit", ds[f].attrs.get("units", ""))
        ax.set_ylabel(f"{f}\n[{u}]", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(f"{site_tag}: gap-free met (daily mean; red = filled)",
                      fontsize=10)
    fig.tight_layout()
    p = os.path.join(plot_dir, f"{site_tag}_met_overview.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  plot -> {p}")

    # ---- zoom: first 30 days (hourly) to inspect the long climatology fill ----
    zoom = times < (times[0] + pd.Timedelta("30D"))
    zfields = ["TA", "VPD", "SW_IN", "LW_IN", "PA"]
    zfields = [f for f in zfields if f in ds.data_vars]
    fig, axes = plt.subplots(len(zfields), 1, figsize=(13, 2.0 * len(zfields)),
                             sharex=True)
    for ax, f in zip(np.atleast_1d(axes), zfields):
        v = np.asarray(ds[f].values).ravel()
        fr = np.asarray(ds[f"{f}_filled"].values).ravel().astype(bool)
        ax.plot(times[zoom], v[zoom], lw=0.7, color="tab:blue", label="series")
        mfill = zoom & fr
        ax.plot(times[mfill], v[mfill], ".", ms=2.5, color="red", label="filled")
        u = ds[f].attrs.get("unit", ds[f].attrs.get("units", ""))
        ax.set_ylabel(f"{f}\n[{u}]", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"{site_tag}: first 30 days (hourly; red = climatology fill)",
                      fontsize=10)
    fig.tight_layout()
    p = os.path.join(plot_dir, f"{site_tag}_met_zoom_first30d.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  plot -> {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver-nc", required=True, help="input <SITE>_driver_v2.nc")
    ap.add_argument("--fullset", required=True,
                    help="raw FLUXNET2015/AMF FULLSET csv for the site")
    ap.add_argument("--out", required=True, help="output gap-free driver nc")
    ap.add_argument("--plot-dir", default="diagnostics/ec_gapfree")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ds = build(args.driver_nc, args.fullset, args.out)
    if not args.no_plot:
        tag = os.path.basename(args.out).replace(".nc", "")
        plot(ds, args.plot_dir, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
