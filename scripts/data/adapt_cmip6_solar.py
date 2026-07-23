#!/usr/bin/env python
"""Adapt a raw input4MIPs solar file into the deck's TSI + 14-band SSI schema.

``download_cmip6_forcing.py --channels solar`` fetches the CMIP6 SOLARIS-HEPPA
solar forcing, whose spectral solar irradiance ``ssi`` is stored at NATIVE
(fine) wavelength resolution.  The AMIP loader
(``forcing/external.py::_load_time_gpt`` via ``SolarConfig.source =
"spectral_file"``) instead wants ``TSI`` (time,) [W/m2] plus ``SSI_frac``
(time, numwl=14) — the FRACTION of TSI in each of the 14 RRTMG-SW shortwave
bands (summing to 1), on a daily ``days since 1850-01-01`` (noleap) axis — the
schema the synthetic ``generate_amip_forcing.make_solar`` emits.

This bins the native ``ssi`` into the 14 bands and normalises to fractions.  The
band boundaries are read from the MODEL's OWN shortwave gas-optics table
(``rrtmgp-gas-sw-g112.nc`` ``bnd_limits_wavenumber``) — NOT a textbook copy — and
put in RRTMG-SW order (the 820-2680 cm-1 overlap band LAST), which is exactly
the order ``_cmip_sw_band_order_to_rrtmgp`` expects on ingest.

The native ``ssi`` is a wavelength-BIN MEAN (``W m-2 nm-1``, ``cell_methods:
wlen: mean``, with ``wlen_bnds``) — NOT point samples — so a band's flux is the
histogram integral ``sum(ssi_bin * overlap(bin, band))`` (exact for bin-mean
data, including variable-width bins and partial band overlaps), and
``SSI_frac`` is each band flux over their sum.  The GREGORIAN calendar of the
raw files is mapped onto the loader's noleap ``days since 1850-01-01`` axis by
calendar date, dropping Feb 29.

Usage::

    python scripts/data/adapt_cmip6_solar.py \\
        --in data/cmip6_forcing/solar/solarforcing-ref-day_*.nc \\
        --out data/cmip6_forcing/solar_cmip6.nc

Consume it with ``run_amip_cmip6_deck.py --solar-source spectral_file
--solar-file <out> --solar-tsi-var TSI --solar-spectral-var SSI_frac
--solar-spectral-band-order rrtmg_sw``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_N_SW_BANDS = 14


def rrtmgsw_band_edges_nm(sw_gas_file: str | None = None) -> np.ndarray:
    """Return the 14 SW band wavelength intervals ``[lo_nm, hi_nm]`` in RRTMG-SW
    (CMIP) band order — the 820-2680 cm-1 overlap band LAST.

    Read from the shipped RRTMGP-SW gas-optics table's ``bnd_limits_wavenumber``
    (the model's own definition), so the binning matches the radiation solver
    exactly.  RRTMGP lists the overlap band FIRST; the loader converts CMIP ->
    RRTMGP with ``np.roll(+1)``, so RRTMG-SW order is ``np.roll(rrtmgp, -1)``.
    A wavenumber band ``[lo_wn, hi_wn]`` maps to ``[1e7/hi_wn, 1e7/lo_wn]`` nm.
    """
    import xarray as xr

    if sw_gas_file is None:
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            DEFAULT_SW_GAS,
        )
        sw_gas_file = DEFAULT_SW_GAS
    with xr.open_dataset(sw_gas_file) as ds:
        wn = np.asarray(ds["bnd_limits_wavenumber"].values, dtype=np.float64)
    if wn.shape != (_N_SW_BANDS, 2):
        raise ValueError(f"expected (14, 2) band limits, got {wn.shape}")
    wn = np.roll(wn, -1, axis=0)                       # RRTMGP -> RRTMG-SW order
    lo_nm = 1.0e7 / wn[:, 1]                            # high wavenumber -> short nm
    hi_nm = 1.0e7 / wn[:, 0]
    return np.stack([lo_nm, hi_nm], axis=1)


def ssi_to_band_fractions(bin_lo: np.ndarray, bin_hi: np.ndarray,
                          ssi: np.ndarray, edges_nm: np.ndarray) -> np.ndarray:
    """Bin spectral ``ssi`` (..., W) into the 14 SW bands and normalise.

    ``ssi[..., j]`` is the BIN-MEAN irradiance (``W m-2 nm-1``) over wavelength
    bin ``[bin_lo[j], bin_hi[j]]`` — the SOLARIS-HEPPA convention
    (``cell_methods: wlen: mean``, with ``wlen_bnds``), NOT a point sample.  The
    flux a bin contributes to a band is therefore ``ssi * overlap_width`` (a
    piecewise-constant / histogram integral), EXACT for bin-mean data and for
    variable-width bins and partial band overlaps — a trapezoid across bin
    centres would be wrong at both.  Returns ``SSI_frac`` (..., 14) summing to 1.
    """
    lo = np.asarray(bin_lo, dtype=np.float64)
    hi = np.asarray(bin_hi, dtype=np.float64)
    ssi = np.asarray(ssi, dtype=np.float64)
    band_lo = edges_nm[:, 0][:, None]                  # (14, 1)
    band_hi = edges_nm[:, 1][:, None]
    # overlap width of every (band, bin) pair, clipped at 0 -> (14, nbins).
    overlap = np.clip(np.minimum(band_hi, hi[None, :])
                      - np.maximum(band_lo, lo[None, :]), 0.0, None)
    flux = ssi @ overlap.T                             # (..., 14)
    total = np.sum(flux, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("integrated band flux is non-positive; check the ssi "
                         "units / wavelength coverage vs the SW bands.")
    return flux / total


def _bin_bounds_nm(ds, wl_name: str, wl_dim: str) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength bin bounds ``(lo, hi)`` [nm], ascending per bin.

    Prefers the file's explicit bounds (``<wl>_bnds`` / ``wlen_bnds``), then the
    bin sizes (``wlenbinsize`` / ``<wl>_binsize``, centred), and finally derives
    bounds from the centres (midpoints, edge-extrapolated) as a last resort.
    """
    centres = np.asarray(ds[wl_name].values, dtype=np.float64)
    bnds_name = next((v for v in (f"{wl_name}_bnds", "wlen_bnds", f"{wl_dim}_bnds",
                                  f"{wl_name}_bounds")
                      if v in ds.variables), None)
    if bnds_name is not None:
        b = np.asarray(ds[bnds_name].values, dtype=np.float64).reshape(-1, 2)
        return np.minimum(b[:, 0], b[:, 1]), np.maximum(b[:, 0], b[:, 1])
    size_name = next((v for v in ("wlenbinsize", f"{wl_name}_binsize",
                                  f"{wl_dim}_binsize")
                      if v in ds.variables), None)
    if size_name is not None:
        half = 0.5 * np.abs(np.asarray(ds[size_name].values, dtype=np.float64))
        return centres - half, centres + half
    # derive from centres: midpoints between neighbours, edges reflected.
    order = np.argsort(centres)
    c = centres[order]
    mid = 0.5 * (c[1:] + c[:-1])
    lo = np.concatenate([[c[0] - (mid[0] - c[0])], mid])
    hi = np.concatenate([mid, [c[-1] + (c[-1] - mid[-1])]])
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return lo[inv], hi[inv]


# ---------------------------------------------------------------------------
# Time axis + variable detection
# ---------------------------------------------------------------------------

# Cumulative days BEFORE each month in a 365-day (noleap) year (index by month-1).
_NOLEAP_CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def _to_days_since_1850(time_da) -> tuple[np.ndarray, np.ndarray]:
    """``(days, keep)`` where ``days`` is days since 1850-01-01 on a 365-day
    (noleap) axis and ``keep`` is a boolean mask that drops Feb 29.

    The raw CMIP6 SOLARIS files use a GREGORIAN calendar; the loader anchors to a
    noleap 'days since 1850-01-01' axis.  A blind ``(year-1850)*365 + (doy-1)``
    keeps Feb 29 and shifts Mar–Dec of every leap year by a day (then duplicates
    the next Jan 1).  Instead map each date by its noleap CALENDAR position
    (``month/day`` -> noleap day-of-year) and DROP Feb 29, so the output axis is
    a clean 365-day year.  A numeric input is assumed already days-since-1850.
    """
    vals = np.asarray(time_da.values)
    if np.issubdtype(vals.dtype, np.number):
        v = vals.astype(np.float64)
        return v, np.ones(v.shape, dtype=bool)
    flat = vals.reshape(-1)
    if np.issubdtype(vals.dtype, np.datetime64):
        # xarray decodes an in-range Gregorian axis to datetime64[ns], whose
        # ``.item()`` is an integer ns count (no .month); go via pandas so we get
        # real year/month/day.  cftime object arrays already carry those attrs.
        import pandas as pd
        stamps = [pd.Timestamp(t) for t in flat]
    else:
        stamps = list(flat)
    days = np.empty(flat.shape, dtype=np.float64)
    keep = np.ones(flat.shape, dtype=bool)
    for i, tt in enumerate(stamps):
        if tt.month == 2 and tt.day == 29:      # leap day has no noleap slot
            keep[i] = False
            days[i] = np.nan
            continue
        noleap_doy = _NOLEAP_CUM[tt.month - 1] + tt.day    # 1..365
        days[i] = (tt.year - 1850) * 365.0 + (noleap_doy - 1)
    return days.reshape(vals.shape), keep.reshape(vals.shape)


def _find_var(ds, candidates, kind: str) -> str:
    lower = {str(v).lower(): v for v in ds.variables}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(
        f"no {kind} variable found (looked for {candidates}; "
        f"have {list(ds.data_vars)})")


def adapt_solar(ds, sw_gas_file: str | None = None) -> object:
    """Pure core: a raw solar Dataset -> the deck TSI + SSI_frac Dataset.

    Detects ``tsi`` / ``ssi`` / wavelength by (case-insensitive) name, bins the
    spectral irradiance into the 14 RRTMG-SW bands, and puts TSI + SSI_frac on
    the ``days since 1850-01-01`` (noleap) axis the loader anchors against.
    """
    import xarray as xr

    edges = rrtmgsw_band_edges_nm(sw_gas_file)
    tsi_v = _find_var(ds, ("tsi", "TSI", "rsdt", "tsi_r"), "TSI")
    ssi_v = _find_var(ds, ("ssi", "SSI", "ssi_r", "solar_irradiance"), "spectral SSI")
    ssi = ds[ssi_v]
    time_name = "time" if "time" in ssi.dims else ssi.dims[0]
    wl_dim = next((d for d in ssi.dims if d != time_name), None)
    if wl_dim is None:
        raise ValueError(f"{ssi_v} has no spectral (wavelength) dimension")
    wl_name = _find_var(ds, (wl_dim, "wavelength", "wl", "lambda", "wvl"),
                        "wavelength")
    bin_lo, bin_hi = _bin_bounds_nm(ds, wl_name, wl_dim)

    # ssi -> (time, wl) with the spectral axis trailing.
    ssi_2d = ssi.transpose(time_name, wl_name, ...)
    ssi_arr = np.asarray(ssi_2d.values, dtype=np.float64)
    ssi_arr = ssi_arr.reshape(ssi_arr.shape[0], ssi_arr.shape[1])

    ssi_frac = ssi_to_band_fractions(bin_lo, bin_hi, ssi_arr, edges)  # (time, 14)
    tsi = np.asarray(ds[tsi_v].values, dtype=np.float64).reshape(-1)
    days, keep = _to_days_since_1850(ds[time_name])
    if not (tsi.shape[0] == ssi_frac.shape[0] == days.shape[0]):
        raise ValueError("TSI / SSI / time lengths disagree "
                         f"({tsi.shape[0]}/{ssi_frac.shape[0]}/{days.shape[0]})")
    # Drop Feb-29 records so the axis is a clean noleap 365-day year.
    if not keep.all():
        days, tsi, ssi_frac = days[keep], tsi[keep], ssi_frac[keep]

    return xr.Dataset(
        data_vars={
            "TSI": (("time",), tsi, {"units": "W m-2"}),
            "SSI_frac": (("time", "numwl"), ssi_frac,
                         {"units": "1",
                          "long_name": "Per-band fraction of TSI (RRTMG-SW "
                                       "order, 820-2680 cm-1 band last)"}),
        },
        coords={"time": ("time", days,
                         {"units": "days since 1850-01-01 00:00:00",
                          "calendar": "noleap"})},
        attrs={"source": "scripts/data/adapt_cmip6_solar.py",
               "comment": "SOLARIS-HEPPA ssi binned to the 14 RRTMG-SW bands "
                          "(edges from rrtmgp-gas-sw-g112.nc)"},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_file", type=Path, required=True,
                   help="Raw input4MIPs solar file (tsi + spectral ssi).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output TSI + SSI_frac NetCDF (deck --solar-file).")
    p.add_argument("--sw-gas-file", type=str, default=None,
                   help="Override the RRTMGP-SW gas-optics table used for the "
                        "band edges (default: the shipped g112 table).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import xarray as xr

    args = parse_args(argv)
    with xr.open_dataset(args.in_file, decode_times=True) as ds:
        merged = adapt_solar(ds, args.sw_gas_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(args.out)
    n = merged.sizes["time"]
    print(f"Wrote {args.out} ({n} records; TSI + 14-band SSI_frac, RRTMG-SW "
          f"order)")
    print("Consume: run_amip_cmip6_deck.py --solar-source spectral_file "
          f"--solar-file {args.out} --solar-tsi-var TSI --solar-spectral-var "
          f"SSI_frac --solar-spectral-band-order rrtmg_sw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
