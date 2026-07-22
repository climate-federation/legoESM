#!/usr/bin/env python
"""Adapt raw input4MIPs GHG concentration files into one deck-consumable file.

``download_cmip6_forcing.py --channels ghg`` fetches the CMIP6 well-mixed
greenhouse-gas concentrations as SEPARATE per-gas input4MIPs files, each with a
CF-long variable name (``mole_fraction_of_carbon_dioxide_in_air`` …) on its own
time axis.  The AMIP loader (``forcing/external.py::_load_ghg_annual_file``)
instead wants ONE file with the SHORT variable names ``CO2/CH4/N2O/CFC_11/
CFC_12`` on a shared ``(time, lat=1, lon=1)`` grid whose time is fractional
years (``units = "year as %Y.%f"``) — exactly the schema the synthetic
``generate_amip_forcing.make_ghg_annual`` emits.

This bridges the two: read the per-gas files, extract each global-mean annual
series, put them on a common fractional-year axis, rename to the short names,
and write the single deck file.  Each gas keeps its native ``units`` attribute
(the loader normalises ppm/ppb/ppt/``mol mol-1`` to a mole fraction itself), so
no unit conversion is done here.

Usage::

    # From a directory of per-gas input4MIPs files (auto-matched by CF name):
    python scripts/data/adapt_cmip6_ghg.py --in-dir data/cmip6_forcing/ghg \\
        --out data/cmip6_forcing/ghg_cmip6.nc

    # Or point at each file explicitly:
    python scripts/data/adapt_cmip6_ghg.py --co2 co2.nc --ch4 ch4.nc \\
        --n2o n2o.nc --cfc11 cfc11.nc --cfc12 cfc12.nc --out ghg_cmip6.nc

Consume it with ``run_amip_cmip6_deck.py --ghg-forcing external --ghg-file
<out>`` (or ``run_amip.py`` directly).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

# Deck short name -> its input4MIPs CF variable name.  CFC-11/CFC-12 use the
# input4MIPs ``cfc11``/``cfc12`` spellings.
GAS_TO_CFVAR: dict[str, str] = {
    "CO2": "mole_fraction_of_carbon_dioxide_in_air",
    "CH4": "mole_fraction_of_methane_in_air",
    "N2O": "mole_fraction_of_nitrous_oxide_in_air",
    "CFC_11": "mole_fraction_of_cfc11_in_air",
    "CFC_12": "mole_fraction_of_cfc12_in_air",
}
# CLI flag stem -> deck short name (for the explicit-file form).
FLAG_TO_GAS = {"co2": "CO2", "ch4": "CH4", "n2o": "N2O",
               "cfc11": "CFC_11", "cfc12": "CFC_12"}


def _to_fractional_year(time_da) -> np.ndarray:
    """Convert a decoded time coordinate to fractional years (``YYYY.f``).

    Handles both the CF ``datetime``/``cftime`` axis (input4MIPs concentration
    files) and an already-decimal-year axis.  A year is ``year + (day_of_year -
    1) / days_in_year`` so an annual mid-year value lands near ``YYYY.5``.
    """
    vals = np.asarray(time_da.values)
    if np.issubdtype(vals.dtype, np.floating) or np.issubdtype(vals.dtype, np.integer):
        # Already numeric — assume it is (fractional) years.
        return vals.astype(np.float64)
    out = np.empty(vals.shape, dtype=np.float64)
    for i, t in enumerate(vals.flat):
        # datetime64 -> python datetime; cftime datetimes already have the attrs.
        tt = t.item() if hasattr(t, "item") else t
        year = tt.year
        doy = tt.timetuple().tm_yday
        # days in this year, calendar-aware for cftime (noleap etc.).
        if hasattr(tt, "daysinmonth"):  # cftime datetime
            start = tt.__class__(year, 1, 1)
            nxt = tt.__class__(year + 1, 1, 1)
            days = (nxt - start).days
        else:
            days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        out.flat[i] = year + (doy - 1) / days
    return out


def _canonical_units(units: str) -> str:
    """Map a units string to the spelling the GHG loader recognises.

    The CMIP6 UoM files use a dotted-exponent scale like ``"1.e-6"`` which the
    loader's ``_GHG_UNIT_TO_MOLE_FRACTION`` does NOT list (it has ``"1e-6"``),
    so it would fall back to a magnitude heuristic + warning.  Canonicalise a
    power-of-ten numeric scale (``1.e-6`` / ``1.0e-6`` / ``1e-6``) to ``1e{exp}``;
    leave any already-named unit (``mol mol-1`` / ``ppm`` …) untouched.
    """
    s = str(units).strip()
    try:
        val = float(s)
    except ValueError:
        return s
    if not np.isfinite(val) or val <= 0.0:
        return s
    exp = round(np.log10(val))
    # atol=0.0 (pure relative): the default atol=1e-8 would make tiny non-powers
    # like 2e-9 spuriously "close" to 1e-9 and mis-canonicalise them.
    if np.isclose(val, 10.0 ** exp, rtol=1e-9, atol=0.0):
        return "1" if exp == 0 else f"1e{exp}"   # -> "1e-6" / "1e-9" / "1e-12"
    return s


def _select_global(var, ds):
    """Reduce a gas DataArray's NON-time dims to the GLOBAL series.

    Real CMIP6 UoM concentration files carry a ``sector`` dimension
    ``(Global, NH, SH)`` — averaging it (the naive reduction) would give
    ``(Global+NH+SH)/3``, NOT the global-mean the loader wants.  For each
    non-time dim: pick the label containing "global" if the dim is labelled;
    squeeze a singleton; else take index 0 (the GMNHSH files order Global first)
    with a loud note so a silent positional guess never passes unnoticed.
    """
    time_name = "time" if "time" in var.dims else var.dims[0]
    for d in [dd for dd in var.dims if dd != time_name]:
        picked = _global_index(ds, d)
        if picked is not None:
            var = var.isel({d: picked})
        elif var.sizes[d] == 1:
            var = var.isel({d: 0})
        else:
            print(f"[adapt_cmip6_ghg] dim {d!r} size {var.sizes[d]} has no "
                  f"resolvable 'global' region (no string labels, no ids/"
                  f"original_names attr); taking index 0 (assumed Global, GMNHSH "
                  f"order).", file=sys.stderr)
            var = var.isel({d: 0})
    return var


def _global_index(ds, dim: str):
    """Index of the GLOBAL region along ``dim``, or None if unresolvable.

    Handles both encodings seen in input4MIPs GHG files:
    * a coordinate variable of STRING region labels (``["Global","NH","SH"]``);
    * the real UoM form — an INTEGER ``sector`` coord (0,1,2) whose region names
      live in a string attribute (``ids`` / ``original_names`` / ``sector``),
      e.g. ``"0: Global; 1: Northern Hemisphere; 2: Southern Hemisphere"``.
    """
    if dim not in ds.variables:
        return None
    coord = ds[dim]
    # 1) direct string labels on the coordinate.
    vals = np.asarray(coord.values).reshape(-1)
    if vals.dtype.kind in ("U", "S", "O"):
        for i, lab in enumerate(vals):
            if "global" in str(lab).lower():
                return i
    # 2) region names carried in an attribute (integer-coded sector).
    for attr in ("ids", "original_names", "sector", "region", "long_name"):
        raw = coord.attrs.get(attr)
        if not raw:
            continue
        tokens = [t.strip() for t in str(raw).replace(",", ";").split(";")]
        for pos, tok in enumerate(tokens):
            if "global" not in tok.lower():
                continue
            # "N: Global": N is a CODED sector value — map it to its POSITION in
            # the coord.  If the code is absent from the coord (or the token has
            # no "N:" prefix) fall back to the token position (ids order == coord
            # order by convention) — never treat the code as a position (a code
            # like "5" is not index 5).
            head = tok.split(":", 1)[0].strip()
            if head.isdigit() and int(head) in set(vals.tolist()):
                return int(np.where(vals == int(head))[0][0])
            return pos
    return None


def _extract_gas_series(ds, cf_var: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(years, values, units)`` for one gas from its Dataset.

    Selects the GLOBAL series along any region/sector/space dim (never averages
    hemispheres) and canonicalises the units to a loader-recognised spelling.
    """
    if cf_var not in ds.variables:
        raise ValueError(
            f"variable {cf_var!r} not in dataset (has {list(ds.data_vars)})")
    var = _select_global(ds[cf_var], ds)
    time_name = "time" if "time" in var.dims else var.dims[0]
    values = np.asarray(var.values, dtype=np.float64).reshape(-1)
    years = _to_fractional_year(ds[time_name])
    if values.shape[0] != years.shape[0]:
        raise ValueError(
            f"{cf_var}: {values.shape[0]} values vs {years.shape[0]} times")
    units = _canonical_units(str(ds[cf_var].attrs.get("units", "1")))
    return years, values, units


def merge_ghg(per_gas: dict) -> object:
    """Merge per-gas Datasets into ONE deck GHG Dataset (the pure, testable
    core — takes/returns xarray objects, no file I/O).

    ``per_gas`` maps a deck short name (``CO2`` …) to that gas's opened Dataset.
    Every gas must resolve to the SAME fractional-year axis (input4MIPs GHG
    concentrations are annual on a common calendar); a mismatch is a hard error
    rather than a silent reindex, so a wrong/misaligned file is caught.
    """
    import xarray as xr

    missing = [g for g in GAS_TO_CFVAR if g not in per_gas]
    if missing:
        raise ValueError(
            f"missing gas file(s) for {missing}; the loader requires all of "
            f"{list(GAS_TO_CFVAR)}")

    ref_years = None
    data_vars = {}
    for gas, cf_var in GAS_TO_CFVAR.items():
        years, values, units = _extract_gas_series(per_gas[gas], cf_var)
        if ref_years is None:
            ref_years = years
        elif not np.allclose(years, ref_years, atol=1e-6):
            raise ValueError(
                f"{gas} time axis differs from the first gas "
                f"({years[:1]}..{years[-1:]} vs {ref_years[:1]}..{ref_years[-1:]}); "
                f"the per-gas files must share one annual axis.")
        # (time, lat=1, lon=1) to match the loader's expected shape.
        data_vars[gas] = (
            ("time", "lat", "lon"),
            values.reshape(-1, 1, 1),
            {"units": units,
             "long_name": f"{gas} mole fraction (CMIP6 input4MIPs)"},
        )

    return xr.Dataset(
        data_vars={k: (d, v, a) for k, (d, v, a) in data_vars.items()},
        coords={
            "time": ("time", ref_years, {"units": "year as %Y.%f"}),
            "lat": ("lat", np.array([0.0], dtype=np.float64)),
            "lon": ("lon", np.array([0.0], dtype=np.float64)),
        },
        attrs={"source": "scripts/data/adapt_cmip6_ghg.py",
               "comment": "per-gas input4MIPs GHG concentrations merged for the "
                          "legoESM AMIP GHG loader"},
    )


def _match_in_dir(in_dir: Path) -> dict:
    """Map each gas to a file in ``in_dir`` by its CF variable name in the
    filename (input4MIPs names encode the variable), else by a substring."""
    import xarray as xr

    files = sorted(glob.glob(str(in_dir / "*.nc")))
    out: dict = {}
    for gas, cf_var in GAS_TO_CFVAR.items():
        # input4MIPs filenames use the hyphenated variable id.
        stem = cf_var.replace("_", "-")
        hit = next((f for f in files if stem in Path(f).name), None)
        if hit is None:
            # fall back: open each and check the variable is present.
            for f in files:
                try:
                    with xr.open_dataset(f, decode_times=True) as ds:
                        if cf_var in ds.variables:
                            hit = f
                            break
                except Exception:
                    continue
        if hit is not None:
            out[gas] = xr.open_dataset(hit, decode_times=True)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=None,
                   help="Directory of per-gas input4MIPs files (auto-matched).")
    for flag in FLAG_TO_GAS:
        p.add_argument(f"--{flag}", type=Path, default=None,
                       help=f"Explicit {FLAG_TO_GAS[flag]} input4MIPs file.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output merged GHG NetCDF (deck --ghg-file).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import xarray as xr

    args = parse_args(argv)
    if args.in_dir is not None:
        per_gas = _match_in_dir(args.in_dir)
    else:
        per_gas = {}
        for flag, gas in FLAG_TO_GAS.items():
            path = getattr(args, flag)
            if path is not None:
                per_gas[gas] = xr.open_dataset(path, decode_times=True)
    if not per_gas:
        print("No input files (pass --in-dir or the per-gas flags).",
              file=sys.stderr)
        return 2

    try:
        merged = merge_ghg(per_gas)
    finally:
        for ds in per_gas.values():
            ds.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(args.out)
    yrs = np.asarray(merged["time"].values)
    print(f"Wrote {args.out} ({yrs.size} years, {yrs[0]:.1f}->{yrs[-1]:.1f}; "
          f"vars {list(GAS_TO_CFVAR)})")
    print("Consume: run_amip_cmip6_deck.py --ghg-forcing external "
          f"--ghg-file {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
