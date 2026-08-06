#!/usr/bin/env python
"""Generate a slab-ocean q-flux (ocean-heat-transport convergence) climatology.

Two modes:

* ``--mode synthetic`` — a zonally-symmetric idealized q-flux: heat is EXPORTED
  from the deep tropics (q-flux < 0 into the mixed layer, matching the observed
  poleward ocean heat transport / divergence there) and IMPORTED poleward of
  the subtropics (q-flux > 0, convergence), with a modest sun-following annual
  cycle.  For idealized coupled slab runs.

* ``--mode derive --flux-file F.nc`` — the STANDARD CMIP slab calibration:
  read a monthly surface-flux climatology from an AMIP-SST-forced run
  (``sw_net_sfc, lw_net_sfc, hfss, hfls``; the fields ``diagnostics.py`` saves)
  and set ``q_flux = -(sw_net_sfc + lw_net_sfc - hfss - hfls)`` so a slab forced
  by this q-flux reproduces the AMIP SST climatology.

  SIGN (CLAUDE.md sign check): the slab budget is
  ``dT/dt = (sw_net + lw_net - shflx - lhflx + q_flux) / C_mix`` with all terms
  the net heat flux INTO the mixed layer (sw/lw +into-ocean, sh/lh +upward hence
  subtracted, q_flux +into the layer).  At fixed-SST equilibrium ``dT/dt = 0``
  forces ``q_flux = -(sw_net + lw_net - shflx - lhflx)`` — the negative of the
  net DOWNWARD surface heat flux the AMIP run diagnosed.

Output: ``(time=12, lat, lon)`` NetCDF with a ``q_flux`` variable [W/m2], noleap
mid-month time axis (``units="days since {year}-01-01"``) — the schema the
q-flux loader (``legoesm.ocean.forcing.qflux``) and the AMIP loader consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np

# Mid-month day-of-year on a 365-day noleap calendar (same anchors as the AMIP
# generator so the loaders' cyclic wrap sees a consistent axis).
_MID_DOY = np.array(
    [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]
)


def make_synthetic_qflux(nlat: int = 73, nlon: int = 144, *,
                         amplitude_W_m2: float = 40.0,
                         seasonal_W_m2: float = 15.0) -> np.ndarray:
    """Idealized zonally-symmetric q-flux climatology, shape ``(12, nlat, nlon)``.

    Pattern: ``q(lat) = -A * (1 - 3 sin^2 lat)`` — negative (heat export /
    divergence) in the deep tropics where ``|sin lat|`` is small, crossing zero
    near ``lat = arcsin(1/sqrt(3)) ~= 35 deg`` and positive (convergence) at high
    latitude, integrating to ~0 over the sphere (an OHT convergence has no net
    source).  A ``seasonal`` term follows the sun (warms the summer hemisphere's
    mid-latitudes), giving a first-order annual cycle.
    """
    lat = np.linspace(90.0, -90.0, nlat)          # degrees, descending (+90..-90)
    latr = np.deg2rad(lat)
    # Annual-mean zonal pattern [W/m2], zero net global integral by construction.
    zonal_mean = -amplitude_W_m2 * (1.0 - 3.0 * np.sin(latr) ** 2)  # (nlat,)
    out = np.empty((12, nlat, nlon), dtype=np.float64)
    for m in range(12):
        # Sun-following seasonal modulation: peak downward-into-ocean transport
        # in the summer hemisphere (phase so July warms the NH).
        phase = np.cos(2.0 * np.pi * (_MID_DOY[m] / 365.0) - np.pi / 6.0)
        seasonal = seasonal_W_m2 * phase * np.sin(latr)  # antisymmetric in lat
        out[m] = (zonal_mean + seasonal)[:, None]        # broadcast over lon
    return out


def derive_qflux_from_fluxes(flux_file: str) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """Derive ``q_flux`` from a monthly surface-flux climatology (CMIP calibration).

    ``flux_file`` must contain ``sw_net_sfc, lw_net_sfc, hfss, hfls`` [W/m2] over
    ``(time, lat, lon)`` with 12 monthly records (the AMIP-run diagnostics).
    Returns ``(q_flux(12,nlat,nlon), lat, lon)``.
    """
    import xarray as xr

    with xr.open_dataset(flux_file, decode_times=False) as ds:
        missing = [v for v in ("sw_net_sfc", "lw_net_sfc", "hfss", "hfls")
                   if v not in ds.variables]
        if missing:
            raise ValueError(
                f"flux file {flux_file} missing {missing}; needs the AMIP "
                "surface-flux diagnostics sw_net_sfc/lw_net_sfc/hfss/hfls")
        sw = np.asarray(ds["sw_net_sfc"].values, dtype=np.float64)
        lw = np.asarray(ds["lw_net_sfc"].values, dtype=np.float64)
        sh = np.asarray(ds["hfss"].values, dtype=np.float64)
        lh = np.asarray(ds["hfls"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
    # q_flux = -(net downward surface heat flux); see the module docstring sign
    # derivation.  sw/lw are +into-ocean, sh/lh are +upward (subtracted).
    net_down = sw + lw - sh - lh
    q_flux = -net_down
    if q_flux.ndim != 3 or q_flux.shape[0] != 12:
        raise ValueError(
            f"derived q_flux must be (12, lat, lon); got {q_flux.shape}")
    return q_flux, lat, lon


def write_qflux(out_path: Path, q_flux: np.ndarray, lat: np.ndarray,
                lon: np.ndarray, *, start_year: int) -> None:
    """Write the ``(12, nlat, nlon)`` q-flux to NetCDF (loader schema)."""
    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.title = "Slab-ocean q-flux (OHT convergence) climatology"
        ds.source = "scripts/data/generate_qflux_climatology.py"
        ds.references = "sign: +W/m2 INTO the mixed layer (config.Q_flux sign)"
        ds.createDimension("time", 12)
        ds.createDimension("lat", lat.shape[0])
        ds.createDimension("lon", lon.shape[0])
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {start_year}-01-01"
        t.calendar = "noleap"
        t[:] = _MID_DOY
        vlat = ds.createVariable("lat", "f8", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = lat
        vlon = ds.createVariable("lon", "f8", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = lon
        v = ds.createVariable("q_flux", "f8", ("time", "lat", "lon"))
        v.units = "W m-2"
        v.long_name = "ocean heat transport convergence (+ into mixed layer)"
        v[:] = q_flux


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path, help="output NetCDF path")
    p.add_argument("--mode", choices=["synthetic", "derive"], default="synthetic")
    p.add_argument("--start-year", type=int, default=1979,
                   help="time-axis epoch year (days since <year>-01-01)")
    p.add_argument("--nlat", type=int, default=73)
    p.add_argument("--nlon", type=int, default=144)
    p.add_argument("--amplitude", type=float, default=40.0,
                   help="synthetic: annual-mean zonal amplitude [W/m2]")
    p.add_argument("--seasonal", type=float, default=15.0,
                   help="synthetic: seasonal-cycle amplitude [W/m2]")
    p.add_argument("--flux-file", type=str, default="",
                   help="derive: monthly AMIP surface-flux climatology NetCDF")
    args = p.parse_args(argv)

    if args.mode == "synthetic":
        q = make_synthetic_qflux(args.nlat, args.nlon,
                                 amplitude_W_m2=args.amplitude,
                                 seasonal_W_m2=args.seasonal)
        lat = np.linspace(90.0, -90.0, args.nlat)
        lon = np.linspace(0.0, 360.0, args.nlon, endpoint=False)
    else:
        if not args.flux_file:
            p.error("--mode derive requires --flux-file")
        q, lat, lon = derive_qflux_from_fluxes(args.flux_file)

    write_qflux(args.out, q, lat, lon, start_year=args.start_year)
    print(f"wrote {args.out}  shape={q.shape}  "
          f"range=[{q.min():.1f}, {q.max():.1f}] W/m2  "
          f"global-mean={q.mean():.2f} W/m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
