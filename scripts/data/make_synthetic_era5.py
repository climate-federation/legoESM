"""Write a TINY SYNTHETIC ERA5-like zarr for SMOKE-TESTING the correction campaign.

This is NOT real reanalysis — it is a small, physically-plausible store in the exact
layout :func:`legoesm.training.era5_to_state.load_era5_slice` expects (long ERA5
variable names on ``(time, level, lat, lon)`` / ``(time, lat, lon)``, pressure
``level`` in hPa descending, ``lat`` 90→-90 descending — the ERA5 convention).  Use
it to validate the WHOLE campaign loop on your machine BEFORE submitting the
multi-day real-ERA5 job::

    python scripts/data/make_synthetic_era5.py /tmp/syn_era5.zarr
    python scripts/experiment/write_amip_clubb_lite_config.py /tmp/cfg.json --nlev 10
    JAX_ENABLE_X64=1 python scripts/run/run_correction_campaign.py \\
        --config /tmp/cfg.json --era5-zarr /tmp/syn_era5.zarr --iterations 1 \\
        --n-worst 4 --out /tmp/out.json

The ``--dry-run`` pre-flight validates CONSTRUCTION; a synthetic full run validates
EXECUTION (the model runs, a column LES spins off + diagnoses, the bias computes).
The fields VARY by level (T, q) and latitude (u) so the vertical interp + horizontal
regrid are genuinely exercised, not trivially uniform.
"""

from __future__ import annotations

import argparse

import numpy as np
from legoesm.training.era5_to_state import WB2_PRESSURE_LEVELS


def build_synthetic_era5_dataset(
    *, nlat: int = 24, nlon: int = 48, ntime: int = 2, levels=WB2_PRESSURE_LEVELS
):
    """A tiny, physically-plausible ERA5-like :class:`xarray.Dataset`.

    ``levels`` are pressure levels [hPa]; defaults to ``WB2_PRESSURE_LEVELS`` (the
    campaign's :class:`TrainingERA5Config` default) so the store loads with no
    ``--era5-n-times`` / levels override.  Profiles: ``T`` decreases with height
    (warmer at high pressure), ``q`` drops sharply aloft, ``u`` varies linearly with
    latitude, ``v`` small; ``p_s`` ~1000 hPa, SST ~290 K, flat surface geopotential.
    """
    import xarray as xr

    p = np.asarray(levels, dtype=np.float64)                       # hPa, descending
    nlev = p.size
    lat = np.linspace(90.0, -90.0, nlat)                           # ERA5 convention (descending)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    frac = (p / 1000.0)                                            # 1 at surface → small aloft
    t_lev = (220.0 + 70.0 * frac**0.3).astype(np.float32)         # ~290 K (sfc) → ~249 K (50 hPa)
    q_lev = (1.0e-2 * frac**3 + 1.0e-6).astype(np.float32)        # moist sfc, dry aloft
    u_lat = (10.0 + 5.0 * (lat / 90.0)).astype(np.float32)        # linear in latitude

    def _bcast_lev(v):  # (nlev,) → (ntime, nlev, nlat, nlon)
        return np.broadcast_to(v[None, :, None, None], (ntime, nlev, nlat, nlon)).astype(np.float32)

    temperature = _bcast_lev(t_lev)
    specific_humidity = _bcast_lev(q_lev)
    u_wind = np.broadcast_to(
        u_lat[None, None, :, None], (ntime, nlev, nlat, nlon)).astype(np.float32)
    v_wind = np.full((ntime, nlev, nlat, nlon), 1.0, dtype=np.float32)
    surface_pressure = np.full((ntime, nlat, nlon), 1.0e5, dtype=np.float32)
    skin_temperature = np.full((ntime, nlat, nlon), 290.0, dtype=np.float32)
    z_sfc = np.zeros((ntime, nlat, nlon), dtype=np.float32)        # flat (no terrain)

    dims3, dims2 = ("time", "level", "lat", "lon"), ("time", "lat", "lon")
    return xr.Dataset(
        {
            "temperature": (dims3, temperature),
            "u_component_of_wind": (dims3, u_wind),
            "v_component_of_wind": (dims3, v_wind),
            "specific_humidity": (dims3, specific_humidity),
            "surface_pressure": (dims2, surface_pressure),
            "skin_temperature": (dims2, skin_temperature),
            "geopotential_at_surface": (dims2, z_sfc),
        },
        coords={"time": np.arange(ntime), "level": p, "lat": lat, "lon": lon},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", help="output zarr path")
    p.add_argument("--nlat", type=int, default=24)
    p.add_argument("--nlon", type=int, default=48)
    p.add_argument("--ntime", type=int, default=2)
    args = p.parse_args(argv)
    ds = build_synthetic_era5_dataset(nlat=args.nlat, nlon=args.nlon, ntime=args.ntime)
    ds.to_zarr(args.out, mode="w")
    print(f"[era5] wrote SYNTHETIC ERA5 smoke store ({args.ntime}t × {len(ds.level)}lev × "
          f"{args.nlat}×{args.nlon}) to {args.out} — NOT real reanalysis.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
