#!/usr/bin/env python
"""Convert the PHC3.0 hydrography into a WOA18-format IC for legoESM.

Why
---
FESOM2's CORE2 hindcast cold-starts from ``phc3.0_winter.nc`` in
``/pool/data/AWICM/FESOM2/INITIAL/phc3.0/`` (``namelist.oce``'s
``tracer_init3d``: ``filelist = 'phc3.0_winter.nc'``, ``varlist = 'salt',
'temp'``, ``t_insitu = .true.``).  legoESM's IC path
(``init_ocean_from_woa``) reads a WOA18-format file: variables ``t_an`` /
``s_an``, shaped ``(time, depth, lat, lon)``, on :data:`WOA_DEPTHS`.  This
script bridges the two so a legoESM run starts from the SAME water mass
FESOM2 starts from -- with ZERO model-code change, the same pattern as
``build_woce_ic_latlon.py``.

Two conversions, both of which change numbers materially:

1. **In-situ -> potential temperature.**  PHC3's ``temp`` is IN-SITU
   temperature; every legoESM EOS and the prognostic tracer are POTENTIAL
   temperature.  FESOM2 does this conversion itself when ``t_insitu =
   .true.`` (``gen_ic3d.F90`` -> ``insitu2pot``), evaluating the adiabatic
   lapse rate at ``p = |z|`` in dbar with reference pressure 0.  We call the
   same algorithm through ``legoesm.ocean.eos.potential_temperature``.
   Skipping it leaves the deep ocean ~0.1 degC too warm at 1000 m and
   ~0.5 degC at 5000 m.

2. **Re-levelling to WOA_DEPTHS.**  ``init_ocean_from_woa`` now reads the
   source depth axis from the FILE, so this is no longer required for
   correctness -- it used to be, when the loader took the axis from
   ``WOA_DEPTHS[:n_depth]`` regardless and a file on PHC3's own 33 levels was
   read as if its deep values sat in the top few hundred metres.  It is kept
   because it gives one common axis and lets the conversion below happen at
   the target depths.

No horizontal regrid is needed: PHC3.0 is already the regular 1 deg lat-lon
grid (180x360, centres at -89.5..89.5 / 0.5..359.5) that WOA18 uses.

Usage
-----
    python scripts/data/build_phc3_ic_for_omip.py \\
        --phc3 /pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc \\
        --out  /scratch/b/b381103/fesom2_comparison/phc3_winter_woa_format.nc
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# The adiabatic-lapse-rate polynomial is evaluated once per grid point in
# float64; x64 must be on before jax.numpy is first imported.
os.environ.setdefault("JAX_ENABLE_X64", "1")


def build_phc3_ic(phc3_path: Path, t_var: str = "temp", s_var: str = "salt"):
    """Return ``(t_an, s_an, lat, lon, depths)`` ready to write.

    ``t_an`` / ``s_an`` have shape ``(1, n_woa_depth, n_lat, n_lon)`` with
    NaN wherever PHC3 has no data (land, below seafloor) -- the same no-data
    convention a real WOA18 file uses, which the downstream bilinear
    interpolation and land masking already handle.
    """
    import xarray as xr

    from legoesm.ocean.eos import potential_temperature
    from legoesm.ocean.init_woa import WOA_DEPTHS, interp_column_to_depths

    ds = xr.open_dataset(phc3_path, decode_times=False)
    for name in (t_var, s_var):
        if name not in ds.data_vars:
            raise ValueError(
                f"{phc3_path}: no variable {name!r}; available "
                f"{list(ds.data_vars)}"
            )
    T_insitu = np.asarray(ds[t_var].values, dtype=np.float64)   # (nz, ny, nx)
    S = np.asarray(ds[s_var].values, dtype=np.float64)
    src_z = np.abs(np.asarray(ds["depth"].values, dtype=np.float64))
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    ds.close()

    if T_insitu.ndim != 3:
        raise ValueError(
            f"expected (depth, lat, lon) for {t_var!r}; got {T_insitu.shape}"
        )
    if not np.all(np.diff(src_z) > 0):
        raise ValueError(
            "PHC3 depth axis is not strictly ascending after abs(); "
            f"got {src_z[:5]}...{src_z[-3:]}"
        )

    # ORDER MATTERS, and this is FESOM2's order: interpolate the IN-SITU
    # field to the target levels FIRST, then convert at the TARGET depth.
    # Converting at the source depths and interpolating the result afterwards
    # is a different operation -- the two do not commute, because the
    # adiabatic correction is a nonlinear function of depth.  Measured on the
    # real PHC3 file, the two orders differ by up to ~1 degC (RMS 0.11 degC)
    # in the deepest extrapolated levels.  FESOM2 interpolates T/S onto its
    # own levels in ``gen_ic3d.F90`` and only then calls ``insitu2pot``,
    # which evaluates ``ptheta`` at the MODEL depth ``abs(Z(nz))``.

    # 1) re-level every column of IN-SITU T and S onto WOA_DEPTHS.
    n_woa = WOA_DEPTHS.size
    ny, nx = T_insitu.shape[1], T_insitu.shape[2]
    Tw_insitu = np.empty((n_woa, ny, nx), dtype=np.float64)
    Sw = np.empty_like(Tw_insitu)
    for j in range(ny):
        for i in range(nx):
            Tw_insitu[:, j, i] = interp_column_to_depths(
                T_insitu[:, j, i], src_z, WOA_DEPTHS)
            Sw[:, j, i] = interp_column_to_depths(S[:, j, i], src_z,
                                                  WOA_DEPTHS)

    # 2) in-situ -> potential temperature AT THE TARGET DEPTHS.
    #    p_dbar = depth in metres, matching FESOM2's insitu2pot.
    #
    #    The conversion pressure is CAPPED at each column's deepest valid
    #    source level.  ``interp_column_to_depths`` holds the deepest valid
    #    value below the data (np.interp edge behaviour), so without the cap a
    #    shelf column's warm bottom value would be adiabatically corrected as
    #    if it sat at 5500 m -- an invented 1.4 degC cooling in cells that are
    #    below the seafloor anyway.  FESOM2 has the same restriction for the
    #    same reason: ``insitu2pot`` loops ``nz = nzmin, nzmax-1``, i.e. WET
    #    LEVELS ONLY.
    valid = np.isfinite(T_insitu)
    any_valid = valid.any(axis=0)
    deepest_idx = np.where(any_valid,
                           valid.shape[0] - 1 - valid[::-1].argmax(axis=0), 0)
    z_valid_max = np.where(any_valid, src_z[deepest_idx], 0.0)   # (ny, nx)
    p_eff = np.minimum(WOA_DEPTHS[:, None, None], z_valid_max[None, :, :])

    Tw = np.asarray(potential_temperature(Sw, Tw_insitu, p_eff, 0.0))
    drop = np.where(np.isfinite(Tw_insitu), Tw_insitu - Tw, np.nan)
    k_max = int(np.nanargmax(np.nanmax(drop, axis=(1, 2))))
    print(f"[phc3] in-situ -> potential: max cooling "
          f"{np.nanmax(drop):.4f} degC at {WOA_DEPTHS[k_max]:.0f} m; "
          f"surface-level max {np.nanmax(drop[0]):.2e} degC; "
          f"deepest valid source level {np.nanmax(z_valid_max):.0f} m")

    return Tw[None], Sw[None], lat, lon, WOA_DEPTHS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--phc3", type=str,
        default="/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc",
        help="PHC3.0 climatology NetCDF (the file FESOM2 initialises from).",
    )
    p.add_argument("--t-var", type=str, default="temp")
    p.add_argument("--s-var", type=str, default="salt")
    p.add_argument("--out", type=str, required=True,
                   help="Output WOA18-format NetCDF path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import xarray as xr

    args = parse_args(argv)
    t_an, s_an, lat, lon, depths = build_phc3_ic(
        Path(args.phc3), t_var=args.t_var, s_var=args.s_var,
    )
    coords = dict(time=("time", [0.0]), depth=("depth", depths),
                  lat=("lat", lat), lon=("lon", lon))
    out = xr.Dataset(
        {
            "t_an": (("time", "depth", "lat", "lon"), t_an),
            "s_an": (("time", "depth", "lat", "lon"), s_an),
        },
        coords=coords,
        attrs={
            "title": "PHC3.0 winter climatology in WOA18 layout",
            "source": str(args.phc3),
            "comment": (
                "temp converted IN-SITU -> POTENTIAL via UNESCO/Bryden "
                "(legoesm.ocean.eos.potential_temperature, p_dbar=depth_m, "
                "p_ref=0), matching FESOM2 t_insitu=.true.; re-levelled onto "
                "legoesm.ocean.init_woa.WOA_DEPTHS"
            ),
        },
    )
    out["t_an"].attrs = dict(units="degC",
                             long_name="potential temperature")
    out["s_an"].attrs = dict(units="psu", long_name="practical salinity")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    print(f"[phc3] wrote {args.out}  t_an{t_an.shape}")
    surf = t_an[0, 0]
    print(f"[phc3] surface theta range "
          f"[{np.nanmin(surf):.2f}, {np.nanmax(surf):.2f}] degC; "
          f"wet fraction {np.isfinite(surf).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
