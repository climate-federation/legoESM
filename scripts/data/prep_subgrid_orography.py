"""Prepare a subgrid-orography stddev (SSO_STDH) NetCDF for orographic GWD.

``run_amip.py --subgrid-orography-file <file>`` feeds
:func:`legoesm.grids.topography.load_subgrid_orography`, which regrids a
regular lat-lon ``SSO_STDH(lat, lon)`` [m] field to the model grid so the
orographic gravity-wave-drag schemes (McFarlane, Lindzen) launch a per-column
stress ``tau_0 ∝ h_topo²`` that is localized to real mountains and ~0 over
ocean.  Without this file the schemes fall back to a single global
``config.h_topo = 500 m`` — a 500 m mountain over the open ocean too.

This computes the stddev of terrain height (elevation clipped to >= 0; ocean
bathymetry does not launch mountain waves) within ``block_deg`` blocks from a
regular lat-lon elevation dataset (the same source ``prep_etopo_topography.py``
consumes, e.g. NOAA ETOPO regridded to 0.25 deg):

    python scripts/data/prep_subgrid_orography.py \\
        --input data/amip/etopo_0p25deg.nc \\
        --out data/amip/sso_stdh_2deg.nc --block-deg 2.0
    # then:  run_amip.py --gravity-wave-drag mcfarlane \\
    #            --subgrid-orography-file data/amip/sso_stdh_2deg.nc

Pick ``block_deg`` ~ the model cell size (2 deg ~ C48).  DISCLOSURE: the
stddev is estimated from the ``fine_res_deg`` samples inside each block, so
variance at scales finer than ``fine_res_deg`` (0.25 deg ~ 25 km) is NOT
captured — this UNDERESTIMATES the true subgrid orographic stddev relative to
an extpar/GMTED product built from ~km-scale relief.  For a C48-class AMIP the
McFarlane launch stress is dominated by the resolved-to-25km band, and the
``h_topo`` tunable bounds (50-2000 m) bracket the residual scale error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Sibling import (scripts/data/ is not a package; running this file puts the
# directory on sys.path[0], but tests import via importlib from another cwd).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_etopo_topography import regrid_elevation_to_latlon  # noqa: E402


def subgrid_orography_stddev(
    ds,
    *,
    var_name: str = "",
    fine_res_deg: float = 0.25,
    block_deg: float = 2.0,
):
    """Per-block stddev of terrain height from a lat-lon elevation dataset.

    PURE (no I/O).  Normalizes the source through
    :func:`prep_etopo_topography.regrid_elevation_to_latlon` (variable
    auto-detect, lon [0,360) ascending, lat ascending, regular
    ``fine_res_deg`` grid), clips elevation to >= 0 m (terrain height above
    sea level; ocean contributes zeros so coastal blocks keep their cliffs),
    then reduces non-overlapping ``block_deg`` blocks to their AREA-WEIGHTED
    population stddev (sample weight ∝ cos(lat), the regular-grid cell area;
    unweighted moments would overweight the poleward rows of each block).
    Returns a Dataset ``SSO_STDH(lat, lon)`` [m] with block-center coords —
    the layout ``load_subgrid_orography`` auto-detects.
    """
    import numpy as np
    import xarray as xr

    factor_f = float(block_deg) / float(fine_res_deg)
    factor = int(round(factor_f))
    if abs(factor_f - factor) > 1e-9 or factor < 2:
        raise ValueError(
            f"block_deg must be an integer multiple (>= 2x) of fine_res_deg; "
            f"got block_deg={block_deg!r}, fine_res_deg={fine_res_deg!r}"
        )

    fine = regrid_elevation_to_latlon(
        ds, var_name=var_name, target_res_deg=float(fine_res_deg)
    )
    elev = np.asarray(fine["elevation"].values, dtype=np.float64)
    lat = np.asarray(fine["lat"].values, dtype=np.float64)
    lon = np.asarray(fine["lon"].values, dtype=np.float64)
    n_lat, n_lon = elev.shape
    if n_lat % factor or n_lon % factor:
        raise ValueError(
            f"global {n_lat}x{n_lon} grid at {fine_res_deg} deg does not tile "
            f"into {block_deg} deg blocks (factor {factor})"
        )

    # Terrain height above sea level: bathymetry launches no mountain waves.
    h = np.clip(elev, 0.0, None)
    hb = h.reshape(n_lat // factor, factor, n_lon // factor, factor)
    # Area weights on a regular lat-lon grid: cell area ∝ cos(lat) (row-wise;
    # constant in lon). Weighted first/second moments so every sample counts
    # by the area it represents inside the block.
    w_row = np.cos(np.deg2rad(lat))
    wb = np.broadcast_to(
        w_row.reshape(n_lat // factor, factor, 1, 1),
        hb.shape,
    )
    w_sum = np.maximum(wb.sum(axis=(1, 3)), 1e-12)
    mu = (wb * hb).sum(axis=(1, 3)) / w_sum
    var = (wb * (hb - mu[:, None, :, None]) ** 2).sum(axis=(1, 3)) / w_sum
    sso = np.sqrt(var)
    lat_b = lat.reshape(-1, factor).mean(axis=1)
    lon_b = lon.reshape(-1, factor).mean(axis=1)

    out = xr.Dataset(
        {"SSO_STDH": (("lat", "lon"), sso)},
        coords={"lat": lat_b, "lon": lon_b},
    )
    out["SSO_STDH"].attrs = {
        "units": "m",
        "long_name": (
            "standard deviation of subgrid orography "
            "(terrain height clipped to >= 0 m)"
        ),
    }
    out["lat"].attrs = {"units": "degrees_north"}
    out["lon"].attrs = {"units": "degrees_east"}
    return out


def main(argv=None) -> int:
    import xarray as xr

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", required=True,
                   help="regular lat-lon elevation NetCDF (ETOPO/GEBCO/...)")
    p.add_argument("--out", required=True, help="output SSO_STDH NetCDF path")
    p.add_argument("--elev-var", default="",
                   help="elevation variable name (default: auto-detect)")
    p.add_argument("--fine-res-deg", type=float, default=0.25,
                   help="normalization grid the stddev is sampled on")
    p.add_argument("--block-deg", type=float, default=2.0,
                   help="block size ~ model cell size (2 deg ~ C48)")
    args = p.parse_args(argv)

    with xr.open_dataset(args.input) as ds:
        out = subgrid_orography_stddev(
            ds,
            var_name=args.elev_var,
            fine_res_deg=args.fine_res_deg,
            block_deg=args.block_deg,
        )
    out.attrs["source"] = str(args.input)
    out.attrs["history"] = (
        f"prep_subgrid_orography.py --fine-res-deg {args.fine_res_deg} "
        f"--block-deg {args.block_deg}"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    sso = out["SSO_STDH"].values
    print(f"[sso] wrote {args.out}: shape {sso.shape}, "
          f"max {float(sso.max()):.1f} m, "
          f"land-blocks>50m {float((sso > 50.0).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
