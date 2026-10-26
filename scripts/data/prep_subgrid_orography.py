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


def subgrid_orography_residual_stddev(
    ds,
    *,
    var_name: str = "",
    fine_res_deg: float = 0.25,
    block_deg: float = 2.0,
    resolved_cutoff_deg: float = 4.0,
    estimator: str = "block_stddev",
):
    """Per-block stddev of the RESIDUAL terrain the model cannot resolve.

    The classic construction above takes the stddev of terrain inside each
    block, which mixes two different things: variance at scales the model
    RESOLVES in its own topography (double-counted drag once the GWD launches
    from it) and genuinely subgrid variance.  Issue #1712, measured: the block
    size acts as an accidental scale decomposition with a ~33x lever on the
    launch stress, and nothing ties it to the model grid.

    This variant makes the decomposition EXPLICIT.  Terrain is smoothed with a
    top-hat running mean of width ``resolved_cutoff_deg`` — the model's
    effective resolution, ~3-4x its cell size, NOT its cell size — and the
    stddev of ``h - smooth(h)`` is taken per block.  A wave much longer than
    the cutoff contributes ~nothing (the model resolves it); a wave much
    shorter contributes its full stddev; with ``estimator="rms"`` the block
    size keeps only the job of LOCATING the answer on the output grid and the
    physics lives in the cutoff, named and stamped in the file (with the
    default estimator the block still shapes it -- see ESTIMATOR).

    Same clipping (terrain >= 0), same area weighting, same output layout as
    the classic construction, so ``load_subgrid_orography`` reads either.
    The smoothing is lon-periodic; in lat the running mean is truncated at
    the poles (renormalised, not padded).

    COASTAL STEPS ARE KEPT, by design and by parity with the classic
    construction (whose docstring says "coastal blocks keep their cliffs"):
    with terrain clipped at 0 and no land mask, a coastal window mixes land
    with sea zeros and the land-sea step contributes ~H^2 f(1-f) of variance
    (GLM review).  A coastal cliff IS orographic forcing, so this is a
    deliberate property, not an oversight -- but a masked variant (weight by
    land fraction in the smooth AND the block moments) is the named follow-up
    if coastal drag is ever tuned against this field.

    ESTIMATOR (#1712, codex review, measured -- no default change made):
    ``"block_stddev"`` (default, unchanged) is the stddev of the residual about
    its BLOCK mean; it also removes residual waves longer than the block but
    shorter than the cutoff (a 2-deg wave at block 1 / cutoff 3.5 keeps 0.39x
    of its RMS), so the block size still shapes the answer.  ``"rms"`` is the
    residual's RMS about zero: it keeps that band in full, but also keeps the
    top-hat's transition-band leak of RESOLVED waves (~(cutoff/lambda)^2/6 of
    their amplitude, e.g. 2.9% of a 30-deg wave at a 4-deg cutoff), which the
    block demeaning suppressed.  Which one a production file uses is a choice.
    """
    import numpy as np
    import xarray as xr

    if estimator not in ("block_stddev", "rms"):
        raise ValueError(
            f"estimator must be 'block_stddev' or 'rms'; got {estimator!r}")
    factor_f = float(block_deg) / float(fine_res_deg)
    factor = int(round(factor_f))
    if abs(factor_f - factor) > 1e-9 or factor < 2:
        raise ValueError(
            f"block_deg must be an integer multiple (>= 2x) of fine_res_deg; "
            f"got block_deg={block_deg!r}, fine_res_deg={fine_res_deg!r}")
    k_f = float(resolved_cutoff_deg) / float(fine_res_deg)
    k = int(round(k_f))
    if abs(k_f - k) > 1e-9 or k < 2:
        raise ValueError(
            f"resolved_cutoff_deg must be an integer multiple (>= 2x) of "
            f"fine_res_deg; got {resolved_cutoff_deg!r} / {fine_res_deg!r}")
    if resolved_cutoff_deg < block_deg:
        raise ValueError(
            f"resolved_cutoff_deg ({resolved_cutoff_deg}) < block_deg "
            f"({block_deg}): the smooth would remove variance INSIDE a block "
            "that the block-stddev is supposed to measure — the residual "
            "construction needs cutoff >= block.")

    fine = regrid_elevation_to_latlon(
        ds, var_name=var_name, target_res_deg=float(fine_res_deg))
    elev = np.asarray(fine["elevation"].values, dtype=np.float64)
    lat = np.asarray(fine["lat"].values, dtype=np.float64)
    lon = np.asarray(fine["lon"].values, dtype=np.float64)
    n_lat, n_lon = elev.shape
    if n_lat % factor or n_lon % factor:
        raise ValueError(
            f"global {n_lat}x{n_lon} grid at {fine_res_deg} deg does not "
            f"tile into {block_deg} deg blocks (factor {factor})")

    h = np.clip(elev, 0.0, None)

    # AREA-WEIGHTED top-hat running mean, lon-periodic, lat-truncated.
    # Weighting by cos(lat) keeps the smooth consistent with the block
    # moments below (a plain boxcar would overweight poleward rows).
    w_row = np.cos(np.deg2rad(lat))[:, None]
    w2 = np.broadcast_to(w_row, h.shape)

    def _running(a, n, axis, periodic):
        csum = np.cumsum(
            np.concatenate([np.zeros_like(np.take(a, [0], axis=axis)), a],
                           axis=axis), axis=axis)
        L = a.shape[axis]
        # EXACTLY n cells per window (codex P1: the earlier 2*(n//2)+1 form
        # smoothed n+1 cells for even n, so a "4 deg" cutoff smoothed
        # 4.25 deg while the metadata claimed 4).  For even n the window is
        # off-centre by half a cell, which is a pure registration shift
        # (GLM: num and den share it, the mean stays exact).
        half_lo = n // 2
        half_hi = n - half_lo
        idx_hi = np.clip(np.arange(L) + half_hi, 0, L)
        idx_lo = np.clip(np.arange(L) - half_lo, 0, L)
        hi = np.take(csum, idx_hi, axis=axis)
        lo = np.take(csum, idx_lo, axis=axis)
        out = hi - lo
        if periodic:
            wrap_hi = np.arange(L) + half_hi - L
            wrap_lo = -(np.arange(L) - half_lo)
            add_hi = np.take(csum, np.clip(wrap_hi, 0, L), axis=axis)
            add_lo = (np.take(csum, [L], axis=axis)
                      - np.take(csum, np.clip(L - wrap_lo, 0, L), axis=axis))
            out = out + np.where(
                np.expand_dims(wrap_hi > 0, 1 - axis) if a.ndim == 2
                else (wrap_hi > 0), add_hi, 0.0)
            out = out + np.where(
                np.expand_dims(wrap_lo > 0, 1 - axis) if a.ndim == 2
                else (wrap_lo > 0), add_lo, 0.0)
        return out

    num = _running(_running(w2 * h, k, 0, False), k, 1, True)
    den = np.maximum(_running(_running(w2, k, 0, False), k, 1, True), 1e-12)
    h_smooth = num / den
    resid = h - h_smooth

    rb = resid.reshape(n_lat // factor, factor, n_lon // factor, factor)
    wb = np.broadcast_to(
        w_row.reshape(n_lat // factor, factor, 1, 1), rb.shape)
    w_sum = np.maximum(wb.sum(axis=(1, 3)), 1e-12)
    if estimator == "rms":
        sso = np.sqrt((wb * rb ** 2).sum(axis=(1, 3)) / w_sum)
    else:
        mu = (wb * rb).sum(axis=(1, 3)) / w_sum
        sso = np.sqrt(
            (wb * (rb - mu[:, None, :, None]) ** 2).sum(axis=(1, 3)) / w_sum)
    lat_b = lat.reshape(-1, factor).mean(axis=1)
    lon_b = lon.reshape(-1, factor).mean(axis=1)

    out = xr.Dataset({"SSO_STDH": (("lat", "lon"), sso)},
                     coords={"lat": lat_b, "lon": lon_b})
    out["SSO_STDH"].attrs = {
        "units": "m",
        "long_name": ("standard deviation of RESIDUAL subgrid orography "
                      "(terrain height >= 0 minus a "
                      f"{resolved_cutoff_deg:g}-deg running mean)"),
    }
    out["lat"].attrs = {"units": "degrees_north"}
    out["lon"].attrs = {"units": "degrees_east"}
    return out


def anchored_sso_scales(cell_deg: float, fine_res_deg: float):
    """``(block_deg, resolved_cutoff_deg)`` anchored to a model grid (#1712).

    block = the model cell, cutoff = ``EFFECTIVE_RESOLUTION_DX`` cells -- the
    reading the loader's scale check applies to explicit-cutoff files, so a
    file built here passes that check on the grid it was built for.  Both are
    snapped DOWN to multiples of ``fine_res_deg``: the cutoff so it never sits
    above the effective resolution, the block to the largest such multiple
    that tiles 180 x 360 deg.
    """
    import numpy as np
    from legoesm.grids.topography import EFFECTIVE_RESOLUTION_DX

    fine = float(fine_res_deg)
    n_lat, n_lon = round(180.0 / fine), round(360.0 / fine)
    n_cell = int(np.floor(float(cell_deg) / fine + 1e-9))
    for f in range(n_cell, 1, -1):
        if n_lat % f == 0 and n_lon % f == 0:
            block = f * fine
            break
    else:
        raise ValueError(
            f"fine_res_deg={fine_res_deg!r} is too coarse for a "
            f"{cell_deg!r}-deg cell: need >= 2 fine samples per block")
    cutoff = int(np.floor(EFFECTIVE_RESOLUTION_DX * float(cell_deg) / fine
                          + 1e-9)) * fine
    return block, cutoff


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
    p.add_argument("--block-deg", type=float, default=None,
                   help="block size ~ model cell size (default 2.0 deg ~ C48)")
    p.add_argument("--residual-estimator", choices=("block_stddev", "rms"),
                   default="block_stddev",
                   help="residual construction only: stddev about the block "
                        "mean (default) or RMS about zero (see "
                        "subgrid_orography_residual_stddev)")
    p.add_argument("--model-cell-deg", type=float, default=None,
                   help="anchor the decomposition to a model grid (#1712): "
                        "block = cell, resolved cutoff = the effective "
                        "resolution, both derived (see anchored_sso_scales). "
                        "For MPAS take the cell from "
                        "legoesm.grids.topography.voronoi_cell_spacing_deg. "
                        "Excludes --block-deg / --resolved-cutoff-deg.")
    p.add_argument("--resolved-cutoff-deg", type=float, default=None,
                   help="EXPLICIT scale decomposition (#1712): smooth the "
                        "terrain with a running mean of this width (the "
                        "model's EFFECTIVE resolution, ~3-4x its cell size) "
                        "and take the block stddev of the residual. Omitted "
                        "(default) = the classic construction, unchanged.")
    args = p.parse_args(argv)
    if args.model_cell_deg is not None:
        if args.block_deg is not None or args.resolved_cutoff_deg is not None:
            p.error("--model-cell-deg derives --block-deg and "
                    "--resolved-cutoff-deg; do not pass them too")
        args.block_deg, args.resolved_cutoff_deg = anchored_sso_scales(
            args.model_cell_deg, args.fine_res_deg)
    elif args.block_deg is None:
        args.block_deg = 2.0
    if (args.residual_estimator != "block_stddev"
            and args.resolved_cutoff_deg is None):
        p.error("--residual-estimator applies only to the residual "
                "construction (--resolved-cutoff-deg or --model-cell-deg)")

    with xr.open_dataset(args.input) as ds:
        if args.resolved_cutoff_deg is not None:
            out = subgrid_orography_residual_stddev(
                ds, var_name=args.elev_var,
                fine_res_deg=args.fine_res_deg, block_deg=args.block_deg,
                resolved_cutoff_deg=args.resolved_cutoff_deg,
                estimator=args.residual_estimator)
        else:
            out = subgrid_orography_stddev(
                ds, var_name=args.elev_var,
                fine_res_deg=args.fine_res_deg, block_deg=args.block_deg)
    out.attrs["source"] = str(args.input)
    # MACHINE-READABLE construction record (#1712).  The loader has to know the
    # file's scale decomposition to tell whether it double-counts orography the
    # model already resolves, and a free-text ``history`` is a pointer, not a
    # citable fact -- it used to be the only record, and the loader ignored it.
    out.attrs["block_deg"] = float(args.block_deg)
    out.attrs["fine_res_deg"] = float(args.fine_res_deg)
    out.attrs["construction"] = (
        "residual_stddev" if args.resolved_cutoff_deg is not None
        else "block_stddev")
    if args.resolved_cutoff_deg is not None:
        out.attrs["resolved_cutoff_deg"] = float(args.resolved_cutoff_deg)
        out.attrs["residual_estimator"] = args.residual_estimator
    if args.model_cell_deg is not None:
        out.attrs["model_cell_deg"] = float(args.model_cell_deg)
    out.attrs["history"] = (
        f"prep_subgrid_orography.py --fine-res-deg {args.fine_res_deg} "
        f"--block-deg {args.block_deg}"
        + (f" --resolved-cutoff-deg {args.resolved_cutoff_deg}"
           if args.resolved_cutoff_deg is not None else "")
        + (f" --residual-estimator {args.residual_estimator}"
           if args.resolved_cutoff_deg is not None else "")
        + (f" (anchored: --model-cell-deg {args.model_cell_deg})"
           if args.model_cell_deg is not None else "")
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
