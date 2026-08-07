#!/usr/bin/env python
"""Repair the latitude axis of CMOR trees written by the MPAS / cubed-sphere lanes.

WHAT WAS WRONG
--------------
``DiagnosticCollector`` labelled its CMIP files with **cell centres**
(``lat = -90+dlat/2 .. 90-dlat/2``) while the unstructured output regridders
(``compute_voronoi_to_latlon_weights`` / ``compute_cubedsphere_to_latlon_
weights``) sampled the model on a **pole-inclusive** grid
(``linspace(-90, 90, nlat)``).  Row ``j`` of every regridded field therefore
holds the model's value at ``-90 + j*180/(nlat-1)`` but is labelled
``-90 + dlat/2 + j*dlat`` — a poleward displacement of ``lat/(nlat-1)``
degrees, up to ``dlat/2`` at the poles and zero at the equator.

For a field that falls off toward the poles this reads systematically LOW.
Measured on the 1979-1980 MPAS AMIP run at nlat=36: ``rsdt`` 337.27 W/m^2
against a prescribed-astronomy truth of 340.41 (-3.15 W/m^2, -0.94%),
carrying ~1.8 W/m^2 of the reported TOA imbalance.  ``areacella`` was always
correct — it is the DATA that sat in the wrong place.

The code defect is fixed upstream (regridders now take the writer's axis
explicitly, gated by ``tests/unit/test_cmip_regrid_lat_axis_consistency.py``).
This tool repairs trees ALREADY on disk, so two years of simulation need not
be re-run to correct a coordinate.

MODES
-----
``interpolate`` (default)
    Resample each field from the pole-inclusive latitudes it was sampled on
    onto the cell centres the file claims.  The published grid, ``lat_bnds``
    and ``areacella`` stay valid and standard, so every downstream
    area-weighted mean becomes correct with no change to the consumer.
    Strictly interpolation (targets lie inside [-90, 90]); no extrapolation.

``relabel``
    Leave every data value untouched and rewrite ``lat`` / ``lat_bnds`` /
    ``areacella`` to describe the pole-inclusive grid the data actually sits
    on.  Loses the standard 5-degree grid but modifies no field value.

Only the affected lanes need this.  The structured (lat-lon / Gaussian) lane
was always self-consistent — pass ``--source-grid`` deliberately and the tool
records it in the file history.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Marks a repaired file so a second pass cannot double-correct it.
FIX_ATTR = "legoesm_lat_axis_fix"

# Grid metrics + bounds are properties of the LABELLED grid, not regridded
# model fields: they must never be interpolated along latitude.
_NEVER_INTERPOLATE = frozenset({
    "areacella", "lat", "lon", "lat_bnds", "lon_bnds", "time", "time_bnds",
    "plev", "plev_bnds", "height", "depth", "depth_bnds",
})

AFFECTED_GRIDS = ("mpas", "cubed_sphere")


def cell_centre_lat(n_lat: int) -> np.ndarray:
    """The latitude axis a CMOR file CLAIMS (cell centres) [degrees]."""
    dlat = 180.0 / n_lat
    return np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, n_lat)


def pole_inclusive_lat(n_lat: int) -> np.ndarray:
    """The latitude axis the broken regridders actually SAMPLED [degrees]."""
    return np.linspace(-90.0, 90.0, n_lat)


def lat_bounds(centres: np.ndarray) -> np.ndarray:
    """Cell edges for a 1-D latitude axis, clamped to the poles."""
    mid = 0.5 * (centres[1:] + centres[:-1])
    edges = np.concatenate([[-90.0], mid, [90.0]])
    return np.stack([edges[:-1], edges[1:]], axis=1)


def cell_area(centres: np.ndarray, n_lon: int, radius: float) -> np.ndarray:
    """Exact spherical-band cell area [m^2] for a lat axis, shape (nlat, nlon)."""
    b = np.deg2rad(lat_bounds(centres))
    band = np.abs(np.sin(b[:, 1]) - np.sin(b[:, 0]))
    row = radius ** 2 * band * (2.0 * np.pi / n_lon)
    return np.broadcast_to(row[:, None], (len(centres), n_lon)).astype(np.float64)


def is_cell_centre_axis(lat: np.ndarray, atol: float = 1e-6) -> bool:
    """True when ``lat`` is the cell-centre axis for its own length."""
    return bool(np.allclose(lat, cell_centre_lat(len(lat)), atol=atol))


def correct_dataset(
    ds: xr.Dataset,
    mode: str = "interpolate",
    source_grid: str = "mpas",
    radius: float | None = None,
) -> xr.Dataset:
    """Return a corrected copy of one CMOR dataset.

    Pure function — no I/O — so it is directly testable.

    Raises
    ------
    ValueError
        On an unknown ``mode`` / ``source_grid``, a dataset with no ``lat``
        dimension, a ``lat`` axis that is not the cell-centre axis (already
        repaired, relabelled, or written by the structured lane), or a
        dataset already carrying the fix stamp.
    """
    if mode not in ("interpolate", "relabel"):
        raise ValueError(
            f"unknown mode {mode!r}; expected 'interpolate' or 'relabel'")
    if source_grid not in AFFECTED_GRIDS:
        raise ValueError(
            f"unknown source_grid {source_grid!r}; expected one of "
            f"{AFFECTED_GRIDS}. The structured lat-lon/Gaussian lane was "
            "never affected and must not be 'corrected'.")
    if FIX_ATTR in ds.attrs:
        raise ValueError(
            f"dataset already carries {FIX_ATTR}={ds.attrs[FIX_ATTR]!r}; "
            "refusing to double-correct.")
    if "lat" not in ds.dims:
        raise ValueError("dataset has no 'lat' dimension")

    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    n_lat = len(lat)
    if not is_cell_centre_axis(lat):
        raise ValueError(
            "'lat' is not the cell-centre axis this fix targets "
            f"(got {lat[0]:.4f}..{lat[-1]:.4f}, expected "
            f"{cell_centre_lat(n_lat)[0]:.4f}..{cell_centre_lat(n_lat)[-1]:.4f}). "
            "The file may already be repaired, relabelled, or from the "
            "structured lane.")

    if radius is None:
        from legoesm import constants
        radius = float(constants.R_earth)

    src = pole_inclusive_lat(n_lat)      # where the data really is
    tgt = lat                            # where the file says it is
    out = ds.copy(deep=True)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if mode == "interpolate":
        for name, da in ds.data_vars.items():
            if name in _NEVER_INTERPOLATE or "lat" not in da.dims:
                continue
            # np.interp along the lat axis; targets are strictly inside
            # [-90, 90] so this never extrapolates.
            axis = da.dims.index("lat")
            arr = np.moveaxis(np.asarray(da.values, dtype=np.float64), axis, 0)
            flat = arr.reshape(n_lat, -1)
            new = np.empty((len(tgt), flat.shape[1]), dtype=np.float64)
            for c in range(flat.shape[1]):
                new[:, c] = np.interp(tgt, src, flat[:, c])
            new = np.moveaxis(new.reshape((len(tgt),) + arr.shape[1:]), 0, axis)
            out[name] = (da.dims, new.astype(da.dtype), dict(da.attrs))
        note = (f"latitude axis repaired by {Path(__file__).name} "
                f"(mode=interpolate, source_grid={source_grid}): fields "
                "resampled from the pole-inclusive grid the "
                f"{source_grid} regridder sampled onto the cell centres the "
                "file labels. lat/lat_bnds/areacella unchanged (they were "
                "already correct).")
    else:  # relabel
        out = out.assign_coords(lat=("lat", src, dict(ds["lat"].attrs)))
        if "lat_bnds" in ds:
            out["lat_bnds"] = (ds["lat_bnds"].dims, lat_bounds(src),
                               dict(ds["lat_bnds"].attrs))
        if "areacella" in ds and "lon" in ds.dims:
            a = ds["areacella"]
            out["areacella"] = (a.dims,
                                cell_area(src, ds.sizes["lon"], radius),
                                dict(a.attrs))
        note = (f"latitude axis repaired by {Path(__file__).name} "
                f"(mode=relabel, source_grid={source_grid}): lat/lat_bnds/"
                "areacella rewritten to the pole-inclusive grid the data was "
                "sampled on. No field value modified.")

    out.attrs[FIX_ATTR] = f"{mode}/{source_grid}/{stamp}"
    out.attrs["history"] = (str(ds.attrs.get("history", "")).rstrip()
                            + f"\n{stamp}: {note}").strip()
    return out


def correct_file(path: Path, dest: Path, mode: str, source_grid: str,
                 dry_run: bool = False) -> str:
    """Correct one NetCDF file. Returns a one-line status string."""
    with xr.open_dataset(path, decode_times=False) as ds:
        ds.load()
        try:
            fixed = correct_dataset(ds, mode=mode, source_grid=source_grid)
        except ValueError as exc:
            return f"SKIP  {path.name}: {exc}"
        if dry_run:
            return f"WOULD {path.name} ({mode})"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fixed.to_netcdf(dest)
    return f"FIXED {path.name} -> {dest}"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tree", type=Path,
                   help="CMOR tree root (the directory holding Amon/, day/, fx/) "
                        "or a single .nc file.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output tree root. Default: <tree>_latfix. Never "
                        "writes in place unless --in-place is given.")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite the input tree (a .bak copy is kept).")
    p.add_argument("--mode", choices=("interpolate", "relabel"),
                   default="interpolate",
                   help="interpolate (default): move the DATA onto the "
                        "labelled cell centres. relabel: move the LABELS "
                        "onto the data.")
    p.add_argument("--source-grid", choices=AFFECTED_GRIDS, default="mpas",
                   help="Native grid the tree was written from. Only these "
                        "lanes were affected.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; write nothing.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tree: Path = args.tree
    if not tree.exists():
        print(f"error: {tree} does not exist", file=sys.stderr)
        return 2

    files = [tree] if tree.is_file() else sorted(tree.rglob("*.nc"))
    if not files:
        print(f"error: no .nc files under {tree}", file=sys.stderr)
        return 2

    if args.in_place:
        out_root = tree
    else:
        out_root = args.out or tree.with_name(tree.name + "_latfix")

    n_fixed = 0
    for f in files:
        if args.in_place:
            if not args.dry_run:
                shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
            dest = f
        else:
            rel = f.name if tree.is_file() else f.relative_to(tree)
            dest = out_root / rel
        # to_netcdf cannot overwrite the file it is reading — stage then move.
        stage = dest.with_suffix(dest.suffix + ".tmp")
        status = correct_file(f, stage, args.mode, args.source_grid,
                              dry_run=args.dry_run)
        if status.startswith("FIXED"):
            stage.replace(dest)
            n_fixed += 1
            status = f"FIXED {f.name} -> {dest}"
        print(status)

    print(f"\n{n_fixed}/{len(files)} file(s) corrected "
          f"(mode={args.mode}, source_grid={args.source_grid})"
          + (" [DRY RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
