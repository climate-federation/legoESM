"""Regrid a spun-up land soil state onto another model grid.

WHY.  The published LMIP soil state (PR #1624, Zenodo record 21986853) is ten
years of spun-up soil moisture, temperature and snow on the 2-degree lat-lon
grid.  The coupled AMIP runs on an MPAS mesh, and its land otherwise COLD
STARTS: soil moisture seeded from the atmosphere's own humidity, which is both
a spin-up shock and least realistic exactly where the land matters most.  The
strict restart loader (rightly) refuses a state whose column count is not the
run's own, so the state must be regridded once, offline, and the result is a
normal land IC any run can name with --land-ic-path.

HOW.  Nearest LAND neighbour on the sphere.  Soil state is a per-column
quantity, not a conserved flux, so nearest-neighbour is the right operation;
what matters is never mixing ocean placeholders into land columns.  "Land" on
the source is the spin-up's OWN definition — the surfdata's land+lake+glacier
cover on the source grid, through the same loader the spin-up used — not a
re-derived mask.  Every target column receives its nearest land source column;
target ocean columns carry coastal values the land model never reads (the
driver applies its own mask downstream).

The output carries the soil-column stamp (layer thicknesses) so the loader can
refuse it on any run whose soil column differs — the source file predates the
stamp, and adding it here is what makes the strict check able to work.

KNOWN CARRIED DEFECTS of the published state, documented in its runbook and
deliberately NOT altered here: episodic runoff spikes in ~12% of land cells
(finite values), and an unbounded ice-sheet snowpack.  Altering data in a
regridder would be a hidden choice; if those need fixing it happens upstream,
visibly.

Usage:
    regrid_land_ic.py --source data/lmip_soil_ic/restart_1985_d000h00.npz \
        --surfdata data/legoesm_surfdata_c260716.nc \
        --target-grid mpas --target-resolution 4 --out <path.npz> \
        --source-soil-column 10,3.0,2.0 \
        --source-soil-hydraulics clapp_hornberger surfdata_cosby \
            data/legoesm_surfdata_c260716.nc

    The soil-hydraulics attestation is needed while the source predates the
    stamp.  To stamp an IC already on its grid without regridding it (arrays
    unchanged), add --stamp-only and give only --source, --out and
    --source-soil-hydraulics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

_STATE_FIELDS = ("T_soil", "psi_soil", "theta_soil", "runoff_surface",
                 "runoff_subsurface", "snow_depth", "snow_age")
# Per-file fields copied unchanged.  Anything else a source carries (the optional
# water reservoirs, canopy water, TgC, carbon pools, frozen fraction) is
# per-column state this script does not remap, so such a source is refused
# rather than written with another grid's columns attached.
_COPIED_FIELDS = ("restart_version", "land_mode", "t_end_s", "n_steps_completed",
                  "metadata_json", "soil_dz", "soil_z_interface",
                  "soil_hydraulics_json")
# Belongs to the SOURCE grid's columns; a regridded state is converted in full
# on load, so it is dropped rather than attached to the wrong columns.
_DROPPED_FIELDS = ("soil_hydraulics_column_sig",)


def _cols_rad(grid) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (lat, lon) in radians, in the surfdata loader's order.

    Mirrors run_lmip_biophys.grid_latlon_rad — the function that defined the
    source state's column order — so source and target are addressed the same
    way. Duplicated here rather than imported because run scripts are not
    importable modules; a drift test pins the two (test_regrid_land_ic.py).
    """
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat, lon = np.asarray(grid.lat2d), np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat, lon = np.asarray(grid.latCell), np.asarray(grid.lonCell)
    else:
        lat, lon = np.asarray(grid.lat), np.asarray(grid.lon)
    return lat.ravel().astype(np.float64), lon.ravel().astype(np.float64)


def _source_land_mask(surfdata: str, grid, land_ncol: int) -> np.ndarray:
    """The spin-up's own land definition on its own grid.

    f_land + f_lake + f_glacier > 0 from the harmonized surfdata regridded by
    the SAME loader the spin-up used (run_lmip_biophys.py's no-mask-file
    branch), so the mask cannot disagree with the state it selects from.
    """
    from legoesm.land.boundary_data import init_land_surface_data
    from legoesm.land.config import MultiLayerLandConfig

    _, _, gsd = init_land_surface_data(
        surfdata, grid, MultiLayerLandConfig(), 0.0)

    def cover1d(a):
        a = np.asarray(a)
        return a[0] if a.ndim == 2 else a

    frac = (cover1d(gsd.f_land) + cover1d(gsd.f_lake)
            + cover1d(gsd.f_glacier)).ravel()
    if frac.size != land_ncol:
        raise SystemExit(
            f"surfdata mask has {frac.size} columns but the soil state has "
            f"{land_ncol}: wrong source grid.")
    return frac > 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="spun-up land restart .npz")
    ap.add_argument("--surfdata", default=None,
                    help="harmonized surfdata that defines the source land mask")
    ap.add_argument("--target-grid", default=None,
                    help="target grid type (e.g. mpas, latlon)")
    ap.add_argument("--target-resolution", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-soil-column", default=None,
                    help="attest the SOURCE state's soil column as "
                         "'n_layers,depth_m,growth' (e.g. '10,3.0,2.0'). The "
                         "source file predates the soil-column stamp, so its "
                         "column cannot be read off the file; the operator "
                         "must state it, it is checked against the calibrated "
                         "column, and it is recorded in provenance. Without "
                         "this a source on a same-layer-count but different-"
                         "depth column would be silently certified as "
                         "calibrated (codex).")
    ap.add_argument("--source-soil-hydraulics", nargs=3, default=None,
                    metavar=("CURVE", "SOURCE", "PARAMETER_FILE"),
                    help="attest the soil hydraulics an UNSTAMPED source was "
                         "evolved under, e.g. clapp_hornberger surfdata_cosby "
                         "data/legoesm_surfdata_c260716.nc. Refused when the "
                         "source already carries a stamp. Recorded as attested, "
                         "with the source's md5.")
    ap.add_argument("--stamp-only", action="store_true",
                    help="do not regrid: write --source unchanged plus the "
                         "attested soil-hydraulics stamp to --out (for an IC "
                         "already on its target grid).")
    args = ap.parse_args(argv)
    if not args.stamp_only:
        missing = [f for f in ("surfdata", "target_grid", "target_resolution",
                               "source_soil_column")
                   if getattr(args, f) is None]
        if missing:
            ap.error("regridding needs " + ", ".join(
                "--" + m.replace("_", "-") for m in missing))

    src = np.load(args.source, allow_pickle=False)
    extra = sorted(set(src.files) - set(_STATE_FIELDS) - set(_COPIED_FIELDS)
                   - set(_DROPPED_FIELDS))
    if extra and not args.stamp_only:
        raise SystemExit(
            f"source carries per-column fields this script does not remap: "
            f"{extra}. Refusing rather than attaching them to the wrong columns.")
    from legoesm.land.restart import file_md5, soil_hydraulics_stamp
    hyd_json = None
    if "soil_hydraulics_json" in src.files:
        if args.source_soil_hydraulics:
            raise SystemExit(
                "the source already records its soil hydraulics; "
                "--source-soil-hydraulics would overrule the file.")
    else:
        if not args.source_soil_hydraulics:
            raise SystemExit(
                "the source records no soil hydraulics; state them with "
                "--source-soil-hydraulics CURVE SOURCE PARAMETER_FILE.")
        curve, source, pfile = args.source_soil_hydraulics
        stamp = soil_hydraulics_stamp(curve, source, pfile)
        stamp.update(attested=True, attested_source_md5=file_md5(args.source))
        hyd_json = np.array(json.dumps(stamp))
    if args.stamp_only:
        if hyd_json is None:
            raise SystemExit("--stamp-only: the source is already stamped.")
        out = {k: src[k] for k in src.files}
        out["soil_hydraulics_json"] = hyd_json
        np.savez_compressed(args.out, **out)
        print(f"wrote {args.out}: {args.source} unchanged plus its "
              "soil-hydraulics stamp")
        return 0
    meta = json.loads(str(src["metadata_json"]))
    ncol_src = src["T_soil"].shape[0]

    from legoesm.grids.factory import create_grid
    src_grid = create_grid(meta["grid_type"], resolution=int(meta["resolution"]))
    src_lat, src_lon = _cols_rad(src_grid)
    if src_lat.size != ncol_src:
        raise SystemExit(
            f"reconstructed source grid has {src_lat.size} columns, the state "
            f"has {ncol_src}: the metadata does not describe this file.")

    dst_grid = create_grid(args.target_grid, resolution=args.target_resolution)
    dst_lat, dst_lon = _cols_rad(dst_grid)

    from legoesm.coupler.grid_remap import nearest_column_map
    land = _source_land_mask(args.surfdata, src_grid, ncol_src)
    n_land = int(land.sum())
    nearest = nearest_column_map(src_lat, src_lon, dst_lat, dst_lon,
                                 src_valid=land)

    out = {k: src[k] for k in src.files
           if k not in _STATE_FIELDS and k not in _DROPPED_FIELDS}
    if hyd_json is not None:
        out["soil_hydraulics_json"] = hyd_json
    for k in _STATE_FIELDS:
        out[k] = np.asarray(src[k])[nearest]

    # Refuse to write garbage rather than stamp it as an IC.
    for k in ("T_soil", "psi_soil", "theta_soil"):
        if not np.all(np.isfinite(out[k])):
            raise SystemExit(f"{k}: non-finite values after regrid; refusing.")

    # The soil-column stamp: her template's own column, computed by the model's
    # own grid builder, never typed in by hand.
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    cal_grid = biophysics_lmip_two_leaf_setup()["soil_grid"]
    n_str, d_str, g_str = args.source_soil_column.split(",")
    attested = SoilGridConfig(n_layers=int(n_str), total_depth=float(d_str),
                              growth_factor=float(g_str))
    if (attested.n_layers != cal_grid.n_layers
            or attested.total_depth != cal_grid.total_depth
            or attested.growth_factor != cal_grid.growth_factor):
        raise SystemExit(
            f"attested source column {args.source_soil_column} is not the "
            f"calibrated column ({cal_grid.n_layers},{cal_grid.total_depth},"
            f"{cal_grid.growth_factor}): refusing to stamp a state as a "
            "column it is not on.")
    if src["T_soil"].shape[1] != attested.n_layers:
        raise SystemExit(
            f"source has {src['T_soil'].shape[1]} layers but the attestation "
            f"says {attested.n_layers}.")
    out["soil_dz"] = np.asarray(make_soil_grid(cal_grid).dz, dtype=np.float64)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    meta.update({
        "grid_type": args.target_grid,
        "resolution": args.target_resolution,
        "regridded_from": pathlib.Path(args.source).name,
        "regridded_from_md5": hashlib.md5(
            pathlib.Path(args.source).read_bytes()).hexdigest(),
        "regrid_script": "scripts/data/regrid_land_ic.py",
        "regrid_git_sha": sha,
        "regrid_n_source_land": n_land,
        "source_soil_column_attested": args.source_soil_column,
    })
    out["metadata_json"] = np.str_(json.dumps(meta))

    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {dst_lat.size} columns from {n_land} source land "
          f"columns ({meta['grid_type']} {meta['resolution']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
