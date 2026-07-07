#!/usr/bin/env python
"""Build the global land-carbon initial-condition ("finidat") from real maps.

Stage A of the global carbon IC map
(``docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md``): sample
the (PFT x climate) space that occurs globally, spin each sampled archetype to a
verified semi-analytic soil-carbon equilibrium, and assign each grid cell the
cover-weighted mix of its archetypes' equilibria -- so a global DifferLand run
starts *at equilibrium* instead of cold, without spinning up every cell.

This driver only *wires* the already-committed Stage-A pipeline (Tasks 2-5); it
implements no new numerics:

    reduce_climatology_to_features   (climate_features.py)  -- monthly clim -> features
    build_archetypes                 (global_init.py)       -- (PFT x climate) k-means
    equilibrate_archetypes           (global_init.py)       -- semi-analytic spin-up
    map_to_grid                      (global_init.py)       -- cover-weighted pool mix

Real-data inputs and the assembler used
---------------------------------------
* **PFT cover + soil texture** -- read via the repo's canonical surface-data
  assembler :func:`legoesm.land.global_surface_data.load_global_surface_data`
  under the ``"legoesm_surfdata"`` preset.  That single call CONSUMES the
  harmonized ``legoesm_surfdata`` NetCDF -- CLM5 cover/PFT
  (``land.surface_data.sources.clm5_surfdata.read_clm5_cover_veg``) + HWSD v2.0
  soil (``land.surface_data.sources.hwsd2``) -- built by
  ``scripts/data/build_legoesm_surfdata.py`` (``surface_data.assemble.build_v1_surfdata``)
  and CONSERVATIVELY regrids every field onto the target lat-lon grid.  We take
  the per-cell 17-PFT weights directly and derive the per-cell USDA soil-texture
  class from the regridded topsoil sand/clay via
  :func:`legoesm.land.soil_texture.usda_texture_index` (the repo's texture
  triangle).  No hand-rolled regridding.
* **Monthly climatology (T, precip, SW-down, net radiation)** -- read from a
  monthly-climatology NetCDF and regridded onto the SAME target grid with the
  repo's :func:`legoesm.grids.regridding.conservative_regrid_latlon`.  NOTE: the
  design spec points at ``tools.forcing.amip`` for climate, but that module only
  loads prescribed **SST/SIC** (no land 2 m T / precip / SW / net radiation), so
  the four land climate fields are read from a documented climatology NetCDF
  here.  The four fields must be provided by the Task-8 data-prep step (e.g. an
  ERA5 monthly-mean file); this driver never fabricates climate.

``--dry-run-synthetic`` fabricates a tiny world in-process (a handful of cells,
3 PFTs) and runs the FULL pipeline + writes both ``.npz`` files, so the driver
is exercised end-to-end WITHOUT any data files.  Every real-data loader/assembler
import is deferred to FUNCTION scope so the dry run -- and importing this module
-- never requires the surface-data packages or the data files.

Login-node policy: the equilibration JIT-compiles the coupled land+carbon model,
so run this via ``sbatch``/``srun`` on a compute node, never on the login node.

Outputs (both under ``<--output>/``)
------------------------------------
* ``global_carbon_ic.npz`` -- per-pool ``(ncell,)`` CarbonState + ``lat``/``lon``
  + ``dominant_pft`` / ``pft_present`` (the "there" PFT ids) + cover weights.
* ``archetypes.npz`` -- the ``ArchetypeTable`` + per-archetype equilibrium pools
  + the QC bundle (GPP/NPP/SOC/biomass/drift), the reusable ``(PFT,climate)->pools``
  lookup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from legoesm.land.carbon.climate_features import reduce_climatology_to_features
from legoesm.land.carbon.global_init import (
    build_archetypes,
    equilibrate_archetypes,
    map_to_grid,
)

from legoesm import constants

# --- global regular-lat-lon grid geometry (degrees; grid layout, not physics) ---
_LAT_SPAN_DEG = 180.0          # pole-to-pole latitude span [deg]
_LON_SPAN_DEG = 360.0          # full longitude span [deg]

# --- synthetic-world PFT indices (into CLM5_PFT_NAMES) for --dry-run-synthetic ---
_PFT_TROPICAL_TREE = 4         # broadleaf_evergreen_tropical (woody)
_PFT_TEMPERATE_TREE = 7        # broadleaf_deciduous_temperate (woody)
_PFT_C3_GRASS = 13             # c3_grass (herbaceous)


class GlobalCarbonInputs:
    """Per-cell driver inputs on a common ``(ncell,)`` land vector.

    A tiny plain container (not a config): the shared hand-off between the
    real-data / synthetic loaders and the archetype pipeline.
    """

    __slots__ = (
        "pft_weights", "monthly_t_k", "monthly_precip", "monthly_sw",
        "monthly_netrad", "soil_class", "land_mask", "cell_lat_deg",
        "cell_lon_deg", "n_pft",
    )

    def __init__(self, pft_weights, monthly_t_k, monthly_precip, monthly_sw,
                 monthly_netrad, soil_class, land_mask, cell_lat_deg,
                 cell_lon_deg):
        self.pft_weights = np.asarray(pft_weights, float)      # (ncell, n_pft)
        self.monthly_t_k = np.asarray(monthly_t_k, float)      # (ncell, 12)
        self.monthly_precip = np.asarray(monthly_precip, float)
        self.monthly_sw = np.asarray(monthly_sw, float)
        self.monthly_netrad = np.asarray(monthly_netrad, float)
        self.soil_class = np.asarray(soil_class, dtype=object)  # (ncell,) str
        self.land_mask = np.asarray(land_mask, bool)           # (ncell,)
        self.cell_lat_deg = np.asarray(cell_lat_deg, float)    # (ncell,)
        self.cell_lon_deg = np.asarray(cell_lon_deg, float)    # (ncell,)
        self.n_pft = int(self.pft_weights.shape[1])


# ===========================================================================
# Synthetic world (--dry-run-synthetic): no files, no assembler, no grid
# ===========================================================================
def build_synthetic_inputs() -> GlobalCarbonInputs:
    """Fabricate a tiny (6-cell, 3-PFT) world exercising the full pipeline.

    Two woody PFTs (tropical + temperate tree) and one herbaceous PFT (C3 grass),
    with pure and mixed cells, so ``equilibrate_archetypes`` sees BOTH a woody and
    a herbaceous ``(is_woody, soil_class)`` group.  A NH-phased seasonal T cycle
    gives each cell a well-defined seasonal amplitude; precip/SW/net-radiation are
    per-cell constants.  Everything is land; soil is a single texture so the two
    groups are exactly {woody, herbaceous}.
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    n_pft = int(N_PFT_CLM5)
    # (pft indices+weights, mean T [K], seasonal half-amp [K], precip [kg/m2/s],
    #  SW-down [W/m2], net radiation [W/m2]) per cell.
    cells = [
        ({_PFT_TROPICAL_TREE: 1.0},                    298.0, 2.0, 6.0e-5, 230.0, 110.0),
        ({_PFT_TEMPERATE_TREE: 1.0},                   283.0, 12.0, 2.5e-5, 190.0, 80.0),
        ({_PFT_C3_GRASS: 1.0},                         288.0, 8.0, 3.0e-5, 210.0, 90.0),
        ({_PFT_TROPICAL_TREE: 0.5, _PFT_TEMPERATE_TREE: 0.5}, 293.0, 6.0, 4.0e-5, 210.0, 95.0),
        ({_PFT_TEMPERATE_TREE: 0.6, _PFT_C3_GRASS: 0.4}, 285.0, 10.0, 3.0e-5, 195.0, 85.0),
        ({_PFT_TROPICAL_TREE: 1.0},                    300.0, 1.5, 7.0e-5, 235.0, 115.0),
    ]
    ncell = len(cells)
    months = np.arange(12)
    # NH-phased annual T cycle: reduce_climatology_to_features recovers the
    # half-amplitude from 0.5*(max-min).
    seasonal_shape = np.sin(2.0 * np.pi * months / 12.0)

    pft_weights = np.zeros((ncell, n_pft))
    monthly_t = np.zeros((ncell, 12))
    monthly_pr = np.zeros((ncell, 12))
    monthly_sw = np.zeros((ncell, 12))
    monthly_nr = np.zeros((ncell, 12))
    for c, (wmap, mean_t, amp, pr, sw, nr) in enumerate(cells):
        for p, wt in wmap.items():
            pft_weights[c, p] = wt
        monthly_t[c] = mean_t + amp * seasonal_shape
        monthly_pr[c] = pr
        monthly_sw[c] = sw
        monthly_nr[c] = nr

    soil_class = np.array(["loam"] * ncell, dtype=object)
    land_mask = np.ones(ncell, bool)
    # A small idealised lat/lon spread (metadata only; unused by the pipeline).
    cell_lat = np.linspace(-30.0, 60.0, ncell)
    cell_lon = np.linspace(0.0, 300.0, ncell)
    return GlobalCarbonInputs(
        pft_weights, monthly_t, monthly_pr, monthly_sw, monthly_nr,
        soil_class, land_mask, cell_lat, cell_lon)


# ===========================================================================
# Real-data world: canonical surface-data assembler + climatology NetCDF
# ===========================================================================
def _fill_nonfinite(a: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the field's finite mean.

    Ocean / no-data cells (dropped by ``conservative_regrid_latlon``) return NaN;
    those cells are excluded from clustering by the land mask anyway, but
    ``build_archetypes`` standardises features over ALL cells, so a NaN would
    poison the global mean/std.  Filling with the finite mean keeps the
    standardisation well-defined without touching any land-cell value.
    """
    a = np.asarray(a, float)
    bad = ~np.isfinite(a)
    if bad.any():
        fill = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
        a = np.where(bad, fill, a)
    return a


def _load_monthly_climatology(args, tgt_lat_deg, tgt_lon_deg):
    """Read a monthly-climatology NetCDF and regrid to the target grid.

    Returns ``(monthly_t_k, monthly_precip, monthly_sw, monthly_netrad)`` each
    ``(ncell, 12)`` on the target grid's flattened ``(i_lat, i_lon)`` (row-major)
    ordering -- the SAME ordering ``load_global_surface_data`` produces, so PFT /
    soil / climate share one ``(ncell,)`` vector.  Uses the repo's conservative
    lat-lon regridder; no hand-rolled regridding.

    The file must expose the four fields (variable names configurable) shaped
    ``(time=12, lat, lon)`` with 1-D ``lat``/``lon`` coordinates in degrees.  T is
    Kelvin unless ``--clim-t-in-celsius`` is set (then ``constants.T_freeze`` is
    added).
    """
    import xarray as xr
    from legoesm.grids.regridding import conservative_regrid_latlon

    tgt_lat_deg = np.asarray(tgt_lat_deg, float)
    tgt_lon_deg = np.asarray(tgt_lon_deg, float)
    n_lat, n_lon = tgt_lat_deg.size, tgt_lon_deg.size

    ds = xr.open_dataset(args.climatology)
    try:
        src_lat = np.asarray(ds[args.clim_lat].values, float)
        src_lon = np.asarray(ds[args.clim_lon].values, float)

        def _regrid(var_name):
            da = ds[var_name]
            other = [d for d in da.dims
                     if d not in (args.clim_lat, args.clim_lon)]
            if len(other) != 1:
                raise ValueError(
                    f"climatology variable {var_name!r} must have exactly one "
                    f"non-spatial (time) dimension; got dims {da.dims}.")
            # (lat, lon, time) -> conservative regrid -> (n_lat, n_lon, 12).
            arr = da.transpose(args.clim_lat, args.clim_lon, other[0]).values
            arr = np.asarray(arr, float)
            if arr.shape[-1] != 12:
                raise ValueError(
                    f"climatology variable {var_name!r} time axis has "
                    f"{arr.shape[-1]} entries; expected 12 monthly means.")
            out = conservative_regrid_latlon(
                arr, src_lat, src_lon, tgt_lat_deg, tgt_lon_deg)
            return _fill_nonfinite(out).reshape(n_lat * n_lon, 12)

        monthly_t = _regrid(args.clim_t_var)
        monthly_pr = _regrid(args.clim_precip_var)
        monthly_sw = _regrid(args.clim_sw_var)
        monthly_nr = _regrid(args.clim_netrad_var)
    finally:
        ds.close()

    if args.clim_t_in_celsius:
        monthly_t = monthly_t + constants.T_freeze
    return monthly_t, monthly_pr, monthly_sw, monthly_nr


def load_real_inputs(args) -> GlobalCarbonInputs:
    """Load + regrid the real PFT / soil / climate maps to one ``(ncell,)`` vector.

    PFT cover and soil come through the canonical
    :func:`legoesm.land.global_surface_data.load_global_surface_data`
    (``"legoesm_surfdata"`` preset) assembler; the per-cell USDA soil-texture
    class is derived from the regridded topsoil sand/clay by
    :func:`legoesm.land.soil_texture.usda_texture_index`.  Climate comes from the
    monthly-climatology NetCDF via :func:`_load_monthly_climatology`.
    """
    # Deferred (function-scope) imports: keep module import + --dry-run-synthetic
    # free of the surface-data packages / data files.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.land.global_surface_data import (
        get_surfdata_preset,
        load_global_surface_data,
    )
    from legoesm.land.soil_texture import USDA_TEXTURES, usda_texture_index

    if not args.surfdata:
        raise SystemExit(
            "--surfdata (harmonized legoesm_surfdata NetCDF) is required for the "
            "real-data build; build it with scripts/data/build_legoesm_surfdata.py "
            "or pass --dry-run-synthetic.")
    if not args.climatology:
        raise SystemExit(
            "--climatology (monthly-mean T/precip/SW/netrad NetCDF) is required "
            "for the real-data build, or pass --dry-run-synthetic.")

    res = float(args.resolution_deg)
    n_lat = int(round(_LAT_SPAN_DEG / res))
    n_lon = int(round(_LON_SPAN_DEG / res))
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

    preset = get_surfdata_preset("legoesm_surfdata")._replace(
        surf_path=args.surfdata, landuse_path=args.surfdata, veg_path=args.surfdata)
    gsd = load_global_surface_data(preset, grid)

    # Per-cell 17-PFT cover weights (first/only year) on the target grid.
    pft_weights = np.asarray(gsd.pft_frac[0], float)               # (ncell, n_pft)
    # Topsoil (layer 0) sand/clay [fraction -> percent] -> USDA texture class.
    sand_pct = np.asarray(gsd.sand_frac[:, 0], float) * 100.0
    clay_pct = np.asarray(gsd.clay_frac[:, 0], float) * 100.0
    tex_idx = np.asarray(usda_texture_index(sand_pct, clay_pct))
    soil_class = np.array([USDA_TEXTURES[int(i)] for i in tex_idx], dtype=object)
    land_mask = np.asarray(gsd.f_land[0], float) > 0.0             # (ncell,)

    # Per-cell lat/lon [deg] in the SAME row-major (i_lat, i_lon) order as gsd's
    # ncol (load_global_surface_data ravels lat2d/lon2d).
    cell_lat = np.rad2deg(np.asarray(grid.lat2d, float)).ravel()
    cell_lon = np.rad2deg(np.asarray(grid.lon2d, float)).ravel()
    tgt_lat_1d = np.rad2deg(np.asarray(grid.lat, float))
    tgt_lon_1d = np.rad2deg(np.asarray(grid.lon, float))

    monthly_t, monthly_pr, monthly_sw, monthly_nr = _load_monthly_climatology(
        args, tgt_lat_1d, tgt_lon_1d)

    return GlobalCarbonInputs(
        pft_weights, monthly_t, monthly_pr, monthly_sw, monthly_nr,
        soil_class, land_mask, cell_lat, cell_lon)


# ===========================================================================
# QC summary + outputs
# ===========================================================================
def _print_qc_summary(table, qc, n_arch) -> None:
    """Print archetype count and per-PFT SOC / biomass ranges from the QC bundle."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES

    som = np.asarray(qc["som_kgC"], float)
    bio = np.asarray(qc["biomass_kgC"], float)
    drift = np.asarray(qc["drift_frac_per_yr"], float)
    pft_id = np.asarray(table.pft_id, int)

    print(f"[global_carbon_ic] {n_arch} archetypes across "
          f"{len(np.unique(pft_id))} PFTs")
    print(f"[global_carbon_ic] SOC  range {som.min():8.2f} .. {som.max():8.2f} kgC/m2")
    print(f"[global_carbon_ic] biomass range {bio.min():8.2f} .. {bio.max():8.2f} kgC/m2")
    print(f"[global_carbon_ic] |drift| max {np.abs(drift).max():.2e} /yr "
          f"(verify-segment total-C)")
    print("[global_carbon_ic] per-PFT SOC / biomass [kgC/m2] ranges:")
    for p in np.unique(pft_id):
        sel = pft_id == p
        name = CLM5_PFT_NAMES[int(p)] if int(p) < len(CLM5_PFT_NAMES) else f"pft{p}"
        print(f"    {name:36s} n={int(sel.sum()):3d}  "
              f"SOC [{som[sel].min():7.2f}, {som[sel].max():7.2f}]  "
              f"biomass [{bio[sel].min():7.2f}, {bio[sel].max():7.2f}]")


def _write_outputs(out_dir, inputs, grid_state, table, eq, qc, w_min, res_deg):
    """Write ``global_carbon_ic.npz`` (finidat) + ``archetypes.npz`` (lookup)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finidat_path = out / "global_carbon_ic.npz"
    archetypes_path = out / "archetypes.npz"

    pft_weights = inputs.pft_weights
    pft_present = pft_weights >= w_min                              # (ncell, n_pft) "there"
    has_cover = pft_weights.max(axis=1) > 0.0
    dominant_pft = np.where(has_cover, pft_weights.argmax(axis=1), -1).astype(np.int64)

    from legoesm.land.surface_params import CLM5_PFT_NAMES
    pft_names = np.asarray(CLM5_PFT_NAMES[:inputs.n_pft], dtype="U40")

    # --- finidat: per-pool (ncell,) CarbonState + geometry + PFT ids ---
    finidat = {f: np.asarray(getattr(grid_state, f), float) for f in grid_state._fields}
    finidat.update(
        lat=inputs.cell_lat_deg, lon=inputs.cell_lon_deg,
        land_mask=inputs.land_mask,
        dominant_pft=dominant_pft, pft_present=pft_present,
        pft_weights=pft_weights, pft_names=pft_names,
        resolution_deg=np.asarray(float(res_deg)),
    )
    np.savez(finidat_path, **finidat)

    # --- archetypes: ArchetypeTable + equilibrium pools + QC lookup ---
    arch = {}
    for f in table._fields:
        vals = np.asarray(getattr(table, f))
        # soil_class is a string/object column -> store as fixed-width unicode
        # (no object pickling in the npz).
        arch[f] = vals.astype("U40") if vals.dtype == object else vals
    for f in eq._fields:
        arch[f"eq_{f}"] = np.asarray(getattr(eq, f), float)
    for k, v in qc.items():
        arch[f"qc_{k}"] = np.asarray(v, float)
    arch["pft_names"] = pft_names
    np.savez(archetypes_path, **arch)

    return finidat_path, archetypes_path


# ===========================================================================
# CLI + orchestration
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- archetype / spin-up controls (design defaults) ---
    p.add_argument("--resolution-deg", type=float, default=1.0,
                   help="target regular lat-lon resolution [deg] (real-data path)")
    p.add_argument("--k-per-pft", type=int, default=12,
                   help="climate archetypes per PFT (upper bound; k-means)")
    p.add_argument("--w-min", type=float, default=0.05,
                   help="min cover weight for a (cell,PFT) pair to be occupied")
    p.add_argument("--n-spinup", type=int, default=200,
                   help="transient spin-up years before the analytic reset")
    p.add_argument("--n-verify", type=int, default=40,
                   help="verification years after the analytic reset")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="sub-daily spin-up timestep [s]")
    p.add_argument("--n-layers", type=int, default=10, help="soil layers")
    p.add_argument("--soil-depth", type=float, default=3.0,
                   help="soil column depth [m]")
    p.add_argument("--seed", type=int, default=0, help="base k-means RNG seed")
    p.add_argument("--output", type=str, default="results/global_carbon_ic",
                   help="output DIRECTORY for the two .npz files")
    # --- real-data inputs (ignored under --dry-run-synthetic) ---
    p.add_argument("--surfdata", type=str, default="",
                   help="harmonized legoesm_surfdata NetCDF (CLM5 cover/PFT + HWSD soil)")
    p.add_argument("--climatology", type=str, default="",
                   help="monthly-climatology NetCDF (12-month T/precip/SW/netrad)")
    p.add_argument("--clim-t-var", type=str, default="tas",
                   help="climatology 2 m air-temperature variable name")
    p.add_argument("--clim-precip-var", type=str, default="pr",
                   help="climatology precipitation-rate variable name [kg/m2/s]")
    p.add_argument("--clim-sw-var", type=str, default="rsds",
                   help="climatology down-shortwave variable name [W/m2]")
    p.add_argument("--clim-netrad-var", type=str, default="netrad",
                   help="climatology surface net-radiation variable name [W/m2]")
    p.add_argument("--clim-lat", type=str, default="lat",
                   help="climatology latitude coordinate name")
    p.add_argument("--clim-lon", type=str, default="lon",
                   help="climatology longitude coordinate name")
    p.add_argument("--clim-t-in-celsius", action="store_true",
                   help="climatology T is in Celsius (add constants.T_freeze)")
    p.add_argument("--dry-run-synthetic", action="store_true",
                   help="fabricate a tiny world and run the full pipeline with no files")
    return p


def main(argv=None):
    """Build the global carbon IC and write both .npz files; returns their paths."""
    args = build_arg_parser().parse_args(argv)

    if args.dry_run_synthetic:
        print("[global_carbon_ic] --dry-run-synthetic: fabricating a tiny world")
        inputs = build_synthetic_inputs()
    else:
        inputs = load_real_inputs(args)

    ncell = inputs.pft_weights.shape[0]
    print(f"[global_carbon_ic] {ncell} cells, {inputs.n_pft} PFTs, "
          f"{int(inputs.land_mask.sum())} land cells")

    # Stage A: climate features -> archetypes.
    features = reduce_climatology_to_features(
        inputs.monthly_t_k, inputs.monthly_precip, inputs.monthly_sw,
        inputs.monthly_netrad)
    table, cell_id, cell_w = build_archetypes(
        inputs.pft_weights, features, inputs.soil_class, inputs.land_mask,
        k_per_pft=args.k_per_pft, w_min=args.w_min, seed=args.seed)
    n_arch = int(np.asarray(table.pft_id).shape[0])
    print(f"[global_carbon_ic] built {n_arch} archetypes "
          f"(k_per_pft={args.k_per_pft}, w_min={args.w_min})")

    # Stage B: spin every archetype to a verified equilibrium.
    print(f"[global_carbon_ic] equilibrating (n_spinup={args.n_spinup}, "
          f"n_verify={args.n_verify}, dt={args.dt}s, n_layers={args.n_layers}, "
          f"soil_depth={args.soil_depth}m) ...")
    eq, qc = equilibrate_archetypes(
        table, n_spinup=args.n_spinup, n_verify=args.n_verify, dt=args.dt,
        n_layers=args.n_layers, soil_depth=args.soil_depth)

    # Stage C: cover-weighted map of archetype equilibria onto the grid.
    grid_state = map_to_grid(cell_id, cell_w, eq)

    _print_qc_summary(table, qc, n_arch)

    finidat_path, archetypes_path = _write_outputs(
        args.output, inputs, grid_state, table, eq, qc, args.w_min,
        float(args.resolution_deg))
    print(f"[global_carbon_ic] wrote {finidat_path}")
    print(f"[global_carbon_ic] wrote {archetypes_path}")
    return finidat_path, archetypes_path


if __name__ == "__main__":
    main()
