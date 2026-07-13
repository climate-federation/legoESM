#!/usr/bin/env python
"""Global **biophysics-only** land driver forced by CRU-JRA reanalysis (LMIP).

This is the M3 driver of the LMIP forcing workplan (``docs/land/lmip_s3_scope.md``).
It is a copy of the ``run_lmip_smoke.py`` template with the *idealised* per-step
forcing replaced by **real CRU-JRA reanalysis** (CLM datm format), disaggregated
from 6-hourly to the model timestep and streamed through ``lax.scan`` as an
explicit per-step input (SegmentForcing doctrine).  ``run_lmip_smoke.py`` is kept
untouched as the synthetic-forcing smoke test.

Configuration matches ``run_lmip_smoke`` exactly: ``MultiLayerLandConfig`` with
prescribed seasonal LAI (CLM5 monthly climatology, one-year cycle) and
**carbon disabled** (``carbon="none"``) — energy + water + snow + soil
temperature only.  No NBP; the carbon cycle is a later workstream.

Default timestep is **1 h** (``--dt 3600``); pass ``--dt 1800`` for 30-min steps.
Default grid is **latlon ~2°** (``--grid-type latlon --resolution 90`` -> 90x180).

Usage::

    # synthetic forcing (no data needed), quick CI-scale smoke
    JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
        --surfdata data/legoesm_surfdata_v1.nc --resolution 24 --n-steps 48

    # real CRU-JRA on Derecho
    JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
        --surfdata data/legoesm_surfdata_v1.nc --resolution 90 \\
        --forcing-dir $SCRATCH/crujra --year 2023 --start-doy 196 \\
        --n-steps 240 --dt 3600 --output lmip_biophys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.land.config import MultiLayerLandConfig, LandConfig, resolve_land_config
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.canopy import CanopyConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.stomata import StomataConfig
from legoesm import constants
from legoesm.land.multilayer_land import (
    step_multilayer_land,
    step_multilayer_land_with_diagnostics,
    init_multilayer_land_state,
)
from legoesm.land.slab_land import step_land
from legoesm.land.boundary_data import init_land_surface_data, make_step_land_params_updater
from legoesm.land.forcing import stage_forcing, stage_forcing_years
from legoesm.land.output_tapes import (
    accumulate_tape_step, build_slot_indices, finalize_tape,
    init_tape_accumulator, load_output_config,
)
from legoesm.land.restart import (
    load_land_restart,
    save_land_restart,
    merge_land_restart_into_template,
)

U_MIN = 1.0
_SEC_PER_DAY = 86400.0


def make_grid(grid_type: str, resolution: int):
    """Model grid with the ModelDriver/run_amip ``--resolution N`` convention
    (latlon -> N x 2N; cubed_sphere -> CN; gaussian -> TN).  Mirrors
    ``run_lmip_smoke.make_grid`` (kept separate so that smoke driver is untouched).
    """
    from legoesm.driver.config import normalize_grid_type
    gt = normalize_grid_type(grid_type)
    if gt == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        return create_cubed_sphere(resolution)
    if gt == "gaussian":
        from legoesm.grids.gaussian import create_gaussian_grid
        return create_gaussian_grid(resolution)
    if gt == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(resolution)
    raise ValueError(f"unsupported grid_type {grid_type!r}")


def grid_latlon_rad(grid):
    """Per-column ``(lat, lon)`` in radians (the surfdata loader's order)."""
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat, lon = np.asarray(grid.lat2d), np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat, lon = np.asarray(grid.latCell), np.asarray(grid.lonCell)
    else:
        lat, lon = np.asarray(grid.lat), np.asarray(grid.lon)
    return jnp.asarray(lat.ravel()), jnp.asarray(lon.ravel())


def build_model_times(start_doy: float, dt: float, n_steps: int, *, synthetic: bool):
    """Model step times in seconds since the forcing-year start.

    Synthetic forcing starts its clock at 0, so synthetic runs start at day 0
    regardless of ``--start-doy`` (a warning is printed)."""
    t0 = 0.0 if synthetic else float(start_doy) * _SEC_PER_DAY
    return t0 + dt * np.arange(n_steps, dtype=np.float64)


def _args_from_config(cfg, cli_args) -> argparse.Namespace:
    """Build the internal argument namespace from a validated LMIPConfig.
    CLI still supplies ``--output-dir`` and (optional) ``--restart-from``
    overrides so a chained run doesn't need a config edit."""
    ns = argparse.Namespace(
        output=cli_args.output_dir,
        grid_type=cfg.grid["type"],
        resolution=int(cfg.grid["resolution"]),
        land_mode=cfg.physics["land_mode"],
        surface_scheme=cfg.physics["surface_scheme"],
        bulk=cfg.physics["bulk_scheme"],
        # Composable physics knobs (defaults = the AMIP-consistent multilayer
        # canopy).  See lmip_config.validate_config for bounds/validation.
        stomatal_model=cfg.physics.get("stomatal_model", "ball_berry"),
        stomata_enabled=bool(cfg.physics.get("stomata_enabled", False)),
        vc_max25=cfg.physics.get("vc_max25", None),
        g1=cfg.physics.get("g1", None),
        gs_max=cfg.physics.get("gs_max", None),
        snow_albedo=bool(cfg.physics.get("snow_albedo_feedback", True)),
        enable_freeze_thaw=bool(cfg.physics.get("enable_freeze_thaw", False)),
        surfdata=cfg.surfdata["path"],
        forcing_dir=cfg.forcing.get("data_dir", ""),
        prefix=cfg.forcing.get("prefix", ""),
        suffix=cfg.forcing.get("suffix", ""),
        year=int(cfg.forcing["year_start"]),
        year_end=int(cfg.forcing["year_end"]),
        start_doy=float(cfg.time["start_doy"]),
        dt=float(cfg.time["dt"]),
        n_steps=int(cfg.time["n_steps"]),
        k_neighbors=int(cfg.forcing.get("k_neighbors", 4)),
        land_mask_file=cfg.land_mask_file,
        land_frac_min=cfg.land_frac_min,
        # CLI overrides the config here for chaining ergonomics.
        restart_from=cli_args.restart_from or cfg.restart.get("from", ""),
        output_config="",                        # embedded output block is used directly
        _cfg_output_tapes=cfg.output,            # -> load_output_config indirection below
        _cfg_luc=cfg.raw.get("land_use_change") or {},   # E_LUC bookkeeping block
        _cfg_land_cover_dataset=cfg.surfdata.get("land_cover_dataset", "clm5"),
    )
    return ns


def _nonfinite_per_col(tree, ncol: int):
    """Per-column bool ``(ncol,)``: True where ANY state leaf is non-finite in
    that column.  Columns are independent in the offline land model, so this
    lets the driver revert only the failing columns (not the whole grid).
    Assumes every per-column leaf has axis 0 = the column axis; non-column
    leaves (wrong leading dim / scalars) are skipped.
    """
    flags = []
    for leaf in jax.tree_util.tree_leaves(tree):
        if getattr(leaf, "ndim", 0) < 1 or leaf.shape[0] != ncol:
            continue
        nf = ~jnp.isfinite(leaf)
        if leaf.ndim > 1:
            nf = jnp.any(nf, axis=tuple(range(1, leaf.ndim)))
        flags.append(nf)
    if not flags:
        return jnp.zeros(ncol, dtype=bool)
    return jnp.any(jnp.stack(flags, axis=0), axis=0)


def resolve_lulcc(land_cover_dataset: str, n_cover_years: int, cover_years) -> str:
    """Validate the declared land-cover dataset against the loaded surfdata and
    return a one-line transient-cover status for the run banner.

    ``land_cover_dataset`` (from ``surfdata.land_cover_dataset``) is the config's
    declared cover source: ``clm5`` = the static single-year base; any of
    ``luh2|luh3|hyde|pongratz|kk10`` = a transient anthropogenic reconstruction.

    The transient-cover engine (``make_step_land_params_updater`` +
    per-step ``year`` threading) keys off the surfdata's NUMBER OF COVER YEARS,
    not off this field — so a run that *declares* a reconstruction but is handed a
    single-year surfdata would SILENTLY apply no land-use change.  That is the
    "no silent no-op" failure the codebase forbids, so raise instead: the
    reconstruction must first be baked into a transient surfdata
    (``scripts/data/build_anthropogenic_surfdata.py --dataset <name>`` for
    HYDE/Pongratz/KK10, or ``build_luh2_transient_surfdata`` for LUH2/3) and
    ``surfdata.path`` pointed at it.  ``clm5`` + a multi-year surfdata is allowed
    (transient cover still applies) but the banner flags the provenance mismatch.
    """
    from legoesm.land.surface_data.datasets import validate_land_cover_dataset
    validate_land_cover_dataset(land_cover_dataset)          # known-name guard (raises)
    transient = n_cover_years > 1
    if land_cover_dataset != "clm5" and not transient:
        raise SystemExit(
            f"land_cover_dataset={land_cover_dataset!r} declares a transient LULCC "
            f"reconstruction, but the surfdata carries a single cover year — no "
            f"land-use change would be applied (silent no-op).  Build a transient "
            f"surfdata first (scripts/data/build_anthropogenic_surfdata.py "
            f"--dataset {land_cover_dataset}) and set surfdata.path to it, or use "
            f"land_cover_dataset=clm5 for a static-cover run.")
    if transient:
        y0, y1 = int(cover_years[0]), int(cover_years[-1])
        prov = "" if land_cover_dataset != "clm5" else " [dataset=clm5 but surfdata is transient]"
        return f"transient LULCC ON ({land_cover_dataset}, {n_cover_years} cover years {y0}-{y1}){prov}"
    return f"static cover ({land_cover_dataset})"


def _report_eluc(args, gsd) -> None:
    """Compute + report annual E_LUC when land-use-change bookkeeping is enabled.

    A post-run diagnostic: the bookkeeping is annual and independent of the
    biophysics scan, so it runs once over the transient cover series (no effect
    on the physics run).  No-op unless ``land_use_change.scheme == "bookkeeping"``.
    """
    luc_block = getattr(args, "_cfg_luc", None) or {}
    if luc_block.get("scheme", "none") != "bookkeeping":
        return
    from legoesm.land.land_use_change import (
        LandUseChangeConfig, annual_eluc_series, validate_luc_config)

    fields = LandUseChangeConfig._fields
    luc_cfg = LandUseChangeConfig(**{k: v for k, v in luc_block.items() if k in fields})
    validate_luc_config(luc_cfg)
    nyear = int(np.asarray(gsd.pft_frac).shape[0])
    if nyear <= 1:
        print("E_LUC: bookkeeping enabled but surfdata is single-year (static "
              "cover) — no land-use transitions to bookkeep.")
        return

    eluc_pgc, _ = annual_eluc_series(gsd.pft_frac, gsd.cell_area, luc_cfg)
    years = np.asarray(gsd.years).astype(int)
    eluc = np.asarray(eluc_pgc)
    print(f"E_LUC (bookkeeping): {int(years[0])}-{int(years[-1])} | "
          f"cumulative {eluc.sum():.4f} PgC | mean {eluc.mean():.4f} PgC/yr | "
          f"final year {eluc[-1]:.4f} PgC/yr")
    # Fidelity: the bookkeeping is driven by NET year-to-year cover change. For a
    # dataset that natively carries gross transitions (LUH2/LUH3) this understates
    # shifting-cultivation emissions until the gross-transition path is wired.
    from legoesm.land.surface_data.datasets import has_gross_transitions
    dataset = getattr(args, "_cfg_land_cover_dataset", "clm5")
    if has_gross_transitions(dataset):
        print(f"  NOTE: {dataset} carries native gross transitions, but E_LUC here "
              f"uses NET cover change — gross-transition emissions are understated.")
    out = Path(f"{args.output}.eluc_annual.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([years, eluc]),
               header="year  E_LUC_PgC_per_yr", fmt=["%d", "%.6e"])
    print(f"  wrote {out}")


def run(args) -> int:
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args.grid_type, args.resolution)
    lat_rad, lon_rad = grid_latlon_rad(grid)
    ncol = lat_rad.shape[0]

    # Estimate the PEAK forcing pytree memory (ONE year at a time — chunked
    # scan runs per-year).  Fail fast with a concrete size + suggested fix if
    # a single year's forcing wouldn't fit in a soft budget (24 GiB —
    # comfortable on a 40 GB A100).  1 year of hourly at 2° ≈ 16 GiB float64.
    _ATM_TO_SURFACE_N_FIELDS = 14                # T, q, p, wind, sw, lw, precip, snow, cos_z, ρ, CO2, ...
    _BYTES_PER_ELEM = 8                           # float64
    _STEPS_PER_YEAR_MAX = int(365.0 * 86400.0 / max(float(args.dt), 1.0))
    peak_year_steps = min(int(args.n_steps), _STEPS_PER_YEAR_MAX)
    est_bytes = peak_year_steps * ncol * _ATM_TO_SURFACE_N_FIELDS * _BYTES_PER_ELEM
    est_gib = est_bytes / (1024 ** 3)
    _FORCING_BUDGET_GIB = 24.0
    if est_gib > _FORCING_BUDGET_GIB:
        print(f"ERROR: per-year forcing pytree ({est_gib:.1f} GiB) exceeds the "
              f"soft budget ({_FORCING_BUDGET_GIB} GiB).", file=sys.stderr)
        print(f"  peak_year_steps={peak_year_steps} x ncol={ncol} x "
              f"~{_ATM_TO_SURFACE_N_FIELDS} fields x {_BYTES_PER_ELEM} B\n",
              file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  - Coarser grid (e.g. biophysics/smoke_4deg template)",
              file=sys.stderr)
        print("  - Larger dt (fewer steps per year)", file=sys.stderr)
        print("  - CPU with more host RAM: qsub -l select=…:mem=128GB",
              file=sys.stderr)
        return 2

    # --- land config: SAME as run_lmip_smoke (carbon stays at its default
    #     "none"); --surface-scheme picks two-leaf canopy or SimpleSEB. ---
    # Explicit dispatch — raise on an unknown selector rather than silently
    # defaulting (dispatch hardening; validate_config also restricts these, so
    # this is the defense that catches a NEW scheme wired in without touching
    # the driver).
    if args.surface_scheme == "two_leaf_canopy":
        # The two-leaf canopy carries its OWN mechanistic Ball-Berry/Medlyn
        # stomata (selected by CanopyConfig.stomatal_model); per-column Vcmax25
        # comes from the surfdata PFT map.
        surf = CanopyConfig(max_iters=50, tol=1e-2, stomatal_model=args.stomatal_model)
    elif args.surface_scheme == "simple_seb":
        surf = SimpleSEBConfig()
    else:
        raise ValueError(
            f"unknown surface_scheme {args.surface_scheme!r} "
            "(expected 'two_leaf_canopy' or 'simple_seb')")
    # StomataConfig governs the SimpleSEB interactive-stomata path (the Jarvis
    # beta amip_sota disabled for over-transpiration, #730); the two-leaf canopy
    # ignores it (it has its own leaf conductance).  Only override a scalar the
    # user actually set (None -> land default / per-PFT surfdata value).
    _stom = {"enabled": bool(args.stomata_enabled), "stomata_model": args.stomatal_model}
    if args.vc_max25 is not None:
        _stom["Vc_max25"] = float(args.vc_max25)
    if args.gs_max is not None:
        _stom["gs_max"] = float(args.gs_max)
    if args.g1 is not None:
        _stom["g1_bb" if args.stomatal_model == "ball_berry" else "g1_med"] = float(args.g1)
    stomata = StomataConfig(**_stom)
    if args.land_mode == "multilayer":
        # NOTE: the soil-layer count is set by the surfdata loader's remap grid
        # (init_land_surface_data -> gsd), so the model SoilGrid must match it;
        # we use the loader default (8-layer Richards).  Exact AMIP parity
        # (10-layer/3 m via clm_multilayer_setup) is a documented follow-up.
        base_cfg = MultiLayerLandConfig(
            surface_scheme=surf, soil_grid=SoilGridConfig(),
            bulk_scheme=args.bulk, snow_albedo_feedback=bool(args.snow_albedo),
            stomata=stomata,
            # Soil-water latent zero-curtain: off is bit-identical sensible-only
            # heat; on stabilises freezing boreal/Arctic columns.  Preserved
            # through init_land_surface_data (which only _replace()s hydraulics).
            thermal=SoilThermalConfig(enable_freeze_thaw=bool(args.enable_freeze_thaw)))
        # Diagnostics variant so the scan can tape GPP (the canopy's surface_out.gpp
        # is dropped from the TileResponse when carbon is off).  Same _impl as
        # step_multilayer_land — the 4th return (SurfaceFluxOutput) is already
        # computed, so this adds no cost.
        step_fn = step_multilayer_land_with_diagnostics
    elif args.land_mode == "slab":
        base_cfg = LandConfig(surface_scheme=surf)
        step_fn = step_land
    else:
        raise ValueError(
            f"unknown land_mode {args.land_mode!r} (expected 'multilayer' or 'slab')")
    base_cfg = resolve_land_config(args.land_mode, base_cfg)
    is_multilayer = (args.land_mode == "multilayer")

    config, _params_nominal, gsd = init_land_surface_data(
        args.surfdata, grid, base_cfg, args.start_doy)

    # --- CRU-JRA forcing: load -> regrid -> disaggregate to the model steps. ---
    # Year range: --year-end defaults to --year (single-year, backward-compat).
    # A larger --year-end triggers multi-year contiguous forcing.
    year_start = int(args.year)
    year_end = int(args.year_end) if args.year_end is not None else year_start
    if year_end < year_start:
        raise SystemExit(f"--year-end ({year_end}) < --year ({year_start})")
    multi_year = year_end > year_start

    # Per-year forcing-file check.  When a data_dir is set, EVERY year in the
    # range must have its Solr file staged.  Silent fallback to synthetic for
    # a missing intermediate year would load a fake full-year climatology and
    # blow up device memory (~30 GB on GPU for one year of 6h global fake
    # forcing).  Fail fast with a clean list + the exact fix command.
    if args.forcing_dir:
        missing = []
        for y in range(year_start, year_end + 1):
            p = Path(args.forcing_dir) / f"{args.prefix}.Solr.{y}{args.suffix}.nc"
            if not p.exists():
                missing.append((y, p))
        if missing:
            print("ERROR: CRU-JRA forcing not staged for the requested years:",
                  file=sys.stderr)
            for y, p in missing:
                print(f"  year {y}: missing {p}", file=sys.stderr)
            print("\nStage them first from a login node (repo root):",
                  file=sys.stderr)
            for y, _ in missing:
                print(f"  ./scripts/data/download_lmip_data.sh --year {y}",
                      file=sys.stderr)
            return 2
        synthetic = False
    else:
        synthetic = True
        if args.start_doy != 0.0:
            print("(synthetic forcing starts at day 0; --start-doy ignored)")

    dt = float(args.dt)
    model_times_s = build_model_times(args.start_doy, dt, args.n_steps, synthetic=synthetic)
    forcing_desc = ("synthetic" if synthetic
                    else f"CRU-JRA {year_start}"
                    + (f"-{year_end}" if multi_year else ""))
    _ft = bool(getattr(getattr(config, "thermal", None), "enable_freeze_thaw", False))
    # --- LULCC option: validate the declared land-cover dataset against the loaded
    # surfdata (fail fast on a declared-reconstruction / static-surfdata mismatch),
    # and validate the E_LUC bookkeeping config UP FRONT so a bad knob fails before
    # the run instead of in the post-run _report_eluc. ---
    _lc_dataset = getattr(args, "_cfg_land_cover_dataset", "clm5")
    _n_cover_years = int(np.asarray(gsd.pft_frac).shape[0])
    _lulcc_status = resolve_lulcc(_lc_dataset, _n_cover_years, np.asarray(gsd.years))
    _luc_block = getattr(args, "_cfg_luc", None) or {}
    _eluc_on = _luc_block.get("scheme", "none") == "bookkeeping"
    if _eluc_on:
        from legoesm.land.land_use_change import LandUseChangeConfig, validate_luc_config
        _fields = LandUseChangeConfig._fields
        validate_luc_config(
            LandUseChangeConfig(**{k: v for k, v in _luc_block.items() if k in _fields}))
    print(f"grid={args.grid_type} | {ncol} columns | surface={args.surface_scheme} | "
          f"carbon={config.carbon.scheme} | freeze_thaw={'on' if _ft else 'off'} | "
          f"cover={_lulcc_status} | E_LUC={'on' if _eluc_on else 'off'} | "
          f"dt={dt:.0f}s | n_steps={args.n_steps} | forcing={forcing_desc}")

    # Precompute per-year masks over model_times_s.  We stage forcing +
    # scan ONE YEAR AT A TIME in a Python loop below, so peak device memory
    # is one year's forcing pytree — not all N years concatenated (which
    # OOM'd the A100 at 5-yr × 2° hourly).
    tq = np.asarray(model_times_s, dtype=np.float64)
    _SEC_PER_YEAR = _SEC_PER_DAY * 365.0                    # noleap
    year_masks = []                                          # list[(year, mask)]
    for k, y in enumerate(range(year_start, year_end + 1)):
        t_lo, t_hi = k * _SEC_PER_YEAR, (k + 1) * _SEC_PER_YEAR
        mask = (tq >= t_lo) & (tq < t_hi)
        if mask.any():
            year_masks.append((y, mask))
    if not year_masks:
        raise SystemExit(
            f"no model times fall within years [{year_start}, {year_end}]")

    # T0 seed for cold-start soil temperature: stage just the first year's
    # first step (one AtmToSurface slice, ~100 KB) and extract T_lowest[0].
    # --restart-from overrides this anyway.
    allow_syn = synthetic
    _first_year, _first_mask = year_masks[0]
    _tq0_local = tq[_first_mask][:1]                          # already relative to year_start
    _seed_forcing = stage_forcing(
        lat_rad, lon_rad, _tq0_local,
        year=_first_year, data_dir=(None if synthetic else args.forcing_dir),
        prefix=args.prefix, suffix=args.suffix,
        k_neighbors=args.k_neighbors, allow_synthetic=allow_syn)
    T0 = _seed_forcing.T_lowest[0]
    del _seed_forcing                                          # free before real staging
    if args.land_mode == "slab":
        from legoesm.core.field import Field
        from legoesm.land.state import LandState
        z = lambda: jnp.zeros(ncol)
        state = LandState(
            T_soil=Field(T0, name="T_soil", units="K"),
            W_bucket=Field(jnp.full(ncol, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(z(), name="snow_depth", units="kg/m2"),
            snow_age=Field(z(), name="snow_age", units="s"),
            runoff=z(),
        )
    else:
        if args.restart_from:
            # Warm start from a prior end-state — bypass the cold-init T_soil
            # broadcast so the loaded profile survives verbatim.  The restart
            # round-trips only core prognostic fields; graft them onto a fresh
            # template so the OPTIONAL structural fields (surface_water,
            # snow_bands, ice_bands, canopy_state, …) match the lax.scan carry
            # structure (a bare loaded state has them at their None defaults,
            # which mismatches the array-valued step output — see
            # merge_land_restart_into_template).
            _loaded, restart_meta = load_land_restart(
                args.restart_from,
                expected_land_mode="multilayer",
                expected_ncol=ncol,
                expected_n_layers=config.soil_grid.n_layers)
            _template = init_multilayer_land_state(ncol, config, T_init=288.0)
            state = merge_land_restart_into_template(_loaded, _template)
            print(f"restart: loaded state from {args.restart_from} "
                  f"(t_end_s={restart_meta['t_end_s']:.1f}, "
                  f"steps_completed={restart_meta['n_steps_completed']})")
        else:
            state = init_multilayer_land_state(ncol, config, T_init=288.0)
            state = state._replace(T_soil=jnp.broadcast_to(T0[:, None], state.T_soil.shape))

    update_land_params = make_step_land_params_updater(gsd, config.surface_scheme)

    # ----- output tapes (CLM-style history streams; see output_tapes.py) -----
    if getattr(args, "_cfg_output_tapes", None) is not None:
        # Config-driven path: build TapeSpec list from the embedded output block.
        from legoesm.land.output_tapes import TapeSpec
        tape_specs = [TapeSpec(name=t["name"], freq=t["freq"],
                                average=t.get("average", "mean"),
                                vars=tuple(t["vars"]))
                      for t in args._cfg_output_tapes["tapes"]]
    else:
        tape_specs = load_output_config(args.output_config or None)
    tape_slots = {}                                # (slot_idx, n_slots, slot_times) per tape
    tape_accums = {}
    for tape in tape_specs:
        slot_idx, n_slots, slot_times = build_slot_indices(model_times_s, tape.freq)
        tape_slots[tape.name] = (jnp.asarray(slot_idx), n_slots, slot_times)
        tape_accums[tape.name] = init_tape_accumulator(tape, n_slots, ncol)
    print("tapes: " + " | ".join(
        f"{t.name}(freq={t.freq},avg={t.average},vars={len(t.vars)})" for t in tape_specs))

    _ZEROS = jnp.zeros(ncol)                       # slab-mode placeholder for multilayer-only vars

    # ----- scan body: (state, tape_accums, revert_count) -> next. -----
    def _step_body(carry, xs):
        state, accums, revert_count = carry
        forcing_t, doy_t, year_t, per_tape_slot = xs
        theta_top_t = (state.theta_soil[:, 0] if is_multilayer else jnp.full(ncol, 0.2))
        land_params_t, lai_diag = update_land_params(theta_top_t, doy_t, year_t)
        # Multilayer uses the diagnostics variant (4-tuple) so surface_out.gpp is
        # reachable; slab keeps the 3-tuple.  ``is_multilayer`` is static.
        if is_multilayer:
            new_state, resp, _, surf_out = step_fn(
                state, forcing_t, config, U_MIN, dt,
                lat=lat_rad, land_params=land_params_t, doy=doy_t)
        else:
            new_state, resp, _ = step_fn(
                state, forcing_t, config, U_MIN, dt,
                lat=lat_rad, land_params=land_params_t, doy=doy_t)
            surf_out = None
        # GPP [gC/m2/day]: the canopy's gross primary production (surface_out.gpp,
        # gC/m2/s).  None for schemes that don't produce it (simple_seb biophysics)
        # -> reported as 0.  ET [mm/day]: latent-heat-equivalent evapotranspiration
        # lhflx / L_v (positive = surface -> atmosphere; over snow this is the
        # sublimation-equivalent water flux).
        if surf_out is not None and surf_out.gpp is not None:
            gpp_day = surf_out.gpp * _SEC_PER_DAY
        else:
            gpp_day = _ZEROS
        et_mmday = resp.lhflx / constants.L_v * _SEC_PER_DAY
        # Transpiration + soil-evaporation split [mm/day]: the canopy's per-component
        # latent (LE_canopy = sunlit+shaded leaf transpiration, LE_soil = ground
        # evaporation), converted to a water flux.  None for simple_seb (single skin,
        # no canopy/soil partition) -> 0.  transp + soil_evap ~ ET (modulo snow
        # sublimation).  Net radiation [W/m2], positive INTO the surface = absorbed
        # SW + net LW = sw_down*(1-albedo) + lw_down - lw_up (scheme-agnostic; the
        # reported albedo/lw_up already reflect the canopy RT).
        if surf_out is not None and surf_out.LE_canopy is not None:
            transp = surf_out.LE_canopy / constants.L_v * _SEC_PER_DAY
            soil_evap = surf_out.LE_soil / constants.L_v * _SEC_PER_DAY
        else:
            transp = _ZEROS
            soil_evap = _ZEROS
        rnet = (forcing_t.sw_down * (1.0 - resp.albedo)
                + forcing_t.lw_down - resp.lw_up)
        # Available variables per step -> selected by each tape's spec.
        values = {
            "T_sfc": resp.T_sfc, "albedo": resp.albedo,
            "shflx": resp.shflx, "lhflx": resp.lhflx,
            "runoff": resp.freshwater_flux,
            "precip": forcing_t.precip_total,
            "LAI": lai_diag,
            "GPP": gpp_day,
            "ET": et_mmday,
            "transp": transp,
            "soil_evap": soil_evap,
            "Rnet": rnet,
        }
        if is_multilayer:
            values["T_soil_top"] = new_state.T_soil[:, 0]
            values["theta_soil_top"] = new_state.theta_soil[:, 0]
            values["snow_depth"] = new_state.snow_depth
        else:
            values["T_soil_top"] = _ZEROS
            values["theta_soil_top"] = _ZEROS
            values["snow_depth"] = _ZEROS
        # --- atomic per-column NaN-revert guard (ported from run_ec_site) ---
        # Columns are independent, so if a column's state update goes non-finite,
        # revert THAT column to its previous state (jnp.where): a diverging boreal
        # cell can no longer poison its own future steps (it holds a finite state
        # and may recover from a transient), and it never corrupts the run-level
        # PASS/FAIL.  The reverted step's diagnostics are untrustworthy, so mask
        # them to NaN; ``reverted`` (0/1) is tape-able as a per-cell failure-rate
        # map and ``revert_count`` accumulates a per-cell total for the summary.
        reverted = _nonfinite_per_col(new_state, ncol)          # (ncol,) bool
        def _revert(n, o):
            if getattr(n, "ndim", 0) < 1 or n.shape[0] != ncol:
                return n
            m = reverted.reshape((ncol,) + (1,) * (n.ndim - 1))
            return jnp.where(m, o, n)
        new_state = jax.tree_util.tree_map(_revert, new_state, state)
        revert_count = revert_count + reverted.astype(revert_count.dtype)
        values = {k: jnp.where(reverted, jnp.nan, v) for k, v in values.items()}
        values["reverted"] = reverted.astype(jnp.float64)
        new_accums = {}
        for tape in tape_specs:                    # unrolled at trace time
            new_accums[tape.name] = accumulate_tape_step(
                accums[tape.name], tape, per_tape_slot[tape.name],
                {v: values[v] for v in tape.vars})
        return (new_state, new_accums, revert_count), None

    # ----- CHUNKED SCAN: stage forcing + lax.scan one year at a time.  --------
    # Tape accumulators are sized for the WHOLE run and threaded across chunks;
    # slot indices are already global (build_slot_indices returns indices into
    # the whole run's slot count), so slicing by year_mask keeps the accumulation
    # naturally aligned.  Peak forcing memory = ONE year (not N × N years).
    # JAX caches the scan compilation on shape+function, so year 2+ reuse the
    # compiled artifact from year 1 (partial-year edges may recompile once).
    # Global slot indices — computed once, sliced per year.
    slot_idx_global = {name: t[0] for name, t in tape_slots.items()}
    total_steps = int(sum(m.sum() for _, m in year_masks))
    # Every requested step MUST fall inside a staged forcing year.  A step whose
    # time lands past the last year's window would be silently DROPPED (fewer
    # steps integrated than asked), and the auto-saved restart still records
    # n_steps_completed=n_steps / t_end from the FULL clock -> a chained warm
    # start would resume at the wrong model time.  Fail fast instead.
    if total_steps != int(args.n_steps):
        raise SystemExit(
            f"{int(args.n_steps) - total_steps} of {args.n_steps} model steps fall "
            f"outside the forcing years [{year_start}, {year_end}] "
            f"({total_steps} would be integrated).  Extend --year-end, or reduce "
            f"--n-steps / --dt so the run fits the staged years "
            f"(1 noleap year = {int(_SEC_PER_YEAR / dt)} steps at dt={dt:.0f}s).")
    print(f"stepping {total_steps} timestep(s) across {len(year_masks)} year chunk(s) "
          f"(lax.scan per year) ...")

    # ---- output setup + per-year flush helpers ----
    # A multi-year run flushes each COMPLETED year to its own annual NetCDF
    # (lmip_biophys.<tape>.<year>.nc) + a resumable restart right after that
    # year's scan, so a wall-clock timeout keeps every finished year instead of
    # losing the whole run.  The final combined file (lmip_biophys.<tape>.nc) is
    # still written at the end for a run that finishes.
    lat_deg = np.rad2deg(np.asarray(lat_rad)); lon_deg = np.rad2deg(np.asarray(lon_rad))
    is_latlon = args.grid_type == "latlon"
    if is_latlon:
        nlat, nlon = args.resolution, 2 * args.resolution
        assert nlat * nlon == ncol, f"latlon reshape mismatch: {nlat}*{nlon} != {ncol}"
        lat_1d = lat_deg.reshape(nlat, nlon)[:, 0]
        lon_1d = lon_deg.reshape(nlat, nlon)[0, :]

    def _cover1d(a):
        a = np.asarray(a)
        return a[0] if a.ndim == 2 else a
    if args.land_mask_file:
        from legoesm.grids.topography import load_land_fraction
        land_fraction = np.asarray(load_land_fraction(grid, args.land_mask_file)).ravel()
    else:
        land_fraction = (_cover1d(gsd.f_land) + _cover1d(gsd.f_lake)
                         + _cover1d(gsd.f_glacier))
    land = land_fraction >= args.land_frac_min

    def _flush_tapes(accums, slot_ids_by_tape, label):
        """Write each tape's selected slots to ``lmip_biophys.<tape>[.<label>].nc``.
        ``label`` empty -> the combined whole-run file; a year string -> that
        year's annual file.  Latlon uses the (time, lat, lon) rectangular layout;
        other grids fall back to (time, ncol)."""
        import xarray as xr
        masked = lambda a: np.where(land, np.asarray(a, np.float64), np.nan)
        dims = ("time", "lat", "lon") if is_latlon else ("time", "ncol")
        for tape in tape_specs:
            ids = np.asarray(slot_ids_by_tape[tape.name])
            if ids.size == 0:
                continue
            finalized = finalize_tape(accums[tape.name], tape)   # var -> (n_slots, ncol)
            _, _, slot_times = tape_slots[tape.name]
            st = np.asarray(slot_times)[ids]

            def pack(arr):
                arr2 = np.stack([masked(arr[i]) for i in ids])
                return arr2.reshape(ids.size, nlat, nlon) if is_latlon else arr2

            data_vars = {v: (dims, pack(finalized[v])) for v in tape.vars}
            coords = {"time": (("time",), st / _SEC_PER_DAY)}
            if is_latlon:
                coords.update({"lat": (("lat",), lat_1d), "lon": (("lon",), lon_1d)})
            else:
                coords.update({"lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)})
            attrs = {
                "forcing": "synthetic" if synthetic else f"CRU-JRA {year_start}"
                           + (f"-{year_end}" if year_end > year_start else ""),
                "dt": dt, "start_doy": args.start_doy,
                "grid_type": args.grid_type, "surface_scheme": args.surface_scheme,
                "carbon": config.carbon.scheme,
                "tape_name": tape.name, "tape_freq": tape.freq, "tape_average": tape.average,
                "time_units": "days since year_start Jan 1 (noleap)",
                "year_label": label or "all",
            }
            ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
            suffix = f".{label}" if label else ""
            nc = out_dir / f"lmip_biophys.{tape.name}{suffix}.nc"
            ds.to_netcdf(nc)
            print(f"wrote {nc} ({ids.size} {tape.freq} slots, "
                  f"layout={'lat,lon' if is_latlon else 'ncol'})")

    def _save_restart(cur_state, t_end_s, n_completed):
        """Save a chained-run seed named by the model time it represents
        (restart_<YEAR>_d<DDD>h<HH>.npz, noleap).  Multilayer only."""
        if not is_multilayer:
            return
        try:
            days_since_start = t_end_s / _SEC_PER_DAY
            year_offset = int(days_since_start // 365)
            year_final = year_start + year_offset
            doy_float = days_since_start - year_offset * 365.0
            doy_int = int(doy_float)
            hour_of_day = int(round((doy_float - doy_int) * 24.0)) % 24
            restart_name = f"restart_{year_final:04d}_d{doy_int:03d}h{hour_of_day:02d}.npz"
            restart_meta = {
                "grid_type": args.grid_type, "resolution": args.resolution,
                "surface_scheme": args.surface_scheme, "bulk_scheme": args.bulk,
                # Sourced from the CONSTRUCTED config (not args) so provenance
                # reflects the physics actually run.
                "enable_freeze_thaw": bool(config.thermal.enable_freeze_thaw),
                "year": year_start, "year_end": year_end, "dt": dt,
                "n_steps": args.n_steps, "start_doy": args.start_doy,
                "forcing": ("synthetic" if synthetic else "CRU-JRA"),
                "year_final": year_final, "doy_final": doy_int, "hour_final": hour_of_day,
            }
            rp = save_land_restart(
                out_dir / restart_name, cur_state,
                land_mode="multilayer", t_end_s=t_end_s,
                n_steps_completed=n_completed, metadata=restart_meta)
            print(f"wrote {rp}")
        except Exception as e:  # noqa: BLE001
            print(f"(restart write skipped: {e})")

    steps_done = 0
    revert_count = jnp.zeros(ncol)                  # per-cell NaN-revert tally
    for k, (year, mask) in enumerate(year_masks):
        # Year-local model times: the year's forcing clock resets to 0 at Jan 1.
        tq_year = tq[mask]
        n_step_year = tq_year.size
        tq_local = tq_year - k * _SEC_PER_YEAR
        forcing_year = stage_forcing(
            lat_rad, lon_rad, tq_local,
            year=year, data_dir=(None if synthetic else args.forcing_dir),
            prefix=args.prefix, suffix=args.suffix,
            k_neighbors=args.k_neighbors, allow_synthetic=allow_syn)
        doy_year = jnp.asarray(tq_year / _SEC_PER_DAY)
        # Transient cover: broadcast this chunk's calendar year across its steps as
        # a TRACED scan input (not a Python constant baked into the closure) so
        # interp_annual selects the right LUH2 slice WITHOUT recompiling the scan
        # each year (SegmentForcing doctrine).
        year_xs = jnp.full(n_step_year, float(year))
        # Slice each tape's GLOBAL slot indices to just this year's steps.
        slot_year_xs = {name: idx[mask] for name, idx in slot_idx_global.items()}
        print(f"  year {year} ({n_step_year} steps) ...")
        (state, tape_accums, revert_count), _ = jax.lax.scan(
            _step_body, (state, tape_accums, revert_count),
            (forcing_year, doy_year, year_xs, slot_year_xs))
        del forcing_year, doy_year, year_xs, slot_year_xs      # free before next year
        steps_done += n_step_year
        # Flush THIS year's completed tape slots + a resumable restart, so a
        # wall-clock timeout keeps every finished year (annual output).  Only for
        # multi-year runs; a single-year run gets the combined file below.
        if multi_year:
            try:
                year_ids = {
                    t.name: np.unique(np.asarray(slot_idx_global[t.name])[np.asarray(mask)])
                    for t in tape_specs}
                _flush_tapes(tape_accums, year_ids, f"{year:04d}")
                _save_restart(state, float(tq_year[-1] + dt), steps_done)
            except Exception as e:  # noqa: BLE001
                print(f"(year {year} annual flush skipped: {e})")

    # --- E_LUC land-use-change bookkeeping (post-run annual diagnostic). ---
    _report_eluc(args, gsd)

    # --- PASS/FAIL: final soil top-layer T must be finite over land (uses the
    #     ``land`` mask computed before the year loop). ---
    # Multilayer T_soil is a raw (ncol, n_layers) array; slab T_soil is a 1-D
    # Field (.data holds the (ncol,) array).  Validate the REAL slab state, not a
    # zeros placeholder — otherwise a slab NaN blow-up would silently PASS.
    T_final = np.asarray(
        state.T_soil[:, 0] if is_multilayer else state.T_soil.data).ravel()
    nan_land = int(np.isnan(T_final[land]).sum())
    finite = np.all(np.isfinite(T_final[land]))
    status = "PASS" if (nan_land == 0 and finite) else "FAIL"
    print(f"land cells: {int(land.sum())} | NaN final T_soil_top over land: {nan_land} -> {status}")
    if is_multilayer:
        def rng(a):
            return f"[{np.nanmin(a[land]):.2f}, {np.nanmax(a[land]):.2f}]"
        print(f"  final T_soil_top {rng(T_final)} K | "
              f"theta_top {rng(np.asarray(state.theta_soil[:, 0]))} | "
              f"snow_depth {rng(np.asarray(state.snow_depth))} kg/m2")

    # --- NaN-revert diagnostics: how many land cells needed the atomic revert
    #     guard, and where.  A non-zero count = the physics diverged on those cells
    #     (boreal/Arctic; see docs/land/boreal_nan_diagnosis_plan.md).  The run
    #     still finishes with a finite state instead of NaN-poisoning. ---
    rc = np.asarray(revert_count)
    n_reverted_cells = int((rc[land] > 0).sum())
    if n_reverted_cells:
        print(f"NaN-revert guard: {n_reverted_cells}/{int(land.sum())} land cells "
              f"reverted >=1 step | {int(rc[land].sum())} total cell-steps | "
              f"worst cell {int(rc[land].max())} steps (fluxes masked; tape var "
              f"'reverted' = per-cell revert fraction).")
    else:
        print("NaN-revert guard: no land cell required a revert (fully finite).")
    try:
        import xarray as xr
        rc_map = np.where(land, rc, np.nan)
        if is_latlon:
            rc_da = xr.DataArray(rc_map.reshape(nlat, nlon), dims=("lat", "lon"),
                                 coords={"lat": lat_1d, "lon": lon_1d})
        else:
            rc_da = xr.DataArray(rc_map, dims=("ncol",), coords={
                "lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)})
        xr.Dataset({"revert_count": rc_da},
                   attrs={"desc": "per-cell count of NaN-revert steps"}).to_netcdf(
            out_dir / "lmip_biophys.reverts.nc")
        print(f"wrote {out_dir / 'lmip_biophys.reverts.nc'}")
    except Exception as e:  # noqa: BLE001
        print(f"(revert-map write skipped: {e})")

    # --- final COMBINED whole-run NetCDF (lmip_biophys.<tape>.nc) + end-of-run
    #     restart.  A multi-year run already flushed per-year annual files +
    #     restarts inside the loop; this combined file is the convenience output
    #     for a finished run (and the sole output for a single-year run). ---
    try:
        all_ids = {t.name: np.arange(tape_slots[t.name][1]) for t in tape_specs}
        _flush_tapes(tape_accums, all_ids, "")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")
    _save_restart(state, float(model_times_s[-1] + dt), args.n_steps)

    return 0 if status == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    """Minimal CLI: --config points at a fully-resolved YAML config (typically
    generated by scripts/run/init_experiment.py).  --output-dir + optional
    --restart-from are the only knobs kept outside the YAML — they change every
    run and belong on the command line for chaining ergonomics."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="resolved YAML config (see templates/land/ + "
                         "scripts/run/init_experiment.py)")
    ap.add_argument("--output-dir", required=True,
                    help="experiment output directory (auto-created)")
    ap.add_argument("--restart-from", default="",
                    help="override the config's restart.from field — the most "
                         "common per-run change (chained warm starts)")
    return ap


def main(argv=None) -> int:
    """Public driver entry point.

    Parses ``argv`` (or ``sys.argv[1:]`` if None), loads the config, and
    returns the driver's integer exit code.  Callers who want the classic
    "sys.exit on failure" behaviour should wrap: ``sys.exit(main())``.

    Suitable for direct use from tests — no need to reach into
    ``_args_from_config`` or ``run`` privately.
    """
    from legoesm.land.lmip_config import load_config
    cli_args = build_parser().parse_args(argv)
    cfg = load_config(cli_args.config)
    return run(_args_from_config(cfg, cli_args))


if __name__ == "__main__":
    sys.exit(main())
