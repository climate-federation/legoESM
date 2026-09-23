#!/usr/bin/env python
"""AIMIP training on the lat-lon C-grid primitive-equation dycore.

Moves the AIMIP family comparison off the spectral Gaussian PE path onto
the **lat-lon C-grid PE** carry-based stack, which the GPU benchmark
(scripts/cluster/scaling_ginsburg/aimip_grid_throughput.sbatch) showed is
~9-34x faster per step on GPU and makes radiation-flux supervision native
(the SegmentCarry carries TOA + surface fluxes; the spectral state does
not).

Families (this driver — Phase 1):
  * ``classical``  — tune the physics-pipeline scheme parameters
    (C_H/C_E, albedos, SBM ... — the gray optical depths left the trainable
    set on 2026-08-11) via ``train_physics_params``.
    The *scheme combination* itself (convection/turbulence/...) is selected
    by CLI flags; sweeping combinations = many invocations of this driver.
  * ``column_nn``  — neural column physics (Rasp-style MLP) via
    ``train_neural_gcm``.
SFNO is NOT trained by this single-GPU launcher; that path moved to the
SPMD-shardable WB scale trainer (``run_amip.py --variants sfno_full`` /
``training.scale_build``). Passing ``--variants sfno`` here fails fast.

Radiation fluxes (TOA reflected SW, OLR, surface net SW/LW) enter the
TRAINING LOSS via ``LossConfig.w_flux_*`` so the learned/tuned models
match ERA5's radiative response and generalise across climates.  ERA5
flux targets come from the WB2 ``mean_*_radiation_flux`` variables
(``TrainingERA5Config.load_radiation_fluxes=True``).

Everything reuses validated building blocks: ``ModelDriver`` (model +
forcing climatology), ``era5_to_latlon_carry`` (flux-aware),
``train_physics_params`` / ``train_neural_gcm``, ``build_training_segment``
+ ``single_day_rollout`` for eval, and ``combined_loss`` (which now
includes the flux term).

Example (tiny smoke):
    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run/run_aimip_latlon.py \\
        --smoke --output-dir results/aimip_latlon_smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

logger = logging.getLogger("aimip_latlon")

REPO = Path(__file__).resolve().parents[2]

# TRAINING supervision horizon: rollout length AND target lead, kept in
# lockstep.  Passed as ``rollout_hours`` to the trainers and used for the
# matched target lead in ``load_window_pairs``.
#
# Raised 6 -> 24 h on 2026-09-18 (user decision), because the reason for the
# 6 h cap turned out not to hold.  The cap cited job 8533825 for "a 24 h
# rollout explodes the adjoint to NaN through the 144-step differentiable
# dycore even when the forward is finite".  Re-measured on current code
# (jobs 9829660 sweep, 9829661 with JAX's NaN detector armed; 32x8,
# dt=600 s, adjoint w.r.t. the initial state): NO NaN is created anywhere in
# the backward pass at 144 steps, in either the current scheme set or the
# June-2026 one, and the current lane's adjoint is BOUNDED — max|adjoint|
# rises once inside the first eight steps and then sits between 1e3 and 1e5
# out to 24 h, with no exponential branch.  Kessler and the mass-flux scheme
# are what damp it; the June set (gray radiation, SBM convection, no
# microphysics) instead grows at 0.28/step and reaches 3e13 by 144 steps.
#
# So this value is only safe for a scheme set whose adjoint is bounded.
# Re-run scripts/validate/adjoint_horizon_sweep.py before trusting 24 h on a
# different one; the argument is about CONDITIONING, not about the absence
# of a NaN, and 3e13 is finite but useless to an optimiser.
#
# Cost: 144 steps per supervised sample instead of 36.
_ROLLOUT_HOURS = 24

# EVALUATION horizon, deliberately NOT raised with the training horizon.
# The headline RMSE here is quoted against the canonical AIMIP scorecard
# (``run_aimip.py::_evaluate_variant``), which scores a free 6 h forecast; a
# 24 h eval would silently redefine every number and break comparability
# with every result already on record.  Changing the training horizon is the
# variable under test, so the evaluation protocol is held fixed.
_EVAL_HOURS = 6


# ---------------------------------------------------------------------------
# Config build (reuse run_amip's parser so the 70-field ExperimentConfig is
# constructed exactly like a production AMIP run).
# ---------------------------------------------------------------------------

def _load_run_amip():
    spec = importlib.util.spec_from_file_location(
        "_run_amip_mod", REPO / "scripts" / "run" / "run_amip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_latlon_config(args):
    """Build an ExperimentConfig for a lat-lon C-grid AMIP run."""
    ra = _load_run_amip()
    # MPAS reuses this whole driver: the trainers (train_physics_params /
    # train_neural_gcm) take ``model`` and ``grid`` as ARGUMENTS and never
    # construct either, so the grid is a two-value swap here plus the carry
    # converter below. A third near-duplicate driver would be copy-paste.
    if args.grid == "mpas":
        # REFUSED, with the blockers named. An earlier version of this flag
        # claimed MPAS "just worked" because the trainers take model+grid as
        # arguments; that was wrong and was never actually run. What breaks,
        # verified in the source:
        #   1. --resolution means LATITUDE COUNT here and SCVT SUBDIVISION
        #      LEVEL for MPAS (the builder caps it at 10), so --n-lat 72 does
        #      not even construct a mesh;
        #   2. era5_to_mpas_carry returns an MPASCarry, which is NOT a
        #      SegmentCarry and has none of the held_* flux fields this
        #      driver's loss and eval read;
        #   3. the SST regrid, build_training_segment and the eval accumulators
        #      all index grid.lat / grid.lon / cos_lat and a cell-centred v,
        #      none of which an SCVT mesh provides.
        # Refusing microphysics/turbulence (the previous guard) addressed none
        # of these. MPAS needs its own carry/forcing/eval path; until then a
        # loud refusal beats a flag that looks supported.
        raise SystemExit(
            "--grid mpas is NOT supported by this driver yet.\n"
            "  * --resolution is a latitude count here but an SCVT subdivision "
            "level for MPAS (max 10), so no mesh is built;\n"
            "  * era5_to_mpas_carry returns MPASCarry, not the SegmentCarry "
            "(with held_* radiative fluxes) this driver trains and scores on;\n"
            "  * the SST regrid, segment builder and eval all assume "
            "grid.lat/grid.lon/cos_lat and a cell-centred v.\n"
            "Use --grid latlon. An MPAS AIMIP lane needs an MPAS-specific "
            "carry + forcing + eval path first."
        )
    _grid_type, _disc = {
        "latlon": ("latlon", "latlon_cgrid"),
    }[args.grid]
    argv = [
        "--grid-type", _grid_type,
        "--discretization", _disc,
        "--resolution", str(args.n_lat),
        "--nlev", str(args.n_lev),
        "--vertical-coord", "sigma",
        "--dt", str(args.dt),
        # Unique per-run driver output dir: ModelDriver.setup() writes a
        # run_manifest, and concurrent sweep tasks sharing the default
        # results/amip/<grid>_<date> dir collide ("refuses to mix two
        # runs").  Point it at this run's own output dir.
        "--output", str(args.output_dir),
        "--radiation", args.radiation,
        "--convection", args.convection,
        "--turbulence", args.turbulence,
        "--microphysics", args.microphysics,
        "--clouds", args.clouds,
        "--gravity-wave-drag", args.gravity_wave_drag,
        # Match the training-segment sub-cycling (also resolves the None that
        # skipping _postprocess_args would leave).
        "--rad-update-steps", str(args.rad_update_steps),
        # Lat-lon C-grid needs pole-CFL relief; the Fourier polar filter
        # lets the dycore take a much larger stable dt than the ~30 s the
        # raw pole cells force.  The effective (CFL-clamped) dt is read
        # back from the model after setup and used for the rollout.
        "--use-polar-filter",
        "--days", "1",
        "--dataset", "analytical",   # IC/forcing replaced by ERA5 below
        "--ic", "standard",
    ]
    parsed = ra.build_arg_parser().parse_args(argv)
    # Land surface model.  This lane builds its ExperimentConfig from a FIXED
    # argv that passes no land flags and no --topography, so until now it
    # resolved to "none" — meaning NO land surface model at all: land skin
    # temperature falls back to the neighbouring prescribed SST minus a lapse
    # rate, with no soil, no water store and no stomatal control.  That was a
    # default nobody had chosen, which is the shape of failure this repo has
    # paid for before, so the choice is now explicit and printed.
    #
    # The switches come from the land package's own mapping rather than being
    # spelled out here, so this driver and run_amip cannot drift apart.
    from legoesm.land.config import describe_land_model, land_model_switches
    for _field, _value in land_model_switches(args.land_model).items():
        setattr(parsed, _field, _value)
    if getattr(args, "land_mask_file", ""):
        parsed.land_mask_file = args.land_mask_file
    logger.info(
        "Land surface model: %s (slab_land_active=%s, use_multilayer_land=%s, "
        "land_mask_file=%s)",
        describe_land_model(
            slab_land_active=parsed.slab_land_active,
            use_multilayer_land=parsed.use_multilayer_land,
            has_land_mask=bool(getattr(parsed, "land_mask_file", ""))),
        parsed.slab_land_active, parsed.use_multilayer_land,
        getattr(parsed, "land_mask_file", "") or "<none>")
    return ra.build_config_from_args(parsed)


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------

def make_loss_config(args):
    from legoesm.training.losses import LossConfig
    return LossConfig(
        w_T=1.0, w_u=0.5, w_v=0.5, w_q=0.3, w_ps=0.3,
        w_bias_T=10.0,
        # Radiation-flux supervision (TOA + surface).
        w_flux_rsut=args.w_flux_rsut,
        w_flux_olr=args.w_flux_olr,
        w_flux_sfc_sw=args.w_flux_sfc_sw,
        w_flux_sfc_lw=args.w_flux_sfc_lw,
        w_bias_flux_olr=args.w_bias_flux_olr,
        w_bias_flux_rsut=args.w_bias_flux_rsut,
        flux_scale=20.0,
        normalize_by_scale=True,
        level_weighting="pressure",
        # GenCast-style multi-step autoregressive supervision (empty ->
        # single 6 h horizon).  multi_step_rollout_loss reads these off
        # loss_config, so no trainer-signature change is needed.
        multi_step_hours=tuple(args.multi_step_hours),
        multi_step_weights=tuple(args.multi_step_weights),
    )


# ---------------------------------------------------------------------------
# ERA5 window loading -> (ic, target, forcing) triples on the model grid
# ---------------------------------------------------------------------------

def _parse_windows(spec: str):
    """'2015:0:3,2016:180:3' -> [(2015,0,3), (2016,180,3)]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        y, off, n = chunk.split(":")
        out.append((int(y), int(off), int(n)))
    return out


def load_window_pairs(grid, sigma, windows, *, rollout_hours, forcing_ctx,
                      era5_zarr=None, microphysics="none", turbulence="none",
                      multi_step_hours=(), grid_kind="latlon"):
    """Build (initial_carries, target_carries, forcings) on the model grid.

    Mirrors ``neural_gcm_spectral.load_training_data`` window->time-index
    logic, but produces lat-lon carries (flux-aware) and an aligned
    SegmentForcing per IC.  Pairs are formed inside each window only (no
    cross-window leakage).

    ``multi_step_hours`` empty -> one target ``rollout_hours`` ahead per IC
    (each ``targets[i]`` is a single carry; the single-horizon path).
    Non-empty -> GenCast-style multi-step: each ``targets[i]`` is a TUPLE
    of carries, one per lead in ``sorted(multi_step_hours)``, consumed by
    ``multi_step_rollout_loss``.  Use the single-target form for EVAL (the
    eval rolls one 6 h forecast) and the multi-step form for TRAINING.
    """
    from legoesm.driver.compiled_segments import pack_forcing
    from legoesm.training.era5_to_state import (
        TrainingERA5Config, open_era5_zarr, load_era5_slice,
        era5_to_latlon_carry, era5_to_mpas_carry, regrid_2d_to_gaussian,
    )

    def _to_carry(era5, grid_or_mesh, sigma_, microphysics_, turbulence_):
        """ERA5 slice -> SegmentCarry on whichever horizontal grid is in use.

        The two converters do NOT take the same arguments.
        ``era5_to_latlon_carry`` accepts ``microphysics`` / ``turbulence``,
        which size the tracer slots the physics pipeline expects;
        ``era5_to_mpas_carry`` has no such parameters (it builds the
        edge-normal wind state and the scalar cell fields only). Passing them
        anyway would be a TypeError; silently DROPPING them on the MPAS path
        without saying so would be worse — a run configured with microphysics
        would start from a carry that has nowhere to put the condensate. So the
        MPAS path refuses the combination instead of guessing.
        """
        if grid_kind == "mpas":  # unreachable: build_latlon_config refuses first
            if microphysics_ not in (None, "none") or \
                    turbulence_ not in (None, "none"):
                raise NotImplementedError(
                    "era5_to_mpas_carry() takes no microphysics/turbulence "
                    f"arguments, but this run asks for microphysics="
                    f"{microphysics_!r} turbulence={turbulence_!r}. Those size "
                    "the tracer slots on the lat-lon path; the MPAS converter "
                    "does not build them, so the run would start from a carry "
                    "with nowhere to put condensate/TKE. Extend "
                    "era5_to_mpas_carry first, or run --grid mpas with "
                    "--microphysics none --turbulence none."
                )
            return era5_to_mpas_carry(era5, grid_or_mesh, sigma_)
        return era5_to_latlon_carry(
            era5, grid_or_mesh, sigma_,
            microphysics=microphysics_, turbulence=turbulence_,
        )

    era5_cfg = TrainingERA5Config(dt_hours=6, load_radiation_fluxes=True)
    if era5_zarr:
        era5_cfg = era5_cfg._replace(zarr_store=era5_zarr)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    # Radiation-flux targets come from a separate clean store (ARCO-ERA5);
    # open it once here and thread it into every snapshot load.
    flux_ds = open_era5_zarr(era5_cfg.flux_zarr) if era5_cfg.flux_zarr else ds

    era5_dt_hours = era5_cfg.dt_hours
    # Lead(s) in ERA5-snapshot units.  Single-horizon: [rollout_hours].
    # Multi-step: the sorted multi_step_hours leads (each a target snapshot).
    leads_h = sorted(int(h) for h in multi_step_hours) or [int(rollout_hours)]
    for h in leads_h:
        if h % era5_dt_hours != 0:
            raise ValueError(
                f"lead {h}h (rollout_hours/multi_step_hours) must be a "
                f"multiple of era5_dt_hours={era5_dt_hours}."
            )
    strides = [h // era5_dt_hours for h in leads_h]   # snapshot units IC->target
    max_stride = max(strides)
    is_multi = bool(multi_step_hours)
    snaps_per_day = 24 // era5_dt_hours

    times = np.asarray(ds.time.values, dtype="datetime64[ns]")

    def _year_to_idx(year: int) -> int:
        return int(np.searchsorted(times, np.datetime64(f"{year:04d}-01-01")))

    def _doy(tidx: int) -> float:
        d = times[tidx]
        yr0 = d.astype("datetime64[Y]")
        return float((d - yr0) / np.timedelta64(1, "D")) + 1.0

    # Forcing climatology fields (o3 / aerosol / solar) are physics-scheme
    # independent — taken once from the driver context, reused per window
    # (the per-IC seasonal signal rides ``day_of_year``).
    solar_weights = forcing_ctx["solar_weights"]
    s_0 = forcing_ctx["current_s_0"]
    o3_vmr = forcing_ctx["o3_vmr"]
    aerosol_od = forcing_ctx["aerosol_od"]

    ics, targets, forcings = [], [], []
    for (year, day_off, n_days) in windows:
        if n_days < 1:
            raise ValueError(f"window {(year, day_off, n_days)} has n_days<1")
        start = _year_to_idx(year) + day_off * snaps_per_day
        n_ics = n_days * snaps_per_day
        # Load the contiguous snapshot block (ICs + the max-lead lookahead).
        block = {}
        for s in range(n_ics + max_stride):
            tidx = start + s
            era5 = load_era5_slice(era5_cfg, tidx, ds=ds, flux_ds=flux_ds)
            block[s] = (_to_carry(era5, grid, sigma, microphysics, turbulence),
                        era5, tidx)
        for d in range(n_ics):
            ic_carry, ic_era5, ic_tidx = block[d]
            # Single-horizon -> one target carry; multi-step -> a tuple of
            # carries (one per lead) for multi_step_rollout_loss.
            if is_multi:
                target_carry = tuple(block[d + s][0] for s in strides)
            else:
                target_carry, _, _ = block[d + strides[0]]
            # The IC must NOT carry the observed ERA5 fluxes (they are
            # loaded for the TARGET only): otherwise the model's predicted
            # held_* can leak the IC's observed fluxes and fake a perfect
            # flux score (column_nn's untrained ~0-tendency NN left the
            # ERA5-seeded held_* almost untouched).  Zero them on the IC —
            # radiation/physics recompute them on step 1.
            ic_carry = ic_carry._replace(
                held_dT_rad=jnp.zeros_like(ic_carry.held_dT_rad),
                held_sw_net_sfc=jnp.zeros_like(ic_carry.held_sw_net_sfc),
                held_lw_net_sfc=jnp.zeros_like(ic_carry.held_lw_net_sfc),
                held_sw_up_toa=jnp.zeros_like(ic_carry.held_sw_up_toa),
                held_lw_up_toa=jnp.zeros_like(ic_carry.held_lw_up_toa),
                held_sw_down_toa=jnp.zeros_like(ic_carry.held_sw_down_toa),
            )
            sst = jnp.asarray(
                regrid_2d_to_gaussian(ic_era5.sst, ic_era5.lat, ic_era5.lon, grid)
            )
            forcing = pack_forcing(
                sst=sst,
                sic=jnp.zeros_like(sst),
                day_of_year=jnp.asarray(_doy(ic_tidx)),
                seconds_of_day=jnp.asarray(0.0),
                solar_weights=solar_weights,
                s_0=s_0,
                o3_vmr=o3_vmr,
                aerosol_od=aerosol_od,
            )
            ics.append(ic_carry)
            targets.append(target_carry)
            forcings.append(forcing)
    # Guard (codex review #2): the flux loss only makes sense when the
    # target carries hold REAL ERA5 fluxes.  Zero-filled targets + active
    # flux weights would silently train the model's radiation toward zero.
    # We always load fluxes here; verify they actually arrived (catches an
    # all-zero ERA5 read / regrid failure) on concrete arrays before any JIT.
    # Unwrap a multi-step target tuple to one carry for the sanity checks
    # (every lead carries the same ERA5 flux variables).  NB a SegmentCarry
    # IS a NamedTuple (== tuple subclass), so isinstance(targets[0], tuple)
    # can NOT distinguish a single carry from a multi-step tuple-of-carries —
    # key off is_multi (else targets[0][0] grabs the carry's first FIELD, an
    # array, and .held_lw_up_toa blows up on the single-target eval load).
    _t0 = (targets[0][0] if (ics and is_multi)
           else (targets[0] if ics else None))
    if ics:
        import numpy as _np
        tgt_olr = _np.asarray(_t0.held_lw_up_toa)
        if not _np.isfinite(tgt_olr).all():
            raise RuntimeError(
                "Target carry OLR has non-finite values "
                f"(nan_frac={_np.isnan(tgt_olr).mean():.3f}) — the ERA5 "
                "radiation-flux target is corrupted.  NaN targets would NaN "
                "out the flux loss.  Check the WB2 mean_*_radiation_flux "
                "variables (scripts/tmp/_probe_wb2_flux_values.py)."
            )
        if float(_np.abs(tgt_olr).max()) == 0.0:
            raise RuntimeError(
                "Target carry radiation fluxes are all zero despite "
                "load_radiation_fluxes=True — ERA5 flux load/regrid failed. "
                "Flux-loss supervision would push the model toward zero "
                "radiation.  Check the WB2 mean_*_radiation_flux variables."
            )
    logger.info("Loaded %d (ic,target,forcing) pairs across %d window(s)%s; "
                "target OLR mean ~%.1f W/m^2",
                len(ics), len(windows),
                f" (multi-step leads={leads_h}h)" if is_multi else "",
                float(np.asarray(_t0.held_lw_up_toa).mean()) if ics else 0.0)
    return ics, targets, forcings


# ---------------------------------------------------------------------------
# Held-out evaluation: per-variable + radiation-flux RMSE/bias vs ERA5
# ---------------------------------------------------------------------------

def _area_weighted(field, cos_lat):
    """Area-weighted mean over all axes (lat axis weighted by cos lat)."""
    from legoesm.training.losses import lat_weighted_mean
    return float(lat_weighted_mean(field, cos_lat))


def evaluate(run_seg_raw, ics, targets, forcings, grid, dt, hours=_EVAL_HOURS):
    """RMSE + bias per prognostic + radiation flux, averaged over windows.

    METRIC PARITY with the canonical AIMIP scorecard (``run_aimip.py``
    ``_evaluate_variant``): the headline ``T``/``u``/``v`` RMSE is the
    SINGLE mid-level (sigma ~0.5, ≈500 hPa) cross-section — the
    WeatherBench T@500hPa convention — NOT the full 3-D column.  The
    full-column number (kept as ``T_column``) is dominated by the
    boundary-layer + near-vacuum stratosphere/sponge levels where the
    free 6 h forecast has its largest errors, so it runs ~3x higher than
    the 500 hPa number and is NOT comparable to the spectral scorecard.
    ``T_sfc`` is the near-surface (last sigma) level, matching the
    canonical ``T_sfc`` entry used in the AIMIP intercomparison plot.
    """
    from legoesm.training.dycore_rollout import single_day_rollout

    cos_lat = jnp.asarray(grid.cos_lat)
    acc = {}

    def _accum(name, pred, tgt):
        """2-D field RMSE/bias (or full 3-D column when arrays are 3-D)."""
        d = pred - tgt
        rmse = np.sqrt(_area_weighted(d ** 2, cos_lat))
        bias = _area_weighted(d, cos_lat)
        acc.setdefault(name, {"rmse": [], "bias": []})
        acc[name]["rmse"].append(rmse)
        acc[name]["bias"].append(bias)

    def _accum_level(name, pred3d, tgt3d, level):
        """Single-level (lat,lon) slice of a 3-D (lat,lon,nlev) field —
        the canonical AIMIP metric (500 hPa mid-level / near-surface)."""
        _accum(name, pred3d[..., level], tgt3d[..., level])

    diag = {}  # non-RMSE model diagnostics (no ERA5 target)

    def _diag_precip(pred):
        # precip_accum [kg/m^2] over the rollout -> mm/day (1 kg/m^2 = 1 mm).
        pa = getattr(pred, "precip_accum", None)
        if pa is None:
            return
        rate = jnp.asarray(pa) / max(float(hours), 1e-9) * 24.0
        diag.setdefault("precip_mm_day", {"mean": [], "max": []})
        diag["precip_mm_day"]["mean"].append(_area_weighted(rate, cos_lat))
        diag["precip_mm_day"]["max"].append(float(np.max(np.asarray(rate))))

    for ic, tgt, forcing in zip(ics, targets, forcings):
        pred = single_day_rollout(ic, forcing, run_seg_raw, dt=dt, hours=hours)
        nlev = pred.T.shape[-1]
        mid = nlev // 2          # ~500 hPa (sigma ~0.5) — WeatherBench convention
        surf = nlev - 1          # near-surface (last sigma level)
        # Headline metrics: single mid-level slice = canonical AIMIP parity.
        _accum_level("T", pred.T, tgt.T, mid)
        _accum_level("u", pred.u, tgt.u, mid)
        _accum_level("v", pred.v, tgt.v, mid)
        _accum_level("T_sfc", pred.T, tgt.T, surf)     # near-surface T
        _accum_level("q", pred.q_v, tgt.q_v, surf)     # humidity is surface-heavy
        # Full-column diagnostic (NOT comparable to the spectral scorecard —
        # inflated ~3x by boundary-layer + stratosphere/sponge levels).
        _accum("T_column", pred.T, tgt.T)
        _accum("ps", pred.p_s, tgt.p_s)
        # Radiation fluxes (only meaningful when targets carry ERA5 fluxes).
        _accum("olr", pred.held_lw_up_toa, tgt.held_lw_up_toa)
        _accum("rsut", pred.held_sw_up_toa, tgt.held_sw_up_toa)
        _accum("sfc_net_sw", pred.held_sw_net_sfc, tgt.held_sw_net_sfc)
        _accum("sfc_net_lw", pred.held_lw_net_sfc, tgt.held_lw_net_sfc)
        # Precipitation has no ERA5 target here -> report model rate as a
        # physical sanity diagnostic (global-mean ~3 mm/day expected).
        _diag_precip(pred)

    out = {k: {"rmse": float(np.mean(v["rmse"])),
               "bias": float(np.mean(v["bias"]))}
           for k, v in acc.items()}
    for k, v in diag.items():
        out[k] = {"mean": float(np.mean(v["mean"])), "max": float(np.max(v["max"]))}
    return out


# ---------------------------------------------------------------------------
# Variant training
# ---------------------------------------------------------------------------

def train_variant(variant, model, grid, sigma, physics_pipeline, config,
                  ics, targets, forcings, eval_data, loss_config, args, outdir):
    from legoesm.training.training_driver import (
        train_physics_params, train_neural_gcm, build_training_segment,
    )
    from legoesm.training.dycore_rollout import single_day_rollout  # noqa: F401

    t0 = time.time()
    if variant == "classical":
        trained, hist = train_physics_params(
            model, grid, sigma, physics_pipeline, ics, targets, forcings,
            n_epochs=args.epochs, lr=args.lr, dt=args.dt,
            rollout_hours=_ROLLOUT_HOURS, microphysics=args.microphysics,
            rad_update_steps=args.rad_update_steps,
            rad_stop_gradient=args.radiation_as_forcing,
            loss_config=loss_config, log_every=1,
        )
        step_unified = physics_pipeline.build_step_unified(
            rad_stop_gradient=args.radiation_as_forcing)
        seg = build_training_segment(
            model, step_unified, grid, sigma, args.dt,
            microphysics=args.microphysics,
            rad_update_steps=args.rad_update_steps,
            **trained.to_segment_kwargs(),
        )
    elif variant == "column_nn":
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics, make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter
        key = jax.random.PRNGKey(args.nn_seed)
        nn = NeuralPhysics(
            nlev=args.n_lev, hidden_dim=args.nn_hidden,
            n_layers=args.nn_layers, key=key,
            residual_scale=args.nn_residual_scale,
        )
        trained, hist = train_neural_gcm(
            model, grid, sigma, nn, ics, targets, forcings,
            n_epochs=args.epochs, lr=args.lr, dt=args.dt,
            rollout_hours=_ROLLOUT_HOURS,
            loss_config=loss_config, log_every=1,
        )
        adapter = make_adapter(grid)
        step_unified = make_neural_step_unified(trained, adapter)
        seg = build_training_segment(model, step_unified, grid, sigma, args.dt)
    else:
        raise ValueError(
            f"unknown variant {variant!r}; supports 'classical', 'column_nn'."
        )

    train_seconds = time.time() - t0
    eval_metrics = evaluate(
        seg.raw, eval_data[0], eval_data[1], eval_data[2], grid, args.dt,
    )

    vdir = outdir / variant
    vdir.mkdir(parents=True, exist_ok=True)
    ckpt = vdir / "params.eqx"
    eqx.tree_serialise_leaves(str(ckpt), trained)

    return {
        "variant": variant,
        "train_loss_history": [float(x) for x in hist],
        "train_seconds": train_seconds,
        "eval_metrics": eval_metrics,
        "checkpoint": str(ckpt),
        "schemes": {
            "convection": args.convection, "turbulence": args.turbulence,
            "microphysics": args.microphysics, "clouds": args.clouds,
            "gravity_wave_drag": args.gravity_wave_drag,
            "radiation": args.radiation,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="AIMIP lat-lon C-grid training")
    p.add_argument("--n-lat", type=int, default=72)
    p.add_argument("--n-lev", type=int, default=8)
    p.add_argument("--dt", type=float, default=600.0)
    # AMIP-like runs MUST use the band model (rrtmgp): gray radiation ignores
    # o3/aerosol/water-vapour spectral structure, so its TOA/surface fluxes
    # cannot generalize to a different climate — the whole point of flux
    # supervision.  Gray stays selectable for cheap debug only.
    p.add_argument("--radiation", default="rrtmgp")
    # Convection default = mass_flux (DETRAINING).  With microphysics=kessler,
    # an ADJUSTMENT convection scheme (sbm/dca/kuo) precipitates the column
    # drying DIRECTLY *and* kessler rains the same vapour -> precip
    # double-counted (measured ~26 mm/day vs ~3 expected).  A detraining scheme
    # (mass_flux/edmf/...) routes convective condensate to the q_c cloud bucket
    # so kessler is the SOLE surface-precip owner — physically correct + no
    # double-count.  sbm stays selectable (pair it with microphysics=none).
    p.add_argument("--convection", default="mass_flux")
    p.add_argument("--turbulence", default="louis")
    # Full AMIP physics stack: warm-rain microphysics (kessler — cheap,
    # differentiable, fast-compiling; morrison/p3 + rrtmgp blow the compile
    # budget) and non-orographic spectral gravity-wave drag (hines — active
    # without subgrid-orography input, which the lat-lon setup lacks; mcfarlane
    # would be inert here).  Applied to the CLASSICAL physics variant; the
    # NN-replacement variants subsume these into the learned tendencies.
    p.add_argument("--microphysics", default="kessler")
    p.add_argument("--clouds", default="xu_randall")
    p.add_argument("--gravity-wave-drag", default="hines")
    p.add_argument("--allow-unfilled-families", action="store_true",
                   help="Permit a classical run with a family set to 'none'. "
                        "For ablations only: the result is NOT comparable to "
                        "a complete classical model.")
    # Radiation sub-cycling for the TRAINING rollout: run rrtmgp every N
    # dynamics steps, NOT every step (radiation varies slowly; GCMs update it
    # ~hourly).  rrtmgp is the dominant cost of the classical variant, so the
    # default sub-cycles (N=6 -> ~hourly at dt~600s, ~15 min at the T106
    # CFL dt~155s) — never every step.  Moot for the NN-replacement variants.
    p.add_argument("--rad-update-steps", type=int, default=6)
    # Radiation-as-forcing: stop_gradient rrtmgp in the rollout so its adjoint
    # never enters the backward graph.  rrtmgp's reverse-mode is the dominant
    # XLA compile cost and it GROWS with grid size (>10 h at true T106) — this
    # collapses it so high-res differentiable training is feasible.  The state
    # loss still trains convection/turbulence/surface; TOA/surface fluxes
    # follow once the state matches ERA5.  Trade-off: the flux loss can no
    # longer tune radiation params (albedo) — that needs the cheap 2-scalar
    # forward-mode path (follow-up).  Classical variant only.
    p.add_argument("--radiation-as-forcing", action="store_true", default=False)
    # GenCast-style multi-step autoregressive supervision (the canonical
    # AIMIP protocol; spectral got classical 6.8->4.05K with it).  The
    # per-segment gradient is TRUNCATED (stop_gradient between segments).
    # The old justification for that — "the full-rollout adjoint NaNs past
    # ~6h" — does NOT reproduce on current code (see _ROLLOUT_HOURS above:
    # no NaN at 144 steps with the detector armed).  Truncation is now a
    # conditioning choice, not a necessity, and whether it still earns its
    # bias is an open question worth measuring.  Empty -> a single rollout
    # at _ROLLOUT_HOURS.
    p.add_argument("--multi-step-hours", type=int, nargs="*", default=[],
                   help="autoregressive leads in hours (multiples of 6), "
                        "e.g. 6 12 18 24; empty -> a single rollout at the "
                        "training horizon (24 h).")
    p.add_argument("--multi-step-weights", type=float, nargs="*", default=[],
                   help="per-lead loss weights (match --multi-step-hours); "
                        "empty -> uniform.")
    p.add_argument("--variants", default="classical,column_nn",
                   help="comma list: classical,column_nn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-windows", default="2015:0:3,2016:180:3")
    p.add_argument("--eval-windows", default="2017:0:3")
    p.add_argument("--nn-hidden", type=int, default=256)
    p.add_argument("--nn-layers", type=int, default=4)
    p.add_argument("--nn-seed", type=int, default=0)
    # Output tendency scale for the column NN.  The raw network output is
    # O(1); NeuralPhysics' own 0.01 default treats it as a per-SECOND
    # tendency (dT/dt ~ 0.01 K/s = 36 K/hr) which blows up the multi-step
    # forward at init.  Measured: 1e-4 still NaN'd by sample 3, 1e-5 is
    # stable (untrained-NN rollout ≈ pure dynamics) and trains — use 1e-5.
    p.add_argument("--nn-residual-scale", type=float, default=1.0e-5)
    # Radiation-flux loss weights (TOA + surface).
    p.add_argument("--w-flux-olr", type=float, default=1.0)
    p.add_argument("--w-flux-rsut", type=float, default=0.5)
    p.add_argument("--w-flux-sfc-sw", type=float, default=0.5)
    p.add_argument("--w-flux-sfc-lw", type=float, default=0.5)
    p.add_argument("--w-bias-flux-olr", type=float, default=10.0)
    p.add_argument("--w-bias-flux-rsut", type=float, default=2.0)
    p.add_argument(
        "--grid", choices=("latlon", "mpas"), default="latlon",
        help="Horizontal grid. 'latlon' = the lat-lon C-grid PE (default, "
             "unchanged). 'mpas' = the SCVT Voronoi mesh + TRiSK PE, which "
             "reuses this entire driver because the carry-based trainers take "
             "model and grid as arguments. MPAS currently requires "
             "--microphysics none --turbulence none (era5_to_mpas_carry builds "
             "no tracer slots); it refuses loudly rather than dropping them.")
    p.add_argument("--era5-zarr", default=None)
    p.add_argument("--output-dir", default="results/aimip_latlon")
    p.add_argument("--land-model", choices=("none", "slab", "multilayer"),
                   default="none",
                   help="Land surface model for the physics rollout, same "
                        "three names the coupled driver and run_amip use. "
                        "DEFAULT 'none' preserves what this lane has always "
                        "done, which is to run with NO land surface model: "
                        "land skin temperature falls back to the neighbouring "
                        "prescribed SST minus a lapse rate, with no soil, no "
                        "water store and no stomatal control. 'slab' and "
                        "'multilayer' need somewhere for land to exist, so "
                        "pass --land-mask-file with them (this lane builds a "
                        "flat topography, and the driver refuses a land tile "
                        "that would have no land under it).")
    p.add_argument("--land-mask-file", type=str, default="",
                   help="Land-mask NetCDF forwarded to the ExperimentConfig. "
                        "Required to give --land-model slab/multilayer any "
                        "land to act on, since this lane runs flat topography.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config (n_lat=32, 1 epoch, 1+1 short windows)")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if args.smoke:
        args.n_lat = 32
        args.n_lev = 8
        args.epochs = 1
        args.train_windows = "2015:0:1"
        args.eval_windows = "2017:0:1"

    # Validate variants up front (before model build / ERA5 load) so a bad or
    # retired token fails fast and cleanly, not mid-loop where a SystemExit
    # would bypass the per-variant except and skip the scorecard write. The
    # SFNO path moved to the WB scale trainer (run_amip.py --variants
    # sfno_full); this launcher no longer trains SFNO.
    _SUPPORTED_VARIANTS = ("classical", "column_nn")
    _sel_variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    _bad = sorted(_sel_variants - set(_SUPPORTED_VARIANTS))
    if _bad:
        raise SystemExit(
            f"unsupported variant(s) {_bad}; supported: "
            f"{list(_SUPPORTED_VARIANTS)}. The SFNO path moved to the WB scale "
            "trainer: run_amip.py --variants sfno_full."
        )

    # Radiation pin, EVERY variant: an AIMIP run uses rrtmgp. This used to
    # guard `classical` only, so a column_nn invocation could pass
    # --radiation gray (Claude review). Only --smoke is exempt.
    from legoesm.training.campaign_driver import validate_campaign_radiation
    validate_campaign_radiation(
        str(args.radiation), campaign="aimip", smoke=bool(args.smoke),
    )

    if "classical" in _sel_variants:
        # ... and the same one-parameterization-per-family rule the spectral
        # lane enforces. This lane builds its physics through physics_pipeline
        # rather than make_aimip_classical_spectral_physics, so the factory
        # gate never sees it (codex round 7) — hence the explicit call.
        from legoesm.training.aimip_params import (
            validate_classical_scheme_set,
        )
        validate_classical_scheme_set(
            convection=str(args.convection),
            turbulence=str(args.turbulence),
            cloud=str(args.clouds),
            microphysics=str(args.microphysics),
            radiation=str(args.radiation),
            gwd=str(args.gravity_wave_drag),
            allow_unfilled=bool(
                getattr(args, "allow_unfilled_families", False)),
        )

    # Microphysics is now threaded through build_training_segment (the carry
    # already carries q_c/q_r/conv_prog/precip_accum), so a real scheme runs
    # in the CLASSICAL physics variant.  The NN-replacement variant (column_nn)
    # keeps microphysics='none' inside its segment — the network subsumes
    # condensation — so the scheme only changes the physics model.  Warn if a
    # non-none scheme is requested while only NN variants are selected (it
    # would have no effect there).
    _nn_only = _sel_variants <= {"column_nn"}
    if args.microphysics != "none" and _nn_only:
        logger.warning(
            "microphysics=%s requested but only NN-replacement variants are "
            "selected; the network subsumes condensation, so the microphysics "
            "scheme has no effect. It applies to the 'classical' variant.",
            args.microphysics,
        )
    # Guard (codex review #5): per-window ozone/aerosol composition is not
    # threaded; the forcing climatology is taken once and reused (only
    # day_of_year varies).  Gray radiation ignores o3/aerosol so this is
    # exact for gray; warn for spectral radiation.
    if args.radiation != "gray":
        logger.warning(
            "radiation=%s but o3/aerosol composition is fixed across "
            "windows (reused from the driver context). Exact for gray; "
            "thread per-window composition before trusting rrtmgp fluxes.",
            args.radiation,
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("AIMIP lat-lon training | devices=%s", jax.devices())

    from legoesm.driver.model_driver import ModelDriver

    config = build_latlon_config(args)
    driver = ModelDriver(config)
    driver.setup()
    grid, sigma, model = driver.grid, driver.sigma, driver.model

    # CFL-safe dt for the rollout.  ``model.effective_dt`` only applies the
    # equatorial-ADVECTIVE clamp; the gravity-wave CFL is stricter and is
    # what actually blows up the high-resolution rollout (n_lat=160:
    # effective_dt=334s but the gravity-wave limit is ~180s -> NaN).  Cap by
    # the gravity-wave CFL on the MERIDIONAL spacing dy=R*pi/n_lat (the
    # polar filter relaxes the tiny zonal pole spacing to ~dy), matching the
    # driver's own reduction.
    from legoesm.core.cfl import cfl_max_dt
    from legoesm import constants as _const
    dy_min = float(_const.R_earth) * float(np.pi) / float(args.n_lat)
    dt_cfl = cfl_max_dt(dy_min, wave_speed=400.0, cfl_number=0.7)  # gravity wave
    dt_eff = min(float(getattr(model, "effective_dt", args.dt)), dt_cfl)
    if dt_eff != args.dt:
        logger.info("CFL-safe dt=%.1fs (requested %.1fs, effective_dt=%.1fs, "
                    "gravity-CFL=%.1fs); %.0f steps per 24h",
                    dt_eff, args.dt, float(getattr(model, "effective_dt", args.dt)),
                    dt_cfl, 86400.0 / dt_eff)
    args.dt = dt_eff

    from legoesm.driver.physics_pipeline import build_physics_pipeline
    physics_pipeline = build_physics_pipeline(grid, sigma, config)

    # Forcing climatology (o3 / aerosol / solar) from the driver context.
    forcing_ctx = driver._prepare_run_context(0, config.start_day,
                                              restore_carry=False)

    loss_config = make_loss_config(args)
    train_windows = _parse_windows(args.train_windows)
    eval_windows = _parse_windows(args.eval_windows)

    logger.info("Loading training data ...")
    # Training uses multi-step targets (tuple per IC) when --multi-step-hours
    # is set; eval ALWAYS uses single 6h targets (evaluate() rolls one 6h
    # forecast and _accum expects a single target carry).
    ics, targets, forcings = load_window_pairs(
        grid, sigma, train_windows,
        grid_kind=args.grid,
        rollout_hours=_ROLLOUT_HOURS, forcing_ctx=forcing_ctx,
        era5_zarr=args.era5_zarr,
        microphysics=args.microphysics, turbulence=args.turbulence,
        multi_step_hours=tuple(args.multi_step_hours),
    )
    logger.info("Loading eval data ...")
    eval_data = load_window_pairs(
        grid, sigma, eval_windows,
        grid_kind=args.grid,
        rollout_hours=_EVAL_HOURS, forcing_ctx=forcing_ctx,
        era5_zarr=args.era5_zarr,
        microphysics=args.microphysics, turbulence=args.turbulence,
    )

    scorecard = {}
    out_json = outdir / "aimip_latlon_scorecard.json"
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        logger.info("===== training variant: %s =====", variant)
        try:
            scorecard[variant] = train_variant(
                variant, model, grid, sigma, physics_pipeline, config,
                ics, targets, forcings, eval_data, loss_config, args, outdir,
            )
            logger.info("  %s eval: %s", variant,
                        json.dumps(scorecard[variant]["eval_metrics"], indent=2))
        except Exception as exc:  # one variant's failure must not lose the others
            logger.exception("variant %s failed", variant)
            scorecard[variant] = {"variant": variant, "error": repr(exc)[:500]}
        # Write incrementally so a later variant's crash never discards an
        # earlier variant's completed results.
        out_json.write_text(json.dumps(scorecard, indent=2))

    logger.info("Wrote scorecard -> %s", out_json)

    # EXIT CODE MUST REFLECT THE OUTCOME. This returned 0 while BOTH variants
    # died on a TypeError, so a batch job reported success and left a scorecard
    # whose every entry was an "error" string (2026-08-08). A per-variant
    # try/except is right — one arm's crash must not discard the other's
    # results — but swallowing it into the exit status is not.
    failed = sorted(v for v, r in scorecard.items() if "error" in r)
    if failed:
        logger.error(
            "%d/%d variant(s) FAILED: %s (see the per-variant 'error' field in "
            "%s)", len(failed), len(scorecard), ", ".join(failed), out_json)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
