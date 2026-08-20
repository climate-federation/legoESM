#!/usr/bin/env python
"""Offline global land spin-up driver (#746 item 1).

Cycles the standalone multilayer land model under repeating
seasonal/diurnal forcing for N years until the soil column equilibrates,
then writes a land RESTART (``save_land_restart``) that the coupled AMIP
driver loads as its land initial condition (``run_amip --land-ic``).

This is the standard land-model workflow the #730/#746 discussion calls
for: a cold-start soil column in a coupled AMIP run drives the land
cloud-albedo cold trap (cold BL -> high RH -> low cloud -> high albedo ->
stays cold); an OFFLINE spin-up equilibrates the deep-soil temperature and
moisture first, so the coupled run starts from a physically-settled land
state instead of the ~day-0 shock.

It reuses ``run_lmip_smoke``'s grid / surfdata / idealised-forcing / land-
step machinery verbatim (imported, not copied) and adds the three things a
spin-up needs on top of the smoke's single-year scan:

  * multi-year cycling (outer Python loop over years; inner ``lax.scan``
    over the year's steps, so each year's carry is O(1) memory);
  * per-year EQUILIBRIUM metrics — the year-over-year drift of the
    land-mean deep-soil temperature and column moisture — which is exactly
    the datum #746 item 1 needs to decide *spin-up* (drift -> 0, the land
    is converging toward balance) vs *structural* (drift pinned, needs a
    cold-BL cloud-response fix, not more spin-up);
  * a land restart at the end + optional per-year checkpointing and
    ``--restart-from`` chaining, so a long spin-up survives a walltime kill
    (the #769/#770 land-restart API).

Forcing is the idealised seasonal/diurnal cycle from ``run_lmip_smoke``
(solar geometry + latitudinal/seasonal/diurnal T); it is repeated every
year, so equilibrium here means "settled under the climatological seasonal
cycle", the right target for a coupled-AMIP land IC.  (A CRU-JRA-forced
variant is the TRENDY/LMIP follow-up; the reader exists in
``legoesm.land.forcing`` but the coupled-AMIP IC only needs a
climatologically-settled column, which the idealised cycle provides.)

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_land_spinup.py \\
        --surfdata data/legoesm_surfdata_v1.nc --grid-type cubed_sphere \\
        --resolution 12 --years 20 --output land_spinup_c12 \\
        --restart-out land_spinup_c12/land_ic.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_SECONDS_PER_YEAR = 365.0 * 86400.0


def _load_smoke():
    """Import ``run_lmip_smoke`` as a module to reuse its grid/forcing/land
    machinery (scripts are not a package; mirror scale_build._load_run_amip)."""
    path = _PROJECT_ROOT / "scripts" / "run" / "run_lmip_smoke.py"
    spec = importlib.util.spec_from_file_location("run_lmip_smoke_for_spinup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _land_mean(field_col, land_mask):
    """Land-masked mean of a per-column (or per-column-per-layer) array."""
    a = np.asarray(field_col)
    if a.ndim == 2:                      # (ncol, n_layers) -> deep-column mean
        a = a.mean(axis=1)
    return float(np.mean(a[land_mask]))


def _build_year_scan(smoke, step_fn, config, update_land_params, cover_year,
                     lat_rad, lon_rad, ncol, dt, is_multilayer, steps_per_year):
    """A JAX ``lax.scan`` over one year's land steps (year-relative step index).

    ``t0_s`` (seconds since the run's Jan 1) sets the absolute calendar so
    the seasonal forcing is continuous across chained years/restarts.
    """
    dt_days = dt / 86400.0
    dt_hours = dt / 3600.0
    U_MIN = smoke.U_MIN

    def _step_body(state, args):
        step_i, t0_s = args
        t_s = t0_s + step_i.astype(jnp.float64) * dt
        doy_t = jnp.mod(t_s / 86400.0, 365.0)
        hour_t = jnp.mod(t_s / 3600.0, 24.0)
        forcing_t = smoke.make_global_forcing(lat_rad, lon_rad, doy_t, hour_t)
        theta_top_t = (state.theta_soil[:, 0] if is_multilayer
                       else jnp.full(ncol, 0.2))
        land_params_t, _lai = update_land_params(theta_top_t, doy_t, cover_year)
        new_state, _resp, _ = step_fn(
            state, forcing_t, config, U_MIN, dt,
            lat=lat_rad, land_params=land_params_t, doy=doy_t)
        return new_state, None

    def run_year(state, t0_s):
        steps = jnp.arange(steps_per_year)
        # t0_s is a traced jit arg — broadcast it (no float(), which would try
        # to concretise the tracer) so every step in the year shares the same
        # absolute calendar origin.
        t0 = jnp.full((steps_per_year,), t0_s, dtype=jnp.float64)
        state, _ = jax.lax.scan(_step_body, state, (steps, t0))
        return state

    return jax.jit(run_year)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surfdata", default="data/legoesm_surfdata_v1.nc")
    ap.add_argument("--grid-type", default="cubed_sphere",
                    choices=["latlon", "gaussian", "cubed_sphere", "voronoi"])
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--surface-scheme", default="two_leaf_canopy",
                    choices=["two_leaf_canopy", "simple_seb"])
    ap.add_argument("--years", type=int, default=20,
                    help="number of seasonal cycles to spin up")
    ap.add_argument("--dt", type=float, default=1800.0,
                    help="land timestep [s]; keep <= ~1800 for soil-column "
                         "stability (the multilayer soil thermal/Richards "
                         "columns blow up at multi-hour dt)")
    ap.add_argument("--seconds-per-year", type=float, default=_SECONDS_PER_YEAR,
                    help="length of one spin-up outer step [s] (default 365 d). "
                         "At the 365-day default each outer step spans a FULL "
                         "annual forcing cycle (doy wraps every 365 d), so the "
                         "per-step drift is the true year-over-year settling. A "
                         "SHORT value (e.g. a few days) is only a fast WIRING "
                         "smoke — it advances the real forcing calendar a few "
                         "days per outer step (it does NOT replay the annual "
                         "cycle), exercising the loop/metrics/restart plumbing "
                         "at a stable dt without a physical multi-year spin-up.")
    ap.add_argument("--land-frac-min", type=float, default=0.5)
    ap.add_argument("--land-mask-file", default="")
    ap.add_argument("--output", default="land_spinup",
                    help="output dir (metrics + per-year checkpoints)")
    ap.add_argument("--restart-out", default=None,
                    help="final land-IC restart path (.npz); "
                         "default <output>/land_ic.npz")
    ap.add_argument("--restart-from", default=None,
                    help="resume from a prior land restart (.npz); the "
                         "calendar picks up at its t_end_s")
    ap.add_argument("--checkpoint-every-years", type=int, default=5,
                    help="write a chainable land restart every N years (0=off)")
    ap.add_argument("--equilibrium-tol-K", type=float, default=0.05,
                    help="deep-soil-T year-over-year drift [K/yr] below which "
                         "the spin-up is reported CONVERGED")
    args = ap.parse_args(argv)

    smoke = _load_smoke()
    from legoesm.land.config import (
        MultiLayerLandConfig, LandConfig, resolve_land_config)
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.canopy import CanopyConfig
    from legoesm.land.surface_scheme import SimpleSEBConfig
    from legoesm.land.multilayer_land import (
        step_multilayer_land, init_multilayer_land_state)
    from legoesm.land.boundary_data import (
        init_land_surface_data, make_step_land_params_updater)
    from legoesm.land.restart import (
        save_land_restart, load_land_restart,
        merge_land_restart_into_template)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = smoke.make_grid(args.grid_type, args.resolution)
    lat_rad, lon_rad = smoke.grid_latlon_rad(grid)
    ncol = int(lat_rad.shape[0])
    dt = float(args.dt)
    steps_per_year = int(round(float(args.seconds_per_year) / dt))
    # The EFFECTIVE year length is the actual simulated time of one inner scan
    # (steps_per_year * dt), NOT the requested seconds_per_year — otherwise, if
    # seconds_per_year is not an exact multiple of dt, the recorded calendar
    # time (t_end_s) would drift from the stepped time and a --restart-from
    # resume would skip/repeat forcing each year (codex #746 finding).  With
    # this, t0/t_end/year-index all equal the exact stepped time.
    sec_per_year = steps_per_year * dt

    # Land config (multilayer only — the spin-up target; slab has no deep soil
    # column to equilibrate and no restart format).
    surf = (CanopyConfig(max_iters=50, tol=1e-2)
            if args.surface_scheme == "two_leaf_canopy" else SimpleSEBConfig())
    base_cfg = resolve_land_config(
        "multilayer",
        MultiLayerLandConfig(surface_scheme=surf, soil_grid=SoilGridConfig()))
    config, _params, gsd = init_land_surface_data(
        args.surfdata, grid, base_cfg, 1.0)   # day_of_year (positional)
    update_land_params = make_step_land_params_updater(gsd, config.surface_scheme)
    n_layers = int(config.soil_grid.n_layers)
    # Layer thicknesses identify the column that a consumer must match: the layer
    # COUNT alone does not (8 layers can span 3 m or 6.375 m).  Recorded on every
    # restart this driver writes, and checked on every one it reads.
    from legoesm.land.soil_grid import make_soil_grid as _make_soil_grid
    soil_dz = _make_soil_grid(config.soil_grid).dz

    # Land mask for the equilibrium diagnostics (ocean columns are inert but
    # would dilute the land-mean drift).
    def _cover1d(a):
        a = np.asarray(a)
        return a[0] if a.ndim == 2 else a
    if args.land_mask_file:
        from legoesm.grids.topography import load_land_fraction
        land_frac = np.asarray(load_land_fraction(grid, args.land_mask_file)).ravel()
    else:
        land_frac = (_cover1d(gsd.f_land) + _cover1d(gsd.f_lake)
                     + _cover1d(gsd.f_glacier))
    land_mask = land_frac >= args.land_frac_min

    # State: resume from a restart, or cold-start.
    year0, t0_s = 0, 0.0
    if args.restart_from:
        loaded, meta = load_land_restart(
            args.restart_from, expected_land_mode="multilayer",
            expected_ncol=ncol, expected_n_layers=n_layers,
            expected_soil_dz=soil_dz)
        # Graft the restart's prognostic columns onto a canonical cold-start
        # template so the loaded state has the full structure the inner
        # lax.scan (step_multilayer_land output) requires — the restart only
        # round-trips the core fields; the optional structural fields are None.
        _template = init_multilayer_land_state(ncol, config, T_init=288.0)
        state = merge_land_restart_into_template(loaded, _template)
        t0_s = float(meta["t_end_s"])
        year0 = int(round(t0_s / sec_per_year))
        print(f"resumed from {args.restart_from}: year {year0}, t={t0_s:.0f}s")
    else:
        state = init_multilayer_land_state(ncol, config, T_init=288.0)

    # S3 spin-up cycles a fixed climate at FIXED cover; single-year surfdata makes
    # interp_annual return the one slice for any year.
    cover_year = jnp.asarray(float(np.asarray(gsd.years)[0]))
    run_year = _build_year_scan(
        smoke, step_multilayer_land, config, update_land_params, cover_year,
        lat_rad, lon_rad, ncol, dt, True, steps_per_year)

    print(f"spin-up: {args.grid_type} res{args.resolution} | {ncol} cols "
          f"({int(land_mask.sum())} land) | {args.years} yr x {steps_per_year} "
          f"steps @ dt={dt:.0f}s | n_layers={n_layers}")

    prev = {"T_deep": _land_mean(state.T_soil, land_mask),
            "theta": _land_mean(state.theta_soil, land_mask)}
    metrics = []
    converged = False
    for y in range(year0, year0 + args.years):
        state = run_year(state, float(y) * sec_per_year)
        jax.block_until_ready(state)
        cur = {"T_deep": _land_mean(state.T_soil, land_mask),
               "theta": _land_mean(state.theta_soil, land_mask)}
        d_T = cur["T_deep"] - prev["T_deep"]
        d_theta = cur["theta"] - prev["theta"]
        metrics.append({
            "year": y + 1, "T_soil_land_mean": cur["T_deep"],
            "theta_land_mean": cur["theta"],
            "drift_T_K_per_yr": d_T, "drift_theta_per_yr": d_theta,
        })
        print(f"  year {y+1:3d}: T_soil={cur['T_deep']:.3f} K "
              f"(drift {d_T:+.4f} K/yr) | theta={cur['theta']:.4f} "
              f"(drift {d_theta:+.5f}/yr)")
        prev = cur
        if abs(d_T) < args.equilibrium_tol_K:
            converged = True
        if (args.checkpoint_every_years > 0
                and (y + 1) % args.checkpoint_every_years == 0):
            ck = out_dir / f"land_spinup_year_{y+1:04d}.npz"
            save_land_restart(
                ck, state, land_mode="multilayer",
                t_end_s=float(y + 1) * sec_per_year,
                n_steps_completed=(y + 1) * steps_per_year,
                soil_dz=soil_dz,
                metadata={"grid_type": args.grid_type,
                          "resolution": args.resolution, "dt": dt})

    restart_out = Path(args.restart_out) if args.restart_out else out_dir / "land_ic.npz"
    save_land_restart(
        restart_out, state, land_mode="multilayer",
        t_end_s=float(year0 + args.years) * sec_per_year,
        n_steps_completed=(year0 + args.years) * steps_per_year,
        soil_dz=soil_dz,
        metadata={"grid_type": args.grid_type, "resolution": args.resolution,
                  "dt": dt, "years": year0 + args.years,
                  "surface_scheme": args.surface_scheme})

    # Verdict for #746 item 1: is the land converging (spin-up) or pinned
    # (structural)?  The LAST year's |drift| vs the tolerance decides.
    last = metrics[-1]
    verdict = ("CONVERGED" if abs(last["drift_T_K_per_yr"]) < args.equilibrium_tol_K
               else "NOT_CONVERGED")
    summary = {
        "grid_type": args.grid_type, "resolution": args.resolution,
        "ncol": ncol, "n_land": int(land_mask.sum()), "years": year0 + args.years,
        "dt": dt, "steps_per_year": steps_per_year,
        "equilibrium_tol_K": args.equilibrium_tol_K,
        "final_drift_T_K_per_yr": last["drift_T_K_per_yr"],
        "final_drift_theta_per_yr": last["drift_theta_per_yr"],
        "verdict": verdict, "restart": str(restart_out), "per_year": metrics,
    }
    (out_dir / "spinup_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nverdict: {verdict} (final drift "
          f"{last['drift_T_K_per_yr']:+.4f} K/yr, tol {args.equilibrium_tol_K}) "
          f"-> {restart_out}")
    print(f"wrote {out_dir/'spinup_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
