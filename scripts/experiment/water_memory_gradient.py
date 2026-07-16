"""Water-memory gradient: d(late-season GPP) / d(spring soil water), per cell.

Differentiable-model demonstration (paper "what new questions can we ask") — the
sensitivity of late-growing-season GPP to a per-cell perturbation of the spring soil
water state, obtained by ONE reverse-mode pass through a checkpointed May->Sep scan.
Land columns are independent (no lateral flow), so
    d( sum_i mask_i * late_GPP_i ) / d( dtheta_j )
is DIAGONAL = the per-cell memory map M_j = d(late_GPP_j)/d(dtheta_j) -- the whole
global field for ~one model cost.  Uncalibrated: the *pattern* (memory concentrated
in water-limited regions) is the capability demonstration, not the numbers.

Design (chosen with the user):
  * perturbation = initial column SOIL WATER (dtheta), psi kept consistent;
  * window = May 1 -> Sep 30 (fixed for now; per-cell growing season is a planned
    follow-up -- pass a per-cell ``late_flags`` when that lands);
  * spring state = a spin-up restart forward-integrated (NO grad) to the May-1
    perturbation point.

Feasibility (all cleared, see ``--selftest`` and tests/land/test_water_memory_gradient):
  A differentiability (finite grad through the two-leaf Newton solve; AD==FD),
  B NaN isolation (a bad cell's NaN stays in its own gradient entry -> mask output),
  C checkpointing (jax.checkpoint on the scan body: exact + bounds memory).

Real-data run (Derecho): assemble ``state0`` from a spin-up restart and
``forcing_window`` from the CRU-JRA loader (``run_lmip_biophys`` setup helpers), pass
the driver's per-step ``update_land_params`` as ``land_params_fn``, and the good-cell
mask from a forward NaN scan.  ``restart_1985`` requires the 1975-1985 spin-up to
finish; an interim restart gives an earlier pattern look.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_hydraulics import psi_from_theta
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _step_and_gpp(state, forcing_t, doy_t, config, lat, dt, land_params_fn):
    """One land step -> (new_state, gpp [gC/m2/s]).  Mirrors run_lmip_biophys's
    per-step body (per-step land_params from theta_top + doy)."""
    ncol = state.theta_soil.shape[0]
    lp = None if land_params_fn is None else land_params_fn(state.theta_soil[:, 0], doy_t)
    new_state, _resp, _c, sout = step_multilayer_land_with_diagnostics(
        state, forcing_t, config, U_min=1.0, dt=dt, lat=lat, doy=doy_t, land_params=lp)
    gpp = sout.gpp if sout.gpp is not None else jnp.zeros(ncol)
    return new_state, gpp


def spin_forward(state, forcing_seq, doy_seq, config, *, lat, dt, land_params_fn=None):
    """No-grad forward integration to reach the perturbation point (e.g. Jan->May).
    ``forcing_seq`` is a time-stacked AtmToSurface (leading axis = steps)."""
    def body(st, xs):
        Fi, doy_i = xs
        new_st, _ = _step_and_gpp(st, Fi, doy_i, config, lat, dt, land_params_fn)
        return new_st, None
    state_end, _ = jax.lax.scan(body, state, (forcing_seq, doy_seq))
    return jax.lax.stop_gradient(state_end)


def water_memory_map(
    state0, forcing_seq, doy_seq, late_flags, config, *,
    lat, dt, good_mask, land_params_fn=None, use_checkpoint=True,
):
    """Per-cell memory map M_i = d( sum over late steps of GPP_i ) / d(dtheta_i)
    [gC/m^2 per (m^3/m^3)], via one reverse-mode pass over the (checkpointed) window
    scan.  ``dtheta`` perturbs the initial column soil water (all layers), psi kept
    consistent.  ``late_flags`` (nsteps,) selects the steps counted in the late-GPP
    sum; ``good_mask`` (ncol,) excludes NaN cells (their gradient entry is left NaN and
    masked -- columns are independent, so it does not contaminate good cells).

    Returns ``(memory_map (ncol,), late_gpp (ncol,))``."""
    ncol = state0.theta_soil.shape[0]

    def loss(dtheta):
        theta_p = state0.theta_soil + dtheta[:, None]
        st = state0._replace(theta_soil=theta_p,
                             psi_soil=psi_from_theta(theta_p, config.hydraulics))

        def body(carry, xs):
            st, acc = carry
            Fi, doy_i, latef = xs
            new_st, gpp = _step_and_gpp(st, Fi, doy_i, config, lat, dt, land_params_fn)
            return (new_st, acc + jnp.where(latef, gpp, 0.0)), None

        body_fn = jax.checkpoint(body) if use_checkpoint else body
        (_stf, late_gpp), _ = jax.lax.scan(
            body_fn, (st, jnp.zeros(ncol)), (forcing_seq, doy_seq, late_flags))
        # sum only good cells; where() keeps the summed VALUE finite even if a bad cell
        # is NaN, and column independence confines a bad cell's NaN to its own grad entry
        return jnp.sum(jnp.where(good_mask, late_gpp, 0.0)) * dt, late_gpp

    # value_and_grad with has_aux -> ((value, aux), grad)
    (_val, late_gpp), M = jax.value_and_grad(loss, has_aux=True)(jnp.zeros(ncol))
    return M, late_gpp


def water_memory_map_percell(
    state0, forcing_seq, doy_seq, config, *,
    lat, dt, onset_step, late_lo_steps, late_hi_steps=None, good_mask,
    land_params_fn=None, use_checkpoint=True,
):
    """Per-cell GROWING-SEASON memory: inject the perturbation dtheta_i into cell i's
    soil water at ITS growing-season onset step s_i (snow-free + thawed), and sum GPP
    over the per-cell window [s_i + late_lo, s_i + late_hi) (late_hi=None -> to end).

        M_i = d( sum_{s_i+lo <= t < s_i+hi} GPP_i(t) ) / d( dtheta_i injected at s_i )

    Because dtheta_i enters the state only AT/AFTER thaw, the reverse-mode gradient
    never flows back through the near-singular spring-thaw canopy steps (TODO-1) -- so
    boreal/temperate cells are INCLUDED over their real growing season without the
    poleward contamination, structurally rather than by masking.  A BOUNDED window
    (late_hi finite) is a since-onset time bin -> the building block of the per-cell
    growing-season kernel.  Diagonal (columns independent).  state0 is the spun
    perturbation-point state.  Returns ``(M (ncol,), late_gpp (ncol,))``."""
    ncol = state0.theta_soil.shape[0]
    nsteps = int(np.asarray(forcing_seq.T_lowest).shape[0])
    onset = jnp.asarray(onset_step)
    lo = onset + int(late_lo_steps)
    hi = (onset + int(late_hi_steps)) if late_hi_steps is not None \
        else jnp.full_like(onset, nsteps + 1)
    steps = jnp.arange(nsteps)

    def loss(dtheta):
        def body(carry, xs):
            st, acc = carry
            Fi, doy_i, step_i = xs
            # inject dtheta into all soil layers at this cell's onset step; keep psi
            # consistent there (mixed-form Richards reads both psi and theta).
            is_on = (step_i == onset)                                  # (ncol,)
            theta_i = st.theta_soil + jnp.where(is_on[:, None], dtheta[:, None], 0.0)
            psi_i = jnp.where(is_on[:, None],
                              psi_from_theta(theta_i, config.hydraulics), st.psi_soil)
            st = st._replace(theta_soil=theta_i, psi_soil=psi_i)
            new_st, gpp = _step_and_gpp(st, Fi, doy_i, config, lat, dt, land_params_fn)
            in_win = (step_i >= lo) & (step_i < hi)                     # per-cell window
            acc = acc + jnp.where(in_win, gpp, 0.0)
            return (new_st, acc), None

        body_fn = jax.checkpoint(body) if use_checkpoint else body
        (_stf, late_gpp), _ = jax.lax.scan(
            body_fn, (state0, jnp.zeros(ncol)), (forcing_seq, doy_seq, steps))
        return jnp.sum(jnp.where(good_mask, late_gpp, 0.0)) * dt, late_gpp

    (_val, late_gpp), M = jax.value_and_grad(loss, has_aux=True)(jnp.zeros(ncol))
    return M, late_gpp


def growing_season_kernel(
    state0, forcing_seq, doy_seq, config, *,
    lat, dt, onset_step, bin_steps, n_bins, good_mask, land_params_fn=None,
    use_checkpoint=True,
):
    """Per-cell growing-season memory KERNEL, phenology-aligned:
      K[b, i] = d( GPP_i in [onset_i + b*bin, onset_i + (b+1)*bin) ) / d( dtheta_i @ onset )
    i.e. the memory decay binned by TIME-SINCE-ONSET (bin b = the b-th interval after
    green-up), so bin b is comparable across cells regardless of calendar onset.  One
    reverse pass per bin (each a bounded-window ``water_memory_map_percell``); by
    construction the bins sum to the full since-onset map (late_lo=0).  Late bins are
    empty (0) for cells whose window ends first.  Returns ``K (n_bins, ncol)``."""
    rows = []
    for b in range(n_bins):
        Kb, _ = water_memory_map_percell(
            state0, forcing_seq, doy_seq, config, lat=lat, dt=dt,
            onset_step=onset_step, late_lo_steps=b * bin_steps,
            late_hi_steps=(b + 1) * bin_steps, good_mask=good_mask,
            land_params_fn=land_params_fn, use_checkpoint=use_checkpoint)
        rows.append(np.asarray(Kb))
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Synthetic case (local selftest) — time-varying diurnal forcing, no external data
# ---------------------------------------------------------------------------
def synthetic_window(ncol, nsteps, dt=3600.0, seed=0):
    """A drying, warm, water-limited window with a DIURNAL forcing cycle (exercises
    the time-varying scanned-forcing path).  Zero precip -> extra initial soil water
    props up late-window GPP = the memory signal."""
    rng = np.random.default_rng(seed)
    lat = jnp.asarray(np.deg2rad(rng.uniform(25.0, 55.0, ncol)))
    hours = np.arange(nsteps) * (dt / 3600.0)
    day_phase = 2 * np.pi * (hours % 24) / 24.0
    sw = np.clip(700.0 * np.sin(day_phase - np.pi / 2), 0.0, None)      # noon-peaked
    cosz = np.clip(np.sin(day_phase - np.pi / 2), 0.05, 1.0)
    Tair = 293.0 + 6.0 * np.sin(day_phase - np.pi / 2 - 0.3)            # diurnal T
    doy_seq = jnp.asarray(180.0 + hours / 24.0)

    def col(v):  # (nsteps,) -> (nsteps, ncol)
        return jnp.asarray(np.repeat(v[:, None], ncol, axis=1))

    F = AtmToSurface(
        T_lowest=col(Tair), q_lowest=col(np.full(nsteps, 0.004)),   # dry air = ET demand
        u_lowest=col(np.full(nsteps, 3.0)), v_lowest=col(np.full(nsteps, 1.0)),
        p_lowest=col(np.full(nsteps, 97000.0)), p_surface=col(np.full(nsteps, 101325.0)),
        rho_lowest=col(np.full(nsteps, 1.15)), sw_down=col(sw),
        lw_down=col(np.full(nsteps, 350.0)), cos_zenith=col(cosz),
        precip_total=col(np.zeros(nsteps)), precip_snow=col(np.zeros(nsteps)),
        co2_ppmv=col(np.full(nsteps, 400.0)),
        has_radiation=jnp.ones(nsteps), has_precipitation=jnp.ones(nsteps))
    late_flags = jnp.asarray(np.arange(nsteps) >= int(nsteps * 0.5))
    return F, doy_seq, late_flags, lat


# Two-leaf canopy is the LMIP scheme AND the only one that produces GPP (SimpleSEB
# returns gpp=None -> a trivially-zero memory gradient).
_SYNTH_CFG = MultiLayerLandConfig(
    surface_scheme=TwoLeafCanopyConfig(max_iters=30, tol=1e-2), snow_scheme="single")


def _selftest(ncol=128, nsteps=240):
    cfg = _SYNTH_CFG
    F, doy_seq, late_flags, lat = synthetic_window(ncol, nsteps)
    # DRY start (theta_init 0.20, near wilting 0.15) so the column is water-limited and
    # the memory signal (d late-GPP / d spring water) is non-zero.
    st0 = init_multilayer_land_state(ncol, cfg, T_init=296.0, theta_init=0.20, TgC_init=23.0)
    good = jnp.ones(ncol, bool)
    dt = 3600.0
    print(f"=== selftest ncol={ncol} nsteps={nsteps} ({nsteps*dt/86400:.1f}d) ===")

    t = time.time()
    M, late = water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=dt,
                               good_mask=good, use_checkpoint=True)
    M = np.asarray(M.block_until_ready())
    print(f"[A] grad finite: {bool(np.all(np.isfinite(M)))} | "
          f"map range [{M.min():.3e}, {M.max():.3e}] | ({time.time()-t:.1f}s w/ compile)")

    def S(dtheta):
        _m, _ = water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=dt,
                                 good_mask=good, use_checkpoint=True)
        # value via a separate closure
        theta_p = st0.theta_soil + dtheta[:, None]
        st = st0._replace(theta_soil=theta_p, psi_soil=psi_from_theta(theta_p, cfg.hydraulics))
        def body(carry, xs):
            st, acc = carry
            Fi, doy_i, latef = xs
            ns, gpp = _step_and_gpp(st, Fi, doy_i, cfg, lat, dt, None)
            return (ns, acc + jnp.where(latef, gpp, 0.0)), None
        (_s, lg), _ = jax.lax.scan(body, (st, jnp.zeros(ncol)), (F, doy_seq, late_flags))
        return float(jnp.sum(lg) * dt)
    eps = 1e-4
    for k in (0, ncol - 1):
        fd = (S(jnp.zeros(ncol).at[k].set(eps)) - S(jnp.zeros(ncol).at[k].set(-eps))) / (2 * eps)
        print(f"    cell {k}: AD={M[k]:.4e} FD={fd:.4e} rel.err={abs(M[k]-fd)/(abs(fd)+1e-30):.2e}")

    M2, _ = water_memory_map(st0, F, doy_seq, late_flags, cfg, lat=lat, dt=dt,
                             good_mask=good, use_checkpoint=False)
    print(f"[C] checkpoint parity: max|ckpt-nockpt|={float(np.max(np.abs(np.asarray(M2)-M))):.2e}")

    Fbad = F._replace(sw_down=F.sw_down.at[:, 0].set(jnp.nan))
    Mb, _ = water_memory_map(st0, Fbad, doy_seq, late_flags, cfg, lat=lat, dt=dt,
                             good_mask=good.at[0].set(False), use_checkpoint=True)
    Mb = np.asarray(Mb)
    print(f"[B] NaN-cell isolation: good cells finite={bool(np.all(np.isfinite(Mb[1:])))} "
          f"(cell0={Mb[0]:.2e}, masked)")


# ---------------------------------------------------------------------------
# Real-data run (Derecho): spin-up restart + CRU-JRA forcing -> global memory map
# ---------------------------------------------------------------------------
# noleap calendar month-start day-of-year edges (Jan=1 ... Dec=12 via searchsorted)
_MONTH_STARTS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def monthly_memory_kernel(
    state0, forcing_seq, doy_seq, config, *,
    lat, dt, good_mask, land_params_fn=None, use_checkpoint=True,
):
    """Temporal memory kernel: K[m, i] = d( month-m integrated GPP_i ) / d(theta_i May)
    [gC/m^2 per (m^3/m^3)], for every calendar month spanned by the window.  One
    reverse pass PER MONTH (late_flags = that month's steps), reusing the exact tested
    ``water_memory_map`` path -- forward-mode jvp is unavailable because the canopy
    Newton solver is custom_vjp (reverse only).  By construction, summing K over the
    late-window months reproduces the integrated map (a consistency check).  Returns
    ``(month_labels (n_months,), K (n_months, ncol))``."""
    doy = np.asarray(doy_seq)
    month = np.searchsorted(_MONTH_STARTS, doy, side="right")   # 1..12 per step
    labels = sorted(set(int(m) for m in month))
    rows = []
    for m in labels:
        flags = jnp.asarray(month == m)
        Km, _ = water_memory_map(
            state0, forcing_seq, doy_seq, flags, config, lat=lat, dt=dt,
            good_mask=good_mask, land_params_fn=land_params_fn,
            use_checkpoint=use_checkpoint)
        rows.append(np.asarray(Km))
    return np.array(labels), np.stack(rows)


def _forward_window_diag(state0, forcing_seq, doy_seq, config, lat, dt, land_params_fn,
                         *, snow_thresh=1.0, freeze_margin=0.5, min_onset_doy=0.0):
    """No-grad forward pass over the window -> per-cell diagnostics:
      good       : GPP + state stay FINITE throughout (no revert guard here, so a NaN
                   cell stays NaN; exclude it from the summed loss).
      max_snow   : max snow_depth [kg/m2] over the window.
      min_tsoil  : min top-soil temperature [K] over the window.
      onset_step : FIRST window step at which the cell is snow-free (< snow_thresh) AND
                   thawed (T_soil_top > T_freeze + freeze_margin) AND doy >= min_onset_doy
                   -- the growing-season onset.  ``nsteps`` (sentinel) if never reached.
    max_snow/min_tsoil flag SNOW / FREEZE-THAW cells where the canopy Newton solve is
    near-singular (TODO-1) -> a contaminated *gradient* even with finite forward GPP.
    onset_step drives the per-cell growing-season mode, which injects the perturbation
    only AFTER thaw so the gradient never flows back through the singularity."""
    ncol = state0.theta_soil.shape[0]
    nsteps = int(np.asarray(forcing_seq.T_lowest).shape[0])
    steps = jnp.arange(nsteps)                                  # int64 under x64, int32 else

    def body(carry, xs):
        st, ok, msnow, mtsoil, onset, found = carry
        Fi, doy_i, step_i = xs
        # onset criterion on the INCOMING state (so injecting at onset_step lands on a
        # snow-free, thawed cell) -- first step meeting it, at/after min_onset_doy.
        is_free = ((st.snow_depth < snow_thresh)
                   & (st.T_soil[:, 0] > constants.T_freeze + freeze_margin)
                   & (doy_i >= min_onset_doy))
        newly = is_free & (~found)
        onset = jnp.where(newly, step_i, onset)
        found = found | is_free
        new_st, gpp = _step_and_gpp(st, Fi, doy_i, config, lat, dt, land_params_fn)
        ok = ok & jnp.isfinite(gpp) & jnp.isfinite(new_st.theta_soil).all(axis=-1)
        msnow = jnp.maximum(msnow, new_st.snow_depth)
        mtsoil = jnp.minimum(mtsoil, new_st.T_soil[:, 0])
        return (new_st, ok, msnow, mtsoil, onset, found), None

    # onset sentinel dtype MUST match ``steps`` (jnp.where promotes it in-loop),
    # else lax.scan rejects the carry as int32-in / int64-out under x64.
    init = (state0, jnp.ones(ncol, bool), jnp.zeros(ncol), jnp.full(ncol, 1e3),
            jnp.full(ncol, nsteps, dtype=steps.dtype), jnp.zeros(ncol, bool))
    (_stf, ok, msnow, mtsoil, onset, _f), _ = jax.lax.scan(
        body, init, (forcing_seq, doy_seq, steps))
    return ok, msnow, mtsoil, onset


def run_real(args) -> int:
    """Global memory map from a spin-up restart + real CRU-JRA forcing.  Reuses the
    LMIP driver's grid / surfdata / forcing / restart setup so the physics is
    identical to the production run."""
    jax.config.update("jax_enable_x64", True)
    import importlib.util
    from legoesm.land.config import resolve_land_config
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.soil_thermal import SoilThermalConfig
    from legoesm.land.boundary_data import (
        init_land_surface_data, make_step_land_params_updater)
    from legoesm.land.forcing import stage_forcing
    from legoesm.land.restart import load_land_restart, merge_land_restart_into_template

    # reuse the driver's grid helpers (local to run_lmip_biophys) by path import
    drv_path = Path(__file__).resolve().parents[1] / "run" / "run_lmip_biophys.py"
    spec = importlib.util.spec_from_file_location("_run_lmip_biophys", drv_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    grid = drv.make_grid(args.grid_type, args.resolution)
    lat_rad, lon_rad = drv.grid_latlon_rad(grid)
    ncol = int(lat_rad.shape[0])
    dt = float(args.dt)
    year = int(args.year)
    DAY = 86400.0

    # production LMIP physics (single snow scheme; two-leaf canopy + MOST + freeze/thaw)
    base_cfg = MultiLayerLandConfig(
        surface_scheme=TwoLeafCanopyConfig(max_iters=50, tol=1e-2),
        soil_grid=SoilGridConfig(), bulk_scheme="most", snow_scheme="single",
        snow_albedo_feedback=True, thermal=SoilThermalConfig(enable_freeze_thaw=True))
    base_cfg = resolve_land_config("multilayer", base_cfg)
    config, _p, gsd = init_land_surface_data(args.surfdata, grid, base_cfg, 0.0)

    update_lp = make_step_land_params_updater(gsd, config.surface_scheme)
    lp_fn = lambda theta_top, doy: update_lp(theta_top, doy, float(year))[0]

    # step times [s since Jan 1]: spin Jan1->perturb_doy (no grad); window perturb_doy->end
    n_spin = int(args.perturb_doy * DAY / dt)
    n_win = int((args.window_end_doy - args.perturb_doy) * DAY / dt)
    spin_t = dt * np.arange(n_spin)
    win_t = args.perturb_doy * DAY + dt * np.arange(n_win)
    fk = dict(year=year, data_dir=args.forcing_dir, prefix=args.prefix,
              suffix=args.suffix, k_neighbors=args.k_neighbors)
    print(f"grid={args.grid_type} R{args.resolution} | {ncol} cols | year {year} | "
          f"spin Jan1->doy{args.perturb_doy} ({n_spin} steps) | "
          f"window doy{args.perturb_doy}->{args.window_end_doy} ({n_win} steps) | "
          f"late-GPP from doy{args.late_doy}")

    F_spin = stage_forcing(lat_rad, lon_rad, spin_t, **fk)
    doy_spin = jnp.asarray(spin_t / DAY)

    # Jan-1 state: a spin-up restart (its resolution MUST match --resolution), or a
    # cold start seeded from the Jan-1 air temperature (Jan->May spin then partly
    # equilibrates the spring soil moisture).  Cold start decouples the demo
    # resolution from the (2 deg) spin-up restart -> a cheap 4 deg first look.
    template = init_multilayer_land_state(ncol, config, T_init=288.0)
    if args.restart:
        loaded, meta = load_land_restart(
            args.restart, expected_land_mode="multilayer", expected_ncol=ncol,
            expected_n_layers=config.soil_grid.n_layers)
        state_jan1 = merge_land_restart_into_template(loaded, template)
        print(f"restart: {args.restart} (t_end_s={meta['t_end_s']:.0f})")
    else:
        T0 = F_spin.T_lowest[0]                                 # first-step air temp
        state_jan1 = template._replace(
            T_soil=jnp.broadcast_to(T0[:, None], template.T_soil.shape))
        print("cold start (no restart): T_soil seeded from Jan-1 air temp; the "
              "Jan->May spin partly equilibrates spring soil moisture")

    state_p = spin_forward(state_jan1, F_spin, doy_spin, config,
                           lat=lat_rad, dt=dt, land_params_fn=lp_fn)
    del F_spin
    print("spin complete -> perturbation-point state")

    F_win = stage_forcing(lat_rad, lon_rad, win_t, **fk)
    doy_win = jnp.asarray(win_t / DAY)
    late_flags = jnp.asarray((win_t / DAY) >= args.late_doy)

    # per-cell onset: first snow-free/thawed window step at/after --onset-min-doy
    # (default = window start).  Drives the growing-season mode.
    min_onset = args.onset_min_doy if args.onset_min_doy is not None else args.perturb_doy
    good, max_snow, min_tsoil, onset_step = _forward_window_diag(
        state_p, F_win, doy_win, config, lat_rad, dt, lp_fn,
        snow_thresh=args.onset_snow_thresh, min_onset_doy=min_onset)
    max_snow = np.asarray(max_snow); min_tsoil = np.asarray(min_tsoil)
    onset_np = np.asarray(onset_step)
    nwin = int(win_t.shape[0])
    onset_found = onset_np < nwin                                  # reached onset in window
    onset_doy = np.where(onset_found, min_onset + onset_np * dt / DAY, np.nan)
    # snow-free AND never-froze over the window -> away from the spring-thaw canopy
    # singularity (TODO-1) that contaminates the boreal/arctic gradient.
    snowfree_warm = (max_snow < 1.0) & (min_tsoil > constants.T_freeze - 0.5)
    print(f"good cells (finite over window): {int(np.asarray(good).sum())}/{ncol} | "
          f"snow-free & non-freezing: {int(snowfree_warm.sum())} | "
          f"reached growing-season onset: {int(onset_found.sum())}")

    good_np = np.asarray(good)
    if args.growing_season:
        # per-cell mode: inject at each cell's onset, sum GPP over its late window.
        # Excludes cells that never reach onset (permanent snow/ice); no thaw in the
        # differentiated gradient path -> no snowfree mask needed.
        late_lag = int(args.late_lag_days * DAY / dt)
        good_np = good_np & onset_found
        print(f"growing-season mode: inject at per-cell onset, late window = onset+"
              f"{args.late_lag_days:.0f}d..end ({int(good_np.sum())} cells)")
        M, late_gpp = water_memory_map_percell(
            state_p, F_win, doy_win, config, lat=lat_rad, dt=dt,
            onset_step=onset_step, late_lo_steps=late_lag,
            good_mask=jnp.asarray(good_np), land_params_fn=lp_fn, use_checkpoint=True)
    else:
        M, late_gpp = water_memory_map(
            state_p, F_win, doy_win, late_flags, config, lat=lat_rad, dt=dt,
            good_mask=good, land_params_fn=lp_fn, use_checkpoint=True)
        if args.snowfree_only:
            good_np = good_np & snowfree_warm
    M_raw = np.asarray(M)
    # The forward good_mask only guarantees the forward STATE is finite -- but a cell
    # can have finite GPP yet a huge/NaN GRADIENT if the backward pass ran near a
    # singularity (arid-Richards near-zero moisture capacity; spring-thaw canopy Newton
    # -- the documented TODO-1/TODO-2 regimes). Exclude cells whose |gradient| is
    # non-finite or unphysically large (a clear singularity signature). Disclosed +
    # counted, not silently clipped.
    grad_ok = np.isfinite(M_raw) & (np.abs(M_raw) < args.max_abs_grad)
    keep = good_np & grad_ok
    n_fwd = int((~good_np).sum())
    n_grad = int((good_np & ~grad_ok).sum())
    M = np.where(keep, M_raw, np.nan)
    # water_memory_map returns late_gpp as sum of per-step GPP RATES [gC/m2/s];
    # multiply by dt for the time-integrated late-window GPP [gC/m2].
    late_gpp = np.where(keep, np.asarray(late_gpp) * dt, np.nan)
    print(f"memory map: kept {int(keep.sum())}/{ncol} | forward-excluded {n_fwd} | "
          f"gradient-blowup excluded {n_grad} (|M|>={args.max_abs_grad:g} or non-finite) "
          f"| range [{np.nanmin(M):.3e}, {np.nanmax(M):.3e}] gC/m2 per (m3/m3)")

    # --- temporal memory kernel (optional) ---
    months = kernel = None
    kernel_since_onset = False
    if args.kernel and args.growing_season:
        # per-cell growing-season kernel: bins by TIME-SINCE-ONSET (phenology-aligned,
        # thaw-clean).  One reverse pass per bin.
        bin_days = args.kernel_bin_days
        bin_steps = int(bin_days * DAY / dt)
        n_bins = int(np.ceil((args.window_end_doy - min_onset) / bin_days))
        print(f"computing growing-season kernel: {n_bins} bins of {bin_days:.0f}d "
              f"since onset (one reverse pass per bin) ...")
        kernel = growing_season_kernel(
            state_p, F_win, doy_win, config, lat=lat_rad, dt=dt,
            onset_step=onset_step, bin_steps=bin_steps, n_bins=n_bins,
            good_mask=jnp.asarray(good_np), land_params_fn=lp_fn, use_checkpoint=True)
        kernel = np.where(keep[None, :] & (np.abs(kernel) < args.max_abs_grad)
                          & np.isfinite(kernel), kernel, np.nan)
        months = np.arange(n_bins)                                  # bin index since onset
        kernel_since_onset = True
        for b, row in enumerate(kernel):
            print(f"  bin {b} (onset+{b*bin_days:.0f}..{(b+1)*bin_days:.0f}d): "
                  f"mean|K| over kept = {np.nanmean(np.abs(row)):.3e}")
    elif args.kernel:
        print("computing monthly memory kernel (one reverse pass per month) ...")
        months, kernel = monthly_memory_kernel(
            state_p, F_win, doy_win, config, lat=lat_rad, dt=dt,
            good_mask=good, land_params_fn=lp_fn, use_checkpoint=True)
        # same keep + per-element gradient-plausibility on each month
        kernel = np.where(keep[None, :] & (np.abs(kernel) < args.max_abs_grad)
                          & np.isfinite(kernel), kernel, np.nan)
        # consistency: kernel summed over the LATE months == the integrated map
        late_m = months >= int(np.searchsorted(_MONTH_STARTS, args.late_doy, side="right"))
        resid = np.nanmax(np.abs(np.nansum(kernel[late_m], axis=0) - M))
        for m, row in zip(months, kernel):
            print(f"  month {int(m):2d}: mean|dGPP/dtheta| over kept = "
                  f"{np.nanmean(np.abs(row)):.3e}")
        print(f"  consistency (sum_late-months kernel vs integrated map): "
              f"max|resid|={resid:.2e}")

    _write_map(args.out, M, late_gpp, keep, lat_rad, lon_rad,
               args.resolution, year, args, months=months, kernel=kernel,
               max_snow=max_snow, min_tsoil=min_tsoil, onset_doy=onset_doy,
               kernel_since_onset=kernel_since_onset)
    return 0


def _write_map(out, M, late_gpp, good, lat_rad, lon_rad, resolution, year, args,
               *, months=None, kernel=None, max_snow=None, min_tsoil=None,
               onset_doy=None, kernel_since_onset=False):
    import xarray as xr
    nlat, nlon = resolution, 2 * resolution
    lat = np.rad2deg(np.asarray(lat_rad)).reshape(nlat, nlon)[:, 0]
    lon = np.rad2deg(np.asarray(lon_rad)).reshape(nlat, nlon)[0, :]
    _ln = ("d(late-growing-season GPP) / d(growing-season-onset soil water) [per-cell]"
           if args.growing_season else "d(late-season GPP) / d(spring soil water)")
    data = {
        "dGPP_dtheta": (("lat", "lon"), M.reshape(nlat, nlon),
                        {"long_name": _ln, "units": "gC m-2 per (m3 m-3)"}),
        "late_gpp": (("lat", "lon"), late_gpp.reshape(nlat, nlon),
                     {"long_name": "late-window GPP (unperturbed)", "units": "gC m-2"}),
        "good_cell": (("lat", "lon"), good.reshape(nlat, nlon).astype("i1"),
                      {"long_name": "1 = finite over window (else masked)"}),
    }
    if max_snow is not None:
        data["max_snow"] = (("lat", "lon"), max_snow.reshape(nlat, nlon),
                            {"long_name": "max snow_depth over window (thaw-cell flag)",
                             "units": "kg m-2"})
        data["min_tsoil_top"] = (("lat", "lon"), min_tsoil.reshape(nlat, nlon),
                                 {"long_name": "min top-soil T over window (freeze flag)",
                                  "units": "K"})
    if onset_doy is not None:
        data["onset_doy"] = (("lat", "lon"), onset_doy.reshape(nlat, nlon),
                             {"long_name": "growing-season onset day-of-year (per cell)",
                              "units": "day"})
    coords = {"lat": ("lat", lat, {"units": "degrees_north"}),
              "lon": ("lon", lon, {"units": "degrees_east"})}
    if kernel is not None and kernel_since_onset:
        # phenology-aligned kernel: bins are time-since-onset (comparable across cells)
        data["dGPP_dtheta_since_onset"] = (
            ("bin", "lat", "lon"),
            kernel.reshape(kernel.shape[0], nlat, nlon),
            {"long_name": "d(GPP in bin since onset) / d(onset soil water) "
                          "[per-cell growing-season memory kernel]",
             "units": "gC m-2 per (m3 m-3)"})
        coords["bin"] = ("bin", np.asarray(months),
                         {"long_name": f"bins of {args.kernel_bin_days:.0f} days since "
                          "growing-season onset (0 = onset)"})
    elif kernel is not None:
        data["dGPP_dtheta_monthly"] = (
            ("month", "lat", "lon"),
            kernel.reshape(kernel.shape[0], nlat, nlon),
            {"long_name": "d(month GPP) / d(spring soil water) [temporal memory kernel]",
             "units": "gC m-2 per (m3 m-3)"})
        coords["month"] = ("month", np.asarray(months),
                           {"long_name": "calendar month (noleap)"})
    ds = xr.Dataset(
        data, coords=coords,
        attrs={"title": "water-memory gradient (differentiable land demo)",
               "Conventions": "CF-1.8", "year": year, "restart": args.restart or "",
               "perturb_doy": args.perturb_doy, "window_end_doy": args.window_end_doy,
               "late_doy": args.late_doy, "note": "uncalibrated; pattern is the demo"})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic feasibility checks (A/B/C) locally")
    ap.add_argument("--ncol", type=int, default=128)
    ap.add_argument("--nsteps", type=int, default=240)
    # --- real-data run (Derecho) ---
    ap.add_argument("--run", action="store_true", help="real-data global memory map")
    ap.add_argument("--restart", help="spin-up restart (.npz, Jan-1 state); its "
                    "resolution MUST match --resolution. Omit for a cold start.")
    ap.add_argument("--surfdata", help="surfdata NetCDF")
    ap.add_argument("--forcing-dir", dest="forcing_dir", help="CRU-JRA data dir")
    ap.add_argument("--prefix",
                    default="clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--year", type=int, default=1985)
    ap.add_argument("--grid-type", dest="grid_type", default="latlon")
    ap.add_argument("--resolution", type=int, default=45)          # 45 = 4deg
    ap.add_argument("--k-neighbors", dest="k_neighbors", type=int, default=4)
    ap.add_argument("--dt", type=float, default=3600.0)
    ap.add_argument("--perturb-doy", dest="perturb_doy", type=float, default=120.0)   # May 1
    ap.add_argument("--window-end-doy", dest="window_end_doy", type=float, default=273.0)  # Sep 30
    ap.add_argument("--late-doy", dest="late_doy", type=float, default=181.0)          # Jul 1
    ap.add_argument("--kernel", action="store_true",
                    help="also emit the monthly temporal memory kernel dGPP(month)/dtheta_May "
                         "(one reverse pass per month; shows the memory decay/timescale)")
    ap.add_argument("--max-abs-grad", dest="max_abs_grad", type=float, default=1e5,
                    help="mask cells whose |gradient| exceeds this (unphysical -> a "
                         "near-singularity in the arid/thaw regimes; default 1e5)")
    ap.add_argument("--snowfree-only", dest="snowfree_only", action="store_true",
                    help="restrict the map to cells that are snow-free AND never freeze "
                         "over the window -> excludes the spring-thaw canopy singularity "
                         "(TODO-1) contaminating the boreal/arctic gradient")
    ap.add_argument("--growing-season", dest="growing_season", action="store_true",
                    help="PER-CELL growing-season mode: inject the perturbation at each "
                         "cell's snow-free/thawed onset and sum GPP over its late window "
                         "-> structurally avoids the thaw singularity (supersedes "
                         "--snowfree-only; includes boreal cells over their real season)")
    ap.add_argument("--late-lag-days", dest="late_lag_days", type=float, default=45.0,
                    help="growing-season mode: late window starts onset + this many days")
    ap.add_argument("--onset-min-doy", dest="onset_min_doy", type=float, default=None,
                    help="growing-season mode: earliest allowed onset doy (default = "
                         "--perturb-doy, the window start)")
    ap.add_argument("--onset-snow-thresh", dest="onset_snow_thresh", type=float,
                    default=1.0, help="snow_depth [kg/m2] below which a cell is 'snow-free'")
    ap.add_argument("--kernel-bin-days", dest="kernel_bin_days", type=float, default=30.0,
                    help="growing-season kernel (--kernel --growing-season): bin width in "
                         "days since onset (default 30 = ~monthly)")
    ap.add_argument("--out", default="results/water_memory/memory_map.nc")
    args = ap.parse_args(argv)

    if args.selftest:
        jax.config.update("jax_enable_x64", True)
        _selftest(args.ncol, args.nsteps)
        return 0
    if args.run:
        # --restart is OPTIONAL (cold start if omitted); surfdata + forcing required.
        missing = [f"--{k}" for k in ("surfdata", "forcing_dir")
                   if not getattr(args, k.replace("-", "_"), None)]
        if missing:
            ap.error(f"--run needs {', '.join(missing)}")
        return run_real(args)
    ap.error("use --selftest (local) or --run --restart ... --surfdata ... "
             "--forcing-dir ... (Derecho). See the module docstring.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
