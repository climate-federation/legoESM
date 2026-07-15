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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic feasibility checks (A/B/C) locally")
    ap.add_argument("--ncol", type=int, default=128)
    ap.add_argument("--nsteps", type=int, default=96)
    args = ap.parse_args(argv)
    if args.selftest:
        jax.config.update("jax_enable_x64", True)
        _selftest(args.ncol, args.nsteps)
        return 0
    ap.error("real-data --run path is wired on Derecho (needs a spin-up restart + "
             "CRU-JRA forcing); use --selftest locally. See the module docstring.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
