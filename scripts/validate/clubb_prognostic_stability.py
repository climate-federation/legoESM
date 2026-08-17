"""Characterize the prognostic CLUBB single-column stability envelope.

Sweeps the dry-static stratification ``dtheta/dz`` (and optionally surface
heating) and integrates the *prognostic* CLUBB closure
(:func:`legoesm.atmosphere.physics.turbulence.clubb.integrate_clubb_column`)
for many steps, reporting per-case whether the carried ``wp2`` stays bounded and
the mean-temperature grid-scale-oscillation metric ``osc =
sum|T[k-1]-2T[k]+T[k+1]|`` (a 2-dz checkerboard detector).

Purpose (iter 49): pin the boundary of the multi-step instability discovered in
iter 48. Findings so far:
  * The MOIST / well-stratified regime (``dtheta/dz >= ~4e-3``) is stable.
  * A deep, near-dry, weakly-stratified column under SUSTAINED surface heating
    grows ``wp2`` and develops grid-scale ``T`` noise. The growth worsens with
    step COUNT at fixed physical time -> a genuine growing mode, NOT a
    Courant/forward-Euler timestep limit.
  * The wp2 buoyancy-production sign, the ``tau`` family, and
    ``calc_stability_correction`` all match CLUBB-JAX bit/round-off — i.e. the
    per-piece port is faithful.

RETRACTED (2026-08-12, #1508). This file used to add: "The standalone SCM driver
advances the means with CLUBB alone (no dynamical-core numerical diffusion to
damp grid-scale noise), so part of this is a driver-exposure artifact a coupled
run would damp." That is wrong. The growth was CLUBB's own missing surface
second-moment boundary condition (``calc_sfc_varnce``): with the level-0 row
left at its initial seed, the Cauchy-Schwarz floor pinned the surface theta_l
variance near 9e2 K^2. With the BC ported, the ``--mode production`` column runs
a full simulated day (it reached non-finite T in 92 steps before), and the bare
nu=0 dry column stays bounded without the host-diffusion stand-in.

Run: ``JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/clubb_prognostic_stability.py``
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    clubb_turbulence,
    clubb_turbulence_prognostic,
    init_clubb_moments,
    pack_clubb_moments,
    CLUBBMomentState,
)
from legoesm.atmosphere.physics._shared import virtual_temperature  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402

from legoesm import constants  # noqa: E402


def _production_column(ncol=1, nlev=30, p_s=1.0e5):
    """A mid-latitude column on the PRODUCTION MPAS grid (#1508).

    L30 sigma, troposphere + stratosphere, realistic moisture — the vertical
    structure is production-SHAPED — it is a synthetic single column, NOT a
    column extracted from an MPAS/AMIP run — as opposed to the deep dry
    idealisations the sweep above uses.
    """
    sigma_half = np.linspace(0.0, 1.0, nlev + 1) ** 1.2   # denser near surface
    p_half = np.tile(np.maximum(sigma_half * p_s, 1.0e3), (ncol, 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    # A monotone T(p): 288 K at the surface falling to ~252 K at the model
    # top.  Shape parameters are a synthetic profile, not a fit — this is a
    # stand-in for a real sounding, and no result here should depend on
    # their exact values.
    T = np.maximum(288.0 - 65.0 * (1.0 - (p_full / p_s) ** 0.19), 200.0)
    # RH ~ 80% near the surface, decaying with height (model saturation curve,
    # never a re-derived Tetens — a divergent curve manufactures supersaturation).
    q_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p_full)))
    q_v = np.maximum(0.8 * q_sat * (p_full / p_s) ** 2, 1.0e-6)
    z_half = np.tile(
        -8000.0 * np.log(np.maximum(p_half[0] / p_s, 1e-6)), (ncol, 1))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return dict(u=jnp.full((ncol, nlev), 10.0), v=jnp.zeros((ncol, nlev)),
                T=jnp.asarray(T), q_v=jnp.asarray(q_v),
                p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
                z_full=jnp.asarray(z_full), z_half=jnp.asarray(z_half),
                q_sfc=jnp.asarray(q_v[:, -1]))


def production(dt=75.0, nsteps=1152, nlev=30, sfc_dT=2.0, arm="prognostic",
               report_every=48):
    """Drive a PRODUCTION scheme entry at the #1508 configuration.

    ``prognostic`` (``clubb_turbulence_prognostic``, which reached the dycore
    floor inside one simulated day in AMIP) vs ``diagnostic``
    (``clubb_turbulence``, which ran cleanly from the identical state).  Same
    column, dt, step count and forcing.

    **What this can and cannot show.**  It is a BARE column: the means are
    advanced by forward Euler from the CLUBB tendency alone, with no dynamical
    core, no advective resupply, no moisture fixer and — the one that matters —
    no host numerical diffusion.  So a blowup here localises an instability in
    the bare closure+driver COMBINATION, not necessarily a defect in the closure
    alone, and it does NOT establish the production failure mode; the diagnostic
    arm is the control.

    This paragraph used to say that a missing host diffusion "has already
    produced one FALSE instability on this scheme (iter 48-51)".  Retracted
    (#1508): that instability was real and was the missing surface
    second-moment BC.  With :func:`calc_sfc_varnce` ported, the bare nu=0 dry
    column is bounded without the stand-in.

    The two arms are the two production entries, not one flag on one function:
    they carry different state (a single ``wp2`` field vs 15 packed moments).
    That is the comparison of interest, but it is not a one-variable A/B.
    """
    col = _production_column(nlev=nlev)
    cfg = CLUBBConfig(prognostic=(arm == "prognostic"))
    T_sfc = col["T"][:, -1] + sfc_dT
    idx = {n: i for i, n in enumerate(CLUBBMomentState._fields)}
    moments = pack_clubb_moments(
        init_clubb_moments(1, nlev, cfg, dtype=col["T"].dtype))
    wp2_diag = jnp.full((1, nlev), cfg.tke_min, dtype=col["T"].dtype)

    @jax.jit
    def _step(u, v, T, q, m, wp2d):
        rho = col["p_full"] / (constants.R_d * virtual_temperature(T, q))
        if arm == "prognostic":
            out, m = clubb_turbulence_prognostic(
                u, v, T, q, m, col["p_full"], col["p_half"], col["z_full"],
                col["z_half"], T_sfc, col["q_sfc"], rho, dt, cfg)
        else:
            out, wp2d = clubb_turbulence(
                u, v, T, q, wp2d, col["p_full"], col["p_half"], col["z_full"],
                col["z_half"], T_sfc, col["q_sfc"], rho, dt, cfg)
        return (u + dt * out.du_dt, v + dt * out.dv_dt, T + dt * out.dT_dt,
                q + dt * out.dq_v_dt, m, wp2d)

    def _wp2_max(m, wp2d):
        """The arms carry wp2 in DIFFERENT places; reading the packed array in
        the diagnostic arm would report an untouched init value (and reading
        field 6 reports up2, not wp2 — a mislabelled column, pre-merge codex)."""
        if arm == "prognostic":
            return float(np.max(np.abs(np.asarray(m)[:, idx["wp2"], :])))
        return float(np.max(np.abs(np.asarray(wp2d))))

    print(f"# arm={arm} dt={dt} nsteps={nsteps} nlev={nlev} sfc_dT={sfc_dT}")
    print(f"# {'step':>6} {'day':>6} {'Tmin':>8} {'Tmax':>9} {'|u|max':>9} "
          f"{'wp2max':>10} {'thlp2_sfc':>11} {'qmax':>9}")
    u, v, T, q, m, wp2d = (col["u"], col["v"], col["T"], col["q_v"],
                           moments, wp2_diag)
    for k in range(nsteps):
        u, v, T, q, m, wp2d = _step(u, v, T, q, m, wp2d)
        # Test finiteness FIRST: a nan-aware reduction would hide the very
        # failure this exists to find.
        blew = not bool(np.isfinite(np.asarray(T)).all())
        if blew or k % report_every == 0 or k == nsteps - 1:
            Ta, ua = np.asarray(T), np.asarray(u)
            # The diagnostic arm carries no moment state, so its packed array
            # is still the untouched seed: printing it would read as a healthy
            # thlp2 rather than as "not applicable" (pre-merge codex).
            thlp2_sfc = (f"{float(np.asarray(m)[0, idx['thlp2'], 0]):11.4e}"
                         if arm == "prognostic" else f"{'n/a':>11}")
            print(f"  {k + 1:6d} {(k + 1) * dt / 86400.0:6.2f} "
                  f"{np.min(Ta):8.2f} {np.max(Ta):9.2f} "
                  f"{np.max(np.abs(ua)):9.3f} {_wp2_max(m, wp2d):10.3e} "
                  f"{thlp2_sfc} {np.max(np.asarray(q)):9.2e}")
        if blew:
            print(f"  BLOWUP at step {k + 1} "
                  f"(day {(k + 1) * dt / 86400.0:.3f})")
            return False
    return True


def _column(ncol, nlev, dtheta_dz, q0):
    z_half = jnp.asarray(np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.asarray(np.tile(np.linspace(2.0e4, 1.0e5, nlev + 1), (ncol, 1)))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    theta = 290.0 + dtheta_dz * np.asarray(z_full)
    T = jnp.asarray(theta * np.asarray(exner))
    u = jnp.full((ncol, nlev), 8.0)
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.full((ncol, nlev), q0)
    return dict(u=u, v=v, T=T, q_v=q_v, p_full=p_full, p_half=p_half,
                z_full=z_full, z_half=z_half, q_sfc=jnp.full((ncol,), q0))


def characterize(dt=200.0, nsteps=40, nlev=24, q0=1e-5, heat=1.0):
    cfg = CLUBBConfig()
    print(f"# dt={dt} nsteps={nsteps} nlev={nlev} q0={q0} surface_dT={heat}")
    print(f"# {'dtheta/dz':>10} {'wp2max':>10} {'osc':>8} {'Tmin':>7} {'verdict':>9}")
    for dthdz in (8e-3, 4e-3, 2e-3, 1e-3):
        col = _column(2, nlev, dthdz, q0)
        T_sfc = col["T"][:, -1] + heat
        _, _, T_f, _, m_f, _ = integrate_clubb_column(
            col["u"], col["v"], col["T"], col["q_v"], col["p_full"],
            col["p_half"], col["z_full"], col["z_half"], T_sfc, col["q_sfc"],
            dt, nsteps, cfg)
        Ta = np.asarray(T_f)[0]
        osc = float(np.sum(np.abs(Ta[:-2] - 2 * Ta[1:-1] + Ta[2:])))
        wp2max = float(np.max(np.asarray(m_f.wp2)))
        verdict = "STABLE" if (wp2max < 100.0 and Ta.min() > 100.0) else "UNSTABLE"
        print(f"  {dthdz:10.0e} {wp2max:10.3f} {osc:8.2f} {Ta.min():7.1f} {verdict:>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sweep", "production"), default="sweep",
                    help="sweep: the dtheta/dz stability envelope (iter 49). "
                         "production: the #1508 config through the real scheme "
                         "entry, prognostic vs diagnostic.")
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--surface-dt", type=float, default=None,
                    help="T_sfc minus the lowest model level [K].")
    ap.add_argument("--arm", choices=("prognostic", "diagnostic"),
                    default="prognostic")
    args = ap.parse_args()
    if args.mode == "production":
        ok = production(dt=args.dt if args.dt is not None else 75.0,
                        nsteps=args.nsteps if args.nsteps is not None else 1152,
                        nlev=args.nlev,
                        sfc_dT=(args.surface_dt if args.surface_dt is not None
                                else 2.0),
                        arm=args.arm)
        raise SystemExit(0 if ok else 1)
    characterize(dt=args.dt if args.dt is not None else 200.0,
                 nsteps=args.nsteps if args.nsteps is not None else 40,
                 heat=args.surface_dt if args.surface_dt is not None else 1.0)
