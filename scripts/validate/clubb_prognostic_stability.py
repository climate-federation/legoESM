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
    per-piece port is faithful. The standalone SCM driver advances the means with
    CLUBB alone (no dynamical-core numerical diffusion to damp grid-scale noise),
    so part of this is a driver-exposure artifact a coupled run would damp.

Run: ``JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/clubb_prognostic_stability.py``
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402

from legoesm import constants  # noqa: E402


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
    ap.add_argument("--nsteps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=200.0)
    ap.add_argument("--surface-dt", type=float, default=1.0)
    args = ap.parse_args()
    characterize(dt=args.dt, nsteps=args.nsteps, heat=args.surface_dt)
