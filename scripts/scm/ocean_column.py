"""Ocean single-column model demonstrations on the legoESM ocean SCM.

Three idealized process studies that mirror the atmosphere SCM demo
scripts in this directory, driven through
:class:`legoesm.ocean.scm.OceanColumnModel`:

  * ``convection`` — convective adjustment of a statically unstable column
    (cold/dense surface over warm/light deep water); enhanced-diffusion
    convection mixes it back toward neutral while conserving heat.
  * ``heating``    — a prescribed surface heat flux warms the column with an
    exact column-heat-content budget.
  * ``ekman``      — wind-driven Ekman layer: the surface current veers to
    the right of the wind (northern hemisphere) and the inertial-period-
    averaged transport matches −τ/(ρf).  Demonstrates the stable
    Crank-Nicolson Coriolis sub-step.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/scm/ocean_column.py convection
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/scm/ocean_column.py ekman --hours 96
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig


def _heat_content(state, dz) -> float:
    T = np.asarray(state.T.data[0, 0, 0])
    return float(np.sum(T * dz)) * constants.rho_ocean * constants.c_sw


def run_convection(args) -> int:
    nlev = args.nlev
    T0 = jnp.linspace(8.0, 16.0, nlev)  # unstable: dense surface over light deep
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=args.dt, T_profile=T0,
        H_max=200.0, dz_surface=5.0, dz_deep=15.0,
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    hc0 = _heat_content(scm.state, dz)
    nsteps = int(args.hours * 3600.0 / args.dt)
    scm.run(nsteps=nsteps, save_every=max(1, nsteps // 10))
    Tf = np.asarray(scm.state.T.data[0, 0, 0])
    print(f"convection: {nsteps} steps, dt={args.dt:.0f}s")
    print(f"  T std   init {float(np.std(np.asarray(T0))):.4f} -> final {Tf.std():.4f}")
    print(f"  T range init {float(T0[0]):.2f}..{float(T0[-1]):.2f} "
          f"-> final {Tf.min():.3f}..{Tf.max():.3f}")
    print(f"  heat-content rel drift: {(_heat_content(scm.state, dz)-hc0)/abs(hc0):.2e}")
    return 0


def run_heating(args) -> int:
    nlev = args.nlev
    q = args.q_net
    scm = OceanColumnModel.create(
        physics_config=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            convection=OceanConvectionConfig(scheme="none"),
        ),
        nlev=nlev, dt=args.dt, T_profile=jnp.full((nlev,), 10.0),
        H_max=1000.0, forcing=OceanSCMForcing(q_net=lambda t: q),
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    hc0 = _heat_content(scm.state, dz)
    nsteps = int(args.hours * 3600.0 / args.dt)
    scm.run(nsteps=nsteps, save_every=max(1, nsteps // 10))
    dHC = _heat_content(scm.state, dz) - hc0
    expected = q * nsteps * args.dt
    print(f"heating: Q_net={q:.1f} W/m^2, {nsteps} steps")
    print(f"  surface T: 10.000 -> {float(scm.state.T.data[0,0,0,0]):.4f} degC")
    print(f"  d(heat content)={dHC:.4e}  expected={expected:.4e}  "
          f"rel_err={abs(dHC-expected)/abs(expected):.2e}")
    return 0


def run_ekman(args) -> int:
    nlev, f, tau = args.nlev, 1.0e-4, 0.1
    scm = OceanColumnModel.create(
        physics_config=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant"),
            convection=OceanConvectionConfig(scheme="none"),
        ),
        nlev=nlev, dt=args.dt, T_profile=jnp.full((nlev,), 15.0),
        H_max=800.0, latitude_deg=43.0,
        forcing=OceanSCMForcing(f_c=f, tau_x=lambda t: tau),
        A_v_background=1.0e-2,
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    Tin = 2.0 * np.pi / f
    nsteps = int(args.hours * 3600.0 / args.dt)
    scm.run(nsteps=nsteps, save_every=10**9)
    # average over one inertial period
    nper = int(Tin / args.dt)
    Us, Vs, us, vs = [], [], [], []
    for _ in range(nper):
        scm.step()
        u = np.asarray(scm.state.u.data[0, 0, 0])
        v = np.asarray(scm.state.v.data[0, 0, 0])
        Us.append(np.sum(u * dz)); Vs.append(np.sum(v * dz))
        us.append(u[0]); vs.append(v[0])
    My_th = -tau / (constants.rho_ocean * f)
    print(f"ekman: f={f:.1e}, tau_x={tau}, {nsteps} steps "
          f"({nsteps*args.dt/Tin:.1f} inertial periods spin-up)")
    print(f"  avg surface (u,v)=({np.mean(us):.4f}, {np.mean(vs):.4f}) m/s "
          f"(veers right of +x wind)")
    print(f"  avg transport My={np.mean(Vs):.4f}  theory={My_th:.4f}  "
          f"rel_err={abs(np.mean(Vs)-My_th)/abs(My_th):.3f}")
    print(f"  max|vel|={max(np.max(np.abs(np.asarray(scm.state.u.data))), np.max(np.abs(np.asarray(scm.state.v.data)))):.4f} "
          f"(bounded => CN-Coriolis stable)")
    return 0


_CASES = {"convection": run_convection, "heating": run_heating, "ekman": run_ekman}


def main(argv=None) -> int:
    jax.config.update("jax_enable_x64", True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=sorted(_CASES))
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--dt", type=float, default=1200.0, help="time step [s]")
    p.add_argument("--hours", type=float, default=120.0, help="duration [h]")
    p.add_argument("--q-net", dest="q_net", type=float, default=200.0,
                   help="surface heat flux [W/m^2] (heating case)")
    args = p.parse_args(argv)
    return _CASES[args.case](args)


if __name__ == "__main__":
    sys.exit(main())
