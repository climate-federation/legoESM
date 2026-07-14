"""Radiative-convective equilibrium (RCE) demo for the SCM (issue #277).

Runs a tropical single column with gray radiation + Louis turbulence +
Kessler microphysics + simple mass-flux convection for ``--days`` model
days, then reports final-state statistics. This is the canonical
sanity test for any SCM: temperature should equilibrate to a moist
adiabat under the heating/cooling balance, with surface T near the
prescribed initial value and stratospheric T near the radiation
equilibrium temperature (~200 K for the gray scheme).

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/matrix/run_scm_test_matrix.py rce --days 50

Exits non-zero if the final state is non-finite or fails coarse RCE
sanity checks: surface T in (240, 320) K; stratospheric T in
(150, 270) K; full column q_v >= -1e-8 kg/kg (positivity floor).
"""

from __future__ import annotations

import argparse
import sys

import jax.numpy as jnp

from legoesm.atmosphere.physics import (
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
    MicrophysicsConfig,
    ConvectionConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel


def build_initial_profiles(nlev: int, p_s: float, T_sfc: float):
    """Moist tropical sounding: tropospheric lapse 6.5 K/km capped at
    200 K, plus near-saturated near-surface humidity decaying upward.
    """
    sigma = jnp.linspace(0.01, 1.0, nlev)
    H = 8.0e3
    z = -H * jnp.log(jnp.maximum(sigma, 1e-3))
    T = jnp.maximum(T_sfc - 6.5e-3 * z, 200.0)
    q_v = 1.8e-2 * jnp.exp(-z / 3.0e3)
    return T, q_v


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--days", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=600.0, help="physics dt [s]")
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--latitude-deg", type=float, default=0.0)


def run(args: argparse.Namespace) -> int:
    T0, qv0 = build_initial_profiles(args.nlev, p_s=1.0e5, T_sfc=args.T_sfc)

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="mass_flux"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )

    nsteps = int((args.days * 86400.0) / args.dt)
    print(f"[SCM RCE] nlev={args.nlev}  dt={args.dt} s  "
          f"nsteps={nsteps}  days={args.days}  lat={args.latitude_deg}")

    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=args.nlev, dt=args.dt,
        T_profile=T0, q_v_profile=qv0,
        latitude_deg=args.latitude_deg,
        time_integrator="forward_euler",
    )

    final, hist = scm.run(
        nsteps=nsteps, save_every=max(1, nsteps // 20),
    )

    T_final = final.T.data[0, 0, 0]
    q_final = final.tracers["q_v"].data[0, 0, 0]
    T_sfc_f = float(T_final[-1])
    T_top_f = float(T_final[0])
    q_sfc_f = float(q_final[-1])
    p_s_f = float(final.p_s.data[0, 0, 0])

    print(f"[result] T_sfc_final = {T_sfc_f:.2f} K   "
          f"T_top_final = {T_top_f:.2f} K")
    print(f"[result] q_v_sfc_final = {q_sfc_f*1e3:.3f} g/kg   "
          f"p_s_final = {p_s_f:.1f} Pa")
    print(f"[history] T_sfc trajectory (subsampled): "
          f"{[float(t[-1]) for t in hist.T]}")

    ok = True
    if not jnp.all(jnp.isfinite(T_final)):
        print("[FAIL] non-finite final T"); ok = False
    if not (240.0 < T_sfc_f < 320.0):
        print(f"[FAIL] T_sfc out of plausible RCE range: {T_sfc_f}"); ok = False
    if not (150.0 < T_top_f < 270.0):
        print(f"[FAIL] T_top out of plausible RCE range: {T_top_f}"); ok = False
    if not jnp.all(q_final >= -1e-8):
        print("[FAIL] q_v went substantially negative"); ok = False

    print("[OK] RCE sanity passed" if ok else "[FAIL] RCE sanity failed")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
