"""Silvestri et al. 2024 §4 driver — 2D decaying turbulence, one (scheme, N) run.

Integrates the 2D vorticity harness (``ocean/experiments/silvestri_turbulence_2d``)
from the Ishiko IC to t=6 (~18 eddy turnovers, T_e≈0.33) and reports the paper's
Fig-4/5 metrics:

  * TIME SERIES: integrated kinetic energy and enstrophy.
  * SPECTRA at t=3.6 (~11 turnovers): isotropic energy + enstrophy spectra.
  * SNAPSHOTS: vorticity at t=3.6 and final (Fig 3).

Schemes (Table 1): DNS / Leith1 / Leith2 / W5D / W9D / W5V / W9V. The paper's
DNS is 4096²; coarse runs 64²→1024². W9V should converge to DNS at the coarsest
resolution and keep more energy than Leith. One (scheme, N) per invocation.

Usage::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_silvestri_turbulence_2d.py \\
        --scheme W9V --N 256 --t-end 6 --out results/silvestri_turb2d
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def run(scheme_name, N, t_end, cfl, Re, out, seed, tag=""):
    import os
    import jax
    import jax.numpy as jnp
    import legoesm.ocean.experiments.silvestri_turbulence_2d as T2
    import legoesm.ocean.diagnostics as D

    os.makedirs(out, exist_ok=True)
    scheme = T2.SILVESTRI_TURB2D_SCHEMES[scheme_name]
    rhs = T2.make_rhs(scheme, Re=Re, N=N)
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=seed))

    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    dx = 2.0 * np.pi / N

    # Fixed dt from a CFL on the IC velocity (max |u| is ~initial in decaying turb).
    u0, v0 = T2.velocity_from_vorticity(zeta, kx, ky, k2_inv)
    umax0 = float(jnp.max(jnp.sqrt(u0 ** 2 + v0 ** 2)))
    dt = cfl * dx / max(umax0, 1e-6)
    n_steps = int(np.ceil(t_end / dt))
    record_every = max(1, n_steps // 120)
    snap_step = int(round(3.6 / dt))            # snapshot/spectra at t≈3.6

    import functools

    @jax.jit
    def _step(z):
        return T2.step_ssp_rk3(z, dt, rhs)

    print(f"== Turb2D §4: {scheme_name} N={N} Re={Re:g} t_end={t_end} | "
          f"T_e≈{T2.target_enstrophy()**-0.5:.3f} dt={dt:.2e} steps={n_steps} "
          f"umax0={umax0:.3f} ==", flush=True)

    ts, kes, enss = [], [], []
    ke0, ens0 = T2.total_energy_enstrophy(zeta, N)
    ts.append(0.0); kes.append(ke0); enss.append(ens0)
    zeta_36 = None
    t0 = time.time()
    for s in range(1, n_steps + 1):
        zeta = _step(zeta)
        if s == snap_step:
            jax.block_until_ready(zeta)
            zeta_36 = np.asarray(zeta)
        if s % record_every == 0 or s == n_steps:
            jax.block_until_ready(zeta)
            ke, ens = T2.total_energy_enstrophy(zeta, N)
            ts.append(s * dt); kes.append(ke); enss.append(ens)
            if not np.isfinite(ke):
                print(f"   *** BLEW UP at step {s} (t={s*dt:.2f})"); break
    finite = bool(np.isfinite(float(jnp.sum(zeta))))
    print(f"   done {n_steps} steps in {time.time()-t0:.0f}s; "
          f"KE {ke0:.3e}->{kes[-1]:.3e}  enstrophy {ens0:.3e}->{enss[-1]:.3e}", flush=True)

    # Isotropic spectra at t≈3.6 (the paper's comparison time).
    if zeta_36 is None:
        zeta_36 = np.asarray(zeta)
    z36 = jnp.asarray(zeta_36)
    u36, v36 = T2.velocity_from_vorticity(z36, kx, ky, k2_inv)
    k_e, P_e = D.isotropic_energy_spectrum(u36, v36, dx)
    k_z, P_z = D.isotropic_enstrophy_spectrum(z36, dx)

    np.savez_compressed(
        f"{out}/turb2d_{scheme_name}_N{N}{tag}.npz",
        scheme=scheme_name, N=N, Re=Re, t=np.array(ts), ke=np.array(kes),
        enstrophy=np.array(enss), k_energy=np.asarray(k_e), P_energy=np.asarray(P_e),
        k_enstrophy=np.asarray(k_z), P_enstrophy=np.asarray(P_z),
        zeta_36=zeta_36, zeta_final=np.asarray(zeta))
    print(f"VERDICT scheme={scheme_name} N={N} | stable={finite} "
          f"KE_final={kes[-1]:.4e} enstrophy_final={enss[-1]:.4e} "
          f"KE_ratio={kes[-1]/ke0:.3f} ens_ratio={enss[-1]/ens0:.3f}", flush=True)
    return "STABLE" if finite else "BLEW UP"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="W9V",
                    choices=("DNS", "Leith1", "Leith2", "W5D", "W9D", "W5V", "W9V"))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--t-end", type=float, default=6.0)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--Re", type=float, default=3.3e4)
    ap.add_argument("--out", default="results/silvestri_turb2d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    run(args.scheme, args.N, args.t_end, args.cfl, args.Re, args.out, args.seed,
        tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
