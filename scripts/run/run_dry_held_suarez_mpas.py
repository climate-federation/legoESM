#!/usr/bin/env python
"""Dry Held-Suarez on the MPAS Voronoi dycore — 4th grid control.

Completes the grid comparison: cube cd-grid super-rotates (no midlat jets),
spectral gives textbook midlat jets, lat-lon collapses axisymmetric.  Does the
MPAS Voronoi dycore produce midlat jets (viable) or super-rotate (like cube)?
Also tests --nu-del2-scale (the MPAS Laplacian viscosity, analog of the cube's
A_h eddy-killer).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig)
from legoesm.atmosphere.held_suarez import (
    held_suarez_init_mpas, held_suarez_forcing_mpas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)   # ~10242 cells ~ C32
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--nu-del2-scale", type=float, default=0.0,
                    help="Scale the MPAS del2 (Laplacian) viscosity (A_h analog; "
                         "default 0 = off, the cleanest eddy test).")
    ap.add_argument("--nu-del4-div", type=float, default=86400.0,
                    help="nu_del4 = dx^4/div. Smaller div = MORE scale-selective "
                         "biharmonic damping for stability (864000 blows up ~day70).")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    mesh = create_voronoi_mesh(args.level)
    sigma = create_sigma_coordinate(args.nlev)
    dx = float(np.sqrt(np.mean(np.asarray(mesh.areaCell))))
    nu_del4 = dx ** 4 / args.nu_del4_div   # scale-selective biharmonic
    nu_del2 = args.nu_del2_scale * 0.01 * dx ** 2
    print(f"Dry HS MPAS level{args.level} ({len(np.asarray(mesh.latCell))} cells) / L{args.nlev}, "
          f"{args.days}d, dt={args.dt}s, nu_del2={nu_del2:.2e} nu_del4={nu_del4:.2e}")
    print(f"  JAX devices: {jax.devices()}")

    cfg = MPASPrimitiveEquationConfig(nu_del2=nu_del2, nu_del4=nu_del4, fix_mass=True)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    state = held_suarez_init_mpas(mesh, sigma)

    @jax.jit
    def step(s):
        return model.step(s, args.dt, held_suarez_forcing_mpas)

    lat = np.asarray(mesh.latCell) * 180.0 / np.pi  # (nCells,)
    area = np.asarray(mesh.areaCell)
    sig_full = np.asarray(sigma.sigma_full)
    k_sfc = int(np.argmax(sig_full))
    bins = np.arange(-90, 91, 5.0); bc = 0.5 * (bins[:-1] + bins[1:])
    band = [(lat >= bins[b]) & (lat < bins[b+1]) for b in range(len(bc))]
    n_steps = int(args.days * 86400 / args.dt)
    diag_every = max(1, int(1 * 86400 / args.dt))

    def diag(s):
        ue, vn = reconstruct_cell_velocity(s.u.data, mesh)  # (nCells,nlev) east/north
        ue = np.asarray(ue); vn = np.asarray(vn)
        uz = np.full((len(bc), ue.shape[1]), np.nan)
        eke_num = 0.0; w = 0.0
        for b in range(len(bc)):
            m = band[b]
            if m.sum() < 3:
                continue
            ub = np.average(ue[m], axis=0, weights=area[m])
            vb = np.average(vn[m], axis=0, weights=area[m])
            uz[b] = ub
            ek = 0.5 * ((ue[m]-ub)**2 + (vn[m]-vb)**2)
            eke_num += np.sum(area[m][:, None]*ek); w += np.sum(area[m])*ek.shape[1]
        return uz, eke_num/w

    t0 = time.time(); state = step(state); jax.block_until_ready(state)
    print(f"  JIT done ({time.time()-t0:.1f}s)")
    eke_t = []
    for i in range(1, n_steps):
        state = step(state)
        if i % diag_every == 0:
            jax.block_until_ready(state)
            uz, eke = diag(state); day = (i+1)*args.dt/86400
            jk = np.unravel_index(np.nanargmax(uz), uz.shape)
            mw = float(np.nanmax(np.abs(uz)))
            if not np.isfinite(mw) or mw > 1000:
                print(f"  *** BLOWUP day {day:.1f}"); break
            eke_t.append((day, eke))
            if i % (diag_every*5) == 0:
                print(f"  day {day:6.1f}: jet_umax={uz[jk]:6.1f} m/s @lat{bc[jk[0]]:.0f}  eddyKE={eke:.3e}")

    uz, eke = diag(state)
    np.savez(args.out, uz=uz, lat_bins=bc, sigma_full=sig_full,
             eke_t=np.array(eke_t) if eke_t else np.zeros((0, 2)))
    if not np.isfinite(uz).any():
        print("FINAL state non-finite (blew up); eke_t saved. Last good day:",
              eke_t[-1][0] if eke_t else "none")
        return
    jk = np.unravel_index(np.nanargmax(uz), uz.shape)
    print(f"\nFINAL MPAS HS jet: zonal-mean u max = {uz[jk]:.1f} m/s at lat {bc[jk[0]]:.0f}")
    print("Surface zonal-mean u by lat:")
    for b in range(0, len(bc), 3):
        print(f"  lat {bc[b]:+5.0f}: {uz[b, k_sfc]:+6.1f} m/s")
    print(f"midlat jets + surface westerlies => MPAS viable. Saved {args.out}")


if __name__ == "__main__":
    main()
