#!/usr/bin/env python
"""Dry Held-Suarez on the SPECTRAL (Gaussian) dycore — grid-comparison control.

The cd-grid cubed-sphere dry HS produces spurious EQUATORIAL SUPER-ROTATION with
no midlatitude eddy-driven jets (surface easterlies everywhere).  The equator runs
along cubed-sphere panel edges — a classic grid-imprinting / super-rotation source.

The spectral transform method has NO grid imprinting and is the dycore that DEFINES
the Held-Suarez benchmark (~28 m/s midlat jets at ~45 deg).  If spectral here gives
proper midlatitude jets, the cube dycore is the culprit; if it ALSO super-rotates,
the problem is in the forcing/setup, not the grid.

spectral_pe_to_grid returns TRUE eastward/northward winds on the Gaussian grid, so
the zonal mean is a clean lon-average (no rotation needed).

Usage:
    JAX_ENABLE_X64=1 python scripts/run/run_dry_held_suarez_spectral.py \
        --nmax 42 --nlev 30 --days 200 --out out.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
from legoesm.core.precision import set_policy, PrecisionPolicy
set_policy(PrecisionPolicy.fp64())  # spectral transforms require float64
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPrimitiveEquationModel, SpectralPEConfig,
    isothermal_rest_state_spectral, spectral_pe_to_grid)
from legoesm.atmosphere.held_suarez import held_suarez_forcing_spectral


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=42)
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"Dry Held-Suarez SPECTRAL T{args.nmax} / L{args.nlev}, {args.days} d, dt={args.dt}s")
    print(f"  JAX devices: {jax.devices()}")

    grid = create_gaussian_grid(args.nmax)
    sigma = create_sigma_coordinate(args.nlev)
    cfg = SpectralPEConfig(
        hyperdiff_coeff=2.338e15 * (21.0 / args.nmax) ** 4,
        spectral_filter_order=8, spectral_filter_strength=0.01)
    model = SpectralPrimitiveEquationModel(grid, sigma, cfg)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

    @jax.jit
    def step(s):
        return model.step(s, args.dt, physics_fn=held_suarez_forcing_spectral)

    lat = np.asarray(grid.lat) * 180.0 / np.pi  # (nlat,)
    n_steps = int(args.days * 86400 / args.dt)
    diag_every = max(1, int(1 * 86400 / args.dt))

    sig_full = np.asarray(sigma.sigma_full)
    k_sfc = int(np.argmax(sig_full))  # lowest model level (sigma nearest 1)

    def diag(s):
        f = spectral_pe_to_grid(s, grid, sigma)
        u = np.asarray(f['u']); v = np.asarray(f['v'])  # (nlat,nlon,nlev) eastward
        uz = u.mean(1)  # zonal mean over lon -> (nlat,nlev)
        # eddy KE: deviation from zonal mean, cos-lat weighted, lon+vert mean
        up = u - uz[:, None, :]; vp = v - v.mean(1)[:, None, :]
        eke_lat = 0.5 * (up**2 + vp**2).mean((1, 2))  # per lat
        eke = float(np.average(eke_lat, weights=np.cos(np.deg2rad(lat))))
        return uz, eke

    t0 = time.time(); state = step(state); jax.block_until_ready(state)
    print(f"  JIT done ({time.time()-t0:.1f}s)")
    eke_t = []
    for i in range(1, n_steps):
        state = step(state)
        if i % diag_every == 0:
            jax.block_until_ready(state)
            uz, eke = diag(state); day = (i+1)*args.dt/86400
            jk = np.unravel_index(np.nanargmax(uz), uz.shape)  # (lat_idx, lev_idx)
            mw = float(np.nanmax(np.abs(uz)))
            if not np.isfinite(mw) or mw > 1000:
                print(f"  *** BLOWUP day {day:.1f}"); break
            eke_t.append((day, eke))
            if i % (diag_every*5) == 0:
                print(f"  day {day:6.1f}: jet_umax={uz[jk]:6.1f} m/s @lat{lat[jk[0]]:.0f} "
                      f"sig_lev{jk[1]}  eddyKE={eke:.3e}")

    uz, eke = diag(state)
    np.savez(args.out, uz=uz, lat=lat, sigma_full=np.asarray(sigma.sigma_full),
             eke_t=np.array(eke_t) if eke_t else np.zeros((0,2)))
    jk = np.unravel_index(np.nanargmax(uz), uz.shape)  # (lat_idx, lev_idx)
    # surface zonal-mean wind by latitude (eddy-driven westerlies?)
    print(f"\nFINAL spectral HS jet: zonal-mean u max = {uz[jk]:.1f} m/s at lat {lat[jk[0]]:.0f}")
    print("Surface (lowest level) zonal-mean u by lat:")
    for j in range(0, len(lat), max(1, len(lat)//10)):
        print(f"  lat {lat[j]:+5.0f}: {uz[j, k_sfc]:+6.1f} m/s")
    print(f"HS94 benchmark: ~28 m/s midlat jet @~45 deg + surface westerlies @midlat. Saved {args.out}")


if __name__ == "__main__":
    main()
