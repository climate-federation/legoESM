#!/usr/bin/env python
"""Dry Held-Suarez on the LAT-LON C-grid dycore — third grid control.

Cube cd-grid super-rotates (no midlat jets); spectral gives textbook midlat jets.
Lat-lon is another GRID-POINT dycore: if it ALSO gives midlat jets like spectral,
the failure is SPECIFIC to the cubed-sphere cd-grid formulation; if it super-rotates
like the cube, the issue is broader (grid-point / shared code).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse, time, math
import numpy as np
import jax, jax.numpy as jnp
from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
    hydrostatic_to_cgrid, cgrid_to_hydrostatic)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_init_latlon, held_suarez_forcing_latlon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlat", type=int, default=64)
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    n_lat, n_lon = args.nlat, 2 * args.nlat

    grid = create_latlon_grid(n_lat, n_lon)
    sigma = create_sigma_coordinate(args.nlev)
    R = float(grid.radius)
    dx_pole = R * grid.dlon * math.cos(math.pi / 2 - grid.dlat / 2)
    dt = min(200.0, 0.8 * dx_pole / 300.0)
    A_h = min(0.01 * (R * math.pi / n_lat) ** 2, 0.4 * dx_pole ** 2 / dt)
    print(f"Dry HS LAT-LON {n_lat}x{n_lon} / L{args.nlev}, {args.days}d, dt={dt:.1f}s, A_h={A_h:.2e}")
    print(f"  JAX devices: {jax.devices()}")

    cfg = CGridLatLonPrimitiveEquationConfig(A_h=A_h, fix_mass=True)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    state = hydrostatic_to_cgrid(held_suarez_init_latlon(grid, sigma), grid)

    @jax.jit
    def step(s):
        return model.step(s, dt, physics_fn=held_suarez_forcing_latlon)

    lat = np.asarray(grid.lat) * 180.0 / np.pi
    sig_full = np.asarray(sigma.sigma_full)
    k_sfc = int(np.argmax(sig_full))
    n_steps = int(args.days * 86400 / dt)
    diag_every = max(1, int(1 * 86400 / dt))

    def diag(s):
        h = cgrid_to_hydrostatic(s, grid)
        u = np.asarray(h.u.data); v = np.asarray(h.v.data)  # (nlat,nlon,nlev)
        uz = u.mean(1)
        up = u - uz[:, None, :]; vp = v - v.mean(1)[:, None, :]
        eke = float(np.average(0.5 * (up**2 + vp**2).mean((1, 2)),
                               weights=np.cos(np.deg2rad(lat))))
        return uz, eke

    t0 = time.time(); state = step(state); jax.block_until_ready(state)
    print(f"  JIT done ({time.time()-t0:.1f}s)")
    eke_t = []
    for i in range(1, n_steps):
        state = step(state)
        if i % diag_every == 0:
            jax.block_until_ready(state)
            uz, eke = diag(state); day = (i+1)*dt/86400
            jk = np.unravel_index(np.nanargmax(uz), uz.shape)
            mw = float(np.nanmax(np.abs(uz)))
            if not np.isfinite(mw) or mw > 1000:
                print(f"  *** BLOWUP day {day:.1f}"); break
            eke_t.append((day, eke))
            if i % (diag_every*5) == 0:
                print(f"  day {day:6.1f}: jet_umax={uz[jk]:6.1f} m/s @lat{lat[jk[0]]:.0f} "
                      f"lev{jk[1]}  eddyKE={eke:.3e}")

    uz, eke = diag(state)
    np.savez(args.out, uz=uz, lat=lat, sigma_full=sig_full,
             eke_t=np.array(eke_t) if eke_t else np.zeros((0, 2)))
    jk = np.unravel_index(np.nanargmax(uz), uz.shape)
    print(f"\nFINAL latlon HS jet: zonal-mean u max = {uz[jk]:.1f} m/s at lat {lat[jk[0]]:.0f}")
    print("Surface zonal-mean u by lat:")
    for j in range(0, len(lat), max(1, len(lat)//10)):
        print(f"  lat {lat[j]:+5.0f}: {uz[j, k_sfc]:+6.1f} m/s")
    print(f"midlat jets + surface westerlies => grid-point OK, cube-SPECIFIC. Saved {args.out}")


if __name__ == "__main__":
    main()
