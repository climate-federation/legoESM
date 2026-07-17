#!/usr/bin/env python
"""Dry Held-Suarez (1994) on the cubed sphere — pure dynamical-core jet test.

NO radiation / convection / moist physics / ERA5 IC: just HS Newtonian
relaxation + Rayleigh drag on the SAME cd-grid dycore the AMIP runs use
(built via create_atmosphere_dycore from a real experiment config so the
diffusion/divergence-damping match the proven-stable AMIP setup), started
from the HS analytic isothermal-rest state.

Diagnoses whether the dycore MAINTAINS a realistic eddy-driven jet
(~28 m/s at ~45 deg, sigma~0.25) or spuriously dissipates it (the AMIP
wind-collapse symptom).  Healthy dry-HS jet => dycore is fine, AMIP collapse
is a physics/IC coupling problem.  Weak dry-HS jet => dycore over-dissipates;
resolution-test the fix with --n 96.

Usage:
    JAX_ENABLE_X64=1 python scripts/run/run_dry_held_suarez_cube.py \
        --config <amip experiment_config.json> --n 32 --days 200 --out out.npz
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
from legoesm.driver.config import load_experiment_config
from legoesm.driver.component_factory import create_atmosphere_dycore
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init, held_suarez_forcing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="/scratch/b/b309165/amip_c32l45_orogwd_1979_60d_25800992/experiment_config.json")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--nlev", type=int, default=45)
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--hyperdiff-scale", type=float, default=1.0,
                    help="Scale the dycore biharmonic hyperdiffusion (probe over-dissipation).")
    ap.add_argument("--div-damp-scale", type=float, default=1.0,
                    help="Scale the dycore divergence damping.")
    ap.add_argument("--vcoord", type=str, default="hybrid", choices=["hybrid", "sigma"],
                    help="Vertical coordinate (sigma matches the spectral control).")
    ap.add_argument("--ah-scale", type=float, default=1.0,
                    help="Scale the 2nd-order Laplacian viscosity A_h (the suspected "
                         "eddy-killer; default A_h damps a 3000km eddy in ~5h). "
                         "0 = off, rely on scale-selective hyperdiffusion.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    # Match the requested resolution (the dycore diffusion is recomputed for the
    # passed grid inside create_atmosphere_dycore).
    if getattr(cfg.grid, "resolution", None) != args.n:
        cfg = cfg._replace(grid=cfg.grid._replace(resolution=args.n))
    # Optional diffusion-strength probe (is the cd-grid over-dissipating the jet?).
    if args.hyperdiff_scale != 1.0 or args.div_damp_scale != 1.0:
        cfg = cfg._replace(dycore=cfg.dycore._replace(
            hyperdiff_scale=args.hyperdiff_scale,
            div_damp_scale=args.div_damp_scale))
        print(f"  diffusion probe: hyperdiff_scale={args.hyperdiff_scale} "
              f"div_damp_scale={args.div_damp_scale}")
    dt = args.dt if args.dt is not None else float(cfg.dycore.dt)

    print(f"Dry Held-Suarez (AMIP dycore): C{args.n} / L{args.nlev}, {args.days} d, dt={dt}s")
    print(f"  config: {args.config}")
    print(f"  JAX devices: {jax.devices()}")

    grid = create_cubed_sphere(args.n)
    if args.vcoord == "sigma":
        from legoesm.grids.vertical import create_sigma_coordinate
        sigma = create_sigma_coordinate(args.nlev)
    else:
        sigma = standard_hybrid_levels(args.nlev)
    model = create_atmosphere_dycore(cfg, grid, sigma)
    if args.ah_scale != 1.0:
        old = model.config.A_h
        model.config = model.config._replace(A_h=old * args.ah_scale)
        print(f"  A_h override: {old:.3e} -> {model.config.A_h:.3e} m2/s (scale {args.ah_scale})")
    state = held_suarez_init(grid, sigma)

    @jax.jit
    def step(s):
        return model.step(s, dt, physics_fn=held_suarez_forcing)

    n_steps = int(args.days * 86400 / dt)
    diag_every = max(1, int(2 * 86400 / dt))

    cosA = np.asarray(grid.cos_angle); sinA = np.asarray(grid.sin_angle)
    latd = np.asarray(grid.lat) * 180.0 / np.pi
    bins = np.arange(-90, 91, 5.0); bc = 0.5 * (bins[:-1] + bins[1:])

    # precompute lat-band masks + area weights for eddy-KE (deviation from
    # zonal symmetry) — the cleanest "are baroclinic eddies present" metric.
    area = np.asarray(grid.area)
    band_masks = [(latd >= bins[b]) & (latd < bins[b+1]) for b in range(len(bc))]

    def jet_diag(s):
        u = np.asarray(s.u.data); v = np.asarray(s.v.data)
        ue = u * cosA[..., None] - v * sinA[..., None]
        uz = np.full((len(bc), u.shape[-1]), np.nan)
        for b in range(len(bc)):
            if band_masks[b].any():
                uz[b] = ue[band_masks[b]].mean(0)
        return uz

    def eddy_ke(s):
        """Area-weighted eddy KE [m2/s2] = deviation of (u,v) from the
        zonal (lat-band) mean, vertically averaged.  Tiny => zonally
        symmetric flow (no baroclinic eddies)."""
        u = np.asarray(s.u.data); v = np.asarray(s.v.data)
        eke_num = 0.0; w_sum = 0.0
        for b in range(len(bc)):
            m = band_masks[b]
            if m.sum() < 3:
                continue
            uw = area[m][:, None]
            ub = np.average(u[m], axis=0, weights=area[m])
            vb = np.average(v[m], axis=0, weights=area[m])
            ek = 0.5 * ((u[m] - ub) ** 2 + (v[m] - vb) ** 2)  # (ncell,nlev)
            eke_num += np.sum(uw * ek)
            w_sum += np.sum(uw) * ek.shape[1]
        return eke_num / w_sum

    # sample eddy KE daily so the growth/decay curve resolves the
    # baroclinic-instability e-folding (~0.5-1 /day if active).
    diag_every = max(1, int(1 * 86400 / dt))
    eke_t = []  # (day, eddy_ke)

    t0 = time.time()
    state = step(state); jax.block_until_ready(state)
    print(f"  JIT done ({time.time()-t0:.1f}s)")

    for i in range(1, n_steps):
        state = step(state)
        if i % diag_every == 0:
            jax.block_until_ready(state)
            mw = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            day = (i+1)*dt/86400
            if not np.isfinite(mw) or mw > 1000:
                print(f"  *** BLOWUP day {day:.1f} max_wind={mw:.1f}"); break
            eke = float(eddy_ke(state)); eke_t.append((day, eke))
            if i % (diag_every*5) == 0:
                uz = jet_diag(state); jk = np.unravel_index(np.nanargmax(uz), uz.shape)
                print(f"  day {day:6.1f}: max_wind={mw:5.1f}  jet_east_max={uz[jk]:5.1f} m/s @lat{bc[jk[0]]:.0f}  eddyKE={eke:.3e} m2/s2")

    uz = jet_diag(state)
    eke_arr = np.array(eke_t) if eke_t else np.zeros((0, 2))
    np.savez(args.out, uz=uz, lat_bins=bc, eke_t=eke_arr,
             sigma_full=np.asarray(sigma.sigma_full),
             u=np.asarray(state.u.data), v=np.asarray(state.v.data),
             T=np.asarray(state.T.data), p_s=np.asarray(state.p_s.data),
             lat=latd, cos_angle=cosA, sin_angle=sinA)
    jk = np.unravel_index(np.nanargmax(uz), uz.shape)
    print(f"\nFINAL dry-HS jet: eastward zonal-mean max = {uz[jk]:.1f} m/s "
          f"at lat {bc[jk[0]]:.0f}, sigma_lev {jk[1]} (sigma={float(sigma.sigma_full[jk[1]]):.3f})")
    print(f"HS94 benchmark: ~28 m/s at ~45 deg.  Saved {args.out}")


if __name__ == "__main__":
    main()
