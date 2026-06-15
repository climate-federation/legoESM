"""Silvestri et al. 2024 §5 baroclinic-jet driver — one (scheme, resolution) run.

REWRITTEN 2026-06-15 to use the CANONICAL recipe + scheme presets + Phase-2
diagnostics (the prior version inlined the grid/IC/thermal-wind/sponge — a
parallel system, now replaced by ``ocean/experiments/silvestri_baroclinic_jet.py``
and ``ocean/experiments/silvestri_schemes.py``).

Integrates ``build_silvestri_baroclinic_jet_setup`` forward for ``--days`` (paper:
1000) applying the τ=50-day zonal-mean restoring each step, and reports the
paper's Fig-7/8/9/10 metrics:

  * TIME SERIES: total KE, eddy KE, eddy APE.
  * ZONAL SPECTRA (final state): eddy energy + enstrophy.
  * EFFECTIVE RESOLUTION: zonal-mean buoyancy section; deformation radius L_d.
  * SNAPSHOT: surface relative vorticity (Fig 7).
  * gridscale_frac (top-quartile-wavenumber energy) + saturation flag.

W9V should be the most energetic (lowest implicit dissipation); the dispersive
SM2/QG2 less so, with ringing at fine resolution. Emits a parseable VERDICT and
saves the metric arrays for ``scripts/plot/plot_silvestri_comparison.py``.

One (scheme, resolution) per invocation — matrix-friendly (pin a GPU per job,
re-launch on kill). Schemes: UP3 / W9V / W9D / SM2 / QG2.

Usage::

    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 .venv/bin/python \\
        scripts/run/run_silvestri_baroclinic_jet.py --scheme W9V --resolution 80x64 \\
        --days 1000 --dt 900
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np

_SPD = 86400.0


def _metrics(state, area, h, N2, g_alphaT, T_ref):
    """(TKE, EKE, eddy_APE) volume integrals from the prognostic state."""
    import legoesm.ocean.diagnostics as D
    u_h, v_h = D.velocity_to_cell_centre(state.u.data, state.v.data)
    b = g_alphaT * (state.T.data - T_ref)                 # buoyancy [m/s²]
    tke = float(D.domain_kinetic_energy(u_h, v_h, area, h))
    eke = float(D.eddy_kinetic_energy(u_h, v_h, area, h))
    ape = float(D.eddy_available_potential_energy(b, N2, area, h))
    return tke, eke, ape


def _final_spatial(state, grid, dx, g_alphaT, T_ref):
    """Zonal eddy-energy + enstrophy spectra, zonal-mean buoyancy, surface
    vorticity (Fig 7), gridscale_frac."""
    import jax.numpy as jnp
    import legoesm.ocean.diagnostics as D
    u_h, v_h = D.velocity_to_cell_centre(state.u.data, state.v.data)
    zeta = D.relative_vorticity_cell_centre(state.u.data, state.v.data, grid)
    # Surface-layer (k=0) zonal spectra (paper averages the top 200 m).
    k_e, P_eu = D.zonal_power_spectrum(u_h[..., 0], dx)
    _, P_ev = D.zonal_power_spectrum(v_h[..., 0], dx)
    P_energy = 0.5 * (np.asarray(P_eu) + np.asarray(P_ev))
    k_z, P_z = D.zonal_power_spectrum(zeta[..., 0], dx)
    P_ens = 0.5 * np.asarray(P_z)
    b = g_alphaT * (state.T.data - T_ref)
    bzm = np.asarray(D.zonal_mean(b))                     # (n_lat, nlev)
    zeta_surf = np.asarray(zeta[..., 0])                 # (n_lat, n_lon) Fig 7
    Ptot = float(np.sum(P_energy[1:]))
    khi = len(P_energy) - max(1, len(P_energy) // 4)
    gs_frac = float(np.sum(P_energy[khi:]) / Ptot) if Ptot > 0 else 0.0
    return (np.asarray(k_e), P_energy, np.asarray(k_z), P_ens, bzm, zeta_surf, gs_frac)


def _deformation_radius_km(state, cfg, T_ref, g_alphaT):
    """L_d = N·H/(π|f₀|) from the column-mean stratification (Eq 54 estimate)."""
    from legoesm import constants
    b = g_alphaT * (np.asarray(state.T.data) - T_ref)
    dz = cfg.H_max / b.shape[-1]
    N2_col = np.clip(np.abs(np.gradient(b, dz, axis=-1)), 1e-12, None)
    N_bar = np.sqrt(np.mean(N2_col))
    f0 = abs(2 * constants.Omega * np.sin(np.radians(cfg.lat_center)))
    return float(N_bar * cfg.H_max / (np.pi * f0) / 1000.0)


def run(scheme, resolution, days, dt, out, nlev, tag="", stabilize=False):
    import os
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
        build_silvestri_baroclinic_jet_setup, SilvestriJetConfig, restore_state,
    )
    from legoesm.ocean.experiments.silvestri_schemes import scheme_label
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm import constants

    parts = resolution.split("x")
    n_lat = int(parts[0]); n_lon = int(parts[1]) if len(parts) > 1 else int(parts[0])
    os.makedirs(out, exist_ok=True)

    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        cfg_j = SilvestriJetConfig()
        recipe = build_silvestri_baroclinic_jet_setup(
            n_lat=n_lat, n_lon=n_lon, scheme=scheme, nlev=nlev, config=cfg_j,
            stabilize=stabilize)
        grid, z, cfg, state, restoring = (
            recipe.grid, recipe.z_coord, recipe.model_config,
            recipe.initial_state, recipe.restoring)
        model = LatLonCGridOceanModel(grid, z, cfg)

        area = grid.area
        h = jnp.broadcast_to(jnp.asarray(z.dz_ref), state.T.data.shape)
        g_alphaT = constants.g * cfg_j.alpha_T
        dx = float(constants.R_earth * np.cos(np.radians(cfg_j.lat_center)) * grid.dlon)

        if cfg.outer_integrator == "ab2" and state.T_incr_prev is None:
            def _z(d):
                return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                             dims=d.dims, units=d.units)
            state = state._replace(
                T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
                u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))

        steps_per_block = max(1, int(round(_SPD / dt)))

        def _body(s, _):
            s = model.step(s, dt)
            s = restore_state(s, restoring, dt)          # τ=50d zonal-mean restoring
            return s, None

        @partial(jax.jit, static_argnames=("n",))
        def _block(s, n):
            s, _ = jax.lax.scan(_body, s, None, length=n)
            return s

        print(f"== Silvestri jet: {scheme_label(scheme)} {n_lat}x{n_lon}, dt={dt:.0f}s, "
              f"{days:.0f}d, nlev={nlev}, fp64 | mom={cfg.momentum_advection} "
              f"smooth={cfg.weno_smoothness} friction={cfg.lateral_friction_scheme} | "
              f"N2={cfg_j.N2:g} Db={cfg_j.delta_b:g} tau={cfg_j.restoring_timescale_days:g}d ==",
              flush=True)

        ts, tkes, ekes, apes, umax_s = [], [], [], [], []
        m = _metrics(state, area, h, cfg_j.N2, g_alphaT, cfg_j.T_ref_C)
        ts.append(0.0); tkes.append(m[0]); ekes.append(m[1]); apes.append(m[2])
        umax_s.append(float(jnp.max(jnp.abs(state.u.data))))
        t0 = time.time(); blew = False
        n_days = int(round(days))
        for d in range(1, n_days + 1):
            state = _block(state, steps_per_block)
            jax.block_until_ready(state.u.data)
            umax = float(jnp.max(jnp.abs(state.u.data)))
            tke, eke, ape = _metrics(state, area, h, cfg_j.N2, g_alphaT, cfg_j.T_ref_C)
            ts.append(d * _SPD); tkes.append(tke); ekes.append(eke)
            apes.append(ape); umax_s.append(umax)
            if not bool(jnp.all(jnp.isfinite(state.u.data))) or umax > 50.0:
                print(f"   *** BLEW UP at day {d}: max|u|={umax:.3e}"); blew = True; break
            if d % 25 == 0 or d == n_days:
                print(f"   day {d:4d}: TKE={tke:.3e} EKE={eke:.3e} APE={ape:.3e} "
                      f"max|u|={umax:.3f}  {time.time()-t0:5.0f}s", flush=True)

        ts = np.array(ts); tkes = np.array(tkes); ekes = np.array(ekes); apes = np.array(apes)
        sat, eke_sat = False, float("nan")
        if not blew and len(ekes) >= 9:
            tail = ekes[-max(3, len(ekes) // 3):]
            eke_sat = float(tail.mean())
            cv = float(tail.std() / tail.mean()) if tail.mean() > 0 else 9.9
            sat = (cv < 0.25) and (eke_sat > 0.3 * ekes.max())

        k_e = P_e = k_z = P_z = bzm = zeta_surf = None
        gs_frac = float("nan"); Ld = float("nan")
        if not blew:
            k_e, P_e, k_z, P_z, bzm, zeta_surf, gs_frac = _final_spatial(
                state, grid, dx, g_alphaT, cfg_j.T_ref_C)
            Ld = _deformation_radius_km(state, cfg_j, cfg_j.T_ref_C, g_alphaT)

        print(f"   SATURATION: saturated={sat} EKE_sat={eke_sat:.3e} "
              f"L_d={Ld:.2f}km gridscale_frac={gs_frac:.4f}", flush=True)
        print(f"VERDICT scheme={scheme} res={n_lat}x{n_lon} days={n_days} dt={dt:.0f} | "
              f"stable={not blew} saturated={sat} TKE_sat={tkes[-1]:.3e} "
              f"EKE_sat={eke_sat:.3e} L_d_km={Ld:.2f} gridscale_frac={gs_frac:.4f} "
              f"max_u={umax_s[-1]:.3f}", flush=True)

        np.savez_compressed(
            f"{out}/silvestri_jet_{scheme}_{n_lat}x{n_lon}{tag}.npz",
            scheme=scheme, t=ts, tke=tkes, eke=ekes, ape=apes,
            umax=np.array(umax_s), blew=blew, saturated=sat, eke_sat=eke_sat,
            L_d_km=Ld, gridscale_frac=gs_frac,
            k_energy=k_e, P_energy=P_e, k_enstrophy=k_z, P_enstrophy=P_z,
            zonal_mean_buoyancy=bzm, zeta_surface=zeta_surf)
        return "BLEW UP" if blew else "STABLE"
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="W9V",
                    choices=("UP3", "W9V", "W9D", "SM2", "QG2"))
    ap.add_argument("--resolution", default="80x64", help="n_lat x n_lon")
    ap.add_argument("--days", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=900.0)
    ap.add_argument("--nlev", type=int, default=50)
    ap.add_argument("--out", default="results/silvestri_jet")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stabilize", action="store_true",
                    help="Apply the eddy-resolving dissipation backstop (A_h=1000+"
                         "C_smag=0.1) to the no-closure WENO schemes — needed at "
                         "eddy-resolving res in legoESM; NOT paper-faithful (the "
                         "paper's WENO has no closure).")
    args = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    run(args.scheme, args.resolution, args.days, args.dt, args.out, args.nlev,
        tag=args.tag, stabilize=args.stabilize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
