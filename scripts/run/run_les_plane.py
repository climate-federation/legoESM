"""Reference boundary-layer LES on the plane NH dycore: GABLS1 + Wangara.

Two canonical ABL intercomparison cases, run as 3D large-eddy simulation on the
compressible plane non-hydrostatic dycore with the **dynamic Smagorinsky** SGS
closure (Germano 1991 / Lilly 1992; ``smagorinsky_dynamic=True``):

* **GABLS1** (Beare et al. 2006, BLM 118; Cuxart et al. 2006) — stable boundary
  layer. Neutral θ=265 K mixed layer below 100 m, capping inversion 0.01 K/m
  above; geostrophic ug=8 m/s, f=1.39e-4 (73°N); surface cooled at 0.25 K/hr
  (prescribed T_s, bulk neutral-log drag). The reference SBL equilibrates to a
  ~150-200 m depth with a super-geostrophic low-level jet near its top.

* **Wangara** Day 33 (Clarke 1971; Yamada & Mellor 1975) — convective boundary
  layer. θ≈277 K, prescribed cosine surface sensible-heat flux peaking 0.216 K
  m/s at 13:00 LT (+ moisture flux), f for −34.5°S. The CBL grows to ~1-1.5 km
  by mid-afternoon with a well-mixed θ and convective velocity w*~1-2 m/s.

Vertical transport is operator-split into an IMPLICIT mixing-length PBL column
(`_apply_pbl_column`): each step the dycore runs first (advection + acoustic +
horizontal dynamic-Smagorinsky SGS + Coriolis), then a jitted, time-traced
backward-Euler Thomas solve applies vertical diffusion of u/v/θ'/q_v with the
surface flux as the bottom boundary condition (bulk neutral-log drag + bulk T_s
flux for GABLS, prescribed cosine fluxes for Wangara). K_v is a Louis-type
`l²·|∂U/∂z|·f(Ri)` closure with a small floor (damps the 2Δz vertical mode) and
a modest cap (so it does NOT parametrise away the resolved CBL convection — that
is left to the dycore's resolved eddies + 3D dynamic SGS). The implicit solve is
unconditionally stable, replacing an earlier explicit single-level source that
injected a 2Δz mode (GABLS) and blew up under strong convective heating (Wangara).

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run/run_les_plane.py --case gabls1 \\
       --nx 32 --ny 32 --nlev 64 --dx 12.5 --H 400 --dz-sfc 6.25 \\
       --dt 0.5 --hours 2 --output results/les_gabls1
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.core.field import Field
from legoesm.timestepping.tridiagonal import thomas_solve
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Case definitions: θ_ref(z), geostrophic wind, Coriolis, surface forcing.    #
# --------------------------------------------------------------------------- #
_KAPPA_VK = constants.kappa_von_karman


def gabls1_theta_ref(z):
    """GABLS1 initial θ(z): 265 K below 100 m, +0.01 K/m capping inversion."""
    return jnp.where(z > 100.0, 265.0 + 0.01 * (z - 100.0), 265.0)


def wangara_theta_ref(z):
    """Wangara Day-33 initial θ(z): ~277 K mixed layer, capping inversion above
    ~1 km (simplified sounding — the convective growth is flux-driven)."""
    return jnp.where(z > 1000.0, 277.0 + 0.008 * (z - 1000.0), 277.0)


_CASES = {
    "gabls1": dict(
        theta_ref_fn=gabls1_theta_ref,
        f_c=1.39e-4, ug=8.0, vg=0.0, z0=0.1,
        moist=False,
        # defaults (Beare 2006 domain is 400³ m @ dx=6.25 m; coarsened here).
        nx=32, ny=32, nlev=64, dx=12.5, H=400.0, dz_sfc=6.25,
        dt=0.5, hours=2.0,
    ),
    "wangara": dict(
        theta_ref_fn=wangara_theta_ref,
        f_c=2.0 * 7.2921e-5 * float(np.sin(np.deg2rad(-34.5))),
        ug=-8.0, vg=0.0, z0=0.1,
        moist=True,
        nx=48, ny=48, nlev=50, dx=100.0, H=2500.0, dz_sfc=20.0,
        dt=1.0, hours=4.0,
    ),
}


def _implicit_vertical_diffusion(phi_asc, K_iface, dz_asc, dzc, dt,
                                 flux_sfc=None, drag_sfc=None):
    """Backward-Euler implicit vertical diffusion of a column field.

    ``phi_asc`` ascending in z (index 0 = surface), shape (...,n). ``K_iface``
    diffusivity at the n-1 interior interfaces (index k = between k, k+1).
    Surface (k=0 bottom) BC: a prescribed flux ``flux_sfc`` [φ·m/s, +ve into the
    column] and/or an implicit drag ``drag_sfc`` [m/s] (bottom flux = −drag·φ_0).
    No-flux top. Unconditionally stable ⇒ no CFL limit on the vertical mixing /
    surface forcing (fixes the explicit single-level blow-up). Thomas solve."""
    n = phi_asc.shape[-1]
    # Interface coupling coefficient r_k = dt·K_k / (dz_k_center) for k=0..n-2.
    r = dt * K_iface / dzc                       # (..., n-1)
    z = jnp.zeros(phi_asc.shape[:-1] + (1,))
    # Sub-diagonal a (coupling to k-1): a_k = -r_{k-1}/dz_k for k>=1.
    a = -jnp.concatenate([z, r], axis=-1) / dz_asc
    # Super-diagonal c (coupling to k+1): c_k = -r_k/dz_k for k<=n-2.
    c = -jnp.concatenate([r, z], axis=-1) / dz_asc
    b = 1.0 - a - c
    d = phi_asc
    if drag_sfc is not None:                     # implicit surface drag on φ_0
        add0 = dt * drag_sfc / dz_asc[..., 0]
        b = b.at[..., 0].add(add0)
    if flux_sfc is not None:                      # prescribed surface flux into φ_0
        d = d.at[..., 0].add(dt * flux_sfc / dz_asc[..., 0])
    return thomas_solve(a, b, c, d)


def _apply_pbl_column(state, t, *, case, hc, z0, dt, ug, vg):
    """Implicit mixing-length PBL column: vertical diffusion of u, v, θ', q_v
    with the surface flux as the bottom boundary condition.

    Replaces the crude single-level explicit surface source (which injected an
    undamped 2Δz vertical mode and blew up under strong convective heating).
    K_v is a Louis-type mixing-length closure ``l²·|∂U/∂z|·f(Ri)`` with a small
    floor; the implicit Thomas solve is unconditionally stable. The dynamic
    Smagorinsky still provides the resolved HORIZONTAL SGS inside the dycore."""
    g = constants.g
    # Ascending-z views (state stores top-down: index 0 = top).
    u = state.u.data[..., ::-1]
    v = state.v.data[..., ::-1]
    thp = state.theta_prime.data[..., ::-1]
    theta_ref = hc.theta_ref[::-1]
    z_full = hc.z_full[::-1]                       # ascending (index 0 = surface)
    dz = hc.dz[::-1]
    theta = theta_ref + thp                        # full θ, ascending

    uc, vc = u, v                                  # A-grid approx for the column
    # Interface (k between k,k+1) gradients + Richardson number.
    dzc = z_full[1:] - z_full[:-1]                 # (n-1,) centre spacing
    dUdz = jnp.sqrt(
        ((uc[..., 1:] - uc[..., :-1]) / dzc) ** 2
        + ((vc[..., 1:] - vc[..., :-1]) / dzc) ** 2) + 1e-6
    dthdz = (theta[..., 1:] - theta[..., :-1]) / dzc
    theta_i = 0.5 * (theta[..., 1:] + theta[..., :-1])
    Ri = g / theta_i * dthdz / (dUdz ** 2)
    # Louis-type stability function: stable (Ri>0) shuts mixing off by Ri_c=0.25;
    # unstable (Ri<0) enhances. Mixing length l = κz/(1+κz/λ), λ=40 m.
    z_i = 0.5 * (z_full[1:] + z_full[:-1])
    lmix = _KAPPA_VK * z_i / (1.0 + _KAPPA_VK * z_i / 40.0)
    # Stable (Ri>0): Ri_c=0.25 shut-off (sustains the SBL gradient). Unstable
    # (Ri<0): keep f_stab≈1 (NOT the big √(1−16Ri) enhancement) so the column
    # provides only baseline surface-layer mixing — super-adiabatic instability
    # is then left to trigger RESOLVED convection + the dycore 3D dynamic SGS,
    # rather than being parametrised away (which killed the Wangara CBL eddies).
    f_stab = jnp.where(
        Ri > 0.0, jnp.clip(1.0 - Ri / 0.25, 0.0, 1.0) ** 2, 1.0)
    # Small numerical floor, CONFINED to the lower domain (z<300 m) so it damps
    # the near-surface 2Δz mode without eroding the free-atmosphere geostrophic
    # wind / the stable LLJ aloft (the surface-flux-as-BC, not this floor, is the
    # primary 2Δz cure — codex review). Decays as exp(−z/150 m).
    K_floor = 0.02 * jnp.exp(-z_i / 150.0)
    K_iface = lmix ** 2 * dUdz * f_stab + K_floor
    K_iface = jnp.minimum(K_iface, 20.0)           # cap (don't over-mix the CBL)

    z1 = z_full[..., 0] if z_full.ndim else z_full
    z1 = z_full[0]
    Umag = jnp.sqrt(uc[..., 0] ** 2 + vc[..., 0] ** 2 + 0.01)
    Cd = (_KAPPA_VK / jnp.log(z1 / z0)) ** 2
    drag = Cd * Umag                               # implicit momentum drag [m/s]

    new_u = _implicit_vertical_diffusion(u, K_iface, dz, dzc, dt, drag_sfc=drag)
    new_v = _implicit_vertical_diffusion(v, K_iface, dz, dzc, dt, drag_sfc=drag)

    if case == "gabls1":
        theta_s = 265.0 - 0.25 * t / 3600.0
        F_th = Cd * Umag * (theta_s - theta[..., 0])      # bulk kinematic w'θ'
        new_th = _implicit_vertical_diffusion(thp, K_iface, dz, dzc, dt,
                                              flux_sfc=F_th)
        upd = dict(theta_prime=state.theta_prime.replace(data=new_th[..., ::-1]))
    else:                                          # Wangara prescribed fluxes
        t_hr = 9.0 + t / 3600.0
        cos_t = jnp.cos((t_hr - 13.0) / 11.0 * jnp.pi)
        F_th = 0.216 * cos_t
        F_qv = 2.29e-5 * cos_t
        new_th = _implicit_vertical_diffusion(thp, K_iface, dz, dzc, dt,
                                              flux_sfc=F_th)
        qv = state.tracers.data[..., 0][..., ::-1]
        new_qv = _implicit_vertical_diffusion(qv, K_iface, dz, dzc, dt,
                                              flux_sfc=F_qv)
        new_tr = state.tracers.data.at[..., 0].set(
            jnp.clip(new_qv[..., ::-1], 0.0, None))
        upd = dict(theta_prime=state.theta_prime.replace(data=new_th[..., ::-1]),
                   tracers=state.tracers.replace(data=new_tr))
    return state._replace(
        u=state.u.replace(data=new_u[..., ::-1]),
        v=state.v.replace(data=new_v[..., ::-1]),
        **upd)


def build(args):
    spec = _CASES[args.case]
    nx, ny, nlev = args.nx, args.ny, args.nlev
    dx, H, dz_sfc = args.dx, args.H, args.dz_sfc
    f_c = spec["f_c"]
    ug, vg, z0 = spec["ug"], spec["vg"], spec["z0"]

    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="f_plane", f0=f_c)
    hc = create_stretched_height_coordinate(
        nlev, H=H, dz_sfc=dz_sfc, theta_ref_fn=spec["theta_ref_fn"],
        p_sfc=1.0e5)
    # Geostrophic reference wind for the SAM-form Coriolis (f·(u−ug0)).
    ug0 = jnp.full(nlev, ug)
    vg0 = jnp.full(nlev, vg)
    hc = hc._replace(u_geo0=ug0, v_geo0=vg0)
    tm = make_flat_plane_terrain_metric(grid, hc)

    cfg = CompressibleEulerConfig(
        use_coriolis=True,
        semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True,
        n_acoustic_substeps=args.n_acoustic_substeps,
        # Off-centre the SI vertical acoustic solve to damp the vertical 2Δz
        # (Nyquist) computational mode — without it (β=0, fully centred) the
        # single-level surface-flux forcing piles into an undamped 2Δz column
        # mode once the stable BL turbulence collapses. β=0.2 ≈ the RCE value.
        acoustic_off_centering=args.off_centering,
        si_w_vertical_filter_nu=args.si_w_filter,
        # LES dynamic Smagorinsky.
        smagorinsky_cs=0.17,                 # fallback / initial value
        smagorinsky_dynamic=not args.static_sgs,
        smagorinsky_dynamic_cs_max=0.3,
        smagorinsky_prandtl=1.0,
        smagorinsky_wall_damping=True,       # LES: cap l_m at κz near the wall
        smagorinsky_delta_max=1.0e30,        # fine grid: no horizontal Δ cap
        sgs_vertical_diffusion=True,   # 3D dynamic SGS for resolved eddies; column adds surface coupling
        # weak biharmonic for the 2Δ acoustic mode; small sponge at the top.
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        sponge_coeff=0.05, sponge_width=0.2 * H, sponge_w_only=True,
        sponge_profile_shape="sam_rational",
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)

    # IC: rest state (hydrostatic θ_ref) + geostrophic mean wind + θ' seed.
    dtype = grid.area_T.dtype
    state = make_rest_state(grid, hc, dtype=dtype)
    u0 = jnp.full((ny, nx, nlev), ug, dtype=dtype)
    v0 = jnp.full((ny, nx, nlev), vg, dtype=dtype)
    # Band-limited θ' seed in the bottom quarter to trigger turbulence.
    key = jax.random.PRNGKey(0)
    seed = 0.1 * jax.random.normal(key, (ny, nx, nlev), dtype=dtype)
    zmask = (hc.z_full < 0.5 * H).astype(dtype)
    seed = seed * zmask
    state = state._replace(
        u=state.u.replace(data=u0),
        v=state.v.replace(data=v0),
        theta_prime=state.theta_prime.replace(data=seed),
    )
    if spec["moist"]:
        tracers = jnp.zeros((ny, nx, nlev, 1), dtype=dtype)
        state = state._replace(
            tracers=state.tracers.replace(data=tracers))

    surf = jax.jit(partial(
        _apply_pbl_column, case=args.case, hc=hc, z0=z0,
        dt=args.dt, ug=ug, vg=vg))
    return model, state, surf, grid, hc, spec


def _diagnostics(state, hc, spec):
    """Domain-mean BL diagnostics from the current state."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    w = np.asarray(state.w.data)
    thp = np.asarray(state.theta_prime.data)
    z = np.asarray(hc.z_full)
    theta = np.asarray(hc.theta_ref) + thp.mean(axis=(0, 1))
    wvar = (w[..., :-1] ** 2).mean(axis=(0, 1))          # ~w'² (incl mean≈0)
    umean = u.mean(axis=(0, 1))
    vmean = v.mean(axis=(0, 1))
    spd = np.sqrt(umean ** 2 + vmean ** 2)
    # BL depth: height of max vertical θ gradient (inversion base), ascending z.
    order = np.argsort(z)
    zs, ths = z[order], theta[order]
    dthdz = np.gradient(ths, zs)
    h_bl = zs[3 + int(np.argmax(dthdz[3:]))] if len(zs) > 6 else np.nan
    return dict(
        max_w=float(np.abs(w).max()),
        wvar_max=float(wvar.max()),
        h_bl=float(h_bl),
        spd_max=float(spd.max()),
        spd_max_z=float(z[np.argmax(spd)]),
        theta_sfc=float(theta[order][0]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(_CASES), required=True)
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--nlev", type=int)
    p.add_argument("--dx", type=float)
    p.add_argument("--H", type=float)
    p.add_argument("--dz-sfc", type=float)
    p.add_argument("--dt", type=float)
    p.add_argument("--hours", type=float)
    p.add_argument("--off-centering", type=float, default=0.2,
                   help="SI vertical-acoustic off-centring β (damps the 2Δz "
                        "vertical mode). 0 = centred (undamped).")
    p.add_argument("--si-w-filter", type=float, default=0.1,
                   help="Vertical Laplacian filter ν on w inside the SI solve "
                        "(damps 2Δz w noise).")
    p.add_argument("--static-sgs", action="store_true",
                   help="Use fixed-C_s Smagorinsky (diagnostic) instead of the "
                        "dynamic coefficient.")
    p.add_argument("--n-acoustic-substeps", type=int, default=8,
                   help="Horizontal acoustic substeps per dt. Need "
                        "c·(dt/n)/dx < 1 (c≈340): for dx=25 m, dt=0.5 s ⇒ n≳7.")
    p.add_argument("--hyperdiff", type=float, default=1.0e-2)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    # Fill unset args from the per-case defaults.
    spec = _CASES[args.case]
    for k in ("nx", "ny", "nlev", "dx", "H", "dz_sfc", "dt", "hours"):
        if getattr(args, k) is None:
            setattr(args, k, spec[k])
    args.output = args.output or Path(f"results/les_{args.case}")
    args.output.mkdir(parents=True, exist_ok=True)

    model, state, surf, grid, hc, spec = build(args)
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[LES {args.case}] nx={args.nx} ny={args.ny} nlev={args.nlev} "
          f"dx={args.dx} m H={args.H} m dz_sfc={args.dz_sfc} m")
    print(f"  dt={args.dt} s  steps={nsteps}  hours={args.hours}  "
          f"f_c={spec['f_c']:.3e}  ug={spec['ug']}  dynamic-Smagorinsky")
    print(f"{'step':>7} {'t[h]':>6} {'max|w|':>9} {'wvar_max':>9} "
          f"{'h_bl[m]':>8} {'spd_max':>8} {'@z[m]':>7} {'th_sfc':>8}")

    import time
    t0 = time.time()
    for i in range(nsteps):
        t = i * args.dt
        state = model.step(state, dt=args.dt, physics_fn=None)
        state = surf(state, jnp.asarray(t))
        if (i + 1) % args.print_every == 0 or i == 0:
            d = _diagnostics(state, hc, spec)
            mw = d["max_w"]
            if not np.isfinite(mw) or mw > 100.0:
                print(f"[BLOWUP] step {i+1} max|w|={mw}")
                return 1
            print(f"{i+1:7d} {(i+1)*args.dt/3600:6.3f} {d['max_w']:9.3f} "
                  f"{d['wvar_max']:9.4f} {d['h_bl']:8.1f} {d['spd_max']:8.3f} "
                  f"{d['spd_max_z']:7.1f} {d['theta_sfc']:8.3f}")
    wall = time.time() - t0
    d = _diagnostics(state, hc, spec)
    print(f"\n[DONE] wall={wall:.1f}s  {nsteps/wall:.1f} steps/s")
    print(f"  final: BL depth={d['h_bl']:.1f} m  max|w|={d['max_w']:.3f}  "
          f"jet spd={d['spd_max']:.3f} @ z={d['spd_max_z']:.1f} m  "
          f"θ_sfc={d['theta_sfc']:.3f} K")
    # Dump final mean profiles for assessment.
    z = np.asarray(hc.z_full)
    np.savez(args.output / "final_profiles.npz",
             z=z, theta=np.asarray(hc.theta_ref) + np.asarray(
                 state.theta_prime.data).mean(axis=(0, 1)),
             u=np.asarray(state.u.data).mean(axis=(0, 1)),
             v=np.asarray(state.v.data).mean(axis=(0, 1)),
             wvar=(np.asarray(state.w.data)[..., :-1] ** 2).mean(axis=(0, 1)))
    print(f"  profiles -> {args.output}/final_profiles.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
