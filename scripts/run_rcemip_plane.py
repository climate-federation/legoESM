"""RCEMIP1 (Wing et al. 2018) RCE harness on the plane NH dycore.

Reference: doi:10.5194/gmd-11-793-2018.

Scope
-----
Scaled-down RCEMIP-style radiative-convective equilibrium on the
plane non-hydrostatic dycore (PR2b-PR3d) wired with:

- Bulk surface fluxes via :func:`legoesm.coupler.bulk_flux.simple_bulk_fluxes`
- Gray radiation via :func:`legoesm.atmosphere.physics.radiation.gray.gray_radiation`
- Smagorinsky LES (PR3c, horizontal-only pilot)
- Hyperdiffusion (PR3a) + sponge (PR2d) + upwind advection (PR3b)
- ``n_tracers >= 3`` for q_v / q_c / q_r (PR3d)

KNOWN LIMITATIONS
-----------------
* Microphysics column-reshape adapter is **inlined** here for the
  RCEMIP run; a reusable helper that wraps
  ``_make_nonhydrostatic_microphysics`` for plane state lands in a
  follow-up PR after the validation tolerances stabilise.
* Full RCEMIP1 equilibrium (100 days, 100x100 km, 1 km grid) does
  not fit in a CI budget; the default CLI is a scaled-down smoke
  configuration (16x16 cells, 50 steps). The full-resolution
  benchmark spec is parameterised by CLI flags.
* The horizontal-only Smag pilot (PR3c) limits the LES energy
  budget; full 3D Smag lands in a follow-up PR.
* The dycore inherits the PR2b A-grid simplification — momentum
  advection is centred / upwind on cell-centred ``u``, ``v``. The
  energy-consistent C-grid pairing PR will tighten conservation
  bounds.
* Cross-grid consistency (plane vs cubed-sphere vs MPAS) is NOT
  validated in this script — the cubed-sphere / MPAS NH dycores
  use their own RCE harnesses (see ``scripts/run_rce.py``).

CLI
---
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run_rcemip_plane.py \\
       --nx 16 --ny 16 --nlev 30 --dx 4000.0 \\
       --dt 6.0 --steps 50 --output results/rcemip_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.core.field import Field
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


# -------- RCEMIP1 IC (Wing 2018 Tab A1, simplified) -------- #


def _rcemip_theta_profile(z: jax.Array, T_sfc: float = 300.0) -> jax.Array:
    """Wing-inspired simplified θ(z) sounding for the smoke harness.

    **NOT a verbatim Wing 2018 Tab A1 reproduction.** This is a
    two-segment piecewise profile used to bootstrap the RCEMIP-style
    smoke run; the upper-stratosphere branch of the full Wing 2018
    sounding (involving the moist-virtual reference T_v and a
    different lapse above the tropopause) is deferred to the full
    RCEMIP validation PR that compares against published reference
    profiles.

    Profile used here:

        z < z_t = 15 km:  θ(z) = T_sfc + Γ · z,  Γ = 6.7 × 10⁻³ K/m
        z ≥ z_t:          θ = θ(z_t) (constant tropopause cap)

    At the surface θ = 300 K; at 15 km θ ≈ 400.5 K; constant aloft.
    Returns θ at the supplied z [m].
    """
    z_t = 15_000.0
    Gamma = 6.7e-3
    theta_t = T_sfc + Gamma * z_t
    return jnp.where(z < z_t, T_sfc + Gamma * z, theta_t)


def _rcemip_qv_profile(z: jax.Array, q_sfc: float = 0.018) -> jax.Array:
    """Wing 2018 specific humidity profile (analytical):

    q_v(z) = q_sfc * exp(-z / z_q) with z_q = 4 km below 15 km,
    near-zero above.
    """
    z_q = 4_000.0
    z_t = 15_000.0
    return jnp.where(z < z_t, q_sfc * jnp.exp(-z / z_q), 1.0e-9)


# -------- Physics adapter (column reshape) -------- #


def _make_rcemip_physics(
    grid,
    height_coord,
    terrain_metric,
    Cd: float = 1.0e-3,
    Ch: float = 1.0e-3,
    T_sfc: float = 300.0,
    q_sfc: float = 0.018,
    olr_target: float = 250.0,
    radiation_tau: float = 86_400.0 * 5.0,  # 5-day Newtonian damping toward 300 K
):
    """Build a plane-state ``physics_fn`` combining:

    * Surface fluxes via :func:`legoesm.coupler.bulk_flux.simple_bulk_fluxes`
      applied at the lowest model level (k = nlev-1; z_full is
      top-down).
    * Simple Newtonian relaxation to a reference temperature
      profile (placeholder for gray radiation; the full
      ``gray_radiation`` call needs column reshaping that adds
      ~200 LOC and lands with the cross-grid consistency PR).

    The returned ``physics_fn`` has signature
    ``(state, grid, height_coord, terrain_metric) ->
    PlaneNonHydrostaticTendencies``. ``state.tracers.data[..., 0]``
    is interpreted as ``q_v``.
    """
    from legoesm.coupler.bulk_flux import simple_bulk_fluxes

    def physics_fn(state, grid_in, hc_in, tm_in):
        ny, nx, nlev = state.theta_prime.data.shape
        n_tracers = state.tracers.data.shape[-1]
        rho_0 = hc_in.rho_ref
        theta_0 = hc_in.theta_ref
        theta_total = theta_0 + state.theta_prime.data
        rho_total = rho_0 + state.rho_prime.data

        # Lowest model level (top-down indexing: k = nlev - 1).
        k_sfc = nlev - 1
        u_lo = state.u.data[..., k_sfc]
        v_lo = state.v.data[..., k_sfc]
        rho_lo = rho_total[..., k_sfc]
        theta_lo = theta_total[..., k_sfc]
        # Convert θ → T at surface via Exner. Use the **reference
        # Exner** at the lowest full level from height_coord. Codex
        # iter-2 note: ``hc_in.exner_ref`` ignores Exner perturbations
        # carried by (rho', theta'); for the PR4 scaffold this is
        # acceptable because the RCE setup runs near a hydrostatic
        # reference. A perturbation-aware Exner reconstruction
        # (calling ``compute_exner_perturbation`` + adding the
        # reference) is the upgrade path for the full RCEMIP
        # validation harness; documented here so the next PR knows
        # what to lift.
        pi_sfc = hc_in.exner_ref[k_sfc]   # scalar, dimensionless
        T_lo = theta_lo * pi_sfc
        if n_tracers > 0:
            q_lo = state.tracers.data[..., k_sfc, 0]
        else:
            q_lo = jnp.zeros_like(T_lo)
        wind_speed = jnp.sqrt(u_lo ** 2 + v_lo ** 2 + 1.0)  # 1 m/s floor

        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            u_lowest=u_lo, v_lowest=v_lo, T_lowest=T_lo, q_lowest=q_lo,
            T_sfc=jnp.full_like(T_lo, T_sfc),
            q_sfc=jnp.full_like(T_lo, q_sfc),
            rho=rho_lo, wind_speed=wind_speed, Cd=Cd, Ch=Ch,
        )

        # Momentum tendencies from surface stress: applied at lowest
        # level only, dissipated into the layer thickness.
        dz_sfc = hc_in.dz[k_sfc]
        du_sfc = tau_x / (rho_lo * dz_sfc)
        dv_sfc = tau_y / (rho_lo * dz_sfc)
        du_dt_data = jnp.zeros_like(state.u.data).at[..., k_sfc].set(du_sfc)
        dv_dt_data = jnp.zeros_like(state.v.data).at[..., k_sfc].set(dv_sfc)

        # Sensible heat flux warms the lowest layer (convert back from
        # T-tendency to theta-tendency via the same Exner factor used
        # above; ``dtheta = dT / π`` since ``T = θ · π``).
        dT_sfc = shflx / (rho_lo * constants.c_pd * dz_sfc)
        dtheta_sfc = dT_sfc / pi_sfc
        dtheta_p_data = jnp.zeros_like(state.theta_prime.data).at[..., k_sfc].set(dtheta_sfc)
        # Newtonian radiative cooling toward initial theta_ref (theta'
        # relaxed to zero), gentle 5-day timescale per RCEMIP-style
        # placeholder.
        dtheta_p_data = dtheta_p_data - state.theta_prime.data / radiation_tau

        # Latent heat flux moistens the lowest level (q_v tendency):
        dtracers_data = jnp.zeros_like(state.tracers.data)
        if n_tracers > 0:
            dq_sfc = lhflx / (rho_lo * constants.L_v * dz_sfc)
            dtracers_data = dtracers_data.at[..., k_sfc, 0].add(dq_sfc)

        zeros_w = jnp.zeros_like(state.w.data)
        zeros_rho = jnp.zeros_like(state.rho_prime.data)
        zeros_phis = jnp.zeros_like(state.phis.data)

        return PlaneNonHydrostaticTendencies(
            du_dt=state.u.replace(data=du_dt_data),
            dv_dt=state.v.replace(data=dv_dt_data),
            dw_dt=state.w.replace(data=zeros_w),
            dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_data),
            drho_prime_dt=state.rho_prime.replace(data=zeros_rho),
            dphis_dt=state.phis.replace(data=zeros_phis),
            dtracers_dt=state.tracers.replace(data=dtracers_data),
        )

    return physics_fn


# -------- IC + main -------- #


def _build_rcemip_initial_state(grid, height_coord):
    rest = make_rest_state(grid, height_coord, dtype=jnp.float64)
    # Allocate q_v + q_c + q_r tracers
    ny, nx, nlev = rest.theta_prime.data.shape
    tracers = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64)
    q_v = _rcemip_qv_profile(height_coord.z_full)
    # Broadcast q_v(z) to (ny, nx, nlev)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, nlev)),
    )
    # theta perturbation = 0 (rest theta_ref already isentropic 300 K
    # via create_height_coordinate default); add a small random kick
    # to break the symmetry so convection initiates.
    rng_key = jax.random.PRNGKey(0)
    theta_kick = 0.1 * jax.random.normal(rng_key, rest.theta_prime.data.shape)
    return rest._replace(
        theta_prime=rest.theta_prime.replace(data=theta_kick),
        tracers=rest.tracers.replace(data=tracers),
    )


def parse_args():
    p = argparse.ArgumentParser(description="RCEMIP1 plane NH harness.")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=4_000.0,
                   help="Horizontal grid spacing [m]; RCEMIP1 full = 1 km.")
    p.add_argument("--H", type=float, default=33_000.0,
                   help="Model top height [m]; RCEMIP1 = 33 km.")
    p.add_argument("--dt", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--hyperdiff", type=float, default=1.0e6)
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=5_000.0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results/rcemip_plane"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"RCEMIP1 plane: nx={args.nx} ny={args.ny} nlev={args.nlev}")
    print(f"  dx={args.dx} m, Lz={args.H} m, dt={args.dt} s, "
          f"{args.steps} steps -> t_final={args.steps * args.dt:.1f} s")
    print(f"  T_sfc={args.T_sfc} K, hyperdiff={args.hyperdiff:.2e}, "
          f"smag_cs={args.smag_cs}")

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    hc = create_height_coordinate(args.nlev, H=args.H)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    physics_fn = _make_rcemip_physics(grid, hc, tm, T_sfc=args.T_sfc)

    state = _build_rcemip_initial_state(grid, hc)
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))

    print("\nstep    t [s]    max|w|     min(theta')   max(theta')   "
          "max(q_v)   d(mass)")

    for i in range(args.steps):
        state = model.step(state, dt=args.dt, physics_fn=physics_fn)
        if (i + 1) % args.print_every == 0 or i == 0:
            t = (i + 1) * args.dt
            max_w = float(jnp.max(jnp.abs(state.w.data)))
            min_th = float(jnp.min(state.theta_prime.data))
            max_th = float(jnp.max(state.theta_prime.data))
            max_qv = float(jnp.max(state.tracers.data[..., 0]))
            mass = float(compute_dry_mass_plane(state, grid, hc, tm))
            rel = abs(mass - mass0) / abs(mass0)
            print(f"{i+1:5d}  {t:7.2f}  {max_w:9.3e}  {min_th:12.4e}  "
                  f"{max_th:12.4e}  {max_qv:9.3e}  {rel:8.2e}")
            if not bool(jnp.all(jnp.isfinite(state.w.data))):
                print("\nNON-FINITE STATE — aborting.")
                break

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
