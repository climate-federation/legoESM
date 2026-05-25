"""RCEMIP1 (Wing et al. 2018) RCE harness on the plane NH dycore.

Reference: doi:10.5194/gmd-11-793-2018.

Scope
-----
RCEMIP-style radiative-convective equilibrium on the plane
non-hydrostatic dycore (PR2b-PR3d) wired with the same factory
dispatch the cubed-sphere / MPAS NH harnesses use:

- Bulk surface fluxes via :func:`legoesm.coupler.bulk_flux.simple_bulk_fluxes`
- Radiation via :func:`legoesm.atmosphere.physics.radiation.integration.make_radiation_physics`
  with ``model_type="plane"`` (selects gray or RRTMGP from the
  RadiationConfig scheme literal)
- Microphysics via
  :func:`legoesm.atmosphere.physics.microphysics.integration.make_microphysics_physics`
  with ``model_type="plane"`` (selects kessler, morrison, sundqvist,
  seifert_beheng, thompson, ml_emulator, or "none" from the
  MicrophysicsConfig scheme literal)
- Smagorinsky LES (PR3c, horizontal-only pilot)
- Hyperdiffusion (PR3a) + sponge (PR2d) + upwind advection (PR3b)
- ``n_tracers >= 3`` for q_v / q_c / q_r (PR3d)

CLI
---
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run_rcemip_plane.py \\
       --nx 16 --ny 16 --nlev 30 --dx 4000.0 --dt 6.0 \\
       --steps 50 --radiation gray --microphysics kessler \\
       --output results/rcemip_smoke
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
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.field import Field
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


# -------- RCEMIP1 IC (Wing 2018 Tab A1, simplified) -------- #


def _rcemip_theta_profile(z: jax.Array, T_sfc: float = 300.0) -> jax.Array:
    """Wing-inspired simplified θ(z) sounding for the smoke harness.

    Two-segment piecewise profile used to bootstrap the RCEMIP-style
    smoke run; the upper-stratosphere branch of the full Wing 2018
    sounding is deferred to the full RCEMIP validation PR.

        z < z_t = 15 km:  θ(z) = T_sfc + Γ · z,  Γ = 6.7 × 10⁻³ K/m
        z ≥ z_t:          θ = θ(z_t) (constant tropopause cap)
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


# -------- Surface-flux physics_fn (bulk_flux only) -------- #


def _make_surface_flux_physics(
    grid, height_coord, terrain_metric,
    Cd: float, Ch: float, T_sfc: float, q_sfc: float,
):
    """Lowest-level bulk surface fluxes via :mod:`coupler.bulk_flux`.

    Returns a ``physics_fn(state, grid, hc, tm) ->
    PlaneNonHydrostaticTendencies`` whose only non-zero tendencies
    are momentum drag + sensible-heat + latent-heat at the lowest
    model level (``k = nlev - 1`` under top-down indexing).
    """
    from legoesm.coupler.bulk_flux import simple_bulk_fluxes

    def physics_fn(state, grid_in, hc_in, tm_in):
        ny, nx, nlev = state.theta_prime.data.shape
        n_tracers = state.tracers.data.shape[-1]
        rho_0 = hc_in.rho_ref
        theta_0 = hc_in.theta_ref
        theta_total = theta_0 + state.theta_prime.data
        rho_total = rho_0 + state.rho_prime.data

        k_sfc = nlev - 1
        u_lo = state.u.data[..., k_sfc]
        v_lo = state.v.data[..., k_sfc]
        rho_lo = rho_total[..., k_sfc]
        theta_lo = theta_total[..., k_sfc]
        # Reference-Exner-based θ→T conversion at lowest level. See
        # the docstring of the PR4 scaffold variant of this function
        # for the caveat about perturbation-Exner; same caveat
        # applies here.
        pi_sfc = hc_in.exner_ref[k_sfc]
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

        dz_sfc = hc_in.dz[k_sfc]
        du_sfc = tau_x / (rho_lo * dz_sfc)
        dv_sfc = tau_y / (rho_lo * dz_sfc)
        du_dt_data = jnp.zeros_like(state.u.data).at[..., k_sfc].set(du_sfc)
        dv_dt_data = jnp.zeros_like(state.v.data).at[..., k_sfc].set(dv_sfc)

        dT_sfc = shflx / (rho_lo * constants.c_pd * dz_sfc)
        dtheta_sfc = dT_sfc / pi_sfc
        dtheta_p_data = jnp.zeros_like(
            state.theta_prime.data
        ).at[..., k_sfc].set(dtheta_sfc)

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


# -------- Physics composer -------- #


def _sum_plane_tendencies(*tendencies):
    """Sum a list of ``PlaneNonHydrostaticTendencies`` field-wise.

    Each input is a NamedTuple of ``Field`` leaves; the output is one
    NamedTuple whose ``.data`` arrays are the element-wise sum across
    all inputs. Field metadata (dims/units/name) comes from the FIRST
    tendency input. Raises ``ValueError`` on an empty input (caller
    bug: there's no canonical empty tendency).
    """
    if not tendencies:
        raise ValueError(
            "_sum_plane_tendencies requires at least one tendency; "
            "got an empty argument list."
        )
    if len(tendencies) == 1:
        return tendencies[0]
    head = tendencies[0]
    # Codex review 2026-05-24: dims must match across all tendencies
    # so the sum is semantically well-defined; mismatched dims would
    # indicate a wiring bug (e.g., feeding NH cubed-sphere tendencies
    # into the plane composer).
    for t in tendencies[1:]:
        for fname in head._fields:
            if getattr(t, fname).dims != getattr(head, fname).dims:
                raise ValueError(
                    f"plane tendency dim mismatch on field {fname!r}: "
                    f"first={getattr(head, fname).dims!r} "
                    f"vs other={getattr(t, fname).dims!r}"
                )
    out = {}
    for fname in head._fields:
        head_f = getattr(head, fname)
        summed = sum(
            (getattr(t, fname).data for t in tendencies[1:]),
            start=head_f.data,
        )
        out[fname] = head_f.replace(data=summed)
    return PlaneNonHydrostaticTendencies(**out)


def make_rcemip_physics(
    grid, height_coord, terrain_metric,
    radiation_config: RadiationConfig | None,
    microphysics_config: MicrophysicsConfig | None,
    dt: float,
    Cd: float = 1.0e-3, Ch: float = 1.0e-3,
    T_sfc: float = 300.0, q_sfc: float = 0.018,
):
    """Compose RCEMIP physics_fn from surface_fluxes + radiation + microphysics.

    Each component is built by the canonical factory (no plane-specific
    inlining beyond surface fluxes). Pass ``radiation_config=None`` or
    ``microphysics_config=None`` to skip either branch.
    """
    physics_fns = [_make_surface_flux_physics(
        grid, height_coord, terrain_metric,
        Cd=Cd, Ch=Ch, T_sfc=T_sfc, q_sfc=q_sfc,
    )]
    if radiation_config is not None:
        physics_fns.append(make_radiation_physics(
            radiation_config, model_type="plane",
        ))
    if microphysics_config is not None:
        physics_fns.append(make_microphysics_physics(
            microphysics_config, model_type="plane", dt=dt,
        ))

    def physics_fn(state, grid_in, hc_in, tm_in):
        tendencies = [
            fn(state, grid_in, hc_in, tm_in) for fn in physics_fns
        ]
        return _sum_plane_tendencies(*tendencies)

    return physics_fn


# -------- IC + main -------- #


def _build_rcemip_initial_state(grid, height_coord):
    rest = make_rest_state(grid, height_coord, dtype=jnp.float64)
    ny, nx, nlev = rest.theta_prime.data.shape
    tracers = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64)
    q_v = _rcemip_qv_profile(height_coord.z_full)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, nlev)),
    )
    rng_key = jax.random.PRNGKey(0)
    theta_kick = 0.1 * jax.random.normal(rng_key, rest.theta_prime.data.shape)
    return rest._replace(
        theta_prime=rest.theta_prime.replace(data=theta_kick),
        tracers=rest.tracers.replace(data=tracers),
    )


def _build_radiation_config(scheme: str) -> RadiationConfig | None:
    if scheme == "none":
        return None
    if scheme not in ("gray", "rrtmgp"):
        raise ValueError(
            f"Unknown --radiation: {scheme!r}; "
            f"choose from 'gray', 'rrtmgp', 'none'."
        )
    return RadiationConfig(scheme=scheme)


def _build_microphysics_config(scheme: str) -> MicrophysicsConfig | None:
    if scheme == "none":
        return None
    valid = ("kessler", "morrison", "sundqvist",
             "seifert_beheng", "thompson", "ml_emulator")
    if scheme not in valid:
        raise ValueError(
            f"Unknown --microphysics: {scheme!r}; "
            f"choose from {valid + ('none',)}."
        )
    return MicrophysicsConfig(scheme=scheme)


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
    p.add_argument("--radiation", choices=["gray", "rrtmgp", "none"],
                   default="gray",
                   help="Radiation scheme. 'none' skips the radiation branch.")
    p.add_argument("--microphysics",
                   choices=["kessler", "morrison", "sundqvist",
                            "seifert_beheng", "thompson", "ml_emulator",
                            "none"],
                   default="kessler",
                   help="Microphysics scheme. 'none' skips the branch.")
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
    print(f"  radiation={args.radiation}, microphysics={args.microphysics}")

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

    radiation_config = _build_radiation_config(args.radiation)
    microphysics_config = _build_microphysics_config(args.microphysics)
    physics_fn = make_rcemip_physics(
        grid, hc, tm,
        radiation_config=radiation_config,
        microphysics_config=microphysics_config,
        dt=args.dt,
        T_sfc=args.T_sfc,
    )

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
