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
from legoesm.grids.vertical import (
    create_height_coordinate,
    create_stretched_height_coordinate,
)


# JAX x64 toggle happens at IMPORT TIME (before argparse). Two paths:
# 1. LEGOESM_RCEMIP_PLANE_FP32=1 in the env -> x64 stays OFF -> fp32
#    arithmetic stays fp32. This is the supported fp32 path.
# 2. Anything else -> x64 ON -> fp64 default (and fp32 arrays will
#    auto-promote to fp64 if mixed with any fp64 literal).
# Per codex iter-... HIGH#3: a previous --precision float32 flag was
# DEAD because the module-level toggle ran before argparse. We deleted
# the broken _enable_x64_if_needed shim and now require the env var.
# main() will refuse --precision float32 without the env var to make
# the contract explicit.
import os as _os
if _os.environ.get("LEGOESM_RCEMIP_PLANE_FP32") != "1":
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
    """Compose RCEMIP physics_fn — surface + radiation + microphysics.

    Returns a single ``physics_fn(state, grid, hc, tm)`` that calls each
    branch and sums tendencies. Radiation is called every outer step
    here; for production with RRTMGP use
    :func:`make_rcemip_physics_gated_rad` which caches radiation across
    a configurable interval.
    """
    physics_fns = []
    if Cd > 0 or Ch > 0:
        physics_fns.append(_make_surface_flux_physics(
            grid, height_coord, terrain_metric,
            Cd=Cd, Ch=Ch, T_sfc=T_sfc, q_sfc=q_sfc,
        ))
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


def split_rad_from_other_physics(
    grid, height_coord, terrain_metric,
    radiation_config: RadiationConfig | None,
    microphysics_config: MicrophysicsConfig | None,
    dt: float,
    Cd: float = 1.0e-3, Ch: float = 1.0e-3,
    T_sfc: float = 300.0, q_sfc: float = 0.018,
):
    """Build TWO separate physics callables for the gated-radiation pattern.

    Returns ``(non_rad_physics_fn, rad_physics_fn_or_None)``:
    - non_rad_physics_fn(state) — runs every dycore outer step
      (surface fluxes + microphysics). Cheap.
    - rad_physics_fn(state) — runs only every ``radiation_interval``
      outer steps; returned tendency is cached + applied as forward
      Euler increments between refreshes. None if radiation_config is None.

    Both follow the standard ``physics_fn(state, grid, hc, tm) ->
    PlaneNonHydrostaticTendencies`` signature.
    """
    non_rad_fns = []
    if Cd > 0 or Ch > 0:
        non_rad_fns.append(_make_surface_flux_physics(
            grid, height_coord, terrain_metric,
            Cd=Cd, Ch=Ch, T_sfc=T_sfc, q_sfc=q_sfc,
        ))
    if microphysics_config is not None:
        non_rad_fns.append(make_microphysics_physics(
            microphysics_config, model_type="plane", dt=dt,
        ))

    def non_rad_physics_fn(state, grid_in, hc_in, tm_in):
        if not non_rad_fns:
            return _zero_tendencies(state)
        tendencies = [
            fn(state, grid_in, hc_in, tm_in) for fn in non_rad_fns
        ]
        return _sum_plane_tendencies(*tendencies)

    rad_physics_fn = None
    if radiation_config is not None:
        rad_physics_fn = make_radiation_physics(
            radiation_config, model_type="plane",
        )
    return non_rad_physics_fn, rad_physics_fn


def _zero_tendencies(state):
    """All-zero PlaneNonHydrostaticTendencies matching state's pytree shape."""
    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
        dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
        dtheta_prime_dt=state.theta_prime.replace(
            data=jnp.zeros_like(state.theta_prime.data)),
        drho_prime_dt=state.rho_prime.replace(
            data=jnp.zeros_like(state.rho_prime.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        dtracers_dt=state.tracers.replace(
            data=jnp.zeros_like(state.tracers.data)),
    )


def apply_radiation_forward_euler(state, rad_tend, dt):
    """Forward-Euler apply of cached radiation tendency over dt.

    Radiation tends to be slow (~K/day in tropos, ~10K/day at strato).
    Over a 5-min radiation interval, forward Euler error << RK3 dycore
    error on the same fields. Standard treatment in operational CRMs
    (SAM, WRF, CM1 all forward-Euler their radiation increment).
    """
    new_theta_p = state.theta_prime.data + dt * rad_tend.dtheta_prime_dt.data
    return state._replace(
        theta_prime=state.theta_prime.replace(data=new_theta_p),
    )


# -------- IC + main -------- #


def _build_rcemip_initial_state(grid, height_coord, dtype=jnp.float64,
                                theta_noise_amp=0.1, n_seed_lev=4,
                                n_tracers=3):
    """RCEMIP1 IC: rest state + q_v profile + small theta noise.

    theta noise restricted to the bottom ``n_seed_lev`` levels (Wing
    2018 symmetry breaker) — applying noise everywhere causes
    spurious upper-tropospheric buoyancy gradients that NaN within
    ~10 steps at dx=2 km regardless of dt. Zero-mean horizontal so
    total energy is conserved at IC.

    ``n_tracers``: 3 = q_v, q_c, q_r (Kessler/Sundqvist/Thompson layout
    head); 9 = full Morrison/Seifert-Beheng with N_c, N_r, N_i + ice
    classes. The microphysics integration validates the slot count.
    """
    rest = make_rest_state(grid, height_coord, dtype=dtype)
    ny, nx, nlev = rest.theta_prime.data.shape
    tracers = jnp.zeros((ny, nx, nlev, n_tracers), dtype=dtype)
    q_v = _rcemip_qv_profile(height_coord.z_full).astype(dtype)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, nlev)),
    )
    n_seed_lev = min(n_seed_lev, nlev)
    rng_key = jax.random.PRNGKey(0)
    theta_noise = jax.random.uniform(
        rng_key, shape=(ny, nx, n_seed_lev),
        minval=-theta_noise_amp, maxval=theta_noise_amp, dtype=dtype,
    )
    theta_noise = theta_noise - jnp.mean(theta_noise, axis=(0, 1),
                                         keepdims=True)
    theta_p = jnp.zeros_like(rest.theta_prime.data)
    # Bottom 4 levels in top-down indexing = LAST 4 array entries.
    theta_p = theta_p.at[..., -n_seed_lev:].set(theta_noise)
    return rest._replace(
        theta_prime=rest.theta_prime.replace(data=theta_p),
        tracers=rest.tracers.replace(data=tracers),
    )


def _build_radiation_config(scheme: str,
                            update_interval_steps: int = 1) -> RadiationConfig | None:
    if scheme == "none":
        return None
    if scheme not in ("gray", "rrtmgp"):
        raise ValueError(
            f"Unknown --radiation: {scheme!r}; "
            f"choose from 'gray', 'rrtmgp', 'none'."
        )
    return RadiationConfig(
        scheme=scheme, update_interval_steps=update_interval_steps,
    )


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
    p.add_argument("--semi-implicit", action="store_true",
                   help="Use semi-implicit acoustic substepping (lifts "
                        "vertical CFL). Recommended for long runs at dx>=2km.")
    p.add_argument("--n-acoustic-substeps", type=int, default=6,
                   help="Acoustic substeps per RK3 stage. Default 6 "
                        "matches CompressibleEulerConfig default.")
    p.add_argument("--off-centering", type=float, default=0.1,
                   help="Skamarock-Klemp 2008 off-centering beta for the "
                        "acoustic mode. 0.1 = recommended for moist-convective "
                        "stability; 0.0 = centred (less damping but can grow "
                        "acoustic noise on long runs).")
    p.add_argument("--implicit-buoyancy", action="store_true", default=True,
                   help="Klemp-Wilhelmson 1978 implicit buoyancy in the "
                        "semi-implicit acoustic substep. Closes the w<->theta "
                        "gravity-wave feedback at coarse vertical resolution; "
                        "default ON for RCE (was OFF in iter-78 bench config).")
    p.add_argument("--no-implicit-buoyancy", dest="implicit_buoyancy",
                   action="store_false")
    p.add_argument("--theta-noise-amp", type=float, default=0.0,
                   help="Initial theta' perturbation amplitude [K] at bottom 4 "
                        "levels. 0 = clean Wing IC (stable at dt up to 10 s "
                        "per iter-9/14); 0.1 = Wing 2018 standard symmetry "
                        "breaker (blows up at dx>=2 km without LES — iter-212 "
                        "in run_rce_mpi_long.py).")
    p.add_argument("--no-physics", action="store_true",
                   help="Skip the physics_fn entirely. Use for dry-dycore "
                        "stability probes.")
    p.add_argument("--no-surface-flux", action="store_true",
                   help="Disable surface bulk fluxes (Cd=Ch=0). For stability "
                        "diagnostics — without surface fluxes RCE cannot reach "
                        "physical equilibrium but the dycore alone can be "
                        "tested.")
    p.add_argument("--advection",
                   choices=["upwind1", "van_leer", "weno5"],
                   default="van_leer",
                   help="Horizontal advection scheme. van_leer (default) "
                        "= 2nd-order TVD, monotone, stencil 4; needed for "
                        "stability at dt>=10s with default hyperdiff. weno5 "
                        "= 5th-order WENO-Z, much less grid-scale noise; "
                        "upwind1 = 1st-order (smoke runs only).")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64",
                   help="fp32 ~5-9x faster than fp64 on consumer GPU "
                        "(fp64 ALU 1:32 ratio); fp32 sufficient for RCE.")
    p.add_argument("--radiation-interval", type=int, default=150,
                   help="Radiation update interval in outer steps. RCEMIP / "
                        "CRM standard: refresh every 5 min sim time. At "
                        "dt=2s, interval=150 = 5 min refresh; interval=900 = "
                        "30 min (less aggressive). 1 = every step (heavy with "
                        "RRTMGP). Gated mode (interval>1) applies cached "
                        "radiation tendency as forward Euler increments "
                        "between recomputes — standard SAM/WRF/CM1 practice.")
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
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="Emit a surface-snapshot PNG every N steps "
                        "(0 = off). At dt=20s, 4320 steps = 1 sim day.")
    p.add_argument("--stretched-vertical", action="store_true",
                   help="Use create_stretched_height_coordinate (RCEMIP1: "
                        "nlev=74, geometric stretching from dz_sfc=50m near "
                        "surface to ~1500m at model top). Default uses the "
                        "uniform create_height_coordinate.")
    p.add_argument("--dz-sfc", type=float, default=50.0,
                   help="Surface-layer thickness [m] for stretched vertical "
                        "coordinate. RCEMIP1 standard = 50 m.")
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

    # Per codex iter-... HIGH#3: explicit contract check. fp32 must be
    # requested via the env var BEFORE Python import time; --precision
    # alone is insufficient because jax.config.update("jax_enable_x64",
    # True) runs at module load.
    if args.precision == "float32" and _os.environ.get(
        "LEGOESM_RCEMIP_PLANE_FP32"
    ) != "1":
        raise SystemExit(
            "--precision float32 requires LEGOESM_RCEMIP_PLANE_FP32=1 in "
            "the environment BEFORE python launch (the jax x64 toggle "
            "runs at module import time, before argparse). Example: "
            "LEGOESM_RCEMIP_PLANE_FP32=1 .venv/bin/python "
            "scripts/run_rcemip_plane.py --precision float32 ..."
        )
    dtype = jnp.float32 if args.precision == "float32" else jnp.float64
    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=dtype,
    )
    if args.stretched_vertical:
        hc = create_stretched_height_coordinate(
            args.nlev, H=args.H, dz_sfc=args.dz_sfc,
        )
    else:
        hc = create_height_coordinate(args.nlev, H=args.H)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=args.semi_implicit,
        n_acoustic_substeps=args.n_acoustic_substeps,
        acoustic_off_centering=args.off_centering,
        implicit_buoyancy=args.implicit_buoyancy,
        horizontal_advection_scheme=args.advection,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)

    radiation_config = _build_radiation_config(
        args.radiation, update_interval_steps=args.radiation_interval,
    )
    microphysics_config = _build_microphysics_config(args.microphysics)
    if args.no_physics:
        physics_fn = None
        rad_physics_fn = None
    elif args.radiation_interval > 1 and radiation_config is not None:
        # Gated radiation: split heavy radiation from light per-step physics.
        # rad_tendency cached for radiation_interval outer steps, applied as
        # forward Euler each step. Standard CRM treatment (SAM, WRF, CM1).
        physics_fn, rad_physics_fn = split_rad_from_other_physics(
            grid, hc, tm,
            radiation_config=radiation_config,
            microphysics_config=microphysics_config,
            dt=args.dt, T_sfc=args.T_sfc,
            Cd=0.0 if args.no_surface_flux else 1.0e-3,
            Ch=0.0 if args.no_surface_flux else 1.0e-3,
        )
        sim_refresh_s = args.radiation_interval * args.dt
        print(f"  RADIATION GATED: refresh every {args.radiation_interval} "
              f"steps = {sim_refresh_s:.0f} s sim time "
              f"(RCEMIP typical 300-1800 s).")
        # Per codex iter-... MEDIUM#7: warn if interval is well outside
        # the operational CRM range (5-30 min sim).
        if sim_refresh_s > 1800.0:
            print(f"  WARN: radiation refresh interval {sim_refresh_s:.0f} s "
                  f"> 1800 s (30 min). Slow-process error grows linearly "
                  f"with interval; SAM/WRF/CM1 typical max = 30 min.")
        elif sim_refresh_s < 60.0:
            print(f"  WARN: radiation refresh interval {sim_refresh_s:.0f} s "
                  f"< 60 s. RRTMGP cost dominates the run; consider "
                  f"--radiation-interval >= {int(300 / args.dt)}.")
    else:
        physics_fn = make_rcemip_physics(
            grid, hc, tm,
            radiation_config=radiation_config,
            microphysics_config=microphysics_config,
            dt=args.dt,
            T_sfc=args.T_sfc,
            Cd=0.0 if args.no_surface_flux else 1.0e-3,
            Ch=0.0 if args.no_surface_flux else 1.0e-3,
        )
        rad_physics_fn = None

    # Tracer slot count per scheme (codex iter-... MEDIUM#5):
    #   morrison / seifert_beheng / p3: 9 slots (q_v, q_c, q_r, q_i,
    #     q_s, q_g, N_c, N_r, N_i)
    #   thompson: 7 slots (q_v, q_c, q_r, q_i, q_s, q_g, N_i)
    #   kessler / sundqvist / ml_emulator / none: 3 slots (q_v, q_c, q_r)
    # The microphysics integration validates the slot count at JIT time
    # and raises ValueError if too few — but we allocate generously
    # here to surface schema errors at parse time, not deep in JIT.
    if args.microphysics in ("morrison", "seifert_beheng", "p3"):
        n_tracers = 9
    elif args.microphysics == "thompson":
        n_tracers = 7
    else:
        n_tracers = 3
    state = _build_rcemip_initial_state(
        grid, hc, dtype=dtype, theta_noise_amp=args.theta_noise_amp,
        n_tracers=n_tracers,
    )
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))

    snap_dir = args.output / "snapshots"
    if args.snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)

    print("\nstep    t [s]    max|w|     min(theta')   max(theta')   "
          "max(q_v)   d(mass)")

    # Gated-radiation runtime state. rad_physics_fn is None when
    # radiation is either off or runs every step inside physics_fn.
    # Bind grid/hc/tm via closure (Python statics) — they're pytrees of
    # arrays; passing as JIT args would require static_argnums=hashable
    # which they aren't. Closure capture is safe: the wrapper recompiles
    # iff the state's shape/dtype changes, not on every call.
    if rad_physics_fn is not None:
        def _rad_wrapper(s):
            return rad_physics_fn(s, grid, hc, tm)
        rad_jit = jax.jit(_rad_wrapper)
    else:
        rad_jit = None
    cached_rad_tend = None

    for i in range(args.steps):
        if rad_jit is not None and (
            i % args.radiation_interval == 0 or cached_rad_tend is None
        ):
            cached_rad_tend = rad_jit(state)
        if cached_rad_tend is not None:
            state = apply_radiation_forward_euler(
                state, cached_rad_tend, args.dt,
            )
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
                  f"{max_th:12.4e}  {max_qv:9.3e}  {rel:8.2e}", flush=True)
            if not bool(jnp.all(jnp.isfinite(state.w.data))):
                print("\nNON-FINITE STATE — aborting.")
                break
        if args.snapshot_every > 0 and (i + 1) % args.snapshot_every == 0:
            _emit_surface_snapshot_png(
                snap_dir, i + 1, (i + 1) * args.dt, state, grid, hc,
            )
            _emit_profile_npz(
                snap_dir, i + 1, (i + 1) * args.dt, state, hc,
            )

    if args.snapshot_every > 0:
        _render_profile_evolution_png(
            snap_dir, args.output / "profile_evolution.png",
        )

    print(f"\nOutput: {args.output}")


def _emit_profile_npz(snap_dir: Path, step: int, t_s: float,
                       state, hc) -> None:
    """Save horizontal-mean vertical profiles per snapshot day.

    Profiles dumped: T(z), theta'(z), q_v(z), q_c(z), w_RMS(z), CWV(z).
    Read back by render_profile_evolution_png at end of run.
    """
    import numpy as np
    import jax.numpy as _jnp
    nlev = state.theta_prime.data.shape[-1]
    theta_p = np.asarray(state.theta_prime.data)
    rho_p = np.asarray(state.rho_prime.data)
    # w lives at half levels (nlev+1); average to full levels (nlev)
    # so the profile axis aligns with theta/qv/etc.
    w_half = np.asarray(state.w.data)
    w = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    theta_0 = np.asarray(hc.theta_ref)
    rho_0 = np.asarray(hc.rho_ref)
    pi_0 = np.asarray(hc.exner_ref)
    theta_total = theta_0 + theta_p
    # Hydrostatic Exner -> Temperature at full levels (cheap diagnostic).
    T = theta_total * pi_0
    q_v = np.asarray(state.tracers.data[..., 0])
    n_tr = state.tracers.data.shape[-1]
    q_c = np.asarray(state.tracers.data[..., 1]) if n_tr > 1 else None
    # Horizontal means over (ny, nx)
    np.savez(snap_dir / f"profile_step_{step:08d}.npz",
             step=step, t_s=t_s, z=np.asarray(hc.z_full),
             T_mean=T.mean(axis=(0, 1)),
             theta_mean=theta_total.mean(axis=(0, 1)),
             theta_p_mean=theta_p.mean(axis=(0, 1)),
             theta_p_std=theta_p.std(axis=(0, 1)),
             qv_mean=q_v.mean(axis=(0, 1)),
             qv_std=q_v.std(axis=(0, 1)),
             qc_mean=(q_c.mean(axis=(0, 1)) if q_c is not None
                      else np.zeros(nlev)),
             w_RMS=np.sqrt((w ** 2).mean(axis=(0, 1))),
             rho_mean=(rho_0 + rho_p.mean(axis=(0, 1))))


def _render_profile_evolution_png(snap_dir: Path, out: Path) -> None:
    """Compose a 5-panel profile-vs-day PNG from saved profile_*.npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    npz_files = sorted(snap_dir.glob("profile_step_*.npz"))
    if not npz_files:
        print(f"  no profile npz found in {snap_dir}; skip evolution PNG")
        return
    data = [np.load(f) for f in npz_files]
    days = np.array([d["t_s"] for d in data]) / 86400.0
    z_km = data[0]["z"] / 1000.0
    fig, axes = plt.subplots(1, 5, figsize=(18, 7), sharey=True)
    panels = [
        ("theta_mean", "θ(z) [K]", "viridis"),
        ("qv_mean", "q_v(z) [kg/kg]", "plasma"),
        ("qc_mean", "q_c(z) [kg/kg]", "Blues"),
        ("w_RMS", "w_RMS(z) [m/s]", "magma"),
        ("theta_p_std", "θ' std(z) [K]", "inferno"),
    ]
    cmap = plt.get_cmap("viridis", len(data))
    for ax, (key, label, _cm) in zip(axes, panels):
        for i, d in enumerate(data):
            ax.plot(d[key], z_km, color=cmap(i / max(1, len(data) - 1)),
                    linewidth=0.7, alpha=0.7)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("height [km]")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=days[0],
                                                   vmax=days[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04,
                        label="day")
    fig.suptitle("RCE horizontal-mean profile evolution", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _emit_surface_snapshot_png(snap_dir: Path, step: int, t_s: float,
                                state, grid, hc) -> None:
    """Write a 2x2 panel PNG of surface fields at this timestep.

    Panels: (q_v surface), (theta' surface), (max(w) column max),
    (column-integrated water vapor). Top-down: k_sfc = nlev-1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k_sfc = state.theta_prime.data.shape[-1] - 1
    qv_sfc = np.asarray(state.tracers.data[..., k_sfc, 0])
    th_sfc = np.asarray(state.theta_prime.data[..., k_sfc])
    w_col_max = np.asarray(jnp.max(jnp.abs(state.w.data), axis=-1))
    # CWV = sum(rho_v dz) = sum(q_v * rho_dry * dz)
    q_v = np.asarray(state.tracers.data[..., 0])  # (ny, nx, nlev)
    rho_total = np.asarray(hc.rho_ref + state.rho_prime.data)  # (ny, nx, nlev)
    dz = np.asarray(hc.dz)  # (nlev,)
    cwv = np.sum(q_v * rho_total * dz, axis=-1)  # kg/m^2

    day = t_s / 86400.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (qv_sfc, "q_v surface [kg/kg]", "BrBG"),
        (th_sfc, "theta' surface [K]", "RdBu_r"),
        (w_col_max, "max|w| over column [m/s]", "viridis"),
        (cwv, "column water vapor [kg/m^2]", "Blues"),
    ]
    for ax, (data, label, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"RCE plane CRM — day {day:.2f} (step {step})",
                 fontsize=12)
    fig.tight_layout()
    out = snap_dir / f"day_{day:07.2f}_step_{step:08d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
