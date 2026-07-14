"""Bisect: find which component limits dt on plane CRM.

Tests 132x132 grid, dx=2km, uniform z, IC: rest+small theta kick.
Each level adds one component; report max-stable-dt.

Levels:
  L1: bare dycore (no physics, no fix_mass, no positivity, no smag)
  L2: + Smagorinsky LES (smag_cs=0.2)
  L3: + hyperdiff (1e6)
  L4: + sponge (default)
  L5: + mass fixer
  L6: + surface flux only
  L7: + microphysics
  L8: + radiation
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_qv_ref_fn, make_wing2018_theta_ref_fn,
)
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

T_SFC_K = 300.0
Q_SFC_FRAC = 0.018
Z_T = 15_000.0
GAMMA_TROP = 6.7e-3


def make_state_and_hc(nx, ny, nlev, dx, H, with_qv=True):
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx, dtype=jnp.float64,
    )
    # Equilibrium CRM debug: T_v0 = SST (param renamed T_sfc -> T_v0).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=T_SFC_K, q_sfc=Q_SFC_FRAC, z_t=Z_T, Gamma=GAMMA_TROP,
    )
    hc = create_height_coordinate(nlev, H=H, theta_ref_fn=theta_fn)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    if with_qv:
        qv_fn = make_wing2018_qv_ref_fn(q_sfc=Q_SFC_FRAC, z_t=Z_T)
        qv_p = qv_fn(hc.z_full)
        new_tr = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64)
        new_tr = new_tr.at[..., 0].set(
            qv_p[None, None, :] * jnp.ones((ny, nx, nlev)),
        )
        state = state._replace(
            tracers=state.tracers.replace(data=new_tr),
        )
    rng = jax.random.PRNGKey(0)
    kick = 0.1 * jax.random.normal(rng, state.theta_prime.data.shape)
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=kick),
    )
    return grid, hc, tm, state


def run_test(name, cfg, dt, n_steps, grid, hc, tm, state, physics_fn=None):
    model = PlaneCompressibleEulerModel(grid, hc, tm, config=cfg)
    for k in range(n_steps):
        state = model.step(state, dt=dt, physics_fn=physics_fn)
        max_w = float(jnp.max(jnp.abs(state.w.data)))
        rho_min = float(jnp.min(state.rho_prime.data + hc.rho_ref))
        finite = bool(jnp.all(jnp.isfinite(state.w.data)))
        if not finite or rho_min < 0:
            return False, k, max_w, rho_min
        if max_w > 200:
            return False, k, max_w, rho_min
    return True, n_steps, max_w, rho_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=132)
    ap.add_argument("--ny", type=int, default=132)
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--dx", type=float, default=2_000.0)
    ap.add_argument("--H", type=float, default=33_000.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n-acoustic-substeps", type=int, default=24)
    args = ap.parse_args()

    grid, hc, tm, state = make_state_and_hc(
        args.nx, args.ny, args.nlev, args.dx, args.H,
    )

    def cfg_base(**kw):
        defaults = dict(
            n_acoustic_substeps=args.n_acoustic_substeps,
            use_coriolis=False,
            sponge_coeff=0.0, sponge_width=0.0,
            hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0,
            hyperdiff_w_coeff=0.0,
            smagorinsky_cs=0.0, smagorinsky_prandtl=1.0,
            fix_mass=False, anchor_mass_to_initial=False,
        )
        defaults.update(kw)
        return CompressibleEulerConfig(**defaults)

    dt_grid = [12, 8, 6, 4, 2, 1]

    levels = [
        ("L1 bare dycore", cfg_base(), None),
        ("L2 +Smag(0.2)", cfg_base(smagorinsky_cs=0.2), None),
        ("L3 +Smag+hyperdiff(1e6)",
         cfg_base(smagorinsky_cs=0.2, hyperdiff_coeff=1e6,
                  hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6), None),
        ("L4 +sponge(0.05,5km)",
         cfg_base(smagorinsky_cs=0.2, hyperdiff_coeff=1e6,
                  hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
                  sponge_coeff=0.05, sponge_width=5_000.0), None),
        ("L5 +fix_mass",
         cfg_base(smagorinsky_cs=0.2, hyperdiff_coeff=1e6,
                  hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
                  sponge_coeff=0.05, sponge_width=5_000.0,
                  fix_mass=True, anchor_mass_to_initial=True), None),
    ]
    # Add physics-bearing levels reusing L5 cfg.
    cfg_full = cfg_base(
        smagorinsky_cs=0.2, hyperdiff_coeff=1e6,
        hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
        sponge_coeff=0.05, sponge_width=5_000.0,
        fix_mass=True, anchor_mass_to_initial=True,
    )

    def make_radiation_only_fn():
        return make_radiation_physics(
            RadiationConfig(scheme="gray"), model_type="plane",
        )

    def make_micro_only_fn(dt):
        return make_microphysics_physics(
            MicrophysicsConfig(scheme="kessler"),
            model_type="plane", dt=dt,
        )

    print(f"# Grid: {args.nx}x{args.ny}x{args.nlev}, "
          f"dx={args.dx}, H={args.H}, "
          f"n_acoustic_substeps={args.n_acoustic_substeps}, "
          f"steps={args.steps}")
    print(f"# Format: dt -> result(steps_completed, max|w|, rho_min)")

    for name, cfg, physics_fn in levels:
        line = [f"\n{name}:"]
        for dt in dt_grid:
            _, _, _, state_fresh = make_state_and_hc(
                args.nx, args.ny, args.nlev, args.dx, args.H,
            )
            ok, k, mw, rm = run_test(
                name, cfg, dt, args.steps, grid, hc, tm, state_fresh,
                physics_fn=physics_fn,
            )
            tag = "OK" if ok else f"FAIL@step{k}"
            line.append(
                f"  dt={dt:5.1f}: {tag:14s} max|w|={mw:6.2f} "
                f"rho_min={rm:.2e}"
            )
        print("\n".join(line), flush=True)

    # Physics levels separately (microphysics dt depends on outer dt).
    for label, fn_builder in (
        ("L6 +surface_flux only", None),  # placeholder; build inline
        ("L7 +microphysics(Kessler) only", "micro"),
        ("L8 +radiation(gray) only", "rad"),
    ):
        line = [f"\n{label}:"]
        for dt in dt_grid:
            _, _, _, state_fresh = make_state_and_hc(
                args.nx, args.ny, args.nlev, args.dx, args.H,
            )
            if label.startswith("L6"):
                # No physics_fn — surface flux can be tested separately
                # but it's not wired through model.step here.
                line.append(f"  dt={dt:5.1f}: skipped (surface flux only)")
                continue
            elif fn_builder == "micro":
                pfn = make_micro_only_fn(dt)
            elif fn_builder == "rad":
                pfn = make_radiation_only_fn()
            else:
                pfn = None
            ok, k, mw, rm = run_test(
                label, cfg_full, dt, args.steps,
                grid, hc, tm, state_fresh, physics_fn=pfn,
            )
            tag = "OK" if ok else f"FAIL@step{k}"
            line.append(
                f"  dt={dt:5.1f}: {tag:14s} max|w|={mw:6.2f} "
                f"rho_min={rm:.2e}"
            )
        print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
