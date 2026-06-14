"""Compare TVD vs DST-3 tracer advection on Eady baroclinic instability.

Usage:
  JAX_ENABLE_X64=1 python3 scripts/compare_dst3_eady.py [--days 200] [--resolution 100x50]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def run_eady(scheme: str, n_lat: int, n_lon: int, days: float, dt: float = 300.0):
    """Run Eady uniform experiment with given tracer advection scheme."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.experiments.eady_uniform import (
        EadyUniformConfig, create_initial_conditions as eu_ic,
        create_forcings as eu_forcings, compute_sponge_mask)
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface, interp_cell_to_vface)

    eu_config = EadyUniformConfig()
    z_coord = create_ocean_z_star(n_levels=20, H_max=eu_config.H_max)
    grid, _ = create_regional_latlon_grid(
        n_lat, n_lon, eu_config.lat_south, eu_config.lat_north,
        lon_west=eu_config.lon_west, lon_east=eu_config.lon_east,
        periodic_x=True)

    physics = eu_forcings("latlon_channel", None, eu_config)

    # Enable KPP for all schemes — it is a physical parameterization
    # (boundary layer mixing, interior shear instability) not a numerical fix.
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    physics = physics._replace(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
    )

    kw = dict(
        n_barotropic_substeps=30,
        A_h=eu_config.A_h, B_h=eu_config.B_h, C_smag=eu_config.C_smag,
        K_h=eu_config.K_h, K_bih=eu_config.K_bih,
        A_v=eu_config.A_v, K_v=eu_config.K_v,
        bottom_drag_r=eu_config.bottom_drag_coeff,
        eos="linear",
        eos_linear=LinearEOSConfig(
            alpha_T=eu_config.alpha_T, rho_ref=eu_config.rho_0,
            T_ref=eu_config.T_ref_C, S_ref=eu_config.S_uniform),
        barotropic_diffusion_alpha=eu_config.barotropic_diffusion_alpha,
        barotropic_div_damp=eu_config.barotropic_div_damp,
        tracer_advection=scheme,
        physics=physics,
    )
    cfg = LatLonCGridOceanConfig(**kw)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = eu_ic("latlon_channel", grid, z_coord, eu_config)

    # Sponge relaxation
    gamma = compute_sponge_mask(grid, eu_config)
    T_init = jnp.array(state.T.data)
    decay_T = jnp.array(np.exp(-dt * np.array(gamma))[..., np.newaxis])
    decay_u = jnp.array(np.exp(-dt * np.array(
        interp_cell_to_uface(jnp.array(gamma))))[..., np.newaxis])
    decay_v = jnp.array(np.exp(-dt * np.array(
        interp_cell_to_vface(jnp.array(gamma))))[..., np.newaxis])

    T_mean_init = float(jnp.mean(state.T.data))
    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)

    print(f"\n{'='*60}")
    print(f"Running Eady {n_lat}x{n_lon} ({days:.0f} days) with {scheme}")
    print(f"  dt={dt}s, n_steps={n_steps}, diag_every={diag_every}")
    print(f"{'='*60}")

    t0 = time.time()
    for i in range(n_steps):
        state = model.step(state, dt)
        # Sponge
        T_new = state.T.data * decay_T + T_init * (1.0 - decay_T)
        u_new = state.u.data * decay_u
        v_new = state.v.data * decay_v
        sponge_kw = dict(
            u=Field(u_new, name="u", dims=state.u.dims, units=state.u.units),
            v=Field(v_new, name="v", dims=state.v.dims, units=state.v.units),
            T=Field(T_new, name="T", dims=state.T.dims, units=state.T.units))
        # SOM moments must also be decayed by the sponge to stay consistent
        # with the relaxed tracer mean.  decay_T is exp(-dt*gamma).
        if state.T_som is not None:
            sponge_kw["T_som"] = state.T_som.replace(
                data=state.T_som.data * decay_T[..., jnp.newaxis])
        if state.S_som is not None:
            sponge_kw["S_som"] = state.S_som.replace(
                data=state.S_som.data * decay_T[..., jnp.newaxis])
        state = state._replace(**sponge_kw)

        if (i + 1) % diag_every == 0:
            jax.block_until_ready(state.T.data)
            has_nan = bool(jnp.any(jnp.isnan(state.T.data)))
            T_mean = float(jnp.mean(state.T.data))
            max_u = float(jnp.max(jnp.abs(state.u.data)))
            day = (i + 1) * dt / 86400
            print(f"  day {day:6.1f}: T_mean={T_mean:.6f}, "
                  f"max_u={max_u:.4f}, nan={has_nan}")
            if has_nan:
                print("  *** BLOWUP ***")
                break

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0
    has_nan = bool(jnp.any(jnp.isnan(state.T.data)))
    T_mean_final = float(jnp.mean(state.T.data))
    T_drift = abs(T_mean_final - T_mean_init)
    max_speed = float(jnp.max(jnp.abs(state.u.data)))

    print(f"\n  RESULT: {'PASS' if not has_nan else 'FAIL'}")
    print(f"  T_drift={T_drift:.6f}, max_speed={max_speed:.4f}, "
          f"wall={wall:.1f}s")
    return dict(scheme=scheme, ok=not has_nan, T_drift=T_drift,
                max_speed=max_speed, wall=wall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--resolution", default="100x50",
                        help="n_lon x n_lat (e.g. 100x50 for ~20km)")
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--schemes", default="tvd,dst3",
                        help="Comma-separated schemes to compare")
    args = parser.parse_args()

    n_lon, n_lat = [int(x) for x in args.resolution.split("x")]
    schemes = args.schemes.split(",")

    results = []
    for scheme in schemes:
        r = run_eady(scheme, n_lat, n_lon, args.days, args.dt)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY ({n_lat}x{n_lon}, {args.days:.0f} days)")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {r['scheme']:12s}: {status}  T_drift={r['T_drift']:.6f}  "
              f"max_u={r['max_speed']:.4f}  wall={r['wall']:.1f}s")


if __name__ == "__main__":
    main()
