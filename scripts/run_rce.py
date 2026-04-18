#!/usr/bin/env python
"""Moist Radiative-Convective Equilibrium (RCE) with slab ocean or slab land.

Aquaplanet or land-planet experiment:
- Gray radiation (Frierson 2006, moisture-dependent LW optical depth)
- SBM convection + large-scale condensation
- Bulk aerodynamic boundary-layer coupling
- Slab ocean (50 m mixed layer) or slab land (1 m soil + bucket hydrology)
- Rayleigh friction (BL drag + weak free-atmosphere drag)

Operator-split coupling: dynamics step, then physics step each timestep.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_rce.py --days 200
    JAX_ENABLE_X64=1 python scripts/run_rce.py --mode land --days 100
    JAX_ENABLE_X64=1 python scripts/run_rce.py --resolution 24 --dt 300 --days 500
    JAX_ENABLE_X64=1 python scripts/run_rce.py --grid-type gaussian --truncation 21
    JAX_ENABLE_X64=1 python scripts/run_rce.py --grid-type latlon --resolution 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Moist RCE experiment")
    parser.add_argument("--mode", choices=["ocean", "land"], default="ocean",
                        help="Surface type: slab ocean or slab land")
    parser.add_argument("--ocean-mode", choices=["prescribed", "slab", "two_layer"],
                        default="slab",
                        help="Ocean coupling: prescribed (fixed SST), slab (50m), "
                             "two_layer (50m mixed + 200m deep)")
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=16,
                        help="Grid resolution (N for cubed-sphere, n_max for spectral, etc.)")
    parser.add_argument("--nlev", type=int, default=20)
    parser.add_argument("--dt", type=float, default=None,
                        help="Timestep [s] (auto: 600 for N<=24, 300 for N>24)")
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--sst-init", type=float, default=300.0,
                        help="Initial SST [K] (ocean) or soil T [K] (land)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "gaussian", "latlon", "voronoi"],
                        help="Horizontal grid type")
    parser.add_argument("--discretization", type=str, default="cdgrid",
                        choices=["cdgrid", "spectral", "latlon_fv", "mpas"],
                        help="Discretization method")
    parser.add_argument("--truncation", type=int, default=None,
                        help="Spectral truncation (T21, T42, etc.). Sets grid_type=gaussian.")
    args = parser.parse_args()

    # Auto-configure for spectral discretization
    if args.discretization == "spectral" or args.truncation is not None:
        args.discretization = "spectral"
        args.grid_type = "gaussian"
        if args.truncation is not None:
            args.resolution = args.truncation

    N = args.resolution
    NLEV = args.nlev
    DT = args.dt or (300.0 if N > 24 else 600.0)
    OUTPUT_DIR = Path(args.output or f"results/rce_{args.mode}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Grid, vertical coordinate, dynamical core
    # ---------------------------------------------------------------
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.driver.component_factory import create_atmosphere_dycore, compute_diffusion
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig

    grid_type = args.grid_type
    discretization = args.discretization

    if grid_type == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        grid = create_cubed_sphere(N)
    elif grid_type == "gaussian":
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(N)
    elif grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(N)
    elif grid_type == "voronoi":
        from legoesm.grids.voronoi import create_voronoi_mesh
        grid = create_voronoi_mesh(N, lloyd_iterations=50)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    sigma = create_sigma_coordinate(NLEV)
    shape_2d = grid.grid_shape_2d

    config = ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=N, nlev=NLEV),
        dycore=DycoreConfig(discretization=discretization, dt=DT),
    )
    model = create_atmosphere_dycore(config, grid, sigma)
    HYPERDIFF = compute_diffusion(grid, config.dycore).hyperdiff

    # ---------------------------------------------------------------
    # Initial atmospheric state (isothermal 280 K, at rest)
    # ---------------------------------------------------------------
    if grid_type == "cubed_sphere":
        from legoesm.atmosphere.held_suarez import held_suarez_init
        state = held_suarez_init(grid, sigma, T_init=280.0)
    elif grid_type == "voronoi":
        from legoesm.atmosphere.held_suarez import held_suarez_init_mpas
        state = held_suarez_init_mpas(grid, sigma, T_init=280.0)
    else:
        from legoesm.atmosphere.held_suarez import held_suarez_init_latlon
        state = held_suarez_init_latlon(grid, sigma, T_init=280.0)

    # Moisture: 60% RH with sigma^2 vertical decay
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio

    p_full_init = state.p_s.data[..., None] * sigma.sigma_full
    q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
    q_v = 0.6 * q_sat_init * sigma.sigma_full ** 2
    q_v = jnp.minimum(q_v, q_sat_init)

    # ---------------------------------------------------------------
    # Surface configuration
    # ---------------------------------------------------------------
    IS_LAND = args.mode == "land"
    OCEAN_MODE = args.ocean_mode if not IS_LAND else "slab"  # land uses slab soil
    IS_TWO_LAYER = OCEAN_MODE == "two_layer"
    IS_PRESCRIBED = OCEAN_MODE == "prescribed"
    if not IS_LAND:
        from legoesm.ocean.simple_ocean import SimpleOceanConfig
        sfc_config = SimpleOceanConfig(mode=OCEAN_MODE, h_mix=50.0, albedo_ocean=0.06,
                                       h_deep=200.0, k_mix=1.0e-4)
        C_sfc = sfc_config.rho_ocean * sfc_config.c_ocean * sfc_config.h_mix
        T_sfc = jnp.full(shape_2d, args.sst_init)
        T_deep = jnp.full(shape_2d, 278.0) if IS_TWO_LAYER else None
        sfc_albedo = 0.06
        T_freeze = sfc_config.T_freeze
        W_bucket = jnp.zeros(shape_2d)  # dummy, unused
        W_max = 1.0                     # dummy
        beta_min = 1.0                  # dummy
    else:
        # Slab land: thin soil (1 m), fast thermal response
        C_soil = 2.0e6     # volumetric heat capacity [J/m3/K]
        h_soil = 1.0       # soil depth [m]
        C_sfc = C_soil * h_soil
        T_sfc = jnp.full(shape_2d, args.sst_init)
        T_deep = None
        sfc_albedo = 0.25
        T_freeze = constants.T_freeze  # 273.15 K (freshwater)
        # Bucket hydrology
        W_max = 150.0      # kg/m2
        beta_min = 0.1     # minimum evaporation efficiency
        W_bucket = jnp.full(shape_2d, 0.75 * W_max)  # 75% saturated

    C_H = 4.4e-3  # bulk transfer coefficient (heat and moisture; Frierson 2006)

    # ---------------------------------------------------------------
    # Physics configuration
    # ---------------------------------------------------------------
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.diagnostics.column_integrals import column_water_vapor

    # Grid-specific hyperdiffusion (spectral/voronoi handle diffusion in dycore)
    _apply_hyperdiff = None
    if grid_type == "cubed_sphere":
        from legoesm.core.operators_3d import hyperdiffusion_3d
        _apply_hyperdiff = lambda q, coeff: hyperdiffusion_3d(q, grid, coeff)
    elif grid_type == "latlon":
        from legoesm.core.operators_latlon_3d import hyperdiffusion_3d as hyperdiffusion_3d_ll
        _apply_hyperdiff = lambda q, coeff: hyperdiffusion_3d_ll(q, grid, coeff)

    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=sfc_albedo, perpetual_equinox=True,
    )
    sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

    # Rayleigh friction profile (Frierson 2006: BL only, no free-atmosphere drag)
    sigma_b = 0.7
    k_f = ((1.0 / 86400.0) * jnp.maximum(0.0,
               (sigma.sigma_full - sigma_b) / (1.0 - sigma_b)))
    fric_decay = jnp.exp(-k_f * DT)

    S_0 = gray_config.S_0
    dsigma = sigma.dsigma

    # ---------------------------------------------------------------
    # JIT-compiled physics step
    # ---------------------------------------------------------------
    @jax.jit
    def physics_step(T, p_s, q_v, u, v, T_sfc, T_deep, W_bkt, lat, dt):
        """Operator-split physics: radiation + convection + BL + surface."""
        nlev = T.shape[-1]
        ncol = T[..., 0].size
        p_full = p_s[..., None] * sigma.sigma_full
        p_half = p_s[..., None] * sigma.sigma_half

        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)

        insol = perpetual_equinox_insolation(lat_col, S_0)

        # (a) Gray radiation
        rad = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        dT_rad = rad.heating_rate.reshape(T.shape)

        # (b) SBM convection
        conv = sbm_convection(
            T=T_col, q_v=q_v_col,
            p_full=p_full_col, p_half=p_half_col,
            dt=dt, config=sbm_config,
        )
        dT_conv = conv.dT_dt.reshape(T.shape)
        dq_conv = conv.dq_v_dt.reshape(T.shape)
        precip = conv.precipitation.reshape(p_s.shape)

        # (c) Bulk aerodynamic BL coupling
        rho_low = (p_s * sigma.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (sigma.sigma_half[-1] - sigma.sigma_half[-2])

        # Sensible heat flux (positive upward)
        shflx = rho_low * constants.c_pd * C_H * wind * (T_sfc - T[..., -1])

        # Latent heat flux — ocean: beta=1; land: bucket moisture
        q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
        if IS_LAND:
            beta = jnp.maximum(beta_min, W_bkt / W_max)
        else:
            beta = 1.0
        lhflx = rho_low * constants.L_v * C_H * wind * beta * (q_sat_sfc - q_v[..., -1])
        lhflx = jnp.maximum(lhflx, 0.0)  # no dew in simple scheme
        evap = lhflx / constants.L_v

        # Atmospheric tendencies (lowest level)
        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
        dq_BL = constants.g * evap / dp_low

        # (d) Surface energy balance
        sw_net = (rad.sw_flux_down[:, -1] - rad.sw_flux_up[:, -1]).reshape(p_s.shape)
        lw_net = (rad.lw_flux_down[:, -1] - rad.lw_flux_up[:, -1]).reshape(p_s.shape)

        if IS_PRESCRIBED:
            T_sfc_new = T_sfc  # fixed SST
        elif IS_TWO_LAYER:
            # Mixed layer + deep layer with vertical diffusion
            rho_o, c_o = sfc_config.rho_ocean, sfc_config.c_ocean
            d_mid = 0.5 * (sfc_config.h_mix + sfc_config.h_deep)
            F_mix = rho_o * c_o * sfc_config.k_mix * (T_sfc - T_deep) / d_mid
            C_mix = rho_o * c_o * sfc_config.h_mix
            C_deep_v = rho_o * c_o * sfc_config.h_deep
            dT_sfc_dt = (sw_net + lw_net - shflx - lhflx - F_mix) / C_mix
            dT_deep_dt = F_mix / C_deep_v
            T_sfc_new = jnp.maximum(T_sfc + dt * dT_sfc_dt, T_freeze)
            T_deep = T_deep + dt * dT_deep_dt
        else:
            dT_sfc_dt = (sw_net + lw_net - shflx - lhflx) / C_sfc
            T_sfc_new = jnp.maximum(T_sfc + dt * dT_sfc_dt, T_freeze)

        # (e) Bucket hydrology update (land only — traced at JIT time)
        if IS_LAND:
            W_new = jnp.clip(W_bkt + dt * (precip - evap), 0.0, W_max)
        else:
            W_new = W_bkt

        # (f) Total atmospheric tendencies
        dT_dt = dT_rad + dT_conv
        dT_dt = dT_dt.at[..., -1].add(dT_BL)
        dq_dt = dq_conv
        dq_dt = dq_dt.at[..., -1].add(dq_BL)

        # Hyperdiffusion on q_v (dampen 2Δx noise; spectral/voronoi handle in dycore)
        if _apply_hyperdiff is not None:
            dq_dt = dq_dt + _apply_hyperdiff(q_v, HYPERDIFF)

        return dT_dt, dq_dt, T_sfc_new, T_deep, W_new, precip

    # ---------------------------------------------------------------
    # Time integration
    # ---------------------------------------------------------------
    n_steps = int(args.days * 86400 / DT)
    diag_interval = int(args.diag_days * 86400 / DT)

    _ocean_label = OCEAN_MODE if not IS_LAND else "slab_soil"
    print("=" * 70)
    print(f"  Moist RCE: {_ocean_label} {args.mode} + gray radiation + SBM convection")
    print("=" * 70)
    _grid_labels = {
        "cubed_sphere": f"C{N}", "gaussian": f"T{N}",
        "latlon": f"LL{N}", "voronoi": f"V{N}",
    }
    print(f"  Grid:     {_grid_labels[grid_type]}/L{NLEV} ({grid_type}/{discretization})"
          f",  dt={DT:.0f}s,  {args.days} days")
    print(f"  Surface:  T_init={args.sst_init:.0f}K, C_sfc={C_sfc:.2e} J/m2/K")
    print()
    print(f"  {'Day':>6s}  {'<T_sfc>':>8s}  {'<T_atm>':>8s}  {'<Precip>':>8s}"
          f"  {'<CWV>':>6s}  {'max|v|':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

    t_start = time.time()

    for step in range(n_steps):
        # (1) Dynamics only (no inline physics)
        state = model.step(state, DT)

        # (2) Operator-split physics
        _T_deep_in = T_deep if IS_TWO_LAYER else jnp.zeros(shape_2d)
        dT_dt, dq_dt, T_sfc, _T_deep_out, W_bucket, precip = physics_step(
            state.T.data, state.p_s.data, q_v,
            state.u.data, state.v.data,
            T_sfc, _T_deep_in, W_bucket, grid.grid_lat, DT,
        )
        if IS_TWO_LAYER:
            T_deep = _T_deep_out
        new_T = state.T.data + DT * dT_dt
        q_v = jnp.maximum(q_v + DT * dq_dt, 0.0)

        # (3) Large-scale condensation (saturation adjustment)
        q_sat = saturation_mixing_ratio(
            new_T, state.p_s.data[..., None] * sigma.sigma_full,
        )
        excess = jnp.maximum(q_v - q_sat, 0.0)
        q_v = q_v - excess
        new_T = new_T + constants.L_v * excess / constants.c_pd
        state = state._replace(T=state.T.replace(data=new_T))

        # Large-scale precipitation: column-integrated condensation [kg/m2/s]
        ls_precip = jnp.sum(excess * state.p_s.data[..., None] * dsigma,
                            axis=-1) / constants.g / DT
        precip = precip + ls_precip  # total = convective + large-scale
        if IS_LAND:
            W_bucket = jnp.clip(W_bucket + DT * ls_precip, 0.0, W_max)

        # (4) Rayleigh friction
        state = state._replace(
            u=state.u.replace(data=state.u.data * fric_decay),
            v=state.v.replace(data=state.v.data * fric_decay),
        )

        # Diagnostics
        if (step + 1) % diag_interval == 0:
            jax.block_until_ready(state.u.data)
            day = (step + 1) * DT / 86400.0
            mean_sfc = float(jnp.mean(T_sfc))
            mean_T = float(jnp.mean(state.T.data))
            max_v = float(jnp.max(jnp.sqrt(
                state.u.data ** 2 + state.v.data ** 2)))
            mean_precip = float(jnp.mean(precip)) * 86400.0
            cwv = column_water_vapor(q_v, state.p_s.data, dsigma)
            mean_cwv = float(jnp.mean(cwv))

            print(f"  {day:6.0f}  {mean_sfc:8.2f}  {mean_T:8.2f}"
                  f"  {mean_precip:8.2f}  {mean_cwv:6.1f}  {max_v:8.2f}")

            if not jnp.all(jnp.isfinite(state.u.data)) or max_v > 500:
                print(f"  BLOWUP at day {day:.0f}")
                break

    jax.block_until_ready(state.u.data)
    total = time.time() - t_start
    print(f"\n  Complete: {total:.1f}s wall time")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
