#!/usr/bin/env python
"""Run all atmosphere dycore, ocean dycore and RCE tests at ~2.5° resolution.

Covers cubed-sphere (C36), lat-lon (72×144), and spectral (T42) grids.
Short integrations (a few days) to verify stability and correctness.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_all_25deg_tests.py
"""

import sys
import time
import traceback

sys.stdout.reconfigure(line_buffering=True)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

RESULTS = []


def record(name, status, wall_time, notes=""):
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!"}[status]
    RESULTS.append({"name": name, "status": status, "wall": wall_time, "notes": notes})
    print(f"  {icon} {status:5s} | {name:<55s} | {wall_time:6.1f}s | {notes}")


def check_finite(arrays: dict) -> bool:
    for name, arr in arrays.items():
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


# =========================================================================
print("=" * 72)
print("  legoESM ~2.5° Resolution Test Suite")
print("=" * 72)
print(f"  Backend: {jax.default_backend()}")
print(f"  X64:     {jax.config.jax_enable_x64}")
print(f"  Devices: {jax.devices()}")
print()

# =========================================================================
# ATMOSPHERE DYCORES — CUBED-SPHERE (C36)
# =========================================================================
N_CS = 36   # ~2.5°
DT_SW = 300.0
DT_HYDRO = 200.0  # Tighter CFL for baroclinic wave (strong jets)
DT_NH = 2.0        # Tighter CFL for NH at C36
NLEV = 20

print("=" * 72)
print("  ATMOSPHERE DYCORES — CUBED-SPHERE C36 (~2.5°)")
print("=" * 72)

# --- 1. Shallow Water FV — Williamson Test 2 (5 days) ---
test_name = "SW FV Williamson 2 (C36, 5 days)"
print(f"\n  [{1}] {test_name}")
try:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import (
        ShallowWaterModel, ShallowWaterConfig,
    )
    from legoesm.atmosphere.dynamics.williamson import williamson_test2, williamson_test5

    grid_cs = create_cubed_sphere(N_CS)
    state_init = williamson_test2(grid_cs)
    HYPERDIFF_SW = 5e16 * (48 / N_CS) ** 4
    config = ShallowWaterConfig(
        hyperdiff_coeff=HYPERDIFF_SW,
        edge_blend_strength=0.25,
    )
    model = ShallowWaterModel(grid_cs, config)

    state = state_init
    n_steps = int(5 * 86400 / DT_SW)
    t0 = time.time()
    for i in range(n_steps):
        state = model.step(state, DT_SW)
    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    h_err = float(jnp.sqrt(jnp.mean((state.h.data - state_init.h.data) ** 2)))
    ok = check_finite({"h": state.h.data, "u": state.u.data, "v": state.v.data})
    record(test_name, "PASS" if ok else "FAIL", wall, f"L2 err={h_err:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 2. Shallow Water FV — Williamson Test 5 (15 days) ---
test_name = "SW FV Williamson 5 (C36, 15 days)"
print(f"\n  [{2}] {test_name}")
try:
    state_init5 = williamson_test5(grid_cs)
    config5 = ShallowWaterConfig(
        hyperdiff_coeff=HYPERDIFF_SW,
        edge_blend_strength=0.25,
    )
    model5 = ShallowWaterModel(grid_cs, config5)
    state = state_init5
    n_steps = int(15 * 86400 / DT_SW)
    t0 = time.time()
    for i in range(n_steps):
        state = model5.step(state, DT_SW)
    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    mass_init = float(jnp.mean(state_init5.h.data))
    mass_final = float(jnp.mean(state.h.data))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    ok = check_finite({"h": state.h.data, "u": state.u.data, "v": state.v.data})
    record(test_name, "PASS" if ok else "FAIL", wall, f"mass drift={mass_drift:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 3. Hydrostatic PE FV — Held-Suarez (10 days) ---
test_name = "Hydro FV Held-Suarez (C36/L20, 10 days)"
print(f"\n  [{3}] {test_name}")
try:
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel, PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez import (
        held_suarez_forcing, held_suarez_init,
    )
    from legoesm.core.operators import global_integral

    sigma = create_sigma_coordinate(NLEV)
    HYPERDIFF = 5e16 * (48 / N_CS) ** 4
    pe_config = PrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF,
        hyperdiff_ps_coeff=HYPERDIFF,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model_pe = PrimitiveEquationModel(grid_cs, sigma, pe_config)
    state = held_suarez_init(grid_cs, sigma)
    mass_init = float(global_integral(state.p_s, grid_cs))

    n_steps = int(10 * 86400 / DT_HYDRO)
    t0 = time.time()
    for i in range(n_steps):
        state = model_pe.step_with_physics(state, DT_HYDRO, held_suarez_forcing)
    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    mass_final = float(global_integral(state.p_s, grid_cs))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data, "p_s": state.p_s.data})
    record(test_name, "PASS" if ok else "FAIL", wall, f"mass drift={mass_drift:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 4. Hydrostatic PE FV — Baroclinic Wave (10 days) ---
test_name = "Hydro FV Baroclinic Wave (C36/L20, 10 days)"
print(f"\n  [{4}] {test_name}")
try:
    from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init

    state = baroclinic_wave_init(grid_cs, sigma, perturbed=True)
    mass_init = float(global_integral(state.p_s, grid_cs))

    n_steps = int(10 * 86400 / DT_HYDRO)
    t0 = time.time()
    for i in range(n_steps):
        state = model_pe.step(state, DT_HYDRO)
    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    mass_final = float(global_integral(state.p_s, grid_cs))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    u_max = float(jnp.max(jnp.abs(state.u.data)))
    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data, "p_s": state.p_s.data})
    record(test_name, "PASS" if ok else "FAIL", wall, f"|u|_max={u_max:.1f} m/s, mass={mass_drift:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 5. Non-Hydrostatic FV — DCMIP-2025 TC1 Gravity Waves (3 hours) ---
test_name = "NH FV DCMIP-2025 TC1 (C36/L20, 3h)"
print(f"\n  [{5}] {test_name}")
try:
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc1_init

    state, height_coord, terrain_metric = dcmip25_tc1_init(grid_cs, n_levels=NLEV)
    nh_config = CompressibleEulerConfig(
        n_acoustic_substeps=10,  # More substeps for CFL at C36
        semi_implicit_acoustic=True,  # Tridiagonal solve for acoustic substeps
        sponge_width=10000.0,
        sponge_coeff=0.05,
        hyperdiff_coeff=5e16 * (48 / N_CS) ** 4,
        edge_blend_uv=0.15,
        edge_blend_w=0.15,
        edge_blend_theta=0.15,
        edge_blend_rho=0.15,
        edge_blend_tracers=0.15,
    )
    model_nh = CompressibleEulerModel(grid_cs, height_coord, terrain_metric, nh_config)

    n_steps = int(3 * 3600 / DT_NH)
    t0 = time.time()
    for i in range(n_steps):
        state = model_nh.step(state, DT_NH)
    jax.block_until_ready(state.theta_prime.data)
    wall = time.time() - t0

    ok = check_finite({
        "theta_prime": state.theta_prime.data,
        "u": state.u.data,
        "v": state.v.data,
        "w": state.w.data,
        "rho_prime": state.rho_prime.data,
    })
    w_max = float(jnp.max(jnp.abs(state.w.data)))
    record(test_name, "PASS" if ok else "FAIL", wall, f"|w|_max={w_max:.4f} m/s")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# =========================================================================
# ATMOSPHERE DYCORES — LAT-LON (72×144)
# =========================================================================
N_LAT = 72
N_LON = 144

print("\n" + "=" * 72)
print("  ATMOSPHERE DYCORES — LAT-LON 72×144 (~2.5°)")
print("=" * 72)

# --- 6. Hydrostatic PE Lat-Lon — Held-Suarez (10 days) ---
test_name = "Hydro LatLon Held-Suarez (72x144/L20, 10 days)"
print(f"\n  [{6}] {test_name}")
try:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
        LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez_latlon import (
        held_suarez_forcing_latlon, held_suarez_init_latlon,
    )
    from legoesm.core.operators_latlon import global_integral as global_integral_ll

    grid_ll = create_latlon_grid(N_LAT, N_LON)
    sigma_ll = create_sigma_coordinate(NLEV)

    HYPERDIFF_LL = 2e16 * (64 / N_LAT) ** 4
    ll_config = LatLonPrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF_LL,
        hyperdiff_ps_coeff=HYPERDIFF_LL,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model_ll = LatLonPrimitiveEquationModel(grid_ll, sigma_ll, ll_config)
    state = held_suarez_init_latlon(grid_ll, sigma_ll)
    mass_init = float(global_integral_ll(state.p_s, grid_ll))

    n_steps = int(10 * 86400 / DT_HYDRO)
    t0 = time.time()
    for i in range(n_steps):
        state = model_ll.step_with_physics(state, DT_HYDRO, held_suarez_forcing_latlon)
    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    mass_final = float(global_integral_ll(state.p_s, grid_ll))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data, "p_s": state.p_s.data})
    record(test_name, "PASS" if ok else "FAIL", wall, f"mass drift={mass_drift:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# =========================================================================
# ATMOSPHERE DYCORES — SPECTRAL (T42)
# =========================================================================
T_SPEC = 42   # ~2.8° equivalent
DT_SW_SPEC = 60.0   # Explicit SW needs small dt
DT_SPEC = 600.0      # SI allows larger dt for PE

print("\n" + "=" * 72)
print("  ATMOSPHERE DYCORES — SPECTRAL T42 (~2.8°)")
print("=" * 72)

# --- 7. Spectral Shallow Water — Williamson Test 2 (5 days) ---
test_name = "SW Spectral Williamson 2 (T42, 5 days)"
print(f"\n  [{7}] {test_name}")
try:
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig, spectral_sw_tendencies,
        williamson_test2_spectral, spectral_to_grid,
    )
    from legoesm.timestepping.ssp_rk3 import ssp_rk3_step

    grid_spec = create_gaussian_grid(T_SPEC)
    state_init_spec = williamson_test2_spectral(grid_spec)
    a_spec = grid_spec.radius
    eig_max_sw = T_SPEC * (T_SPEC + 1) / (a_spec * a_spec)
    HYPERDIFF_SW_SPEC = 1.0 / (1.0 * 3600.0 * eig_max_sw ** 2)
    sw_spec_config = SpectralSWConfig(hyperdiff_coeff=HYPERDIFF_SW_SPEC)

    def tendency_sw(s):
        return spectral_sw_tendencies(s, grid_spec, sw_spec_config)

    step_sw_jit = jax.jit(lambda s, dt: ssp_rk3_step(s, tendency_sw, dt))
    state = state_init_spec

    n_steps = int(5 * 86400 / DT_SW_SPEC)
    t0 = time.time()
    for i in range(n_steps):
        state = step_sw_jit(state, DT_SW_SPEC)
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    fields = spectral_to_grid(state, grid_spec)
    ok = check_finite({"h": fields["h"], "u": fields["u"], "v": fields["v"]})
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 8. Spectral Shallow Water — Williamson Test 5 (15 days) ---
test_name = "SW Spectral Williamson 5 (T42, 15 days)"
print(f"\n  [{8}] {test_name}")
try:
    from legoesm.atmosphere.dynamics.spectral_sw import williamson_test5_spectral

    state_init5_spec = williamson_test5_spectral(grid_spec)
    state = state_init5_spec

    n_steps = int(15 * 86400 / DT_SW_SPEC)
    t0 = time.time()
    for i in range(n_steps):
        state = step_sw_jit(state, DT_SW_SPEC)
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    fields = spectral_to_grid(state, grid_spec)
    ok = check_finite({"h": fields["h"], "u": fields["u"], "v": fields["v"]})
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 9. Spectral PE — Held-Suarez (10 days) ---
test_name = "Hydro Spectral Held-Suarez (T42/L20, 10 days)"
print(f"\n  [{9}] {test_name}")
try:
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral

    sigma_spec = create_sigma_coordinate(NLEV)
    a = grid_spec.radius
    eig_max = T_SPEC * (T_SPEC + 1) / (a * a)
    HYPERDIFF_SPEC = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

    spec_pe_config = SpectralPEConfig(
        hyperdiff_coeff=HYPERDIFF_SPEC,
        hyperdiff_order=2,
        semi_implicit=True,  # Required for stability at T42
        si_T_ref=300.0,
    )
    model_spec_pe = SpectralPrimitiveEquationModel(grid_spec, sigma_spec, spec_pe_config)
    state = isothermal_rest_state_spectral(grid_spec, sigma_spec, T_init=300.0)

    n_steps = int(10 * 86400 / DT_SPEC)
    t0 = time.time()
    for i in range(n_steps):
        state = model_spec_pe.step_with_physics(state, DT_SPEC, held_suarez_forcing_spectral)
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    ok = check_finite({
        "vor_hat": state.vor_hat.data,
        "div_hat": state.div_hat.data,
        "T_hat": state.T_hat.data,
        "lnps_hat": state.lnps_hat.data,
    })
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 10. Spectral PE — Baroclinic Wave (10 days) ---
test_name = "Hydro Spectral Baroclinic Wave (T42/L20, 10 days)"
print(f"\n  [{10}] {test_name}")
try:
    from legoesm.atmosphere.dynamics.spectral_pe import baroclinic_wave_init_spectral

    state = baroclinic_wave_init_spectral(grid_spec, sigma_spec)

    n_steps = int(10 * 86400 / DT_SPEC)
    t0 = time.time()
    for i in range(n_steps):
        state = model_spec_pe.step(state, DT_SPEC)
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    ok = check_finite({
        "vor_hat": state.vor_hat.data,
        "div_hat": state.div_hat.data,
        "T_hat": state.T_hat.data,
    })
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 11. Spectral NH — DCMIP-2025 TC1 (3 hours) ---
test_name = "NH Spectral DCMIP-2025 TC1 (T42/L20, 3h)"
print(f"\n  [{11}] {test_name}")
try:
    from legoesm.atmosphere.dynamics.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    state, hc_spec, tm_spec = dcmip25_tc1_init_spectral(grid_spec, n_levels=NLEV)
    hyperdiff_nh = 1.0 / (0.5 * 3600.0 * eig_max ** 2)
    nh_spec_config = SpectralNHConfig(
        n_acoustic_substeps=6,
        sponge_width=10000.0,
        sponge_coeff=0.05,
        hyperdiff_coeff=hyperdiff_nh,
    )
    model_nh_spec = SpectralCompressibleEulerModel(
        grid_spec, hc_spec, tm_spec, nh_spec_config,
    )

    n_steps = int(3 * 3600 / DT_NH)
    t0 = time.time()
    for i in range(n_steps):
        state = model_nh_spec.step(state, DT_NH)
    jax.block_until_ready(state.theta_prime_hat.data)
    wall = time.time() - t0

    ok = check_finite({
        "theta_prime_hat": state.theta_prime_hat.data,
        "rho_prime_hat": state.rho_prime_hat.data,
    })
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# =========================================================================
# OCEAN DYCORES — CUBED-SPHERE (C36)
# =========================================================================
DT_OCEAN = 1800.0
N_OCEAN_LEV = 20

print("\n" + "=" * 72)
print("  OCEAN DYCORES — CUBED-SPHERE C36 (~2.5°)")
print("=" * 72)

# --- 12. Ocean FV — Rest State Adjustment (5 days) ---
test_name = "Ocean FV Rest State (C36/L20, 5 days)"
print(f"\n  [{12}] {test_name}")
try:
    from legoesm.ocean import (
        OceanModel, OceanConfig,
        create_ocean_z_star, rest_state_ocean,
    )
    from legoesm.ocean.conservation import _ocean_area_sum

    z_coord = create_ocean_z_star(N_OCEAN_LEV)
    ocean_config = OceanConfig(
        A_h=1e4,
        K_h=1e3,
        A_v=1e-3,
        K_v=1e-4,
        n_barotropic_substeps=30,
        hyperdiff_coeff=0.0,
        use_conservation_fixer=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
    )
    ocean_model = OceanModel(grid_cs, z_coord, ocean_config)
    state = rest_state_ocean(grid_cs, z_coord, land_lat_threshold=90.0)

    n_steps = int(5 * 86400 / DT_OCEAN)
    t0 = time.time()
    for i in range(n_steps):
        state = ocean_model.step(state, DT_OCEAN)
    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({
        "T": state.T.data, "S": state.S.data,
        "u": state.u.data, "v": state.v.data,
        "eta": state.eta.data,
    })
    u_max = float(jnp.max(jnp.abs(state.u.data)))
    eta_max = float(jnp.max(jnp.abs(state.eta.data)))
    record(test_name, "PASS" if ok else "FAIL", wall,
           f"|u|_max={u_max:.2e}, |eta|_max={eta_max:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 13. Ocean FV — Barotropic Gravity Wave (5 days) ---
test_name = "Ocean FV Barotropic Wave (C36/L20, 5 days)"
print(f"\n  [{13}] {test_name}")
try:
    from legoesm.core.field import Field

    state = rest_state_ocean(grid_cs, z_coord, land_lat_threshold=90.0)
    # Gaussian SSH perturbation at equator
    lat = grid_cs.lat
    lon = grid_cs.lon
    eta_pert = 1.0 * jnp.exp(-((lat ** 2 + (lon - jnp.pi) ** 2) / (2 * 0.1 ** 2)))
    state = state._replace(eta=state.eta.replace(data=eta_pert))

    n_steps = int(5 * 86400 / DT_OCEAN)
    t0 = time.time()
    for i in range(n_steps):
        state = ocean_model.step(state, DT_OCEAN)
    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0

    ok = check_finite({
        "T": state.T.data, "S": state.S.data,
        "u": state.u.data, "v": state.v.data,
        "eta": state.eta.data,
    })
    eta_max = float(jnp.max(jnp.abs(state.eta.data)))
    ke = float(jnp.mean(state.u.data ** 2 + state.v.data ** 2))
    record(test_name, "PASS" if ok else "FAIL", wall,
           f"|eta|_max={eta_max:.4f} m, KE={ke:.2e}")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# --- 14. Spectral Ocean — Rest State (5 days) ---
test_name = "Ocean Spectral Rest State (T42/L20, 5 days)"
print(f"\n  [{14}] {test_name}")
try:
    from legoesm.ocean import SpectralOceanModel, SpectralOceanConfig

    spec_ocean_config = SpectralOceanConfig(
        A_h=1e4,
        K_h=1e3,
        A_v=1e-3,
        K_v=1e-4,
        hyperdiff_coeff=HYPERDIFF_SPEC,
    )
    spec_ocean_model = SpectralOceanModel(grid_spec, z_coord, spec_ocean_config)

    from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
    state = rest_state_spectral_ocean(grid_spec, z_coord, land_lat_threshold=90.0)

    n_steps = int(5 * 86400 / DT_OCEAN)
    t0 = time.time()
    for i in range(n_steps):
        state = spec_ocean_model.step(state, DT_OCEAN)
    jax.block_until_ready(state.T_hat.data)
    wall = time.time() - t0

    ok = check_finite({
        "T_hat": state.T_hat.data,
        "S_hat": state.S_hat.data,
        "vor_hat": state.vor_hat.data,
    })
    record(test_name, "PASS" if ok else "FAIL", wall)
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# =========================================================================
# RCE — CUBED-SPHERE (C36)
# =========================================================================
DT_RCE = 300.0

print("\n" + "=" * 72)
print("  RCE — CUBED-SPHERE C36 (~2.5°)")
print("=" * 72)

# --- 15. RCE Slab Ocean (10 days) ---
test_name = "RCE Slab Ocean (C36/L20, 10 days)"
print(f"\n  [{15}] {test_name}")
try:
    from legoesm import constants
    from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.ocean.simple_ocean import SimpleOceanConfig

    # Reuse grid_cs, sigma from above
    rce_state = held_suarez_init(grid_cs, sigma, T_init=280.0)

    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=0.31, perpetual_equinox=True,
    )
    sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)
    ocean_slab_config = SimpleOceanConfig(
        mode="slab", h_mix=50.0, rho_ocean=1025.0, c_ocean=3994.0,
        Q_flux=0.0, albedo_ocean=0.06, emissivity_ocean=0.97,
        Cd_ocean=1.5e-3, Ch_ocean=1.5e-3, U_min=1.0, T_freeze=271.35,
    )
    _C_mix = ocean_slab_config.rho_ocean * ocean_slab_config.c_ocean * ocean_slab_config.h_mix

    # Moisture init
    p_full_init = rce_state.p_s.data[..., None] * sigma.sigma_full
    q_sat_init = saturation_mixing_ratio(rce_state.T.data, p_full_init)
    q_v = 0.6 * q_sat_init * sigma.sigma_full ** 2
    q_v = jnp.minimum(q_v, q_sat_init)
    ocean_sst = jnp.full((6, N_CS, N_CS), 300.0)

    # Rayleigh friction parameters
    _sigma_b = 0.7
    _k_f_max = 1.0 / 86400.0
    _k_free = 0.1 / 86400.0
    _k_f = _k_free + _k_f_max * jnp.maximum(0.0, (sigma.sigma_full - _sigma_b) / (1.0 - _sigma_b))
    _fric_decay = jnp.exp(-_k_f * DT_RCE)

    @jax.jit
    def physics_step_rce(T, p_s, q_v, u, v, ocean_sst, lat, dt):
        nlev = sigma.sigma_full.shape[0]
        shape_3d = T.shape
        shape_2d = p_s.shape
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        p_full = p_s[..., None] * sigma.sigma_full
        p_half = p_s[..., None] * sigma.sigma_half
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        sst_col = ocean_sst.reshape(ncol)
        lat_col = lat.reshape(ncol)
        insol = perpetual_equinox_insolation(lat_col, gray_config.S_0)
        rad_out = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=sst_col, lat=lat_col, q_v=q_v_col,
            insolation=insol, config=gray_config,
        )
        dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)
        conv_out = sbm_convection(
            T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
            dt=dt, config=sbm_config,
        )
        dT_dt_conv = conv_out.dT_dt.reshape(shape_3d)
        dq_v_dt_conv = conv_out.dq_v_dt.reshape(shape_3d)
        dT_dt = dT_dt_rad + dT_dt_conv
        dq_v_dt = dq_v_dt_conv

        # BL coupling
        T_sfc = T[..., -1]
        q_sat_sfc = saturation_mixing_ratio(ocean_sst, p_s)
        wind_sfc = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_bl = p_s * sigma.dsigma[-1]
        SH = ocean_slab_config.Ch_ocean * wind_sfc * (ocean_sst - T_sfc) * constants.c_pd * (p_s / (constants.R_d * T_sfc))
        LH = ocean_slab_config.Ch_ocean * wind_sfc * (q_sat_sfc - q_v[..., -1]) * constants.L_v * (p_s / (constants.R_d * T_sfc))
        LH = jnp.maximum(LH, 0.0)
        dT_bl = SH * constants.g / (constants.c_pd * dp_bl)
        dq_bl = LH * constants.g / (constants.L_v * dp_bl)
        dT_dt = dT_dt.at[..., -1].add(dT_bl)
        dq_v_dt = dq_v_dt.at[..., -1].add(dq_bl)

        # Large-scale condensation
        q_sat = saturation_mixing_ratio(T + dT_dt * dt, p_full)
        q_new = q_v + dq_v_dt * dt
        excess = jnp.maximum(q_new - q_sat, 0.0)
        dT_dt = dT_dt + (constants.L_v / constants.c_pd) * excess / dt
        dq_v_dt = dq_v_dt - excess / dt

        # Ocean energy balance
        sw_sfc = insol.reshape(shape_2d) * (1.0 - gray_config.sfc_albedo)
        lw_net = rad_out.lw_flux_down_sfc.reshape(shape_2d) - rad_out.lw_flux_up_sfc.reshape(shape_2d)
        Q_ocean = sw_sfc + lw_net - SH - LH
        sst_new = ocean_sst + Q_ocean * dt / _C_mix
        sst_new = jnp.maximum(sst_new, ocean_slab_config.T_freeze)

        precip = jnp.sum(jnp.maximum(excess, 0.0) * dp_bl[..., None] / constants.g, axis=-1) if excess.ndim > 3 else jnp.zeros(shape_2d)
        return dT_dt, dq_v_dt, sst_new, precip

    state = rce_state
    lat_3d = grid_cs.lat

    n_steps = int(10 * 86400 / DT_RCE)
    t0 = time.time()
    for i in range(n_steps):
        # Dynamics step
        state = model_pe.step(state, DT_RCE)
        # Physics step
        dT_dt, dq_v_dt, ocean_sst, _ = physics_step_rce(
            state.T.data, state.p_s.data, q_v,
            state.u.data, state.v.data, ocean_sst, lat_3d, DT_RCE,
        )
        new_T = state.T.data + dT_dt * DT_RCE
        state = state._replace(
            T=state.T.replace(data=new_T),
            u=state.u.replace(data=state.u.data * _fric_decay),
            v=state.v.replace(data=state.v.data * _fric_decay),
        )
        q_v = jnp.maximum(q_v + dq_v_dt * DT_RCE, 0.0)
    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    T_mean = float(jnp.mean(state.T.data))
    sst_mean = float(jnp.mean(ocean_sst))
    ok = check_finite({
        "T": state.T.data, "u": state.u.data,
        "v": state.v.data, "p_s": state.p_s.data,
        "q_v": q_v, "sst": ocean_sst,
    })
    record(test_name, "PASS" if ok else "FAIL", wall,
           f"T_mean={T_mean:.1f}K, SST={sst_mean:.1f}K")
except Exception as e:
    record(test_name, "ERROR", 0, str(e)[:80])
    traceback.print_exc()

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)

n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
n_error = sum(1 for r in RESULTS if r["status"] == "ERROR")
total_wall = sum(r["wall"] for r in RESULTS)

print(f"\n  PASS: {n_pass}  |  FAIL: {n_fail}  |  ERROR: {n_error}  |  Total: {len(RESULTS)}")
print(f"  Total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)\n")

for r in RESULTS:
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!"}[r["status"]]
    print(f"  {icon} {r['status']:5s} | {r['name']:<55s} | {r['wall']:6.1f}s | {r['notes']}")

print("\n" + "=" * 72)
if n_fail + n_error > 0:
    print(f"  SOME TESTS FAILED ({n_fail} fail, {n_error} error)")
else:
    print("  ALL TESTS PASSED")
print("=" * 72)
