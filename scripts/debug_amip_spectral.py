#!/usr/bin/env python
"""Quick diagnostic: test spectral PE stability with and without physics."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np

TRUNC = 21
NLEV = 20
DT = 600.0

from legoesm.grids.gaussian import (
    create_gaussian_grid, sh_analysis_3d, sh_synthesis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel, SpectralPEConfig,
    SpectralHydrostaticState, isothermal_rest_state_spectral,
    spectral_pe_to_grid, spectral_pe_tendencies,
)
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants

grid = create_gaussian_grid(TRUNC)
sigma = create_sigma_coordinate(NLEV)
a = grid.radius
eig_max = TRUNC * (TRUNC + 1) / (a**2)

HYPERDIFF = 1.0 / (0.15 * 3600.0 * eig_max**2)

pe_config = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_order=2,
    time_integrator="ssp_rk54",
    semi_implicit=False,
)

# ---- Test 1: Pure dynamics (no physics) ----
print("=" * 60)
print("TEST 1: Pure dynamics, no physics, isothermal rest state")
print("=" * 60)

state = isothermal_rest_state_spectral(grid, sigma, T_init=270.0)

@jax.jit
def step_pure(state):
    def tendency_fn(s):
        return spectral_pe_tendencies(s, grid, sigma, pe_config)
    return ssp_rk54_step(state, tendency_fn, DT)

for i in range(500):
    state = step_pure(state)
    if (i + 1) % 50 == 0:
        jax.block_until_ready(state.vor_hat.data)
        fields = spectral_pe_to_grid(state, grid, sigma)
        T = fields['T']
        u = fields['u']
        v = fields['v']
        p_s = fields['p_s']
        day = (i + 1) * DT / 86400.0
        max_v = float(jnp.max(jnp.sqrt(u**2 + v**2)))
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        ps_mean = float(jnp.mean(p_s))
        vor_max = float(jnp.max(jnp.abs(state.vor_hat.data)))
        div_max = float(jnp.max(jnp.abs(state.div_hat.data)))
        finite = bool(jnp.all(jnp.isfinite(state.vor_hat.data)))
        print(f"  Step {i+1:4d} (day {day:5.1f}): T=[{T_min:.2f}, {T_mean:.2f}, {T_max:.2f}] "
              f"max|v|={max_v:.4f} ps={ps_mean:.1f} "
              f"vor_max={vor_max:.2e} div_max={div_max:.2e} finite={finite}")
        if not finite:
            print("  BLOWUP in pure dynamics!")
            break

print()

# ---- Test 2: Dynamics + radiation (using physics_fn) ----
print("=" * 60)
print("TEST 2: Dynamics + gray radiation + friction (via step_with_physics)")
print("=" * 60)

from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.grids.gaussian import sh_analysis_oc2_3d, sh_analysis_dmu_3d

state2 = isothermal_rest_state_spectral(grid, sigma, T_init=270.0)
shape_2d = (grid.n_lat, grid.n_lon)
shape_3d = (grid.n_lat, grid.n_lon, NLEV)

# Moisture
p_s_init = 1.0e5
q_v = 0.5 * saturation_mixing_ratio(
    jnp.full(shape_3d, 270.0),
    jnp.broadcast_to(p_s_init * sigma.sigma_full, shape_3d),
) * sigma.sigma_full**2

# Analytical SST
lat_deg = np.degrees(np.asarray(grid.lat))
sst_1d = 27.0 * (1.0 - np.sin(np.radians(lat_deg))**2) + 273.15
sst = jnp.broadcast_to(jnp.array(sst_1d)[:, None], shape_2d)
sic = jnp.zeros(shape_2d)

lat_2d = jnp.broadcast_to(grid.lat[:, None], shape_2d)
gray_config = GrayRadiationConfig(tau_equator=7.2, tau_pole=1.8, S_0=1360.0, sfc_albedo=0.06)

_T_ice = 271.35
_k_f_max = 1.0 / 86400.0
_k_free = 0.1 / 86400.0
_sigma_b = 0.85
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (sigma.sigma_full - _sigma_b) / (1.0 - _sigma_b))

# Direct RK54 step with physics inside tendency (avoiding static closure issue)
@jax.jit
def step_with_phys(state, q_v_in):
    def tendency_fn(s):
        fields = spectral_pe_to_grid(s, grid, sigma)
        T_g = fields['T']
        u_g = fields['u']
        v_g = fields['v']
        p_s_g = fields['p_s']

        T_sfc = sst  # constant SST for this test
        p_full = p_s_g[..., None] * sigma.sigma_full
        p_half = p_s_g[..., None] * sigma.sigma_half

        T_col = T_g.reshape(-1, NLEV)
        p_full_col = p_full.reshape(-1, NLEV)
        p_half_col = p_half.reshape(-1, NLEV + 1)
        q_v_col = q_v_in.reshape(-1, NLEV)
        T_sfc_col = T_sfc.reshape(-1)
        lat_col = lat_2d.reshape(-1)

        insol = daily_mean_insolation(lat_col, 1.0, 1360.0)
        rad_out = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

        du_dt = -_k_f * u_g
        dv_dt = -_k_f * v_g

        _im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        _one_over_a = 1.0 / a
        cos_lat_3d = grid.cos_lat[:, None, None]
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        dvor_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + _one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )
        ddiv_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - _one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )
        dT_hat = sh_analysis_3d(grid, dT_dt_rad)
        dlnps_hat = jnp.zeros_like(s.lnps_hat.data)

        phys = SpectralHydrostaticState(
            vor_hat=s.vor_hat.replace(data=dvor_hat),
            div_hat=s.div_hat.replace(data=ddiv_hat),
            T_hat=s.T_hat.replace(data=dT_hat),
            lnps_hat=s.lnps_hat.replace(data=dlnps_hat),
            phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
        )
        return spectral_pe_tendencies(s, grid, sigma, pe_config, phys)

    return ssp_rk54_step(state, tendency_fn, DT)

for i in range(500):
    state2 = step_with_phys(state2, q_v)
    if (i + 1) % 50 == 0:
        jax.block_until_ready(state2.vor_hat.data)
        fields = spectral_pe_to_grid(state2, grid, sigma)
        T = fields['T']
        u = fields['u']
        v = fields['v']
        p_s = fields['p_s']
        day = (i + 1) * DT / 86400.0
        max_v = float(jnp.max(jnp.sqrt(u**2 + v**2)))
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        ps_mean = float(jnp.mean(p_s))
        finite = bool(jnp.all(jnp.isfinite(state2.vor_hat.data)))
        print(f"  Step {i+1:4d} (day {day:5.1f}): T=[{T_min:.2f}, {T_mean:.2f}, {T_max:.2f}] "
              f"max|v|={max_v:.4f} ps={ps_mean:.1f} finite={finite}")
        if not finite:
            print("  BLOWUP in dynamics+physics!")
            break

# ---- Test 3: Same but using model.step_with_physics (to test closure issue) ----
print()
print("=" * 60)
print("TEST 3: Using model.step_with_physics (closure-based physics_fn)")
print("=" * 60)

model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)
state3 = isothermal_rest_state_spectral(grid, sigma, T_init=270.0)

_fs = {'sst': sst, 'sic': sic, 'day': 1.0, 'q_v': q_v}

def closure_physics_fn(state, grid, sigma_coord):
    fields = spectral_pe_to_grid(state, grid, sigma_coord)
    T_g = fields['T']
    u_g = fields['u']
    v_g = fields['v']
    p_s_g = fields['p_s']

    T_sfc = _fs['sst']
    p_full = p_s_g[..., None] * sigma_coord.sigma_full
    p_half = p_s_g[..., None] * sigma_coord.sigma_half

    T_col = T_g.reshape(-1, NLEV)
    p_full_col = p_full.reshape(-1, NLEV)
    p_half_col = p_half.reshape(-1, NLEV + 1)
    q_v_col = _fs['q_v'].reshape(-1, NLEV)
    T_sfc_col = T_sfc.reshape(-1)
    lat_col = lat_2d.reshape(-1)

    insol = daily_mean_insolation(lat_col, 1.0, 1360.0)
    rad_out = gray_radiation(
        T=T_col, p_full=p_full_col, p_half=p_half_col,
        sfc_temperature=T_sfc_col, lat=lat_col,
        q_v=q_v_col, insolation=insol, config=gray_config,
    )
    dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

    du_dt = -_k_f * u_g
    dv_dt = -_k_f * v_g

    _im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    _one_over_a = 1.0 / a
    cos_lat_3d = grid.cos_lat[:, None, None]
    du_cos = du_dt * cos_lat_3d
    dv_cos = dv_dt * cos_lat_3d

    dvor_hat = (
        _im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
        + _one_over_a * sh_analysis_dmu_3d(grid, du_cos)
    )
    ddiv_hat = (
        _im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
        - _one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
    )
    dT_hat = sh_analysis_3d(grid, dT_dt_rad)
    dlnps_hat = jnp.zeros_like(state.lnps_hat.data)

    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
    )

for i in range(500):
    state3 = model.step_with_physics(state3, DT, closure_physics_fn)
    if (i + 1) % 50 == 0:
        jax.block_until_ready(state3.vor_hat.data)
        fields = spectral_pe_to_grid(state3, grid, sigma)
        T = fields['T']
        u = fields['u']
        v = fields['v']
        day = (i + 1) * DT / 86400.0
        max_v = float(jnp.max(jnp.sqrt(u**2 + v**2)))
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        finite = bool(jnp.all(jnp.isfinite(state3.vor_hat.data)))
        print(f"  Step {i+1:4d} (day {day:5.1f}): T=[{T_min:.2f}, {T_mean:.2f}, {T_max:.2f}] "
              f"max|v|={max_v:.4f} finite={finite}")
        if not finite:
            print("  BLOWUP in model.step_with_physics!")
            break

# ---- Test 4: Semi-implicit, pure dynamics ----
print()
print("=" * 60)
print("TEST 4: Semi-implicit, pure dynamics, isothermal rest state")
print("=" * 60)

si_config = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_order=2,
    time_integrator="ssp_rk3",
    semi_implicit=True,
    si_T_ref=270.0,
    si_alpha=0.5,
)
si_model = SpectralPrimitiveEquationModel(grid, sigma, si_config)
state4 = isothermal_rest_state_spectral(grid, sigma, T_init=270.0)

for i in range(500):
    state4 = si_model.step(state4, DT)
    if (i + 1) % 50 == 0:
        jax.block_until_ready(state4.vor_hat.data)
        fields = spectral_pe_to_grid(state4, grid, sigma)
        T = fields['T']
        u = fields['u']
        v = fields['v']
        p_s = fields['p_s']
        day = (i + 1) * DT / 86400.0
        max_v = float(jnp.max(jnp.sqrt(u**2 + v**2)))
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        ps_mean = float(jnp.mean(p_s))
        finite = bool(jnp.all(jnp.isfinite(state4.vor_hat.data)))
        print(f"  Step {i+1:4d} (day {day:5.1f}): T=[{T_min:.2f}, {T_mean:.2f}, {T_max:.2f}] "
              f"max|v|={max_v:.4f} ps={ps_mean:.1f} finite={finite}")
        if not finite:
            print("  BLOWUP in SI pure dynamics!")
            break

# ---- Test 5: Semi-implicit + physics ----
print()
print("=" * 60)
print("TEST 5: Semi-implicit + physics (radiation + friction)")
print("=" * 60)

si_model2 = SpectralPrimitiveEquationModel(grid, sigma, si_config)
state5 = isothermal_rest_state_spectral(grid, sigma, T_init=270.0)

for i in range(500):
    state5 = si_model2.step_with_physics(state5, DT, closure_physics_fn)
    if (i + 1) % 50 == 0:
        jax.block_until_ready(state5.vor_hat.data)
        fields = spectral_pe_to_grid(state5, grid, sigma)
        T = fields['T']
        u = fields['u']
        v = fields['v']
        day = (i + 1) * DT / 86400.0
        max_v = float(jnp.max(jnp.sqrt(u**2 + v**2)))
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        finite = bool(jnp.all(jnp.isfinite(state5.vor_hat.data)))
        print(f"  Step {i+1:4d} (day {day:5.1f}): T=[{T_min:.2f}, {T_mean:.2f}, {T_max:.2f}] "
              f"max|v|={max_v:.4f} finite={finite}")
        if not finite:
            print("  BLOWUP in SI + physics!")
            break

print("\nDone.")
