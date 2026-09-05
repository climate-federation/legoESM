"""Source-level controls for SI3's selectable Lemieux-2016 landfast arm."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.dynamics import (
    SI3_LANDFAST_L16_ORACLE_OPERANDS,
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    si3_cgrid_aevp_solver,
    si3_cgrid_landfast_l16_basal_stress,
)

_SHAPE = (7, 7)
_DT_S = 1200.0  # ORCA1 namelist_cfg ice time step used by the target deck
_CONCENTRATION = 0.8
_ICE_VOLUME_M = 4.0
_SNOW_VOLUME_M = 0.1
_DEPTH_M = 10.0
_AREA_M2 = 9.0e6
_ECCENTRICITY = 2.0
_CREEP_S_INV = 1.0e-12
_STRENGTH_PA = 2.0e4
_STRENGTH_DECAY = 20.0
_RHO_SNOW = 330.0
_RHO_ICE = 917.0
_RHO_WATER = 1026.0
_RHO_OCEAN = 1026.0
_GRAVITY = 9.80665
_RN_ISHLAT = 2.0
_LF_DEPFRA = 0.125  # ORCA1 namelist_ice_ref:59
_LF_BFR_N_M3 = 15.0  # ORCA1 namelist_ice_ref:61
_LF_RELAX_S_INV = 1.0e-5  # ORCA1 namelist_ice_ref:62
_LF_TENSILE = 0.05  # ORCA1 namelist_ice_ref:63
_ICEBERG_MASK = 0.25
_INITIAL_SPEED_M_S = 0.02
_PLANT_BFR_N_M3 = 14.0
_PLANT_DEPTH_M = 9.0
_DYNAMIC_BRANCH_AIR_STRESS_PA = -0.1
_AEVP_ALPHA_FLOOR = 50.0  # icedyn_rhg_evp.F90:446,467,474
_LANDFAST_SPEED_FLOOR_M_S = 5.0e-5  # icedyn_rhg_evp.F90:541,592,647,699


def _fixture():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    one = jnp.ones(_SHAPE, dtype=jnp.float64)
    zero = jnp.zeros(_SHAPE, dtype=jnp.float64)
    metric = jnp.full(_SHAPE, math.sqrt(_AREA_M2), dtype=jnp.float64)
    area = jnp.full(_SHAPE, _AREA_M2, dtype=jnp.float64)
    metrics = SI3CGridMetrics(
        metric,
        metric,
        metric,
        metric,
        metric,
        metric,
        metric,
        metric,
        area,
        area,
        area,
        area,
    )
    forcing = SI3CGridAEVPForcing(
        concentration_t=jnp.full(_SHAPE, _CONCENTRATION, dtype=jnp.float64),
        ice_volume_t=jnp.full(_SHAPE, _ICE_VOLUME_M, dtype=jnp.float64),
        snow_volume_t=jnp.full(_SHAPE, _SNOW_VOLUME_M, dtype=jnp.float64),
        pond_volume_t=zero,
        lid_volume_t=zero,
        air_stress_u_t=zero,
        air_stress_v_t=zero,
        drag_io_t=zero,
        ocean_u_u=zero,
        ocean_v_v=zero,
        ssh_t=zero,
        coriolis_t=zero,
        tmask_t=one,
        umask_u=one,
        vmask_v=one,
        fast_tmask=zero,
        depth_t=jnp.full(_SHAPE, _DEPTH_M, dtype=jnp.float64),
        depth_u=jnp.full(_SHAPE, _DEPTH_M, dtype=jnp.float64),
        depth_v=jnp.full(_SHAPE, _DEPTH_M, dtype=jnp.float64),
        iceberg_tmask=zero,
        iceberg_umask=zero,
        iceberg_vmask=zero,
    )
    config = SI3CGridAEVPConfig(
        scheme="si3_aevp",
        staggering="si3_c_grid",
        dt_s=_DT_S,
        n_subcycles=1,
        eccentricity=_ECCENTRICITY,
        creep_limit_s_inv=_CREEP_S_INV,
        strength_parameter_pa=_STRENGTH_PA,
        strength_decay=_STRENGTH_DECAY,
        rho_snow=_RHO_SNOW,
        rho_ice=_RHO_ICE,
        rho_water=_RHO_WATER,
        rho_ocean=_RHO_OCEAN,
        gravity=_GRAVITY,
        rn_ishlat=_RN_ISHLAT,
        halo_width=2,
        category_count=1,
        landfast=True,
        landfast_depth_fraction=_LF_DEPFRA,
        landfast_basal_friction_n_m3=_LF_BFR_N_M3,
        landfast_relaxation_s_inv=_LF_RELAX_S_INV,
        landfast_tensile_fraction=_LF_TENSILE,
        convergence_check=0,
    )
    velocity = jnp.full(_SHAPE, _INITIAL_SPEED_M_S, dtype=jnp.float64)
    state = SI3CGridAEVPState(velocity, velocity, zero, zero, zero)
    return state, forcing, metrics, config


def test_basal_coefficients_match_hand_written_depth_and_iceberg_branches():
    _, forcing, metrics, config = _fixture()
    area_fraction = forcing.concentration_t
    base_u, base_v, base_t = jax.jit(
        lambda: si3_cgrid_landfast_l16_basal_stress(
            forcing,
            metrics,
            config,
            area_fraction,
            area_fraction,
        )
    )()
    critical = _CONCENTRATION * _LF_DEPFRA * _DEPTH_M
    expected = (
        -_LF_BFR_N_M3
        * max(0.0, _ICE_VOLUME_M - critical)
        * math.exp(-_STRENGTH_DECAY * (1.0 - _CONCENTRATION))
    )
    np.testing.assert_array_equal(base_u, np.full(_SHAPE, expected))
    np.testing.assert_array_equal(base_v, np.full(_SHAPE, expected))
    np.testing.assert_array_equal(base_t, np.full(_SHAPE, expected))

    iceberg = forcing._replace(
        iceberg_umask=forcing.iceberg_umask.at[3, 3].set(_ICEBERG_MASK)
    )
    planted_u, _, _ = si3_cgrid_landfast_l16_basal_stress(
        iceberg,
        metrics,
        config,
        area_fraction,
        area_fraction,
    )
    assert float(planted_u[3, 3]) == -_LF_BFR_N_M3 * _ICEBERG_MASK


def test_landfast_solver_jits_and_basal_plant_moves_scored_velocity_row():
    state, forcing, metrics, config = _fixture()
    forcing = forcing._replace(
        air_stress_u_t=jnp.full(
            _SHAPE, _DYNAMIC_BRANCH_AIR_STRESS_PA, dtype=jnp.float64
        ),
        air_stress_v_t=jnp.full(
            _SHAPE, _DYNAMIC_BRANCH_AIR_STRESS_PA, dtype=jnp.float64
        ),
    )
    solve = jax.jit(
        lambda current_forcing: si3_cgrid_aevp_solver(
            state,
            current_forcing,
            metrics,
            config,
        )
    )
    baseline = solve(forcing)
    planted = solve(
        forcing._replace(
            depth_u=jnp.full(_SHAPE, _PLANT_DEPTH_M, dtype=jnp.float64)
        ),
    )
    jax.block_until_ready((baseline, planted))
    baseline_row = np.asarray(baseline.u_ice_u)[2:-2, 2:-2]
    planted_row = np.asarray(planted.u_ice_u)[2:-2, 2:-2]
    assert np.all(np.isfinite(baseline_row))
    assert np.count_nonzero(baseline_row != planted_row) > 0


def test_landfast_depth_path_has_finite_nonzero_gradient():
    state, forcing, metrics, config = _fixture()
    forcing = forcing._replace(
        air_stress_u_t=jnp.full(
            _SHAPE, _DYNAMIC_BRANCH_AIR_STRESS_PA, dtype=jnp.float64
        )
    )

    def interior_velocity_sum(depth):
        current_forcing = forcing._replace(
            depth_u=jnp.full(_SHAPE, depth, dtype=jnp.float64)
        )
        result = si3_cgrid_aevp_solver(state, current_forcing, metrics, config)
        return jnp.sum(result.u_ice_u[2:-2, 2:-2])

    derivative = jax.jit(jax.grad(interior_velocity_sum))(
        jnp.asarray(_PLANT_DEPTH_M, dtype=jnp.float64)
    )
    assert np.isfinite(float(derivative))
    assert float(derivative) != 0.0


def test_static_friction_branch_matches_hand_written_aevp_update():
    state, forcing, metrics, config = _fixture()
    result = jax.jit(
        lambda: si3_cgrid_aevp_solver(state, forcing, metrics, config)
    )()
    mass = _RHO_SNOW * _SNOW_VOLUME_M + _RHO_ICE * _ICE_VOLUME_M
    strength = (
        _STRENGTH_PA
        * _ICE_VOLUME_M
        * math.exp(-_STRENGTH_DECAY * (1.0 - _CONCENTRATION))
    )
    p_over_delta = strength / _CREEP_S_INV
    alpha_inner = 0.5 * p_over_delta * (1.0 / _AREA_M2) * (_DT_S / mass)
    beta = max(_AEVP_ALPHA_FLOOR, math.pi * math.sqrt(alpha_inner))
    decay = max(0.0, beta - _DT_S * _LF_RELAX_S_INV)
    expected = (_INITIAL_SPEED_M_S + _INITIAL_SPEED_M_S * decay) / (beta + 1.0)
    assert float(result.u_ice_u[3, 3]) == expected
    assert float(result.v_ice_v[3, 3]) == expected


def test_dynamic_basal_denominator_matches_hand_algebra():
    state, forcing, metrics, config = _fixture()
    forcing = forcing._replace(
        air_stress_u_t=jnp.full(
            _SHAPE, _DYNAMIC_BRANCH_AIR_STRESS_PA, dtype=jnp.float64
        ),
        air_stress_v_t=jnp.full(
            _SHAPE, _DYNAMIC_BRANCH_AIR_STRESS_PA, dtype=jnp.float64
        ),
    )
    result = jax.jit(
        lambda: si3_cgrid_aevp_solver(state, forcing, metrics, config)
    )()

    mass = _RHO_SNOW * _SNOW_VOLUME_M + _RHO_ICE * _ICE_VOLUME_M
    attenuation = math.exp(-_STRENGTH_DECAY * (1.0 - _CONCENTRATION))
    strength = _STRENGTH_PA * _ICE_VOLUME_M * attenuation
    p_over_delta = strength / _CREEP_S_INV
    alpha_inner = 0.5 * p_over_delta * (1.0 / _AREA_M2) * (_DT_S / mass)
    beta = max(_AEVP_ALPHA_FLOOR, math.pi * math.sqrt(alpha_inner))

    critical = _CONCENTRATION * _LF_DEPFRA * _DEPTH_M
    basal = -_LF_BFR_N_M3 * max(0.0, _ICE_VOLUME_M - critical) * attenuation
    basal_speed = _LANDFAST_SPEED_FLOOR_M_S + math.sqrt(
        _INITIAL_SPEED_M_S * _INITIAL_SPEED_M_S
        + _INITIAL_SPEED_M_S * _INITIAL_SPEED_M_S
    )
    mass_over_dt = mass / _DT_S
    tau_air = _CONCENTRATION * _DYNAMIC_BRANCH_AIR_STRESS_PA
    numerator = mass_over_dt * (
        beta * _INITIAL_SPEED_M_S + _INITIAL_SPEED_M_S
    ) + tau_air
    denominator = mass_over_dt * (beta + 1.0) - basal / basal_speed
    expected_velocity = numerator / denominator

    assert float(result.u_ice_u[3, 3]) == expected_velocity
    assert float(result.v_ice_v[3, 3]) == expected_velocity


def test_tensile_stress_matches_hand_written_nonzero_strain_update():
    state, forcing, metrics, config = _fixture()
    x_increment = np.float64(1.0e-3)
    u_ice = jnp.broadcast_to(
        jnp.arange(_SHAPE[0], dtype=jnp.float64)[:, None] * x_increment,
        _SHAPE,
    )
    zero = jnp.zeros(_SHAPE, dtype=jnp.float64)
    strained = state._replace(u_ice_u=u_ice, v_ice_v=zero)
    result = jax.jit(
        lambda: si3_cgrid_aevp_solver(strained, forcing, metrics, config)
    )()

    metric = np.float64(math.sqrt(_AREA_M2))
    divergence = np.float64(metric * x_increment)
    divergence = np.float64(divergence / np.float64(_AREA_M2))
    tension = divergence
    inverse_eccentricity_square = np.float64(1.0 / (_ECCENTRICITY * _ECCENTRICITY))
    delta = np.float64(tension * tension)
    delta = np.float64(delta * inverse_eccentricity_square)
    delta = np.float64(divergence * divergence + delta)
    delta = np.float64(math.sqrt(delta))
    attenuation = np.float64(
        math.exp(-_STRENGTH_DECAY * (1.0 - _CONCENTRATION))
    )
    strength = np.float64(_STRENGTH_PA * _ICE_VOLUME_M)
    strength = np.float64(strength * attenuation)
    p_over_delta = np.float64(strength / np.float64(delta + _CREEP_S_INV))
    mass = np.float64(_RHO_SNOW * _SNOW_VOLUME_M + _RHO_ICE * _ICE_VOLUME_M)
    alpha_inner = np.float64(0.5 * p_over_delta)
    alpha_inner = np.float64(alpha_inner * np.float64(1.0 / _AREA_M2))
    alpha_inner = np.float64(alpha_inner * np.float64(_DT_S / mass))
    alpha = np.float64(
        max(_AEVP_ALPHA_FLOOR, math.pi * math.sqrt(alpha_inner))
    )
    target = np.float64(divergence * np.float64(1.0 + _LF_TENSILE))
    target = np.float64(
        target - np.float64(delta * np.float64(1.0 - _LF_TENSILE))
    )
    target = np.float64(p_over_delta * target)
    expected = np.float64(target / np.float64(alpha + 1.0))
    assert float(result.stress1_t[3, 3]) == float(expected)


def test_landfast_rejects_non_orca1_coefficients_and_off_is_noop():
    state, forcing, metrics, config = _fixture()
    with pytest.raises(ValueError, match="do not match ORCA1"):
        si3_cgrid_aevp_solver(
            state,
            forcing,
            metrics,
            config._replace(landfast_basal_friction_n_m3=_PLANT_BFR_N_M3),
        )
    off = config._replace(landfast=False)
    base = si3_cgrid_landfast_l16_basal_stress(
        forcing,
        metrics,
        off,
        forcing.concentration_t,
        forcing.concentration_t,
    )
    assert all(np.count_nonzero(np.asarray(value)) == 0 for value in base)


def test_future_orca2_variant_operand_registry_is_complete():
    assert SI3_LANDFAST_L16_ORACLE_OPERANDS == (
        "concentration_t",
        "ice_volume_t",
        "depth_t",
        "depth_u",
        "depth_v",
        "iceberg_tmask",
        "iceberg_umask",
        "iceberg_vmask",
        "fast_tmask",
        "landfast_depth_fraction",
        "landfast_basal_friction_n_m3",
        "landfast_relaxation_s_inv",
        "landfast_tensile_fraction",
    )
