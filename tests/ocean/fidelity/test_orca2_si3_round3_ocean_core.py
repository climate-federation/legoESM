from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.bulk_flux_omip import nemo_orca2_ocean_consumption_step
from legoesm.ocean.forcing.nemo_surface import (
    NemoFwbCase2Carry,
    nemo_fwb_case2_cold_start,
    nemo_fwb_case2_step,
)
from legoesm.ocean.freshwater import FreshwaterForcing
from legoesm.ocean.state import OceanSurfaceForcing


def _fwb(carry, *, kt, annual_steps=2920, mass_b=None, reduction_mask=None):
    shape = (2, 3)
    one = jnp.ones(shape, dtype=jnp.float64)
    if mass_b is None:
        mass_b = jnp.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64
        )
    if reduction_mask is None:
        reduction_mask = jnp.asarray(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=jnp.float64
        )
    return nemo_fwb_case2_step(
        carry,
        kt=kt,
        annual_steps=annual_steps,
        forcing_stride=2,
        ssh=jnp.asarray(
            [[0.1, 0.2, 0.0], [0.3, 0.4, 0.5]], dtype=jnp.float64
        ),
        snwice_mass_b=mass_b,
        cell_area=jnp.asarray(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=jnp.float64
        ),
        surface_mask=jnp.asarray(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64
        ),
        # smask0_i additionally removes the duplicated wet north-fold point.
        interior_surface_mask=reduction_mask,
        reference_density=constants.rho_ocean_nemo,
        seconds_per_day=86400.0,
        days_per_year=365.0,
        sst_mean=4.0 * one,
        emp=1.0e-5 * one,
        qns=-30.0 * one,
        heat_capacity=constants.c_p_seawater,
    )


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def test_case2_cold_c2_c3_emp_corr_is_zero_but_state_machine_is_live():
    carry = nemo_fwb_case2_cold_start(0.0)
    first = _fwb(carry, kt=1)
    assert float(first.carry.emp_corr) == 0.0
    assert float(first.carry.a_fwb) != 999.0
    assert float(first.carry.a_fwb_b) == float(first.carry.a_fwb)
    current = first
    for kt in range(2, 101):
        current = _fwb(current.carry, kt=kt)
    assert float(current.carry.emp_corr) == 0.0

    live = _fwb(NemoFwbCase2Carry(2.0, 1.0, 0.0), kt=2921)
    assert float(live.carry.emp_corr) != 0.0


def test_case2_exact_fortran_values_qns_and_reduction_mask_plants_fire():
    """sbcfwb.F90:362-373,389-390 with distinct smask0_i and smask0."""
    first = _fwb(nemo_fwb_case2_cold_start(0.0), kt=1)
    # The four owned cells have area sum 16 and mass-weighted height
    # 4.756530214424951 m3, so a_fwb = total*rho0/area = 305.0125 kg/m2.
    first_a = np.float64(float.fromhex("0x1.3103333333333p+8"))
    np.testing.assert_array_equal(_bits(first.carry.a_fwb), _bits(first_a))
    np.testing.assert_array_equal(_bits(first.carry.a_fwb_b), _bits(first_a))
    np.testing.assert_array_equal(_bits(first.carry.emp_corr), _bits(0.0))

    # A second annual sample adds exactly one kg/m2 over every owned cell:
    # a_fwb=306.0125, a_fwb_b=305.0125, and emp_corr=1/(86400*365).
    second = _fwb(
        first.carry,
        kt=17,
        annual_steps=16,
        mass_b=jnp.arange(2.0, 8.0, dtype=jnp.float64).reshape(2, 3),
    )
    expected_carry = NemoFwbCase2Carry(
        np.float64(float.fromhex("0x1.3203333333333p+8")),
        first_a,
        np.float64(float.fromhex("0x1.106291eb9624ep-25")),
    )
    np.testing.assert_array_equal(_bits(second.carry), _bits(expected_carry))

    emp_wet = np.float64(float.fromhex("0x1.509bbb2022253p-17"))
    emp_dry = np.float64(float.fromhex("0x1.4f8b588e368f1p-17"))
    qns_wet = np.float64(float.fromhex("-0x1.e00212eba380ap+4"))
    qns_dry = np.float64(float.fromhex("-0x1.e000000000000p+4"))
    expected_emp = np.asarray(
        [[emp_wet, emp_wet, emp_dry], [emp_wet, emp_wet, emp_wet]]
    )
    expected_qns = np.asarray(
        [[qns_wet, qns_wet, qns_dry], [qns_wet, qns_wet, qns_wet]]
    )
    np.testing.assert_array_equal(_bits(second.emp), _bits(expected_emp))
    np.testing.assert_array_equal(_bits(second.qns), _bits(expected_qns))

    smask0 = jnp.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64
    )
    unmasked_sum_plant = _fwb(
        nemo_fwb_case2_cold_start(0.0),
        kt=1,
        reduction_mask=jnp.ones_like(smask0),
    )
    smask0_sum_plant = _fwb(
        nemo_fwb_case2_cold_start(0.0), kt=1, reduction_mask=smask0
    )
    assert not np.array_equal(_bits(unmasked_sum_plant.carry.a_fwb), _bits(first_a))
    assert not np.array_equal(_bits(smask0_sum_plant.carry.a_fwb), _bits(first_a))

    qns_omitted_plant = np.full((2, 3), -30.0, dtype=np.float64)
    emp_sign_flipped_plant = np.asarray(1.0e-5 - second.carry.emp_corr * smask0)
    assert not np.array_equal(_bits(qns_omitted_plant), _bits(expected_qns))
    assert not np.array_equal(_bits(emp_sign_flipped_plant), _bits(expected_emp))


def test_case2_odd_application_and_even_step_plant_fire():
    correction = jnp.asarray(2.0e-7, dtype=jnp.float64)
    carry = NemoFwbCase2Carry(3.0, 3.0, correction)
    even = _fwb(carry, kt=2)
    odd = _fwb(carry, kt=3)
    base_emp = jnp.full((2, 3), 1.0e-5, dtype=jnp.float64)
    np.testing.assert_array_equal(_bits(even.emp), _bits(base_emp))
    assert not np.array_equal(_bits(odd.emp), _bits(base_emp))

    planted_even = base_emp + correction * jnp.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64
    )
    assert not np.array_equal(_bits(even.emp), _bits(planted_even))


def test_case2_annual_guard_and_snwice_mass_b_plants_fire():
    carry = NemoFwbCase2Carry(7.0, 6.0, 3.0e-8)
    inside_year = _fwb(carry, kt=17, mass_b=jnp.full((2, 3), 9.0e8))
    np.testing.assert_array_equal(_bits(inside_year.carry), _bits(carry))

    planted_wrong_cadence = _fwb(carry, kt=17, annual_steps=16)
    assert not np.array_equal(
        _bits(planted_wrong_cadence.carry), _bits(inside_year.carry)
    )

    mass_b = jnp.arange(1.0, 7.0, dtype=jnp.float64).reshape(2, 3)
    annual = _fwb(carry, kt=17, annual_steps=16, mass_b=mass_b)
    snwice_fmass_plant = _fwb(
        carry, kt=17, annual_steps=16, mass_b=jnp.zeros_like(mass_b)
    )
    assert not np.array_equal(
        _bits(annual.carry.a_fwb), _bits(snwice_fmass_plant.carry.a_fwb)
    )


def test_case2_restart_carries_emp_corr_plant_fire():
    restored = NemoFwbCase2Carry(4.0, 3.0, jnp.asarray(8.0e-7))
    even = _fwb(restored, kt=2)
    resumed = _fwb(even.carry, kt=3)
    reset_plant = _fwb(nemo_fwb_case2_cold_start(0.0), kt=3)
    np.testing.assert_array_equal(_bits(even.carry), _bits(restored))
    assert not np.array_equal(_bits(resumed.emp), _bits(reset_plant.emp))


@pytest.fixture(scope="module")
def real_consumption_fixture():
    """Small production C-grid fixture from the existing drag/vmix tests."""
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig,
        VerticalMixingConfig,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import (
        create_ocean_z_star,
        create_partial_cell_coordinate,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z_ref = create_ocean_z_star(n_levels=5, H_max=600.0)
    bathymetry = jnp.full((6, 8), 372.0, dtype=jnp.float64)
    z_coord = create_partial_cell_coordinate(z_ref, bathymetry)
    state = rest_state_latlon_cgrid_ocean(
        grid,
        z_ref,
        T_water_init_C=20.0,
        T_deep=2.0,
        S_uniform=35.0,
        H_bathy_override=bathymetry,
        nemo_prognostic_barotropic_velocity=True,
    )
    state = state._replace(
        u=state.u.replace(
            data=state.u.data + 0.01 * state.u_mask.data[..., None]
        ),
        v=state.v.replace(
            data=state.v.data - 0.007 * state.v_mask.data[..., None]
        ),
        tke=Field(
            data=(
                jnp.full(state.T.data.shape[:-1] + (4,), 0.01)
                * state.land_mask.data[..., None]
            ),
            name="tke",
            dims=("lat", "lon", "level"),
            units="m^2/s^2",
        ),
    )
    tke = TKEConfig(
        prognostic=True,
        tke_mxl_choice=3,
        nemo_mxlice=2,
        nemo_mxl0_surface_tmask=True,
        tke_preclosure_coeff_source="carried_previous_step",
    )
    base_physics = OceanPhysicsConfig()
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke),
        lateral_mixing=type(base_physics.lateral_mixing)(scheme="none"),
        surface_forcing=type(base_physics.surface_forcing)(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    config = LatLonCGridOceanConfig.from_flat(
        A_h=0.0,
        B_h=0.0,
        K_h=0.0,
        K_bih=0.0,
        hyperdiff_coeff=0.0,
        A_v=1.0e-4,
        K_v=1.0e-5,
        n_barotropic_substeps=5,
        physics=physics,
        implicit_vertical_mixing=True,
        barotropic_solver="explicit_substep",
    )
    config = recipe._nemo_linear_bottom_drag(config)._replace(
        implicit_ice_ocean_top_drag=True
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)
    zero = jnp.zeros_like(state.eta.data)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    return SimpleNamespace(dt_s=60.0), state, model, zero, freshwater


def _real_consumption_output(fixture, hm_i, rcdu_ice):
    ocean, state, model, zero, freshwater = fixture
    wet = state.land_mask.data
    surface = OceanSurfaceForcing(
        tau_x=zero,
        tau_y=zero,
        taum=0.2 * wet,
        ice_concentration=0.5 * wet,
        ice_mean_thickness_hm_i=hm_i * wet,
        ice_ocean_top_drag_rCdu=rcdu_ice * wet,
    )
    return nemo_orca2_ocean_consumption_step(
        ocean,
        state,
        surface_forcing=surface,
        freshwater_forcing=freshwater,
        model=model,
        nemo_kt=1,
    )


def test_shared_core_real_tke_and_top_drag_routes_plants_fire(
    real_consumption_fixture,
):
    baseline = _real_consumption_output(real_consumption_fixture, 0.2, -1.0e-3)
    changed_hm = _real_consumption_output(real_consumption_fixture, 2.0, -1.0e-3)
    changed_drag = _real_consumption_output(real_consumption_fixture, 0.2, -2.0e-3)

    assert not np.array_equal(
        _bits(baseline.tke_avm.data), _bits(changed_hm.tke_avm.data)
    )
    assert not np.array_equal(_bits(baseline.u.data), _bits(changed_drag.u.data))
    assert not np.array_equal(_bits(baseline.v.data), _bits(changed_drag.v.data))


def test_shared_core_real_routes_have_finite_reverse_mode_gradient(
    real_consumption_fixture,
):
    def loss(parameters):
        out = _real_consumption_output(
            real_consumption_fixture, parameters[0], parameters[1]
        )
        return (
            jnp.sum(out.tke_avm.data)
            + jnp.sum(out.u.data * out.u.data)
            + jnp.sum(out.v.data * out.v.data)
        )

    gradient = jax.grad(loss)(jnp.asarray([0.2, -1.0e-3], dtype=jnp.float64))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(gradient != 0.0))
