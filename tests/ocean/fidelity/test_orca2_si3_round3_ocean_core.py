from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.bulk_flux_omip import nemo_orca2_ocean_consumption_step
from legoesm.ocean.forcing.nemo_surface import (
    NemoFwbCase2Carry,
    nemo_fwb_case2_cold_start,
    nemo_fwb_case2_step,
)
from legoesm.ocean.freshwater import FreshwaterForcing
from legoesm.ocean.state import OceanSurfaceForcing


def _fwb(carry, *, kt, annual_steps=2920, mass_b=None):
    shape = (2, 3)
    one = jnp.ones(shape, dtype=jnp.float64)
    if mass_b is None:
        mass_b = jnp.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64
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


def _surface(x):
    one = jnp.ones((2, 2), dtype=jnp.float64)
    qns = x * one
    qsr = 2.0 * x * one
    emp = 3.0e-6 * x * one
    sfx = 4.0e-6 * x * one
    return OceanSurfaceForcing(
        sw_down=qsr,
        q_net=qns + qsr,
        tau_x=5.0 * x * one,
        tau_y=6.0 * x * one,
        freshwater=-emp,
        taum=7.0 * x * one,
        utau_u_native=8.0 * x * one,
        vtau_v_native=9.0 * x * one,
        q_nonsolar=qns,
        salt_flux=1.0e-3 * sfx,
        salt_flux_sfx=sfx,
        ice_concentration=0.5 * one,
        ice_ocean_top_drag_rCdu=-1.1e-3 * x * one,
        ice_mean_thickness_hm_i=1.2 * x * one,
    )


def _freshwater(x):
    z = jnp.zeros((2, 2), dtype=jnp.float64)
    return FreshwaterForcing(z, 3.0e-6 * x * jnp.ones_like(z), z, z, z)


class _RouteModel:
    def __init__(self):
        self.surface = None

    def step(self, state, dt, *, freshwater, surface_forcing, nemo_kt):
        del freshwater, nemo_kt
        self.surface = surface_forcing
        return state + dt * (
            surface_forcing.ice_mean_thickness_hm_i
            + surface_forcing.ice_ocean_top_drag_rCdu
        )


def test_shared_core_routes_hm_i_and_rcdu_top_drag_plants_fire():
    model = _RouteModel()
    surface = _surface(jnp.asarray(1.0, dtype=jnp.float64))
    ocean = SimpleNamespace(dt_s=2.0)
    state = jnp.zeros((2, 2), dtype=jnp.float64)
    actual = nemo_orca2_ocean_consumption_step(
        ocean,
        state,
        surface_forcing=surface,
        freshwater_forcing=_freshwater(1.0),
        model=model,
        nemo_kt=1,
    )
    assert model.surface is surface

    no_hm_i = surface._replace(ice_mean_thickness_hm_i=jnp.zeros_like(state))
    no_rcdu = surface._replace(ice_ocean_top_drag_rCdu=jnp.zeros_like(state))
    planted_hm = nemo_orca2_ocean_consumption_step(
        ocean, state, surface_forcing=no_hm_i,
        freshwater_forcing=_freshwater(1.0), model=_RouteModel(), nemo_kt=1
    )
    planted_rcdu = nemo_orca2_ocean_consumption_step(
        ocean, state, surface_forcing=no_rcdu,
        freshwater_forcing=_freshwater(1.0), model=_RouteModel(), nemo_kt=1
    )
    assert not np.array_equal(_bits(actual), _bits(planted_hm))
    assert not np.array_equal(_bits(actual), _bits(planted_rcdu))


def test_shared_core_has_finite_reverse_mode_gradient():
    ocean = SimpleNamespace(dt_s=0.25)
    state = jnp.zeros((2, 2), dtype=jnp.float64)

    def loss(x):
        return jnp.sum(nemo_orca2_ocean_consumption_step(
            ocean,
            state,
            surface_forcing=_surface(x),
            freshwater_forcing=_freshwater(x),
            model=_RouteModel(),
            nemo_kt=1,
        ))

    gradient = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
