"""Focused unit tests for the selected rung-3.6 exchange identity."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.coupler.ocean_forcing import (
    NemoSI3ExchangeConfig,
    nemo_si3_fwb_step,
    nemo_si3_ssm_step,
    nemo_si3_tra_sbc_rk3,
)
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_coupled_card
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.sea_ice import _nemo_si3_ice_update_tau
from legoesm.ocean.physics.shortwave_penetration import nemo_rgb_one_layer_rhs


def test_coupled_card_binds_decision_six_and_libm() -> None:
    card = build_c1d_omip_l3_coupled_card()
    assert card.slab_depth_m == 10.0
    assert card.exchange == NemoSI3ExchangeConfig()
    assert card.ice.precision_policy.transcendentals == "libm"


def test_ssm_first_step_has_three_seeded_copies() -> None:
    values = tuple(jnp.asarray([float(k)]) for k in range(1, 8))
    got = nemo_si3_ssm_step(values, values, kt=1, config=NemoSI3ExchangeConfig())
    # U/V and non-temperature members are unchanged by the four-copy mean.
    for index in (0, 1, 3, 4, 5, 6):
        np.testing.assert_array_equal(np.asarray(got[index]), np.asarray(values[index]))


def test_fwb_uses_registered_delayed_domain_sum() -> None:
    got = nemo_si3_fwb_step(
        emp=jnp.asarray([3.0]), qns=jnp.asarray([5.0]),
        snwice_fmass=jnp.asarray([2.0]), area=jnp.asarray([10.0]),
        mask=jnp.asarray([1.0]), emp_ext=jnp.asarray([0.0]),
        emp_corr=jnp.asarray([0.0]), domain_sum=jnp.asarray([10.0]),
        heat_capacity=jnp.asarray([2.0]), sst=jnp.asarray([-1.0]),
        active=jnp.asarray([True]), config=NemoSI3ExchangeConfig(),
    )
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray([2.0]))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray([3.0]))
    np.testing.assert_array_equal(np.asarray(got[2]), np.asarray([-1.0]))


def test_tau_implicit_arm_does_not_subtract_ocean_velocity() -> None:
    got = _nemo_si3_ice_update_tau(
        u_ocean=jnp.asarray([1.0]), v_ocean=jnp.asarray([0.0]),
        u_ice=jnp.asarray([2.0]), u_ice_west=jnp.asarray([2.0]),
        v_ice=jnp.asarray([0.0]), v_ice_south=jnp.asarray([0.0]),
        drag_io=jnp.asarray([0.01]), ice_fraction=jnp.asarray([0.5]),
        tmod_before=jnp.asarray([0.0]), taum_before=jnp.asarray([0.0]),
        utau_ocean=jnp.asarray([0.0]), vtau_ocean=jnp.asarray([0.0]),
        refresh=jnp.asarray([True]),
        rho_ocean=NEMO_SI3_CONSTANTS_CONFIG.rho_ocean,
    )
    assert float(got["rCdU_ice"][0]) < 0.0
    assert float(got["utau"][0]) > 0.0


def test_rk3_and_one_layer_rgb_source_rows() -> None:
    config = NemoSI3ExchangeConfig()
    t, s = nemo_si3_tra_sbc_rk3(
        tendency_t=1.0, tendency_s=2.0, emp=3.0, qns=4.0,
        salt_flux_pss=5.0, layer_thickness=2.0, inverse_density=0.5,
        inverse_heat_capacity=0.25, temperature=6.0, salinity=7.0,
        stage=3, config=config,
    )
    assert float(t) == 1.25
    assert float(s) == 3.25
    rhs = nemo_rgb_one_layer_rhs(1.0, 8.0, 2.0, 0.25)
    assert float(rhs) == 2.0
