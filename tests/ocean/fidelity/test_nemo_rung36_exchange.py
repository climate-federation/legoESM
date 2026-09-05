"""Focused unit tests for the selected rung-3.6 exchange identity."""

from __future__ import annotations

import struct

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy
from legoesm.coupler.ocean_forcing import (
    NemoSI3ExchangeConfig,
    nemo_si3_fwb_step,
    nemo_si3_ssm_step,
    nemo_si3_tra_sbc_rk3,
)
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_coupled_card
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.sea_ice import _nemo_si3_ice_update_tau
from legoesm.ocean.eos import nemo_eos_fzp
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    nemo_freshwater_eta_tendency,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_c1d_omip_l3_slab_ocean_card,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_stagger_surface_stress,
    nemo_top_drag_rate_faces,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_flux_form_barotropic_velocity_update,
    nemo_literal_ssh_forward,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.physics.shortwave_penetration import nemo_rgb_one_layer_rhs
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_exchange_gate import (
    GateError,
    _nemo_bilinear_months,
    _nemo_monthly_interp,
    _ssm,
    evaluate as evaluate_exchange,
)
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
    STAGGER_FIELDS,
    _records as _ocean_records,
)


def test_rung36_exchange_gate_names_missing_stream(tmp_path) -> None:
    with pytest.raises(GateError, match="missing required rung-3.6 stream"):
        evaluate_exchange(tmp_path)


def test_coupled_card_binds_decision_six_and_libm() -> None:
    card = build_c1d_omip_l3_coupled_card()
    assert card.slab_depth_m == 10.0
    assert card.ocean_dt_seconds == 3600.0
    assert card.ice_dt_seconds == 14400.0
    assert card.ice_cadence == 4
    assert card.exchange == NemoSI3ExchangeConfig()
    assert card.ice.precision_policy.transcendentals == "libm"
    ocean = build_c1d_omip_l3_slab_ocean_card()
    assert ocean.recipe.model_config.surface_stress_implicit is True
    assert (
        ocean.recipe.model_config.barotropic.nemo_stage_mean_imposition
        is True
    )


def test_rung36_freshwater_ssh_uses_stp2d_statement_order() -> None:
    """NEMO stp2d.F90:248-251 stores r1_rho0 before multiplying emp."""
    zero = jnp.asarray([[0.0]], dtype=jnp.float64)
    # The ocean card receives -emp as ice_fw (positive means ice melt).
    ice_fw = jnp.asarray([[-5.1736057719122322e-5]], dtype=jnp.float64)
    forcing = FreshwaterForcing(zero, zero, zero, ice_fw, zero)
    got = nemo_freshwater_eta_tendency(forcing, 1026.0)
    expected = np.float64(1.0 / 1026.0) * np.asarray(ice_fw)
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_ssm_first_step_has_three_seeded_copies() -> None:
    values = tuple(jnp.asarray([float(k)]) for k in range(1, 8))
    got = nemo_si3_ssm_step(values, values, kt=1, config=NemoSI3ExchangeConfig())
    # U/V and non-temperature members are unchanged by the four-copy mean.
    for index in (0, 1, 3, 4, 5, 6):
        np.testing.assert_array_equal(np.asarray(got[index]), np.asarray(values[index]))


def test_fzp_keeps_nemo_source_statement_boundaries() -> None:
    """TEOS-10 arm follows eosbn2.F90:1676-1682 assignment order."""
    salinity = np.asarray([34.0, 34.125, 35.0], dtype=np.float64)
    inv_s0 = np.float64(1.0) / np.float64(35.16504)
    zs = np.sqrt(np.abs(salinity) * inv_s0)
    poly = (((((np.float64(1.46873e-3) * zs - np.float64(9.64972e-3)) * zs
               + np.float64(2.28348e-2)) * zs - np.float64(3.12775e-2)) * zs
             + np.float64(2.07679e-2)) * zs - np.float64(5.87701e-2))
    expected = poly * salinity
    np.testing.assert_array_equal(np.asarray(nemo_eos_fzp(salinity)), expected)


def test_rung36_stress_and_top_drag_use_nemo_face_association() -> None:
    utau = jnp.asarray([[0.0119902083]], dtype=jnp.float64)
    vtau = jnp.asarray([[0.00365089335]], dtype=jnp.float64)
    one_t = jnp.ones((1, 1), dtype=jnp.float64)
    one_u = jnp.ones((1, 2), dtype=jnp.float64)
    one_v = jnp.ones((2, 1), dtype=jnp.float64)
    got_u, got_v = nemo_stagger_surface_stress(
        utau, vtau, one_u, one_v, one_t, jnp.float64)
    np.testing.assert_array_equal(np.asarray(got_u), [[float(utau[0, 0])] * 2])
    np.testing.assert_array_equal(np.asarray(got_v), [[float(vtau[0, 0])]] * 2)

    raw = jnp.asarray([[-5.0e-5]], dtype=jnp.float64)
    top_u, top_v = nemo_top_drag_rate_faces(raw, jnp.float64)
    np.testing.assert_array_equal(np.asarray(top_u), [[5.0e-5, 5.0e-5]])
    np.testing.assert_array_equal(np.asarray(top_v), [[5.0e-5], [5.0e-5]])


def test_rung36_flux_form_substep_uses_transport_not_velocity_form() -> None:
    dtype = jnp.float64
    dt_e = dtype(5.7052297939778134)
    ssh_now = dtype(-1.6666666666666679)
    ssh_frc = dtype(5.04250075e-8)
    ssh_after = nemo_literal_ssh_forward(
        ssh_now, dt_e, ssh_frc, dtype(0.0), dtype(1.0))
    depth_zero = dtype(9.999999999999998)
    depth_now = depth_zero + ssh_now
    depth_after = depth_zero + ssh_after
    forcing = dtype(1.40236354e-6)
    got = nemo_flux_form_barotropic_velocity_update(
        dtype(0.0), dt_e, depth_now, depth_after, depth_after,
        depth_now, depth_after, dtype(0.0), dtype(0.0), forcing,
        dtype(1.0))
    expected = ((depth_now * dtype(0.0)
                 + dt_e * (depth_after * dtype(0.0)
                           + depth_after * dtype(0.0)
                           + depth_now * forcing))
                * (dtype(1.0) / depth_after))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    velocity_form = dt_e * forcing
    assert np.asarray(got).view(np.uint64) != np.asarray(velocity_form).view(np.uint64)


def test_rung36_flux_form_accepts_registered_nemo_inverse_depth() -> None:
    """The operand-frame replay must not recompute dynspg_ts z1_hu."""
    values = dict(
        velocity_now=jnp.float64(2.0e-5), dt_e=jnp.float64(5.7),
        depth_now=jnp.float64(8.3), depth_back=jnp.float64(8.4),
        depth_mid=jnp.float64(8.35), depth_forcing=jnp.float64(8.3),
        depth_after=jnp.float64(8.2), pressure_gradient=jnp.float64(0.0),
        trend=jnp.float64(2.0e-8), forcing=jnp.float64(1.0e-6),
        mask=jnp.float64(1.0),
    )
    derived = nemo_flux_form_barotropic_velocity_update(**values)
    registered = nemo_flux_form_barotropic_velocity_update(
        **values, inverse_depth_after=jnp.float64(1.0) / values["depth_after"])
    np.testing.assert_array_equal(np.asarray(registered), np.asarray(derived))


def test_c1d_slab_ocean_card_reuses_shared_rk3_stack() -> None:
    card = build_c1d_omip_l3_slab_ocean_card()
    cfg = card.recipe.model_config
    assert card.dt_s == 3600.0 and card.n_steps == 8760
    assert card.meridionally_periodic is True
    assert card.precision_policy == PrecisionPolicy.fp64(
        transcendentals="libm")
    assert cfg.momentum_advection == "off"
    assert cfg.vertical_momentum_scheme == "off"
    assert np.all(np.asarray(card.recipe.initial_state.v_mask.data) == 1.0)
    assert cfg.momentum_time_integrator == "rk3_ws"
    assert cfg.tracer_time_integrator == "rk3_ws"
    assert cfg.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3"
    assert cfg.barotropic.n_barotropic_substeps == 631
    assert cfg.barotropic_drag_substep is True
    assert cfg.zdf_drag_in_matrix is True
    assert cfg.zdf_baroclinic_only is True
    # Construction exercises the fail-closed RK3 selector: off/upwind3/off
    # is the single resolved C1D ln_dynadv_OFF program.
    LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)


def test_rung36_ocean_schema_rejects_false_header(tmp_path) -> None:
    path = tmp_path / "stagger.bin"
    with path.open("wb") as stream:
        stream.write(b"NEMO_L3STG__001 ")
        stream.write(struct.pack("=5i", 1, 1, 1, len(STAGGER_FIELDS) - 1, 64))
        stream.write(np.zeros(len(STAGGER_FIELDS), dtype=np.float64).tobytes())
    with pytest.raises(ValueError, match="untrue header"):
        _ocean_records(
            path, b"NEMO_L3STG__001 ", "5i", len(STAGGER_FIELDS))


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


def test_chlorophyll_replay_preserves_weight_and_time_association() -> None:
    source = np.arange(24, dtype=np.float32).reshape(12, 1, 2)
    monthly = _nemo_bilinear_months(
        source, [1, 2, 1, 2],
        [np.float64(0.0), np.float64(0.25),
         np.float64(0.0), np.float64(0.75)],
    )
    np.testing.assert_array_equal(monthly, source[:, 0, 1].astype(np.float64))
    hourly = _nemo_monthly_interp(monthly, 2)
    assert hourly.shape == (2,)
    assert hourly[1] != hourly[0]


def test_ssm_registry_rejects_positive_but_wrong_time_level(tmp_path) -> None:
    path = tmp_path / "ssm.bin"
    with path.open("wb") as stream:
        for stage in (0, 1):
            stream.write(b"NEMO_L3SSM__001 ")
            stream.write(struct.pack("=7i", 1, 1, 2, 2, stage, 15, 64))
            stream.write(np.zeros(15, dtype=np.float64).tobytes())
    with pytest.raises(ValueError, match="Kbb/Kmm"):
        _ssm(path)
