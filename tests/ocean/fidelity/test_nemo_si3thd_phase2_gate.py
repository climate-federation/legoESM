"""Direct tests for the selectable SI3 column and its NEMO phase-2 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.bitz_lipscomb import (
    SI3SurfaceForcing,
    _si3_zdf_bl99_step,
    ice_enthalpy_from_temperature,
    ice_temperature_from_enthalpy,
    option2_salinity_profile,
    snow_enthalpy_from_temperature,
)
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
from legoesm.ice.config import SI3ThermoConfig, validate_si3_thermo_config
from legoesm.ice.constants_config import (
    NEMO_SI3_CONSTANTS_CONFIG,
    IceConstantsConfig,
)
from legoesm.ice.scm import IceColumnModel

set_policy(PrecisionPolicy.fp64())


def _forcing() -> SI3SurfaceForcing:
    def a(value):
        return jnp.asarray([value], dtype=jnp.float64)

    return SI3SurfaceForcing(
        qns_ice=a(-442.5), qsr_ice=a(0.0), dqns_ice=a(-7.0),
        qtr_ice_top=a(0.0), t_bottom=a(271.27), sss=a(34.7),
        evaporation=a(0.0), snow_precipitation=a(5.0e-6),
        qprec_ice=a(0.0), qcn_ice_bottom=a(0.0),
        qsb_ice_bottom=a(0.0), fhld=a(0.0), qlead=a(0.0),
    )


def _zdf_inputs():
    c = NEMO_SI3_CONSTANTS_CONFIG
    bulk = jnp.asarray([6.3], dtype=jnp.float64)
    sal = option2_salinity_profile(bulk, jnp.asarray([34.7], dtype=jnp.float64))
    ti = jnp.asarray([[257.0, 261.0, 266.0]], dtype=jnp.float64)
    ts = jnp.asarray([[255.0, 260.0, 266.0]], dtype=jnp.float64)
    return (
        ice_enthalpy_from_temperature(ti, sal, c),
        snow_enthalpy_from_temperature(ts, c),
        sal,
        jnp.asarray([2.0], dtype=jnp.float64),
        jnp.asarray([0.2], dtype=jnp.float64),
        jnp.asarray([270.0], dtype=jnp.float64),
    )


def test_nemo_constants_and_enthalpy_roundtrip_are_fp64() -> None:
    c = NEMO_SI3_CONSTANTS_CONFIG
    sal = jnp.asarray([[3.15, 6.3, 9.45]], dtype=jnp.float64)
    temp = jnp.asarray([[258.0, 263.0, 269.0]], dtype=jnp.float64)
    energy = ice_enthalpy_from_temperature(temp, sal, c)
    recovered = ice_temperature_from_enthalpy(energy, sal, c)
    assert energy.dtype == recovered.dtype == jnp.float64
    np.testing.assert_allclose(recovered, temp, rtol=0.0, atol=2.0e-13)
    assert (c.c_ice, c.latent_fusion, c.k_ice, c.k_snow, c.lead_albedo) == (
        2096.7, 333360.1, 2.034396, 0.5, 0.066,
    )


def test_selector_rejects_frankenstein_identity() -> None:
    card = build_c1d_omip_l3_card()
    validate_si3_thermo_config(card.config)
    with pytest.raises(ValueError, match="ORCA1-resolved"):
        validate_si3_thermo_config(
            card.config._replace(si3=SI3ThermoConfig(n_snow_layers=2))
        )
    with pytest.raises(ValueError, match="mixing canonical"):
        validate_si3_thermo_config(
            card.config._replace(ice_constants=IceConstantsConfig())
        )


def test_zdf_is_eager_jit_and_reverse_mode_safe() -> None:
    c = NEMO_SI3_CONSTANTS_CONFIG
    e_i, e_s, sal, h_i, h_s, t_su = _zdf_inputs()
    forcing = _forcing()

    def run(qns):
        return _si3_zdf_bl99_step(
            e_i, e_s, sal, h_i, h_s, t_su,
            forcing._replace(qns_ice=qns), 3600.0, c,
        ).T_surface

    eager = run(forcing.qns_ice)
    compiled = jax.jit(run)(forcing.qns_ice)
    gradient = jax.grad(lambda q: jnp.sum(run(jnp.asarray([q]))))(
        forcing.qns_ice[0]
    )
    np.testing.assert_array_equal(np.asarray(compiled), np.asarray(eager))
    assert eager.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(eager)))
    assert bool(jnp.isfinite(gradient))


def test_existing_column_driver_dispatches_selected_si3() -> None:
    card = build_c1d_omip_l3_card()
    model = IceColumnModel.create(
        forcing=_forcing(), config=card.config, ncol=1, dt=card.dt_seconds,
        h_ice_init=2.0, h_snow_init=0.2, concentration_init=0.9,
        T_ice_init=260.0, S_ice_init=6.3,
    )
    response = model.step()
    assert model.state.e_ice.data.shape == (1, 3)
    assert model.state.e_snow.data.shape == (1, 3)
    assert model.state.e_ice.data.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(model.state.e_ice.data)))
    assert response.T_sfc.dtype == jnp.float64


@pytest.fixture(scope="module")
def gate_module():
    path = Path("scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py")
    spec = importlib.util.spec_from_file_location("nemo_si3thd_phase2_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_finds_preregistered_first_divergence(gate_module) -> None:
    result = gate_module.run()
    assert result["status"] == "DEBT"
    assert result["backend"] == "cpu"
    assert result["dtypes"] == ["float64"]
    assert result["sweep_steps_examined"] == 1
    assert result["preregistered_hypothesis"] == "CONFIRMED"
    assert result["first_divergence"]["name"] == "kt1.POST_ZDF.t_su"
    assert result["first_divergence"]["normalized_max_abs"] > 1.0e-15
    assert all(
        row["status"] == "AT-BAR"
        for row in result["rows"]
        if row["name"].startswith("geometry_ic.")
        or row["name"].startswith("kt1.ENTRY.")
    )


def test_gate_binary_cursor_advances_to_second_step(gate_module) -> None:
    """The kt=1 stop does not hide a broken later-step stream cursor."""

    card = build_c1d_omip_l3_card()
    with (
        (card.oracle_root / "oracle_si3_thd_frames.bin").open("rb") as thd,
        (card.oracle_root / "oracle_si3_exchange_frames.bin").open("rb") as xchg,
    ):
        for kt in (1, 2):
            frames = gate_module._read_thd_step(thd, kt)
            exchange = gate_module._read_exchange_step(xchg, kt)
            assert len(frames) == 8
            assert exchange["qns_ice"].dtype == np.float64


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"plant_geometry": True}, "geometry/IC gate"),
        ({"plant_stage": True}, "ENTRY gate"),
        ({"plant_selector": True}, "ORCA1-resolved"),
    ],
)
def test_gate_plants_exit_red(gate_module, kwargs, error) -> None:
    with pytest.raises((gate_module.GateError, ValueError), match=error):
        gate_module.run(**kwargs)
