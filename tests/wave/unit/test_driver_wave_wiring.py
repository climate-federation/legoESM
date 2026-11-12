"""Driver-side wave wiring: config defaults, refusal paths, CLI round-trip."""

from __future__ import annotations

import pytest

from legoesm.driver.coupled_config import CoupledConfig
from legoesm.driver.coupled_esm_driver import CoupledESMDriver
from legoesm.wave.config import WaveConfig, WaveTimeConfig
from tests.wave.unit.conftest import tiny_grid, tiny_network


class _Stub:
    def __init__(self, config, *, dynamic=False):
        self.coupled_cfg = config
        self._is_dynamic_ocean = dynamic


def _wave_config():
    return WaveConfig(
        grid=tiny_grid(), network=tiny_network(),
        time=WaveTimeConfig(dt_gse_s=0),
    )


def test_wave_defaults_are_off():
    config = CoupledConfig()
    assert config.wave_mode == "off"
    assert config.wave_config is None
    assert config.wave_weights_path == ""
    assert config.kpp_langmuir is False


def test_coupled_config_validate_strict_rejects_unknown_wave_mode():
    with pytest.raises(ValueError, match="wave_mode"):
        CoupledConfig(wave_mode="sometimes").validate_strict()
    CoupledConfig().validate_strict()
    CoupledConfig(wave_mode="active").validate_strict()


def test_unknown_wave_mode_raises():
    with pytest.raises(ValueError, match="wave_mode"):
        CoupledESMDriver._init_wave(_Stub(CoupledConfig(wave_mode="sometimes")))


def test_active_requires_config():
    with pytest.raises(ValueError, match="wave_config"):
        CoupledESMDriver._init_wave(_Stub(CoupledConfig(wave_mode="active")))


def test_active_requires_weights():
    config = CoupledConfig(wave_mode="active", wave_config=_wave_config())
    with pytest.raises(ValueError, match="wave_weights_path"):
        CoupledESMDriver._init_wave(_Stub(config))


def test_active_requires_dynamic_ocean():
    config = CoupledConfig(
        wave_mode="active", wave_config=_wave_config(),
        wave_weights_path="/tmp/does-not-matter.npz")
    with pytest.raises(ValueError, match="ocean_mode='dynamic'"):
        CoupledESMDriver._init_wave(_Stub(config, dynamic=False))


def test_cli_flags_round_trip():
    from scripts.run.run_coupled import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.wave_mode == "off"
    assert args.wave_config == ""
    assert args.wave_weights == ""
    assert args.kpp_langmuir is False
    args = parser.parse_args([
        "--wave-mode", "active",
        "--wave-config", "deck.json",
        "--wave-weights", "weights.npz",
        "--kpp-langmuir",
    ])
    assert args.wave_mode == "active"
    assert args.wave_config == "deck.json"
    assert args.wave_weights == "weights.npz"
    assert args.kpp_langmuir is True


def test_step_wave_adds_batch_axis_and_returns_ocean_shaped_stokes(
        tiny_config, tiny_state):
    """The coupling seam must feed the model (1, lon, lat) and stage (lat, lon).

    Regression: the first coupled smoke passed the remapped wind as 2-D
    (lon, lat) and the wave model rejected it.
    """
    import types

    import jax.numpy as jnp

    from legoesm.wave.model import prepare_env

    seen = {}

    class _FakeModel:
        def advance(self, state, env, u10, v10):
            seen["shapes"] = (state.spectra.shape, u10.shape, v10.shape)
            return state

    class _Stub:
        coupled_cfg = CoupledConfig(wave_mode="active")
        _wave_model = _FakeModel()
        _wave_state = tiny_state
        _wave_env = prepare_env(tiny_config, tiny_state.depth)
        _wave_atm_remap = types.SimpleNamespace(a2o=None)
        _wave_ocean_remap = types.SimpleNamespace(o2a=None)
        _wave_stokes = None
        _wave_diag = None
        _wave_diag_logged = False

    # Surface fields are (lat, lon); the remap is identity in this stub.
    n_lon, n_lat = tiny_state.depth.shape[1:]  # wave state order is (lon, lat)
    sfc = types.SimpleNamespace(
        tau_x=jnp.full((n_lat, n_lon), 0.1),
        tau_y=jnp.zeros((n_lat, n_lon)),
        z0=jnp.full((n_lat, n_lon), 1e-4),
    )
    atm = types.SimpleNamespace(rho_lowest=jnp.full((n_lat, n_lon), 1.2))

    stub = _Stub()
    CoupledESMDriver._step_wave(stub, sfc, atm)

    spectra_shape, u_shape, v_shape = seen["shapes"]
    assert u_shape == (1, n_lon, n_lat)
    assert v_shape == (1, n_lon, n_lat)
    assert stub._wave_stokes.shape == (n_lat, n_lon)

    # Exercise the same seam with the real integrator as well: the fake model
    # above checks shapes but cannot catch dtype or JAX scan failures.
    import jax
    from legoesm.wave.model import WaveModel

    stub._wave_model = WaveModel(tiny_config, key=jax.random.key(0))
    stub._wave_state = tiny_state
    CoupledESMDriver._step_wave(stub, sfc, atm)
    assert stub._wave_state.spectra.dtype == tiny_state.spectra.dtype
    assert bool(jnp.all(jnp.isfinite(stub._wave_state.spectra)))
    assert bool(jnp.any(stub._wave_state.spectra != tiny_state.spectra))
    assert bool(jnp.all(jnp.isfinite(stub._wave_stokes)))


def _kpp_ocean_config(*, enable_langmuir=False):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        KPPConfig, VerticalMixingConfig)
    from legoesm.ocean.state import LatLonCGridOceanConfig

    base = LatLonCGridOceanConfig.from_flat()
    return base._replace(physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp",
            kpp=KPPConfig(enable_langmuir=enable_langmuir))))


def test_kpp_langmuir_switch_enables_it_on_an_existing_kpp_stack():
    class _Stub:
        coupled_cfg = CoupledConfig(
            kpp_langmuir=True, ocean_config=_kpp_ocean_config())

    config = CoupledESMDriver._build_latlon_ocean_config(_Stub())
    assert config.physics is not None
    assert config.physics.vertical_mixing.scheme == "kpp"
    assert config.physics.vertical_mixing.kpp.enable_langmuir is True


def test_kpp_langmuir_refuses_a_card_without_a_physics_stack():
    """No KPP stack -> hard error; never invent K_conv values here."""

    class _Stub:
        coupled_cfg = CoupledConfig(kpp_langmuir=True)

    with pytest.raises(ValueError, match="existing KPP"):
        CoupledESMDriver._build_latlon_ocean_config(_Stub())


def test_kpp_langmuir_refuses_a_non_kpp_closure():
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig)
    from legoesm.ocean.state import LatLonCGridOceanConfig

    card = LatLonCGridOceanConfig.from_flat()._replace(
        physics=OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="constant")))

    class _Stub:
        coupled_cfg = CoupledConfig(kpp_langmuir=True, ocean_config=card)

    with pytest.raises(ValueError, match="scheme='kpp'"):
        CoupledESMDriver._build_latlon_ocean_config(_Stub())


def test_kpp_langmuir_tripole_builder_enables_it():
    """The tripole card already carries KPP; the flag must reach it."""

    class _Stub:
        coupled_cfg = CoupledConfig(kpp_langmuir=True)

    config = CoupledESMDriver._build_tripole_ocean_config(_Stub())
    assert config.physics.vertical_mixing.scheme == "kpp"
    assert config.physics.vertical_mixing.kpp.enable_langmuir is True


def test_step_wave_raises_when_active_without_a_model():
    import types
    stub = types.SimpleNamespace(
        coupled_cfg=CoupledConfig(wave_mode="active"), _wave_model=None)
    with pytest.raises(RuntimeError, match="not initialised"):
        CoupledESMDriver._step_wave(stub, object(), object())


def test_kpp_langmuir_off_keeps_the_legacy_stack():
    class _Stub:
        coupled_cfg = CoupledConfig()

    config = CoupledESMDriver._build_latlon_ocean_config(_Stub())
    assert config.physics is None


def test_wave_clock_refuses_partial_coupling_before_surface_advance():
    import types
    config = _wave_config()._replace(
        time=_wave_config().time._replace(
            dt_global_s=600, dt_stot_s=600))
    stub = types.SimpleNamespace(
        _wave_model=object(),
        coupled_cfg=CoupledConfig(coupling_dt=600., wave_config=config))
    with pytest.raises(ValueError, match="clocks"):
        CoupledESMDriver._segment_hook(stub, None, 0., 300.)
