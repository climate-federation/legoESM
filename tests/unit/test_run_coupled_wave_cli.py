"""CLI surface of the optional coupled wave component (run_coupled)."""

from __future__ import annotations

import pytest

import scripts.run.run_coupled as run_coupled
from legoesm.driver.coupled_config import CoupledConfig


def test_wave_flags_default_off():
    args = run_coupled.build_parser().parse_args([])
    assert args.wave_mode == "off"
    assert args.wave_config == ""
    assert args.wave_weights == ""
    assert args.kpp_langmuir is False


def test_wave_flags_round_trip():
    args = run_coupled.build_parser().parse_args([
        "--wave-mode", "active",
        "--wave-config", "deck.json",
        "--wave-weights", "weights.npz",
        "--kpp-langmuir",
    ])
    assert args.wave_mode == "active"
    assert args.wave_config == "deck.json"
    assert args.wave_weights == "weights.npz"
    assert args.kpp_langmuir is True


def test_unknown_wave_mode_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        run_coupled.build_parser().parse_args(["--wave-mode", "sometimes"])


def test_kpp_langmuir_requires_a_dynamic_ocean(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_coupled.main(["--kpp-langmuir", "--ocean", "slab"])
    assert excinfo.value.code == 2
    assert "--ocean dynamic" in capsys.readouterr().err


def test_active_wave_requires_config(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_coupled.main([
            "--wave-mode", "active", "--ocean", "dynamic",
            "--ocean-grid", "latlon:8",
        ])
    assert excinfo.value.code == 2
    assert "requires --wave-config" in capsys.readouterr().err


@pytest.mark.parametrize(("flags", "message"), [
    (["--atm-owns-surface"], "requires the coupler-owned surface exchange"),
    (["--grid", "voronoi"], "does not support Voronoi-to-wave remapping"),
])
def test_active_wave_refuses_unsupported_coupling(flags, message, capsys):
    with pytest.raises(SystemExit) as exc:
        run_coupled.main(["--wave-mode", "active", *flags])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_coupled_config_defaults_keep_waves_off():
    config = CoupledConfig()
    assert config.wave_mode == "off"
    assert config.wave_config is None
    assert config.wave_weights_path == ""
    assert config.kpp_langmuir is False


def test_wave_off_cli_preserves_prescribed_langmuir(tmp_path):
    import jax.numpy as jnp

    from legoesm.ocean.physics.vertical_mixing.kpp import _kpp_langmuir_enhancement

    args = [
        "--grid", "latlon", "--ocean", "dynamic", "--resolution", "8",
        "--output", str(tmp_path),
    ]
    baseline = run_coupled.main(args, build_only=True)
    driver = run_coupled.main([*args, "--kpp-langmuir"], build_only=True)
    baseline_card = baseline._build_latlon_ocean_config()
    card = driver._build_latlon_ocean_config()

    assert baseline_card.physics is None
    assert card._replace(physics=baseline_card.physics) == baseline_card
    assert card.physics.vertical_mixing.scheme == "kpp"
    assert card.physics.convection.scheme == "enhanced_diffusion"
    assert driver.coupled_cfg.wave_mode == "off"
    driver._init_wave()
    assert driver._wave_stokes is None
    kpp = card.physics.vertical_mixing.kpp
    assert kpp.enable_langmuir
    u_star = jnp.array(0.01)
    assert float(_kpp_langmuir_enhancement(u_star, driver._wave_stokes, kpp)) == (
        pytest.approx(1.51703464, rel=1e-6))
    assert float(_kpp_langmuir_enhancement(u_star, jnp.zeros_like(u_star), kpp)) == 1


@pytest.mark.parametrize("neural", [False, True])
def test_wave_weights_required_only_for_neural_config(tmp_path, capsys, neural):
    import json
    from tests.wave.unit.test_wave_coupling import _global_wave_config

    config = _global_wave_config(lon_step=45.0, lat_step=45.0)
    if not neural:
        config = config._replace(network=config.network._replace(
            parameterization="none"))
    path = tmp_path / "wave.json"
    path.write_text(json.dumps({
        name: value._asdict() for name, value in config._asdict().items()}))
    args = ["--wave-mode", "active", "--wave-config", str(path),
            "--ocean", "dynamic", "--ocean-grid", "latlon:8"]
    if neural:
        with pytest.raises(SystemExit) as exc:
            run_coupled.main(args, build_only=True)
        assert exc.value.code == 2
        assert "--wave-weights is required" in capsys.readouterr().err
    else:
        driver = run_coupled.main(args, build_only=True)
        assert driver.coupled_cfg.wave_weights_path == ""
        assert driver.coupled_cfg.wave_mode == "active"

        assert driver._build_latlon_ocean_config().physics is None
        enabled = run_coupled.main([*args, "--kpp-langmuir"], build_only=True)
        physics = enabled._build_latlon_ocean_config().physics
        assert physics.vertical_mixing.scheme == "kpp"
        assert physics.vertical_mixing.kpp.enable_langmuir is True
        assert physics.surface_forcing.scheme == "none"
