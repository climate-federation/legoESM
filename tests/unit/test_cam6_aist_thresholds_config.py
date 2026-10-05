"""CAM6 aist ramp thresholds (cldfrc2m rhmini/rhmaxi/rhminis/rhmaxis) as run
settings: CLI -> ExperimentConfig -> the MPAS lane's built CloudConfig, which
feeds radiation cloud cover and the in-cloud warm-rain ast."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_NAMES = ("rhmini", "rhmaxi", "rhminis", "rhmaxis")
_DECK = (pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
         / "amip_production.yaml")


def _cam6(**kw):
    """A minimal valid cam6_clubb ExperimentConfig (MPAS lane)."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=8),
        dycore=DycoreConfig(discretization="mpas"), microphysics="morrison",
        turbulence="clubb", radiation="rrtmgp", cloud_scheme="cam6_clubb",
        use_clubb_cloud_fraction=True, **kw)


def _args(argv):
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    p = build_arg_parser()
    return build_config_from_args(_postprocess_args(
        p.parse_args(["--dataset", "analytical", *argv]), p))


def test_cli_round_trip_and_default_none():
    cfg = _args([])
    assert all(getattr(cfg, f"cloud_cam6_{n}") is None for n in _NAMES)
    cfg = _args(["--cloud-cam6-rhmini", "0.85", "--cloud-cam6-rhmaxi", "1.05",
                 "--cloud-cam6-rhminis", "0.9", "--cloud-cam6-rhmaxis", "1.1"])
    assert (cfg.cloud_cam6_rhmini, cfg.cloud_cam6_rhmaxi, cfg.cloud_cam6_rhminis,
            cfg.cloud_cam6_rhmaxis) == (0.85, 1.05, 0.9, 1.1)


@pytest.mark.parametrize("name,bad", [("rhmini", 0.995), ("rhmaxi", 0.99),
                                      ("rhminis", 1.01), ("rhmaxis", 1.2)])
def test_validate_strict_one_range(name, bad):
    with pytest.raises(ValueError, match=f"cloud_cam6_{name}.*out of range"):
        _cam6(**{f"cloud_cam6_{name}": bad}).validate_strict()


def test_rhmini_range_is_the_param_spec_range():
    from legoesm.atmosphere.physics.clouds import config as cc
    spec = cc.__param_spec__["CloudConfig"]["params"]["cam6_rhmini"]["bounds"]
    assert tuple(spec) == (0.5, 0.99)


def test_cli_value_reaches_the_mpas_cloud_config_and_moves_aist():
    """Non-vacuous: the MPAS lane's built CloudConfig carries the CLI value,
    and aist on an rhi = 0.9 cell responds to it."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        cam6_ice_stratus_fraction,
    )
    from legoesm.driver.model_driver import _standalone_cloud_config
    cc = _standalone_cloud_config(_args(["--cloud-cam6-rhmini", "0.85"]),
                                  "cam6_clubb", allow_convective_cloud=True)
    assert cc.cam6_rhmini == 0.85
    base = _standalone_cloud_config(_args([]), "cam6_clubb",
                                    allow_convective_cloud=True)
    from legoesm.thermo import saturation_mixing_ratio
    # near 0 C esl ~ esi, so rhi ~ 0.93: inside both ramps, off the 0.999 cap
    T, p = jnp.full((1, 1), 270.0), jnp.full((1, 1), 5.0e4)
    qs = saturation_mixing_ratio(T, p)
    args = (0.9 * qs, T, p, jnp.full((1, 1), 1e-5), jnp.zeros(1))
    ptop = jnp.full((1, 1), 4.9e4)          # tropospheric at lat 0
    a0 = float(cam6_ice_stratus_fraction(*args, base, ptop)[0, 0])
    a1 = float(cam6_ice_stratus_fraction(*args, cc, ptop)[0, 0])
    assert 0.0 < a1 < a0 < 0.999


def test_combined_hands_the_radiation_cloud_config_to_microphysics(monkeypatch):
    """The MPAS physics step's microphysics gets config.radiation.cloud_config
    (the object _standalone_cloud_config built), not a default."""
    from legoesm.atmosphere.physics import combined
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    seen = {}
    real = combined.make_microphysics_physics

    def spy(*a, cloud_config=None, **k):
        seen["cc"] = cloud_config
        return real(*a, cloud_config=cloud_config, **k)
    monkeypatch.setattr(combined, "make_microphysics_physics", spy)
    cc = CloudConfig(scheme="cam6_clubb", cam6_rhmini=0.85)
    combined.make_physics(combined.PhysicsConfig(
        radiation=RadiationConfig(scheme="none", cloud_config=cc),
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(scheme="kessler")),
        model_type="mpas", dt=600.0)
    assert seen["cc"] is cc


def test_production_deck_records_the_code_defaults_zero_physics_change():
    from scripts.run.run_amip import build_arg_parser

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import _standalone_cloud_config
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(_DECK, build_arg_parser())
    vals = {f"cloud_cam6_{n}": keys[f"cloud_cam6_{n}"] for n in _NAMES}
    assert vals == {"cloud_cam6_rhmini": 0.80, "cloud_cam6_rhmaxi": 1.0,
                    "cloud_cam6_rhminis": 1.0, "cloud_cam6_rhmaxis": 1.0}
    _cam6(**vals).validate_strict()
    with_deck = _standalone_cloud_config(ExperimentConfig(**vals), "cam6_clubb",
                                         allow_convective_cloud=True)
    unset = _standalone_cloud_config(ExperimentConfig(), "cam6_clubb",
                                     allow_convective_cloud=True)
    assert with_deck == unset
    assert np.isclose(with_deck.cam6_rhmini, 0.80)


def test_all_four_reach_the_built_config_and_legal_bounds_pass():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import _standalone_cloud_config
    cc = _standalone_cloud_config(_args([
        "--cloud-cam6-rhmini", "0.7", "--cloud-cam6-rhmaxi", "1.1",
        "--cloud-cam6-rhminis", "0.85", "--cloud-cam6-rhmaxis", "1.05"]),
        "cam6_clubb", allow_convective_cloud=True)
    assert (cc.cam6_rhmini, cc.cam6_rhmaxi, cc.cam6_rhminis,
            cc.cam6_rhmaxis) == (0.7, 1.1, 0.85, 1.05)
    # inclusive bounds: CAM6 non-CLUBB namelist values must be legal
    for name, v in (("rhmini", 0.5), ("rhmini", 0.99), ("rhmaxi", 1.1),
                    ("rhminis", 0.85), ("rhmaxis", 1.1)):
        _cam6(**{f"cloud_cam6_{name}": v}).validate_strict()
    ExperimentConfig().validate_strict()          # unset: no cam6 needed


@pytest.mark.parametrize("name", _NAMES)
def test_threshold_refused_unless_cam6_clubb(name):
    """Any of the four on a non-cam6 cloud scheme would be silently inert:
    refused at validation (user decision 2026-10-01)."""
    from legoesm.driver.config import ExperimentConfig
    for scheme in ("none", "sundqvist"):
        with pytest.raises(ValueError, match="read only by"):
            ExperimentConfig(cloud_scheme=scheme,
                             **{f"cloud_cam6_{name}": 0.9 if name == "rhminis"
                                else (0.8 if name == "rhmini" else 1.0)}
                             ).validate_strict()
