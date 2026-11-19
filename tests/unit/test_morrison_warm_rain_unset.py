"""Unset Morrison warm-rain law resolves by flavour (mg -> kk2000_cam6,
sam -> kk2000) at the driver and the scheme, and every committed deck that
runs Morrison names its law."""
from pathlib import Path

import pytest
import yaml

from legoesm.atmosphere.physics.microphysics import integration as mi
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig, MorrisonConfig, resolve_warm_rain_scheme)
from legoesm.atmosphere.physics.microphysics.morrison import (
    resolve_morrison_flavor)
from legoesm.driver.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("flavor, law", [("mg", "kk2000_cam6"),
                                         ("sam", "kk2000")])
def test_unset_law_resolves_by_flavour(flavor, law):
    cfg = MorrisonConfig(morrison_flavor=flavor)
    assert cfg.warm_rain_scheme is None
    assert resolve_warm_rain_scheme(cfg) == law
    assert resolve_morrison_flavor(cfg).warm_rain_scheme == law
    explicit = cfg._replace(warm_rain_scheme="seifert_beheng_sb2001")
    assert resolve_morrison_flavor(explicit).warm_rain_scheme == (
        "seifert_beheng_sb2001")


def test_unset_law_with_unknown_flavour_raises():
    with pytest.raises(ValueError, match="morrison_flavor"):
        resolve_warm_rain_scheme(MorrisonConfig(morrison_flavor="bogus"))


def test_driver_default_is_unset_and_explicit_needs_morrison():
    assert ExperimentConfig._field_defaults["morrison_warm_rain_scheme"] is None
    ExperimentConfig(microphysics="kessler").validate_strict()
    with pytest.raises(ValueError, match="requires microphysics='morrison'"):
        ExperimentConfig(microphysics="kessler",
                         morrison_warm_rain_scheme="kk2000").validate_strict()


@pytest.mark.parametrize("flavor", ["mg", "sam"])
def test_driver_kk_factors_accept_the_resolved_unset_law(flavor):
    ExperimentConfig(microphysics="morrison", morrison_flavor=flavor,
                     morrison_autocon_fact=0.5).validate_strict()


def test_guard_runs_before_the_lane_factory_and_reads_the_resolved_law(
        monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("lane factory reached")
    monkeypatch.setattr(mi, "_make_plane_microphysics", boom)
    with pytest.raises(ValueError, match="relvar"):
        mi.make_microphysics_physics(
            MicrophysicsConfig(scheme="morrison"), "plane", 10.0)
    sam = MicrophysicsConfig(scheme="morrison",
                             morrison=MorrisonConfig(morrison_flavor="sam"))
    with pytest.raises(RuntimeError, match="lane factory reached"):
        mi.make_microphysics_physics(sam, "plane", 10.0)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("flavor, wired", [("mg", True), ("sam", False)])
def test_make_physics_wires_clubb_relvar_for_the_resolved_law(
        monkeypatch, flavor, wired):
    from legoesm.atmosphere.physics import combined
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    seen = {}

    def spy(*a, relvar_from_clubb=False, **k):
        seen["relvar"] = relvar_from_clubb
        raise _Stop
    monkeypatch.setattr(combined, "make_microphysics_physics", spy)
    cfg = combined.PhysicsConfig(
        radiation=RadiationConfig(scheme="none",
                                  cloud_config=CloudConfig(scheme="cam6_clubb")),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", liquid_partition=True,
                                    clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(
            scheme="morrison", morrison=MorrisonConfig(morrison_flavor=flavor)))
    with pytest.raises(_Stop):
        combined.make_physics(cfg, model_type="mpas", dt=600.0,
                              cld_macmic_num_steps=2)
    assert seen["relvar"] is wired


def test_aimip_classical_factory_pins_kk2000_for_morrison(monkeypatch):
    from legoesm.atmosphere.physics import combined
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics)
    seen = {}

    def spy(physics_config, *a, **k):
        seen["law"] = physics_config.microphysics.morrison.warm_rain_scheme
        raise _Stop
    monkeypatch.setattr(combined, "make_physics", spy)
    with pytest.raises(_Stop):
        make_aimip_classical_spectral_physics(
            AIMIPClassicalParams.from_defaults(),
            create_gaussian_grid(11, dealiasing="quadratic"), dt=1800.0,
            radiation="gray", microphysics_scheme="morrison")
    assert seen["law"] == "kk2000"


def _scan_configs():
    """(deck laws, AIMIP-classical files selecting Morrison) over every
    committed YAML; ``include:`` bases are merged as the run loaders do."""
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    decks, aimip = {}, {}
    for p in sorted((ROOT / "config").rglob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        if not isinstance(d, dict):
            continue
        if "include" in d:
            d = read_yaml_with_includes(str(p))
        rel = p.relative_to(ROOT).as_posix()
        cl = d.get("classical") if isinstance(d.get("classical"), dict) else {}
        if d.get("microphysics") == "morrison":
            decks[rel] = (d.get("morrison_warm_rain_scheme"),
                          bool(d.get("clubb_liquid_partition")))
        elif cl.get("microphysics") == "morrison":
            pin = ((cl.get("param_fixed") or {})
                   .get("atm.micro.MorrisonConfig") or {})
            decks[rel] = (pin.get("warm_rain_scheme"),
                          bool(cl.get("clubb_liquid_partition")))
        for a in (d, d.get("cfg_overrides") or {}):
            if isinstance(a, dict) and a.get("aimip_microphysics") == "morrison":
                pin = ((a.get("param_fixed") or {})
                       .get("atm.micro.MorrisonConfig") or {})
                aimip[rel] = pin.get("warm_rain_scheme")
    return decks, aimip


def test_every_morrison_deck_names_its_warm_rain_law():
    decks, _ = _scan_configs()
    assert "config/amip/amip_production.yaml" in decks
    unnamed = sorted(k for k, (law, _) in decks.items() if law is None)
    assert not unnamed, unnamed
    # the CAM6 law only where make_physics can wire CLUBB's relvar
    cam6 = {k for k, (law, part) in decks.items() if law == "kk2000_cam6"}
    assert cam6 and all(decks[k][1] for k in cam6), sorted(cam6)
    assert {law for law, _ in decks.values()} == {"kk2000", "kk2000_cam6"}


def test_aimip_sweep_configs_rely_on_the_classical_factory_pin():
    """These select Morrison through the AIMIP/WB classical factory, which
    pins kk2000 (test above); none overrides it."""
    _, aimip = _scan_configs()
    assert aimip == {
        "config/aimip/sweep/stage1/combo_micro_morrison/variant_classical.yaml": None,
        "config/aimip/wbcompare/classical_sweep/sbm_ysu/suite.yaml": None,
        "config/aimip/wbcompare/classical_sweep/sbm_ysu/variant_classical.yaml": None,
        "config/wb/sweep/stage1/combo_micro_morrison/variant_classical.yaml": None,
    }
