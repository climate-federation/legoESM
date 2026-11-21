"""The two one-variable A/B decks for the ported MPAS options: each differs
from amip_production.yaml in its own rows only, reaches the dycore object the
run steps, and adds no validation error on the production lane."""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config, read_yaml_with_includes

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_DECKS = {
    "amip_production_ppm.yaml": {"mpas_vert_advection_scheme": "ppm",
                                 "mpas_vert_ppm_limiter_sharpness": 1.0e4},
    "amip_production_vert4_tracer.yaml": {"mpas_nu_vert4_tracer": 0.0},
}


def _cfg(deck):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


def _errors(cfg):
    try:
        cfg.validate_strict()
    except ValueError as exc:
        return set(str(exc).splitlines())
    return set()


@pytest.mark.parametrize("name,rows", _DECKS.items())
def test_deck_differs_from_production_only_in_its_rows(name, rows):
    prod, deck = read_yaml_with_includes(_PROD), read_yaml_with_includes(_AMIP / name)
    diff = {k for k in prod.keys() | deck.keys() if prod.get(k) != deck.get(k)}
    assert diff == set(rows)
    assert all(deck[k] == v for k, v in rows.items())


@pytest.mark.parametrize("name,rows", _DECKS.items())
def test_deck_reaches_the_dycore_and_validates(name, rows):
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.grids.voronoi import create_voronoi_mesh
    prod, cfg = _cfg(_PROD), _cfg(_AMIP / name)
    assert _errors(cfg) <= _errors(prod)
    model = create_atmosphere_dycore(cfg, create_voronoi_mesh(2), make_cam6_l32_levels())
    if "mpas_vert_advection_scheme" in rows:
        assert model.config.vert_advection_scheme == "ppm"
        assert model.config.vert_ppm_limiter_sharpness == 1.0e4
    else:
        assert model.config.nu_vert4_tracer == 0.0
        assert model.config.nu_vert4_T == cfg.dycore.mpas_nu_vert4_T > 0.0
