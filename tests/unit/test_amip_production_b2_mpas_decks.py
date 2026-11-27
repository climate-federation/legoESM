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
    "amip_production_ppm.yaml": {"vertical_form": "legacy",
                                 "mpas_vert_advection_scheme": "ppm",
                                 "mpas_vert_ppm_limiter_sharpness": 1.0e4,
                                 "tracer_vertical_scheme": None},
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
        assert model.config.vertical_form == "legacy"
        assert model.config.vert_advection_scheme == "ppm"
        assert model.config.vert_ppm_limiter_sharpness == 1.0e4
    else:
        assert model.config.nu_vert4_tracer == 0.0
        assert model.config.nu_vert4_T == cfg.dycore.mpas_nu_vert4_T > 0.0


def test_ppm_deck_steps_and_ppm_on_as83_is_refused():
    """The PPM arm takes one MPAS step through the factory-built dycore; ppm
    with production's shared-core vertical form fails validate_strict."""
    import jax.numpy as jnp
    import numpy as np

    from legoesm.core.field import Field
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.unit.test_mpas_atmosphere import (
        _add_perturbation_hydro, _make_hydrostatic_state,
    )
    cfg = _cfg(_AMIP / "amip_production_ppm.yaml")
    mesh, coord = create_voronoi_mesh(2), make_cam6_l32_levels()
    model = create_atmosphere_dycore(cfg, mesh, coord)
    nlev = coord.A_full.shape[0]
    state = _add_perturbation_hydro(_make_hydrostatic_state(mesh, nlev), mesh, nlev)
    qv = jnp.asarray(np.linspace(1e-6, 1.5e-2, nlev)[None, :].repeat(mesh.nCells, 0))
    state = state._replace(tracers={"q_v": Field(
        data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")})
    out = model.step(state, 300.0)
    assert np.isfinite(np.asarray(out.T.data)).all()
    assert np.isfinite(np.asarray(out.tracers["q_v"].data)).all()
    bad = cfg._replace(dycore=cfg.dycore._replace(vertical_form="as83"))
    assert any("wired into the MPAS legacy vertical form only" in e for e in _errors(bad))
