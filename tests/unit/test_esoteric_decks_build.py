"""Every E&M2009 deck in config/esoteric/ builds through run_coupled."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rc = _load("_run_coupled_esoteric", "scripts/run/run_coupled.py")
mg = _load("_make_geography_esoteric", "scripts/data/make_geography.py")
DECKS = sorted((REPO / "config" / "esoteric").glob("em2009_*.yaml"))
CASES = [p for p in DECKS if p.name != "em2009_base.yaml"]


def test_four_cases():
    assert [p.stem for p in CASES] == ["em2009_aquaplanet", "em2009_drake",
                                       "em2009_eqpas", "em2009_ridge"]


@pytest.mark.parametrize("deck", CASES, ids=lambda p: p.stem)
def test_deck_builds(deck, tmp_path):
    shape = deck.stem.split("_")[1]
    geo = str(tmp_path / f"{shape}.nc")
    mg.idealized(shape, barrier_lon=None if shape == "aquaplanet" else 0.0).to_netcdf(geo)
    d = rc.main(["--config", str(deck), "--geography", geo, "--output",
                 str(tmp_path / "run"), "--days", "1"], build_only=True)
    a, c = d._atm.config, d.coupled_cfg
    assert a.geography == geo and a.grid.resolution == 5 and a.dycore.dt == 225.0
    assert c.atm_owns_surface and c.land_mode == "none" and c.ocean_mode == "dynamic"
    assert a.use_multilayer_land and a.gravity_wave_drag == "none" and not a.unfused_radiation
    assert a.zm_land_fraction == ("none" if shape == "aquaplanet" else "required")
    assert a.ic == "standard" and a.T_init == 300.0
    a.validate_strict()      # the run validates the resolved config; so must the deck


def test_atmosphere_owned_land_keeps_its_rows_and_mesh_alias_is_eligible(tmp_path):
    from legoesm.driver.config import land_stress_eligibility, ocean_flux_surface_eligibility
    geo = str(tmp_path / "ridge.nc")
    mg.idealized("ridge", barrier_lon=0.0).to_netcdf(geo)
    d = rc.main(["--config", str(REPO / "config/esoteric/em2009_ridge.yaml"),
                 "--geography", geo, "--output", str(tmp_path / "run"), "--days", "1"],
                build_only=True)
    a = d._atm.config
    assert a.snow_albedo_feedback          # production row survives atm_owns_surface
    v = a._replace(grid=a.grid._replace(grid_type="voronoi"))
    assert ocean_flux_surface_eligibility(v)[0] and land_stress_eligibility(v)[0]
