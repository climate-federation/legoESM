"""``land_fraction_source``: where the column land fraction comes from.

topography (default) / era5 (lat-lon, needs a named store) / zero (lat-lon).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.model_driver import ModelDriver


def _cfg(grid_type="latlon", column_lane=False, **kw):
    disc = {"latlon": "latlon_cgrid", "cubed_sphere": "cdgrid"}[grid_type]
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=8 if grid_type == "latlon" else 4,
                        nlev=3),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization=disc, column_lane=column_lane),
        topography="flat", radiation="gray", days=1, **kw)


def _errors(cfg):
    with pytest.raises(ValueError) as e:
        cfg.validate_strict()
    return str(e.value)


def test_validate_rejects_unknown_source():
    assert "land_fraction_source" in _errors(_cfg(land_fraction_source="lsm"))


def test_validate_era5_needs_a_store():
    assert "land_fraction_era5_zarr" in _errors(_cfg(land_fraction_source="era5"))


@pytest.mark.parametrize("src,kw", [("era5", {"land_fraction_era5_zarr": "x.zarr"}),
                                     ("zero", {})])
def test_validate_era5_and_zero_are_latlon_only(src, kw):
    assert "lat-lon grid" in _errors(
        _cfg("cubed_sphere", land_fraction_source=src, **kw))


@pytest.mark.parametrize("column_lane", [False, True])
def test_driver_takes_the_era5_land_fraction(monkeypatch, tmp_path, column_lane):
    import legoesm.training.aimip_spatial as sp
    seen = {}

    def fake(grid, era5_config=None):    # (lat, lon) like the real regrid
        seen["store"] = era5_config.zarr_store
        seen["flux"] = era5_config.flux_zarr
        return jnp.full((len(grid.lat), len(grid.lon)), 0.25)

    monkeypatch.setattr(sp, "era5_land_fraction", fake)
    d = ModelDriver(_cfg(column_lane=column_lane, land_fraction_source="era5",
                         land_fraction_era5_zarr="my.zarr"),
                    output_dir=str(tmp_path / "run"))
    f = np.asarray(d.static_land_fraction())
    assert (seen["store"], seen["flux"]) == ("my.zarr", "")
    assert f.shape == tuple(d.grid.grid_shape_2d)
    assert np.all(f == 0.25)          # flat topography alone would give 0
    d.setup()                         # builds the physics pipeline
    assert d.physics.run_has_no_land is False   # land present: no zero bypass


def test_driver_zero_source_is_explicit_zeros(monkeypatch, tmp_path):
    import legoesm.grids.topography as topo

    def land_everywhere(grid, config=None):   # terrain that says "all land"
        return (jnp.zeros(grid.grid_shape_2d), jnp.ones(grid.grid_shape_2d))

    monkeypatch.setattr(topo, "load_real_topography", land_everywhere)
    d = ModelDriver(_cfg(land_fraction_source="zero")._replace(
                        topography="terrain.nc"),
                    output_dir=str(tmp_path / "run"))
    f = d.static_land_fraction()
    assert f is not None and not bool(jnp.any(f))
    # marked land-free, so McFarlane gets an explicit zero land fraction
    d.setup()                         # builds the physics pipeline
    assert d.physics.run_has_no_land is True


def test_run_amip_flags_reach_the_config():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args)
    p = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(p.parse_args([
        "--dataset", "analytical", "--land-fraction-source", "era5",
        "--land-fraction-era5-zarr", "s.zarr"]), p))
    assert (cfg.land_fraction_source, cfg.land_fraction_era5_zarr) == ("era5", "s.zarr")
    cfg0 = build_config_from_args(_postprocess_args(
        p.parse_args(["--dataset", "analytical"]), p))
    assert cfg0.land_fraction_source == "topography"


def _latlon_script():
    path = Path(__file__).resolve().parents[2] / "scripts/run/run_aimip_latlon.py"
    spec = importlib.util.spec_from_file_location("_aimip_latlon_lfs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("argv", [[], ["--era5-zarr", "other.zarr"]])
def test_aimip_latlon_names_era5_and_its_store(argv):
    from legoesm.training.era5_to_state import TrainingERA5Config
    mod = _latlon_script()
    cfg = mod.build_latlon_config(mod.build_parser().parse_args(argv))
    assert cfg.land_fraction_source == "era5"
    assert cfg.land_fraction_era5_zarr == (
        argv[1] if argv else TrainingERA5Config().zarr_store)



def _msgs(cfg):
    try:
        cfg.validate_strict()
    except ValueError as e:
        return str(e)
    return ""


def test_mask_file_with_era5_or_zero_is_refused():
    for src, kw in (("era5", {"land_fraction_era5_zarr": "x.zarr"}), ("zero", {})):
        assert "two coastlines" in _msgs(
            _cfg(land_fraction_source=src, land_mask_path="lsm.nc", **kw))


def test_land_tile_on_flat_terrain_accepts_era5_not_zero():
    tile = "land tile was requested"
    assert tile in _msgs(_cfg(slab_land_active=True))
    assert tile not in _msgs(_cfg(slab_land_active=True,
                                  land_fraction_source="era5",
                                  land_fraction_era5_zarr="x.zarr"))
    for kw in ({"slab_land_active": True}, {"use_multilayer_land": True}):
        assert "land physics on no land" in _msgs(
            _cfg(land_fraction_source="zero", **kw))
