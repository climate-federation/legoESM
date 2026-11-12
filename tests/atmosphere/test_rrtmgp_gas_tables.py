"""RRTMGP gas-optics table selection (``ExperimentConfig.rrtmgp_gas_tables``).

"reduced" = LW g128 / SW g112 (module default, byte-identical: the lane fields
stay ""), "full" = LW g256 / SW g224.  The production AMIP deck selects "full".
The per-band CMIP6 solar spectrum must be expanded to the g-points of the SAME
SW table the radiation runs, so the run's solar config carries that table too.
"""
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as rrtmgp_mod
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import (
    rrtmgp_config_from_experiment,
    solar_config_from_experiment,
)
from legoesm.forcing import external

REPO = Path(__file__).resolve().parents[2]


def _stem(path):
    return Path(path).name.split(".")[0]  # .nc or .zarr store


def test_builder_maps_table_set_to_files():
    red = rrtmgp_config_from_experiment(ExperimentConfig())
    assert (red.lw_gas_file, red.sw_gas_file) == ("", "")
    full = rrtmgp_config_from_experiment(ExperimentConfig(rrtmgp_gas_tables="full"))
    assert _stem(full.lw_gas_file) == "rrtmgp-gas-lw-g256"
    assert _stem(full.sw_gas_file) == "rrtmgp-gas-sw-g224"
    assert Path(full.lw_gas_file).exists() and Path(full.sw_gas_file).exists()


def test_unknown_table_set_raises():
    with pytest.raises(ValueError, match="rrtmgp_gas_tables"):
        rrtmgp_mod.gas_table_files("g256")
    with pytest.raises(ValueError, match="rrtmgp_gas_tables"):
        ExperimentConfig(rrtmgp_gas_tables="ful").validate_strict()


def test_full_tables_refused_where_nothing_reads_them():
    with pytest.raises(ValueError, match="rrtmgp_gas_tables"):
        ExperimentConfig(rrtmgp_gas_tables="full", radiation="gray").validate_strict()


def test_full_tables_load_with_reference_gpoint_counts():
    full = rrtmgp_mod.RRTMGP.from_legoesm_config(
        rrtmgp_config_from_experiment(ExperimentConfig(rrtmgp_gas_tables="full")))
    red = rrtmgp_mod.RRTMGP.from_legoesm_config(
        rrtmgp_config_from_experiment(ExperimentConfig()))
    assert (full.optics_lib.n_gpt_lw, full.optics_lib.n_gpt_sw) == (256, 224)
    assert (red.optics_lib.n_gpt_lw, red.optics_lib.n_gpt_sw) == (128, 112)


def _ssi_file(path, bands):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("numwl", bands.size)
        ds.createVariable("time", "f8", ("time",))[:] = [0.0, 400.0]
        ds.createVariable("TSI", "f8", ("time",))[:] = [1361.0, 1361.0]
        ds.createVariable("SSI_frac", "f8", ("time", "numwl"))[:] = np.stack([bands, bands])


def test_run_solar_config_expands_to_the_run_sw_table(tmp_path):
    """The run's solar config (the one the driver builds) expands a 14-band
    spectrum to 112 g-points with the reduced tables and to 224 with the full
    ones, preserving each band's total."""
    f = tmp_path / "ssi.nc"
    bands = np.linspace(1.0, 2.0, 14)
    bands = bands / bands.sum()
    _ssi_file(f, bands)
    kw = dict(solar_source="spectral_file", solar_file=str(f), solar_tsi_var="TSI",
              solar_spectral_var="SSI_frac", solar_spectral_band_order="as_is")
    out = {}
    for t in ("reduced", "full"):
        cfg = solar_config_from_experiment(ExperimentConfig(rrtmgp_gas_tables=t, **kw))
        spec = external.get_solar_forcing_at_time(cfg, 10.0)["solar_fraction_by_gpt"]
        out[t] = np.asarray(spec)
    assert out["reduced"].shape == (112,) and out["full"].shape == (224,)
    sw = rrtmgp_mod.gas_table_files("full")[1]
    with netCDF4.Dataset(sw) as ds:
        lims = np.asarray(ds["bnd_limits_gpt"][:], int)
    per_band = np.array([out["full"][a - 1:b].sum() for a, b in lims])
    np.testing.assert_allclose(per_band, bands, rtol=1e-12)


def test_cli_flag_round_trips():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    p = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(p.parse_args(["--dataset", "analytical"]), p))
    assert cfg.rrtmgp_gas_tables == "reduced"
    cfg = build_config_from_args(_postprocess_args(p.parse_args(
        ["--dataset", "analytical", "--rrtmgp-gas-tables", "full"]), p))
    assert cfg.rrtmgp_gas_tables == "full"


def test_production_deck_selects_full_tables():
    """Through run_amip's own --config loader (choices-validated) into the
    ExperimentConfig the run builds, and on to the RRTMGP files."""
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    p = build_arg_parser()
    keys = load_yaml_config(str(REPO / "config/amip/amip_production.yaml"), p)
    assert keys["rrtmgp_gas_tables"] == "full"
    p.set_defaults(rrtmgp_gas_tables=keys["rrtmgp_gas_tables"])
    cfg = build_config_from_args(_postprocess_args(p.parse_args(["--dataset", "analytical"]), p))
    assert _stem(rrtmgp_config_from_experiment(cfg).lw_gas_file) == "rrtmgp-gas-lw-g256"


def test_one_legal_set_everywhere():
    """validate_strict, the CLI choices and the file map name the same sets."""
    from scripts.run.run_amip import build_arg_parser
    choices = next(a.choices for a in build_arg_parser()._actions
                   if a.dest == "rrtmgp_gas_tables")
    assert set(choices) == set(rrtmgp_mod.GAS_TABLE_FILES)
    for t in rrtmgp_mod.GAS_TABLE_FILES:
        ExperimentConfig(rrtmgp_gas_tables=t, radiation="rrtmgp").validate_strict()
