"""The raised-lid sigma layout: the production L30 troposphere kept bit-identical
below sigma 0.109, nlev-27 log-spaced stratospheric layers up to sigma_top;
the default layout stays byte-identical; the config gates hold."""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.config import ExperimentConfig, GridConfig
from legoesm.grids.vertical import SIGMA_LAYOUTS, create_sigma_coordinate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "scripts", "run"))
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402

EXPECTED_DP_HPA = [1.119, 1.744, 2.720, 4.241, 6.613, 10.312, 16.080, 25.073, 39.097]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_default_layout_is_the_legacy_linspace(dtype):
    legacy = jnp.linspace(0.01, 1.0, 31, dtype=dtype)          # the pre-layout construction
    for kw in ({}, {"layout": "standard"}):
        sh = create_sigma_coordinate(30, dtype=dtype, **kw).sigma_half
        assert sh.dtype == dtype and np.array_equal(np.asarray(sh), np.asarray(legacy))
    assert SIGMA_LAYOUTS == ("standard", "l30_trop_logstrat")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_l36_retained_interfaces_bit_identical_in_each_dtype(dtype):
    c = create_sigma_coordinate(36, sigma_top=0.002, layout="l30_trop_logstrat", dtype=dtype)
    ref = np.asarray(jnp.linspace(0.01, 1.0, 31, dtype=dtype))[3:]
    assert c.sigma_half.dtype == dtype
    assert np.array_equal(np.asarray(c.sigma_half)[9:], ref)
    assert np.all(np.diff(np.asarray(c.sigma_half)) > 0)
    assert bool(jnp.all(jnp.isfinite(c.alpha)))


def test_layout_builds_under_jit():
    f = jax.jit(lambda: create_sigma_coordinate(36, sigma_top=0.002, layout="l30_trop_logstrat",
                                                dtype=jnp.float64).sigma_half)
    assert np.asarray(f()).shape == (37,)


def test_l36_keeps_the_l30_troposphere_and_logs_the_stratosphere():
    c = create_sigma_coordinate(36, sigma_top=0.002, layout="l30_trop_logstrat",
                                dtype=jnp.float64)
    sh = np.asarray(c.sigma_half)
    ref = np.asarray(jnp.linspace(0.01, 1.0, 31, dtype=jnp.float64))[3:]
    assert np.array_equal(sh[9:], ref)                     # bit-identical retained interfaces
    assert sh[0] == 0.002 and np.all(np.diff(sh) > 0)
    dln = np.diff(np.log(sh[:10]))
    assert np.max(np.abs(dln - dln[0])) < 1e-12
    np.testing.assert_allclose(np.diff(sh[:10]) * 1000.0, EXPECTED_DP_HPA, atol=1e-2)
    assert c.sigma_full.shape == (36,)


@pytest.mark.parametrize("kw", [dict(layout="bogus"),
                                dict(n_levels=27, layout="l30_trop_logstrat"),
                                dict(sigma_top=0.2, layout="l30_trop_logstrat"),
                                dict(sigma_top=0.108999, layout="l30_trop_logstrat"),
                                dict(layout="l30_trop_logstrat", tropopause_refine=3.0)])
def test_invalid_layout_requests_raise(kw):
    n = kw.pop("n_levels", 36)
    top = kw.pop("sigma_top", 0.002)
    with pytest.raises(ValueError):
        create_sigma_coordinate(n, sigma_top=top, dtype=jnp.float64, **kw)


def test_config_gates_and_wiring():
    assert GridConfig().sigma_layout == "standard"
    ExperimentConfig(grid=GridConfig(vertical_coord="sigma", nlev=36, sigma_top=0.002,
                                     sigma_layout="l30_trop_logstrat")).validate_strict()
    for bad in (dict(nlev=27), dict(nlev=36, tropopause_refine=3.0), dict(nlev=36, sigma_top=0.2),
                dict(nlev=36, sigma_top=0.108)):
        with pytest.raises(ValueError, match="sigma_layout"):
            ExperimentConfig(grid=GridConfig(vertical_coord="sigma", sigma_layout="l30_trop_logstrat",
                                             **{"sigma_top": 0.002, **bad})).validate_strict()
    with pytest.raises(ValueError, match="sigma_layout"):
        ExperimentConfig(grid=GridConfig(vertical_coord="hybrid",
                                         sigma_layout="l30_trop_logstrat")).validate_strict()
    from legoesm.driver.config import DycoreConfig
    with pytest.raises(ValueError, match="sigma_top"):
        ExperimentConfig(dycore=DycoreConfig(model_type="nonhydrostatic"),
                         grid=GridConfig(vertical_coord="sigma", sigma_top=0.002)).validate_strict()
    with pytest.raises(ValueError, match="sigma_layout"):
        ExperimentConfig(dycore=DycoreConfig(model_type="nonhydrostatic"),
                         grid=GridConfig(vertical_coord="sigma", nlev=36, sigma_top=0.002,
                                         sigma_layout="l30_trop_logstrat")).validate_strict()
    ExperimentConfig(dycore=DycoreConfig(model_type="nonhydrostatic")).validate_strict()
    args = build_arg_parser().parse_args(["--sigma-layout", "l30_trop_logstrat",
                                          "--sigma-top", "0.002", "--nlev", "36"])
    assert build_config_from_args(args).grid.sigma_layout == "l30_trop_logstrat"


def test_yaml_forwards_the_layout(tmp_path):
    from legoesm.config import Config
    yml = tmp_path / "deck.yaml"
    yml.write_text("grid:\n  type: mpas\n  vertical_coord: sigma\n  n_levels: 36\n"
                   "  sigma_top: 0.002\n  sigma_layout: l30_trop_logstrat\n")
    g = Config.from_yaml(str(yml)).to_experiment_config().grid
    assert g.sigma_layout == "l30_trop_logstrat" and g.nlev == 36
