"""Bound all-g-point gas optics: gradients, misuse and the timing bench.

Their numerics are held to the Fortran oracle in tests/atmosphere/test_rrtmgp_rte_oracle.py.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

_SPEC = importlib.util.spec_from_file_location(
    "bench_rrtmgp_solve", Path(__file__).parents[2] / "scripts/bench/bench_rrtmgp_solve.py")
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _solver():
    return RRTMGP.from_legoesm_config(RRTMGPConfig(
        overhead_layer=True, use_scan=True, include_clouds=True, gpoint_batch_size=16))


def test_gradient_is_finite_and_nonzero_outside_the_table():
    t, kw = bench.inputs(4, 16)
    t = t - 60.0  # the top layers fall below the table's 160 K
    s = _solver()
    g = jax.grad(lambda x: jnp.sum(s.solve_columns(T=x, sfc_temperature=x[:, -1], **kw).heating_rate))(t)
    assert np.isfinite(g).all() and np.abs(g).max() > 0


def test_reading_gas_optics_needs_the_matching_spectrum_bound():
    s = _solver().optics_lib
    x = jnp.full((2, 1, 4), 250.0)
    with pytest.raises(ValueError, match="with_all_gpts"):
        s.compute_lw_optical_properties(x * 200.0, x, x * 1e20, jnp.array(0))
    lw_only = s.with_all_gpts(True, x * 200.0, x, x * 1e20, None)
    with pytest.raises(ValueError, match="with_all_gpts"):
        lw_only.compute_sw_optical_properties(x * 200.0, x, x * 1e20, jnp.array(0))


def test_bench_inputs_have_level_and_layer_shapes():
    t, kw = bench.inputs(4, 8)
    assert t.shape == (4, 8) and kw["p_half"].shape == (4, 9)


@pytest.mark.parametrize("start,end", [(-1, 0), (0, 2), (3, 3)])
def test_malformed_minor_band_limits_are_refused(start, end):
    side = dict(n_minor_absrb_lower=2, minor_lower_bnd_start=[start], minor_lower_bnd_end=[end],
                n_minor_absrb_upper=2, minor_upper_bnd_start=[2], minor_upper_bnd_end=[2])
    names = ("kminor_lower", "kminor_upper", "idx_minor_gases_lower", "idx_minor_gases_upper",
             "minor_lower_scales_with_density", "minor_upper_scales_with_density",
             "idx_scaling_gases_lower", "idx_scaling_gases_upper", "lower_scale_by_complement",
             "upper_scale_by_complement", "minor_lower_gpt_shift", "minor_upper_gpt_shift")
    lookup = SimpleNamespace(bnd_lims_gpt=[[0, 1]], idx_h2o=0, **side, **dict.fromkeys(names))
    cells = SimpleNamespace(temperature=jnp.zeros(1), vmr_fields=None, mix=lambda *_: {})
    with pytest.raises(ValueError, match="minor absorber band limits"):
        gas_optics._minor_band(lookup, cells, jnp.ones(1), 0)
