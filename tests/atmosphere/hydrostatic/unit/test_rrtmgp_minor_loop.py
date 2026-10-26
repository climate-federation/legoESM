"""RRTMGP minor-absorber loop: tight trip count == full masked loop, bitwise.

``_compute_minor_optical_depth`` walks each band's absorber range with a
static trip count equal to the widest band's range.  It must reproduce a
loop over every absorber (masked to the band) bit for bit -- values and
gradients -- on the production LW/SW tables, which include empty bands.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
    gas_optics,
    lookup_gas_optics_longwave,
    lookup_gas_optics_shortwave,
    lookup_volume_mixing_ratio,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import (
    constants as optics_constants,
)
from tests.legoesm_paths import legoesm_source_path

_DATA = "atmosphere/physics/radiation/rrtmgp/optics/rrtmgp_data/"
_TABLES = {
    "lw": (lookup_gas_optics_longwave, "rrtmgp-gas-lw-g128.nc"),
    "sw": (lookup_gas_optics_shortwave, "rrtmgp-gas-sw-g112.nc"),
}


@pytest.fixture(scope="module", params=sorted(_TABLES))
def table(request):
    module, name = _TABLES[request.param]
    lookup = module.from_data_file(str(legoesm_source_path(_DATA + name)))
    vmr_lib = lookup_volume_mixing_ratio.LookupVolumeMixingRatio(
        global_means={
            optics_constants.DRY_AIR_KEY: optics_constants.DRY_AIR_VMR,
            "co2": 415e-6, "ch4": 1900e-9, "n2o": 332e-9, "o2": 0.20948,
            "n2": 0.78084, "co": 1.5e-7, "ccl4": 7.5e-11, "cfc11": 2.2e-10,
            "cfc12": 5.0e-10, "cfc22": 2.4e-10, "cf4": 8.5e-11, "no2": 3.0e-10,
        },
        profiles=None,
    )
    return lookup, vmr_lib


def _profile(lookup, dtype):
    rng = np.random.default_rng(11)
    shape = (6, 9)
    T = rng.uniform(120.0, 370.0, shape)  # beyond the table on both ends
    p = np.exp(rng.uniform(np.log(1.0), np.log(1.1e5), shape))
    h2o = rng.uniform(0.0, 0.03, shape)
    h2o[0, :] = 0.0  # zero-vapour column
    o3 = rng.uniform(0.0, 1e-5, shape)
    cast = lambda a: jnp.asarray(a, dtype=dtype)
    vmr = {lookup.idx_h2o: cast(h2o), lookup.idx_o3: cast(o3)}
    return cast(T), cast(p), cast(np.full(shape, 1e24)), vmr


def _tau(lookup, vmr_lib, T, p, mol, vmr, lower, n_trips):
    gpts = jnp.arange(lookup.n_gpt)
    f = lambda T_, g: gas_optics._compute_minor_optical_depth(
        lookup, vmr_lib, mol, T_, p, g, lower, vmr, n_trips=n_trips)
    tau = jax.vmap(lambda g: f(T, g))(gpts)
    grad = jax.grad(lambda T_: jnp.sum(jax.vmap(lambda g: f(T_, g))(gpts)))(T)
    return np.asarray(tau), np.asarray(grad)


def _bits(a):
    return a.view(np.uint64 if a.dtype == np.float64 else np.uint32)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("lower", [True, False])
def test_tight_loop_is_bitwise_the_full_masked_loop(table, dtype, lower):
    lookup, vmr_lib = table
    T, p, mol, vmr = _profile(lookup, dtype)
    n = lookup.n_minor_absrb_lower if lower else lookup.n_minor_absrb_upper
    starts = np.asarray(lookup.minor_lower_bnd_start if lower
                        else lookup.minor_upper_bnd_start)
    assert np.any(starts == n), "no empty band: sentinel path untested"
    tight, g_tight = _tau(lookup, vmr_lib, T, p, mol, vmr, lower, None)
    full, g_full = _tau(lookup, vmr_lib, T, p, mol, vmr, lower, n)
    assert np.all(np.isfinite(tight)) and np.any(tight > 0)
    np.testing.assert_array_equal(_bits(tight), _bits(full))
    np.testing.assert_array_equal(_bits(g_tight), _bits(g_full))


def test_short_trip_count_drops_terms(table):
    """Non-vacuity: a trip count below the widest band changes the answer."""
    lookup, vmr_lib = table
    T, p, mol, vmr = _profile(lookup, jnp.float64)
    tight, _ = _tau(lookup, vmr_lib, T, p, mol, vmr, True, None)
    short, _ = _tau(lookup, vmr_lib, T, p, mol, vmr, True, 1)
    assert not np.array_equal(tight, short)
