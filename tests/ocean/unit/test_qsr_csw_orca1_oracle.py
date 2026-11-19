"""ORCA1 oracle: the penetrating-shortwave deposit uses the run's c_sw.

NEMO 5.0.1 ORCA1 ``tra_qsr`` (RGB, ``ln_qsr_rgb``) deposits ``qsr`` with
``r1_rho0_rcp = 1/(rho0*rcp)``, rcp = 3991.86795711963.  legoESM's
``_bc_external_surface_forcing`` took the run's ``c_sw`` for the non-solar
surface term but called the shortwave kernels without it, so the penetrating
deposit used the library default 3994.0 whenever a card set another value
(0.05 % too little heat, and a column heat budget that no longer closed).

Fixture: 60 wet columns from two ranks of the tier-3 QSR record (provenance in
``fixtures/orca1_t3/qsr_csw.json``).  Levels 1-8 sit above every NEMO
extinction cutoff (finding Q1, not this defect), where NEMO's 20-seed spread is
1.1e-14, so the tolerance is 1e-12 (campaign vote C2).
"""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as PE

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "qsr_csw.npz"
_NLEV_SCORED = 8   # NEMO levels 1..8: above the IR/R/G/B extinction levels


@pytest.fixture(scope="module")
def fx():
    set_policy(PrecisionPolicy.fp64())
    return dict(np.load(_FIX))


def _call(fx, c_sw):
    sw = jnp.asarray(fx["qsr"])
    tmask = jnp.asarray(fx["tmask"])
    h_k = jnp.asarray(fx["e3t_live"]) * tmask
    zeros = jnp.zeros_like(h_k)
    e3t_1d = jnp.asarray(fx["e3t_1d"])
    z_coord = SimpleNamespace(
        dz_ref=e3t_1d,
        z_half_ref=-jnp.concatenate([jnp.zeros(1), jnp.cumsum(e3t_1d)]))
    # q_net = qsr: the non-solar surface term is exactly zero, so dT is the
    # penetrating deposit alone -- the quantity NEMO's tra_qsr adds.
    sf = SimpleNamespace(q_net=sw, sw_down=sw, chl=jnp.asarray(fx["chl"]))
    out = PE._bc_external_surface_forcing(
        zeros, zeros, zeros, zeros, sf, zeros, zeros, zeros, zeros, h_k,
        z_coord, jnp.ones_like(sw), None, float(fx["rho0"]),
        tmask[..., 0], tmask, shortwave_scheme="auto", c_sw=c_sw)
    return np.asarray(out[2]), np.asarray(h_k)


def test_rgb_deposit_matches_nemo_with_run_c_sw(fx):
    dT, _ = _call(fx, float(fx["rcp"]))
    ref = fx["dT"]
    wet = fx["tmask"][..., :_NLEV_SCORED] > 0
    a, b = dT[..., :_NLEV_SCORED][wet], ref[..., :_NLEV_SCORED][wet]
    rel = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    assert wet.sum() > 400
    assert rel.max() <= 1e-12, rel.max()


def test_column_heat_closes_with_run_c_sw(fx):
    """Energy: rho0*c_sw*sum(dT*h) equals the absorbed shortwave (q_net)."""
    c_sw = float(fx["rcp"])
    dT, h = _call(fx, c_sw)
    heat = float(fx["rho0"]) * c_sw * np.sum(dT * h, axis=-1)
    sw = fx["qsr"]
    # Everything that penetrates is deposited somewhere in the column (the
    # seabed remainder goes to the deepest wet cell), so closure is exact to
    # rounding of a 74-term sum.
    np.testing.assert_allclose(heat, sw, rtol=1e-12, atol=0)


def test_default_c_sw_bit_identical(fx):
    """c_sw=None keeps the library value: card runs (c_sw 3994) unchanged."""
    from legoesm.ocean.eos import c_sw as lib_c_sw
    a, _ = _call(fx, None)
    b, _ = _call(fx, lib_c_sw)
    assert np.array_equal(a, b)
