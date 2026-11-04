"""Oracle test: runoff heat content vs NEMO 5.0.1 ORCA1 (tier-3 RNF unit, defect P2).

NEMO (ln_rnf_tem=.false., sbcrnf.F90:148) sets the runoff temperature content
``rnf_tsc(:,:,jp_tem) = MAX(sst_m, 0) * rnf * r1_rho0``: river water is fresh,
so it never enters below 0 degC.  The port used the raw SST, giving negative
heat content under sub-zero SST.  Fixture: 60 wet runoff columns (40 with
sst_m < 0) with NEMO's recorded rnf, sst_m, r1_rho0 and rnf_tsc, plus the
per-column gap of the smooth production default; provenance in
``rnf_heat_clamp.json``.

RULE 5 (smooth default, hard limit via the config field
``runoff_temperature_floor_sharpness``): hard limit vs NEMO at rel 1e-12 (C2:
NEMO spread 6.7e-16); smooth default vs the stored per-column gap, in kelvin
(heat gap / runoff water flux), at rel 1e-6 / abs 1e-12 K; gradient through the switch with ``check_grads``.
"""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from legoesm.ocean.freshwater import FreshwaterForcing, real_freshwater_entry
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.state import LatLonCGridOceanConfig

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "rnf_heat_clamp.npz"
_NLEV = 3


def _column_heat(rnf, sst, rho_0, **kw):
    """Runoff-only top-cell entry (spread=None): column heat [K m/s]."""
    z = jnp.zeros_like(rnf)
    fw = FreshwaterForcing(precip=z, evap=z, runoff=rnf, ice_fw=z, restoring=None)
    mask = jnp.ones_like(rnf)
    h_k = jnp.ones(rnf.shape + (_NLEV,))
    T = jnp.broadcast_to(sst[:, None], h_k.shape)
    _, heat = real_freshwater_entry(fw, rnf / rho_0 * mask, h_k, mask, rho_0, T, **kw)
    return jnp.sum(heat, axis=-1)


def _load():
    d = np.load(_FIX)
    return (jnp.asarray(d["rnf"]), jnp.asarray(d["sst_m"]), 1.0 / float(d["r1_rho0"]),
            np.asarray(d["nemo_rnf_tsc_T"]), np.asarray(d["smooth_gap_default"]))


def test_runoff_heat_matches_nemo_hard_limit():
    rnf, sst, rho_0, nemo, _ = _load()
    assert int((sst < 0).sum()) >= 40          # fixture exercises the clamp
    got = np.asarray(_column_heat(rnf, sst, rho_0, runoff_t_floor_sharpness=math.inf))
    np.testing.assert_allclose(got, nemo, rtol=1e-12, atol=0.0)


def _gap_K(heat, nemo, rnf, rho_0):
    """Gap in temperature units [K]: heat difference / runoff water flux."""
    return (np.asarray(heat) - nemo) / (np.asarray(rnf) / rho_0)


def test_runoff_heat_smooth_default_recorded_gap():
    rnf, sst, rho_0, nemo, gap = _load()
    stored = gap / (np.asarray(rnf) / rho_0)                  # [K], up to 4e-4 K
    assert stored.max() > 1e-4                                  # gap is resolved
    got = _gap_K(_column_heat(rnf, sst, rho_0), nemo, rnf, rho_0)   # production default
    np.testing.assert_allclose(got, stored, rtol=1e-6, atol=1e-12)
    # The stored gap is the LogSumExp overshoot ln(1+exp(-s|T|))/s, s = 100.
    s = 100.0
    analytic = np.log1p(np.exp(-s * np.abs(np.asarray(sst)))) / s
    np.testing.assert_allclose(stored, analytic, rtol=1e-6, atol=1e-12)
    # Planted violation: the hard limit must NOT pass the smooth-gap check.
    hard = _gap_K(_column_heat(rnf, sst, rho_0, runoff_t_floor_sharpness=math.inf),
                  nemo, rnf, rho_0)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_allclose(hard, stored, rtol=1e-6, atol=1e-12)


def test_config_default_is_the_smooth_width():
    for cfg in (LatLonCGridOceanConfig(), MPASOceanConfig()):
        assert cfg.runoff_temperature_floor_sharpness == 100.0


def test_runoff_heat_gradient_across_floor():
    rnf = jnp.asarray([1.0e-3, 1.0e-3, 1.0e-3])
    f = lambda s: _column_heat(rnf, s, 1026.0)   # noqa: E731
    for s0 in (-0.005, 0.0, 0.005):
        s = jnp.full((3,), s0)
        check_grads(f, (s,), order=2, modes=("fwd", "rev"))
        g = jax.grad(lambda x: jnp.sum(f(x)))(s)
        assert np.all(np.isfinite(np.asarray(g))) and np.all(np.asarray(g) > 0.0)
