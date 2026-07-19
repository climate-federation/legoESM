"""Faithfulness pins for the Andreae (2009) AOD -> CCN inversion.

Target: ``ccn_from_aod`` and ``specified_nc_field`` in
``legoesm.atmosphere.physics.microphysics.aerosol_activation``.

Most-trustful source
--------------------
Andreae (2009), Atmos. Chem. Phys. 9, 543-556 (doi:10.5194/acp-9-543-2009): the
global power-law fit AOT_500 = 0.0027 * CCN_0.4^0.640 (CCN in cm^-3, r^2=0.88).
The module INVERTS it, N_ccn = (AOT / c)^(1/b), clips to the observationally
constrained range, and converts cm^-3 -> m^-3.

This module is LIVE glue for both the coupled physics_pipeline and the combined-
physics microphysics/radiation factories but was never oracle-pinned (the
separate ``arg_activation`` Abdul-Razzak-Ghan scheme has its own test).

Certification (test-only)
-------------------------
1. Exact inversion N = (max(aod, c*n_min^b)/c)^(1/b) * 1e6 vs an independent
   reimplementation typed from Andreae (2009), to round-off.
2. Departure canaries: the exponent is the RECIPROCAL 1/b (not b); the cm^-3 ->
   m^-3 factor is 1e6; the [n_min, n_max] clip binds; coefficient plumbing.
3. Physical realism: AOT=0.075 -> ~200 cm^-3 (paper clean-continental average).
4. Gradient-safety at aod=0; specified_nc_field column-sum + broadcast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
    CCNFromAODConfig,
    ccn_from_aod,
    specified_nc_field,
)

jax.config.update("jax_enable_x64", True)

_CFG = CCNFromAODConfig()   # aot_coeff=0.0027, aot_exponent=0.640, n_min=10, n_max=1e4 cm^-3

# Published Andreae (2009) fit coefficients, typed INDEPENDENTLY of the SUT config
# so the default-path pins would catch a wrong CCNFromAODConfig default.
_C_PUB, _B_PUB = 0.0027, 0.640            # AOT500 = 0.0027 * CCN0.4^0.640
_NMIN_PUB, _NMAX_PUB = 10.0, 1.0e4        # cm^-3 clip range


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _inv_cm3(aod, c, b, nmin, nmax):
    """Independent Andreae-2009 inversion N=(AOT/c)^(1/b), clipped [nmin,nmax], cm^-3."""
    aod = np.asarray(aod, dtype=np.float64)
    aod_floor = c * nmin ** b
    n = (np.maximum(aod, aod_floor) / c) ** (1.0 / b)
    return np.clip(n, nmin, nmax)


def _lit_m3(aod):
    """Default-path oracle from the PUBLISHED literals (not the SUT config)."""
    return _inv_cm3(aod, _C_PUB, _B_PUB, _NMIN_PUB, _NMAX_PUB) * 1.0e6


def _oracle_m3(aod, cfg):
    """Config-parameterised oracle, for the explicit non-default plumbing tests."""
    return _inv_cm3(aod, cfg.aot_coeff, cfg.aot_exponent,
                    cfg.n_ccn_min_cm3, cfg.n_ccn_max_cm3) * 1.0e6


# AOD values whose diagnosed N lands strictly inside (n_min, n_max) -> unclipped.
_AOD = np.array([0.02, 0.05, 0.1, 0.3])


# ---------------------------------------------------------------------------
# 1. Exact inversion.
# ---------------------------------------------------------------------------
def test_ccn_from_aod_exact_inversion():
    # Default config vs the PUBLISHED-literal oracle -> also catches a wrong default.
    got = np.asarray(ccn_from_aod(_a(_AOD), _CFG))
    np.testing.assert_allclose(got, _lit_m3(_AOD), rtol=1e-12)


def test_ccn_returns_per_m3_not_per_cm3():
    # The cm^-3 -> m^-3 conversion is a factor of 1e6; a dropped conversion would
    # return values ~1e6x too small.
    got = np.asarray(ccn_from_aod(_a(_AOD), _CFG))
    np.testing.assert_allclose(got, _lit_m3(_AOD), rtol=1e-12)
    assert np.all(got > 1.0e6)                                # >= n_min(10 cm^-3)*1e6 = 1e7


# ---------------------------------------------------------------------------
# 2. Departure canaries.
# ---------------------------------------------------------------------------
def test_exponent_is_reciprocal_b():
    # N scales as aod^(1/b); an erroneous aod^b would give a different ratio.
    a1, a2 = 0.05, 0.20                                       # both unclipped
    n1 = float(ccn_from_aod(_a([a1]), _CFG)[0])
    n2 = float(ccn_from_aod(_a([a2]), _CFG)[0])
    ratio = n2 / n1
    expect = (a2 / a1) ** (1.0 / _B_PUB)
    np.testing.assert_allclose(ratio, expect, rtol=1e-12)
    # and it is NOT the wrong (forward) exponent b:
    assert abs(ratio - (a2 / a1) ** _B_PUB) > 1.0


def test_clip_floor_and_cap_bind():
    aod_floor = _C_PUB * _NMIN_PUB ** _B_PUB
    below = float(ccn_from_aod(_a([0.5 * aod_floor]), _CFG)[0])
    np.testing.assert_allclose(below, _NMIN_PUB * 1.0e6, rtol=1e-12)   # floor
    above = float(ccn_from_aod(_a([50.0]), _CFG)[0])          # huge AOD -> cap
    np.testing.assert_allclose(above, _NMAX_PUB * 1.0e6, rtol=1e-12)   # cap


def test_monotone_increasing_in_aod():
    got = np.asarray(ccn_from_aod(_a(_AOD), _CFG))
    assert np.all(np.diff(got) > 0.0)


def test_coefficient_plumbing():
    # Non-default coeff/exponent must flow through to the result (not hard-coded).
    cfg = CCNFromAODConfig(aot_coeff=0.004, aot_exponent=0.5)
    got = np.asarray(ccn_from_aod(_a(_AOD), cfg))
    np.testing.assert_allclose(got, _oracle_m3(_AOD, cfg), rtol=1e-12)
    # differs materially from the default parameterisation:
    assert np.max(np.abs(got - _lit_m3(_AOD))) > 1.0e6


# ---------------------------------------------------------------------------
# 3. Physical realism (Andreae 2009 clean-continental point).
# ---------------------------------------------------------------------------
def test_andreae_clean_continental_point():
    # AOT=0.075 -> 180.21 cm^-3 (published inversion; paper clean-continental ~200+-90).
    n_cm3 = float(ccn_from_aod(_a([0.075]), _CFG)[0]) / 1.0e6
    np.testing.assert_allclose(n_cm3, _inv_cm3(0.075, _C_PUB, _B_PUB, _NMIN_PUB, _NMAX_PUB),
                               rtol=1e-9)
    assert 150.0 < n_cm3 < 260.0                              # within the paper's spread


# ---------------------------------------------------------------------------
# 4. Gradient safety + specified_nc_field.
# ---------------------------------------------------------------------------
def test_floor_guards_gradient_for_subunit_reciprocal_exponent():
    # The aod_floor exists to keep the fractional power AD-safe. For the default
    # exponent (1/b=1.5625>1) the derivative is already 0 at aod=0, so the floor is
    # masked there. It genuinely bites when 1/b<1 (aot_exponent>1, in-bounds): then
    # (aod/c)^(1/b) has an INFINITE derivative at 0 and, chained through the clip,
    # yields a NaN gradient WITHOUT the floor. With the floor the module stays finite.
    cfg = CCNFromAODConfig(aot_exponent=1.5)                  # 1/b = 0.667 < 1
    g = float(jax.grad(lambda a: ccn_from_aod(jnp.reshape(a, (1,)), cfg)[0])(0.0))
    assert np.isfinite(g)                                     # NaN if the floor is removed


def test_differentiable_unclipped():
    g = jax.grad(lambda a: jnp.sum(ccn_from_aod(a, _CFG)))(_a(_AOD))
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_specified_nc_field_column_sum_and_broadcast():
    # Per-layer AOD -> column sum (axis=-1) -> ccn_from_aod -> broadcast across levels.
    per_layer = _a([[0.01, 0.02, 0.02], [0.10, 0.15, 0.05]])  # (ncol=2, nlev=3)
    target = (2, 3)
    got = np.asarray(specified_nc_field(per_layer, target, _CFG))
    col_aod = np.sum(np.asarray(per_layer), axis=-1)          # [0.05, 0.30]
    ref_col = _lit_m3(col_aod)                                # (2,) published-literal oracle
    np.testing.assert_allclose(got, np.broadcast_to(ref_col[:, None], target), rtol=1e-12)
    # every level in a column is identical (broadcast, not per-layer):
    assert np.all(got[:, 0:1] == got)


def test_specified_nc_field_forwards_config():
    # A non-default config must flow through specified_nc_field (not a hard-coded
    # inversion): the output must track the config-parameterised oracle, and differ
    # materially from the default parameterisation.
    per_layer = _a([[0.01, 0.02, 0.02], [0.10, 0.15, 0.05]])
    target = (2, 3)
    cfg = CCNFromAODConfig(aot_coeff=0.004, aot_exponent=0.5)
    got = np.asarray(specified_nc_field(per_layer, target, cfg))
    col_aod = np.sum(np.asarray(per_layer), axis=-1)
    ref_col = _oracle_m3(col_aod, cfg)
    np.testing.assert_allclose(got, np.broadcast_to(ref_col[:, None], target), rtol=1e-12)
    default = np.asarray(specified_nc_field(per_layer, target, _CFG))
    assert np.max(np.abs(got - default)) > 1.0e6


def test_specified_nc_field_differentiable():
    per_layer = _a([[0.01, 0.02, 0.02], [0.10, 0.15, 0.05]])
    g = jax.grad(lambda x: jnp.sum(specified_nc_field(x, (2, 3), _CFG)))(per_layer)
    assert np.all(np.isfinite(np.asarray(g)))
