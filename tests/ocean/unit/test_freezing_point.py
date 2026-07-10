"""Salinity(-pressure)-dependent seawater freezing point (``eos.freezing_point``).

MED-1.  The freeze checks that cap SST / trigger ice formation historically used
a single FIXED constant ``constants.T_freeze_ocean`` (271.35 K, ~-1.8 C).  Real
seawater freezes along a LIQUIDUS that decreases with salinity (and, weakly,
pressure): at S = 35 PSU, p = 0 the true value is ~-1.92 C.  ``freezing_point``
exposes three schemes; this suite exercises the real function directly.

Checks:
  * ``constant`` scheme is byte-identical to ``constants.T_freeze_ocean`` (scalar
    + array), so the default wiring in the ocean/slab freeze paths is inert;
  * ``linear_S`` and ``unesco`` give 0 C at S=0 (pure water) and the expected
    depressions at S=35 (unesco ~-1.92 C, NOT -1.8);
  * SIGN: T_f is monotone DECREASING in salinity, and pressure LOWERS it;
  * the function is differentiable (grad finite, incl. at S=0), jit-safe, and
    vmap-safe, and floors negative-S overshoots without NaN;
  * unknown scheme raises ``ValueError`` (dispatch hardening);
  * the consumer configs default to ``scheme="constant"`` (byte-identical opt-in).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    VALID_FREEZE_SCHEMES,
    FreezingPointConfig,
    freezing_point,
)

# Reference depression in degC (temperature difference => same value in K).
_T0_K = constants.T_freeze  # pure-water freezing point (0 degC reference) [K]


# ---------------------------------------------------------------------------
# constant scheme — byte-identical to the historical fixed value
# ---------------------------------------------------------------------------

def test_constant_scheme_byte_identical_scalar():
    """``constant`` returns exactly ``constants.T_freeze_ocean`` (any S/p)."""
    for S in (0.0, 35.0, 40.0):
        tf = freezing_point(S, 0.0, scheme="constant")
        assert float(tf) == constants.T_freeze_ocean  # exact, not approx
    # Salinity/pressure are ignored by the constant scheme.
    assert float(freezing_point(35.0, 1.0e7, scheme="constant")) == constants.T_freeze_ocean


def test_constant_scheme_byte_identical_array():
    """Array input -> per-cell field of the fixed constant, exactly."""
    S = jnp.array([0.0, 30.0, 34.7, 40.0])
    tf = freezing_point(S, scheme="constant")
    assert tf.shape == S.shape
    assert np.array_equal(np.asarray(tf), np.full(S.shape, constants.T_freeze_ocean))


def test_default_scheme_is_constant():
    """The default (no ``scheme=``) is the byte-identical constant."""
    assert float(freezing_point(35.0)) == constants.T_freeze_ocean
    assert FreezingPointConfig().scheme == "constant"


# ---------------------------------------------------------------------------
# pure-water endpoint (S=0 -> 0 degC) for the liquidus schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_pure_water_freezes_at_zero_C(scheme):
    """S=0, p=0 -> T0 = constants.T_freeze (0 degC) for the liquidus schemes."""
    tf = float(freezing_point(0.0, 0.0, scheme=scheme))
    assert tf == pytest.approx(_T0_K, abs=1e-9)
    assert (tf - _T0_K) == pytest.approx(0.0, abs=1e-9)  # 0 degC


# ---------------------------------------------------------------------------
# the headline value: S=35 -> ~-1.92 C (unesco), NOT -1.8
# ---------------------------------------------------------------------------

def test_unesco_at_S35_is_minus_1p92_C():
    """UNESCO/Millero at S=35, p=0 is ~-1.922 C (271.23 K), not -1.8."""
    tf_K = float(freezing_point(35.0, 0.0, scheme="unesco"))
    tf_C = tf_K - _T0_K
    assert tf_C == pytest.approx(-1.9223, abs=5e-3)
    assert tf_C == pytest.approx(-1.92, abs=1e-2)      # spec target
    # Strictly colder than the old fixed -1.8 C constant.
    assert tf_K < constants.T_freeze_ocean
    # Distinct from the fixed constant by ~0.12 C.
    assert (constants.T_freeze_ocean - tf_K) == pytest.approx(0.12, abs=0.02)


def test_linear_S_slope():
    """linear_S: T_f = T0 - 0.0575*S; at S=35 -> -2.0125 C."""
    tf_C = float(freezing_point(35.0, 0.0, scheme="linear_S")) - _T0_K
    assert tf_C == pytest.approx(-0.0575 * 35.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SIGN gate: monotone decreasing in S; pressure lowers T_f
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_monotone_decreasing_in_salinity(scheme):
    """T_f decreases as S increases (more saline water freezes colder)."""
    S = jnp.array([0.0, 10.0, 20.0, 30.0, 35.0, 40.0])
    tf = np.asarray(freezing_point(S, 0.0, scheme=scheme))
    diffs = np.diff(tf)
    assert np.all(diffs < 0.0), f"{scheme} not strictly decreasing: {tf}"


def test_pressure_lowers_freezing_point_unesco():
    """Deeper (higher p) water freezes colder for the pressure-aware scheme."""
    tf_surface = float(freezing_point(35.0, 0.0, scheme="unesco"))
    tf_deep = float(freezing_point(35.0, 1.0e7, scheme="unesco"))  # 1000 dbar
    assert tf_deep < tf_surface
    # -7.53e-4 degC/dbar * 1000 dbar = -0.753 degC lowering.
    assert (tf_deep - tf_surface) == pytest.approx(-0.753, abs=1e-6)


def test_linear_S_is_pressure_independent():
    """linear_S ignores pressure (surface liquidus, per MOM6 linear form)."""
    a = float(freezing_point(35.0, 0.0, scheme="linear_S"))
    b = float(freezing_point(35.0, 1.0e7, scheme="linear_S"))
    assert a == b


# ---------------------------------------------------------------------------
# differentiability / jit / vmap / negative-S guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_grad_finite_and_negative(scheme):
    """d T_f / d S is finite everywhere (incl. S=0) and negative for S>0.

    At exactly S=0 the ``jnp.maximum(S, 0)`` guard's subgradient is tie-broken by
    JAX, so only FINITENESS is asserted there (the S**1.5 term is grad-safe at 0,
    unlike S*sqrt(S)); strict negativity is checked at strictly-positive S.
    """
    g = jax.grad(lambda s: freezing_point(s, 0.0, scheme=scheme))
    assert np.isfinite(float(g(0.0))), f"{scheme} grad not finite at S=0"
    for S in (1e-6, 5.0, 35.0):
        dg = float(g(S))
        assert np.isfinite(dg), f"{scheme} grad not finite at S={S}"
        assert dg < 0.0, f"{scheme} grad not negative at S={S}: {dg}"
    # linear_S slope is exactly the coefficient.
    if scheme == "linear_S":
        assert float(g(35.0)) == pytest.approx(-0.0575, abs=1e-12)


def test_jit_safe():
    """freezing_point compiles under jit with the scheme static."""
    f = jax.jit(freezing_point, static_argnames=("scheme",))
    tf = f(35.0, 0.0, scheme="unesco")
    assert float(tf) == pytest.approx(
        float(freezing_point(35.0, 0.0, scheme="unesco")), abs=1e-12
    )


def test_vmap_safe():
    """vmap over a salinity vector matches the direct broadcast."""
    S = jnp.array([30.0, 33.0, 35.0, 37.0])
    direct = freezing_point(S, 0.0, scheme="unesco")
    vmapped = jax.vmap(lambda s: freezing_point(s, 0.0, scheme="unesco"))(S)
    assert np.allclose(np.asarray(direct), np.asarray(vmapped), atol=1e-12)


@pytest.mark.parametrize("scheme", ["linear_S", "unesco"])
def test_negative_salinity_guarded(scheme):
    """A negative-S overshoot is floored at 0 (no NaN, returns T0)."""
    tf = float(freezing_point(-5.0, 0.0, scheme=scheme))
    assert np.isfinite(tf)
    assert tf == pytest.approx(_T0_K, abs=1e-9)  # clamped S=0 -> pure water


# ---------------------------------------------------------------------------
# dispatch hardening + config schema
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises():
    """A typo'd scheme raises ValueError, never silently defaults."""
    with pytest.raises(ValueError, match="Unknown freezing-point scheme"):
        freezing_point(35.0, 0.0, scheme="linearS")  # typo
    with pytest.raises(ValueError):
        freezing_point(35.0, 0.0, scheme="millero")


def test_valid_freeze_schemes_set():
    assert VALID_FREEZE_SCHEMES == frozenset({"constant", "linear_S", "unesco"})
    # Every scheme in the set is actually dispatchable.
    for scheme in VALID_FREEZE_SCHEMES:
        assert np.isfinite(float(freezing_point(35.0, 0.0, scheme=scheme)))


def test_consumer_configs_default_constant():
    """The wired consumer configs default to the byte-identical constant."""
    from legoesm.ocean.simple_ocean import SimpleOceanConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.mpas_config import MPASOceanConfig

    assert SimpleOceanConfig().freezing.scheme == "constant"
    assert LatLonCGridOceanConfig().freezing.scheme == "constant"
    assert MPASOceanConfig().freezing.scheme == "constant"
