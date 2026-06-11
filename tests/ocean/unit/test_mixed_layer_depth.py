"""Unit tests for the density-threshold mixed-layer-depth diagnostic.

Validates ``ocean/diagnostics.mixed_layer_depth`` (de Boyer Montegut / Treguier
2023 OMIP method) on analytic columns with a controllable EOS so the potential
density profile is exact: ``eos_fn = lambda T,S,p: 1000 + T`` makes T itself the
potential-density anomaly sigma_theta, giving closed-form expected MLDs.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.ocean.diagnostics import mixed_layer_depth

# Controllable EOS: rho = 1000 + T  ->  sigma_theta = T.  S, p ignored.
_EOS_SIGMA_IS_T = lambda T, S, p: 1000.0 + T  # noqa: E731

_Z = np.array([5.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0])  # level centres [m]


def _mld(sigma_profile, **kw):
    """MLD for a single column whose sigma_theta == sigma_profile."""
    T = np.asarray(sigma_profile, dtype=np.float64)[None, :]   # (1, nlev)
    S = np.zeros_like(T)
    out = mixed_layer_depth(T, S, _Z, eos_fn=_EOS_SIGMA_IS_T, **kw)
    return float(np.asarray(out)[0])


def test_linear_stratification_closed_form():
    """sigma = a*z  ->  sigma_ref = a*10, crossing dsigma=delta at z=10+delta/a."""
    a = 0.001
    sigma = a * _Z
    # delta=0.03 -> z = 10 + 0.03/0.001 = 40 m
    assert _mld(sigma, delta_sigma=0.03, ref_depth_m=10.0) == pytest.approx(40.0, abs=1e-6)
    # delta=0.01 -> z = 10 + 10 = 20 m
    assert _mld(sigma, delta_sigma=0.01, ref_depth_m=10.0) == pytest.approx(20.0, abs=1e-6)


def test_threshold_0p03_is_deeper_than_0p01():
    """Codex BLOCKER #1: a LARGER density threshold gives a DEEPER MLD."""
    a = 0.001
    sigma = a * _Z
    mld_03 = _mld(sigma, delta_sigma=0.03)
    mld_01 = _mld(sigma, delta_sigma=0.01)
    assert mld_03 >= mld_01
    assert mld_03 == pytest.approx(40.0, abs=1e-6)
    assert mld_01 == pytest.approx(20.0, abs=1e-6)


def test_two_layer_step():
    """Step from 0 to 0.06 between z=30 and z=60: dsigma crosses delta=0.03 at the
    interpolated midpoint -> MLD = 45 m.  (A step landing bit-exactly ON the
    threshold is float-fragile via the 1000+T eos trick and measure-zero for real
    sigma, so this tests the robust interpolated crossing instead.)"""
    sigma = np.array([0.0, 0.0, 0.0, 0.06, 0.06, 0.06, 0.06])  # step at idx3 (z=60)
    # crossing between (z=30, dsig=0) and (z=60, dsig=0.06): 30 + 30*(0.03/0.06) = 45
    assert _mld(sigma, delta_sigma=0.03) == pytest.approx(45.0, abs=1e-6)


def test_fully_mixed_returns_bottom():
    """Uniform sigma -> no crossing -> MLD = bottom_depth."""
    sigma = np.full(_Z.shape, 25.0)
    assert _mld(sigma, delta_sigma=0.03, bottom_depth=np.array([4000.0])) == \
        pytest.approx(4000.0)
    # Without bottom_depth -> deepest wet level centre.
    assert _mld(sigma, delta_sigma=0.03) == pytest.approx(float(_Z[-1]))


def test_virtual_reference_point_crossing_just_below_ref():
    """Crossing between the 10 m virtual point and the first level below it
    must interpolate from (10 m, 0), not from a level shallower than 10 m."""
    # sigma flat 0 above, then jumps so the crossing lies between ref(10) and z=15.
    # sigma_ref interp(z=5:0, z=15:0.2)@10 = 0.1; dsig(z=15)=0.2-0.1=0.1.
    sigma = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # delta=0.05: lower bracket = virtual (10, 0), upper = (15, dsig=0.1).
    # z = 10 + (15-10)*(0.05-0)/(0.1-0) = 10 + 5*0.5 = 12.5 m
    assert _mld(sigma, delta_sigma=0.05, ref_depth_m=10.0) == pytest.approx(12.5, abs=1e-6)


def test_all_false_does_not_return_level_zero():
    """Codex BLOCKER #3: no exceedance must give bottom, never z[0]."""
    sigma = np.linspace(0.0, 0.005, _Z.size)  # never reaches 0.03
    out = _mld(sigma, delta_sigma=0.03, bottom_depth=np.array([3000.0]))
    assert out == pytest.approx(3000.0)
    assert out != pytest.approx(float(_Z[0]))


def test_shallow_column_bottom_above_ref():
    """Sea floor above 10 m -> whole column is the mixed layer -> MLD=bottom."""
    sigma = np.linspace(0.0, 0.1, _Z.size)
    assert _mld(sigma, delta_sigma=0.03, ref_depth_m=10.0,
                bottom_depth=np.array([6.0])) == pytest.approx(6.0)


def test_land_column_is_nan():
    sigma = np.linspace(0.0, 0.1, _Z.size)
    T = sigma[None, :]
    S = np.zeros_like(T)
    wet = np.zeros_like(T)   # fully dry
    out = mixed_layer_depth(T, S, _Z, eos_fn=_EOS_SIGMA_IS_T, wet_mask=wet,
                            bottom_depth=np.array([0.0]))
    assert np.isnan(float(np.asarray(out)[0]))


def test_wet_mask_excludes_dry_levels_from_search():
    """A dense value on a DRY level must not register as the MLD crossing."""
    # Dense spike at idx5 (z=250) but that level is dry; real crossing at idx2.
    sigma = np.array([0.0, 0.0, 0.05, 0.05, 0.05, 99.0, 99.0])
    wet = np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.float64)
    T = sigma[None, :]; S = np.zeros_like(T)
    out = float(np.asarray(mixed_layer_depth(
        T, S, _Z, eos_fn=_EOS_SIGMA_IS_T, delta_sigma=0.03,
        wet_mask=wet[None, :], bottom_depth=np.array([200.0])))[0])
    # crossing at idx2 (z=30) region, NOT the dry spike at 250.
    assert out < 60.0


def test_batched_shapes():
    """Vectorised over a 2-D horizontal field."""
    a = 0.001
    sigma = a * _Z
    T = np.broadcast_to(sigma, (4, 3, _Z.size)).copy()
    S = np.zeros_like(T)
    out = np.asarray(mixed_layer_depth(T, S, _Z, eos_fn=_EOS_SIGMA_IS_T,
                                       delta_sigma=0.03))
    assert out.shape == (4, 3)
    assert np.allclose(out, 40.0, atol=1e-6)


def test_wright_eos_default_smoke():
    """Default nonlinear wright_eos path runs and gives a physical MLD on a
    realistic warm-surface / cool-deep column (stable -> finite MLD)."""
    # Warm fresh surface over cool salty deep -> stable, shallow ML.
    z = np.array([5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 800.0])
    T = np.array([20.0, 20.0, 19.5, 15.0, 10.0, 6.0, 4.0])[None, :]
    S = np.array([35.0, 35.0, 35.0, 35.1, 35.2, 35.3, 35.4])[None, :]
    out = float(np.asarray(mixed_layer_depth(
        T, S, z, delta_sigma=0.03, bottom_depth=np.array([3000.0])))[0])
    assert 10.0 <= out <= 120.0   # physical surface mixed layer
