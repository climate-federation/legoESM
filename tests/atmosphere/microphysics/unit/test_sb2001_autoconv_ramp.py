"""RULE 5 for the SB2001 autoconversion gate ``q_c >= 1e-6`` (issue #2232).

gSAM M2005 (module_mp_graupel.f90:1806/2843) switches autoconversion on with a
hard step at 1e-6 kg/kg cloud water.  The port's default is the C2 quintic
ramp, 0 at 1e-6 and 1 at ``sb2001_autoconv_ramp_top`` (2e-6 kg/kg, codex+GLM
2026-10-29); ``ramp_top <= 1e-6`` is gSAM's step.  RULE 5's three tests (plus config and
validation checks):

1. hard limit matches gSAM: the existing oracle tests run at ``ramp_top=0``
   (test_sb2001_warm_rain_oracle.py, test_gsam_sb_liquid_oracle.py, 1e-10);
   here the hard value is checked to reproduce the pre-ramp kernel exactly;
2. per-case gap at the default, pinned per kernel-fixture point and per
   whole-call column, plus the quintic shape inside the ramp;
3. ``check_grads`` through the ramp at the default (user 2026-10-29):
   q_c = 1e-6 (1 + x), x in {0, 0.5, 0.9}, q_r = 1e-6, N_c = 1e8 /m^3,
   rho = 1.0, rate normalised by gSAM's rate at the ramp top (codex+GLM).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics._warm_rain import autoconversion_sb2001
from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig,
    SeifertBehengConfig,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/sb2001_warm_rain.npz"
DEFAULT = SeifertBehengConfig._field_defaults["sb2001_autoconv_ramp_top"]
GATE = 1.0e-6      # gSAM's PRC threshold [kg/kg]
RTOL, ATOL = 1.0e-6, 1.0e-12

sys.path.insert(0, str(Path(__file__).parent))
import test_gsam_sb_liquid_oracle as column_oracle  # noqa: E402


def _kernel(ramp_top):
    fx = dict(np.load(FIXTURE))
    args = tuple(jnp.asarray(fx[k]) for k in ("q_c", "q_r", "N_c", "rho"))
    prc, dnr, _ = autoconversion_sb2001(*args, ramp_top=ramp_top)
    return fx, np.asarray(prc), np.asarray(dnr)


def test_defaults_agree_between_configs():
    assert DEFAULT == MorrisonConfig._field_defaults["sb2001_autoconv_ramp_top"] == 2.0e-6


@pytest.mark.parametrize("bad", [-1.0e-6, math.nan, math.inf])
def test_invalid_ramp_top_raises(bad):
    q = jnp.asarray([1.5e-6])
    with pytest.raises(ValueError, match="ramp_top"):
        autoconversion_sb2001(q, q, jnp.asarray([1.0e8]), jnp.asarray([1.0]),
                              ramp_top=bad)


def test_hard_limit_is_the_gsam_step():
    """Every ramp_top <= 1e-6 is the same exact step (oracle-pinned at 0.0)."""
    fx, prc0, dnr0 = _kernel(0.0)
    _, prc1, dnr1 = _kernel(GATE)
    np.testing.assert_array_equal(prc0, prc1)
    np.testing.assert_array_equal(dnr0, dnr1)
    assert np.all(prc0[fx["q_c"] < GATE] == 0.0)
    assert np.all(prc0[fx["q_c"] == GATE] > 0.0)   # gSAM fires AT the threshold


def _rel(a, b):
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def test_default_gap_per_kernel_point():
    """KNOWN DIFFERENCE, per fixture point (relative to the gSAM step): 1.0 at
    the 168 points with q_c == 1e-6 (the ramp is 0 there), exactly 0 at every
    other point (the fixture has no q_c strictly inside (1e-6, 2e-6))."""
    fx, prc_h, dnr_h = _kernel(0.0)
    _, prc_d, dnr_d = _kernel(DEFAULT)
    expected = np.where(fx["q_c"] == GATE, 1.0, 0.0)
    assert int(expected.sum()) == 168
    for d, h in ((prc_d, prc_h), (dnr_d, dnr_h)):
        np.testing.assert_allclose(_rel(d, h), expected, rtol=RTOL, atol=ATOL)


def test_default_gap_per_column_is_zero():
    """KNOWN DIFFERENCE on the whole-call gSAM columns: 0 in every
    (group, column) -- no post-adjustment level lands in [1e-6, 2e-6)."""
    fx = column_oracle._fx()
    for g in column_oracle.GROUPS:
        T, q_v, hyd, pf, _, rho, dz, dt, cfg = column_oracle._inputs(fx, g)
        hard = column_oracle.seifert_beheng_step(T, q_v, hyd, pf, rho, dz, dt, cfg)
        dflt = column_oracle.seifert_beheng_step(
            T, q_v, hyd, pf, rho, dz, dt,
            cfg._replace(sb2001_autoconv_ramp_top=DEFAULT))
        for name, a, b in zip(("T", "q_v", "q_c", "q_r", "N_r", "precip"), hard, dflt):
            gap = np.abs(np.asarray(a) - np.asarray(b)).reshape(len(np.asarray(a)), -1)
            np.testing.assert_allclose(gap.max(axis=1), 0.0, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{g} {name}")


@pytest.mark.parametrize("x", [0.25, 0.5, 0.75])
def test_default_ramp_is_the_quintic_inside(x):
    """Inside (1e-6, 2e-6) the default rate is gSAM's times the C2 quintic
    6x^5 - 15x^4 + 10x^3 (the fixture has no interior q_c, so the 168
    threshold points are moved to q_c = 1e-6 (1 + x), keeping q_r, N_c, rho)."""
    fx = dict(np.load(FIXTURE))
    m = fx["q_c"] == GATE
    q_c = jnp.asarray(fx["q_c"][m] * (1.0 + x))
    q_r, N_c, rho = (jnp.asarray(fx[k][m]) for k in ("q_r", "N_c", "rho"))
    hard = np.asarray(autoconversion_sb2001(q_c, q_r, N_c, rho, ramp_top=0.0)[0])
    dflt = np.asarray(autoconversion_sb2001(q_c, q_r, N_c, rho, ramp_top=DEFAULT)[0])
    assert m.sum() == 168 and np.all(hard > 0.0)
    np.testing.assert_allclose(dflt / hard, 6 * x**5 - 15 * x**4 + 10 * x**3,
                               rtol=1.0e-12, atol=0.0)


@pytest.mark.parametrize("x", [0.0, 0.5, 0.9])
def test_default_ramp_gradient(x):
    """RULE 5 gradient test at the default: q_c = 1e-6 (1 + x) with q_r = 1e-6,
    N_c = 1e8 /m^3, rho = 1.0 (user 2026-10-29).  The raw rate is ~1e-18
    kg/kg/s, below check_grads' absolute tolerance, so it is divided by gSAM's
    hard-step rate at the ramp top (q_c = 2e-6) to make the check bind
    (codex+GLM 2026-10-29).  The C2 ramp has zero slope at x = 0, so the
    derivative is exactly 0 there and non-zero inside."""
    q_r, N_c, rho = jnp.asarray(1.0e-6), jnp.asarray(1.0e8), jnp.asarray(1.0)
    ref = float(autoconversion_sb2001(jnp.asarray(2.0 * GATE), q_r, N_c, rho,
                                      ramp_top=0.0)[0])
    assert ref > 0.0

    def f(x):
        return autoconversion_sb2001(GATE * (1.0 + x), q_r, N_c, rho,
                                     ramp_top=DEFAULT)[0] / ref

    x = jnp.asarray(x)
    check_grads(f, (x,), order=2)
    g = float(jax.grad(f)(x))
    assert math.isfinite(g)
    assert g == 0.0 if float(x) == 0.0 else g != 0.0
