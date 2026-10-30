"""RULE 5 for the two SB2001 accretion gates ``q_r >= 1e-8`` and
``q_c >= 1e-8`` kg/kg (issue #2262).

gSAM M2005 (module_mp_graupel.f90:1944/3266) computes accretion only where
both rain and cloud water reach 1e-8 kg/kg, with hard steps.  The port's
default replaces each by the C2 quintic ramp, 0 at 1e-8 and 1 at
``sb2001_accretion_qr_ramp_top`` / ``sb2001_accretion_qc_ramp_top`` (2e-8 kg/kg
each, codex+GLM 2026-10-30); a top <= 1e-8 is gSAM's step.  RULE 5's three
tests per gate (plus config and validation checks):

1. hard limit matches gSAM: the existing oracle tests run at tops = 0
   (test_sb2001_warm_rain_oracle.py, test_gsam_sb_liquid_oracle.py, 1e-10);
   here every top <= 1e-8 is checked to give that same exact step;
2. per-case gap at the default, pinned per kernel-fixture point and per
   whole-call column, plus the quintic shape inside each ramp;
3. ``check_grads`` through each ramp at the default: the gated variable at
   1e-8 (1 + x), x in {0, 0.5, 0.9}, the other at 1e-4, rho = 1.0, rate
   normalised by gSAM's rate at the ramp top (codex+GLM 2026-10-30).
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

from legoesm.atmosphere.physics.microphysics._warm_rain import accretion_sb2001
from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig,
    SeifertBehengConfig,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/sb2001_warm_rain.npz"
FIELDS = ("sb2001_accretion_qr_ramp_top", "sb2001_accretion_qc_ramp_top")
DEFAULT = dict(qr_ramp_top=SeifertBehengConfig._field_defaults[FIELDS[0]],
               qc_ramp_top=SeifertBehengConfig._field_defaults[FIELDS[1]])
HARD = dict(qr_ramp_top=0.0, qc_ramp_top=0.0)
GATE = 1.0e-8      # gSAM's PRA thresholds [kg/kg]
RTOL, ATOL = 1.0e-6, 1.0e-12

sys.path.insert(0, str(Path(__file__).parent))
import test_gsam_sb_liquid_oracle as column_oracle  # noqa: E402


def _fx():
    return dict(np.load(FIXTURE))


def _kernel(fx, tops):
    q_c, q_r, rho = (jnp.asarray(fx[k]) for k in ("q_c", "q_r", "rho"))
    return np.asarray(accretion_sb2001(q_c, q_r, rho, **tops))


def test_defaults_agree_between_configs():
    for f in FIELDS:
        assert (SeifertBehengConfig._field_defaults[f]
                == MorrisonConfig._field_defaults[f] == 2.0e-8)


@pytest.mark.parametrize("which", ["qr_ramp_top", "qc_ramp_top"])
@pytest.mark.parametrize("bad", [-1.0e-8, math.nan, math.inf])
def test_invalid_ramp_top_raises(which, bad):
    q = jnp.asarray([1.5e-8])
    with pytest.raises(ValueError, match=which):
        accretion_sb2001(q, q, jnp.asarray([1.0]), **{**DEFAULT, which: bad})


def test_hard_limit_is_the_gsam_step():
    """Every top <= 1e-8 is the same exact step (oracle-pinned at 0.0)."""
    fx = _fx()
    pra0 = _kernel(fx, HARD)
    np.testing.assert_array_equal(
        pra0, _kernel(fx, dict(qr_ramp_top=GATE, qc_ramp_top=GATE)))
    off = (fx["q_r"] < GATE) | (fx["q_c"] < GATE)
    assert np.all(pra0[off] == 0.0)
    at = (fx["q_r"] == GATE) & (fx["q_c"] >= GATE)
    assert at.sum() == 192 and np.all(pra0[at] > 0.0)  # gSAM fires AT the gate


def _rel(a, b):
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def test_default_gap_per_kernel_point():
    """KNOWN DIFFERENCE, per fixture point (relative to the gSAM step): 1.0 at
    the 192 points with q_r == 1e-8 and q_c >= 1e-8 (the rain ramp is 0
    there; largest absolute change 1.4e-13 kg/kg/s), exactly 0 at every other
    point (no fixture q_r or q_c lies strictly inside (1e-8, 2e-8))."""
    fx = _fx()
    hard, dflt = _kernel(fx, HARD), _kernel(fx, DEFAULT)
    expected = np.where((fx["q_r"] == GATE) & (fx["q_c"] >= GATE), 1.0, 0.0)
    assert int(expected.sum()) == 192
    np.testing.assert_allclose(_rel(dflt, hard), expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.abs(dflt - hard).max(), 1.435e-13, rtol=1.0e-3)


def test_default_gap_per_column_is_zero():
    """KNOWN DIFFERENCE on the whole-call gSAM columns: 0 in every
    (group, column) -- no accretion input lands in [1e-8, 2e-8)."""
    fx = column_oracle._fx()
    for g in column_oracle.GROUPS:
        T, q_v, hyd, pf, _, rho, dz, dt, cfg = column_oracle._inputs(fx, g)
        hard = column_oracle.seifert_beheng_step(T, q_v, hyd, pf, rho, dz, dt, cfg)
        dflt = column_oracle.seifert_beheng_step(
            T, q_v, hyd, pf, rho, dz, dt,
            cfg._replace(**{f: SeifertBehengConfig._field_defaults[f] for f in FIELDS}))
        for name, a, b in zip(("T", "q_v", "q_c", "q_r", "N_r", "precip"), hard, dflt):
            gap = np.abs(np.asarray(a) - np.asarray(b)).reshape(len(np.asarray(a)), -1)
            np.testing.assert_allclose(gap.max(axis=1), 0.0, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{g} {name}")


def _ramp_points(fx, gated):
    """Fixture points to move into one gate's ramp, with the OTHER mixing ratio
    above its own ramp top (codex 2026-10-30): the 192 q_r == 1e-8 points for
    the rain gate; for the cloud gate (no fixture q_c near 1e-8) the points
    with q_c == 5e-7 (the smallest q_c above the gate) and q_r >= 2e-8."""
    if gated == "q_r":
        m = (fx["q_r"] == GATE) & (fx["q_c"] >= 2.0 * GATE)
    else:
        m = (fx["q_c"] == 5.0e-7) & (fx["q_r"] >= 2.0 * GATE)
    assert m.sum() > 0
    return m


@pytest.mark.parametrize("gated", ["q_r", "q_c"])
@pytest.mark.parametrize("x", [0.25, 0.5, 0.75])
def test_default_ramp_is_the_quintic_inside(gated, x):
    """Inside (1e-8, 2e-8) the default rate is gSAM's times the C2 quintic
    6x^5 - 15x^4 + 10x^3."""
    fx = _fx()
    m = _ramp_points(fx, gated)
    q = {k: jnp.asarray(fx[k][m]) for k in ("q_c", "q_r", "rho")}
    q[gated] = jnp.full_like(q[gated], GATE * (1.0 + x))
    hard = np.asarray(accretion_sb2001(q["q_c"], q["q_r"], q["rho"], **HARD))
    dflt = np.asarray(accretion_sb2001(q["q_c"], q["q_r"], q["rho"], **DEFAULT))
    assert np.all(hard > 0.0)
    np.testing.assert_allclose(dflt / hard, 6 * x**5 - 15 * x**4 + 10 * x**3,
                               rtol=1.0e-12, atol=0.0)


@pytest.mark.parametrize("gated", ["q_r", "q_c"])
@pytest.mark.parametrize("x", [0.0, 0.5, 0.9])
def test_default_ramp_gradient(gated, x):
    """RULE 5 gradient test at the default: the gated variable at
    1e-8 (1 + x), the other at 1e-4, rho = 1.0, the rate divided by gSAM's
    hard-step rate at the ramp top (2e-8) so check_grads binds (codex+GLM
    2026-10-30).  The C2 ramp has zero slope at x = 0, so the derivative is
    exactly 0 there and non-zero inside."""
    other, rho = jnp.asarray(1.0e-4), jnp.asarray(1.0)

    def pra(q, tops):
        q_c, q_r = (q, other) if gated == "q_c" else (other, q)
        return accretion_sb2001(q_c, q_r, rho, **tops)

    ref = float(pra(jnp.asarray(2.0 * GATE), HARD))
    assert ref > 0.0

    def f(x):
        return pra(GATE * (1.0 + x), DEFAULT) / ref

    x = jnp.asarray(x)
    check_grads(f, (x,), order=2)
    g = float(jax.grad(f)(x))
    assert math.isfinite(g)
    assert g == 0.0 if float(x) == 0.0 else g != 0.0
