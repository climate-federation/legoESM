"""O8: ``ice_thd_frazil`` with collection off, against an independent per-cell transcription.

NEMO (``ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:419-426``, object code
``comisd``/``ja`` then ``pxor``): ``fraz_frac = +0``; ``ht_i_new = rn_hinew`` where
``0 > qlead`` (ordered, so ``-0`` and NaN are not cooling), else ``+0``.  The oracle
below is a CPython float compare, which has no flush-to-zero.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.nemo_namelist import overlay, parse_namelist  # noqa: E402
from legoesm.ice import bitz_lipscomb as bl  # noqa: E402

DATA = Path(__file__).parent / "data" / "orca2_sas_ice_expref"
SPECIAL = [-5e-324, -2.2e-308, -0.0, 0.0, 5e-324, math.nan, -math.nan, math.inf, -math.inf, -1.0e-300]


def _deck():
    values = overlay(parse_namelist(DATA / "namelist_ice_ref", "ice"), parse_namelist(DATA / "namelist_ice_cfg", "ice"))
    return values[("ice", "namthd_do", "rn_hinew")][0], values[("ice", "namthd_do", "ln_frazil")][0]


def _qlead(shape=(7, 5), seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 300.0, size=shape)
    q.flat[: len(SPECIAL)] = SPECIAL
    return q


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _oracle(q, rn_hinew):
    ht = np.array([rn_hinew if float(v) < 0.0 else 0.0 for v in np.ravel(q)]).reshape(np.shape(q))
    return ht, np.zeros(np.shape(q))


def _mismatches(res, q, rn_hinew):
    ht, ff = _oracle(q, rn_hinew)
    return int(np.sum(_bits(res.ht_i_new) != _bits(ht)) + np.sum(_bits(res.fraz_frac) != _bits(ff)))


def _jit(q, rn_hinew, ln_frazil=False):
    return jax.jit(lambda x: bl.si3_ice_thd_frazil(x, rn_hinew=rn_hinew, ln_frazil=ln_frazil))(jnp.asarray(q))


def test_production_deck_resolves_hinew_and_collection_off():
    rn_hinew, ln_frazil = _deck()
    assert ln_frazil is False
    assert struct.pack("<d", rn_hinew).hex() == struct.pack("<d", 0.1).hex()
    assert Fraction(rn_hinew) != Fraction(1, 10)                     # inexact binary64 operand


@pytest.mark.parametrize("shape", [(7, 5), (182, 149), (3,), ()])
def test_jit_equals_oracle_bitwise_including_signed_zero_nan_and_subnormal(shape):
    rn_hinew, ln_frazil = _deck()
    q = _qlead(shape) if shape != () else np.float64(-5e-324)
    res = _jit(q, rn_hinew, ln_frazil)
    assert res.ht_i_new.shape == np.shape(q) and res.ht_i_new.dtype == jnp.float64
    assert _mismatches(res, q, rn_hinew) == 0


def test_inexact_value_pins():
    rn_hinew, _ = _deck()
    res = _jit(_qlead(), rn_hinew)
    ht = _bits(res.ht_i_new).ravel()
    assert ht[0] == 0x3FB999999999999A                               # -5e-324 is cooling (NEMO: no DAZ)
    assert ht[2] == 0 and ht[5] == 0 and ht[6] == 0                  # -0, NaN, -NaN: +0, not -0
    assert ht[8] == 0x3FB999999999999A and ht[9] == 0x3FB999999999999A
    assert np.all(_bits(res.fraz_frac) == 0)


def test_eager_disable_jit_equals_jit():
    rn_hinew, _ = _deck()
    q = jnp.asarray(_qlead())
    jitted = _jit(q, rn_hinew)
    with jax.disable_jit():
        eager = bl.si3_ice_thd_frazil(q, rn_hinew=rn_hinew, ln_frazil=False)
    for a, b in zip(jitted, eager):
        assert np.array_equal(_bits(a), _bits(b))


@pytest.mark.parametrize("plant", ["float_compare", "le_zero", "nan_cooling", "hinew_ulp"])
def test_production_path_plants_fire(plant, monkeypatch):
    rn_hinew, _ = _deck()
    q = _qlead()
    used = rn_hinew
    if plant == "float_compare":                                      # XLA CPU flushes -5e-324 under jit
        monkeypatch.setattr(bl, "_ieee_lt_zero", lambda x: x < 0.0)
    elif plant == "le_zero":
        monkeypatch.setattr(bl, "_ieee_lt_zero", lambda x: x <= 0.0)
    elif plant == "nan_cooling":
        monkeypatch.setattr(bl, "_ieee_lt_zero", lambda x: ~(x >= 0.0))
    elif plant == "hinew_ulp":
        used = math.nextafter(rn_hinew, 1.0)
    assert _mismatches(_jit(q, used), q, rn_hinew) > 0


def test_refuses_missing_inputs_and_the_collection_arm():
    rn_hinew, _ = _deck()
    q = _qlead()
    with pytest.raises(TypeError):
        bl.si3_ice_thd_frazil(q, rn_hinew=rn_hinew)                  # no default for ln_frazil
    with pytest.raises(TypeError):
        bl.si3_ice_thd_frazil(q, ln_frazil=False)                     # no default for rn_hinew
    for kw in ({"rn_hinew": None, "ln_frazil": False}, {"rn_hinew": rn_hinew, "ln_frazil": True},
               {"rn_hinew": rn_hinew, "ln_frazil": 0}):
        with pytest.raises(ValueError):
            bl.si3_ice_thd_frazil(q, **kw)
    with pytest.raises(ValueError):
        bl.si3_ice_thd_frazil(None, rn_hinew=rn_hinew, ln_frazil=False)


def test_gradients_finite_and_central_difference_on_rn_hinew():
    rn_hinew, _ = _deck()
    q = jnp.asarray(_qlead())
    loss = lambda h, x: jnp.sum(bl.si3_ice_thd_frazil(x, rn_hinew=h, ln_frazil=False).ht_i_new ** 2)  # noqa: E731
    g_h, g_q = jax.jit(jax.grad(loss, argnums=(0, 1)))(rn_hinew, q)
    assert np.isfinite(g_h) and np.all(np.isfinite(np.asarray(g_q)))
    eps = 1e-6
    fd = (loss(rn_hinew + eps, q) - loss(rn_hinew - eps, q)) / (2 * eps)
    assert abs(float(g_h) / float(fd) - 1.0) < 1e-6
    n_cool = sum(1 for v in np.ravel(np.asarray(q)) if float(v) < 0.0)
    assert abs(float(g_h) - 2 * rn_hinew * n_cool) < 1e-9
