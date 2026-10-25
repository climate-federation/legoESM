"""O11: ``ice_thd_da`` lateral melt against an independent per-cell transcription.

NEMO: ``ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:174-216``; MIN/MAX operand
order from ``icethd_da.o`` (``maxsd`` keeps ``y`` only when ``y > 0``; ``minsd`` keeps
``x`` only when ``x < at_i`` and ``q`` only when ``q < a_i``).  The oracle is CPython
binary64 arithmetic (no FMA, no flush) with glibc ``pow`` through ctypes; it shares
no model code.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.core.nemo_namelist import overlay, parse_namelist  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ice import bitz_lipscomb as bl  # noqa: E402
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as ICE  # noqa: E402

DATA = Path(__file__).parent / "data" / "orca2_sas_ice_expref"
NLAY_I, NLAY_S = 10, 5
RDT = 10800.0
_LIBM = ctypes.CDLL("libm.so.6")
_LIBM.pow.argtypes = (ctypes.c_double, ctypes.c_double)
_LIBM.pow.restype = ctypes.c_double
RT0 = float.fromhex("0x1.1126666666666p+8")  # phycst.f90:50 rt0 = 273.15_wp
RHOI, RHOS = 917.0, 330.0                     # phycst.f90:68-69


@pytest.fixture(autouse=True)
def libm_policy():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


def _deck():
    v = overlay(parse_namelist(DATA / "namelist_ice_ref", "ice"), parse_namelist(DATA / "namelist_ice_cfg", "ice"))
    return v[("ice", "namthd_da", "rn_beta")][0], v[("ice", "namthd_da", "rn_dmin")][0]


def _div(a, b):
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def _oracle_cell(c, beta, dmin, nn_icesal):
    """icethd_da.f90:174-215 for one cell, statement by statement."""
    zastar = _div(1.0, 1.0 - _LIBM.pow(_div(dmin, 300.0), _div(1.0, beta)))
    zs = list(c["sz_i"]) if nn_icesal == 4 else [c["s_i"]] * NLAY_I
    a, at = c["a_i"], c["at_i"]
    out = dict(a_i=a, h_i=c["h_i"], h_s=c["h_s"], sfx_lam=c["sfx_lam"], hfx_thd=c["hfx_thd"], wfx_lam=c["wfx_lam"])
    if not a > 0.0:
        return out
    zdfloe = dmin * _LIBM.pow(_div(zastar, zastar - at), beta)
    zperi = _div(at * math.pi, 0.66 * zdfloe)
    y = c["sst"] - (c["t_bo"] - RT0)
    zwlat = 3.0e-6 * _LIBM.pow(y if y > 0.0 else 0.0, 1.36)
    x = zwlat * zperi * RDT
    zda_tot = x if x < at else at
    q = _div(zda_tot * a, at)
    zda = q if q < a else a
    r1dt, r1i, r1s = 1.0 / RDT, 1.0 / NLAY_I, 1.0 / NLAY_S

    def fsum(xs):
        s = 0.0
        for v in xs:
            s = s + v
        return s

    hi, hs = c["h_i"], c["h_s"]
    out["sfx_lam"] = c["sfx_lam"] + RHOI * zda * r1dt * hi * r1i * fsum(zs)
    out["hfx_thd"] = c["hfx_thd"] - zda * r1dt * (hi * r1i * fsum(c["e_i"]) + hs * r1s * fsum(c["e_s"]))
    out["wfx_lam"] = c["wfx_lam"] + zda * r1dt * (RHOI * hi + RHOS * hs)
    out["a_i"] = a - zda
    if out["a_i"] == 0.0:
        out["h_i"], out["h_s"] = 0.0, 0.0
    return out


EDGE = [  # (name, overrides): each pins one object-code decision
    ("a_zero", dict(a_i=0.0)),
    ("a_negzero", dict(a_i=-0.0)),
    ("a_nan", dict(a_i=math.nan)),
    ("sst_nan_max_gives_zero", dict(sst=math.nan)),
    ("sst_below_freezing", dict(sst=-1.9, t_bo=RT0 - 1.8)),
    ("dT_negzero", dict(sst=-0.0, t_bo=RT0)),
    ("total_melt_zeroes_h", dict(at_i=0.5, a_i=0.5, sst=40.0, t_bo=RT0 - 1.8)),
    ("min_nan_operands", dict(at_i=0.0, a_i=0.3, sst=math.inf)),
]


def _cells(n=257, seed=0):
    rng = np.random.default_rng(seed)
    at = rng.uniform(0.02, 0.997, n)
    cells = []
    for k in range(n):
        cells.append(dict(
            at_i=at[k], a_i=at[k] * rng.uniform(0.05, 1.0), h_i=rng.uniform(0.05, 4.0),
            h_s=rng.uniform(0.0, 0.6), e_i=list(rng.uniform(2.5e8, 3.4e8, NLAY_I) * rng.choice([1.0, 1e-3], NLAY_I)),
            e_s=list(rng.uniform(0.9e8, 1.2e8, NLAY_S)), s_i=rng.uniform(1.0, 12.0),
            sz_i=list(rng.uniform(0.5, 15.0, NLAY_I)), sst=rng.uniform(-2.0, 6.0), t_bo=RT0 + rng.uniform(-2.0, -1.6),
            sfx_lam=rng.normal(0, 1e-3), hfx_thd=rng.normal(0, 50.0), wfx_lam=rng.normal(0, 1e-3)))
    for k, (_, over) in enumerate(EDGE):
        cells[k].update(over)
    return cells


def _arrays(cells):
    return {k: np.array([c[k] for c in cells], dtype=np.float64) for k in cells[0]}


def _call(arr, beta, dmin, nn_icesal):
    return bl.si3_ice_thd_da(**arr, rn_beta=beta, rn_dmin=dmin, rDt_ice=RDT, nn_icesal=nn_icesal, constants=ICE)


def _jit_call(arr, beta, dmin, nn_icesal):
    cpu = jax.devices("cpu")[0]
    arr = {k: jax.device_put(jnp.asarray(v), cpu) for k, v in arr.items()}
    return jax.jit(lambda a, b, d: _call(a, b, d, nn_icesal))(arr, beta, dmin)


def _bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def _mismatch(res, cells, beta, dmin, nn_icesal):
    want = [_oracle_cell(c, beta, dmin, nn_icesal) for c in cells]
    bad = {}
    for field in bl.SI3ThdDa._fields:
        diff = _bits(getattr(res, field)) != _bits([w[field] for w in want])
        if diff.any():
            bad[field] = np.flatnonzero(diff).tolist()
    return bad


@pytest.mark.parametrize("nn_icesal", [4, 2])
def test_bitwise_vs_independent_jit_and_eager(nn_icesal):
    beta, dmin = _deck()
    cells = _cells()
    arr = _arrays(cells)
    assert _mismatch(_jit_call(arr, beta, dmin, nn_icesal), cells, beta, dmin, nn_icesal) == {}
    with jax.disable_jit():
        eager = _call(arr, beta, dmin, nn_icesal)
    jitted = _jit_call(arr, beta, dmin, nn_icesal)
    for field in bl.SI3ThdDa._fields:
        assert np.array_equal(_bits(getattr(eager, field)), _bits(getattr(jitted, field))), field


def test_traced_namelist_scalars_bitwise():
    beta, dmin = _deck()
    cells = _cells(seed=3)
    arr = _arrays(cells)
    res = _jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4)
    assert _mismatch(res, cells, beta, dmin, 4) == {}


def test_inexact_pins():
    beta, dmin = _deck()
    assert (beta, dmin) == (1.0, 8.0)
    assert bl._DA_ZDMAX_M == 300.0 and bl._DA_ZCS.hex() == "0x1.51eb851eb851fp-1"
    assert bl._DA_ZM1_M_S.hex() == "0x1.92a737110e454p-19" and bl._DA_ZM2.hex() == "0x1.5c28f5c28f5c3p+0"
    assert bl._RPI == math.pi and bl._RPI.hex() == "0x1.921fb54442d18p+1"
    assert ICE.T0.hex() == RT0.hex() == constants.T_freeze.hex()
    assert (ICE.rho_ice, ICE.rho_snow) == (RHOI, RHOS)
    zastar = _div(1.0, 1.0 - _LIBM.pow(_div(dmin, 300.0), _div(1.0, beta)))
    assert zastar.hex() == "0x1.070381c0e0703p+0"  # 1/(1-8/300), not 300/292 printed


def test_edge_pins():
    beta, dmin = _deck()
    cells = _cells()
    arr = _arrays(cells)
    res = _jit_call(arr, beta, dmin, 4)
    idx = {name: k for k, (name, _) in enumerate(EDGE)}
    for name in ("a_zero", "a_negzero", "a_nan"):
        k = idx[name]
        for f in bl.SI3ThdDa._fields:
            assert _bits(getattr(res, f))[k] == _bits(arr[f])[k], (name, f)
    k = idx["sst_nan_max_gives_zero"]
    assert float(res.a_i[k]) == arr["a_i"][k] and np.isfinite(float(res.hfx_thd[k]))
    for name in ("total_melt_zeroes_h", "min_nan_operands"):
        k = idx[name]
        assert _bits(res.a_i)[k] == 0 and _bits(res.h_i)[k] == 0 and _bits(res.h_s)[k] == 0, name
    k = idx["sst_below_freezing"]
    assert float(res.a_i[k]) == arr["a_i"][k]


def test_positive_subnormal_is_ice_bit_predicate():
    x = jax.device_put(jnp.asarray([5e-324, 0.0, -0.0, math.nan, -5e-324, 1.0]), jax.devices("cpu")[0])
    got = np.asarray(jax.jit(bl._ieee_gt_zero)(x)).tolist()
    assert got == [True, False, False, False, False, True]
    assert not bool(jax.jit(lambda v: v > 0.0)(x)[0])  # XLA CPU flushes: why the bit pattern is read


PLANTS = {
    "max_nan_propagating": ("fortran_max", lambda a, b: jnp.maximum(a, b)),
    "min_nan_propagating": ("fortran_min", lambda a, b: jnp.minimum(a, b)),
    "sum_reversed": ("fortran_sum0", lambda t: bl.nemo_source_round(jnp.sum(t[..., ::-1], axis=-1))),
    "no_source_rounding": ("nemo_source_round", lambda x: x),
    "a_eq_zero_never": ("ieee_is_zero", lambda x: jnp.zeros(jnp.shape(x), bool)),
}


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_plants_fire(plant, monkeypatch):
    beta, dmin = _deck()
    cells = _cells()
    arr = _arrays(cells)
    name, fake = PLANTS[plant]
    monkeypatch.setattr(bl, name, fake)
    bad = _mismatch(_jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4), cells, beta, dmin, 4)
    assert bad, f"plant {plant} did not change any bit"


def test_refusals():
    beta, dmin = _deck()
    arr = _arrays(_cells(n=10))
    for k in ("at_i", "sst", "sz_i", "e_s"):
        with pytest.raises(ValueError):
            _call({**arr, k: None}, beta, dmin, 4)
    with pytest.raises(ValueError):
        bl.si3_ice_thd_da(**arr, rn_beta=beta, rn_dmin=None, rDt_ice=RDT, nn_icesal=4, constants=ICE)
    for bad in (0, 5, True, 4.0):
        with pytest.raises(ValueError):
            _call(arr, beta, dmin, bad)
    with pytest.raises(TypeError):
        bl.si3_ice_thd_da(**arr, rn_beta=beta, rn_dmin=dmin, rDt_ice=RDT, nn_icesal=4)


def test_gradients_finite_and_central_difference():
    beta, dmin = _deck()
    cells = _cells(n=40, seed=7)[len(EDGE):]
    arr = {k: jnp.asarray(v) for k, v in _arrays(cells).items()}

    def loss(dmin_, beta_, sst):
        r = bl.si3_ice_thd_da(**{**arr, "sst": sst}, rn_beta=beta_, rn_dmin=dmin_, rDt_ice=RDT,
                              nn_icesal=4, constants=ICE)
        return jnp.sum(r.a_i) + 1e-3 * jnp.sum(r.hfx_thd)

    args = (jnp.float64(dmin), jnp.float64(beta), arr["sst"])
    grads = jax.grad(loss, argnums=(0, 1, 2))(*args)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    for i, eps in ((0, 1e-4), (1, 1e-6)):
        up = list(args); dn = list(args)
        up[i] = args[i] + eps; dn[i] = args[i] - eps
        fd = (loss(*up) - loss(*dn)) / (2 * eps)
        assert abs(float(grads[i]) - float(fd)) <= 1e-5 * max(1.0, abs(float(fd))), (i, grads[i], fd)
    assert abs(float(grads[0])) > 0 and abs(float(grads[1])) > 0
    # Edge cells (below freezing, a_i = 0, total melt) keep finite gradients too.
    edge = {k: jnp.asarray(v) for k, v in _arrays(_cells(n=len(EDGE) + 2)).items()}
    edge = {**edge, "sst": jnp.where(jnp.isfinite(edge["sst"]), edge["sst"], 1.0),
            "a_i": jnp.where(jnp.isnan(edge["a_i"]), 0.2, edge["a_i"]),
            "at_i": jnp.maximum(edge["at_i"], 0.3)}
    g = jax.grad(lambda d: jnp.sum(bl.si3_ice_thd_da(**edge, rn_beta=beta, rn_dmin=d, rDt_ice=RDT,
                                                       nn_icesal=4, constants=ICE).a_i))(jnp.float64(dmin))
    assert np.isfinite(float(g))


def test_zdmax_division_kept_a_division(monkeypatch):
    """XLA rewrites ``x / 300`` as ``x * (1/300)``; at rn_dmin = 6.2 (in the namelist range 6-10) zastar moves."""
    beta = _deck()[0]
    dmin = 6.2
    assert 1 / (1 - dmin * (1 / 300.0)) != 1 / (1 - dmin / 300.0)
    cells = _cells(seed=5)
    arr = _arrays(cells)
    assert _mismatch(_jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4), cells, beta, dmin, 4) == {}
    monkeypatch.setattr(bl, "_da_zastar", lambda d, b: 1.0 / (1.0 - bl.transcendentals.pow(d / 300.0, 1.0 / b)))
    bad = _mismatch(_jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4), cells, beta, dmin, 4)
    assert bad, "bare 300 constant did not change any bit"
