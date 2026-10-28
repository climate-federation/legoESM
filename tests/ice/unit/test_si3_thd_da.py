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

from legoesm.core.nemo_namelist import overlay, parse_namelist  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ice import si3_thd_da as bl  # noqa: E402
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as ICE  # noqa: E402

from legoesm import constants  # noqa: E402

DATA = Path(__file__).parent / "data" / "orca2_sas_ice_expref"
NLAY_I, NLAY_S = 10, 5
RDT = 10800.0
_LIBM = ctypes.CDLL("libm.so.6")
_LIBM.pow.argtypes = (ctypes.c_double, ctypes.c_double)
_LIBM.pow.restype = ctypes.c_double
RT0 = float.fromhex("0x1.1126666666666p+8")  # phycst.f90:50 rt0 = 273.15_wp
RHOI, RHOS = 917.0, 330.0  # phycst.f90:68-69


@pytest.fixture(autouse=True)
def libm_policy():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(previous)


def _deck():
    v = overlay(
        parse_namelist(DATA / "namelist_ice_ref", "ice"),
        parse_namelist(DATA / "namelist_ice_cfg", "ice"),
    )
    return v[("ice", "namthd_da", "rn_beta")][0], v[("ice", "namthd_da", "rn_dmin")][0]


def _div(a, b):
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def _oracle_cell(c, beta, dmin, nn_icesal, rdt=RDT):
    """icethd_da.f90:174-215 for one cell, statement by statement."""
    nlay_i, nlay_s = len(c["e_i"]), len(c["e_s"])
    zastar = _div(1.0, 1.0 - _LIBM.pow(_div(dmin, 300.0), _div(1.0, beta)))
    zs = list(c["sz_i"]) if nn_icesal == 4 else [c["s_i"]] * nlay_i
    a, at = c["a_i"], c["at_i"]
    out = dict(
        a_i=a,
        h_i=c["h_i"],
        h_s=c["h_s"],
        sfx_lam=c["sfx_lam"],
        hfx_thd=c["hfx_thd"],
        wfx_lam=c["wfx_lam"],
    )
    if not a > 0.0:
        return out
    zdfloe = dmin * _LIBM.pow(_div(zastar, zastar - at), beta)
    zperi = _div(at * math.pi, 0.66 * zdfloe)
    y = c["sst"] - (c["t_bo"] - RT0)
    zwlat = 3.0e-6 * _LIBM.pow(y if y > 0.0 else 0.0, 1.36)
    x = zwlat * zperi * rdt
    zda_tot = x if x < at else at
    q = _div(zda_tot * a, at)
    zda = q if q < a else a
    r1dt, r1i, r1s = 1.0 / rdt, 1.0 / nlay_i, 1.0 / nlay_s

    def fsum(xs):
        s = 0.0
        for v in xs:
            s = s + v
        return s

    hi, hs = c["h_i"], c["h_s"]
    out["sfx_lam"] = c["sfx_lam"] + RHOI * zda * r1dt * hi * r1i * fsum(zs)
    out["hfx_thd"] = c["hfx_thd"] - zda * r1dt * (
        hi * r1i * fsum(c["e_i"]) + hs * r1s * fsum(c["e_s"])
    )
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
        cells.append(
            dict(
                at_i=at[k],
                a_i=at[k] * rng.uniform(0.05, 1.0),
                h_i=rng.uniform(0.05, 4.0),
                h_s=rng.uniform(0.0, 0.6),
                e_i=list(rng.uniform(2.5e8, 3.4e8, NLAY_I) * rng.choice([1.0, 1e-3], NLAY_I)),
                e_s=list(rng.uniform(0.9e8, 1.2e8, NLAY_S)),
                s_i=rng.uniform(1.0, 12.0),
                sz_i=list(rng.uniform(0.5, 15.0, NLAY_I)),
                sst=rng.uniform(-2.0, 6.0),
                t_bo=RT0 + rng.uniform(-2.0, -1.6),
                sfx_lam=rng.normal(0, 1e-3),
                hfx_thd=rng.normal(0, 50.0),
                wfx_lam=rng.normal(0, 1e-3),
            )
        )
    for k, (_, over) in enumerate(EDGE):
        cells[k].update(over)
    return cells


def _arrays(cells):
    return {k: np.array([c[k] for c in cells], dtype=np.float64) for k in cells[0]}


def _call(arr, beta, dmin, nn_icesal):
    return bl.si3_ice_thd_da(
        **arr, rn_beta=beta, rn_dmin=dmin, rDt_ice=RDT, nn_icesal=nn_icesal, constants=ICE
    )


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
    assert (
        bl._DA_ZM1_M_S.hex() == "0x1.92a737110e454p-19"
        and bl._DA_ZM2.hex() == "0x1.5c28f5c28f5c3p+0"
    )
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
    x = jax.device_put(
        jnp.asarray([5e-324, 0.0, -0.0, math.nan, -5e-324, 1.0]), jax.devices("cpu")[0]
    )
    got = np.asarray(jax.jit(bl._ieee_gt_zero)(x)).tolist()
    assert got == [True, False, False, False, False, True]
    assert not bool(
        jax.jit(lambda v: v > 0.0)(x)[0]
    )  # XLA CPU flushes: why the bit pattern is read


PLANTS = {
    "max_nan_propagating": ("fortran_max", lambda a, b: jnp.maximum(a, b)),
    "min_nan_propagating": ("fortran_min", lambda a, b: jnp.minimum(a, b)),
    "sum_reversed": (
        "fortran_sum0",
        lambda t: bl.nemo_source_round(jnp.sum(t[..., ::-1], axis=-1)),
    ),
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
        bl.si3_ice_thd_da(
            **arr, rn_beta=beta, rn_dmin=None, rDt_ice=RDT, nn_icesal=4, constants=ICE
        )
    for bad in (0, 5, True, 4.0):
        with pytest.raises(ValueError):
            _call(arr, beta, dmin, bad)
    with pytest.raises(TypeError):
        bl.si3_ice_thd_da(**arr, rn_beta=beta, rn_dmin=dmin, rDt_ice=RDT, nn_icesal=4)


def test_gradients_finite_and_central_difference():
    beta, dmin = _deck()
    cells = _cells(n=40, seed=7)[len(EDGE) :]
    arr = {k: jnp.asarray(v) for k, v in _arrays(cells).items()}

    def loss(dmin_, beta_, sst):
        r = bl.si3_ice_thd_da(
            **{**arr, "sst": sst},
            rn_beta=beta_,
            rn_dmin=dmin_,
            rDt_ice=RDT,
            nn_icesal=4,
            constants=ICE,
        )
        return jnp.sum(r.a_i) + 1e-3 * jnp.sum(r.hfx_thd)

    args = (jnp.float64(dmin), jnp.float64(beta), arr["sst"])
    grads = jax.grad(loss, argnums=(0, 1, 2))(*args)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    for i, eps in ((0, 1e-4), (1, 1e-6)):
        up, dn = list(args), list(args)
        up[i], dn[i] = args[i] + eps, args[i] - eps
        fd = (loss(*up) - loss(*dn)) / (2 * eps)
        assert abs(float(grads[i]) - float(fd)) <= 1e-5 * max(1.0, abs(float(fd))), (
            i,
            grads[i],
            fd,
        )
    assert abs(float(grads[0])) > 0 and abs(float(grads[1])) > 0
    # Edge cells (below freezing, a_i = 0, total melt) keep finite gradients too.
    edge = {k: jnp.asarray(v) for k, v in _arrays(_cells(n=len(EDGE) + 2)).items()}
    edge = {
        **edge,
        "sst": jnp.where(jnp.isfinite(edge["sst"]), edge["sst"], 1.0),
        "a_i": jnp.where(jnp.isnan(edge["a_i"]), 0.2, edge["a_i"]),
        "at_i": jnp.maximum(edge["at_i"], 0.3),
    }
    g = jax.grad(
        lambda d: jnp.sum(
            bl.si3_ice_thd_da(
                **edge, rn_beta=beta, rn_dmin=d, rDt_ice=RDT, nn_icesal=4, constants=ICE
            ).a_i
        )
    )(jnp.float64(dmin))
    assert np.isfinite(float(g))


def test_zdmax_division_kept_a_division(monkeypatch):
    """XLA turns ``x / 300`` into ``x * (1/300)``; at rn_dmin = 6.2 that moves zastar."""
    beta = _deck()[0]
    dmin = 6.2
    assert 1 / (1 - dmin * (1 / 300.0)) != 1 / (1 - dmin / 300.0)
    cells = _cells(seed=5)
    arr = _arrays(cells)
    assert (
        _mismatch(_jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4), cells, beta, dmin, 4)
        == {}
    )
    monkeypatch.setattr(
        bl, "_da_zastar", lambda d, b: 1.0 / (1.0 - bl.transcendentals.pow(d / 300.0, 1.0 / b))
    )
    bad = _mismatch(_jit_call(arr, jnp.float64(beta), jnp.float64(dmin), 4), cells, beta, dmin, 4)
    assert bad, "bare 300 constant did not change any bit"


# --- positive-subnormal a_i: the a_i > 0 test at si3_thd_da.py:143 -------------------
SUBNORMAL = [  # at_i = 1; -0 accumulators so a taken branch (+0 increments) changes bits
    ("a_subnormal_warm", dict(at_i=1.0, a_i=5e-324, sst=40.0, t_bo=RT0 - 1.8)),
    ("a_subnormal_cold", dict(at_i=1.0, a_i=5e-324, sst=-1.9, t_bo=RT0 - 1.8)),
]


def _subnormal_cells():
    cells = _cells(n=len(EDGE) + len(SUBNORMAL), seed=11)[len(EDGE) :]
    for c, (_, over) in zip(cells, SUBNORMAL, strict=True):
        c.update(over, sfx_lam=-0.0, hfx_thd=-0.0, wfx_lam=-0.0)
    return cells


def _assert_subnormal_pinned(res, cells, beta, dmin):
    want = [_oracle_cell(c, beta, dmin, 4) for c in cells]
    assert all(w["sfx_lam"].hex() == "0x0.0p+0" for w in want)  # oracle takes the branch
    for f in ("sfx_lam", "hfx_thd", "wfx_lam"):
        assert np.array_equal(_bits(getattr(res, f)), _bits([w[f] for w in want])), f
    # D5_PLATFORM_FLOOR: XLA CPU flushes the subnormal operand inside the branch, so a_i
    # empties (and h_i, h_s are zeroed) where binary64 keeps 5e-324; pinned so it is seen.
    assert all(w["a_i"] == 5e-324 for w in want)
    for f in ("a_i", "h_i", "h_s"):
        assert not _bits(getattr(res, f)).any(), f


def test_positive_subnormal_a_i_takes_the_branch():
    beta, dmin = _deck()
    cells = _subnormal_cells()
    _assert_subnormal_pinned(_jit_call(_arrays(cells), beta, dmin, 4), cells, beta, dmin)


def test_positive_subnormal_float_compare_plant_fires(monkeypatch):
    beta, dmin = _deck()
    cells = _subnormal_cells()
    monkeypatch.setattr(bl, "_ieee_gt_zero", lambda v: v > 0.0)  # line 143 as `a_i > 0`
    with pytest.raises(AssertionError):
        _assert_subnormal_pinned(_jit_call(_arrays(cells), beta, dmin, 4), cells, beta, dmin)


# --- MIN/MAX ties: which operand survives -------------------------------------------
def test_fortran_min_max_ties_keep_the_first_operand():
    """minsd/maxsd as compiled: on a +0/-0 tie the FIRST model operand survives."""
    f = jax.jit(lambda a, b: (bl.fortran_min(a, b), bl.fortran_max(a, b)))
    for a, b in ((0.0, -0.0), (-0.0, 0.0)):
        mn, mx = f(jnp.float64(a), jnp.float64(b))
        assert _bits(mn) == _bits(a) and _bits(mx) == _bits(a), (a, b)


def test_fortran_min_max_tie_plant_fires():
    f = jax.jit(lambda a, b: (jnp.minimum(a, b), jnp.maximum(a, b)))
    kept = [
        _bits(r) == _bits(a)
        for a, b in ((0.0, -0.0), (-0.0, 0.0))
        for r in f(jnp.float64(a), jnp.float64(b))
    ]
    assert not all(kept)


def _record_operands(monkeypatch, swap=()):
    """Eager call recording each MIN/MAX site's operands and result, in call order."""
    calls = []
    real = {"fortran_min": bl.fortran_min, "fortran_max": bl.fortran_max}

    def recorder(name):
        def rec(a, b):
            if name in swap:  # the production line written with its operands reversed
                a, b = b, a
            out = real[name](a, b)
            calls.append((name, _bits(a).copy(), _bits(b).copy(), _bits(out).copy()))
            return out

        return rec

    for name in real:
        monkeypatch.setattr(bl, name, recorder(name))
    beta, dmin = _deck()
    arr = _arrays(_cells())
    with jax.disable_jit():
        res = _call(arr, beta, dmin, 4)
    return calls, arr, res


def _assert_operand_order(calls, arr):
    k = [name for name, _ in EDGE].index("dT_negzero")
    assert [c[0] for c in calls] == ["fortran_max", "fortran_min", "fortran_min"]
    (_, a0, b0, out0), (_, a1, _, _), (_, a2, _, _) = calls
    assert not a0.any() and b0[k] == _bits(-0.0) and out0[k] == 0  # :187 MAX(+0, -0) = +0
    assert np.array_equal(a1, _bits(arr["at_i"]))  # :189 minsd keeps at_i on a tie
    assert np.array_equal(a2, _bits(arr["a_i"]))  # :195 minsd keeps a_i on a tie


def test_min_max_operand_order_at_each_site(monkeypatch):
    calls, arr, _ = _record_operands(monkeypatch)
    _assert_operand_order(calls, arr)


@pytest.mark.parametrize("site", ["fortran_max", "fortran_min"])
def test_min_max_operand_order_plants_fire(monkeypatch, site):
    _, _, base = _record_operands(monkeypatch)
    monkeypatch.undo()
    calls, arr, res = _record_operands(monkeypatch, swap=(site,))
    with pytest.raises(AssertionError):
        _assert_operand_order(calls, arr)
    if site == "fortran_max":  # pow(+-0, 1.36) = +0: the :187 tie cannot reach a stored bit
        k = [name for name, _ in EDGE].index("dT_negzero")
        for f in bl.SI3ThdDa._fields:
            assert _bits(getattr(res, f))[k] == _bits(getattr(base, f))[k], f


# --- glibc pow arm: a firing control -------------------------------------------------
def _static_call(arr, beta, dmin):
    """rn_beta/rn_dmin as compile-time constants, as the production config passes them."""
    cpu = jax.devices("cpu")[0]
    arr = {k: jax.device_put(jnp.asarray(v), cpu) for k, v in arr.items()}
    return jax.jit(lambda a: _call(a, beta, dmin, 4))(arr)


def test_glibc_pow_arm_fires_at_a_constant_exponent():
    """XLA rewrites pow at a constant exponent of 3 and then differs from glibc."""
    cells = _cells(seed=13)
    arr = _arrays(cells)
    assert _mismatch(_static_call(arr, 3.0, 8.0), cells, 3.0, 8.0, 4) == {}
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    bad = _mismatch(_static_call(arr, 3.0, 8.0), cells, 3.0, 8.0, 4)
    assert bad, "native pow matched glibc: the libm arm is not observable"
    # On the deck (rn_beta = 1, exponents 1 and 1.36) the arms agree: not observable there.
    beta, dmin = _deck()
    assert _mismatch(_static_call(arr, beta, dmin), cells, beta, dmin, 4) == {}


# --- production dispatch: step_sea_ice -> si3_column_step_arrays -> ice_thd_da ----------
C1D_ICE_CFG = (
    Path(__file__).parents[2].parent
    / "scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg"
)
# (entry a_i, h_s, sst_m [degC]): warm, cold, NaN sst, total melt, not gathered (a_i <= epsi10),
# then random cells so rounding-only plants reach a stored bit (the column drops the fluxes)
_RNG = np.random.default_rng(17)
PROD_CELLS = [
    (0.80, 0.20, 2.5),
    (0.60, 0.10, -1.9),
    (0.70, 0.30, math.nan),
    (0.02, 0.25, 40.0),
    (1.0e-12, 0.10, 3.0),
] + list(zip(_RNG.uniform(0.05, 0.99, 250), _RNG.uniform(0.0, 0.4, 250),
             _RNG.uniform(-1.9, 6.0, 250), strict=True))


def _production_case():
    from legoesm.core.field import Field
    from legoesm.ice.bitz_lipscomb import (
        SI3SurfaceForcing,
        ice_enthalpy_from_temperature,
        option2_salinity_profile,
        snow_enthalpy_from_temperature,
    )
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.state import SI3ColumnState

    card = build_c1d_omip_l3_card()
    n = len(PROD_CELLS)
    a, hs, sst = (np.array(col, dtype=np.float64) for col in zip(*PROD_CELLS, strict=True))
    hi = np.random.default_rng(18).uniform(0.3, 3.0, n)
    sss = np.full(n, 34.0)
    s_bulk = np.full(n, 5.0)
    s_lay = option2_salinity_profile(jnp.asarray(s_bulk), jnp.asarray(sss))
    ice_c = card.config.ice_constants
    t_ice = jnp.broadcast_to(jnp.linspace(265.0, 271.0, 3), (n, 3))
    e_i = ice_enthalpy_from_temperature(t_ice, s_lay, ice_c)
    e_s = snow_enthalpy_from_temperature(jnp.full((n, 3), 266.0), ice_c)

    def fld(v, layered=False):
        dims = ("column", "layer") if layered else ("column",)
        return Field(jnp.asarray(v, jnp.float64), name="x", dims=dims, units="1")

    state = SI3ColumnState(
        fld(a), fld(hi), fld(hs), fld(np.full(n, 268.0)), fld(e_i, True), fld(e_s, True),
        fld(s_bulk), fld(s_lay, True), fld(np.zeros(n)),
    )
    z = jnp.zeros(n)
    forcing = SI3SurfaceForcing(  # qlead > 0: no open-water growth, so ice_thd_do adds +0
        qns_ice=jnp.full(n, -20.0), qsr_ice=jnp.full(n, 30.0), dqns_ice=jnp.full(n, -5.0),
        qtr_ice_top=z, t_bottom=jnp.full(n, 271.25), sss=jnp.asarray(sss), evaporation=z,
        snow_precipitation=z, qprec_ice=z, qcn_ice_bottom=z, qsb_ice_bottom=jnp.full(n, 2.0),
        fhld=z, qlead=jnp.full(n, 5.0), sst_m=jnp.asarray(sst),
    )
    return card, state, forcing


def _lateral_melt_config(card, beta, dmin):
    from legoesm.ice.config import SI3LateralMeltConfig

    si3 = card.config.si3._replace(
        lateral_melt=True, namthd_da=SI3LateralMeltConfig(rn_beta=beta, rn_dmin=dmin))
    return card.config._replace(si3=si3)


def _dispatch(config, state, forcing, dt, traced=False):
    """Jitted step_sea_ice; ``traced`` passes &namthd_da as jit arguments (training override)."""
    from legoesm.ice.config import SI3LateralMeltConfig
    from legoesm.ice.sea_ice import step_sea_ice

    cpu = jax.devices("cpu")[0]
    z = jnp.zeros(state.concentration.data.shape)
    state, forcing = jax.device_put((state, forcing), cpu)
    if not traced:
        out, _ = jax.jit(lambda s, f: step_sea_ice(s, f, z, z, z, config, 1.0, dt))(state, forcing)
        return out

    def step(s, f, beta, dmin):
        si3 = config.si3._replace(namthd_da=SI3LateralMeltConfig(rn_beta=beta, rn_dmin=dmin))
        return step_sea_ice(s, f, z, z, z, config._replace(si3=si3), 1.0, dt)[0]

    nam = config.si3.namthd_da
    return jax.jit(step)(state, forcing, jnp.float64(nam.rn_beta), jnp.float64(nam.rn_dmin))


def _production_expected(card, state, forcing, beta, dmin):
    """Oracle ice_thd_da on the column's post-ice_thd_temp state, then non-cooling ice_thd_do."""
    from legoesm.ice.bitz_lipscomb import SI3ColumnArrays, si3_column_step_arrays

    arrays = SI3ColumnArrays(*(f.data for f in state))
    pre = jax.jit(lambda s, f: si3_column_step_arrays(s, f, card.dt_seconds, ICE, si3=card.config.si3))(
        *jax.device_put((arrays, forcing), jax.devices("cpu")[0])).post_temp2
    pre = jax.tree.map(np.asarray, pre)
    entry_a = np.asarray(state.concentration.data)
    want = {"concentration": [], "h_ice": [], "h_snow": []}
    for k in range(len(entry_a)):
        a, h, hs = float(pre.concentration[k]), float(pre.h_ice[k]), float(pre.h_snow[k])
        if entry_a[k] > 1e-10:  # icethd.f90:181 gather
            o = _oracle_cell(
                dict(at_i=float(entry_a[k]), a_i=a, h_i=h, h_s=hs, e_i=list(pre.e_ice[k]),
                     e_s=list(pre.e_snow[k]), s_i=float(pre.S_bulk[k]), sz_i=list(pre.S_layers[k]),
                     sst=float(forcing.sst_m[k]), t_bo=float(forcing.t_bottom[k]),
                     sfx_lam=0.0, hfx_thd=0.0, wfx_lam=0.0),
                beta, dmin, card.config.si3.salinity_scheme, rdt=card.dt_seconds)
            a, h, hs = o["a_i"], o["h_i"], o["h_s"]
        a_tot = a + 0.0
        v_tot = h * a + 0.0 + 0.0
        want["concentration"].append(a_tot)
        want["h_ice"].append(_div(v_tot, a_tot) if a_tot > 1e-10 else 0.0)
        want["h_snow"].append(hs)
    return want


def _production_mismatch(out, want):
    return {
        f: np.flatnonzero(_bits(getattr(out, f).data) != _bits(want[f])).tolist()
        for f in want
        if (_bits(getattr(out, f).data) != _bits(want[f])).any()
    }


def test_d5_card_keeps_lateral_melt_off():
    deck = parse_namelist(C1D_ICE_CFG, "ice")
    assert deck[("ice", "namthd", "ln_iceda")][0] is False
    card, state, forcing = _production_case()
    assert card.config.si3.lateral_melt is False and card.config.si3.namthd_da is None
    on = _dispatch(_lateral_melt_config(card, *_deck()), state, forcing, card.dt_seconds)
    off = _dispatch(card.config, state, forcing, card.dt_seconds)
    assert not np.array_equal(_bits(on.concentration.data), _bits(off.concentration.data))


@pytest.mark.parametrize(
    "beta, dmin, traced", [(1.0, 8.0, False), (3.0, 8.0, False), (1.0, 6.2, True)])
def test_production_dispatch_bitwise(beta, dmin, traced):
    card, state, forcing = _production_case()
    cfg = _lateral_melt_config(card, beta, dmin)
    out = _dispatch(cfg, state, forcing, card.dt_seconds, traced=traced)
    want = _production_expected(card, state, forcing, beta, dmin)
    assert _production_mismatch(out, want) == {}
    a = np.asarray(out.concentration.data)
    if (beta, dmin) == _deck():
        assert a[3] == 0 and out.h_snow.data[3] == 0  # total melt
    assert _bits(a[4]) == _bits(state.concentration.data[4])  # not gathered
    assert np.isnan(forcing.sst_m[2]) and a[2] == want["concentration"][2]  # MAX(0, NaN) = +0


def test_production_dispatch_calls_ice_thd_da_with_the_config(monkeypatch):
    import legoesm.ice.si3_thd_da as da_mod

    calls, real = [], da_mod.si3_ice_thd_da

    def spy(**kw):
        calls.append({k: kw[k] for k in ("rn_beta", "rn_dmin", "rDt_ice", "nn_icesal")})
        return real(**kw)

    monkeypatch.setattr(da_mod, "si3_ice_thd_da", spy)
    card, state, forcing = _production_case()
    beta, dmin = _deck()
    _dispatch(_lateral_melt_config(card, beta, dmin), state, forcing, card.dt_seconds)
    assert calls == [dict(rn_beta=beta, rn_dmin=dmin, rDt_ice=card.dt_seconds, nn_icesal=2)]
    _dispatch(card.config, state, forcing, card.dt_seconds)
    assert len(calls) == 1  # ln_icedA = .false. never calls it


# (module attribute, replacement, beta, dmin, traced).  Not here, with the reason: source
# rounding and the SUM order move only sfx/hfx/wfx, which this column does not return
# (leaf plants above fire on them); a NaN MIN operand needs at_i <= epsi10, never gathered.
# The 1/300 rewrite happens only on a traced rn_dmin (constants fold exactly).
PRODUCTION_PLANTS = {
    "call_removed": ("_si3_column_thd_da", lambda entry, post, *a: post, 1.0, 8.0, False),
    "max_nan_propagating": ("fortran_max", lambda a, b: jnp.maximum(a, b), 1.0, 8.0, False),
    "a_eq_zero_never": (
        "ieee_is_zero", lambda x: jnp.zeros(jnp.shape(x), bool), 1.0, 8.0, False),
    "zdmax_reciprocal": (
        "_da_zastar", lambda d, b: 1.0 / (1.0 - bl.transcendentals.pow(d / 300.0, 1.0 / b)),
        1.0, 6.2, True),
    "native_pow": (None, None, 3.0, 8.0, False),
}


@pytest.mark.parametrize("plant", sorted(PRODUCTION_PLANTS))
def test_production_dispatch_plants_fire(plant, monkeypatch):
    from legoesm.ice import bitz_lipscomb

    name, fake, beta, dmin, traced = PRODUCTION_PLANTS[plant]
    card, state, forcing = _production_case()
    want = _production_expected(card, state, forcing, beta, dmin)
    if plant == "native_pow":
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    else:
        monkeypatch.setattr(bitz_lipscomb if plant == "call_removed" else bl, name, fake)
    cfg = _lateral_melt_config(card, beta, dmin)
    out = _dispatch(cfg, state, forcing, card.dt_seconds, traced=traced)
    assert _production_mismatch(out, want), f"plant {plant} did not change any bit"
