"""Independent scalar checks for SI3 O10 salinity-4 drainage and flushing."""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as canonical_constants
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG


RDT = 10800.0
R1_RDT = 1.0 / RDT
R1_NLAY = 1.0 / 10.0
RHO_I = 917.0
RHO_S = 330.0
RHO_W = 1000.0
GRAVITY = canonical_constants.g_nemo
NML = {
    "nn_icesal": 4,
    "ln_drainage": True,
    "ln_flushing": True,
    "rn_simin": 0.1,
    "rn_sinew": 0.75,
    "rn_sal_himin": 0.1,
    "nn_liquidus": 2,
    "nn_drainage": 50,
    "nn_flushing": 5,
    "rn_flushrate": 0.3,
    "rn_vbrc": 0.05,
    "nn_sal_scheme": 1,
    "rn_alpha_RJW": 0.03,
    "rn_Rc_RJW": 8.0,
    "ln_sal_chk": False,
}


def _fmax(dst, src):
    """x86 ``maxsd src,dst`` tie survivor, independent of JAX/model code."""

    return dst if dst > src else src


def _fmin(dst, src):
    """x86 ``minsd src,dst`` tie survivor, independent of JAX/model code."""

    return dst if dst < src else src


def _vc19(temperature, T0):
    zt1 = temperature - T0
    zt2 = zt1 * zt1
    zt3 = zt2 * zt1
    return (-18.7 * zt1 - 0.519 * zt2) - 0.00535 * zt3


def _permeability(volumes, *, initial):
    out = []
    for start in range(10):
        total = 0.0
        for value in volumes[start:]:
            if initial and value < 1.0e-6:
                total = 0.0
                break
            perm = ((3.0e-8 * value) * value) * value
            total = total + 1.0 / perm
        if initial:
            out.append((10 - start) / total if total > 0.0 else 0.0)
        else:
            out.append((10 - start) / total)
    return out


def _rjw(salinity, brine, mask, sss, h_i, zmid, zedge, z1_h, *, initial):
    divisor = [_fmax(brine[k], 1.0e-10) for k in range(10)] if initial else brine[:10]
    volumes = [mask[k] * salinity[k] / divisor[k] for k in range(10)]
    permeability = _permeability(volumes, initial=initial)
    rayleigh = []
    for k in range(10):
        value = 4.0e6 * GRAVITY
        value = value * 7.5e-4
        value = value * _fmax(brine[k] - sss, 0.0)
        value = value * permeability[k]
        value = value * h_i
        value = value * (1.0 - zmid[k])
        value = value * ((1.0 / 0.523) * (1.0 / 1.8e-6))
        rayleigh.append(value)
    critical_layer = 11
    zc = 0.0
    excess_max = 0.0
    for k in range(9, -1, -1):
        if rayleigh[k] >= 8.0:
            excess_max = _fmax(excess_max, rayleigh[k] - 8.0)
            zc = zedge[k]
            critical_layer = k + 1
    inverse_depth2 = 1.0 / ((1.0 - zc) * (1.0 - zc))
    out = [0.0] * 10
    for k in range(10):
        if k + 1 >= critical_layer:
            value = -0.03 * excess_max
            value = value * (0.523 * (1.0 / 4.0e6))
            value = value * (zmid[k] - zc)
            value = value * z1_h
            out[k] = value * inverse_depth2
    return out


def _salt_increment(a_i, h_i, zds):
    value = RHO_I * a_i
    value = value * h_i
    value = value * R1_NLAY
    value = value * zds
    return value * R1_RDT


def _oracle_column(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,
                   dh_s_sum, sfx_bri, sfx_res, T0):
    """Literal scalar translation of production ``ice_thd_sal``; no model calls."""

    zmid = [(float(k) - 0.5) * R1_NLAY for k in range(1, 11)]
    zedge = [(float(k) - 1.0) * R1_NLAY for k in range(1, 12)]
    brine = [_vc19(value, T0) for value in t_i] + [sss]
    mask = [1.0 if value - T0 < -1.0e-6 else 0.0 for value in t_i]
    melt = dh_s_sum + dh_i_sum

    z1_h = 1.0 / _fmax(h_i * R1_NLAY, 1.0e-10)
    brine_max = 0.0
    for value in brine[:10]:
        brine_max = _fmax(brine_max, value)
    drain = h_i >= 0.1 and melt >= 0.0 and brine_max > sss and t_su <= t_bo
    velocity = _rjw(sz_i, brine, mask, sss, h_i, zmid, zedge, z1_h, initial=True)
    cfl_test = 0.0
    for value in velocity:
        cfl_test = _fmax(cfl_test, abs((value * RDT) * z1_h))
    ndrain = min(50, math.ceil(cfl_test * 2.0)) if drain else 0
    zdt = RDT / max(1, ndrain)
    cfl_drain = 0.0
    smin_drain = 0.0
    salinity = list(sz_i)
    for iteration in range(ndrain):
        if iteration > 0:
            velocity = _rjw(
                salinity, brine, mask, sss, h_i, zmid, zedge, z1_h, initial=False
            )
        for k in range(10):
            cfl = (velocity[k] * zdt) * z1_h
            zds = -cfl * (brine[k + 1] - brine[k])
            cfl_drain = _fmax(cfl_drain, abs(cfl))
            smin_drain = _fmin(smin_drain, salinity[k] + zds)
            zds = _fmax(zds, -salinity[k] + _fmin(0.1, 0.75 * sss))
            salinity[k] = salinity[k] + zds
            sfx_bri = sfx_bri - _salt_increment(a_i, h_i, zds)

    z1_h = 1.0 / _fmax(h_i * R1_NLAY, 1.0e-10)
    flush = h_i >= 0.1 and melt < 0.0
    flush_velocity = []
    for value in brine[:10]:
        mass = dh_i_sum * RHO_I + dh_s_sum * RHO_S
        velocity_k = (-0.3 * mass) / (RHO_W * (1.0 + 0.8e-3 * value))
        flush_velocity.append(velocity_k * R1_RDT)
    cfl_test = 0.0
    for value in flush_velocity:
        cfl_test = _fmax(cfl_test, abs((value * RDT) * z1_h))
    nflush = min(5, math.ceil(cfl_test * 2.0)) if flush else 0
    zdt = RDT / max(1, nflush)
    cfl_flush = 0.0
    smin_flush = 0.0
    for _ in range(nflush):
        volumes = [mask[k] * salinity[k] / brine[k] for k in range(10)]
        volume_min = 1.0e20
        for value in volumes:
            volume_min = _fmin(volume_min, value)
        if volume_min >= 0.05:
            for k in range(10):
                cfl = (flush_velocity[k] * zdt) * z1_h
                cfl_flush = _fmax(cfl_flush, abs(cfl))
                zds = -cfl * (brine[k] - (0.0 if k == 0 else brine[k - 1]))
                smin_flush = _fmin(smin_flush, salinity[k] + zds)
                zds = _fmax(_fmin(0.0, zds), -salinity[k] + _fmin(0.1, 0.75 * sss))
                salinity[k] = salinity[k] + zds
                sfx_bri = sfx_bri - _salt_increment(a_i, h_i, zds)

    for k in range(10):
        old = salinity[k]
        salinity[k] = _fmin(_fmax(0.1, salinity[k]), 0.75 * sss)
        sfx_res = sfx_res - _salt_increment(a_i, h_i, salinity[k] - old)
    return (
        salinity, sfx_bri, sfx_res, ndrain, nflush,
        cfl_drain, cfl_flush, smin_drain, smin_flush,
    )


def _fixture():
    T0 = NEMO_SI3_CONSTANTS_CONFIG.T0
    n = 8
    h_i = np.array([1.5, 2.2, 0.05, 1.0, 1.8, 0.7, 0.11, 1.2])
    t_su = np.array([T0 - 12.0, T0 - 0.2, T0 - 8.0, T0 - 5.0,
                     T0 - 0.1, T0 - 2.0, T0 - 18.0, T0 - 0.5])
    t_bo = np.array([T0 - 1.8] * n)
    sss = np.array([32.0, 34.0, 30.0, 0.2, 35.0, 28.0, 33.0, 36.0])
    a_i = np.array([0.7, 0.9, 0.4, 0.8, 0.95, 0.6, 0.5, 0.85])
    dh_i_sum = np.array([0.0, -0.08, 0.0, 0.0, -0.4, -0.01, 0.0, -0.03])
    dh_s_sum = np.array([0.0, -0.02, 0.0, 0.0, -0.1, 0.0, 0.0, -0.01])
    levels = np.linspace(0.0, 1.0, 10)
    t_i = np.stack([
        T0 - (2.0 + 15.0 * levels),
        T0 - (0.4 + 2.0 * levels),
        T0 - (3.0 + levels),
        T0 - (1.0 + levels),
        T0 - (0.2 + 1.0 * levels),
        T0 - (0.5 + 3.0 * levels),
        T0 - (1.8 + 0.3 * levels),
        T0 - (0.3 + 2.5 * levels),
    ])
    sz_i = np.stack([
        np.linspace(3.0, 11.0, 10),
        np.linspace(12.0, 4.0, 10),
        np.linspace(2.0, 6.0, 10),
        np.linspace(0.02, 0.3, 10),
        np.linspace(18.0, 5.0, 10),
        np.linspace(8.0, 3.0, 10),
        np.linspace(20.0, 24.0, 10),
        np.linspace(10.0, 2.0, 10),
    ])
    return dict(
        h_i=h_i, t_su=t_su, t_i=t_i, sz_i=sz_i, t_bo=t_bo, sss=sss,
        a_i=a_i, dh_i_sum=dh_i_sum, dh_s_sum=dh_s_sum,
        sfx_bri=np.array([0.0, -0.0, 1.0, -2.0, 3.0, 0.5, -1.0, 2.0]),
        sfx_res=np.array([-0.0, 0.0, -1.0, 2.0, -3.0, -0.5, 1.0, -2.0]),
    )


def _model_kwargs():
    return dict(
        **{k: jnp.asarray(v) for k, v in _fixture().items()},
        namelist=NML, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def _oracle_batch():
    fixture = _fixture()
    rows = [
        _oracle_column(
            **{name: (values[i].tolist() if name in ("t_i", "sz_i") else float(values[i]))
               for name, values in fixture.items()},
            T0=NEMO_SI3_CONSTANTS_CONFIG.T0,
        )
        for i in range(fixture["h_i"].size)
    ]
    return tuple(np.asarray([row[k] for row in rows]) for k in range(9))


def test_salinity4_is_bit_exact_to_independent_scalar_under_jit_and_eager():
    oracle = _oracle_batch()
    eager = None
    with jax.disable_jit():
        eager = bl.si3_ice_thd_sal(**_model_kwargs())
    compiled = jax.jit(lambda: bl.si3_ice_thd_sal(**_model_kwargs()))()
    for actual in (eager, compiled):
        for got, want in zip(actual, oracle):
            if np.issubdtype(np.asarray(want).dtype, np.integer):
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_array_equal(_bits(got), _bits(want))


def test_salinity4_inexact_inputs_and_output_digest_are_pinned():
    assert np.float64(NML["rn_flushrate"]).hex() == "0x1.3333333333333p-2"
    assert np.float64(NML["rn_alpha_RJW"]).hex() == "0x1.eb851eb851eb8p-6"
    assert np.float64(R1_NLAY).hex() == "0x1.999999999999ap-4"
    out = jax.jit(lambda: bl.si3_ice_thd_sal(**_model_kwargs()))()
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(out):
        digest.update(np.asarray(leaf).tobytes())
    assert digest.hexdigest() == "60b8eb225a171059d4e25f39b99f000b3c41b2b0127f1d57af304da67fbc68c3"


def test_salinity4_branch_and_subcycle_coverage_is_real():
    out = jax.jit(lambda: bl.si3_ice_thd_sal(**_model_kwargs()))()
    ndrain = np.asarray(out.ndrainage)
    nflush = np.asarray(out.nflushing)
    assert np.any(ndrain > 1)
    assert np.any(nflush > 1)
    assert np.any(ndrain == 0)
    assert np.any(nflush == 0)
    assert np.any(_bits(out.sfx_bri) != _bits(_model_kwargs()["sfx_bri"]))


def test_salinity4_gradient_is_finite_nonzero_and_matches_central_difference():
    kwargs = _model_kwargs()
    base = kwargs.pop("sz_i")

    def objective(scale):
        out = bl.si3_ice_thd_sal(sz_i=base * scale, **kwargs)
        return jnp.sum(out.sz_i) + jnp.sum(out.sfx_bri) * 1.0e-3

    analytic = jax.jit(jax.grad(objective))(jnp.asarray(1.0, dtype=jnp.float64))
    epsilon = 1.0e-6
    finite = (objective(1.0 + epsilon) - objective(1.0 - epsilon)) / (2.0 * epsilon)
    assert np.isfinite(analytic)
    assert analytic != 0.0
    np.testing.assert_allclose(analytic, finite, rtol=1.0e-5, atol=1.0e-7)


@pytest.mark.parametrize(
    "missing",
    [
        "h_i", "t_su", "t_i", "sz_i", "t_bo", "sss", "a_i",
        "dh_i_sum", "dh_s_sum", "sfx_bri", "sfx_res", "rDt_ice",
        "r1_Dt_ice", "r1_nlay_i", "rho_i", "rho_s", "rho_w", "gravity",
        "constants", "namelist",
    ],
)
def test_salinity4_has_no_default_for_physical_inputs(missing):
    kwargs = _model_kwargs()
    kwargs[missing] = None
    with pytest.raises(ValueError, match="requires|missing"):
        bl.si3_ice_thd_sal(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("nn_icesal", 2), ("ln_drainage", False), ("ln_flushing", False),
        ("nn_liquidus", 1), ("nn_drainage", 49), ("nn_flushing", 4),
        ("nn_sal_scheme", 2), ("ln_sal_chk", True),
    ],
)
def test_salinity4_refuses_nonproduction_selectors(name, value):
    kwargs = _model_kwargs()
    kwargs["namelist"] = {**NML, name: value}
    with pytest.raises(ValueError, match="production selectors"):
        bl.si3_ice_thd_sal(**kwargs)


def test_o10_production_composer_calls_salinity_then_second_temperature(monkeypatch):
    """A removed SAL or TEMP2 call makes this production-path plant fail."""

    token = jnp.asarray([1.0])
    sal_token = jnp.asarray([[7.0] * 10])
    fake_dh = SimpleNamespace(
        h_i=token, t_su=token, sz_i=jnp.ones((1, 10)), a_i=token,
        e_i=jnp.ones((1, 10)),
        dh=SimpleNamespace(dh_i_sum=token, dh_s_sum=token),
        fluxes=SimpleNamespace(sfx_res=token),
    )
    fake_o9 = SimpleNamespace(dh=fake_dh, t_i=jnp.ones((1, 10)))
    calls = []

    monkeypatch.setattr(bl, "si3_ice_thd_category_stages", lambda **kwargs: fake_o9)

    def salinity_spy(**kwargs):
        calls.append("sal")
        assert kwargs["sz_i"] is fake_dh.sz_i
        return bl.SI3SalinityOutputs(
            sal_token, token, token, token.astype(jnp.int32), token.astype(jnp.int32),
            token, token, token, token,
        )

    def temperature_spy(**kwargs):
        calls.append("temp2")
        assert kwargs["sz_i"] is sal_token
        return sal_token + 1.0

    monkeypatch.setattr(bl, "si3_ice_thd_sal", salinity_spy)
    monkeypatch.setattr(bl, "si3_ice_thd_temp", temperature_spy)
    result = bl.si3_ice_thd_o10_category_stages(
        entry={"t_bo": token, "sss": token}, fluxes=None, sfx_bri=token,
        namelist=NML, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    assert calls == ["sal", "temp2"]
    np.testing.assert_array_equal(result.t_i, sal_token + 1.0)
