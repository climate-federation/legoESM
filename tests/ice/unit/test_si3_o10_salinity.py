"""Independent scalar checks for SI3 O10 salinity-4 drainage and flushing."""

from __future__ import annotations

import hashlib
import inspect
import math
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as canonical_constants
from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture


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


def _permeability(volumes, *, initial, events=None):
    out = []
    for start in range(10):
        total = 0.0
        for value in volumes[start:]:
            if initial and value < 1.0e-6:
                if events is not None:
                    events["harmonic_early_exit"] += 1
                total = 0.0
                break
            perm = ((3.0e-8 * value) * value) * value
            total = total + 1.0 / perm
        if initial:
            out.append((10 - start) / total if total > 0.0 else 0.0)
        else:
            out.append((10 - start) / total)
    return out


def _rjw(salinity, brine, mask, sss, h_i, zmid, zedge, z1_h, *, initial,
         events=None):
    divisor = [_fmax(brine[k], 1.0e-10) for k in range(10)] if initial else brine[:10]
    volumes = [mask[k] * salinity[k] / divisor[k] for k in range(10)]
    permeability = _permeability(volumes, initial=initial, events=events)
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


def _temperature_from_enthalpy(e_i, salinity, h_i):
    if h_i <= 0.0:
        return [NEMO_SI3_CONSTANTS_CONFIG.T0] * len(salinity)
    c = NEMO_SI3_CONSTANTS_CONFIG
    inverse_rho = 1.0 / c.rho_ice
    inverse_heat_capacity = 1.0 / c.c_ice
    out = []
    for energy, salt in zip(e_i, salinity):
        melt = -c.liquidus_slope * salt
        b = ((c.c_ocean - c.c_ice) * melt + energy * inverse_rho) - c.latent_fusion
        disc = _fmax(b * b - (4.0 * c.c_ice * c.latent_fusion) * melt, 0.0)
        out.append(c.T0 - (((b + math.sqrt(disc)) * 0.5) * inverse_heat_capacity))
    return out


def _oracle_column(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,
                   dh_s_sum, sfx_bri, sfx_res, T0, events=None):
    """Literal scalar translation of production ``ice_thd_sal``; no model calls."""

    zmid = [(float(k) - 0.5) * R1_NLAY for k in range(1, 11)]
    zedge = [(float(k) - 1.0) * R1_NLAY for k in range(1, 12)]
    brine = [_vc19(value, T0) for value in t_i] + [sss]
    mask = [1.0 if value - T0 < -1.0e-6 else 0.0 for value in t_i]
    if events is not None:
        events["zmsk_zero_layers"] += sum(value == 0.0 for value in mask)
    melt = dh_s_sum + dh_i_sum

    z1_h = 1.0 / _fmax(h_i * R1_NLAY, 1.0e-10)
    brine_max = 0.0
    for value in brine[:10]:
        brine_max = _fmax(brine_max, value)
    drain = h_i >= 0.1 and melt >= 0.0 and brine_max > sss and t_su <= t_bo
    velocity = _rjw(
        sz_i, brine, mask, sss, h_i, zmid, zedge, z1_h, initial=True,
        events=events if drain else None,
    )
    cfl_test = 0.0
    for value in velocity:
        cfl_test = _fmax(cfl_test, abs((value * RDT) * z1_h))
    ndrain = min(50, math.ceil(cfl_test * 2.0)) if drain else 0
    if events is not None:
        events["drain_cap_50"] += drain and math.ceil(cfl_test * 2.0) > 50
        events["drain_iterations_after_first"] += max(0, ndrain - 1)
    zdt = RDT / max(1, ndrain)
    cfl_drain = 0.0
    smin_drain = 0.0
    salinity = list(sz_i)
    for iteration in range(ndrain):
        if iteration > 0:
            velocity = _rjw(
                salinity, brine, mask, sss, h_i, zmid, zedge, z1_h,
                initial=False,
            )
        for k in range(10):
            cfl = (velocity[k] * zdt) * z1_h
            zds = -cfl * (brine[k + 1] - brine[k])
            cfl_drain = _fmax(cfl_drain, abs(cfl))
            smin_drain = _fmin(salinity[k] + zds, smin_drain)
            floor = -salinity[k] + _fmin(0.75 * sss, 0.1)
            if events is not None:
                events["drain_floor_clamp"] += zds < floor
            zds = _fmax(zds, floor)
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
    if events is not None:
        events["flush_cap_5"] += flush and math.ceil(cfl_test * 2.0) > 5
        events["flush_iterations_after_first"] += max(0, nflush - 1)
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
                nonpositive = _fmin(0.0, zds) if k == 8 else _fmin(zds, 0.0)
                floor = -salinity[k] + _fmin(0.75 * sss, 0.1)
                if events is not None:
                    events["flush_min_zero_block"] += zds > 0.0
                    events["flush_floor_clamp"] += nonpositive < floor
                zds = (
                    _fmax(floor, nonpositive)
                    if k in (8, 9)
                    else _fmax(nonpositive, floor)
                )
                salinity[k] = salinity[k] + zds
                sfx_bri = sfx_bri - _salt_increment(a_i, h_i, zds)

    for k in range(10):
        old = salinity[k]
        salinity[k] = _fmin(_fmax(salinity[k], 0.1), 0.75 * sss)
        sfx_res = sfx_res - _salt_increment(a_i, h_i, salinity[k] - old)
    return (
        salinity, sfx_bri, sfx_res, ndrain, nflush,
        cfl_drain, cfl_flush, smin_drain, smin_flush,
    )


def _fixture():
    T0 = NEMO_SI3_CONSTANTS_CONFIG.T0
    n = 14
    h_i = np.array([1.5, 2.2, 0.05, 1.0, 1.8, 0.7, 0.11, 1.2,
                    0.1, 0.11, 0.1, 0.1, 0.17693980726715666, 0.0])
    t_su = np.array([T0 - 12.0, T0 - 0.2, T0 - 8.0, T0 - 5.0,
                     T0 - 0.1, T0 - 2.0, T0 - 18.0, T0 - 0.5,
                     T0 - 3.0, T0 - 3.0, T0 - 1.0, T0 - 1.0, T0 - 5.0,
                     T0])
    t_bo = np.array([T0 - 1.8] * n)
    sss = np.array([32.0, 34.0, 30.0, 0.2, 35.0, 28.0, 33.0, 36.0,
                    32.0, 32.0, 32.0, 32.0, 20.140314206854406, 32.0])
    a_i = np.array([0.7, 0.9, 0.4, 0.8, 0.95, 0.6, 0.5, 0.85,
                    0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    dh_i_sum = np.array([0.0, -0.08, 0.0, 0.0, -0.4, -0.01, 0.0, -0.03,
                         0.0, 0.0, -0.5, -0.5, 0.0, 0.0])
    dh_s_sum = np.array([0.0, -0.02, 0.0, 0.0, -0.1, 0.0, 0.0, -0.01,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
        np.full(10, T0 - 2.0),
        np.array([T0 - 2.0] * 8 + [T0, T0 - 2.0]),
        T0 - np.linspace(0.5, 3.0, 10),
        T0 - np.linspace(3.0, 0.5, 10),
        T0 + np.array([
            -4.4339423128342075, -3.783339432360672, -3.30286327594427,
            -2.6662466539688365, -2.6125098201413213, -1.6846832762683912,
            -1.3798759479405038, -1.1318599582791267, -1.0657063643631135,
            -0.7658493514394027,
        ]),
        np.full(10, T0),
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
        np.full(10, 24.0),
        np.full(10, 24.0),
        np.full(10, 10.0),
        np.full(10, 10.0),
        np.array([
            9.891649073279408, 14.35709691518058, 14.039168399538456,
            8.645953206849674, 8.935850592789969, 14.482049715507154,
            3.2605020539867735, 7.739808978871614, 1.0234779941351597,
            11.72299965885506,
        ]),
        np.full(10, 5.0),
    ])
    return dict(
        h_i=h_i, t_su=t_su, t_i=t_i, sz_i=sz_i, t_bo=t_bo, sss=sss,
        a_i=a_i, dh_i_sum=dh_i_sum, dh_s_sum=dh_s_sum,
        sfx_bri=np.array([0.0, -0.0, 1.0, -2.0, 3.0, 0.5, -1.0, 2.0,
                          -0.0, 0.0, -0.0, 0.0, -0.0, -0.0]),
        sfx_res=np.array([-0.0, 0.0, -1.0, 2.0, -3.0, -0.5, 1.0, -2.0,
                          0.0, -0.0, 0.0, -0.0, 0.0, 0.0]),
    )


def _model_kwargs():
    return dict(
        **{k: jnp.asarray(v) for k, v in _fixture().items()},
        namelist=NML, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


_ARRAY_NAMES = tuple(_fixture())


def _model_from_array_args(*values):
    arrays = dict(zip(_ARRAY_NAMES, values, strict=True))
    return bl.si3_ice_thd_sal(
        **arrays, namelist=NML, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


_JIT_MODEL_FROM_ARRAY_ARGS = jax.jit(_model_from_array_args)


def _array_args():
    fixture = _fixture()
    return tuple(jnp.asarray(fixture[name]) for name in _ARRAY_NAMES)


def _source_mutant(function, old, new):
    """Compile a one-site mutant from the real production function source."""

    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1
    namespace = dict(vars(bl))
    namespace["_barrier_identity"] = lambda value: value
    exec(
        compile(source.replace(old, new), inspect.getsourcefile(function), "exec"),
        namespace,
    )
    return namespace[function.__name__]


def _without_source_rounding(function):
    source = textwrap.dedent(inspect.getsource(function))
    assert "nemo_source_round(" in source
    namespace = dict(vars(bl))
    namespace["_barrier_identity"] = lambda value: value
    exec(
        compile(
            source.replace("nemo_source_round(", "_barrier_identity("),
            inspect.getsourcefile(function),
            "exec",
        ),
        namespace,
    )
    return namespace[function.__name__]


def _rjw_barrier_stress():
    rng = np.random.default_rng(19010)
    n = 2000
    salinity = jnp.asarray(rng.uniform(0.1, 25.0, (n, 10)))
    inner_brine = jnp.asarray(rng.uniform(0.2, 80.0, (n, 10)))
    sss = jnp.asarray(rng.uniform(20.0, 36.0, n))
    brine = jnp.concatenate(
        [jnp.zeros((n, 1)), inner_brine, sss[:, None]], axis=-1
    )
    h_i = jnp.asarray(rng.uniform(0.1, 6.0, n))
    return (
        salinity,
        brine,
        jnp.ones_like(salinity),
        sss,
        h_i,
        jnp.asarray((np.arange(10) + 0.5) / 10.0),
        jnp.asarray(np.arange(11) / 10.0),
        10.0 / h_i,
    )


def _unguarded_effective_permeability(zv_br, *, initial):
    """Pre-fix arithmetic retained only for forward-bit regression proof."""

    nlay = zv_br.shape[-1]
    one = jnp.ones_like(zv_br[..., 0])
    zero = jnp.zeros_like(one)
    out = []
    for jk in range(nlay):
        total = zero
        active = jnp.ones_like(one, dtype=bool)
        for jk2 in range(jk, nlay):
            low = zv_br[..., jk2] < 1.0e-6
            go = active & (~low if initial else jnp.ones_like(low, dtype=bool))
            safe_v = jnp.where(go, zv_br[..., jk2], one)
            permeability = bl.nemo_source_round(3.0e-8 * safe_v)
            permeability = bl.nemo_source_round(permeability * safe_v)
            permeability = bl.nemo_source_round(permeability * safe_v)
            reciprocal = bl.nemo_source_round(one / permeability)
            candidate = bl.nemo_source_round(total + reciprocal)
            total = jnp.where(go, candidate, jnp.where(active & low, zero, total))
            active = go
        count = jnp.asarray(float(nlay - jk), dtype=zv_br.dtype)
        safe_sum = jnp.where(total > 0.0, total, one)
        value = bl.nemo_source_round(count / safe_sum)
        out.append(jnp.where(total > 0.0, value, zero))
    return jnp.stack(out, axis=-1)


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
    with jax.disable_jit():
        eager = _model_from_array_args(*_array_args())
    compiled = _JIT_MODEL_FROM_ARRAY_ARGS(*_array_args())
    for actual in (eager, compiled):
        for got, want in zip(actual, oracle):
            if np.issubdtype(np.asarray(want).dtype, np.integer):
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize(
    "site", ["vc19", "flux_increment", "rayleigh_chain", "loop_body"]
)
def test_real_model_barrier_groups_have_bit_changing_stripped_plants(site, monkeypatch):
    """Every plant strips barriers from the executing model function itself."""

    if site == "vc19":
        rng = np.random.default_rng(19010)
        args = (
            jnp.asarray(
                NEMO_SI3_CONSTANTS_CONFIG.T0 - rng.uniform(0.01, 40.0, 2000)
            ),
            NEMO_SI3_CONSTANTS_CONFIG.T0,
        )
        function = bl._sal4_vc19_brine
        with jax.disable_jit():
            eager = function(*args)
        protected = jax.jit(function)(*args)
        stripped = jax.jit(_without_source_rounding(function))(*args)
        np.testing.assert_array_equal(_bits(protected), _bits(eager))
        assert np.count_nonzero(_bits(stripped) != _bits(eager)) > 0
        return

    if site == "rayleigh_chain":
        args = _rjw_barrier_stress()
        kwargs = {
            "gravity": GRAVITY,
            "rn_alpha_rjw": NML["rn_alpha_RJW"],
            "rn_rc_rjw": NML["rn_Rc_RJW"],
            "initial": True,
        }
        function = bl._sal4_rjw_velocity
        with jax.disable_jit():
            eager = function(*args, **kwargs)
        protected = jax.jit(function, static_argnames=("initial",))(*args, **kwargs)
        stripped = jax.jit(
            _without_source_rounding(function), static_argnames=("initial",)
        )(*args, **kwargs)
        np.testing.assert_array_equal(_bits(protected), _bits(eager))
        assert np.count_nonzero(_bits(stripped) != _bits(eager)) > 0
        return

    args = _array_args()
    with jax.disable_jit():
        eager = _model_from_array_args(*args)
    protected = jax.jit(lambda *values: _model_from_array_args(*values))(*args)
    target = (
        bl._sal4_flux_increment
        if site == "flux_increment"
        else bl._si3_ice_thd_sal4_core
    )
    monkeypatch.setattr(bl, target.__name__, _without_source_rounding(target))
    stripped = jax.jit(lambda *values: _model_from_array_args(*values))(*args)
    for got, want in zip(protected, eager):
        np.testing.assert_array_equal(_bits(got), _bits(want))
    assert sum(
        np.count_nonzero(_bits(got) != _bits(want))
        for got, want in zip(stripped, eager)
    ) > 0


def test_zero_permeability_guard_changes_no_forward_bits_on_fixture_and_fuzz():
    fixture = _fixture()
    temperature = jnp.asarray(fixture["t_i"])
    salinity = jnp.asarray(fixture["sz_i"])
    brine = bl._sal4_vc19_brine(temperature, NEMO_SI3_CONSTANTS_CONFIG.T0)
    mask = jnp.where(
        bl.nemo_source_round(temperature - NEMO_SI3_CONSTANTS_CONFIG.T0) < -1.0e-6,
        1.0,
        0.0,
    )
    numerator = bl.nemo_source_round(mask * salinity)
    full_fixture = bl.nemo_source_round(
        numerator / jnp.where(numerator == 0.0, 1.0, brine)
    )

    rng = np.random.default_rng(1901646)
    fuzz = rng.uniform(1.0e-4, 1.0, (12000, 10))
    fuzz[::7, rng.integers(0, 10, len(fuzz[::7]))] = 0.0
    fuzz[::11, rng.integers(0, 10, len(fuzz[::11]))] = -0.0
    fuzz = jnp.asarray(fuzz)

    for initial, values in (
        (False, full_fixture),
        (True, fuzz),
        (False, fuzz),
    ):
        old = jax.jit(_unguarded_effective_permeability, static_argnames=("initial",))(
            values, initial=initial
        )
        new = jax.jit(bl._sal4_effective_permeability, static_argnames=("initial",))(
            values, initial=initial
        )
        np.testing.assert_array_equal(_bits(new), _bits(old))


def test_salinity4_inexact_inputs_and_output_digest_are_pinned():
    assert np.float64(NML["rn_flushrate"]).hex() == "0x1.3333333333333p-2"
    assert np.float64(NML["rn_alpha_RJW"]).hex() == "0x1.eb851eb851eb8p-6"
    assert np.float64(R1_NLAY).hex() == "0x1.999999999999ap-4"
    out = _JIT_MODEL_FROM_ARRAY_ARGS(*_array_args())
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(out):
        digest.update(np.asarray(leaf).tobytes())
    assert digest.hexdigest() == "1d77233ee682c9e47b266ed263db0b4be98a13d4dcf329d6a6fe80a9dd27a674"


def test_salinity4_branch_and_subcycle_coverage_is_real():
    out = _JIT_MODEL_FROM_ARRAY_ARGS(*_array_args())
    ndrain = np.asarray(out.ndrainage)
    nflush = np.asarray(out.nflushing)
    assert np.any(ndrain > 1)
    assert np.any(nflush > 1)
    assert np.any(ndrain == 0)
    assert np.any(nflush == 0)
    assert np.any(_bits(out.sfx_bri) != _bits(_model_kwargs()["sfx_bri"]))

    events = {
        name: 0 for name in (
            "drain_floor_clamp", "flush_floor_clamp", "flush_min_zero_block",
            "harmonic_early_exit", "zmsk_zero_layers", "drain_cap_50",
            "flush_cap_5", "drain_iterations_after_first",
            "flush_iterations_after_first",
        )
    }
    fixture = _fixture()
    for i in range(fixture["h_i"].size):
        _oracle_column(
            **{
                name: (values[i].tolist() if name in ("t_i", "sz_i") else float(values[i]))
                for name, values in fixture.items()
            },
            T0=NEMO_SI3_CONSTANTS_CONFIG.T0,
            events=events,
        )
    assert events == {
        "drain_floor_clamp": 1,
        "flush_floor_clamp": 11,
        "flush_min_zero_block": 9,
        "harmonic_early_exit": 9,
        "zmsk_zero_layers": 11,
        "drain_cap_50": 1,
        "flush_cap_5": 2,
        "drain_iterations_after_first": 62,
        "flush_iterations_after_first": 9,
    }


def _reachable_flush_role_fixture():
    """Two production-NML columns at the real line-644 signed-zero tie."""

    T0 = NEMO_SI3_CONSTANTS_CONFIG.T0
    temperature = np.full(10, T0 - 0.1)
    arrays = {
        "h_i": [1.0, 1.0], "t_su": [T0 - 1.0, T0 - 1.0],
        "t_i": [temperature, temperature],
        "sz_i": [np.full(10, NML["rn_simin"])] * 2,
        "t_bo": [T0 - 1.8, T0 - 1.8], "sss": [32.0, 32.0],
        "a_i": [0.8, 0.8], "dh_i_sum": [-0.001, -0.001],
        "dh_s_sum": [0.0, 0.0], "sfx_bri": [-0.0, 0.0],
        "sfx_res": [0.0, 0.0],
    }
    return tuple(jnp.asarray(arrays[name]) for name in _ARRAY_NAMES)


def _role_model(namelist, *values):
    arrays = dict(zip(_ARRAY_NAMES, values, strict=True))
    return bl.si3_ice_thd_sal(
        **arrays, namelist=namelist, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


@pytest.mark.parametrize(
    ("site", "old", "new"),
    [
        (
            "flush_inner9",
            "_minsd(zero, zds_raw) if jk == 8 else _minsd(zds_raw, zero)",
            "_minsd(zds_raw, zero) if jk == 8 else _minsd(zds_raw, zero)",
        ),
        (
            "flush_outer",
            "_maxsd(floor_delta, nonpositive) if jk in (8, 9) else _maxsd(nonpositive, floor_delta)",
            "_maxsd(nonpositive, floor_delta) if jk in (8, 9) else _maxsd(floor_delta, nonpositive)",
        ),
    ],
)
def test_reachable_line_644_role_plants_flip_the_brine_flux_zero(
    site, old, new, monkeypatch
):
    args = _reachable_flush_role_fixture()
    temperature = np.asarray(args[_ARRAY_NAMES.index("t_i")])
    salinity = np.asarray(args[_ARRAY_NAMES.index("sz_i")])
    initial_flux = _bits(args[_ARRAY_NAMES.index("sfx_bri")])
    assert np.all(temperature < NEMO_SI3_CONSTANTS_CONFIG.T0 - 1.0e-6)
    assert np.all(salinity == NML["rn_simin"])
    np.testing.assert_array_equal(initial_flux, [np.uint64(1 << 63), 0])

    protected = jax.jit(lambda *values: _role_model(NML, *values))(*args)
    mutant = _source_mutant(bl._si3_ice_thd_sal4_core, old, new)
    monkeypatch.setattr(bl, "_si3_ice_thd_sal4_core", mutant)
    planted = jax.jit(lambda *values: _role_model(NML, *values))(*args)
    changed = [
        np.count_nonzero(_bits(got) != _bits(want))
        for got, want in zip(planted, protected)
    ]
    np.testing.assert_array_equal(protected.nflushing, [1, 1])
    np.testing.assert_array_equal(_bits(protected.sfx_bri), [0, 0])
    np.testing.assert_array_equal(
        _bits(planted.sfx_bri), [np.uint64(1 << 63), 0]
    )
    assert changed == [0, 1, 0, 0, 0, 0, 0, 0, 0], site


def test_masked_signed_zero_paths_are_checked_through_the_salinity_model():
    """Melted boundary columns drive +/-0 volume and -0 ``zs_brmax`` paths."""

    T0 = NEMO_SI3_CONSTANTS_CONFIG.T0
    arrays = {
        "h_i": [0.0, 0.0], "t_su": [T0, T0],
        "t_i": [np.full(10, T0), np.full(10, T0)],
        "sz_i": [np.full(10, 0.0), np.full(10, -0.0)],
        "t_bo": [T0 - 1.8, T0 - 1.8], "sss": [32.0, 32.0],
        "a_i": [0.0, 0.0], "dh_i_sum": [0.0, 0.0],
        "dh_s_sum": [0.0, 0.0], "sfx_bri": [0.0, 0.0],
        "sfx_res": [0.0, 0.0],
    }
    args = tuple(jnp.asarray(arrays[name]) for name in _ARRAY_NAMES)
    mask = jnp.zeros_like(args[_ARRAY_NAMES.index("sz_i")])
    volume_numerator = bl.nemo_source_round(
        mask * args[_ARRAY_NAMES.index("sz_i")]
    )
    assert np.all(_bits(volume_numerator[0]) == 0)
    assert np.all(_bits(volume_numerator[1]) == np.uint64(1 << 63))
    brine = bl._sal4_vc19_brine(
        args[_ARRAY_NAMES.index("t_i")], NEMO_SI3_CONSTANTS_CONFIG.T0
    )
    assert np.all(_bits(brine) == np.uint64(1 << 63))

    out = jax.jit(lambda *values: _role_model(NML, *values))(*args)
    assert np.all(np.asarray(out.ndrainage) == 0)
    assert np.all(np.asarray(out.nflushing) == 0)
    for leaf in out:
        values = np.asarray(leaf)
        if np.issubdtype(values.dtype, np.floating):
            np.testing.assert_array_equal(_bits(values[0]), _bits(values[1]))
        else:
            np.testing.assert_array_equal(values[0], values[1])


def test_temp1_two_million_state_scan_excludes_t0_for_positive_salinity():
    """TEMP1's physical arm stays below T0 throughout a broad fp64 scan."""

    n_energy, n_salinity = 1000, 2000
    energy_axis = jnp.linspace(0.0, 1.0e9, n_energy, dtype=jnp.float64)
    salinity_axis = jnp.linspace(0.1, 35.0, n_salinity, dtype=jnp.float64)
    energy = jnp.broadcast_to(
        energy_axis[:, None], (n_energy, n_salinity)
    ).reshape(-1, 1)
    salinity = jnp.broadcast_to(
        salinity_axis[None, :], (n_energy, n_salinity)
    ).reshape(-1, 1)
    thickness = jnp.ones(n_energy * n_salinity, dtype=jnp.float64)
    temperature = jax.jit(
        lambda h_i, e_i, sz_i: bl.si3_ice_thd_temp(
            h_i=h_i, e_i=e_i, sz_i=sz_i,
            constants=NEMO_SI3_CONSTANTS_CONFIG,
        )
    )(thickness, energy, salinity)
    gap = NEMO_SI3_CONSTANTS_CONFIG.T0 - np.asarray(temperature)
    assert gap.size == 2_000_000
    assert np.count_nonzero(
        np.asarray(temperature) == NEMO_SI3_CONSTANTS_CONFIG.T0
    ) == 0
    assert 5.399e-3 < float(np.min(gap)) < 5.401e-3

    boundary = bl.si3_ice_thd_temp(
        h_i=jnp.asarray([0.0, 1.0]), e_i=jnp.zeros((2, 1)),
        sz_i=jnp.zeros((2, 1)), constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    np.testing.assert_array_equal(boundary, NEMO_SI3_CONSTANTS_CONFIG.T0)


def test_five_category_o9_sal_temp2_composition_matches_scalar_oracle():
    namelist = {**o9_fixture._deck(), **NML}
    categories = o9_fixture._categories(seed=19)
    assert len(categories) == 5
    checked_columns = 0
    salinity_active = 0
    below_floor = 0
    melted_temperature = 0
    thin_positive = 0
    zero_salt_residue = 0
    excluded_from_flush = 0
    flush_eligible = 0
    flush_with_melted_temperature = 0
    for jl, cells in enumerate(categories):
        entry, fluxes = o9_fixture._entry(cells)
        sfx_bri = jnp.asarray(np.linspace(-0.0, jl + 1.0, len(cells)))
        stages = jax.jit(
            lambda e, f, b: bl.si3_ice_thd_o10_category_stages(
                entry=e, fluxes=f, sfx_bri=b, namelist=namelist,
                rDt_ice=RDT, r1_Dt_ice=R1_RDT, r1_nlay_i=R1_NLAY,
                rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W, gravity=GRAVITY,
                constants=NEMO_SI3_CONSTANTS_CONFIG,
            )
        )(entry, fluxes, sfx_bri)
        sal_rows = []
        temp_rows = []
        for i in range(len(cells)):
            o9 = stages.o9
            row = _oracle_column(
                h_i=float(o9.dh.h_i[i]), t_su=float(o9.dh.t_su[i]),
                t_i=np.asarray(o9.t_i[i]).tolist(),
                sz_i=np.asarray(o9.dh.sz_i[i]).tolist(),
                t_bo=float(entry["t_bo"][i]), sss=float(entry["sss"][i]),
                a_i=float(o9.dh.a_i[i]),
                dh_i_sum=float(o9.dh.dh.dh_i_sum[i]),
                dh_s_sum=float(o9.dh.dh.dh_s_sum[i]),
                sfx_bri=float(sfx_bri[i]),
                sfx_res=float(o9.dh.fluxes.sfx_res[i]),
                T0=NEMO_SI3_CONSTANTS_CONFIG.T0,
            )
            sal_rows.append(row)
            temp_rows.append(_temperature_from_enthalpy(
                np.asarray(o9.dh.e_i[i]).tolist(), row[0], float(o9.dh.h_i[i])
            ))

            h_i = float(o9.dh.h_i[i])
            salt = np.asarray(o9.dh.sz_i[i])
            temperature = np.asarray(o9.t_i[i])
            has_melted_temperature = np.all(
                temperature == NEMO_SI3_CONSTANTS_CONFIG.T0
            )
            below_floor += int(np.any(salt < NML["rn_simin"]))
            melted_temperature += int(has_melted_temperature)
            melting = float(o9.dh.dh.dh_i_sum[i] + o9.dh.dh.dh_s_sum[i]) < 0.0
            flush = h_i >= NML["rn_sal_himin"] and melting
            flush_eligible += int(flush)
            flush_with_melted_temperature += int(flush and has_melted_temperature)
            thin_positive += int(
                0.0 < h_i < NML["rn_sal_himin"] and np.all(salt > 0.0)
            )
            zero_salt_residue += int(
                0.0 < h_i < NML["rn_sal_himin"] and np.all(salt == 0.0)
            )
            excluded_from_flush += int(h_i < NML["rn_sal_himin"] and not flush)
            if h_i >= NML["rn_sal_himin"]:
                salinity_active += 1
                assert np.all(salt >= NML["rn_simin"])
                upper = (
                    NEMO_SI3_CONSTANTS_CONFIG.T0
                    - NEMO_SI3_CONSTANTS_CONFIG.liquidus_slope * NML["rn_simin"]
                )
                assert np.all(temperature <= upper)
            checked_columns += 1

        oracle_sal = tuple(np.asarray([row[k] for row in sal_rows]) for k in range(9))
        for got, want in zip(stages.sal, oracle_sal):
            if np.issubdtype(np.asarray(want).dtype, np.integer):
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_array_equal(_bits(got), _bits(want))
        np.testing.assert_array_equal(_bits(stages.t_i), _bits(temp_rows))
    assert checked_columns == sum(len(cells) for cells in categories)
    assert (salinity_active, below_floor, melted_temperature) == (19, 3, 3)
    assert (flush_eligible, flush_with_melted_temperature) == (3, 0)
    assert (thin_positive, zero_salt_residue, excluded_from_flush) == (5, 3, 8)


def test_salinity4_gradient_is_finite_nonzero_and_matches_central_difference():
    kwargs = _model_kwargs()
    # The first eight columns are smooth production states.  The ninth pins an
    # exact final-bound kink and the tenth pins a synthetic zero-mask branch;
    # neither belongs in a two-sided finite-difference claim.
    for name in _ARRAY_NAMES:
        kwargs[name] = kwargs[name][:8]
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


def test_melted_h_i_zero_column_has_finite_temperature_and_salinity_gradients():
    kwargs = _model_kwargs()
    temperature = kwargs.pop("t_i")
    salinity = kwargs.pop("sz_i")
    assert kwargs["h_i"][-1] == 0.0
    assert np.all(np.asarray(temperature[-1]) == NEMO_SI3_CONSTANTS_CONFIG.T0)

    def objective(t_i, sz_i):
        return jnp.sum(bl.si3_ice_thd_sal(t_i=t_i, sz_i=sz_i, **kwargs).sz_i)

    grad_t, grad_s = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        temperature, salinity
    )
    assert np.all(np.isfinite(np.asarray(grad_t[-1])))
    assert np.all(np.isfinite(np.asarray(grad_s[-1])))
    np.testing.assert_array_equal(grad_t[-1], jnp.zeros(10, dtype=jnp.float64))
    np.testing.assert_array_equal(grad_s[-1], jnp.ones(10, dtype=jnp.float64))


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

    tokens = {name: jnp.asarray([float(i)]) for i, name in enumerate((
        "h_i", "t_su", "t_i", "sz_i", "t_bo", "sss", "a_i",
        "dh_i_sum", "dh_s_sum", "sfx_bri", "sfx_res", "e_i",
    ), start=1)}
    sal_token = jnp.asarray([[7.0] * 10])
    fake_dh = SimpleNamespace(
        h_i=tokens["h_i"], t_su=tokens["t_su"], sz_i=tokens["sz_i"],
        a_i=tokens["a_i"], e_i=tokens["e_i"],
        dh=SimpleNamespace(
            dh_i_sum=tokens["dh_i_sum"], dh_s_sum=tokens["dh_s_sum"]
        ),
        fluxes=SimpleNamespace(sfx_res=tokens["sfx_res"]),
    )
    fake_o9 = SimpleNamespace(dh=fake_dh, t_i=tokens["t_i"])
    calls = []
    entry = {"t_bo": tokens["t_bo"], "sss": tokens["sss"]}
    fluxes_token = object()

    def o9_spy(**kwargs):
        assert kwargs["entry"] is entry
        assert kwargs["fluxes"] is fluxes_token
        assert kwargs["namelist"] is NML
        assert kwargs["constants"] is NEMO_SI3_CONSTANTS_CONFIG
        return fake_o9

    monkeypatch.setattr(bl, "si3_ice_thd_category_stages", o9_spy)

    def salinity_spy(**kwargs):
        calls.append("sal")
        expected = {
            "h_i": tokens["h_i"], "t_su": tokens["t_su"],
            "t_i": tokens["t_i"], "sz_i": tokens["sz_i"],
            "t_bo": tokens["t_bo"], "sss": tokens["sss"],
            "a_i": tokens["a_i"], "dh_i_sum": tokens["dh_i_sum"],
            "dh_s_sum": tokens["dh_s_sum"], "sfx_bri": tokens["sfx_bri"],
            "sfx_res": tokens["sfx_res"],
        }
        assert all(kwargs[name] is value for name, value in expected.items())
        assert kwargs["namelist"] is NML
        assert kwargs["constants"] is NEMO_SI3_CONSTANTS_CONFIG
        return bl.SI3SalinityOutputs(
            sal_token, tokens["sfx_bri"], tokens["sfx_res"],
            tokens["h_i"].astype(jnp.int32), tokens["t_su"].astype(jnp.int32),
            tokens["t_i"], tokens["sz_i"], tokens["t_bo"], tokens["sss"],
        )

    def temperature_spy(**kwargs):
        calls.append("temp2")
        assert kwargs["sz_i"] is sal_token
        assert kwargs["h_i"] is tokens["h_i"]
        assert kwargs["e_i"] is tokens["e_i"]
        assert kwargs["constants"] is NEMO_SI3_CONSTANTS_CONFIG
        return sal_token + 1.0

    monkeypatch.setattr(bl, "si3_ice_thd_sal", salinity_spy)
    monkeypatch.setattr(bl, "si3_ice_thd_temp", temperature_spy)
    result = bl.si3_ice_thd_o10_category_stages(
        entry=entry, fluxes=fluxes_token, sfx_bri=tokens["sfx_bri"],
        namelist=NML, rDt_ice=RDT, r1_Dt_ice=R1_RDT,
        r1_nlay_i=R1_NLAY, rho_i=RHO_I, rho_s=RHO_S, rho_w=RHO_W,
        gravity=GRAVITY, constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    assert calls == ["sal", "temp2"]
    np.testing.assert_array_equal(result.t_i, sal_token + 1.0)
