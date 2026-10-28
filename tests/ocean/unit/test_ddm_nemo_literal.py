"""NEMO-literal zdf_ddm (zdfddm.f90:107-175) and trazdf's separate salt matrix."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
    DoubleDiffusionConfig,
    nemo_zdf_ddm,
)
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_nemo_tracer_pair,
    implicit_vertical_diffusion_ocean_tracer_pair_dispatch,
)

CFG = DoubleDiffusionConfig(enabled=True, form="nemo_literal", rn_avts=1.0e-4, rn_hsbfr=1.6)


@pytest.fixture(autouse=True)
def _libm():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _fortran(avt, avm, zdt, zds, rn2, wmask, rn_avts=1.0e-4, rn_hsbfr=1.6):
    """zdfddm.f90:129-172 in Python floats (one rounding per operation)."""
    if abs(zds) <= 1.e-20:
        zds = 1.e-20
    zrau = max(1.e-20, zdt / zds)
    zmsks = 0.0 if rn2 + 1.e-12 <= 0.0 else 1.0 * wmask
    zmskf = 0.0 if zrau <= 1.0 else 1.0
    zmskd1 = 0.0 if zrau >= 1.0 else 1.0
    zmskd2 = 0.0 if zrau >= 0.5 else 1.0
    zmskd3 = 0.0 if (zrau <= 0.5 or zrau >= 1.0) else 1.0
    zinr = 1.0 / zrau
    zrr = zrau / rn_hsbfr
    zrr = zrr * zrr
    zavfs = rn_avts / (1.0 + zrr * zrr * zrr) * zmsks * zmskf
    zavft = 0.7 * zavfs * zinr
    zavdt = 1.3635e-6 * math.exp(4.6 * math.exp(-0.54 * (zinr - 1.0))) * zmsks * zmskd1
    zavds = zavdt * zmsks * ((1.85 * zrau - 0.85) * zmskd3 + 0.15 * zrau * zmskd2)
    avs = avt + zavfs + zavds
    avt_new = avt + zavft + zavdt
    avm_new = avm + max(zavft + zavdt, zavfs + zavds)
    return avt_new, avs, avm_new


def _cases():
    rng = np.random.default_rng(13)
    n = 4000
    zds = rng.normal(0.0, 1e-4, n)
    ratio = np.concatenate([rng.uniform(1.0, 3.0, n // 4),     # salt fingering
                            rng.uniform(0.5, 1.0, n // 4),     # diffusive, upper
                            rng.uniform(0.0, 0.5, n // 4),     # diffusive, lower
                            rng.uniform(-3.0, 0.0, n // 4)])   # single-signed -> 1e-20
    zdt = ratio * zds
    zds[:20] = rng.uniform(-1e-20, 1e-20, 20)                   # :129 floor
    zdt[20:30] = 0.0
    ratio_exact = np.array([0.5, 1.0, 1.6, 2.0])                 # mask edges
    zds[30:34] = 2.0 ** -12
    zdt[30:34] = ratio_exact * zds[30:34]
    rn2 = rng.normal(1e-5, 2e-5, n)
    rn2[40:45] = -1e-12                                         # :135 edge
    rn2[45:50] = -5e-13                                         # unstable, yet "stable" by :135
    wmask = (rng.uniform(size=n) > 0.1).astype(np.float64)
    avt = rng.uniform(1e-5, 1e-2, n)
    avm = rng.uniform(1e-4, 1e-2, n)
    return avt, avm, zdt, zds, rn2, wmask


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_literal_ddm_matches_scalar_fortran_bit_for_bit(compiled):
    avt, avm, zdt, zds, rn2, wmask = _cases()
    # jit: rn_hsbfr is a compile-time constant there, which XLA would fold
    # into a reciprocal multiply (V3 kt 1: 9,735 fingering cells off by an ulp)
    fn = jax.jit(lambda *a: nemo_zdf_ddm(*a, CFG)) if compiled else (
        lambda *a: nemo_zdf_ddm(*a, CFG))
    got = fn(*(jnp.asarray(x) for x in (avt, avm, zdt, zds, rn2, wmask)))
    want = np.array([_fortran(*args) for args in zip(avt, avm, zdt, zds, rn2, wmask)]).T
    for name, g, w in zip(("avt", "avs", "avm"), got, want):
        assert np.array_equal(np.asarray(g), w), (name, int(np.sum(np.asarray(g) != w)))
    avt_new, avs, _ = (np.asarray(x) for x in got)
    # every regime is exercised, and DDM moves both diffusivities
    assert np.sum(avs != avt_new) > 1000 and np.sum(avt_new != avt) > 1000


def test_literal_ddm_refuses_other_forms_and_runs_native_exp():
    args = tuple(jnp.ones(3) for _ in range(6))
    with pytest.raises(ValueError, match="nemo_literal"):
        nemo_zdf_ddm(*args, CFG._replace(form="hybrid"))
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    eager = nemo_zdf_ddm(*args, CFG)
    compiled = jax.jit(lambda *values: nemo_zdf_ddm(*values, CFG))(*args)
    for left, right in zip(eager, compiled, strict=True):
        assert np.isfinite(np.asarray(left)).all()
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _pair_inputs():
    rng = np.random.default_rng(3)
    shape = (5, 7, 6)
    e3t = rng.uniform(5.0, 50.0, shape)
    e3w = rng.uniform(5.0, 50.0, shape[:-1] + (shape[-1] - 1,))
    wet = np.ones(shape, bool)
    wet[0, 0, 4:] = False
    K1 = rng.uniform(1e-5, 1e-2, e3w.shape)
    K2 = K1 + rng.uniform(0.0, 1e-4, e3w.shape)
    c1, c2 = rng.normal(10.0, 3.0, shape) * e3t, rng.normal(35.0, 1.0, shape) * e3t
    return [jnp.asarray(x) for x in (c1, c2, K1, K2, e3t, e3w)] + [jnp.asarray(wet)]


def test_salt_matrix_is_trazdfs_own_assembly_on_avs():
    c1, c2, K1, K2, e3t, e3w, wet = _pair_inputs()
    t, s = implicit_vertical_diffusion_nemo_tracer_pair(c1, c2, K1, e3t, e3w, 5400.0, wet, K_2=K2)
    s_alone, t_alone = implicit_vertical_diffusion_nemo_tracer_pair(c2, c1, K2, e3t, e3w, 5400.0, wet)
    t_shared, s_shared = implicit_vertical_diffusion_nemo_tracer_pair(c1, c2, K1, e3t, e3w, 5400.0, wet)
    assert np.array_equal(np.asarray(t), np.asarray(t_shared))
    assert np.array_equal(np.asarray(s), np.asarray(s_alone))
    assert not np.array_equal(np.asarray(s), np.asarray(s_shared))
    _, s_none = implicit_vertical_diffusion_nemo_tracer_pair(
        c1, c2, K1, e3t, e3w, 5400.0, wet, K_2=None)
    assert np.array_equal(np.asarray(s_none), np.asarray(s_shared))
    with pytest.raises(ValueError, match="K_2 requires"):
        implicit_vertical_diffusion_ocean_tracer_pair_dispatch(
            c1, c2, c1, c2, K1, e3t, e3w, 5400.0, wet, K_2=K2)
