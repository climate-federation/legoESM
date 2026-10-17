"""NEMO rab_3d_t/bn2_t with every source operation rounded (V1-selected).

NEMO 5.0.2 (ORCA2 build, gfortran -O3 without FMA) rounds each multiply/add of
eosbn2.f90:1281-1333 (rab_3d_t, EOS-80 branch) and :1631-1637 (bn2_t). Under
jit XLA contracts pairs of them; V1 kt 1 showed 332,247 rn2 cells off by up to
8.2e-18, which owned the last over-bar viscosity cells.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean import eos as eos_mod
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

x64 = pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")


def _inputs(n=4000, nz=6, seed=3):
    rng = np.random.default_rng(seed)
    T = rng.uniform(-1.8, 30.0, (n, nz))
    S = rng.uniform(30.0, 38.0, (n, nz))
    gdept0 = np.cumsum(rng.uniform(5.0, 300.0, (n, nz)), axis=-1)
    gdepw0 = gdept0[..., :-1] + rng.uniform(0.1, 0.9, (n, nz - 1)) * np.diff(gdept0)
    stretch = 1.0 + rng.uniform(-1e-4, 1e-4, n)
    e3w = np.diff(gdept0) * stretch[:, None]
    return T, S, gdept0, gdepw0, stretch, e3w


def _literal(T, S, gdept0, gdepw0, stretch, e3w, g):
    """Fortran source order, one binary64 rounding per operation (numpy)."""
    c = eos_mod._ROQUET_EOS80
    s = stretch[:, None]
    zh = (gdept0 * s) * c["r1_Z0"]
    zt = T * c["r1_T0"]
    zs = np.sqrt(np.abs(S + c["rdeltaS"]) * c["r1_S0"])

    def poly(P):
        k = lambda n: c[P + n]  # noqa: E731
        zn3 = k("003")
        zn2 = k("012") * zt + k("102") * zs + k("002")
        zn1 = (((k("031") * zt + k("121") * zs + k("021")) * zt
                + (k("211") * zs + k("111")) * zs + k("011")) * zt
               + ((k("301") * zs + k("201")) * zs + k("101")) * zs + k("001"))
        zn0 = (((((k("050") * zt + k("140") * zs + k("040")) * zt
                  + (k("230") * zs + k("130")) * zs + k("030")) * zt
                 + ((k("320") * zs + k("220")) * zs + k("120")) * zs + k("020")) * zt
                + (((k("410") * zs + k("310")) * zs + k("210")) * zs + k("110")) * zs
                + k("010")) * zt
               + ((((k("500") * zs + k("400")) * zs + k("300")) * zs + k("200")) * zs
                  + k("100")) * zs + k("000"))
        return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0

    r1_rho0 = 1.0 / 1026.0
    alpha = poly("ALP") * r1_rho0
    beta = poly("BET") / zs * r1_rho0
    zrw = (((gdepw0 * s) - (gdept0[..., 1:] * s))
           / ((gdept0[..., :-1] * s) - (gdept0[..., 1:] * s)))
    zaw = alpha[..., 1:] * (1.0 - zrw) + alpha[..., :-1] * zrw
    zbw = beta[..., 1:] * (1.0 - zrw) + beta[..., :-1] * zrw
    return g * (zaw * (T[..., :-1] - T[..., 1:])
                - zbw * (S[..., :-1] - S[..., 1:])) / e3w


def _card(rounded, T, S, gdept0, gdepw0, stretch, e3w, g):
    s = stretch[:, None]
    f = jax.jit(lambda *a: eos_mod.compute_buoyancy_frequency_nemo_bn2(
        a[0], a[1], a[2] * s, a[3] * s, g=g, eos_form="eos80", e3w_int=a[4],
        zrw_evaluation="nemo_literal", zrw_gdept_0=a[2], zrw_gdepw_0=a[3],
        zrw_stretch=a[5], source_rounded=rounded))
    return np.asarray(f(*(jnp.asarray(v) for v in (T, S, gdept0, gdepw0, e3w, stretch))))


@x64
def test_source_rounded_bn2_matches_the_fortran_operation_order_under_jit():
    args = _inputs()
    lit = _literal(*args, constants.g)
    got = _card(True, *args, constants.g)
    np.testing.assert_array_equal(got.view(np.uint64), lit.view(np.uint64))


@x64
def test_compiled_default_is_contracted_on_this_cpu():
    """Non-vacuity: the default (legacy, unchanged) path differs from the
    literal order on this host, so the test above can fail."""
    args = _inputs()
    if np.array_equal(_card(False, *args, constants.g), _literal(*args, constants.g)):
        pytest.skip("XLA did not contract on this CPU; the plant is unobservable")


def test_source_rounded_refuses_the_simplified_eos():
    T, S, gdept0, gdepw0, _, e3w = _inputs(n=2, nz=3)
    with pytest.raises(ValueError, match="source_rounded=True is implemented"):
        eos_mod.compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept0),
            jnp.asarray(gdepw0), eos_form="seos", e3w_int=jnp.asarray(e3w),
            source_rounded=True)


def test_flag_refuses_a_closure_that_never_builds_the_step_entry_bundle():
    cfg = SimpleNamespace(
        physics=SimpleNamespace(vertical_mixing=SimpleNamespace(
            scheme="tke", tke=TKEConfig(nemo_bn2_source_rounded=True))),
        gm_redi=None)
    model = SimpleNamespace(z_coord=None, config=cfg)
    with pytest.raises(ValueError, match="nemo_bn2_source_rounded=True needs"):
        LatLonCGridOceanModel._tke_step_entry_n2_bundle(model, None)
