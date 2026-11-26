"""V1 kt-1 tke_tke statements (V1-selected, defaults unchanged).

NEMO 5.0.2's V1/ORCA2 build (gfortran -O3 -fno-tree-vectorize, no FMA, scalar
glibc ``exp``/``sin``) rounds each operation of the recurrences
zdftke.f90:466,472 and evaluates htau (:864) and the nn_etau=1 EXP (:500)
with scalar libm.  V1 kt 1: the card's en was non-bit in 31,802 cells.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.physics.vertical_mixing import tke as tke_mod
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

x64 = pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")


@pytest.fixture
def libm():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _system(n=3000, nw=12, seed=5):
    rng = np.random.default_rng(seed)
    lw = -rng.uniform(1e-3, 3.0, (n, nw))
    up = -rng.uniform(1e-3, 3.0, (n, nw))
    dg = 1.0 - lw - up + rng.uniform(0.0, 1.0, (n, nw))
    rhs = rng.uniform(1e-6, 1e-2, (n, nw))
    en1 = rng.uniform(1e-4, 1e-2, n)
    return up, lw, dg, rhs, en1


def _literal(up, lw, dg, rhs, en1):
    """zdftke.f90:458-473 in Fortran source order, one rounding per op."""
    n = dg.shape[-1]
    zdiag = np.concatenate([(1.0 / en1)[:, None], dg], -1)
    zlw = np.concatenate([np.ones_like(en1)[:, None], lw], -1)
    zup = np.concatenate([np.zeros_like(en1)[:, None], up], -1)
    en = np.concatenate([en1[:, None], rhs], -1)
    for k in range(1, n + 1):
        zdiag[:, k] = zdiag[:, k] - zlw[:, k] * zup[:, k - 1] / zdiag[:, k - 1]
    for k in range(1, n + 1):
        zlw[:, k] = en[:, k] - zlw[:, k] / zdiag[:, k - 1] * zlw[:, k - 1]
    en[:, n] = zlw[:, n] / zdiag[:, n]
    for k in range(n - 1, 0, -1):
        en[:, k] = (zlw[:, k] - zup[:, k] * en[:, k + 1]) / zdiag[:, k]
    return np.maximum(en[:, 1:], 1e-6)


def _card(rounded, up, lw, dg, rhs, en1):
    a = np.concatenate([np.zeros_like(en1)[:, None], lw], -1)
    b = np.concatenate([np.ones_like(en1)[:, None], dg], -1)
    c = np.concatenate([np.zeros_like(en1)[:, None], up], -1)
    r = np.concatenate([en1[:, None], rhs], -1)
    f = jax.jit(lambda *x: tke_mod._nemo_literal_tke_solve(
        *x, 1e-6, source_rounded=rounded))
    return np.asarray(f(*(jnp.asarray(v) for v in (a, b, c, r, en1,
                                                    np.ones_like(rhs)))))


@x64
def test_source_rounded_solve_matches_the_fortran_recurrences_under_jit():
    args = _system()
    np.testing.assert_array_equal(
        _card(True, *args).view(np.uint64), _literal(*args).view(np.uint64))


@x64
def test_compiled_default_solve_is_contracted_on_this_cpu():
    """Non-vacuity: the unrounded compiled solve differs on this host."""
    args = _system()
    if np.array_equal(_card(False, *args), _literal(*args)):
        pytest.skip("XLA did not contract on this CPU; the plant is unobservable")


def test_solve_flag_reaches_the_literal_solver(monkeypatch):
    seen = []
    real = tke_mod._nemo_literal_tke_solve

    def spy(*args, **kwargs):
        seen.append(kwargs.get("source_rounded", False))
        return real(*args, **kwargs)

    monkeypatch.setattr(tke_mod, "_nemo_literal_tke_solve", spy)
    for flag in (True, False):
        cfg = TKEConfig(
            tke_matrix_evaluation="nemo_literal", tke_buoyancy_sink="nemo_explicit",
            tke_solver_evaluation="nemo_literal",
            dissipation_discretization="nemo_1p5_split",
            surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0",
            tke_background=0.0, tke_surface_min=0.0,
            nemo_tke_solve_source_rounded=flag)
        tke_mod._solve_tke_backward_euler(
            e_old=jnp.asarray([[1.0, 0.7]]), K_M_old=jnp.asarray([[0.2, 0.3]]),
            K_H_old=jnp.zeros((1, 2)), P_s=jnp.asarray([[0.01, 0.02]]),
            N2=jnp.zeros((1, 2)), l_eps=jnp.ones((1, 2)),
            dz_half=jnp.asarray([[2.0, 3.0]]), surface_flux=jnp.asarray([0.0]),
            dt=2.0, cfg=cfg, dz_surface=jnp.asarray([1.0]),
            surface_dirichlet=jnp.asarray([0.8]), surface_bc_level="nemo_z0",
            K_M_surface=jnp.asarray([0.25]), w_active=jnp.ones((1, 2)),
            nemo_e3t=jnp.asarray([[1.0, 2.0, 3.0]]),
            dissl_old=jnp.asarray([[0.1, 0.2]]))
        assert seen[-1] is flag


GPHIT = np.linspace(-77.0104, 89.6, 4001)       # ORCA2's gphit span [deg]


@x64
def test_scalar_libm_htau_is_nemos_sin_of_the_folded_phase(libm):
    cfg = TKEConfig(tke_htau_evaluation="nemo_scalar_libm")
    got = np.asarray(jax.jit(lambda x: tke_mod._nemo_etau_htau(
        x, cfg, jnp.float64))(jnp.asarray(GPHIT)))
    c = math.pi / 180.0                             # rpi/180._wp, folded
    want = np.asarray([max(0.5, min(30.0, 45.0 * abs(math.sin(c * g))))
                       for g in GPHIT])
    np.testing.assert_array_equal(got, want)
    # red control: the historical arm round-trips through radians
    legacy = np.asarray(tke_mod._nemo_etau_htau(
        jnp.asarray(np.degrees(np.radians(GPHIT))), TKEConfig(), jnp.float64))
    assert not np.array_equal(legacy, want)


@x64
def test_scalar_libm_etau_exp_is_glibc_exp(libm):
    rng = np.random.default_rng(7)
    lat = rng.uniform(-77.0, 89.6, 64)
    depth = rng.uniform(0.0, 6000.0, (64, 30))
    e, taum = np.zeros_like(depth), np.full(64, 0.1)
    cfg = TKEConfig(etau_mode="below_ml", etau_htau_mode="latitude",
                    tke_htau_evaluation="nemo_scalar_libm",
                    tke_etau_exponential_evaluation="nemo_scalar_libm")

    def run(c):
        # traced latitudes: htau is an array, so the division stays a division
        return np.asarray(jax.jit(lambda d, la: tke_mod.nemo_etau_injection(
            jnp.asarray(e), jnp.asarray(taum), d, c, lat_deg=la))(
                jnp.asarray(depth), jnp.asarray(lat)))

    got = run(cfg)
    c = math.pi / 180.0
    htau = np.asarray([max(0.5, min(30.0, 45.0 * abs(math.sin(c * g)))) for g in lat])
    e_sfc = max(tke_mod._NEMO_TKE_EMIN0, tke_mod._NEMO_TKE_EBB / 1025.0 * 0.1)
    profile = np.vectorize(math.exp)(-depth / htau[:, None])
    want = cfg.etau_frac * e_sfc * profile
    sub = np.abs(want) < np.finfo(np.float64).tiny     # XLA flushes these (D127)
    np.testing.assert_array_equal(got[~sub], want[~sub])
    assert not np.any(got[sub])
    # red control: XLA's own exp on the same operands
    assert not np.array_equal(
        run(cfg._replace(tke_etau_exponential_evaluation="jax_expression")), got)


def test_scalar_literal_arms_use_native_policy_transcendentals():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    try:
        lat = jnp.asarray([10.0], dtype=jnp.float64)
        htau_cfg = TKEConfig(tke_htau_evaluation="nemo_scalar_libm")
        got_htau = tke_mod._nemo_etau_htau(lat, htau_cfg, jnp.float64)
        phase = tke_mod._nemo_binary64_round(
            lat * float.fromhex("0x1.1df46a2529d39p-6"))
        want_htau = jnp.maximum(0.5, jnp.minimum(30.0, 45.0 * jnp.abs(jnp.sin(phase))))
        np.testing.assert_array_equal(np.asarray(got_htau), np.asarray(want_htau))

        e = jnp.zeros((1, 2), dtype=jnp.float64)
        taum = jnp.asarray([0.1], dtype=jnp.float64)
        depth = jnp.asarray([[1.0, 20.0]], dtype=jnp.float64)
        etau_cfg = TKEConfig(
            etau_mode="below_ml", etau_htau_mode="constant10m",
            tke_etau_exponential_evaluation="nemo_scalar_libm")
        got_etau = tke_mod.nemo_etau_injection(e, taum, depth, etau_cfg)
        e_sfc = jnp.maximum(
            tke_mod._NEMO_TKE_EMIN0,
            tke_mod._NEMO_TKE_EBB / constants.rho_ocean * taum,
        )
        want_etau = e + etau_cfg.etau_frac * e_sfc[..., None] * jnp.exp(-depth / 10.0)
        np.testing.assert_array_equal(np.asarray(got_etau), np.asarray(want_etau))
    finally:
        set_policy(before)
