"""update_dwinds_phys_duo: NumPy authority vs an independent loop reference,
its JAX twin, jit==eager, and differentiability.

The vectorized authority is checked against a triple-loop transcription of the
Fortran duo path (``model/fv_grid_utils.F90:3420-3540``) built here from scratch,
so the two share no slicing logic -- a shifted window or a swapped edge sum makes
them disagree.  The unit vectors come from the certified builder
(``compute_fv3_native_wind_vectors``), so this exercises the exact quantities the
production path will feed it.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_physics_coupling import (
    update_dwinds_phys_duo,
    update_dwinds_phys_duo_jax,
)

N, NG = 8, 3
M = N + 2 * NG


def _ref(u, v, u_dt, v_dt, dt, vlon, vlat, es1, ew2, n, ng):
    """Independent triple-loop transcription of the Fortran duo path."""
    dt5 = 0.5 * dt
    m = n + 2 * ng
    v3 = np.zeros((m, m, 3))
    for i in range(ng - 1, ng + n + 1):
        for j in range(ng - 1, ng + n + 1):
            v3[i, j, :] = u_dt[i, j] * vlon[i, j, :] + v_dt[i, j] * vlat[i, j, :]
    un, vn = u.copy(), v.copy()
    for j in range(ng, ng + n + 1):          # u: j in js:je+1, i in is:ie
        for i in range(ng, ng + n):
            ue = v3[i, j - 1, :] + v3[i, j, :]
            un[i, j] += dt5 * float((ue * es1[i, j, :]).sum())
    for j in range(ng, ng + n):              # v: j in js:je, i in is:ie+1
        for i in range(ng, ng + n + 1):
            ve = v3[i - 1, j, :] + v3[i, j, :]
            vn[i, j] += dt5 * float((ve * ew2[i, j, :]).sum())
    return un, vn


@pytest.fixture(scope="module")
def case():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.grids.fv3_native_metrics import compute_fv3_native_wind_vectors
    gs = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                    oracle_conventions=True)["gs6"][0]
    wv = compute_fv3_native_wind_vectors(
        gs["grid_lon"], gs["grid_lat"], gs["agrid_lon"], gs["agrid_lat"])
    rng = np.random.default_rng(0)
    u = rng.normal(size=(M, M + 1))
    v = rng.normal(size=(M + 1, M))
    u_dt = rng.normal(size=(M, M))
    v_dt = rng.normal(size=(M, M))
    return dict(u=u, v=v, u_dt=u_dt, v_dt=v_dt, dt=1800.0,
                vlon=wv["vlon"], vlat=wv["vlat"], es1=wv["es1"], ew2=wv["ew2"])


def _args(c):
    return (c["u"], c["v"], c["u_dt"], c["v_dt"], c["dt"],
            c["vlon"], c["vlat"], c["es1"], c["ew2"], N, NG)


def test_numpy_authority_matches_the_loop_reference(case):
    un, vn = update_dwinds_phys_duo(*_args(case))
    ur, vr = _ref(*_args(case))
    assert np.allclose(un, ur, atol=1e-13, rtol=0), np.abs(un - ur).max()
    assert np.allclose(vn, vr, atol=1e-13, rtol=0), np.abs(vn - vr).max()


def test_inputs_are_not_mutated(case):
    u0, v0 = case["u"].copy(), case["v"].copy()
    update_dwinds_phys_duo(*_args(case))
    assert np.array_equal(case["u"], u0) and np.array_equal(case["v"], v0)


def test_only_the_compute_block_moves(case):
    un, vn = update_dwinds_phys_duo(*_args(case))
    # u increment lives on j in [ng:ng+n+1], i in [ng:ng+n]; the rest is untouched
    du = un - case["u"]
    mask = np.zeros_like(du, dtype=bool)
    mask[NG:NG + N, NG:NG + N + 1] = True
    assert not np.any(du[~mask]), "u changed outside its D-grid compute block"
    dv = vn - case["v"]
    maskv = np.zeros_like(dv, dtype=bool)
    maskv[NG:NG + N + 1, NG:NG + N] = True
    assert not np.any(dv[~maskv]), "v changed outside its D-grid compute block"


def test_jax_twin_matches_numpy_and_jit_equals_eager(case):
    import jax
    un, vn = update_dwinds_phys_duo(*_args(case))
    uj, vj = update_dwinds_phys_duo_jax(*_args(case))
    assert np.allclose(np.asarray(uj), un, atol=1e-12, rtol=0)
    assert np.allclose(np.asarray(vj), vn, atol=1e-12, rtol=0)
    f = jax.jit(update_dwinds_phys_duo_jax, static_argnums=(9, 10))
    ujit, vjit = f(*_args(case))
    assert np.allclose(np.asarray(ujit), np.asarray(uj), atol=1e-12, rtol=0)
    assert np.allclose(np.asarray(vjit), np.asarray(vj), atol=1e-12, rtol=0)


def test_differentiable_wrt_the_tendency(case):
    import jax
    import jax.numpy as jnp
    c = case

    def loss(u_dt):
        uj, vj = update_dwinds_phys_duo_jax(
            c["u"], c["v"], u_dt, c["v_dt"], c["dt"],
            c["vlon"], c["vlat"], c["es1"], c["ew2"], N, NG)
        return jnp.sum(uj ** 2) + jnp.sum(vj ** 2)

    g = jax.grad(loss)(jnp.asarray(c["u_dt"]))
    assert np.all(np.isfinite(np.asarray(g)))
    # linear map -> central FD matches analytic grad at one probed cell
    eps = 1e-3
    idx = (NG + 2, NG + 2)
    pert = jnp.asarray(c["u_dt"]).at[idx].add(eps)
    minus = jnp.asarray(c["u_dt"]).at[idx].add(-eps)
    fd = (loss(pert) - loss(minus)) / (2 * eps)
    assert np.isclose(float(g[idx]), float(fd), rtol=1e-5, atol=1e-6)
