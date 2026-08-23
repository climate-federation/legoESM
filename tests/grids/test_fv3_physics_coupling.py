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

# the FV3 duo lane is fp64; enable x64 explicitly so the JAX-twin numerical
# identity is tested at the production precision, not JAX's default float32.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

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
    assert np.all(du[mask] != 0.0), "some intended u cells were not written"
    dv = vn - case["v"]
    maskv = np.zeros_like(dv, dtype=bool)
    maskv[NG:NG + N + 1, NG:NG + N] = True
    assert not np.any(dv[~maskv]), "v changed outside its D-grid compute block"
    assert np.all(dv[maskv] != 0.0), "some intended v cells were not written"


def test_jax_twin_matches_numpy_and_jit_equals_eager(case):
    import jax
    un, vn = update_dwinds_phys_duo(*_args(case))
    uj, vj = update_dwinds_phys_duo_jax(*_args(case))
    assert np.asarray(uj).dtype == np.float64, "jax twin demoted to float32"
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
    # the probe is non-vacuous only if this cell actually influences the loss
    assert abs(float(g[idx])) > 1e-6, "probed gradient is ~0; FD check is vacuous"
    pert = jnp.asarray(c["u_dt"]).at[idx].add(eps)
    minus = jnp.asarray(c["u_dt"]).at[idx].add(-eps)
    fd = (loss(pert) - loss(minus)) / (2 * eps)
    assert np.isclose(float(g[idx]), float(fd), rtol=1e-5, atol=1e-6)


# ---- Held-Suarez forcing: vectorized authority vs an independent loop ref ----
from legoesm.core.fv3_native_physics_coupling import (  # noqa: E402
    held_suarez_tend,
    _HS_DTY, _HS_DTZ, _HS_KAPPA, _HS_P0, _HS_T0, _HS_T_EQ0, _HS_H0, _HS_SDAY,
    _HS_KA_DAYS, _HS_KS_DAYS, _HS_KF_DAYS, _HS_SIGB, _HS_MS_DAYS, _HS_ST_DAYS,
    _HS_REF_RADIUS, _HS_STRAT_LAPSE, _HS_P_MESO, _HS_P_STRAT, _HS_TAU_PREF,
)

NPZ = 20
R_E = 6.371e6


def _hs_column(n=5, npz=NPZ, seed=1):
    """A plausible sigma-pressure column that triggers meso/strat/tropo."""
    rng = np.random.default_rng(seed)
    ps = 1.0e5 + rng.normal(scale=200.0, size=(n, n))          # ~surface Pa
    # half-level sigma from ~1e-5 (top) to 1 (surface): spans all three regimes
    sig = np.exp(np.linspace(np.log(1e-5), np.log(1.0), npz + 1))
    pe = ps[:, :, None] * sig[None, None, :]                    # (n,n,npz+1)
    peln = np.log(pe)
    delp = pe[:, :, 1:] - pe[:, :, :-1]
    pk = pe ** _HS_KAPPA
    pkz = (pk[:, :, 1:] - pk[:, :, :-1]) / (
        _HS_KAPPA * (peln[:, :, 1:] - peln[:, :, :-1]))
    pt = 250.0 + rng.normal(scale=20.0, size=(n, n, npz))       # theta-like
    ua = rng.normal(scale=10.0, size=(n, n, npz))
    va = rng.normal(scale=10.0, size=(n, n, npz))
    lat = rng.uniform(-1.4, 1.4, size=(n, n))
    return dict(pt=pt, ua=ua, va=va, delp=delp, peln=peln, pkz=pkz, pe=pe,
                lat=lat, pdt=1800.0)


def _hs_ref(pt, ua, va, delp, peln, pkz, pe, lat, pdt, strat, radius=R_E):
    """Independent triple-loop transcription of Held_Suarez_Tend (hswf.F90)."""
    ny, nx, npz = pt.shape
    rdt = 1.0 / pdt
    rr = radius / _HS_REF_RADIUS
    kf = _HS_SDAY * rr
    rkv = pdt / (_HS_KF_DAYS * kf); rka = pdt / (_HS_KA_DAYS * kf)
    rks = pdt / (_HS_KS_DAYS * kf)
    t_ms = _HS_MS_DAYS * rr; t_st = _HS_ST_DAYS * rr
    tau = (t_st - t_ms) / np.log(_HS_TAU_PREF)
    rms = pdt / (t_ms * _HS_SDAY); rmr = 1.0 / (1.0 + rms)
    rsgb = 1.0 / (1.0 - _HS_SIGB); ap0k = 1.0 / _HS_P0 ** _HS_KAPPA
    algpk = np.log(ap0k)
    t_dt = np.zeros((ny, nx, npz)); u_dt = np.zeros((ny, nx, npz))
    v_dt = np.zeros((ny, nx, npz))
    for j in range(ny):
        for i in range(nx):
            ps = pe[j, i, npz]
            plc = delp[j, i, :] / (peln[j, i, 1:] - peln[j, i, :-1])
            teq = np.zeros(npz + 1)
            la = lat[j, i]
            for k in range(npz - 1, -1, -1):
                plk = plc[k]; ptk = pt[j, i, k]; pkzk = pkz[j, i, k]
                tey = ap0k * (_HS_T_EQ0 - _HS_DTY * np.sin(la) ** 2)
                tez = _HS_DTZ * (ap0k / _HS_KAPPA) * np.cos(la) ** 2
                if strat and plk <= _HS_P_MESO:
                    dz = _HS_H0 * np.log(plc[k + 1] / plk)
                    teq[k] = teq[k + 1] - _HS_STRAT_LAPSE * np.cos(la) * dz
                    t_dt[j, i, k] += ((ptk + rms * teq[k]) * rmr - ptk) * rdt
                elif strat and _HS_P_MESO < plk <= _HS_P_STRAT:
                    dz = _HS_H0 * np.log(plc[k + 1] / plk)
                    relx = pdt / ((t_ms + tau * np.log(0.01 * plk)) * _HS_SDAY)
                    teq[k] = teq[k + 1] + _HS_STRAT_LAPSE * np.cos(la) * dz
                    t_dt[j, i, k] += relx * (teq[k] - ptk) / (1.0 + relx) * rdt
                else:
                    sigl = plk / ps
                    f1 = max(0.0, (sigl - _HS_SIGB) * rsgb)
                    tq = tey - tez * (np.log(pkzk) + algpk)
                    teq[k] = max(_HS_T0, tq * pkzk)
                    rkt = rka + (rks - rka) * f1 * np.cos(la) ** 4
                    t_dt[j, i, k] += rkt * (teq[k] - ptk) / (1.0 + rkt) * rdt
                    sf = (sigl - _HS_SIGB) * rsgb * rkv
                    if sf > 0.0:
                        tmp = sf / (1.0 + sf) * rdt
                        u_dt[j, i, k] -= (ua[j, i, k] + u_dt[j, i, k]) * tmp
                        v_dt[j, i, k] -= (va[j, i, k] + v_dt[j, i, k]) * tmp
    return t_dt, u_dt, v_dt


@pytest.mark.parametrize("strat", [True, False])
def test_held_suarez_matches_the_loop_reference(strat):
    c = _hs_column()
    a = held_suarez_tend(c["pt"], c["ua"], c["va"], c["delp"], c["peln"],
                         c["pkz"], c["pe"], c["lat"], c["pdt"], strat=strat,
                         radius=R_E)
    r = _hs_ref(c["pt"], c["ua"], c["va"], c["delp"], c["peln"], c["pkz"],
                c["pe"], c["lat"], c["pdt"], strat)
    for got, ref, nm in zip(a, r, ("t_dt", "u_dt", "v_dt")):
        assert np.allclose(got, ref, atol=1e-15, rtol=1e-11), \
            f"{nm} strat={strat}: {np.abs(got - ref).max():.3e}"


def test_held_suarez_exercises_all_three_regimes():
    """The column must actually hit meso/strat/tropo, else the ref is vacuous."""
    c = _hs_column()
    pl = c["delp"] / (c["peln"][:, :, 1:] - c["peln"][:, :, :-1])
    assert (pl <= _HS_P_MESO).any(), "no mesosphere levels"
    assert ((pl > _HS_P_MESO) & (pl <= _HS_P_STRAT)).any(), "no stratosphere"
    assert (pl > _HS_P_STRAT).any(), "no troposphere"


def test_held_suarez_friction_opposes_wind_and_only_in_boundary_layer():
    c = _hs_column()
    _, u_dt, v_dt = held_suarez_tend(
        c["pt"], c["ua"], c["va"], c["delp"], c["peln"], c["pkz"], c["pe"],
        c["lat"], c["pdt"], strat=True, radius=R_E)
    # drag opposes the wind where it acts
    acts = u_dt != 0.0
    assert np.all(u_dt[acts] * c["ua"][acts] <= 0.0), "friction not opposing u"
    # and only below sigma_b (boundary layer) -- top levels untouched
    assert not u_dt[:, :, 0].any(), "friction acting at the model top"
