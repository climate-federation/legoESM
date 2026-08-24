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
    _HS_KAPPA, _HS_P_MESO, _HS_P_STRAT,   # column build + regime-coverage guard only
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
    """Independent triple-loop transcription of Held_Suarez_Tend (hswf.F90).

    Constants are FROZEN LOCAL literals (hswf.F90), not the production _HS_*, so a
    wrong port coefficient cannot corrupt both sides together (codex finding)."""
    ny, nx, npz = pt.shape
    sday = 86400.0; akap = 2.0 / 7.0; p0 = 1.0e5
    rdt = 1.0 / pdt
    rr = radius / 6371.0e3
    kf = sday * rr
    rkv = pdt / (1.0 * kf); rka = pdt / (40.0 * kf); rks = pdt / (4.0 * kf)
    t_ms = 10.0 * rr; t_st = 40.0 * rr
    tau = (t_st - t_ms) / np.log(100.0)
    rms = pdt / (t_ms * sday); rmr = 1.0 / (1.0 + rms)
    rsgb = 1.0 / (1.0 - 0.7); ap0k = 1.0 / p0 ** akap
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
                tey = ap0k * (315.0 - 60.0 * np.sin(la) ** 2)
                tez = 10.0 * (ap0k / akap) * np.cos(la) ** 2
                if strat and plk <= 1.0e2:
                    dz = 7.0 * np.log(plc[k + 1] / plk)
                    teq[k] = teq[k + 1] - 2.25 * np.cos(la) * dz
                    t_dt[j, i, k] += ((ptk + rms * teq[k]) * rmr - ptk) * rdt
                elif strat and 1.0e2 < plk <= 100.0e2:
                    dz = 7.0 * np.log(plc[k + 1] / plk)
                    relx = pdt / ((t_ms + tau * np.log(0.01 * plk)) * sday)
                    teq[k] = teq[k + 1] + 2.25 * np.cos(la) * dz
                    t_dt[j, i, k] += relx * (teq[k] - ptk) / (1.0 + relx) * rdt
                else:
                    sigl = plk / ps
                    f1 = max(0.0, (sigl - 0.7) * rsgb)
                    tq = tey - tez * (np.log(pkzk) + algpk)
                    teq[k] = max(200.0, tq * pkzk)
                    rkt = rka + (rks - rka) * f1 * np.cos(la) ** 4
                    t_dt[j, i, k] += rkt * (teq[k] - ptk) / (1.0 + rkt) * rdt
                    sf = (sigl - 0.7) * rsgb * rkv
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


# ---------------------------------------------------------------------------
# fv_update_phys_dry_duo: the dry-hydrostatic orchestrator that composes the
# scalar/A-grid update with the certified per-level D-grid wind update.
# ---------------------------------------------------------------------------
from legoesm.core.fv3_native_physics_coupling import (  # noqa: E402
    fv_update_phys_dry_duo,
    fv_update_phys_dry_duo_jax,
)

NPZ = 4


@pytest.fixture(scope="module")
def case3d(case):
    rng = np.random.default_rng(1)
    pt = 250.0 + rng.normal(size=(M, M, NPZ))
    ua = rng.normal(size=(M, M, NPZ))
    va = rng.normal(size=(M, M, NPZ))
    u = rng.normal(size=(M, M + 1, NPZ))
    v = rng.normal(size=(M + 1, M, NPZ))
    u_dt = rng.normal(size=(M, M, NPZ))
    v_dt = rng.normal(size=(M, M, NPZ))
    t_dt = rng.normal(size=(M, M, NPZ)) * 1e-3
    return dict(u=u, v=v, pt=pt, ua=ua, va=va, u_dt=u_dt, v_dt=v_dt, t_dt=t_dt,
                dt=1800.0, vlon=case["vlon"], vlat=case["vlat"],
                es1=case["es1"], ew2=case["ew2"])


def _args3d(c):
    return (c["u"], c["v"], c["pt"], c["ua"], c["va"], c["u_dt"], c["v_dt"],
            c["t_dt"], c["dt"], c["vlon"], c["vlat"], c["es1"], c["ew2"], NG)


def test_orchestrator_scalar_update_is_dry_and_compute_domain_only(case3d):
    """pt/ua/va advance by tendency*dt (dry cp factor = 1.0) on the COMPUTE
    DOMAIN only; halos are left exactly as passed (Fortran loops i=is:ie,
    j=js:je).  A full-array update -- the pre-fix behaviour -- fails the
    halo-unchanged check because t_dt/u_dt/v_dt are nonzero in the halo."""
    c = case3d
    _, _, pt_n, ua_n, va_n = fv_update_phys_dry_duo(*_args3d(c))
    ci = slice(NG, NG + N)
    np.testing.assert_allclose(pt_n[ci, ci], c["pt"][ci, ci] + c["t_dt"][ci, ci] * c["dt"], rtol=0, atol=0)
    np.testing.assert_allclose(ua_n[ci, ci], c["ua"][ci, ci] + c["u_dt"][ci, ci] * c["dt"], rtol=0, atol=0)
    np.testing.assert_allclose(va_n[ci, ci], c["va"][ci, ci] + c["v_dt"][ci, ci] * c["dt"], rtol=0, atol=0)
    for new_, old_ in ((pt_n, c["pt"]), (ua_n, c["ua"]), (va_n, c["va"])):
        halo = np.ones(new_.shape[:2], bool)
        halo[ci, ci] = False
        np.testing.assert_array_equal(new_[halo], old_[halo])


def test_orchestrator_dgrid_is_the_per_level_helper(case3d):
    """u_new/v_new stack the certified 2-D helper applied level by level;
    a mis-indexed loop (wrong k placement) makes this disagree."""
    c = case3d
    u_n, v_n, *_ = fv_update_phys_dry_duo(*_args3d(c))
    for k in range(NPZ):
        uk, vk = update_dwinds_phys_duo(
            c["u"][:, :, k], c["v"][:, :, k],
            c["u_dt"][:, :, k], c["v_dt"][:, :, k],
            c["dt"], c["vlon"], c["vlat"], c["es1"], c["ew2"], N, NG)
        np.testing.assert_allclose(u_n[:, :, k], uk, rtol=0, atol=0)
        np.testing.assert_allclose(v_n[:, :, k], vk, rtol=0, atol=0)


def test_orchestrator_does_not_mutate_inputs(case3d):
    c = case3d
    snap = {k: np.asarray(v).copy() for k, v in c.items()
            if isinstance(v, np.ndarray)}
    fv_update_phys_dry_duo(*_args3d(c))
    for k, v0 in snap.items():
        np.testing.assert_array_equal(c[k], v0, err_msg=f"input {k} mutated")


def test_orchestrator_jax_twin_matches_and_jit_equals_eager(case3d):
    import jax
    c = case3d
    ref = fv_update_phys_dry_duo(*_args3d(c))
    eager = fv_update_phys_dry_duo_jax(*_args3d(c))
    jitted = jax.jit(fv_update_phys_dry_duo_jax, static_argnums=(13,))(*_args3d(c))
    for a, b, g in zip(ref, eager, jitted):
        np.testing.assert_allclose(np.asarray(b), a, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(g), np.asarray(b), rtol=0, atol=1e-12)


def test_orchestrator_differentiable_wrt_tendencies(case3d):
    import jax
    import jax.numpy as jnp
    c = case3d

    def loss(t_dt, u_dt):
        u_n, v_n, pt_n, _, _ = fv_update_phys_dry_duo_jax(
            c["u"], c["v"], c["pt"], c["ua"], c["va"], u_dt, c["v_dt"],
            t_dt, c["dt"], c["vlon"], c["vlat"], c["es1"], c["ew2"], NG)
        return jnp.sum(pt_n ** 2) + jnp.sum(u_n ** 2) + jnp.sum(v_n ** 2)

    gt, gu = jax.grad(loss, argnums=(0, 1))(jnp.asarray(c["t_dt"]),
                                            jnp.asarray(c["u_dt"]))
    assert np.all(np.isfinite(np.asarray(gt))) and np.abs(np.asarray(gt)).max() > 0
    assert np.all(np.isfinite(np.asarray(gu))) and np.abs(np.asarray(gu)).max() > 0
