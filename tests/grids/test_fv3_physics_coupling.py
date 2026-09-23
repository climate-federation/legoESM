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

from legoesm import constants
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
    held_suarez_tend_jax,
    _HS_KAPPA, _HS_P_MESO, _HS_P_STRAT,   # column build + regime-coverage guard only
)

NPZ = 20
R_E = constants.R_earth


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
    rr = radius / constants.R_earth
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


def _hs_args(c, strat):
    return (c["pt"], c["ua"], c["va"], c["delp"], c["peln"], c["pkz"],
            c["pe"], c["lat"], c["pdt"], strat, R_E)


@pytest.mark.parametrize("strat", [True, False])
def test_held_suarez_jax_twin_matches_numpy_and_jit_equals_eager(strat):
    """The lax.scan twin vs a FRESH call of the NumPy authority on the same
    inputs (non-vacuous by construction), then jit==eager.  The fixture's
    regime coverage (meso/strat/tropo all populated) is gated separately by
    test_held_suarez_exercises_all_three_regimes, so the strat=True branch
    genuinely exercises the teq scan carry."""
    import jax
    c = _hs_column()
    ref = held_suarez_tend(*_hs_args(c, strat))
    eager = held_suarez_tend_jax(*_hs_args(c, strat))
    # strat (index 9) drives a Python `if` -> static; radius stays traced.
    jf = jax.jit(held_suarez_tend_jax, static_argnums=(9,))
    jitted = jf(*_hs_args(c, strat))
    for a, b, g, nm in zip(ref, eager, jitted, ("t_dt", "u_dt", "v_dt")):
        assert np.asarray(b).dtype == np.float64, f"{nm}: twin demoted to f32"
        np.testing.assert_allclose(np.asarray(b), a, rtol=0, atol=1e-12,
                                   err_msg=f"{nm} strat={strat} jax vs numpy")
        np.testing.assert_allclose(np.asarray(g), np.asarray(b), rtol=0,
                                   atol=1e-12,
                                   err_msg=f"{nm} strat={strat} jit vs eager")


def test_held_suarez_jax_differentiable_wrt_state():
    import jax
    import jax.numpy as jnp
    from tests.grids.fv3_gate_helpers import (
        assert_fd_gap_at_roundoff_floor,
        gated_check_grads,
    )
    c = _hs_column()
    args = tuple(jnp.asarray(c[k]) for k in ("pt", "ua", "va"))
    rest = tuple(jnp.asarray(c[k]) for k in ("delp", "peln", "pkz", "pe", "lat"))

    def f(pt, ua, va):
        return held_suarez_tend_jax(pt, ua, va, *rest, c["pdt"], strat=True,
                                    radius=R_E)

    # grads flow, finite and nonzero (mirrors the orchestrator's gate)
    def loss(pt, ua, va):
        t_dt, u_dt, v_dt = f(pt, ua, va)
        return jnp.sum(t_dt ** 2) + jnp.sum(u_dt ** 2) + jnp.sum(v_dt ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(*args)
    for g, nm in zip(grads, ("pt", "ua", "va")):
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad wrt {nm}"
        assert np.abs(np.asarray(g)).max() > 0.0, f"identically-zero grad {nm}"

    # The forcing is AFFINE in (pt, ua, va): every mask and coefficient (meso/
    # strat/fric, rkt, relx, tmp, the teq column) depends only on pressure and
    # lat.  So the eps^2 truncation ladder has no signal on these operands
    # (its precondition would refuse) and the affine-case instrument is the
    # roundoff-floor gate -- still FD-vs-VJP, still independent of the AD.
    # margin/atol are provisional headroom (>=1e5 over the paper-derived
    # roundoff); the gate passes empirically, and the floor helper has no
    # measure mode to pin them tighter, so they stay as conservative headroom.
    gated_check_grads("held_suarez_tend_jax pt/ua/va", f, args, order=2,
                      modes=("fwd", "rev"), eps=1e-3, atol=1e-12, rtol=1e-6)
    assert_fd_gap_at_roundoff_floor("held_suarez_tend_jax pt/ua/va", f, args,
                                    margin=10.0, eps=1e-3)


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


# ---------------------------------------------------------------------------
# apply_held_suarez_step: the promoted 3-pass six-face HS orchestration.
# The certified score lives in the parity runner (1.7645e-8 vs the Fortran
# oracle); this section pins (a) ONE implementation total (the script
# re-exports the core module's function object), and (b) the promoted copy
# runs on a real six-face duo context and behaves like the spec: pt/u/v move,
# pt halos and delp/press do not, and a context without the ext bundle is
# refused loudly.
# ---------------------------------------------------------------------------
from legoesm.core.fv3_native_physics_coupling import (  # noqa: E402
    apply_held_suarez_step,
)

KM_HS = 3


def test_apply_held_suarez_step_is_the_scripts_function():
    """No second implementation: the parity script's name resolves to the
    SAME function object as the core module's (PEP 562 re-export)."""
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "validate" \
        / "fv3_native"
    sys.path.insert(0, str(scripts))
    try:
        import full_step_oracle_parity as fsop
        assert fsop.apply_held_suarez_step is apply_held_suarez_step
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture(scope="module")
def hs_case():
    """Tiny six-face duo context (ext bundle on) + synthetic state/press.

    Pressures come from the lane's OWN ``p_var_hydrostatic`` (the same
    producer the driver adapter slices), so the per-face pe/peln/pkz
    layouts are exactly what ``apply_held_suarez_step`` was written
    against — not a hand-rolled lookalike.
    """
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    rng = np.random.default_rng(7)
    ptop, ps = 100.0, 1.0e5
    state = []
    delp6 = []
    for _t in range(6):
        delp = np.full((M, M, KM_HS), (ps - ptop) / KM_HS)
        state.append(dict(
            u=rng.normal(scale=5.0, size=(M, M + 1, KM_HS)),
            v=rng.normal(scale=5.0, size=(M + 1, M, KM_HS)),
            pt=250.0 + 5.0 * rng.normal(size=(M, M, KM_HS)),
            delp=delp,
        ))
        delp6.append(delp)
    press_stacked = p_var_hydrostatic(
        np.stack(delp6), ptop=ptop, akap=FV3_KAPPA, n=N, ng=NG, km=KM_HS)
    press = [{nm: np.asarray(press_stacked[nm][t])
              for nm in ("ps", "pe", "peln", "pk", "pkz")}
             for t in range(6)]
    return ctx, state, press


def test_apply_held_suarez_step_moves_the_right_fields(hs_case):
    """pt moves on the compute window ONLY (the scalar update is
    compute-domain-only); u/v move; delp and press are untouched."""
    ctx, state, press = hs_case
    snap = [{k: v.copy() for k, v in face.items()} for face in state]
    press_snap = [{k: np.asarray(v).copy() for k, v in face.items()}
                  for face in press]
    apply_held_suarez_step(ctx, state, press, dt=1800.0, n=N, ng=NG,
                           km=KM_HS, strat=True, backend="numpy")
    ci = slice(NG, NG + N)
    for t in range(6):
        for nm in ("u", "v", "pt"):
            assert np.isfinite(state[t][nm]).all(), (t, nm)
        # non-vacuous: the forcing actually moved the prognostics
        assert np.abs(state[t]["pt"][ci, ci]
                      - snap[t]["pt"][ci, ci]).max() > 0.0, t
        assert np.abs(state[t]["u"] - snap[t]["u"]).max() > 0.0, t
        assert np.abs(state[t]["v"] - snap[t]["v"]).max() > 0.0, t
        # pt halo untouched (fv_update_phys scalar loop is is:ie, js:je)
        halo = np.ones((M, M), bool)
        halo[ci, ci] = False
        np.testing.assert_array_equal(state[t]["pt"][halo],
                                      snap[t]["pt"][halo])
        # delp and every press field are read-only to the HS step
        np.testing.assert_array_equal(state[t]["delp"], snap[t]["delp"])
        for nm, v in press[t].items():
            np.testing.assert_array_equal(np.asarray(v),
                                          press_snap[t][nm], err_msg=nm)


def test_apply_held_suarez_step_refuses_missing_ectx(hs_case):
    ctx, state, press = hs_case
    ctx_no_ext = {**ctx, "ectx": None}
    with pytest.raises(ValueError, match="ectx"):
        apply_held_suarez_step(ctx_no_ext, state, press, dt=1800.0,
                               n=N, ng=NG, km=KM_HS, backend="numpy")


def test_apply_held_suarez_step_refuses_unknown_backend(hs_case):
    ctx, state, press = hs_case
    with pytest.raises(ValueError, match="backend"):
        apply_held_suarez_step(ctx, state, press, dt=1800.0, n=N, ng=NG,
                               km=KM_HS, backend="fortran")


# ---------------------------------------------------------------------------
# apply_held_suarez_step_sixface_jax: the face-stacked pure-JAX twin used
# where the NumPy authority cannot run (multi-process SPMD, whose leaves are
# not host-addressable).  Pinned to the authority on the SAME synthetic
# six-face case; the authority keeps the certified score.
# ---------------------------------------------------------------------------
from legoesm.core.fv3_native_physics_coupling import (  # noqa: E402
    apply_held_suarez_step_sixface_jax,
    stack_held_suarez_metrics,
)


@pytest.fixture(scope="module")
def hs_sixface_case(hs_case):
    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context
    ctx, state, press = hs_case
    tab = build_jax_duo_stepper_context(ctx).tab
    state6 = {nm: np.stack([face[nm] for face in state])
              for nm in ("u", "v", "pt", "delp")}
    press6 = {nm: np.stack([np.asarray(face[nm]) for face in press])
              for nm in ("pe", "peln", "pkz")}
    amat6, lat6, wv6 = stack_held_suarez_metrics(ctx)
    return ctx, state, press, tab, state6, press6, amat6, lat6, wv6


def _numpy_authority_result(ctx, state, press):
    st = [{k: np.array(v) for k, v in face.items()} for face in state]
    pr = [{k: np.asarray(v) for k, v in face.items()} for face in press]
    apply_held_suarez_step(ctx, st, pr, dt=1800.0, n=N, ng=NG, km=KM_HS,
                           strat=True, backend="numpy")
    return {nm: np.stack([face[nm] for face in st]) for nm in ("u", "v", "pt")}


def test_sixface_jax_matches_the_numpy_authority(hs_sixface_case):
    ctx, state, press, tab, state6, press6, amat6, lat6, wv6 = hs_sixface_case
    ref = _numpy_authority_result(ctx, state, press)
    out = apply_held_suarez_step_sixface_jax(
        state6, press6, tab, amat6, lat6, wv6, dt=1800.0, n=N, ng=NG,
        km=KM_HS, strat=True)
    for nm in ("u", "v", "pt"):
        got = np.asarray(out[nm])
        assert got.dtype == np.float64, nm
        assert np.isfinite(got).all(), nm
        err = np.abs(got - ref[nm]).max()
        assert np.allclose(got, ref[nm], atol=1e-11, rtol=1e-12), (nm, err)
    # non-vacuous: the step moved every prognostic it owns
    for nm in ("u", "v", "pt"):
        assert np.abs(np.asarray(out[nm]) - state6[nm]).max() > 0.0, nm
    # delp is read-only to HS and comes back untouched (same object)
    assert out["delp"] is state6["delp"]


def test_sixface_jax_jit_equals_eager(hs_sixface_case):
    import jax
    ctx, state, press, tab, state6, press6, amat6, lat6, wv6 = hs_sixface_case

    def f(state6, press6):
        return apply_held_suarez_step_sixface_jax(
            state6, press6, tab, amat6, lat6, wv6, dt=1800.0, n=N, ng=NG,
            km=KM_HS, strat=True)
    eager = f(state6, press6)
    jitted = jax.jit(f)(state6, press6)
    for nm in ("u", "v", "pt"):
        np.testing.assert_allclose(np.asarray(jitted[nm]),
                                   np.asarray(eager[nm]), rtol=0, atol=1e-13,
                                   err_msg=nm)


def test_sixface_jax_refuses_wrong_pt_shape(hs_sixface_case):
    ctx, state, press, tab, state6, press6, amat6, lat6, wv6 = hs_sixface_case
    bad = {**state6, "pt": state6["pt"][:, :, :, :KM_HS - 1]}
    with pytest.raises(ValueError, match="pt"):
        apply_held_suarez_step_sixface_jax(
            bad, press6, tab, amat6, lat6, wv6, dt=1800.0, n=N, ng=NG,
            km=KM_HS)


def test_stack_held_suarez_metrics_refuses_missing_ectx(hs_case):
    ctx, _, _ = hs_case
    with pytest.raises(ValueError, match="ectx"):
        stack_held_suarez_metrics({**ctx, "ectx": None})


@pytest.mark.parametrize("pass_name,target", [
    ("PASS0-dgrid", "exchange_dgrid_vector_halos"),
    ("PASS2-agrid", "exchange_agrid_scalar_halos"),
])
def test_sixface_jax_match_is_not_vacuous(hs_sixface_case, monkeypatch,
                                          pass_name, target):
    """Non-vacuity: with either halo exchange replaced by the identity the
    twin must LEAVE the 1e-11 envelope of the authority -- otherwise the
    match test above could not detect a dropped pass."""
    import legoesm.grids.fv3_duo_halos as halos
    ctx, state, press, tab, state6, press6, amat6, lat6, wv6 = hs_sixface_case
    ref = _numpy_authority_result(ctx, state, press)
    if target == "exchange_dgrid_vector_halos":
        monkeypatch.setattr(halos, target, lambda u, v, tab: (u, v))
    else:
        monkeypatch.setattr(halos, target, lambda f, tab, ring="stepper": f)
    out = apply_held_suarez_step_sixface_jax(
        state6, press6, tab, amat6, lat6, wv6, dt=1800.0, n=N, ng=NG,
        km=KM_HS, strat=True)
    worst = max(np.abs(np.asarray(out[nm]) - ref[nm]).max()
                for nm in ("u", "v", "pt"))
    assert worst > 1e-11, (pass_name, worst)
