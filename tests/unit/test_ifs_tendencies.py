"""Unit tests for the cudtdqn port (ifs_convective_tendencies + solver).

Run with JAX_ENABLE_X64=1 (default here); all tolerances are float64 unless
stated otherwise. A synthetic 30-level column, surface-last, is built by
``_column_inputs``; no upstream trigger/ascent modules are involved.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection._ifs_tendencies import IFSTendencyConfig, bidiagonal_forward, ifs_convective_tendencies  # noqa: E501

CFG = IFSTendencyConfig()  # rmfadvw = rmfadvwdd = 0, rmfsoltq = 1

NLEV = 30
KEY = jax.random.PRNGKey(0)


def _column_inputs(nlev=NLEV, with_mass_flux=True):
    """Two synthetic columns: col 0 updraught only, col 1 up+down draught."""
    ncol = 2
    p_half = np.linspace(1.0e4, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    j = np.broadcast_to(np.arange(nlev)[None, :], (ncol, nlev))
    T = 300.0 - 60.0 * j / nlev
    q = 1.0e-2 * np.exp(-j / 10.0) + 1.0e-4
    qs = 1.2 * q
    geo_full = 5.0e4 - 300.0 * (nlev - 1 - j)
    geo_half = 0.5 * (geo_full + np.concatenate([geo_full[:, :1], geo_full[:, :-1]], 1))
    T_h = 0.5 * (T + np.concatenate([T[:, :1], T[:, :-1]], 1))
    q_h = 0.5 * (q + np.concatenate([q[:, :1], q[:, :-1]], 1))

    # flux profiles: zero above cloud top (j < k_ctop-1), decaying downward
    k_ctop = np.array([5, 6])
    k_dtop = np.array([15, 15])
    prof = np.maximum(0.0, 1.0 - j / nlev) * (j >= (k_ctop[:, None] - 1))

    def f(key, scale):
        z = jax.random.normal(jax.random.fold_in(KEY, key), (ncol, nlev))
        return scale * prof * (1.0 + 0.1 * np.asarray(z))

    M_u = f(1, 0.01) if with_mass_flux else np.zeros((ncol, nlev))
    M_d = np.zeros((ncol, nlev))
    M_d[1] = -0.5 * M_u[1]
    return dict(
        T=jnp.array(T), q=jnp.array(q), qs=jnp.array(qs),
        p_full=jnp.array(p_full), p_half=jnp.array(p_half),
        geo_full=jnp.array(geo_full), geo_half=jnp.array(geo_half),
        T_h=jnp.array(T_h), q_h=jnp.array(q_h),
        ldcum=jnp.array([True, True]), ktype=jnp.array([1, 2]),
        k_ctop=jnp.array(k_ctop), k_dtop=jnp.array(k_dtop),
        lddraf=jnp.array([False, True]),
        M_u=jnp.array(M_u), M_d=jnp.array(M_d),
        PMFUS=jnp.array(f(2, 100.0)), PMFUQ=jnp.array(f(3, 1.0e-4)),
        PMFUL=jnp.array(f(4, 1.0e-5)),
        PMFDS=jnp.array(f(5, 50.0)), PMFDQ=jnp.array(f(6, 5.0e-5)),
        PLUDE=jnp.array(f(7, 1.0e-5)), PDMFUP=jnp.array(f(8, 1.0e-5)),
    )


def _call(inp, dt, cfg=CFG):
    return ifs_convective_tendencies(dt=jnp.asarray(dt, inp["T"].dtype), cfg=cfg, **inp)


# 1. bidiagonal_forward solves B[j]U[j] + A[j]U[j-1] = R[j] inside the mask.
def test_bidiagonal_residual():
    ncol = 3
    A = jax.random.uniform(KEY, (ncol, NLEV), minval=-0.5, maxval=0.5)
    B = 1.0 + jax.random.uniform(KEY, (ncol, NLEV))
    R = jax.random.uniform(KEY, (ncol, NLEV), minval=-1.0, maxval=1.0)
    k_ctop = jnp.array([4, 6, 9])
    jlev = jnp.arange(NLEV)[None, :]
    mask = jlev >= (k_ctop[:, None] - 1)
    U = bidiagonal_forward(A, B, R, mask, k_ctop)
    Uprev = jnp.concatenate([jnp.zeros((ncol, 1)), U[:, :-1]], 1)
    resid = jnp.where(mask, B * U + A * Uprev - R, U)
    # tolerance: abs 1e-12 (float64, condition numbers O(1))
    assert jnp.max(jnp.abs(resid)) < 1e-12
    assert jnp.all(U[:, 0] == 0.0)  # outside mask (above cloud top)


# 2. Column moisture budget: telescoping of the flux form (PLUDE=PDMFUP=0).
# 2. Column moisture budget, flux form (zero mass flux configuration).
#
# With M_u = M_d = 0 the bidiagonal system is the identity (B = 1, A = 0), so
# the solver is inactive and the returned tendencies equal the explicit
# flux-form ZDQDT of cudtdqn.F90:445-455; moreover the module's recomputed
# fluxes ZMFUQ/ZMFDQ equal the raw PMFUQ/PMFDQ (the ZQ correction is
# multiplied by the mass flux). PLUDE, PDMFUP and PMFUL are genuine sources
# and are kept non-zero.
#
# Only levels j >= k_ctop - 1 are updated (LLCUMBAS, cudtdqn.F90:410), so the
# upper boundary flux sits at a = k_ctop - 1, PER COLUMN (a = 4 and 5 here).
# The KLEV-indexed flux is the interface above the bottom layer, not a surface
# flux; the bottom branch (cudtdqn.F90:445 bottom) imposes zero convective
# transport at KLEV+1, so the KLEV flux never enters the column budget:
#
#   sum_{k=a..N} m_k dq_k/dt = -Q_a - L_a - sum_{k=a..N-1} D_k - sum_{k=a..N} P_k
#
# with m_k = dp_k/g, Q = PMFUQ + PMFDQ, L = PMFUL, D = PLUDE, P = PDMFUP
# (the PLUDE sum excludes the bottom level, where PLUDE does not appear).
def test_moisture_budget():
    inp = _column_inputs(with_mass_flux=False)  # recomputed fluxes == raw
    _, dq_dt, _ = _call(inp, dt=600.0)
    ncol, nlev = inp["T"].shape
    jlev = np.arange(nlev)[None, :]
    a = np.asarray(inp["k_ctop"]) - 1  # first solved level, per column (4, 5)
    cols = np.arange(ncol)

    dp = np.asarray(inp["p_half"][:, 1:] - inp["p_half"][:, :-1])
    m = dp / constants.g  # kg m-2 per layer

    Q = np.asarray(inp["PMFUQ"]) + np.asarray(inp["PMFDQ"])
    L = np.asarray(inp["PMFUL"])
    D = np.asarray(inp["PLUDE"])
    P = np.asarray(inp["PDMFUP"])

    got = np.sum(np.asarray(dq_dt) * m, axis=1)
    sum_D = np.sum(np.where((jlev >= a[:, None]) & (jlev <= nlev - 2), D, 0.0), axis=1)
    sum_P = np.sum(np.where(jlev >= a[:, None], P, 0.0), axis=1)
    expected = -Q[cols, a] - L[cols, a] - sum_D - sum_P
    # tolerance: abs 1e-12 kg m-2 s-1 (exact telescoping in float64; fluxes
    # are O(1e-4), 30-level roundoff accumulation is ~1e-17)
    assert np.max(np.abs(got - expected)) < 1e-12


# 3. Column enthalpy counterpart for dT_dt.
# 3. Column enthalpy budget and the moist-enthalpy identity.
#
# Part 1 uses the zero-mass-flux configuration (recomputed ZMFUS/ZMFDS equal
# the raw PMFUS/PMFDS, solver inactive), keeping PMFUL/PLUDE/PDMFUP as genuine
# sources. As for moisture, the upper boundary flux is at a = k_ctop - 1 per
# column and the KLEV flux is an internal interface, not a surface flux:
#
#   sum_{k=a..N} m_k c_pd dT_k/dt
#       = -S_a + L_v L_a + L_v sum_{k=a..N-1} D_k + L_v sum_{k=a..N} P_k
#
# Part 2 runs the FULL mass-flux fixture and asserts the moist-enthalpy
# identity, in which the latent-heat sources (PMFUL/PLUDE/PDMFUP) cancel:
#
#   sum_k m_k (c_pd dT_k/dt + L_v dq_k/dt) = -S_a - L_v Q_a
#
# Here S_a/Q_a are the RECOMPUTED (post-ZS/ZQ) fluxes at level a; they are
# reproduced below exactly as in cudtdqn.F90:244-258 with ZIMP = 0. At j = a
# the downdraught window is inactive in this fixture (k_dtop = 15 > a), so
# only the updraught flux carries the -M*ZS/-M*ZQ correction.
def test_enthalpy_budget():
    # ---- Part 1: dry-enthalpy budget, zero mass flux (recomputed == raw) --
    inp = _column_inputs(with_mass_flux=False)
    dT_dt, _, _ = _call(inp, dt=600.0)
    ncol, nlev = inp["T"].shape
    jlev = np.arange(nlev)[None, :]
    a = np.asarray(inp["k_ctop"]) - 1  # first solved level, per column (4, 5)
    cols = np.arange(ncol)

    dp = np.asarray(inp["p_half"][:, 1:] - inp["p_half"][:, :-1])
    m = dp / constants.g  # kg m-2 per layer

    S = np.asarray(inp["PMFUS"]) + np.asarray(inp["PMFDS"])
    L = np.asarray(inp["PMFUL"])
    D = np.asarray(inp["PLUDE"])
    P = np.asarray(inp["PDMFUP"])

    got = np.sum(constants.c_pd * np.asarray(dT_dt) * m, axis=1)
    sum_D = np.sum(np.where((jlev >= a[:, None]) & (jlev <= nlev - 2), D, 0.0), axis=1)
    sum_P = np.sum(np.where(jlev >= a[:, None], P, 0.0), axis=1)
    expected = -S[cols, a] + constants.L_v * (L[cols, a] + sum_D + sum_P)
    # tolerance: abs 1e-6 W m-2 (fluxes O(100), exact telescoping; float64
    # cancellation over 30 levels is ~1e-13)
    assert np.max(np.abs(got - expected)) < 1e-6

    # ---- Part 2: moist-enthalpy identity, FULL mass flux ------------------
    inp2 = _column_inputs(with_mass_flux=True)
    dT2, dq2, _ = _call(inp2, dt=600.0)

    T = inp2["T"]
    q = inp2["q"]
    qs = inp2["qs"]
    geo_full = inp2["geo_full"]
    T_h = inp2["T_h"][:, :nlev]
    q_h = inp2["q_h"][:, :nlev]
    geo_half = inp2["geo_half"][:, :nlev]

    def shift_up(x):  # x[j-1], duplicate at j = 0 (above cloud top: masked)
        return jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)

    T_a, q_a, geo_a = shift_up(T), shift_up(q), shift_up(geo_full)
    # ZS/ZQ as in cudtdqn.F90:244-258 with ZIMP = 1 - RMFSOLTQ = 0
    zgq = (q_h - q_a) / qs
    zgs = (constants.c_pd * (T_h - T_a) + geo_half - geo_a) / (constants.c_pd * T + geo_full)
    zs = constants.c_pd * zgs * T + geo_a + zgs * geo_full
    zq = zgq * qs

    # Recomputed fluxes at level a (downdraught uncorrected there: k_dtop > a)
    S_a = np.asarray(inp2["PMFUS"] - inp2["M_u"] * zs + inp2["PMFDS"])[cols, a]
    Q_a = np.asarray(inp2["PMFUQ"] - inp2["M_u"] * zq + inp2["PMFDQ"])[cols, a]

    got2 = np.sum(
        np.asarray(constants.c_pd * dT2 + constants.L_v * dq2) * m, axis=1
    )
    expected2 = -S_a - constants.L_v * Q_a
    # tolerance: abs 1e-6 W m-2 (fluxes O(100); the identity is exact up to
    # float64 roundoff of the 30-level sum and the ZS/ZQ reproduction)
    assert np.max(np.abs(got2 - expected2)) < 1e-6


# 4. dt -> 0: implicit solution converges (error roughly halves with dt).
def test_small_dt_convergence():
    inp = _column_inputs(with_mass_flux=True)
    dT1 = _call(inp, dt=1200.0)[0]
    dT2 = _call(inp, dt=600.0)[0]
    dT3 = _call(inp, dt=300.0)[0]
    err1 = jnp.max(jnp.abs(dT1 - dT2))
    err2 = jnp.max(jnp.abs(dT2 - dT3))
    assert err2 > 0.0  # non-trivial implicit correction
    assert float(err1) > 1.5 * float(err2)  # first-order convergence


# 5. LDCUM false: tendencies exactly zero everywhere (incl. above cloud top).
def test_no_convection_zero_tendency():
    inp = _column_inputs()
    inp = dict(inp)
    inp["ldcum"] = jnp.array([False, True])
    dT_dt, dq_dt, _ = _call(inp, dt=600.0)
    assert jnp.all(dT_dt[0] == 0.0) and jnp.all(dq_dt[0] == 0.0)
    inp2 = dict(inp)
    inp2["ldcum"] = jnp.array([True, True])
    dT2, dq2, _ = _call(inp2, dt=600.0)
    jlev = jnp.arange(NLEV)
    above = jlev[None, :] < (inp["k_ctop"][:, None] - 1)
    # every level above cloud top exactly zero
    assert jnp.all(jnp.where(above, dT2, 0.0) == 0.0)
    assert jnp.all(jnp.where(above, dq2, 0.0) == 0.0)
    # and something below, so the check is not vacuous
    assert jnp.any(dT2[~above] != 0.0)


# 6. jit/eager parity and a finite, non-zero gradient wrt T.
def test_jit_parity_and_grad():
    inp = _column_inputs()

    def scalar_TdTdT(T):
        dT_dt, _, _ = ifs_convective_tendencies(dt=jnp.asarray(600.0, T.dtype), cfg=CFG, **{**inp, "T": T})
        w = jnp.linspace(0.5, 1.5, NLEV)
        return jnp.sum(dT_dt * w[None, :])

    jitted = jax.jit(scalar_TdTdT)
    eager_out, eager_dT = jax.value_and_grad(scalar_TdTdT)(inp["T"])
    jit_out, jit_dT = jax.value_and_grad(jitted)(inp["T"])
    # tolerance: abs 1e-12 parity (float64, identical ops)
    assert abs(float(eager_out) - float(jit_out)) < 1e-12
    assert jnp.max(jnp.abs(eager_dT - jit_dT)) < 1e-12
    assert jnp.all(jnp.isfinite(eager_dT))
    assert jnp.max(jnp.abs(eager_dT)) > 1.0e-6  # gradient is non-zero
    assert jnp.any(eager_dT == 0.0)  # ...but zero above cloud top (masked)
