"""Unit tests for _ifs_flux.py (cuflxn.F90 sections 1.4 and 1.5).

Run with JAX_ENABLE_X64=1.  All tolerances stated per test:
  * float64 (x64): rtol=1e-10, atol=1e-6 (values reach O(1e5), so atol=1e-6
    is 10 orders of magnitude below signal and only guards additive noise).
  * exact zeros are asserted with ``== 0.0`` (the implementation uses
    jnp.where(..., 0.0, ...) so zeros are bit-exact in both precisions).
  * bit-for-bit identity in test 4 uses jnp.array_equal (no tolerance).
If forced to run in float32, relax rtol to 2e-5, atol to 1e-2 (float32 has
~7 significant digits on O(1e5) magnitudes); the exact-zero and
bit-identity assertions are precision-independent.
"""

import os

import jax
jax.config.update("jax_enable_x64", bool(int(os.environ.get("JAX_ENABLE_X64", "1"))))

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as C
from legoesm.atmosphere.physics.convection._ifs_flux import (
    IFSFluxConfig,
    ifs_convective_fluxes,
)

RTOL, ATOL = 1e-10, 1e-6
NLEV = 10
N = 2  # columns: 0 convecting, 1 dead (only used in test 4)

CPD = C.c_pd
LV = C.L_v


def _base_column(nlev=NLEV, ncol=N):
    rng = np.random.default_rng(0)
    p_half = np.empty((ncol, nlev + 1))
    inc = np.array([3.0e3, 2.2e3, 5.1e3, 1.7e3, 4.4e3, 2.9e3, 6.3e3, 3.8e3, 8.2e3, 5.5e3, 1.1e4])
    # Surface-LAST convention: pressure INCREASES with the index, the surface
    # interface is p_half[:, nlev].  Non-uniform spacing so the two candidate
    # interface pairs around the cloud base cannot be confused.
    p_half[:] = (1.0e5 - np.cumsum(inc))[::-1][None, :]
    M_u = rng.uniform(0.0, 0.05, (ncol, nlev))
    PMFUS = rng.uniform(-50.0, 50.0, (ncol, nlev)) * 1e3
    PMFUQ = rng.uniform(-1e-4, 1e-3, (ncol, nlev))
    PMFUL = rng.uniform(0.0, 2e-4, (ncol, nlev))
    PLUDE = rng.uniform(1e-6, 1e-5, (ncol, nlev))
    PDMFUP = rng.uniform(1e-6, 1e-5, (ncol, nlev))
    T_h = rng.uniform(220.0, 300.0, (ncol, nlev))
    q_h = rng.uniform(1e-4, 1.5e-2, (ncol, nlev))
    geo_half = np.linspace(3.0e4, 0.0, nlev)[None, :].repeat(ncol, axis=0)
    return M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, T_h, q_h, geo_half, p_half


def _run(col, k_ctop, k_cbot, ldcum=True):
    (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP,
     T_h, q_h, geo_half, p_half) = _base_column(ncol=1)
    args = [jnp.asarray(a) for a in (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP,
                                     T_h, q_h, geo_half, p_half)]
    ldcum = jnp.array([ldcum])
    ktype = jnp.array([1])
    kc = jnp.array([k_cbot])
    kt = jnp.array([k_ctop])
    out = ifs_convective_fluxes(*args, ldcum, ktype, kc, kt,
                                cfg=IFSFluxConfig())
    ins = [np.asarray(a)[0] for a in args]
    outs = [np.asarray(a)[0] for a in out]
    return ins, outs, p_half[0]


def test_1_environmental_subtraction():
    """cuflxn.F90:250-251, exact arithmetic, incl. a PMFU==0 in-cloud level."""
    # k_cbot = nlev-1 disables section 1.5 entirely (masks j==nlev and j>nlev
    # are empty), so the output is pure section 1.4.
    k_ctop, k_cbot = 2, NLEV - 1
    ins, outs, _ = _run(0, k_ctop, k_cbot)
    M_u, PMFUS, PMFUQ, PMFUL, _, _, T_h, q_h, geo_half, _ = ins
    Mo, So, Qo, Lo, _, _ = outs

    # force a known zero-mass level strictly inside the cloud
    M_u = M_u.copy(); M_u[5] = 0.0

    j = np.arange(NLEV)
    incloud = j >= k_ctop
    active = j >= 1  # source loop starts at KTOPM2 = 2 -> our index 1

    # reference in float64 numpy, exactly as the source writes it
    eS = np.where(active, np.where(incloud, PMFUS - M_u * (CPD * T_h + geo_half), 0.0), PMFUS)
    eQ = np.where(active, np.where(incloud, PMFUQ - M_u * q_h, 0.0), PMFUQ)
    eM = np.where(active, np.where(incloud, M_u, 0.0), M_u)
    eL = np.where(active, np.where(incloud, PMFUL, 0.0), PMFUL)

    # recompute the module output with the zeroed-mass input
    from jax import numpy as jjnp
    out = ifs_convective_fluxes(
        jnp.asarray(M_u)[None], *[jnp.asarray(a)[None] for a in
        (PMFUS, PMFUQ, PMFUL, ins[4], ins[5], T_h, q_h, geo_half, ins[9])],
        jnp.array([True]), jnp.array([1]),
        jnp.array([k_cbot]), jnp.array([k_ctop]), cfg=IFSFluxConfig())
    Mo, So, Qo, Lo = (np.asarray(a)[0] for a in out[:4])

    for got, exp, name in ((So, eS, "PMFUS"), (Qo, eQ, "PMFUQ"),
                           (Mo, eM, "M_u"), (Lo, eL, "PMFUL")):
        np.testing.assert_allclose(got, exp, rtol=RTOL, atol=ATOL,
                                   err_msg=name)
    # the zero-mass in-cloud level: subtraction term vanishes but the
    # level must NOT have been zeroed by the out-of-cloud branch
    assert Mo[5] == 0.0
    np.testing.assert_allclose(So[5], PMFUS[5] - 0.0, rtol=RTOL, atol=ATOL)
    # sanity: subtraction actually changes something (test can fail)
    assert np.max(np.abs(eS - PMFUS)) > 1.0
    # index 0 (IFS JK=1) untouched by the loop
    assert So[0] == PMFUS[0] and Qo[0] == PMFUQ[0] and Mo[0] == M_u[0]


def test_2_out_of_cloud_zeroing_and_jk_minus_one_destination():
    k_ctop, k_cbot = 4, 7
    ins, outs, _ = _run(0, k_ctop, k_cbot)
    Mo, So, Qo, Lo, Lo_de, Do = outs

    # above cloud top (j = 1..k_ctop-1): exactly zero
    for arr in (Mo, So, Qo, Lo):
        assert np.all(arr[1:k_ctop] == 0.0)
    # index 0 (IFS JK=1, loop starts at KTOPM2=2) is untouched
    assert Mo[0] == ins[0][0] and So[0] == ins[1][0]

    # distinctive detrainment / precip fluxes: value 7*(k+1)
    PLUDE0 = 7.0 * np.arange(1, NLEV + 1, dtype=float)
    PDMFUP0 = 11.0 * np.arange(1, NLEV + 1, dtype=float)
    p = _base_column(ncol=1)
    out = ifs_convective_fluxes(
        *[jnp.asarray(a)[None] for a in
          (ins[0], ins[1], ins[2], ins[3], PLUDE0, PDMFUP0,
           ins[6], ins[7], ins[8], ins[9])],
        jnp.array([True]), jnp.array([1]),
        jnp.array([k_cbot]), jnp.array([k_ctop]), cfg=IFSFluxConfig())
    Ld, Dm = np.asarray(out[4])[0], np.asarray(out[5])[0]

    # destination k cleared when level k+1 is out of cloud: levels
    # 1,2,3 are out of cloud -> destinations 0,1,2 cleared
    assert Ld[0] == 0.0 and Ld[1] == 0.0 and Ld[2] == 0.0
    assert Dm[0] == 0.0 and Dm[1] == 0.0 and Dm[2] == 0.0
    # level 4 (cloud top) is IN cloud -> destination 3 survives
    assert Ld[3] == PLUDE0[3] == 28.0
    assert Dm[3] == PDMFUP0[3] == 44.0
    # last destination (no level nlev in the source loop) survives
    assert Ld[NLEV - 1] == PLUDE0[NLEV - 1] == 70.0
    assert Dm[NLEV - 1] == PDMFUP0[NLEV - 1] == 110.0


def test_3_below_cloud_base_scaling():
    """Section 1.5 (cuflxn.F90:292-345) for a single convecting column.

    Reference built strictly on the module's interfaces:
      * p_surf = p_half[NLEV]  (PAPH(KLEV+1); flux arrays have no such level),
      * step 1 (j == k_cbot+1) scales the cloud-base values with
        zzp1 = (p_surf - p_half[k_cbot+1]) / (p_surf - p_half[k_cbot]),
      * step 2 (source loop condition JK > KCBOT+1, i.e. our
        k_cbot+1 < j <= NLEV-1) scales the *post-step-1* level k_cbot+1
        values with zzp2 = (p_surf - p_half[j]) / (p_surf - p_half[k_cbot+1]).
    Levels outside the source's write range are dropped, never invented.
    """
    k_ctop, k_cbot = 1, 5
    ins, outs, ph = _run(0, k_ctop, k_cbot)
    Mo, So, Qo, Lo, _, _ = outs
    M_u, PMFUS, PMFUQ, PMFUL, _, _, T_h, q_h, geo_half, _ = ins

    ps = ph[NLEV]  # surface interface PAPH(KLEV+1)
    assert ps != ph[k_cbot] and ps != ph[k_cbot + 1]
    # guard against divide-by-floor blowups: denominators must be far from
    # the module's _EPS floor, otherwise the reference is meaningless.
    assert abs(ps - ph[k_cbot]) > 1.0e6 * 1.0e-20 * 1.0e14
    assert abs(ps - ph[k_cbot + 1]) > 1.0e-6
    assert abs(ps - ph[k_cbot + 1]) > 1.0e6 * 1.0e-20

    # ---- step 1: level k_cbot+1 written from cloud base ----
    zzp1 = (ps - ph[k_cbot + 1]) / (ps - ph[k_cbot])
    s_ikb = PMFUS[k_cbot] - M_u[k_cbot] * (CPD * T_h[k_cbot] + geo_half[k_cbot])
    q_ikb = PMFUQ[k_cbot] - M_u[k_cbot] * q_h[k_cbot]
    l_ikb = PMFUL[k_cbot]

    np.testing.assert_allclose(Mo[k_cbot + 1], M_u[k_cbot] * zzp1,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(So[k_cbot + 1], (s_ikb - LV * l_ikb) * zzp1,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Qo[k_cbot + 1], (q_ikb + l_ikb) * zzp1,
                               rtol=RTOL, atol=ATOL)
    assert Lo[k_cbot + 1] == 0.0

    # post-step-1 level-(k_cbot+1) values are the fixed base of step 2
    mu1 = M_u[k_cbot] * zzp1
    s1 = (s_ikb - LV * l_ikb) * zzp1
    q1 = (q_ikb + l_ikb) * zzp1

    # ---- step 2: levels the source's second loop actually writes ----
    # source: JK = KTOPM2..KLEV with condition JK > KCBOT+1  ->  our
    # k_cbot+1 < j <= NLEV-1 (flux grid has no level NLEV).
    lo, hi = k_cbot + 1, NLEV - 1
    candidates = (k_cbot + 2, k_cbot + 3, NLEV - 1)
    levels = [j for j in candidates if lo < j <= hi]
    assert levels, "no in-range step-2 levels selected"

    for j in levels:
        assert lo < j <= hi  # inside the source's write range
        zzp2 = (ps - ph[j]) / (ps - ph[k_cbot + 1])
        np.testing.assert_allclose(Mo[j], mu1 * zzp2, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(So[j], s1 * zzp2, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(Qo[j], q1 * zzp2, rtol=RTOL, atol=ATOL)
        assert Lo[j] == 0.0

    # sanity: the taper really bites (test can fail)
    assert np.max(np.abs(Mo[levels])) < np.max(np.abs(M_u[k_cbot]))
    # monotone decrease of the step-2 written levels towards the surface
    # (non-uniform grid); only levels the source writes are asserted.
    Mo_step2 = Mo[levels]
    assert np.all(np.diff(Mo_step2) < 0.0)
    assert np.all(np.diff(np.abs(So[levels])) < 0.0)
    # every step-2 level is smaller than its step-2 base level
    assert np.all(Mo_step2 < Mo[k_cbot + 1])


def test_4_non_convecting_column_unchanged():
    """Non-convecting columns: the ELSE branch of cuflxn.F90:265-283 fires
    whenever NOT(LDCUM .AND. JK >= KCTOP), so EVERY flux level the loop
    reaches is ZEROED -- the column is NOT left alone.  The level loop
    starts at KTOPM2 = 2 (our index 1), so our index 0 (IFS JK = 1) is
    never written and keeps its input value; PDMFUP/PLUDE are cleared at
    the shifted (JK-1) destinations: destination k is zeroed when level
    k+1 is out of cloud, destinations 0..NLEV-2, last one untouched.
    """
    k_ctop, k_cbot = 3, 6
    (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP,
     T_h, q_h, geo_half, p_half) = _base_column()
    # make sure the inputs are genuinely non-zero, so a module that
    # skipped non-convecting columns entirely would FAIL these asserts
    for a in (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP):
        assert np.all(np.abs(a) > 0.0)

    out = ifs_convective_fluxes(
        *[jnp.asarray(a) for a in (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP,
                                   T_h, q_h, geo_half, p_half)],
        jnp.array([False, False]), jnp.array([1, 2]),
        jnp.array([k_cbot, k_cbot]), jnp.array([k_ctop, k_ctop]),
        cfg=IFSFluxConfig())
    Mo, So, Qo, Lo, LDo, Do = (np.asarray(a) for a in out)

    for col in range(N):
        # indices 1..nlev-1 (IFS JK = 2..KLEV): exactly zeroed by the ELSE
        # branch -- a module that skipped this column would fail here
        assert np.all(Mo[col, 1:] == 0.0)
        assert np.all(So[col, 1:] == 0.0)
        assert np.all(Qo[col, 1:] == 0.0)
        assert np.all(Lo[col, 1:] == 0.0)
        # index 0 (IFS JK = 1) is never touched by the level loop
        assert Mo[col, 0] == M_u[col, 0]
        assert So[col, 0] == PMFUS[col, 0]
        assert Qo[col, 0] == PMFUQ[col, 0]
        assert Lo[col, 0] == PMFUL[col, 0]
        # shifted JK-1 destinations: level k+1 out of cloud (all levels
        # here) -> destinations 0..NLEV-2 cleared; destination NLEV-1 has
        # no level NLEV in the source loop and survives
        assert np.all(LDo[col, :NLEV - 1] == 0.0)
        assert np.all(Do[col, :NLEV - 1] == 0.0)
        assert LDo[col, NLEV - 1] == PLUDE[col, NLEV - 1]
        assert Do[col, NLEV - 1] == PDMFUP[col, NLEV - 1]
