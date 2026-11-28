"""FV3 hydrostatic PG off-centering ``beta`` -- ``grad1_p_update``
(dyn_core.F90:2483-2574, dispatched at :1528 when ``beta > 0``, with
``beta_d = 0`` on the first acoustic sub-step, :440-444).

Kernel gates (the stepper/model gates live in
``tests/matrix/test_cube_pgrad_beta.py`` and the FV3DuoConfig tests):

1. ``beta = 0`` is ``one_grad_p`` BITWISE, whatever the carried
   increment -- the it == 1 weight-zero step and the beta = 0 identity;
2. ``beta = 0.4`` matches an explicit loop transcription of the Fortran
   (u += beta*du; du = new; u = (u + divg2 - divg2 + alpha*du)*rdx),
   including the carried increment and the untouched halo of du/dv;
3. guards.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.core.fv3_dsw_tail_3d import require_pgrad_beta  # noqa: E402
from legoesm.core.fv3_pgrad import grad1_p_update, one_grad_p  # noqa: E402
from tests.grids.test_fv3_pgrad import (  # noqa: E402
    BD,
    _gs_cached,
    _ogp_fixture,
    _ogp_flags,
    _ogp_kw,
)


def _g1_kw(km):
    kw = _ogp_kw(km)
    kw.pop("hydrostatic")
    kw.pop("d_ext")
    return {**kw, **_ogp_flags()}


def _du_dv(u, v, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(u.shape) * 3.0,
            rng.standard_normal(v.shape) * 3.0)


@pytest.mark.parametrize("km", [1, 2, 3])
def test_grad1_p_update_at_zero_weight_is_one_grad_p_bitwise(km):
    """beta_d = 0 with divg2 = 0 (the production d_ext = 0 it == 1 call):
    the carried increment is multiplied by exactly zero and alpha = 1, so
    the winds/pk/gz are one_grad_p's to the bit; the returned du is the
    new PG term (non-vacuity: it is not the carried one).  Nonzero divg2
    at zero weight is gated against the Fortran loops below instead (the
    Fortran grouping may round differently from one_grad_p's)."""
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    divg2 = np.zeros_like(divg2)
    gs = _gs_cached()
    du0, dv0 = _du_dv(u, v)
    want = one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                      **_ogp_kw(km), **_ogp_flags())
    got = grad1_p_update(divg2, u, v, pk, gz, du0, dv0, gs, BD, beta=0.0,
                         **_g1_kw(km))
    for name, g, w in zip(("u", "v", "pk", "gz"), got[:4], want):
        assert np.array_equal(np.asarray(g), np.asarray(w)), name
    assert not np.array_equal(np.asarray(got[4]), du0)


def _loop_reference(u, v, pk_b, gz_b, du, dv, divg2, rdx, rdy, *, km, dt,
                    beta, bd):
    """dyn_core.F90:2543-2572 transcribed loop by loop on the B-grid
    pk/gz (0-based arrays offset by isd/jsd; divg2 offset by is/js)."""
    u, v, du, dv = (np.array(a, dtype=np.float64) for a in (u, v, du, dv))
    o, og = bd.isd, bd.is_            # array origin of u/pk, of divg2
    alpha = 1.0 - beta
    for k in range(km):
        def wk(i, j):
            return pk_b[i - o, j - o, k + 1] - pk_b[i - o, j - o, k]
        for j in range(bd.js, bd.je + 2):
            for i in range(bd.is_, bd.ie + 1):
                I, J = i - o, j - o
                u[I, J, k] = u[I, J, k] + beta * du[I, J, k]
                du[I, J, k] = dt / (wk(i, j) + wk(i + 1, j)) * (
                    (gz_b[I, J, k + 1] - gz_b[I + 1, J, k])
                    * (pk_b[I + 1, J, k + 1] - pk_b[I, J, k])
                    + (gz_b[I, J, k] - gz_b[I + 1, J, k + 1])
                    * (pk_b[I, J, k + 1] - pk_b[I + 1, J, k]))
                u[I, J, k] = (u[I, J, k] + divg2[i - og, j - og]
                              - divg2[i + 1 - og, j - og]
                              + alpha * du[I, J, k]) * rdx[I, J]
        for j in range(bd.js, bd.je + 1):
            for i in range(bd.is_, bd.ie + 2):
                I, J = i - o, j - o
                v[I, J, k] = v[I, J, k] + beta * dv[I, J, k]
                dv[I, J, k] = dt / (wk(i, j) + wk(i, j + 1)) * (
                    (gz_b[I, J, k + 1] - gz_b[I, J + 1, k])
                    * (pk_b[I, J + 1, k + 1] - pk_b[I, J, k])
                    + (gz_b[I, J, k] - gz_b[I, J + 1, k + 1])
                    * (pk_b[I, J, k + 1] - pk_b[I, J + 1, k]))
                v[I, J, k] = (v[I, J, k] + divg2[i - og, j - og]
                              - divg2[i - og, j + 1 - og]
                              + alpha * dv[I, J, k]) * rdy[I, J]
    return u, v, du, dv


@pytest.mark.parametrize("km", [1, 3])
@pytest.mark.parametrize("zero_divg2", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.4])
def test_grad1_p_update_matches_the_fortran_loops(km, zero_divg2, beta):
    delp, pk, gz, u, v, divg2 = _ogp_fixture(km)
    if zero_divg2:
        divg2 = np.zeros_like(divg2)
    gs = _gs_cached()
    du0, dv0 = _du_dv(u, v)
    kw = _g1_kw(km)
    u_j, v_j, pk_j, gz_j, du_j, dv_j = grad1_p_update(
        divg2, u, v, pk, gz, du0, dv0, gs, BD, beta=beta, **kw)
    # the B-grid planes are the shared set-up's (certified via one_grad_p)
    _u, _v, pk_o, gz_o = one_grad_p(u, v, pk, gz, divg2, None, gs, BD,
                                    **_ogp_kw(km), **_ogp_flags())
    assert np.array_equal(np.asarray(pk_j), np.asarray(pk_o))
    assert np.array_equal(np.asarray(gz_j), np.asarray(gz_o))
    u_r, v_r, du_r, dv_r = _loop_reference(
        u, v, np.asarray(pk_j), np.asarray(gz_j), du0, dv0, divg2,
        gs["rdx"], gs["rdy"], km=km, dt=kw["dt"], beta=beta, bd=BD)
    for name, g, r in (("u", u_j, u_r), ("v", v_j, v_r),
                       ("du", du_j, du_r), ("dv", dv_j, dv_r)):
        g = np.asarray(g)
        scale = np.abs(r).max()
        assert np.abs(g - r).max() <= 1e-14 * scale, (
            name, np.abs(g - r).max() / scale)
    # outside the update windows du/dv are the carried values (never written)
    mask_u = np.ones(u.shape, bool)
    mask_u[BD.is_ - BD.isd:BD.ie - BD.isd + 1,
           BD.js - BD.jsd:BD.je - BD.jsd + 2, :] = False
    assert np.array_equal(np.asarray(du_j)[mask_u], du0[mask_u])
    # non-vacuity: the carried increment really entered the winds
    u_nc = grad1_p_update(divg2, u, v, pk, gz, 0.0 * du0, 0.0 * dv0, gs, BD,
                          beta=beta, **kw)[0]
    moved = float(np.abs(np.asarray(u_nc) - np.asarray(u_j)).max())
    assert (moved > 1e-6) if beta > 0 else (moved == 0.0)


def test_grad1_p_update_guards():
    delp, pk, gz, u, v, divg2 = _ogp_fixture(1)
    gs = _gs_cached()
    du0, dv0 = _du_dv(u, v)
    kw = _g1_kw(1)
    for bad in (-0.1, 1.5, True, "0.4"):
        with pytest.raises(ValueError, match="beta"):
            grad1_p_update(divg2, u, v, pk, gz, du0, dv0, gs, BD, beta=bad,
                           **kw)
    with pytest.raises(ValueError, match="divg2"):
        grad1_p_update(divg2[:, :, None], u, v, pk, gz, du0, dv0, gs, BD,
                       beta=0.4, **kw)
    with pytest.raises(ValueError, match="divg2"):
        grad1_p_update(divg2[:-1, :-1], u, v, pk, gz, du0, dv0, gs, BD,
                       beta=0.4, **kw)
    with pytest.raises(ValueError, match="du/dv"):
        grad1_p_update(divg2, u, v, pk, gz, du0[:, :-1], dv0, gs, BD,
                       beta=0.4, **kw)


def test_require_pgrad_beta_contract():
    assert require_pgrad_beta("f", 0.0, 0.0, None) == (0.0, 0.0)
    assert require_pgrad_beta("f", 0.4, 0.0, {}) == (0.4, 0.0)
    assert require_pgrad_beta("f", 0.4, 0.4, {}) == (0.4, 0.4)
    with pytest.raises(ValueError, match="pg_prev"):
        require_pgrad_beta("f", 0.4, 0.4, None)
    with pytest.raises(ValueError, match="pg_prev"):
        require_pgrad_beta("f", 0.0, 0.0, {})
    with pytest.raises(ValueError, match="beta_d"):
        require_pgrad_beta("f", 0.4, 0.2, {})
    with pytest.raises(ValueError, match="beta"):
        require_pgrad_beta("f", True, 0.0, None)


# ---------------------------------------------------------------------
# km>1 acoustic loop: the du/dv carry through peel / lax.scan / peel
# ---------------------------------------------------------------------

def _chain_by_hand(jac, jctx, st, n_split, kw):
    """Reference: the sub-steps called one by one with the du/dv handoff
    written out (beta_d = 0 on it == 1, :440-444; remap on it == n_split)."""
    dt = kw["dt_atmos"] / float(n_split)
    pg = None
    for it in range(1, n_split + 1):
        out = jac.acoustic_substep_3d(
            jctx, st, dt, kw["km"], first_substep=(it == 1),
            ptop=kw["ptop"], akap=kw["akap"], cp_air=kw["cp_air"],
            remap_step=(it == n_split), hydrostatic_logp=kw["logp"],
            beta=kw["beta"], pg_prev=pg, batched=kw["batched"])
        st, pg = out["state"], out["pg"]
    return st


@pytest.mark.parametrize("batched, logp",
                         [(False, False), (True, False), (True, True)])
def test_acoustic_loop_scan_carries_the_pg_increment(batched, logp):
    """n_split = 4 runs two middle sub-steps under lax.scan: the loop must
    equal the hand-chained sub-steps (a dropped/stale du in the scan carry
    moves u at O(1e-4) relative), and differ from beta = 0."""
    from tests.grids import test_fv3_acoustic_3d as T
    from legoesm.core import fv3_acoustic_3d as jac
    jctx = T.build_jax_duo_stepper_context(T.build_six_face_duo_context(
        T.N, T.NG, use_ext_bundle=True, oracle_conventions=True))
    st0 = T.state_3d_to_jax(T._seeded(T.KM))
    kw = dict(dt_atmos=T.DT_ATMOS, km=T.KM, ptop=T.PTOP, akap=T.AKAP,
              cp_air=T.CP_AIR, logp=logp, batched=batched, beta=0.4)

    def loop(beta):
        return jac.acoustic_loop_3d(
            jctx, st0, T.DT_ATMOS, T.KM, n_split=4, ptop=T.PTOP,
            akap=T.AKAP, cp_air=T.CP_AIR, batched=batched,
            hydrostatic_logp=logp, beta=beta)["state"]
    got, b0 = loop(0.4), loop(0.0)
    ref = _chain_by_hand(jac, jctx, st0, 4, kw)
    for nm in ("delp", "pt", "u", "v"):
        g, r = np.asarray(got[nm]), np.asarray(ref[nm])
        rel = np.abs(g - r).max() / np.abs(r).max()
        assert rel < 1e-12, (nm, rel)
    du = np.abs(np.asarray(got["u"]) - np.asarray(b0["u"])).max()
    assert du > 1e-6 * np.abs(np.asarray(b0["u"])).max(), du
