"""CLUBB CAM-branch leaves pinned against independent CLUBB-Fortran mirrors.

These five leaves run CAM-default branches that CLUBB-JAX does NOT implement
(it carries only the ARM/True paths), so until now they were "validated
against the Fortran formula by hand" with no mechanical pin.  Each test here
transcribes the CLUBB Fortran (larson-group/clubb_release, CLUBB_core)
reference INDEPENDENTLY in numpy and pins the JAX leaf at rel 1e-12:

* ``compute_skw_fnc``       — advance_xm_wpxp_module.F90:650-659
  (``Cxb + (Cx-Cxb)*exp(-1/2*(Skw/Cxc)**2)`` with the ``|C-Cb| >
  |C+Cb|*eps/2`` collapse branch)
* ``damp_coefficient``      — advance_xm_wpxp_module.F90:5990-6048
  (Lscale ramp toward ``max_coeff_value`` where ``Lscale_zm < threshold``
  AND ``zm > altitude_threshold``)
* ``compute_C6_C7_Skw_fnc`` — advance_xm_wpxp_module.F90:640-700 composition
  (C6rt/C6thl skewness fn then Lscale damping; C7 skewness fn undamped)
* ``wp2_term_dp1_rhs``      — advance_wp2_wp3_module.F90 (the
  ``l_damp_wp2_using_em = .false.`` CAM branch:
  ``+ (C1_Skw_fnc*invrs_tau)*threshold`` interior, boundary zeros)
* ``wp3_term_pr_turb_rhs``  — advance_wp2_wp3_module.F90:5350-5410 (the
  ``l_use_tke_in_wp3_pr_turb_term = .false.`` CAM branch:
  ``-C*Kh*invrs_dzt*(g/thv_ds*d(wpthvp) - d(upwp*dum_dz) - d(vpwp*dvm_dz))``)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    make_clubb_grid,
)

from legoesm import constants  # noqa: E402

_EPS = 1.0e-10  # constants_clubb eps (branch threshold floor)


def _gr(ng=2, nzt=9):
    nzm = nzt + 1
    zm_1d = np.cumsum(np.concatenate([[0.0], 45.0 * 1.12 ** np.arange(nzm - 1)]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm, nzt


def _skw_fnc_mirror(Cx, Cxb, Cxc, Skw):
    """advance_xm_wpxp_module.F90:650-659, per column."""
    out = np.empty_like(Skw)
    for i in range(Skw.shape[0]):
        if abs(Cx[i] - Cxb[i]) > abs(Cx[i] + Cxb[i]) * _EPS / 2.0:
            out[i] = Cxb[i] + (Cx[i] - Cxb[i]) * np.exp(
                -0.5 * (Skw[i] / Cxc[i]) ** 2)
        else:
            out[i] = Cxb[i]
    return out


def _damp_mirror(coef, fnc, mx, alt_thr, thr, Lscale_zm, zm):
    """advance_xm_wpxp_module.F90:6034-6040, per column."""
    out = fnc.copy()
    for i in range(fnc.shape[0]):
        for k in range(fnc.shape[1]):
            if Lscale_zm[i, k] < thr[i] and zm[i, k] > alt_thr[i]:
                out[i, k] = mx[i] + ((coef[i] - mx[i]) / thr[i]) * Lscale_zm[i, k]
    return out


def test_compute_skw_fnc_fortran_mirror():
    rng = np.random.default_rng(11)
    ng, nz = 3, 12
    Cx = np.array([4.0, 0.5, 2.0])
    Cxb = np.array([6.0, 0.5, 2.0])       # cols 1, 2: |C-Cb| = 0 -> collapse
    Cxc = np.array([1.0, 0.7, 1.3])
    Skw = rng.normal(0.0, 2.0, size=(ng, nz))
    got = np.asarray(C.compute_skw_fnc(
        jnp.asarray(Cx), jnp.asarray(Cxb), jnp.asarray(Cxc), jnp.asarray(Skw)))
    want = _skw_fnc_mirror(Cx, Cxb, Cxc, Skw)
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)
    # The collapse branch is exact Cb, not the smooth expression.
    assert np.all(got[1] == Cxb[1])


def test_damp_coefficient_fortran_mirror():
    rng = np.random.default_rng(12)
    gr, ng, nzm, nzt = _gr()
    coef = np.array([4.0, 2.5])
    mx = np.array([12.0, 8.0])
    alt_thr = np.array([100.0, 250.0])
    thr = np.array([300.0, 150.0])
    Lscale = rng.uniform(5.0, 600.0, size=(ng, nzm))
    fnc = rng.uniform(1.0, 6.0, size=(ng, nzm))
    got = np.asarray(C.damp_coefficient(
        jnp.asarray(coef), jnp.asarray(fnc), jnp.asarray(mx),
        jnp.asarray(alt_thr), jnp.asarray(thr), jnp.asarray(Lscale), gr))
    want = _damp_mirror(coef, fnc, mx, alt_thr, thr, Lscale, np.asarray(gr.zm))
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)
    # Non-vacuous: both branches must actually occur in the fixture.
    damped_mask = (Lscale < thr[:, None]) & (np.asarray(gr.zm) > alt_thr[:, None])
    assert damped_mask.any() and (~damped_mask).any()


def test_compute_C6_C7_skw_fnc_fortran_composition():
    rng = np.random.default_rng(13)
    gr, ng, nzm, nzt = _gr()
    cfg = CLUBBConfig()
    p = cfg.params
    Skw = rng.normal(0.0, 1.5, size=(ng, nzm))
    Lscale = rng.uniform(5.0, 500.0, size=(ng, nzm))
    C6rt, C6thl, C7 = C.compute_C6_C7_Skw_fnc(
        jnp.asarray(Skw), jnp.asarray(Lscale), cfg, gr)

    def cols(v):
        return np.full((ng,), v)

    want_rt = _damp_mirror(
        cols(p.C6rt),
        _skw_fnc_mirror(cols(p.C6rt), cols(p.C6rtb), cols(p.C6rtc), Skw),
        cols(p.C6rt_Lscale0), cols(p.altitude_threshold),
        cols(p.wpxp_L_thresh), Lscale, np.asarray(gr.zm))
    want_thl = _damp_mirror(
        cols(p.C6thl),
        _skw_fnc_mirror(cols(p.C6thl), cols(p.C6thlb), cols(p.C6thlc), Skw),
        cols(p.C6thl_Lscale0), cols(p.altitude_threshold),
        cols(p.wpxp_L_thresh), Lscale, np.asarray(gr.zm))
    want_c7 = _skw_fnc_mirror(cols(p.C7), cols(p.C7b), cols(p.C7c), Skw)
    assert np.allclose(np.asarray(C6rt), want_rt, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(C6thl), want_thl, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(C7), want_c7, rtol=1e-12, atol=0.0)


def test_wp2_term_dp1_rhs_fortran_mirror():
    rng = np.random.default_rng(14)
    ng, nzm = 2, 10
    C1 = rng.uniform(0.5, 3.0, size=(ng, nzm))
    invtau = rng.uniform(1e-4, 1e-2, size=(ng, nzm))
    threshold = 4.0e-4  # w_tol_sqd
    got = np.asarray(C.wp2_term_dp1_rhs(
        jnp.asarray(C1), jnp.asarray(invtau), threshold))
    want = np.zeros((ng, nzm))
    want[:, 1:-1] = C1[:, 1:-1] * invtau[:, 1:-1] * threshold
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)
    assert (got[:, 0] == 0.0).all() and (got[:, -1] == 0.0).all()


def test_wp3_term_pr_turb_rhs_fortran_mirror():
    rng = np.random.default_rng(15)
    gr, ng, nzm, nzt = _gr()
    Cw = np.array([0.5, 1.2])
    Kh = rng.uniform(1.0, 40.0, size=(ng, nzt))
    thv = rng.uniform(295.0, 310.0, size=(ng, nzt))
    wpthvp = rng.normal(0.0, 0.05, size=(ng, nzm))
    dum = rng.normal(0.0, 5e-3, size=(ng, nzm))
    dvm = rng.normal(0.0, 5e-3, size=(ng, nzm))
    upwp = rng.normal(0.0, 0.1, size=(ng, nzm))
    vpwp = rng.normal(0.0, 0.1, size=(ng, nzm))
    got = np.asarray(C.wp3_term_pr_turb_rhs(
        jnp.asarray(Cw), jnp.asarray(Kh), jnp.asarray(wpthvp),
        jnp.asarray(dum), jnp.asarray(dvm), jnp.asarray(upwp),
        jnp.asarray(vpwp), jnp.asarray(thv), gr))
    idzt = np.asarray(gr.invrs_dzt)
    want = np.zeros((ng, nzt))
    # advance_wp2_wp3_module.F90:5399-5407 (1-based k=2..nzt-1; the zm pair
    # bracketing zt level k is (k, k+1) = python (k, k+1) on the ascending
    # grid with zm[k] below / zm[k+1] above zt[k]).
    for i in range(ng):
        for k in range(1, nzt - 1):
            want[i, k] = -Cw[i] * Kh[i, k] * idzt[i, k] * (
                constants.g / thv[i, k] * (wpthvp[i, k + 1] - wpthvp[i, k])
                - (upwp[i, k + 1] * dum[i, k + 1] - upwp[i, k] * dum[i, k])
                - (vpwp[i, k + 1] * dvm[i, k + 1] - vpwp[i, k] * dvm[i, k]))
    assert np.allclose(got, want, rtol=1e-12, atol=1e-18)
    assert (got[:, 0] == 0.0).all() and (got[:, -1] == 0.0).all()
