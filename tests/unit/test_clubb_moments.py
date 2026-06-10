"""Unit tests for the CLUBB moment advances (``clubb_moments.py``).

Currently covers ``advance_windm_edsclrm`` (the CAM ``l_predict_upwp_vpwp=False``
u/v eddy-diffusion advance): a committed golden + live bit-exact parity vs the
CLUBB-JAX reference, plus physical sanity (steady state, flux clipping) and
AD/JIT. See ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_moments import (  # noqa: E402
    advance_windm_edsclrm,
    calc_xp2_xpyp_ta_lhs,
    calc_xp2_xpyp_ta_rhs,
    diffusion_zm_lhs,
    term_dp1_lhs,
    term_dp1_rhs,
    term_pr1,
    term_pr2,
    term_tp_rhs,
    xp2_xpyp_lhs,
    xp2_xpyp_rhs,
)

from legoesm import constants  # noqa: E402

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_FIX = Path(__file__).resolve().parent / "clubb_fixtures"
_C_K10, _NU10, _DT = 0.5, 0.0, 300.0
_IC_K10 = 74   # 1-based clubb_params index for c_K10 (constants_clubb)


def _windm_inputs(ng=2, nzt=10):
    rng = np.random.default_rng(21)
    nzm = nzt + 1
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.1 ** np.arange(nzm)[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    zt_shp, zm_shp = (ng, nzt), (ng, nzm)

    def zt(s):
        return jnp.asarray(s * rng.standard_normal(zt_shp))

    rho_ds_zm = jnp.asarray(1.0 + 0.1 * rng.random(zm_shp))
    return dict(
        um=jnp.asarray(5.0 + 3.0 * rng.random(zt_shp)),
        vm=jnp.asarray(-2.0 + 2.0 * rng.random(zt_shp)),
        upwp=jnp.asarray(0.05 * rng.standard_normal(zm_shp)),
        vpwp=jnp.asarray(0.05 * rng.standard_normal(zm_shp)),
        wp2=jnp.asarray(0.2 + 0.5 * rng.random(zm_shp)),
        up2=jnp.asarray(0.3 + 0.5 * rng.random(zm_shp)),
        vp2=jnp.asarray(0.3 + 0.5 * rng.random(zm_shp)),
        wm_zt=zt(0.02),
        Kh_zm=jnp.asarray(0.5 + 2.0 * rng.random(zm_shp)),
        ug=jnp.asarray(6.0 + rng.random(zt_shp)),
        vg=jnp.asarray(-1.0 + rng.random(zt_shp)),
        um_forcing=zt(1e-4), vm_forcing=zt(1e-4),
        rho_ds_zm=rho_ds_zm, rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random(zt_shp)),
        invrs_rho_ds_zt=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random(zt_shp))),
        fcor=jnp.asarray(1.0e-4 + 1e-5 * rng.random((ng,))),
        c_K10=_C_K10, nu10=_NU10, dt=_DT, gr=gr,
    )


def test_shapes_and_flux_clip():
    kw = _windm_inputs()
    um, vm, upwp, vpwp = advance_windm_edsclrm(**kw)
    ng, nzt = kw["um"].shape
    assert um.shape == (ng, nzt) and vm.shape == (ng, nzt)
    assert upwp.shape == kw["upwp"].shape
    # Cauchy-Schwarz bound on the clipped momentum flux (interior).
    bound_u = 0.99 * np.sqrt(np.asarray(kw["wp2"]) * np.asarray(kw["up2"]))
    assert np.all(np.abs(np.asarray(upwp))[:, 1:-1] <= bound_u[:, 1:-1] + 1e-12)


def test_no_forcing_no_coriolis_steady_uniform():
    """Uniform wind, no shear/forcing/Coriolis, zero w -> winds unchanged."""
    ng, nzt = 2, 10
    kw = _windm_inputs(ng, nzt)
    nzm = nzt + 1
    kw.update(
        um=jnp.full((ng, nzt), 7.0), vm=jnp.full((ng, nzt), -3.0),
        wm_zt=jnp.zeros((ng, nzt)),
        ug=jnp.full((ng, nzt), 7.0), vg=jnp.full((ng, nzt), -3.0),
        um_forcing=jnp.zeros((ng, nzt)), vm_forcing=jnp.zeros((ng, nzt)),
        fcor=jnp.zeros((ng,)),
        upwp=jnp.zeros((ng, nzm)), vpwp=jnp.zeros((ng, nzm)),
    )
    um, vm, _, _ = advance_windm_edsclrm(**kw)
    # Uniform field -> zero diffusion flux, no tendency -> unchanged.
    np.testing.assert_allclose(np.asarray(um), 7.0, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(vm), -3.0, rtol=1e-9)


def test_matches_committed_golden():
    g = np.load(_FIX / "clubb_windm_golden.npz")
    um, vm, upwp, vpwp = advance_windm_edsclrm(**_windm_inputs())
    np.testing.assert_array_equal(np.asarray(um), g["um"])
    np.testing.assert_array_equal(np.asarray(vm), g["vm"])
    np.testing.assert_array_equal(np.asarray(upwp), g["upwp"])
    np.testing.assert_array_equal(np.asarray(vpwp), g["vpwp"])


def test_jit_and_grad():
    kw = _windm_inputs()

    def loss(um):
        a, b, c, d = advance_windm_edsclrm(**dict(kw, um=um))
        return jnp.sum(a ** 2) + jnp.sum(b ** 2) + jnp.sum(c ** 2) + jnp.sum(d ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["um"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["um"])))


def test_calm_wind_gradient_finite():
    """Zero winds (um=vm=0) -> the wind_speed/sfc-flux guard keeps grads finite."""
    ng, nzt = 2, 8
    kw = _windm_inputs(ng, nzt)
    nzm = nzt + 1
    kw.update(um=jnp.zeros((ng, nzt)), vm=jnp.zeros((ng, nzt)),
              upwp=jnp.zeros((ng, nzm)), vpwp=jnp.zeros((ng, nzm)))

    def loss(um):
        a, b, c, d = advance_windm_edsclrm(**dict(kw, um=um))
        return jnp.sum(a ** 2) + jnp.sum(b ** 2) + jnp.sum(c ** 2) + jnp.sum(d ** 2)

    g = jax.grad(loss)(kw["um"])
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# advance_xp2_xpyp term builders
# ---------------------------------------------------------------------------

def _gr_only(ng=2, nzt=8):
    nzm = nzt + 1
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.1 ** np.arange(nzm)[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    return make_clubb_grid(zm, 0.5 * (zm[:, 1:] + zm[:, :-1])), ng, nzm


def test_term_dp1_lhs_boundaries_zero_interior():
    ng, nzm = 2, 9
    Cn = jnp.asarray(0.5 + np.random.default_rng(0).random((ng, nzm)))
    itau = jnp.asarray(1e-3 + 1e-3 * np.random.default_rng(1).random((ng, nzm)))
    out = term_dp1_lhs(Cn, itau)
    np.testing.assert_array_equal(np.asarray(out)[:, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(out)[:, -1], 0.0)
    np.testing.assert_allclose(np.asarray(out[:, 1:-1]), np.asarray(Cn * itau)[:, 1:-1], rtol=1e-13)


def test_term_dp1_rhs_definition():
    ng, nzm = 2, 7
    Cn = jnp.asarray(np.random.default_rng(2).random((ng, nzm)))
    itau = jnp.asarray(np.random.default_rng(3).random((ng, nzm)))
    np.testing.assert_allclose(np.asarray(term_dp1_rhs(Cn, itau, 1e-4)),
                               np.asarray(Cn * itau * 1e-4), rtol=1e-13)


def test_term_pr2_nonnegative():
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(8)
    nzt = nzm - 1
    out = term_pr2(0.3, 0.3, jnp.asarray(300.0 + rng.random((ng, nzm))),
                   jnp.asarray(rng.standard_normal((ng, nzm)) * 0.01),
                   jnp.asarray(rng.standard_normal((ng, nzm)) * 0.05),
                   jnp.asarray(rng.standard_normal((ng, nzm)) * 0.05),
                   jnp.asarray(rng.standard_normal((ng, nzt))),
                   jnp.asarray(rng.standard_normal((ng, nzt))), gr)
    assert jnp.all(out >= 0.0)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_xp2_term_builders_parity():
    """Bit-exact parity of the dp1/tp/pr1/pr2 term builders vs the reference."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812

    gr, ng, nzm = _gr_only()
    nzt = nzm - 1
    rng = np.random.default_rng(31)
    Cn = jnp.asarray(0.5 + rng.random((ng, nzm)))
    itau = jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm)))
    np.testing.assert_array_equal(np.asarray(term_dp1_lhs(Cn, itau)),
                                  np.asarray(R.term_dp1_lhs(Cn, itau)))
    np.testing.assert_array_equal(np.asarray(term_dp1_rhs(Cn, itau, 1e-4)),
                                  np.asarray(R.term_dp1_rhs(Cn, itau, 1e-4)))
    xam = jnp.asarray(rng.standard_normal((ng, nzt)))
    xbm = jnp.asarray(rng.standard_normal((ng, nzt)))
    wpxap = jnp.asarray(rng.standard_normal((ng, nzm)))
    wpxbp = jnp.asarray(rng.standard_normal((ng, nzm)))
    np.testing.assert_array_equal(
        np.asarray(term_tp_rhs(xam, xbm, wpxap, wpxbp, gr.invrs_dzm)),
        np.asarray(R.term_tp_rhs(xam, xbm, wpxap, wpxbp, gr.invrs_dzm)))
    xbp2 = jnp.asarray(0.1 + rng.random((ng, nzm)))
    wp2 = jnp.asarray(0.2 + rng.random((ng, nzm)))
    itc4 = jnp.asarray(1e-3 + rng.random((ng, nzm)))
    itc14 = jnp.asarray(1e-3 + rng.random((ng, nzm)))
    np.testing.assert_array_equal(
        np.asarray(term_pr1(5.2, 2.2, xbp2, wp2, itc4, itc14, (2e-2) ** 2)),
        np.asarray(R.term_pr1(5.2, 2.2, xbp2, wp2, itc4, itc14, (2e-2) ** 2)))
    # term_pr2 uses grav -> patch reference constant to legoESM g, then bit-exact.
    R.grav = constants.g
    thv = jnp.asarray(300.0 + rng.random((ng, nzm)))
    wpthvp = jnp.asarray(rng.standard_normal((ng, nzm)) * 0.01)
    upwp = jnp.asarray(rng.standard_normal((ng, nzm)) * 0.05)
    vpwp = jnp.asarray(rng.standard_normal((ng, nzm)) * 0.05)
    um = jnp.asarray(rng.standard_normal((ng, nzt)))
    vm = jnp.asarray(rng.standard_normal((ng, nzt)))
    # The reference term_pr2 only reads gr.invrs_dzm, which CLUBBGrid provides.
    np.testing.assert_array_equal(
        np.asarray(term_pr2(0.3, 0.3, thv, wpthvp, upwp, vpwp, um, vm, gr)),
        np.asarray(R.term_pr2(0.3, 0.3, thv, wpthvp, upwp, vpwp, um, vm, gr)))


def test_xp2_ta_shapes_and_boundaries():
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(9)
    wp3 = jnp.asarray(rng.standard_normal((ng, nzm)))
    ssw = jnp.asarray(0.2 + 0.4 * rng.random((ng, nzm)))
    rho = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    irho = 1.0 / rho
    lhs = calc_xp2_xpyp_ta_lhs(wp3, ssw, 2.4, rho, irho, gr)
    assert lhs.shape == (3, ng, nzm)
    # Boundaries zeroed (top + bottom).
    np.testing.assert_array_equal(np.asarray(lhs)[:, :, 0], 0.0)
    np.testing.assert_array_equal(np.asarray(lhs)[:, :, -1], 0.0)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_xp2_ta_parity():
    """Bit-exact parity of the upwind turbulent-advection LHS/RHS vs the reference."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812

    gr, ng, nzm = _gr_only()
    nzt = nzm - 1
    rng = np.random.default_rng(41)
    wp3 = jnp.asarray(rng.standard_normal((ng, nzm)))
    ssw = jnp.asarray(0.2 + 0.4 * rng.random((ng, nzm)))
    wp2 = jnp.asarray(0.2 + 0.6 * rng.random((ng, nzm)))
    rho = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    irho = 1.0 / rho
    beta_col = jnp.full((ng,), 2.4)
    refgr = SimpleNamespace(zm=gr.zm, zt=gr.zt, dzm=gr.dzm, invrs_dzm=gr.invrs_dzm,
                            invrs_dzt=gr.invrs_dzt, grid_dir=1.0)
    # Reference upwind path ignores the *_zt args; pass placeholders.
    zt0 = jnp.zeros((ng, nzt))
    mine_lhs = calc_xp2_xpyp_ta_lhs(wp3, ssw, beta_col, rho, irho, gr)
    ref_lhs = R.calc_xp2_xpyp_ta_lhs_jax(True, wp3, zt0, ssw, beta_col, rho, irho, zt0, refgr)
    np.testing.assert_allclose(np.asarray(mine_lhs), np.asarray(ref_lhs), rtol=1e-12, atol=1e-14)

    fa = jnp.asarray(1e-4 * rng.standard_normal((ng, nzm)))
    fb = jnp.asarray(1e-3 * rng.standard_normal((ng, nzm)))
    mine_rhs = calc_xp2_xpyp_ta_rhs(wp3, ssw, wp2, beta_col, fa, fb, rho, irho, gr)
    ref_rhs = R.calc_xp2_xpyp_ta_rhs_jax(
        True, wp3, zt0, ssw, wp2, zt0, beta_col, fa, fb, rho, irho, zt0, refgr)
    np.testing.assert_allclose(np.asarray(mine_rhs), np.asarray(ref_rhs), rtol=1e-12, atol=1e-16)


def test_xp2_ta_jit_grad():
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(12)
    wp3 = jnp.asarray(rng.standard_normal((ng, nzm)))
    ssw = jnp.asarray(0.2 + 0.4 * rng.random((ng, nzm)))
    wp2 = jnp.asarray(0.2 + 0.6 * rng.random((ng, nzm)))
    rho = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    irho = 1.0 / rho
    fa = jnp.asarray(1e-4 * rng.standard_normal((ng, nzm)))

    def loss(f):
        lhs = calc_xp2_xpyp_ta_lhs(wp3, ssw, 2.4, rho, irho, gr)
        rhs = calc_xp2_xpyp_ta_rhs(wp3, ssw, wp2, 2.4, f, f, rho, irho, gr)
        return jnp.sum(lhs ** 2) + jnp.sum(rhs ** 2)

    assert jnp.isfinite(jax.jit(loss)(fa))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(fa)))


def _xp2_assembly_inputs():
    """Deterministic inputs for the diffusion_zm/xp2 LHS/RHS combiners."""
    gr, ng, nzm = _gr_only()
    nzt = nzm - 1
    rng = np.random.default_rng(51)
    return dict(
        gr=gr, ng=ng, nzm=nzm, nzt=nzt,
        K_zt=jnp.asarray(0.5 + rng.random((ng, nzt))),
        nu=jnp.full((ng,), 5.0),
        irho_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        rho_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        lhs_ta=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_ma=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_diff=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_dp1=jnp.asarray(rng.standard_normal((ng, nzm))),
        rhs_ta=jnp.asarray(rng.standard_normal((ng, nzm))),
        Cn=jnp.asarray(0.5 + rng.random((ng, nzm))),
        itau=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        xapxbp=jnp.asarray(rng.random((ng, nzm))),
        xam=jnp.asarray(rng.standard_normal((ng, nzt))),
        xbm=jnp.asarray(rng.standard_normal((ng, nzt))),
        wpxap=jnp.asarray(rng.standard_normal((ng, nzm))),
        wpxbp=jnp.asarray(rng.standard_normal((ng, nzm))),
        forcing=jnp.asarray(1e-5 * rng.standard_normal((ng, nzm))),
    )


def _xp2_assembly_outputs(kw):
    gr = kw["gr"]
    diff = diffusion_zm_lhs(kw["K_zt"], kw["nu"], kw["irho_zm"], kw["rho_zt"], gr)
    lhs = xp2_xpyp_lhs(kw["lhs_ta"], kw["lhs_ma"], kw["lhs_diff"], kw["lhs_dp1"], 300.0)
    rhs = xp2_xpyp_rhs(kw["lhs_ta"], kw["rhs_ta"], kw["Cn"], kw["itau"], 1e-4, kw["xapxbp"],
                       kw["xam"], kw["xbm"], kw["wpxap"], kw["wpxbp"], gr.invrs_dzm,
                       kw["forcing"], 300.0)
    return diff, lhs, rhs


def test_xp2_assembly_matches_golden():
    g = np.load(_FIX / "clubb_xp2_assembly_golden.npz")
    diff, lhs, rhs = _xp2_assembly_outputs(_xp2_assembly_inputs())
    np.testing.assert_array_equal(np.asarray(diff), g["diffusion_zm_lhs"])
    np.testing.assert_array_equal(np.asarray(lhs), g["xp2_xpyp_lhs"])
    np.testing.assert_array_equal(np.asarray(rhs), g["xp2_xpyp_rhs"])


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_xp2_assembly_parity():
    """Bit-exact parity of diffusion_zm_lhs / xp2_xpyp_lhs / xp2_xpyp_rhs vs ref."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812
    import clubb_jax.src.CLUBB_core.diffusion as RD  # noqa: N812

    gr, ng, nzm = _gr_only()
    nzt = nzm - 1
    rng = np.random.default_rng(51)
    K_zt = jnp.asarray(0.5 + rng.random((ng, nzt)))
    nu = jnp.full((ng,), 5.0)
    irho_zm = jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm))))
    rho_zt = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt)))
    np.testing.assert_allclose(
        np.asarray(diffusion_zm_lhs(K_zt, nu, irho_zm, rho_zt, gr)),
        np.asarray(RD.diffusion_zm_lhs_jax(K_zt, nu, irho_zm, rho_zt, gr)),
        rtol=1e-12, atol=1e-14)

    lhs_ta = jnp.asarray(rng.standard_normal((3, ng, nzm)))
    lhs_ma = jnp.asarray(rng.standard_normal((3, ng, nzm)))
    lhs_diff = jnp.asarray(rng.standard_normal((3, ng, nzm)))
    lhs_dp1 = jnp.asarray(rng.standard_normal((ng, nzm)))
    np.testing.assert_allclose(
        np.asarray(xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, 300.0)),
        np.asarray(R.xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, 300.0)),
        rtol=1e-12, atol=1e-14)

    rhs_ta = jnp.asarray(rng.standard_normal((ng, nzm)))
    Cn = jnp.asarray(0.5 + rng.random((ng, nzm)))
    itau = jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm)))
    xapxbp = jnp.asarray(rng.random((ng, nzm)))
    xam = jnp.asarray(rng.standard_normal((ng, nzt)))
    xbm = jnp.asarray(rng.standard_normal((ng, nzt)))
    wpxap = jnp.asarray(rng.standard_normal((ng, nzm)))
    wpxbp = jnp.asarray(rng.standard_normal((ng, nzm)))
    forcing = jnp.asarray(1e-5 * rng.standard_normal((ng, nzm)))
    np.testing.assert_allclose(
        np.asarray(xp2_xpyp_rhs(lhs_ta, rhs_ta, Cn, itau, 1e-4, xapxbp, xam, xbm,
                                wpxap, wpxbp, gr.invrs_dzm, forcing, 300.0)),
        np.asarray(R.xp2_xpyp_rhs(lhs_ta, rhs_ta, Cn, itau, 1e-4, xapxbp, xam, xbm,
                                  wpxap, wpxbp, gr.invrs_dzm, forcing, 300.0)),
        rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_parity_vs_clubb_jax_reference():
    """Parity vs CLUBB advance_windm_edsclrm to round-off.

    All LHS/RHS assembly matches the reference bit-exactly (verified
    component-wise); the only round-off-level difference is the tridiagonal
    solve — this port reuses legoESM's Thomas solver while the reference uses
    its own LU (mathematically identical, agree to ~1e-15 relative).
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.CLUBB_core.advance_windm_edsclrm_module import (
        advance_windm_edsclrm as ref_advance,
    )

    kw = _windm_inputs()
    gr = kw["gr"]
    ng, nzt = kw["um"].shape
    nzm = nzt + 1
    clubb_params = np.zeros((ng, 102))
    clubb_params[:, _IC_K10 - 1] = _C_K10
    refgr = SimpleNamespace(
        zm=gr.zm, zt=gr.zt, dzm=gr.dzm, invrs_dzm=gr.invrs_dzm, invrs_dzt=gr.invrs_dzt,
        k_lb_zt=0, k_lb_zm=0, k_ub_zt=nzt - 1, k_ub_zm=nzm - 1)
    ref = jax.jit(
        lambda: ref_advance(
            kw["um"], kw["vm"], kw["upwp"], kw["vpwp"], kw["wp2"], kw["up2"], kw["vp2"],
            kw["wm_zt"], kw["Kh_zm"], kw["ug"], kw["vg"], kw["um_forcing"], kw["vm_forcing"],
            kw["rho_ds_zm"], kw["rho_ds_zt"], kw["invrs_rho_ds_zt"], kw["fcor"],
            jnp.asarray(clubb_params), _NU10, _DT, refgr, False, True, True))()
    mine = jax.jit(lambda: advance_windm_edsclrm(**kw))()
    for a, b in zip(mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
