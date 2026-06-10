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
    advance_xp2_xpyp,
    calc_up2_vp2_lhs,
    calc_xp2_xpyp_lhs,
    calc_xp2_xpyp_ta_lhs,
    calc_xp2_xpyp_ta_rhs,
    clip_variance,
    diffusion_zm_lhs,
    pos_definite_variances,
    term_dp1_lhs,
    term_dp1_rhs,
    term_ma_zm_lhs,
    term_pr1,
    term_pr2,
    term_tp_rhs,
    xp2_xpyp_lhs,
    xp2_xpyp_rhs,
    xp2_xpyp_uv_rhs,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402

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


# ---------------------------------------------------------------------------
# term_ma_zm_lhs (centered mean advection) + xp2/xpyp LHS-assembly wrappers
# ---------------------------------------------------------------------------

def _weights_zm2zt(gr):
    """calc_zm2zt_weights (grid_class.F90), ascending grid: (ng, nzt, 2)."""
    zm = np.asarray(gr.zm)
    zt = np.asarray(gr.zt)
    total = (zm[:, 1:] - zm[:, :-1]) + 1.0e-30
    w_above = (zt - zm[:, :-1]) / total      # M_ABOVE: weight of zm[k]
    w_below = (zm[:, 1:] - zt) / total        # M_BELOW: weight of zm[k+1]
    return np.stack([w_above, w_below], axis=-1)


def test_term_ma_zm_lhs_uniform_grid_centered():
    """On a uniform grid the zm2zt weights are 1/2 → main diag 0, off-diags ±fac/2."""
    ng, nzm = 2, 9
    zm = jnp.asarray(np.tile(np.linspace(0.0, 1600.0, nzm), (ng, 1)))
    gr = make_clubb_grid(zm, 0.5 * (zm[:, 1:] + zm[:, :-1]))
    rng = np.random.default_rng(11)
    wm_zm = jnp.asarray(0.03 * rng.standard_normal((ng, nzm)))
    band = np.asarray(term_ma_zm_lhs(wm_zm, gr))
    assert band.shape == (3, ng, nzm)
    # Boundaries (k=0, k=nzm-1) are zero rows.
    assert np.allclose(band[:, :, 0], 0.0) and np.allclose(band[:, :, -1], 0.0)
    fac = np.asarray(wm_zm)[:, 1:-1] * np.asarray(gr.invrs_dzm)[:, 1:-1]
    np.testing.assert_allclose(band[0, :, 1:-1], 0.5 * fac, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(band[1, :, 1:-1], 0.0, atol=1e-14)
    np.testing.assert_allclose(band[2, :, 1:-1], -0.5 * fac, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_term_ma_zm_lhs_parity():
    """Bit-exact parity vs mean_adv.term_ma_zm_lhs_jax on a STRETCHED grid.

    The stretched grid is the case where the zm2zt weight convention (which
    column is M_ABOVE vs M_BELOW) actually matters; on a uniform grid both are
    1/2 and a wrong convention would pass silently.
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.mean_adv as RMA  # noqa: N812

    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(73)
    wm_zm = jnp.asarray(0.02 * rng.standard_normal((ng, nzm)))
    refgr = SimpleNamespace(invrs_dzm=gr.invrs_dzm,
                            weights_zm2zt=jnp.asarray(_weights_zm2zt(gr)))
    np.testing.assert_allclose(
        np.asarray(term_ma_zm_lhs(wm_zm, gr)),
        np.asarray(RMA.term_ma_zm_lhs_jax(wm_zm, refgr)),
        rtol=1e-12, atol=1e-14)


def _lhs_wrapper_inputs(seed):
    gr, ng, nzm = _gr_only()
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    return dict(
        gr=gr, ng=ng, nzm=nzm, nzt=nzt,
        lhs_ta=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        lhs_ma=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        Kh_zt=jnp.asarray(0.5 + rng.random((ng, nzt))),
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        Cn=jnp.asarray(0.5 + rng.random((ng, nzm))),
        itau=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        nu=jnp.full((ng,), 5.0),
    )


def test_calc_xp2_xpyp_lhs_self_consistent():
    """The wrapper composes diffusion_zm_lhs + dp1 + xp2_xpyp_lhs (CI-side check)."""
    p = _lhs_wrapper_inputs(91)
    c_K2, gamma, dt = 0.025, 1.5, 300.0
    lhs, lhs_diff, dp1 = calc_xp2_xpyp_lhs(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K2, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], p["Cn"], p["itau"], gamma, dt, p["gr"])
    assert lhs.shape == (3, p["ng"], p["nzm"]) and dp1.shape == (p["ng"], p["nzm"])
    exp_diff = diffusion_zm_lhs(c_K2 * p["Kh_zt"], p["nu"], p["invrs_rho_ds_zm"],
                                p["rho_ds_zt"], p["gr"])
    exp_dp1 = term_dp1_lhs(p["Cn"], p["itau"])
    exp_lhs = xp2_xpyp_lhs(p["lhs_ta"], p["lhs_ma"], exp_diff, exp_dp1 * gamma, dt)
    np.testing.assert_array_equal(np.asarray(lhs_diff), np.asarray(exp_diff))
    np.testing.assert_array_equal(np.asarray(dp1), np.asarray(exp_dp1))
    np.testing.assert_array_equal(np.asarray(lhs), np.asarray(exp_lhs))


def test_calc_up2_vp2_lhs_self_consistent():
    """up2/vp2 wrapper composes Kw9 diffusion + C4/C14 dp1 + xp2_xpyp_lhs."""
    p = _lhs_wrapper_inputs(92)
    ng, nzm = p["ng"], p["nzm"]
    c_K9 = jnp.full((ng,), 0.13)
    C4, C14, gamma, dt = 5.2, 1.0, 1.5, 300.0
    itau_C4 = p["itau"]
    itau_C14 = jnp.asarray(itau_C4) * 1.3
    lhs, lhs_diff, dp1_C4, dp1_C14 = calc_up2_vp2_lhs(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K9, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], C4, C14, itau_C4, itau_C14, gamma, dt, p["gr"])
    assert lhs.shape == (3, ng, nzm)
    exp_diff = diffusion_zm_lhs(c_K9[:, None] * p["Kh_zt"], p["nu"],
                                p["invrs_rho_ds_zm"], p["rho_ds_zt"], p["gr"])
    exp_c4 = term_dp1_lhs((2.0 / 3.0) * C4 * jnp.ones((ng, nzm)), itau_C4)
    exp_c14 = term_dp1_lhs((1.0 / 3.0) * C14 * jnp.ones((ng, nzm)), itau_C14)
    exp_lhs = xp2_xpyp_lhs(p["lhs_ta"], p["lhs_ma"], exp_diff,
                           (exp_c4 + exp_c14) * gamma, dt)
    np.testing.assert_array_equal(np.asarray(lhs_diff), np.asarray(exp_diff))
    np.testing.assert_array_equal(np.asarray(dp1_C4), np.asarray(exp_c4))
    np.testing.assert_array_equal(np.asarray(dp1_C14), np.asarray(exp_c14))
    np.testing.assert_array_equal(np.asarray(lhs), np.asarray(exp_lhs))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_calc_xp2_xpyp_lhs_parity():
    """Bit-exact parity of both LHS-assembly wrappers vs the reference."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812

    p = _lhs_wrapper_inputs(93)
    c_K2, gamma, dt = 0.025, 1.5, 300.0
    mine = calc_xp2_xpyp_lhs(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K2, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], p["Cn"], p["itau"], gamma, dt, p["gr"])
    ref = R.calc_xp2_xpyp_lhs_jax(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K2, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], p["Cn"], p["itau"], gamma, dt, p["gr"])
    for a, b in zip(mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)

    ng, nzm = p["ng"], p["nzm"]
    c_K9 = jnp.full((ng,), 0.13)
    C4, C14 = 5.2, 1.0
    itau_C14 = jnp.asarray(p["itau"]) * 1.3
    mine9 = calc_up2_vp2_lhs(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K9, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], C4, C14, p["itau"], itau_C14, gamma, dt, p["gr"])
    ref9 = R.calc_up2_vp2_lhs_jax(
        p["lhs_ta"], p["lhs_ma"], p["Kh_zt"], c_K9, p["nu"], p["invrs_rho_ds_zm"],
        p["rho_ds_zt"], C4, C14, p["itau"], itau_C14, gamma, dt, p["gr"])
    for a, b in zip(mine9, ref9):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# xp2_xpyp_uv_rhs (up2/vp2 explicit RHS) + pos_definite_variances
# ---------------------------------------------------------------------------

def _uv_rhs_inputs(seed):
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(seed)
    ni = nzm - 2

    def zm(s=1.0):
        return jnp.asarray(s * rng.standard_normal((ng, nzm)))

    return dict(
        gr=gr, ng=ng, nzm=nzm,
        rhs_ta_this=zm(1e-3),
        this_pre=jnp.asarray(0.3 + rng.random((ng, nzm))),
        other_pre=jnp.asarray(0.3 + rng.random((ng, nzm))),
        this_wp=jnp.asarray(0.05 * rng.standard_normal((ng, nzm))),
        this_dvel_dz=jnp.asarray(1e-2 * rng.standard_normal((ng, ni))),
        lhs_splat=jnp.zeros((ng, nzm)),   # CAM C_wp2_splat = 0
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        lhs_ta=jnp.asarray(rng.standard_normal((3, ng, nzm))),
        C_uu_shr=jnp.asarray(0.4 * np.ones((ng, 1))),
        C4=jnp.asarray(5.2 * np.ones((ng, 1))),
        C14=jnp.asarray(1.0 * np.ones((ng, 1))),
        invrs_tau_C4_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        invrs_tau_C14_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        lhs_dp1_C4=jnp.asarray(rng.random((ng, nzm))),
        lhs_dp1_C14=jnp.asarray(rng.random((ng, nzm))),
        pr2=jnp.asarray(np.abs(rng.standard_normal((ng, ni)))),
        omg=-0.5, dt=300.0, w_tol_sqd=float((2.0e-2) ** 2),
    )


def _call_uv_rhs(p):
    return xp2_xpyp_uv_rhs(
        p["rhs_ta_this"], p["this_pre"], p["other_pre"], p["this_wp"],
        p["this_dvel_dz"], p["lhs_splat"], p["wp2"], p["lhs_ta"], p["C_uu_shr"],
        p["C4"], p["C14"], p["invrs_tau_C4_zm"], p["invrs_tau_C14_zm"],
        p["lhs_dp1_C4"], p["lhs_dp1_C14"], p["pr2"], p["omg"], p["dt"],
        p["w_tol_sqd"], False, None)


def test_uv_rhs_boundaries_and_shape():
    p = _uv_rhs_inputs(31)
    rhs = np.asarray(_call_uv_rhs(p))
    assert rhs.shape == (p["ng"], p["nzm"])
    # lower BC carries the current value, upper BC is w_tol_sqd
    np.testing.assert_array_equal(rhs[:, 0], np.asarray(p["this_pre"])[:, 0])
    np.testing.assert_allclose(rhs[:, -1], p["w_tol_sqd"], rtol=0, atol=0)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_uv_rhs_parity():
    """Bit-exact parity vs advance_xp2_xpyp_module.xp2_xpyp_uv_rhs (CAM tree)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812

    p = _uv_rhs_inputs(32)
    ref = R.xp2_xpyp_uv_rhs(
        np.asarray(p["rhs_ta_this"]), np.asarray(p["this_pre"]),
        np.asarray(p["other_pre"]), np.asarray(p["this_wp"]),
        np.asarray(p["this_dvel_dz"]), np.asarray(p["lhs_splat"]),
        np.asarray(p["wp2"]), np.asarray(p["lhs_ta"]), np.asarray(p["C_uu_shr"]),
        np.asarray(p["C4"]), np.asarray(p["C14"]),
        np.asarray(p["invrs_tau_C4_zm"]), np.asarray(p["invrs_tau_C14_zm"]),
        np.asarray(p["lhs_dp1_C4"]), np.asarray(p["lhs_dp1_C14"]),
        np.asarray(p["pr2"]), p["omg"], p["dt"], p["w_tol_sqd"], False, None)
    np.testing.assert_allclose(np.asarray(_call_uv_rhs(p)), np.asarray(ref),
                               rtol=1e-12, atol=1e-14)


def test_uv_rhs_jit_static_coriolis_gate():
    """JIT xp2_xpyp_uv_rhs through the public signature with l_coriolis static.

    Exercises both the gate-off (CAM default) and gate-on branches under jit,
    proving the static feature-gate contract (codex review).
    """
    p = _uv_rhs_inputs(33)
    jf = jax.jit(xp2_xpyp_uv_rhs, static_argnums=(19,))
    off = jf(p["rhs_ta_this"], p["this_pre"], p["other_pre"], p["this_wp"],
             p["this_dvel_dz"], p["lhs_splat"], p["wp2"], p["lhs_ta"],
             p["C_uu_shr"], p["C4"], p["C14"], p["invrs_tau_C4_zm"],
             p["invrs_tau_C14_zm"], p["lhs_dp1_C4"], p["lhs_dp1_C14"], p["pr2"],
             p["omg"], p["dt"], p["w_tol_sqd"], False, None)
    assert jnp.all(jnp.isfinite(off))
    fcor = jnp.asarray(1e-4 * np.ones((p["ng"], 1)))
    on = jf(p["rhs_ta_this"], p["this_pre"], p["other_pre"], p["this_wp"],
            p["this_dvel_dz"], p["lhs_splat"], p["wp2"], p["lhs_ta"],
            p["C_uu_shr"], p["C4"], p["C14"], p["invrs_tau_C4_zm"],
            p["invrs_tau_C14_zm"], p["lhs_dp1_C4"], p["lhs_dp1_C14"], p["pr2"],
            p["omg"], p["dt"], p["w_tol_sqd"], True, fcor)
    assert jnp.all(jnp.isfinite(on))
    # gate-on differs from gate-off only by the -2*fcor*wp interior term
    diff = np.asarray(on)[:, 1:-1] - np.asarray(off)[:, 1:-1]
    exp = -2.0 * np.asarray(fcor) * np.asarray(p["this_wp"])[:, 1:-1]
    np.testing.assert_allclose(diff, exp, rtol=1e-12, atol=1e-14)


def test_pos_definite_variances_jit_static_args():
    """JIT pos_definite_variances with the static hole-fill contract."""
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(43)
    field = jnp.asarray(0.4 + rng.random((ng, nzm)))
    rho_ds = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    jf = jax.jit(pos_definite_variances, static_argnums=(4, 5, 6))
    out = jf(field, rho_ds, gr.dzm, 0.0, 1, nzm - 2, 2)
    assert jnp.all(jnp.isfinite(out))


def test_pos_definite_variances_fills_and_conserves():
    gr, ng, nzm = _gr_only()
    rng = np.random.default_rng(41)
    field = np.asarray(0.4 + rng.random((ng, nzm)))
    field[:, 3] = -0.2   # punch a hole
    field = jnp.asarray(field)
    rho_ds = jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm)))
    out = pos_definite_variances(field, rho_ds, gr.dzm, 0.0, 1, nzm - 2,
                                 fill_holes_type=2)
    assert np.all(np.asarray(out)[:, 1:nzm - 1] >= -1e-12)
    assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# clip_variance + advance_xp2_xpyp main
# ---------------------------------------------------------------------------

def test_clip_variance_floors_interior_keeps_top():
    rng = np.random.default_rng(61)
    xp2 = jnp.asarray(rng.standard_normal((2, 9)))   # has negatives
    out = np.asarray(clip_variance(xp2, 0.5))
    assert np.all(out[:, :-1] >= 0.5 - 1e-12)     # floored over 0..nzm-2
    np.testing.assert_array_equal(out[:, -1], np.asarray(xp2)[:, -1])  # top untouched
    # array threshold + cap
    lo = jnp.asarray(0.1 + rng.random((2, 9)))
    out2 = np.asarray(clip_variance(xp2, lo, threshold_hi=2.0))
    assert np.all(out2[:, :-1] >= np.asarray(lo)[:, :-1] - 1e-12)
    assert np.all(out2[:, :-1] <= 2.0 + 1e-12)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_clip_variance_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.clip_explicit as RC  # noqa: N812
    rng = np.random.default_rng(62)
    xp2 = jnp.asarray(rng.standard_normal((3, 10)))
    np.testing.assert_array_equal(
        np.asarray(clip_variance(xp2, 0.3)),
        np.asarray(RC.clip_variance(xp2, 0.3)))
    lo = jnp.asarray(0.1 + rng.random((3, 10)))
    np.testing.assert_array_equal(
        np.asarray(clip_variance(xp2, lo, 1.5)),
        np.asarray(RC.clip_variance(xp2, lo, 1.5)))


def _advance_xp2_inputs(seed=70, ng=2, nzt=10):
    nzm = nzt + 1
    rng = np.random.default_rng(seed)
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.1 ** np.arange(nzm)[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    cfg = CLUBBConfig()

    def zmf(s=1.0, base=0.0):
        return jnp.asarray(base + s * rng.standard_normal((ng, nzm)))

    def ztf(s=1.0, base=0.0):
        return jnp.asarray(base + s * rng.standard_normal((ng, nzt)))

    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    wp2_zt = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzt)))
    sigma = jnp.asarray(0.1 + 0.3 * rng.random((ng, nzm)))   # < 1
    return dict(
        rtm=ztf(1e-3, 8e-3), thlm=ztf(0.5, 290.0), um=ztf(2.0, 5.0), vm=ztf(1.0),
        rtp2=jnp.asarray(1e-6 + 1e-6 * rng.random((ng, nzm))),
        thlp2=jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm))),
        rtpthlp=jnp.asarray(1e-4 * rng.standard_normal((ng, nzm))),
        up2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        vp2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        wprtp=zmf(1e-4), wpthlp=zmf(1e-2), wpthvp=zmf(1e-2), upwp=zmf(0.05),
        vpwp=zmf(0.05), wp2=wp2, wp2_zt=wp2_zt,
        wp3_on_wp2=zmf(0.1), wp3_on_wp2_zt=ztf(0.1),
        sigma_sqd_w=sigma, thv_ds_zm=zmf(1.0, 300.0),
        Kh_zt=jnp.asarray(1.0 + 3.0 * rng.random((ng, nzt))),
        Cn=jnp.asarray(np.full((ng, nzm), cfg.params.C2rt)),
        invrs_tau_xp2_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        invrs_tau_C4_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        invrs_tau_C14_zm=jnp.asarray(1e-3 + 1e-3 * rng.random((ng, nzm))),
        rho_ds_zm=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzm))),
        rho_ds_zt=jnp.asarray(1.0 + 0.1 * rng.random((ng, nzt))),
        invrs_rho_ds_zm=jnp.asarray(1.0 / (1.0 + 0.1 * rng.random((ng, nzm)))),
        wm_zm=zmf(0.02),
        rtp2_forcing=zmf(1e-8), thlp2_forcing=zmf(1e-5), rtpthlp_forcing=zmf(1e-6),
        nu2=jnp.full((ng,), cfg.params.nu2), nu9=jnp.full((ng,), cfg.params.nu9),
        dt=300.0, gr=gr, config=cfg,
    )


def test_advance_xp2_xpyp_runs_and_bounds():
    kw = _advance_xp2_inputs()
    cfg = kw["config"]
    rtp2, thlp2, rtpthlp, up2, vp2 = advance_xp2_xpyp(**kw)
    ng, nzm = kw["wp2"].shape
    for f in (rtp2, thlp2, rtpthlp, up2, vp2):
        assert f.shape == (ng, nzm) and np.all(np.isfinite(np.asarray(f)))
    # variances floored over interior (0..nzm-2)
    assert np.all(np.asarray(rtp2)[:, :-1] >= cfg.rt_tol ** 2 - 1e-12)
    assert np.all(np.asarray(thlp2)[:, :-1] >= cfg.thl_tol ** 2 - 1e-12)
    assert np.all(np.asarray(up2)[:, :-1] >= cfg.w_tol ** 2 - 1e-12)
    assert np.all(np.asarray(vp2)[:, :-1] >= cfg.w_tol ** 2 - 1e-12)
    # rtpthlp is a covariance: Cauchy-Schwarz bounded on the interior (where
    # clip_covar acts); boundaries carry the solve BC values (upper = 0, left
    # unchanged by clip_covar) — matching the reference (no hole-fill on a
    # covariance). Finiteness already asserted on all levels above.
    bound = 0.99 * np.sqrt(np.asarray(rtp2) * np.asarray(thlp2))
    assert np.all(np.abs(np.asarray(rtpthlp))[:, 1:-1] <= bound[:, 1:-1] + 1e-12)
    np.testing.assert_array_equal(np.asarray(rtpthlp)[:, -1], 0.0)  # upper BC


def test_advance_xp2_xpyp_jit_grad():
    kw = _advance_xp2_inputs()

    def loss(wp2):
        out = advance_xp2_xpyp(**dict(kw, wp2=wp2))
        return sum(jnp.sum(f ** 2) for f in out)

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wp2"])))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_advance_xp2_xpyp_parity():
    """Round-off parity of the full 5-moment advance vs the reference.

    All LHS/RHS builders are individually bit-exact; the only round-off-level
    difference is the tridiagonal solve (legoESM Thomas vs the reference LU).
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module as R  # noqa: N812
    from clubb_jax.src.CLUBB_core import parameter_indices as PI  # noqa: N812
    from clubb_jax.src.derived_types.grid_class import (  # noqa: N812
        Grid,
        calc_zm2zt_weights,
        calc_zt2zm_weights,
    )

    kw = _advance_xp2_inputs(seed=71)
    cfg = kw["config"]
    p = cfg.params
    gr = kw["gr"]
    ng, nzm = kw["wp2"].shape
    nzt = nzm - 1

    clubb_params = np.zeros((ng, 102))
    clubb_params[:, PI.ic_K2 - 1] = p.c_K2
    clubb_params[:, PI.ic_K9 - 1] = p.c_K9
    clubb_params[:, PI.iC2rt - 1] = p.C2rt
    clubb_params[:, PI.iC4 - 1] = p.C4
    clubb_params[:, PI.iC14 - 1] = p.C14
    clubb_params[:, PI.iC_uu_shr - 1] = p.C_uu_shr
    clubb_params[:, PI.iC_uu_buoy - 1] = p.C_uu_buoy
    clubb_params[:, PI.ibeta - 1] = p.beta

    zm_np, zt_np, dzt_np = np.asarray(gr.zm), np.asarray(gr.zt), np.asarray(gr.dzt)
    refgr = Grid(
        nzm=nzm, nzt=nzt, ngrdcol=ng, zm=gr.zm, zt=gr.zt, dzm=gr.dzm, dzt=gr.dzt,
        invrs_dzm=gr.invrs_dzm, invrs_dzt=gr.invrs_dzt,
        weights_zt2zm=jnp.asarray(calc_zt2zm_weights(nzm, nzt, ng, zm_np, zt_np)),
        weights_zm2zt=jnp.asarray(calc_zm2zt_weights(nzm, nzt, ng, zm_np, zt_np, dzt_np)),
        k_lb_zm=0, k_ub_zm=nzm - 1, k_lb_zt=0, k_ub_zt=nzt - 1,
        grid_dir_indx=1, grid_dir=1.0)
    flags = SimpleNamespace(
        l_upwind_xpyp_ta=True, l_lmm_stepping=False,
        l_min_xp2_from_corr_wx=True, fill_holes_type=2,
        l_ho_nontrad_coriolis=False)
    nu_vrd = SimpleNamespace(nu2=np.asarray(kw["nu2"]), nu9=np.asarray(kw["nu9"]))
    lhs_splat = jnp.zeros((ng, nzm))

    # term_pr2 buoyancy uses gravity: the reference CLUBB grav differs from
    # legoESM constants.g by ~0.04%. Patch the reference module constant to the
    # legoESM value so the comparison isolates numerics from the constant basis.
    R.grav = float(constants.g)

    ref = R.advance_xp2_xpyp(
        kw["Kh_zt"], jnp.asarray(clubb_params), kw["dt"], jnp.zeros((ng,)), flags,
        refgr, kw["invrs_rho_ds_zm"], kw["invrs_tau_C14_zm"], kw["invrs_tau_C4_zm"],
        kw["invrs_tau_xp2_zm"], False, lhs_splat, ng, nu_vrd, nzm, kw["rho_ds_zm"],
        kw["rho_ds_zt"], kw["rtm"], kw["rtp2"], kw["rtp2_forcing"], kw["rtpthlp"],
        kw["rtpthlp_forcing"], kw["sigma_sqd_w"], None, kw["thlm"], kw["thlp2"],
        kw["thlp2_forcing"], kw["thv_ds_zm"], kw["um"], kw["up2"], kw["upwp"],
        kw["vm"], kw["vp2"], kw["vpwp"], kw["wm_zm"], kw["wp2"], kw["wp2_zt"],
        kw["wp3_on_wp2"], kw["wp3_on_wp2_zt"], kw["wprtp"], kw["wpthlp"], kw["wpthvp"])

    mine = advance_xp2_xpyp(**kw)
    for a, b in zip(mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
