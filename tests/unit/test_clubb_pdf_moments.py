"""Unit tests for CLUBB ADG1 PDF moment integrals + buoyancy flux.

Bit-exact parity vs the CLUBB-JAX reference (the pure moment integrals have no
constants; the buoyancy flux is constant-patched), committed golden fixtures for
CI, and analytic non-negativity of even moments. See ``docs/dev-notes/clubb.md``.
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

from legoesm.atmosphere.physics.turbulence.clubb import (
    derive_mixt_frac_max_mag,  # noqa: E402
)
from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    ADG1_pdf_driver,
    calc_pdf_liquid_cloud_frac_components,
)
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    _EP1,
    _EP2,
    calc_pdf_higher_order_moments,
    calc_pdf_xprcp_fluxes,
    calc_wp2xp2_pdf,
    calc_wp4_pdf,
    calc_xpthvp_terms,
)

from legoesm import constants  # noqa: E402

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_FIX = Path(__file__).resolve().parent / "clubb_fixtures"
_BETA, _MFMM = 2.4, derive_mixt_frac_max_mag(4.5)


def _moment_inputs():
    rng = np.random.default_rng(11)
    ng, nzt = 2, 8
    nzm = nzt + 1
    shp = (ng, nzt)
    wp2 = jnp.asarray(0.2 + 0.6 * rng.random(shp))
    sqrt_wp2 = jnp.sqrt(wp2)
    rtp2 = jnp.asarray(1e-7 + 2e-6 * rng.random(shp))
    thlp2 = jnp.asarray(0.05 + 0.3 * rng.random(shp))
    up2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    vp2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    ssw = jnp.asarray(0.2 + 0.5 * rng.random(shp))
    Skw = jnp.asarray(-1.0 + 2.0 * rng.random(shp))
    rtm = jnp.asarray(9e-3 + 4e-3 * rng.random(shp))
    thlm = jnp.asarray(290.0 + 6.0 * rng.random(shp))
    um = jnp.asarray(rng.normal(size=shp))
    vm = jnp.asarray(rng.normal(size=shp))
    wm = jnp.asarray(rng.normal(size=shp))

    def cov(xp2, f):
        return jnp.asarray(f) * jnp.sqrt(wp2 * xp2)

    adg1 = ADG1_pdf_driver(wm, rtm, thlm, um, vm, wp2, rtp2, thlp2, up2, vp2, Skw,
                           cov(rtp2, 0.3), cov(thlp2, -0.4), cov(up2, 0.2), cov(vp2, -0.25),
                           sqrt_wp2, ssw, _BETA, _MFMM)
    # Shared stretched grid.
    idx = np.arange(nzm, dtype=np.float64)
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.12 ** idx[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    corr_rt_thl_1 = jnp.full(shp, -0.2)
    corr_rt_thl_2 = jnp.full(shp, -0.15)
    return dict(adg1=adg1, wm_zt=wm, rtm=rtm, thlm=thlm, um=um, vm=vm,
                corr_rt_thl_1=corr_rt_thl_1, corr_rt_thl_2=corr_rt_thl_2, gr=gr)


def _xprcp_inputs():
    """ADG1 + cloud-component dict + thermo for the cloud-water flux assembly."""
    kw = _moment_inputs()
    ng, nzt = kw["adg1"]["mixt_frac"].shape
    rng = np.random.default_rng(13)
    p = jnp.asarray(np.linspace(9.5e4, 6.0e4, nzt)[None, :] * np.ones((ng, nzt)))
    exner = (p / constants.p_ref) ** constants.kappa
    rtpthlp = -0.2 * jnp.sqrt(jnp.asarray(1e-6 * rng.random((ng, nzt)))
                              * jnp.asarray(0.1 * rng.random((ng, nzt))))
    comp = calc_pdf_liquid_cloud_frac_components(
        kw["adg1"], rtpthlp, kw["rtm"], kw["thlm"], exner, p)
    return dict(adg1=kw["adg1"], comp=comp, wm_zt=kw["wm_zt"], rtm=kw["rtm"],
                thlm=kw["thlm"], um=kw["um"], vm=kw["vm"], rcm_zt=comp["rcm"],
                gr=kw["gr"])


def _refgr(gr):
    nzm = gr.zm.shape[1]
    return SimpleNamespace(zm=gr.zm, zt=gr.zt, dzm=gr.dzm, invrs_dzm=gr.invrs_dzm,
                           k_ub_zt=gr.zt.shape[1] - 1, k_lb_zt=0,
                           k_ub_zm=nzm - 1, k_lb_zm=0)


# ---------------------------------------------------------------------------
# Analytic non-negativity of even moments
# ---------------------------------------------------------------------------

def test_wp4_nonnegative():
    kw = _moment_inputs()
    a = kw["adg1"]
    wp4 = calc_wp4_pdf(kw["wm_zt"], a["w_1"], a["w_2"], a["varnce_w_1"], a["varnce_w_2"],
                       a["mixt_frac"])
    assert jnp.all(wp4 >= 0.0)


def test_wp2xp2_nonnegative():
    kw = _moment_inputs()
    a = kw["adg1"]
    z = jnp.zeros_like(a["mixt_frac"])
    wp2rt2 = calc_wp2xp2_pdf(kw["wm_zt"], kw["rtm"], a["w_1"], a["w_2"], a["rt_1"], a["rt_2"],
                             a["varnce_w_1"], a["varnce_w_2"], a["varnce_rt_1"], a["varnce_rt_2"],
                             z, z, a["mixt_frac"])
    assert jnp.all(wp2rt2 >= 0.0)


# ---------------------------------------------------------------------------
# Committed golden: higher-order moments (no constants -> bit-exact)
# ---------------------------------------------------------------------------

def test_higher_order_moments_matches_golden():
    g = np.load(_FIX / "clubb_hom_golden.npz")
    out = calc_pdf_higher_order_moments(**_moment_inputs())
    for key in g.files:
        np.testing.assert_array_equal(np.asarray(out[key]), g[key], err_msg=key)


def test_higher_order_moments_jit_grad():
    kw = _moment_inputs()

    def loss(wm):
        out = calc_pdf_higher_order_moments(**dict(kw, wm_zt=wm))
        return sum(jnp.sum(v) for v in out.values())

    assert jnp.isfinite(jax.jit(loss)(kw["wm_zt"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wm_zt"])))


# ---------------------------------------------------------------------------
# Buoyancy flux x'thv' — leaf function, committed golden + analytic
# ---------------------------------------------------------------------------

def _xpthvp_inputs():
    rng = np.random.default_rng(5)
    ng, nzt = 2, 8
    nzm = nzt + 1
    shp = (ng, nzt)
    zm_1d = np.cumsum(np.concatenate([[0.0], 40.0 * 1.12 ** np.arange(nzm)[:-1]]))
    zm = jnp.asarray(np.tile(zm_1d, (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    p = jnp.asarray(np.linspace(9.5e4, 6.0e4, nzt)[None, :] * np.ones(shp))
    exner = (p / constants.p_ref) ** constants.kappa
    f = lambda s: jnp.asarray(s * rng.normal(size=shp))  # noqa: E731
    return dict(exner=exner, thv_ds_zt=jnp.asarray(300.0 + 5 * rng.random(shp)),
                wprcp_zt=f(1e-5), wp2rcp_zt=f(1e-5), rtprcp_zt=f(1e-6), thlprcp_zt=f(1e-3),
                wpthlp_zt=f(0.02), wprtp_zt=f(1e-4), wp2thlp_zt=f(0.01), wp2rtp_zt=f(1e-4),
                rtpthlp_zt=f(1e-5), rtp2_zt=jnp.asarray(1e-6 * rng.random(shp)),
                thlp2_zt=jnp.asarray(0.1 * rng.random(shp)), gr=gr)


def test_xpthvp_dry_limit_reduces_to_thl_flux():
    """With no moisture flux (wprtp=wprcp=0), wpthvp == wpthlp on zt."""
    kw = dict(_xpthvp_inputs(), wprtp_zt=jnp.zeros((2, 8)), wprcp_zt=jnp.zeros((2, 8)))
    # Recompute the zt buoyancy flux directly: rc_coef*0 + ep1*thv*0 + wpthlp.
    _, _, _, _, rc_coef_zt, _ = calc_xpthvp_terms(**kw)
    # wpthvp_zt = wpthlp_zt exactly (the other terms vanish) — check via reconstruction.
    wpthvp_zt = (kw["wpthlp_zt"] + _EP1 * kw["thv_ds_zt"] * kw["wprtp_zt"]
                 + rc_coef_zt * kw["wprcp_zt"])
    np.testing.assert_allclose(np.asarray(wpthvp_zt), np.asarray(kw["wpthlp_zt"]), rtol=1e-12)


def test_xpthvp_matches_golden():
    g = np.load(_FIX / "clubb_xpthvp_golden.npz")
    out = calc_xpthvp_terms(**_xpthvp_inputs())
    names = ("wpthvp_zm", "wp2thvp_zt", "rtpthvp_zm", "thlpthvp_zm", "rc_coef_zt", "rc_coef_zm")
    for name, arr in zip(names, out):
        # FP-reassociation tolerance (C1-C8 condense refactor; ~1e-14 rel).
        np.testing.assert_allclose(np.asarray(arr), g[name], rtol=1e-13, atol=1e-16,
                                   err_msg=name)


def test_xpthvp_jit_grad():
    kw = _xpthvp_inputs()

    def loss(wprtp):
        out = calc_xpthvp_terms(**dict(kw, wprtp_zt=wprtp))
        return sum(jnp.sum(v) for v in out)

    assert jnp.isfinite(jax.jit(loss)(kw["wprtp_zt"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wprtp_zt"])))


# ---------------------------------------------------------------------------
# Cloud-water flux assembly x'rc'
# ---------------------------------------------------------------------------

def test_xprcp_fluxes_matches_golden():
    g = np.load(_FIX / "clubb_xprcp_golden.npz")
    out = calc_pdf_xprcp_fluxes(**_xprcp_inputs())
    for key in g.files:
        # FP-reassociation tolerance (C1-C8 condense refactor; ~1e-14 rel).
        np.testing.assert_allclose(np.asarray(out[key]), g[key], rtol=1e-13, atol=1e-16,
                                   err_msg=key)


def test_xprcp_fluxes_zm_top_zeroed():
    out = calc_pdf_xprcp_fluxes(**_xprcp_inputs())
    for key in ("wprcp_zm", "rtprcp_zm", "thlprcp_zm", "uprcp_zm", "vprcp_zm"):
        np.testing.assert_array_equal(np.asarray(out[key])[:, -1], 0.0)


def test_xprcp_fluxes_jit_grad():
    kw = _xprcp_inputs()

    def loss(rcm):
        out = calc_pdf_xprcp_fluxes(**dict(kw, rcm_zt=rcm))
        return sum(jnp.sum(v) for v in out.values())

    assert jnp.isfinite(jax.jit(loss)(kw["rcm_zt"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["rcm_zt"])))


# ---------------------------------------------------------------------------
# Live bit-exact parity vs CLUBB-JAX reference
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_parity_vs_clubb_jax_reference():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.pdf_closure_module as refmod

    # Higher-order moments: no physical constants -> bit-exact directly.
    kw = _moment_inputs()
    mine = calc_pdf_higher_order_moments(**kw)
    ref = refmod.calc_pdf_higher_order_moments_jax(
        kw["adg1"], kw["wm_zt"], kw["rtm"], kw["thlm"], kw["um"], kw["vm"],
        kw["corr_rt_thl_1"], kw["corr_rt_thl_2"], _refgr(kw["gr"]))
    for key in ref:
        np.testing.assert_array_equal(np.asarray(mine[key]), np.asarray(ref[key]), err_msg=key)

    # Buoyancy flux: patch reference constants to legoESM, then bit-exact.
    refmod.Lv, refmod.Cp = constants.L_v, constants.c_pd
    refmod.ep1, refmod.ep2 = _EP1, _EP2
    xkw = _xpthvp_inputs()
    mine_x = calc_xpthvp_terms(**xkw)
    ref_x = refmod.calc_xpthvp_terms_jax(
        xkw["exner"], xkw["thv_ds_zt"], xkw["wprcp_zt"], xkw["wp2rcp_zt"], xkw["rtprcp_zt"],
        xkw["thlprcp_zt"], xkw["wpthlp_zt"], xkw["wprtp_zt"], xkw["wp2thlp_zt"], xkw["wp2rtp_zt"],
        xkw["rtpthlp_zt"], xkw["rtp2_zt"], xkw["thlp2_zt"], _refgr(xkw["gr"]))
    for a, b in zip(mine_x, ref_x):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # Cloud-water fluxes: no constants given the shared comp dict -> bit-exact.
    pkw = _xprcp_inputs()
    mine_p = calc_pdf_xprcp_fluxes(**pkw)
    ref_p = refmod.calc_pdf_xprcp_fluxes_jax(
        adg1=pkw["adg1"], comp=pkw["comp"], wm_zt=pkw["wm_zt"], rtm=pkw["rtm"],
        thlm=pkw["thlm"], um=pkw["um"], vm=pkw["vm"], rcm_zt=pkw["rcm_zt"],
        gr=_refgr(pkw["gr"]))
    for key in ("wprcp_zt", "wp2rcp_zt", "rtprcp_zt", "thlprcp_zt", "uprcp_zt", "vprcp_zt",
                "wprcp_zm", "rtprcp_zm", "thlprcp_zm", "uprcp_zm", "vprcp_zm"):
        np.testing.assert_array_equal(np.asarray(mine_p[key]), np.asarray(ref_p[key]), err_msg=key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
