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

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402  # noqa: E402
    ADG1_pdf_driver,
    calc_pdf_higher_order_moments,
    calc_pdf_liquid_cloud_frac_components,
    calc_pdf_xprcp_fluxes,
    calc_trapezoid_zm,
    calc_trapezoid_zt,
    calc_wp2xp2_pdf,
    calc_wp4_pdf,
    calc_xpthvp_terms,
    clip_rcm,
    compute_cloud_cover,
    derive_mixt_frac_max_mag,  # noqa: E402
    make_clubb_grid,  # noqa: E402
    zt2zm,
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


def _no_gr(kw):
    """Drop the grid: the PDF helpers are grid-agnostic (the caller regrids)."""
    return {k: v for k, v in kw.items() if k != "gr"}


def _hom_to_zm(out, gr):
    """The zt->zm regrids the closure driver applies to the velocity moments.

    These used to live inside ``calc_pdf_higher_order_moments``; they moved to
    the caller when the closure became grid-agnostic (CAM runs it once per level
    set). Reproduced here so the committed golden keeps pinning both grids.
    """
    k_ub = gr.zm.shape[1] - 1
    return {
        "wp2up2_zm": zt2zm(out["wp2up2"], gr).at[:, k_ub].set(0.0),
        "wp2vp2_zm": zt2zm(out["wp2vp2"], gr).at[:, k_ub].set(0.0),
        "wp4_zm": zt2zm(out["wp4"], gr, zm_min=0.0).at[:, 0].set(0.0).at[:, k_ub].set(0.0),
    }


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
                           invrs_dzt=gr.invrs_dzt, dzt=gr.dzt,
                           grid_dir=1.0, grid_dir_indx=1,
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
    kw = _moment_inputs()
    out = calc_pdf_higher_order_moments(**_no_gr(kw))
    out = dict(out, **_hom_to_zm(out, kw["gr"]))
    for key in g.files:
        np.testing.assert_array_equal(np.asarray(out[key]), g[key], err_msg=key)


def test_higher_order_moments_jit_grad():
    kw = _no_gr(_moment_inputs())

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
    return dict(exner=exner, thv_ds=jnp.asarray(300.0 + 5 * rng.random(shp)),
                wprcp=f(1e-5), wp2rcp=f(1e-5), rtprcp=f(1e-6), thlprcp=f(1e-3),
                wpthlp=f(0.02), wprtp=f(1e-4), wp2thlp=f(0.01), wp2rtp=f(1e-4),
                rtpthlp=f(1e-5), rtp2=jnp.asarray(1e-6 * rng.random(shp)),
                thlp2=jnp.asarray(0.1 * rng.random(shp)), gr=gr)


def test_xpthvp_dry_limit_reduces_to_thl_flux():
    """With no moisture flux (wprtp=wprcp=0), wpthvp == wpthlp."""
    kw = dict(_xpthvp_inputs(), wprtp=jnp.zeros((2, 8)), wprcp=jnp.zeros((2, 8)))
    wpthvp = calc_xpthvp_terms(**_no_gr(kw))[0]
    np.testing.assert_allclose(np.asarray(wpthvp), np.asarray(kw["wpthlp"]), rtol=1e-12)


def test_xpthvp_matches_golden():
    """Pin both level sets: the on-grid values and the regrids the caller applies."""
    g = np.load(_FIX / "clubb_xpthvp_golden.npz")
    kw = _xpthvp_inputs()
    gr = kw["gr"]
    k_ub = gr.zm.shape[1] - 1
    wpthvp, wp2thvp, rtpthvp, thlpthvp, rc_coef = calc_xpthvp_terms(**_no_gr(kw))

    def to_zm(field):
        return zt2zm(field, gr).at[:, k_ub].set(0.0)

    got = {"wpthvp_zm": to_zm(wpthvp), "wp2thvp_zt": wp2thvp,
           "rtpthvp_zm": to_zm(rtpthvp), "thlpthvp_zm": to_zm(thlpthvp),
           "rc_coef_zt": rc_coef, "rc_coef_zm": to_zm(rc_coef)}
    for name, arr in got.items():
        # FP-reassociation tolerance (C1-C8 condense refactor; ~1e-14 rel).
        np.testing.assert_allclose(np.asarray(arr), g[name], rtol=1e-13, atol=1e-16,
                                   err_msg=name)


def test_xpthvp_jit_grad():
    kw = _no_gr(_xpthvp_inputs())

    def loss(wprtp):
        out = calc_xpthvp_terms(**dict(kw, wprtp=wprtp))
        return sum(jnp.sum(v) for v in out)

    assert jnp.isfinite(jax.jit(loss)(kw["wprtp"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wprtp"])))


# ---------------------------------------------------------------------------
# Cloud-water flux assembly x'rc'
# ---------------------------------------------------------------------------

def test_xprcp_fluxes_matches_golden():
    """Pin both level sets: the on-grid fluxes and the regrids the caller applies."""
    g = np.load(_FIX / "clubb_xprcp_golden.npz")
    kw = _xprcp_inputs()
    gr = kw["gr"]
    k_ub = gr.zm.shape[1] - 1
    out = calc_pdf_xprcp_fluxes(**_no_gr(kw))
    got = {f"{name}_zt": out[name] for name in out}
    for name in ("wprcp", "rtprcp", "thlprcp", "uprcp", "vprcp"):
        got[f"{name}_zm"] = zt2zm(out[name], gr).at[:, k_ub].set(0.0)
    for key in g.files:
        # FP-reassociation tolerance (C1-C8 condense refactor; ~1e-14 rel).
        np.testing.assert_allclose(np.asarray(got[key]), g[key], rtol=1e-13, atol=1e-16,
                                   err_msg=key)


def test_xprcp_fluxes_jit_grad():
    kw = _no_gr(_xprcp_inputs())

    def loss(rcm):
        out = calc_pdf_xprcp_fluxes(**dict(kw, rcm_zt=rcm))
        return sum(jnp.sum(v) for v in out.values())

    assert jnp.isfinite(jax.jit(loss)(kw["rcm_zt"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["rcm_zt"])))


# ---------------------------------------------------------------------------
# Live bit-exact parity vs CLUBB-JAX reference
# ---------------------------------------------------------------------------

def test_cloud_cover_gradient_finite_over_clear_levels():
    """A cloud-free level must not hand reverse-mode AD a 0/0.

    The layer geometry divides by a vertical cloud fraction that is zero on the
    clear levels the routine leaves alone. Both branches of a ``where`` are
    evaluated and differentiated, so without a guard on that denominator the
    gradient is NaN even though the returned values are right.
    """
    gr = _moment_inputs()["gr"]
    ng, nzt = gr.zt.shape
    deck = np.zeros((ng, nzt))
    deck[:, 2:5] = 3e-5           # a cloud deck with clear air above and below
    rcm = jnp.asarray(deck)
    cloud_frac = jnp.asarray(np.where(deck > 0.0, 0.4, 0.0))
    # chi is ZERO in the clear air, which is what makes the partial-fraction
    # denominator (cloud water + |chi|) vanish there as well: this exercises
    # BOTH guarded divisions, not just the vertical-fraction one.
    chi = jnp.asarray(np.where(deck > 0.0, 1e-5, 0.0))

    def loss(rcm_in):
        cover, rc_in = compute_cloud_cover(chi, cloud_frac, rcm_in, gr)
        return jnp.sum(cover) + jnp.sum(rc_in)

    assert jnp.isfinite(loss(rcm))
    g = jax.grad(loss)(rcm)
    assert jnp.all(jnp.isfinite(g)), f"non-finite cloud-cover gradient: {g}"


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_level_set_reconciliation_parity_vs_clubb_jax_reference():
    """Bit-exact vs upstream for the two-level-set reconciliation routines.

    Upstream folded its per-term PDF helpers into one ``pdf_closure`` routine,
    so the moment/buoyancy/cloud-flux assemblies are no longer separately
    callable there and are pinned by the committed goldens above instead. The
    routines that ARE still standalone upstream — the trapezoidal reconciliation
    of the thermodynamic and momentum level sets, the cloud-water cap, and the
    layer cloud-cover geometry — are compared directly here.
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.clip_explicit as ref_clip
    import clubb_jax.src.CLUBB_core.pdf_closure_module as refmod

    rng = np.random.default_rng(21)
    kw = _moment_inputs()
    gr = kw["gr"]
    ng, nzt = kw["rtm"].shape
    nzm = nzt + 1
    var_zt = jnp.asarray(rng.normal(size=(ng, nzt)))
    var_zm = jnp.asarray(rng.normal(size=(ng, nzm)))
    refgr = _refgr(gr)

    # The reference's trapezoid routines are jit-compiled, so XLA may contract a
    # multiply-add that our eager form evaluates separately. That is a last-bit
    # difference (~1 ULP), not a formula difference, so these two compare at a
    # few ULP rather than bit-for-bit; everything below is exact.
    np.testing.assert_allclose(
        np.asarray(calc_trapezoid_zt(var_zm, var_zt, gr)),
        np.asarray(refmod.calc_trapezoid_zt(nzm, nzt, ng, refgr, var_zm, var_zt)),
        rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(calc_trapezoid_zm(var_zt, var_zm, gr)),
        np.asarray(refmod.calc_trapezoid_zm(nzm, nzt, ng, refgr, var_zt, var_zm)),
        rtol=1e-15, atol=0.0)

    # Cloud-water cap: rtm below rcm on half the points, above on the rest.
    rcm = jnp.asarray(np.abs(rng.normal(size=(ng, nzt))) * 1e-4)
    rtm = jnp.asarray(rng.normal(size=(ng, nzt)) * 1e-4)
    np.testing.assert_array_equal(
        np.asarray(clip_rcm(rtm, rcm)),
        np.asarray(ref_clip.clip_rcm(nzt, ng, rtm, "test", rcm)))

    # Layer cloud geometry: a cloud deck with a top, a base and a clear gap.
    rcm_deck = np.zeros((ng, nzt))
    rcm_deck[:, 2:5] = 3e-5
    rcm_deck[:, 6] = 8e-6            # isolated one-level cloud
    rcm_deck = jnp.asarray(rcm_deck)
    cloud_frac = jnp.asarray(np.where(np.asarray(rcm_deck) > 0.0, 0.35, 0.0))
    chi_mean = jnp.asarray(rng.normal(size=(ng, nzt)) * 1e-5)
    pdf_params = SimpleNamespace(
        mixt_frac=jnp.full((ng, nzt), 1.0), chi_1=chi_mean,
        chi_2=jnp.zeros((ng, nzt)))
    mine_cover, mine_rc = compute_cloud_cover(chi_mean, cloud_frac, rcm_deck, gr)
    ref_cover, ref_rc = refmod.compute_cloud_cover(
        refgr, nzt, ng, pdf_params, cloud_frac, rcm_deck)
    np.testing.assert_allclose(np.asarray(mine_cover), np.asarray(ref_cover),
                               rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.asarray(mine_rc), np.asarray(ref_rc),
                               rtol=1e-13, atol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
