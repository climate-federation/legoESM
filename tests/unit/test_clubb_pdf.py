"""Unit tests for the CLUBB ADG1 PDF parameter closure (now in ``clubb.py``).

The ADG1 double-Gaussian must reproduce the input moments exactly — these are
machine-precision analytic oracles (mean, variance, covariance, skewness). Plus
derived-parameter values and a live bit-exact parity vs the CLUBB-JAX reference.

Part of the fuller CLUBB port — see ``docs/dev-notes/clubb.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    derive_lmin,
    derive_mixt_frac_max_mag,
)
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    ADG1_pdf_driver,
    ADG1_w_closure,
    calc_liquid_cloud_frac_component,
    calc_pdf_liquid_cloud_frac,
    transform_pdf_chi_eta_component,
)

from legoesm import constants  # noqa: E402

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_BETA, _MFMM = 2.4, derive_mixt_frac_max_mag(4.5)
_CF_GOLDEN_NPZ = Path(__file__).resolve().parent / "clubb_fixtures" / "clubb_cloudfrac_golden.npz"


def _cloud_inputs(rtm_scale=1.0):
    """ADG1 PDF + thermo fixture spanning clear..cloudy (varying rtm vs rsat)."""
    rng = np.random.default_rng(7)
    ng, nzt = 2, 8
    shp = (ng, nzt)
    wp2 = jnp.asarray(0.2 + 0.6 * rng.random(shp))
    sqrt_wp2 = jnp.sqrt(wp2)
    rtp2 = jnp.asarray(1e-7 + 2e-6 * rng.random(shp))
    thlp2 = jnp.asarray(0.05 + 0.3 * rng.random(shp))
    up2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    vp2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    ssw = jnp.asarray(0.2 + 0.5 * rng.random(shp))
    Skw = jnp.asarray(-1.0 + 2.0 * rng.random(shp))
    # rtm spans dry (top) to near/above saturation (bottom) to exercise cf in [0,1].
    rtm = jnp.asarray(rtm_scale * np.linspace(2e-3, 1.6e-2, nzt)[None, :] * np.ones(shp))
    thlm = jnp.asarray(290.0 + 4.0 * rng.random(shp))
    um = jnp.asarray(rng.normal(size=shp))
    vm = jnp.asarray(rng.normal(size=shp))

    def cov(xp2, f):
        return jnp.asarray(f) * jnp.sqrt(wp2 * xp2)

    adg1 = ADG1_pdf_driver(jnp.asarray(rng.normal(size=shp)), rtm, thlm, um, vm,
                           wp2, rtp2, thlp2, up2, vp2, Skw,
                           cov(rtp2, 0.3), cov(thlp2, -0.4), cov(up2, 0.2), cov(vp2, -0.25),
                           sqrt_wp2, ssw, _BETA, _MFMM)
    p = jnp.asarray(np.linspace(9.5e4, 6.0e4, nzt)[None, :] * np.ones(shp))
    exner = (p / constants.p_ref) ** constants.kappa
    rtpthlp = -0.2 * jnp.sqrt(rtp2 * thlp2)
    return dict(adg1=adg1, rtpthlp=rtpthlp, rtm=rtm, thlm=thlm, exner=exner, p_in_Pa=p)


def _inputs():
    """Deterministic 2x6 fixture with skewness, moisture, and shear covariances."""
    rng = np.random.default_rng(0)
    shp = (2, 6)
    wp2 = jnp.asarray(0.2 + 0.6 * rng.random(shp))
    sqrt_wp2 = jnp.sqrt(wp2)
    rtp2 = jnp.asarray(1e-7 + 1e-6 * rng.random(shp))
    thlp2 = jnp.asarray(0.05 + 0.3 * rng.random(shp))
    up2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    vp2 = jnp.asarray(0.1 + 0.5 * rng.random(shp))
    sigma_sqd_w = jnp.asarray(0.2 + 0.5 * rng.random(shp))  # in (0,1)
    Skw = jnp.asarray(-1.0 + 2.0 * rng.random(shp))         # both signs
    # Covariances with |corr| < 1 (well inside the physical bound).
    def _cov(xp2, frac):
        return jnp.asarray(frac) * jnp.sqrt(wp2 * xp2)
    return dict(
        wm=jnp.asarray(rng.normal(size=shp)),
        rtm=jnp.asarray(8e-3 + 2e-3 * rng.random(shp)),
        thlm=jnp.asarray(295.0 + 10.0 * rng.random(shp)),
        um=jnp.asarray(rng.normal(size=shp)), vm=jnp.asarray(rng.normal(size=shp)),
        wp2=wp2, rtp2=rtp2, thlp2=thlp2, up2=up2, vp2=vp2, Skw=Skw,
        wprtp=_cov(rtp2, 0.3), wpthlp=_cov(thlp2, -0.4),
        upwp=_cov(up2, 0.2), vpwp=_cov(vp2, -0.25),
        sqrt_wp2=sqrt_wp2, sigma_sqd_w=sigma_sqd_w,
        beta=_BETA, mixt_frac_max_mag=_MFMM,
    )


# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------

def test_derive_mixt_frac_max_mag_golden():
    assert derive_mixt_frac_max_mag(4.5) == pytest.approx(0.989665, rel=1e-5)


def test_derive_lmin():
    assert derive_lmin(0.1) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# ADG1 moment-recovery identities (exact by construction)
# ---------------------------------------------------------------------------

def test_w_mean_variance_skewness_recovery():
    kw = _inputs()
    wm, wp2, Skw, ssw, sqrt_wp2 = (kw["wm"], kw["wp2"], kw["Skw"],
                                   kw["sigma_sqd_w"], kw["sqrt_wp2"])
    w_1, w_2, _, _, vw1, vw2, mf = ADG1_w_closure(wm, wp2, Skw, ssw, sqrt_wp2, _MFMM)
    omf = 1.0 - mf
    # Mean.
    mean = mf * w_1 + omf * w_2
    np.testing.assert_allclose(np.asarray(mean), np.asarray(wm), rtol=1e-11, atol=1e-11)
    # Variance: sum_i mf_i*(var_i + (mu_i - wm)^2).
    var = mf * (vw1 + (w_1 - wm) ** 2) + omf * (vw2 + (w_2 - wm) ** 2)
    np.testing.assert_allclose(np.asarray(var), np.asarray(wp2), rtol=1e-10)
    # Skewness: third central moment / wp2^1.5.
    m3 = (mf * ((w_1 - wm) ** 3 + 3.0 * (w_1 - wm) * vw1)
          + omf * ((w_2 - wm) ** 3 + 3.0 * (w_2 - wm) * vw2))
    np.testing.assert_allclose(np.asarray(m3 / wp2 ** 1.5), np.asarray(Skw), rtol=1e-9)


def test_w_components_straddle_mean():
    kw = _inputs()
    w_1, w_2, _, _, _, _, _ = ADG1_w_closure(
        kw["wm"], kw["wp2"], kw["Skw"], kw["sigma_sqd_w"], kw["sqrt_wp2"], _MFMM)
    assert jnp.all(w_1 >= kw["wm"]) and jnp.all(w_2 <= kw["wm"])


def test_responder_mean_and_covariance_recovery():
    kw = _inputs()
    out = ADG1_pdf_driver(**kw)
    mf, omf = out["mixt_frac"], 1.0 - out["mixt_frac"]
    w_1, w_2, wm = out["w_1"], out["w_2"], kw["wm"]
    for var, mean, cov in [("rt", "rtm", "wprtp"), ("thl", "thlm", "wpthlp"),
                           ("u", "um", "upwp"), ("v", "vm", "vpwp")]:
        x_1, x_2 = out[f"{var}_1"], out[f"{var}_2"]
        xm = kw[mean]
        # Responder mean recovery.
        np.testing.assert_allclose(
            np.asarray(mf * x_1 + omf * x_2), np.asarray(xm), rtol=1e-10, atol=1e-10,
            err_msg=f"{var} mean")
        # Covariance w'x' recovery.
        recov = mf * (w_1 - wm) * (x_1 - xm) + omf * (w_2 - wm) * (x_2 - xm)
        np.testing.assert_allclose(
            np.asarray(recov), np.asarray(kw[cov]), rtol=1e-9, atol=1e-12,
            err_msg=f"{var} covariance")
        # Variance recovery (locks the varnce_x split + alpha_x + width_factor).
        vx_1, vx_2 = out[f"varnce_{var}_1"], out[f"varnce_{var}_2"]
        var_tot = mf * (vx_1 + (x_1 - xm) ** 2) + omf * (vx_2 + (x_2 - xm) ** 2)
        np.testing.assert_allclose(
            np.asarray(var_tot), np.asarray(kw[f"{var}p2"]),
            rtol=1e-9, err_msg=f"{var} variance")


def test_mixt_frac_zero_skew_and_bounds():
    kw = _inputs()
    _, _, _, _, _, _, mf0 = ADG1_w_closure(
        kw["wm"], kw["wp2"], jnp.zeros_like(kw["Skw"]), kw["sigma_sqd_w"],
        kw["sqrt_wp2"], _MFMM)
    np.testing.assert_allclose(np.asarray(mf0), 0.5, rtol=1e-12)
    out = ADG1_pdf_driver(**kw)
    assert jnp.all(out["mixt_frac"] >= 1.0 - _MFMM - 1e-12)
    assert jnp.all(out["mixt_frac"] <= _MFMM + 1e-12)


def test_pdf_jit_and_grad_clean():
    kw = _inputs()

    def loss(wp2):
        out = ADG1_pdf_driver(**dict(kw, wp2=wp2, sqrt_wp2=jnp.sqrt(wp2)))
        return sum(jnp.sum(v) for v in out.values())

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    g = jax.grad(loss)(kw["wp2"])
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Liquid cloud fraction
# ---------------------------------------------------------------------------

def test_cloud_frac_bounds_and_rcm_nonneg():
    kw = _cloud_inputs()
    rcm, cf = calc_pdf_liquid_cloud_frac(**kw)
    assert jnp.all(cf >= 0.0) and jnp.all(cf <= 1.0)
    assert jnp.all(rcm >= 0.0)


def test_cloud_frac_dry_limit_clear():
    rcm, cf = calc_pdf_liquid_cloud_frac(**_cloud_inputs(rtm_scale=0.01))  # rt << rsat
    assert float(jnp.mean(cf)) < 0.05      # nearly clear everywhere
    assert float(jnp.max(rcm)) < 1e-4


def test_cloud_frac_moist_limit_overcast():
    rcm, cf = calc_pdf_liquid_cloud_frac(**_cloud_inputs(rtm_scale=30.0))  # rt >> rsat
    np.testing.assert_allclose(np.asarray(cf), 1.0, atol=1e-9)
    assert jnp.all(rcm > 0.0)


def test_cloud_frac_monotonic_in_moisture():
    cfs = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        _, cf = calc_pdf_liquid_cloud_frac(**_cloud_inputs(rtm_scale=scale))
        cfs.append(float(jnp.mean(cf)))
    assert cfs[0] <= cfs[1] <= cfs[2] <= cfs[3]


def test_cloud_frac_matches_committed_golden():
    """Bit-exact vs the committed golden (CI, no reference checkout needed)."""
    golden = np.load(_CF_GOLDEN_NPZ)
    rcm, cf = calc_pdf_liquid_cloud_frac(**_cloud_inputs())
    # FP-reassociation tolerance: the C1-C8 "condense" refactor fused/re-
    # associated the algebra (~1e-14 rel), so round-off (not bit-identity) is
    # the right invariant.
    np.testing.assert_allclose(np.asarray(rcm), golden["rcm"], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.asarray(cf), golden["cloud_frac"], rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clear_cell_gradient_finite_both_dtypes(dtype):
    """Zero-variance clear cell (mean_chi=0, stdev_chi=0) -> finite gradients.

    Guards the dtype-aware denominator floor: the reference 1e-100 underflows in
    float32, which would make the inactive Gaussian branch divide by 0 and NaN
    the reverse-mode gradient.
    """
    def cf_loss(args):
        mean_chi, stdev_chi = args
        cf, rc = calc_liquid_cloud_frac_component(mean_chi, stdev_chi)
        return cf + rc

    # Zero-variance clear cell, slightly-cloudy tiny-variance cell, and LARGE
    # |mean| clear/full cells (stdev=0) — the divide-VJP must stay finite for
    # any mean magnitude, not just small ones.
    for mean_chi in (0.0, 1e-3, -1e-3, 10.0, -10.0):
        args = (jnp.asarray(mean_chi, dtype), jnp.asarray(0.0, dtype))
        g_mean, g_std = jax.grad(cf_loss)(args)
        assert jnp.isfinite(g_mean) and jnp.isfinite(g_std), (dtype, mean_chi)
    # Exactly on the partial<->clear/full cutoff mean_chi = +-5*stdev_chi
    # (active partial branch via strict comparisons): gradient must be finite.
    s = jnp.asarray(0.1, dtype)
    for sign in (1.0, -1.0):
        g = jax.grad(cf_loss)((sign * 5.0 * s, s))
        assert all(jnp.isfinite(x) for x in g), (dtype, sign)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_component_variance_gradient_finite(dtype):
    """alpha_x->0 makes component variances exactly 0; the chi/eta transform's
    sqrt(varnce_rt*varnce_thl) must keep finite reverse-mode gradients."""
    def loss(v):
        vrt, vthl = v
        out = transform_pdf_chi_eta_component(
            jnp.asarray(290.0, dtype), jnp.asarray(0.012, dtype),
            jnp.asarray(0.011, dtype), jnp.asarray(0.9, dtype),
            vrt, vthl, jnp.asarray(0.5, dtype))
        return sum(jnp.sum(x) for x in out)

    g = jax.grad(loss)((jnp.asarray(0.0, dtype), jnp.asarray(0.0, dtype)))
    assert all(jnp.isfinite(x) for x in g), dtype


def test_cloud_frac_jit_and_grad():
    kw = _cloud_inputs()

    def loss(rtm):
        rcm, cf = calc_pdf_liquid_cloud_frac(**dict(kw, rtm=rtm))
        return jnp.sum(cf) + jnp.sum(rcm)

    assert jnp.isfinite(jax.jit(loss)(kw["rtm"]))
    g = jax.grad(loss)(kw["rtm"])
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.skipif(
    not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
    reason="CLUBB-JAX reference tree not present",
)
def test_cloud_frac_parity_vs_clubb_jax_reference():
    """Bit-exact cloud_frac/rcm vs CLUBB-JAX (constants+saturation patched)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.pdf_closure_module as refmod
    from legoesm.atmosphere.physics.turbulence.clubb import (
        sat_mixrat_liq as my_sat,
    )

    refmod.ep = constants.epsilon
    refmod.Lv = constants.L_v
    refmod.Rd = constants.R_d
    refmod.Cp = constants.c_pd
    refmod.sat_mixrat_liq = lambda p, t, _sf: my_sat(p, t)

    kw = _cloud_inputs()
    rcm_m, cf_m = calc_pdf_liquid_cloud_frac(**kw)
    rcm_r, cf_r = refmod.calc_pdf_liquid_cloud_frac_jax(
        adg1=kw["adg1"], rtpthlp_zt=kw["rtpthlp"], rtm=kw["rtm"], thlm=kw["thlm"],
        exner=kw["exner"], p_in_Pa=kw["p_in_Pa"], saturation_formula=3)
    np.testing.assert_array_equal(np.asarray(cf_m), np.asarray(cf_r))
    np.testing.assert_array_equal(np.asarray(rcm_m), np.asarray(rcm_r))
    golden = np.load(_CF_GOLDEN_NPZ)
    np.testing.assert_array_equal(np.asarray(cf_r), golden["cloud_frac"])
    np.testing.assert_array_equal(np.asarray(rcm_r), golden["rcm"])


@pytest.mark.skipif(
    not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
    reason="CLUBB-JAX reference tree not present",
)
def test_parity_vs_clubb_jax_reference():
    """Bit-exact parity vs CLUBB-JAX ADG1_pdf_driver (no constants -> exact)."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.CLUBB_core.adg1_adg2_3d_luhar_pdf import (
        ADG1_pdf_driver as ref_driver,
    )

    kw = _inputs()
    mine = ADG1_pdf_driver(**kw)
    ref = ref_driver(
        kw["wm"], kw["rtm"], kw["thlm"], kw["um"], kw["vm"],
        kw["wp2"], kw["rtp2"], kw["thlp2"], kw["up2"], kw["vp2"],
        kw["Skw"], kw["wprtp"], kw["wpthlp"], kw["upwp"], kw["vpwp"],
        kw["sqrt_wp2"], kw["sigma_sqd_w"], _BETA, _MFMM)
    for key in ref:
        np.testing.assert_array_equal(
            np.asarray(mine[key]), np.asarray(ref[key]),
            err_msg=f"{key} diverged from reference")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
