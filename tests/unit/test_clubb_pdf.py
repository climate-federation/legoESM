"""Unit tests for the CLUBB ADG1 PDF parameter closure (``clubb_pdf.py``).

The ADG1 double-Gaussian must reproduce the input moments exactly — these are
machine-precision analytic oracles (mean, variance, covariance, skewness). Plus
derived-parameter values and a live bit-exact parity vs the CLUBB-JAX reference.

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_config import (  # noqa: E402
    derive_lmin,
    derive_mixt_frac_max_mag,
)
from legoesm.atmosphere.physics.turbulence.clubb_pdf import (  # noqa: E402
    ADG1_pdf_driver,
    ADG1_w_closure,
)

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_BETA, _MFMM = 2.4, derive_mixt_frac_max_mag(4.5)


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
