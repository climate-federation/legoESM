"""Tests for the ADG1-PDF diagnostic closure used by the CLUBB scheme.

Confirms the double-Gaussian PDF (vs clubb_lite's single Gaussian) is exercised
in the live path: bounded cloud fraction, non-negative cloud water, a finite
moist buoyancy flux, and cloud forming in a saturated column but not a dry one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    diagnose_cloud_and_buoyancy,
)
from legoesm.atmosphere.physics.turbulence.clubb import make_clubb_grid  # noqa: E402

from legoesm import constants  # noqa: E402


def _inputs(rtm_scale=1.0, ng=2, nzt=12):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    gr = make_clubb_grid(zm, zt)
    p = jnp.asarray(np.tile(np.linspace(1.0e5, 7.0e4, nzt), (ng, 1)))
    exner = (p / constants.p_ref) ** constants.kappa
    thlm = jnp.full((ng, nzt), 290.0)
    rtm = jnp.asarray(rtm_scale * np.tile(np.linspace(1.5e-2, 3e-3, nzt), (ng, 1)))
    thv_ds = thlm * (1.0 + 0.61 * rtm)
    return dict(thlm=thlm, rtm=rtm, wp2=jnp.full((ng, nzt), 0.5), exner=exner,
                p_in_Pa=p, thv_ds=thv_ds, Kh=jnp.full((ng, nzt), 5.0),
                Lscale=jnp.full((ng, nzt), 100.0), gr=gr, config=CLUBBConfig())


def test_cloud_frac_bounds_rcm_nonneg_wpthvp_finite():
    cf, rcm, wpthvp = diagnose_cloud_and_buoyancy(**_inputs())
    assert jnp.all(cf >= 0.0) and jnp.all(cf <= 1.0)
    assert jnp.all(rcm >= 0.0)
    assert jnp.all(jnp.isfinite(wpthvp))


def test_saturated_column_forms_cloud():
    cf, rcm, _ = diagnose_cloud_and_buoyancy(**_inputs(rtm_scale=1.0))
    assert float(cf.max()) > 0.1
    assert float(rcm.max()) > 0.0


def test_dry_column_is_clear():
    cf, rcm, _ = diagnose_cloud_and_buoyancy(**_inputs(rtm_scale=0.02))
    assert float(cf.max()) < 0.05
    assert float(rcm.max()) < 1e-4


def test_cloud_buoyancy_toggle_removes_cloud_liquid_term_only():
    # D9 control: cloud_buoyancy=False drops the rc_coef*wprcp cloud-liquid term from wpthvp
    # (dry buoyancy) while leaving cloud_frac/rcm — the native PDF diagnostics — UNCHANGED.
    kw = _inputs(rtm_scale=1.0)                      # saturated ⇒ rcm>0 ⇒ the term is nonzero
    cf_on, rcm_on, wpthvp_on = diagnose_cloud_and_buoyancy(**kw)
    kw_off = dict(kw, config=CLUBBConfig(cloud_buoyancy=False))
    cf_off, rcm_off, wpthvp_off = diagnose_cloud_and_buoyancy(**kw_off)
    # diagnostics identical (the PDF cloud is still computed, only its buoyancy feedback is off)
    assert jnp.allclose(cf_on, cf_off) and jnp.allclose(rcm_on, rcm_off)
    # the buoyancy flux DIFFERS where cloud exists (the toggle has a real effect)
    assert float(jnp.max(jnp.abs(wpthvp_on - wpthvp_off))) > 0.0
    assert jnp.all(jnp.isfinite(wpthvp_off))
    # a CLEAR column (rcm≈0) is byte-unchanged by the toggle (no cloud-liquid term to drop)
    kw_dry = _inputs(rtm_scale=0.02)
    _, _, w_dry_on = diagnose_cloud_and_buoyancy(**kw_dry)
    _, _, w_dry_off = diagnose_cloud_and_buoyancy(**dict(kw_dry, config=CLUBBConfig(cloud_buoyancy=False)))
    assert jnp.allclose(w_dry_on, w_dry_off, atol=1e-10)


def test_cloud_source_shared_uses_gridscale_saturation_and_differs_from_native():
    # D9 literal forced-shared: cloud_source="shared" replaces the ADG1 PDF cloud with a
    # grid-scale all-or-nothing saturation adjustment (rcm=max(rt−r_sat,0)); it must DIFFER
    # from the native PDF cloud in a saturated column, stay nonneg + finite, and feed buoyancy.
    kw = _inputs(rtm_scale=1.0)
    cf_n, rcm_n, w_n = diagnose_cloud_and_buoyancy(**kw)
    kw_s = dict(kw, config=CLUBBConfig(cloud_source="shared"))
    cf_s, rcm_s, w_s = diagnose_cloud_and_buoyancy(**kw_s)
    assert jnp.all(rcm_s >= 0.0) and jnp.all(jnp.isfinite(w_s))
    assert jnp.all(cf_s >= 0.0) and jnp.all(cf_s <= 1.0)
    # PDF vs grid-scale differ where cloud exists (the whole point of the D9 contrast)
    assert float(jnp.max(jnp.abs(rcm_s - rcm_n))) > 0.0
    assert float(jnp.max(jnp.abs(w_s - w_n))) > 0.0


def test_cloud_source_unknown_raises():
    # nested scheme-Config dispatch hardening: an unknown cloud_source must raise, not
    # silently run a default cloud.
    with pytest.raises(ValueError):
        diagnose_cloud_and_buoyancy(**dict(_inputs(), config=CLUBBConfig(cloud_source="bogus")))


def test_jit_and_grad():
    kw = _inputs()

    def loss(rtm):
        cf, rcm, wpthvp = diagnose_cloud_and_buoyancy(**dict(kw, rtm=rtm))
        return jnp.sum(cf) + jnp.sum(rcm) + jnp.sum(wpthvp ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["rtm"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["rtm"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
