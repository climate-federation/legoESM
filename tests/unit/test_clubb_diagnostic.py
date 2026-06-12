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

from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
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


def test_jit_and_grad():
    kw = _inputs()

    def loss(rtm):
        cf, rcm, wpthvp = diagnose_cloud_and_buoyancy(**dict(kw, rtm=rtm))
        return jnp.sum(cf) + jnp.sum(rcm) + jnp.sum(wpthvp ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["rtm"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["rtm"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
