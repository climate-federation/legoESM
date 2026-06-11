"""Tests for the CLUBB core diagnostics bundle (``clubb_core.py``).

The constituents (Skw/sigma_sqd_w/em/tau/C6-C7) are each independently
parity-tested; this validates the thin orchestration: the right keys, finite
outputs, correct shapes, and jit/grad cleanliness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_core import compute_clubb_diagnostics  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid  # noqa: E402


def _gr(ng=2, nzt=12):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def _inputs(gr, ng, nzm, seed=0):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def zm(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzm)))

    return dict(
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        wp3=jnp.asarray(0.1 * rng.standard_normal((ng, nzt))),
        up2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        vp2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        thlp2=jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm))),
        rtp2=jnp.asarray(1e-6 + 1e-6 * rng.random((ng, nzm))),
        wpthlp=zm(1e-2), wprtp=zm(1e-4),
        Lscale=jnp.asarray(50.0 + 200.0 * rng.random((ng, nzt))),
        brunt_vaisala_freq_sqd=zm(1e-4),
        gr=gr, config=CLUBBConfig(),
    )


def test_diagnostics_keys_and_shapes():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    out = compute_clubb_diagnostics(**_inputs(gr, ng, nzm))
    expected = {"Skw_zm", "Skw_zt", "wp2_zt", "wp3_zm", "wp3_on_wp2",
                "wp3_on_wp2_zt", "gamma_Skw", "sigma_sqd_w", "em", "sqrt_em_zt",
                "Lscale_zm", "C6rt_Skw_fnc", "C6thl_Skw_fnc", "C7_Skw_fnc",
                "invrs_tau_C1_zm", "invrs_tau_C4_zm", "invrs_tau_C6_zm",
                "invrs_tau_C14_zm", "invrs_tau_xp2_zm", "invrs_tau_wp3_zt"}
    assert expected <= set(out)
    for k, v in out.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    # zm-level fields are (ng, nzm); zt-level are (ng, nzt)
    assert out["Skw_zm"].shape == (ng, nzm) and out["Skw_zt"].shape == (ng, nzt)
    assert out["sigma_sqd_w"].shape == (ng, nzm)
    assert out["invrs_tau_wp3_zt"].shape == (ng, nzt)
    # sigma_sqd_w in (0, 1)
    s = np.asarray(out["sigma_sqd_w"])
    assert np.all((s >= 0.0) & (s < 1.0))


def test_diagnostics_jit_and_grad():
    gr, ng, nzm = _gr()
    kw = _inputs(gr, ng, nzm)

    def loss(wp2):
        out = compute_clubb_diagnostics(**dict(kw, wp2=wp2))
        return (jnp.sum(out["sigma_sqd_w"] ** 2) + jnp.sum(out["invrs_tau_C6_zm"] ** 2)
                + jnp.sum(out["C6rt_Skw_fnc"] ** 2))

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wp2"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
