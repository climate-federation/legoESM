"""Tests for the CLUBB core diagnostics bundle (``clubb_core.py``).

The constituents (Skw/sigma_sqd_w/em/tau/C6-C7) are each independently
parity-tested; this validates the thin orchestration: the right keys, finite
outputs, correct shapes, and jit/grad cleanliness.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
                "Lscale_zm", "Kh_zt", "Kh_zm", "C6rt_Skw_fnc", "C6thl_Skw_fnc",
                "C7_Skw_fnc", "invrs_tau_C1_zm", "invrs_tau_C4_zm",
                "invrs_tau_C6_zm", "invrs_tau_C14_zm", "invrs_tau_xp2_zm",
                "invrs_tau_wp3_zt"}
    assert expected <= set(out)
    for k, v in out.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    # zm-level fields are (ng, nzm); zt-level are (ng, nzt)
    assert out["Skw_zm"].shape == (ng, nzm) and out["Skw_zt"].shape == (ng, nzt)
    assert out["sigma_sqd_w"].shape == (ng, nzm)
    assert out["invrs_tau_wp3_zt"].shape == (ng, nzt)
    assert out["Kh_zt"].shape == (ng, nzt) and out["Kh_zm"].shape == (ng, nzm)
    # sigma_sqd_w in (0, 1); Kh >= 0
    s = np.asarray(out["sigma_sqd_w"])
    assert np.all((s >= 0.0) & (s < 1.0))
    assert np.all(np.asarray(out["Kh_zt"]) >= 0.0) and np.all(np.asarray(out["Kh_zm"]) >= 0.0)


_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_sigma_sqd_w_cam_form_matches_reference():
    """Audit (iter 45): the CAM rt/thl-only ``compute_sigma_sqd_w`` used inside
    ``compute_clubb_diagnostics`` is bit-exact to the full CLUBB-JAX reference
    invoked with ``l_predict_upwp_vpwp=False`` (the CAM default), proving the
    omitted up2/vp2/upwp/vpwp correlation terms are correctly absent."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.sigma_sqd_w_module as R  # noqa: N812
    from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.clubb_helpers import compute_sigma_sqd_w

    gr, ng, nzm = _gr()
    cfg = CLUBBConfig()
    rng = np.random.default_rng(45)
    gamma = jnp.asarray(0.2 + 0.2 * rng.random((ng, nzm)))
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    thlp2 = jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm)))
    rtp2 = jnp.asarray(1e-6 + 1e-6 * rng.random((ng, nzm)))
    wpthlp = jnp.asarray(1e-2 * rng.standard_normal((ng, nzm)))
    wprtp = jnp.asarray(1e-4 * rng.standard_normal((ng, nzm)))
    # Dummy momentum-flux/variance fields the reference ignores when the flag is off.
    dummy = jnp.asarray(rng.standard_normal((ng, nzm)))

    mine = compute_sigma_sqd_w(
        gamma, wp2, thlp2, rtp2, wpthlp, wprtp, gr,
        w_tol=cfg.w_tol, thl_tol=cfg.thl_tol, rt_tol=cfg.rt_tol)
    ref = R.compute_sigma_sqd_w(
        gamma, wp2, thlp2, rtp2, dummy, dummy, wpthlp, wprtp, dummy, dummy,
        False, gr)
    np.testing.assert_array_equal(np.asarray(mine), np.asarray(ref))


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
