"""Tests for the CLUBB dissipation time-scale family (``clubb_tau.py``).

Bit-exact parity vs CLUBB-JAX ``calc_stability_correction`` and analytic checks
of the CAM-default tau family (the formulas are the spec), plus jit/grad.
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
from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid, zt2zm  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb_tau as T  # noqa: E402, N812

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"
_ILAMBDA0 = 66   # parameter_indices.ilambda0_stability_coef (1-based)


def _gr(ng=2, nzt=10):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def _inputs(ng, nzm, seed=0):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    return dict(
        Lscale=jnp.asarray(50.0 + 200.0 * rng.random((ng, nzt))),
        em=jnp.asarray(0.05 + 0.5 * rng.random((ng, nzm))),
        sqrt_em_zt=jnp.asarray(0.3 + 0.5 * rng.random((ng, nzt))),
        brunt=jnp.asarray(1e-4 * rng.standard_normal((ng, nzm))),
    )


def test_stability_correction_bounds():
    gr, ng, nzm = _gr()
    p = _inputs(ng, nzm)
    Lscale_zm = jnp.maximum(zt2zm(p["Lscale"], gr), 0.0)
    sc = np.asarray(T.calc_stability_correction(p["brunt"], Lscale_zm, p["em"], 0.04))
    assert np.all(sc >= 1.0 - 1e-12)        # >= 1 (unstable -> exactly 1)
    assert np.all(sc <= 1.0 + 3.0 + 1e-12)  # capped at 1 + 3
    # unstable layers (N2 <= 0) -> exactly 1
    unstable = np.asarray(p["brunt"]) <= 0.0
    assert np.allclose(sc[unstable], 1.0)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_stability_correction_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.advance_helper_module as R  # noqa: N812
    gr, ng, nzm = _gr()
    p = _inputs(ng, nzm, seed=3)
    Lscale_zm = jnp.maximum(zt2zm(p["Lscale"], gr), 0.0)
    lam = 0.04
    cp = np.zeros((ng, 102))
    cp[:, _ILAMBDA0 - 1] = lam
    np.testing.assert_array_equal(
        np.asarray(T.calc_stability_correction(p["brunt"], Lscale_zm, p["em"], lam)),
        np.asarray(R.calc_stability_correction(p["brunt"], Lscale_zm, p["em"], jnp.asarray(cp))))


def test_tau_family_formulas():
    gr, ng, nzm = _gr()
    p = _inputs(ng, nzm, seed=5)
    cfg = CLUBBConfig()
    out = T.compute_tau_family(p["Lscale"], p["em"], p["sqrt_em_zt"], p["brunt"], gr, cfg)
    taumax = cfg.params.taumax
    em_min = 1.5 * cfg.w_tol ** 2
    tau_zt = np.minimum(np.asarray(p["Lscale"]) / np.asarray(p["sqrt_em_zt"]), taumax)
    Lscale_zm = np.maximum(np.asarray(zt2zm(p["Lscale"], gr)), 0.0)
    tau_zm = np.minimum(Lscale_zm / np.sqrt(np.maximum(em_min, np.asarray(p["em"]))), taumax)
    np.testing.assert_allclose(np.asarray(out["invrs_tau_zt"]), 1.0 / tau_zt, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out["invrs_tau_zm"]), 1.0 / tau_zm, rtol=1e-12)
    # C-family aliasing
    np.testing.assert_array_equal(np.asarray(out["invrs_tau_C4_zm"]), np.asarray(out["invrs_tau_zm"]))
    np.testing.assert_array_equal(np.asarray(out["invrs_tau_C14_zm"]), np.asarray(out["invrs_tau_zm"]))
    np.testing.assert_array_equal(np.asarray(out["invrs_tau_xp2_zm"]), np.asarray(out["invrs_tau_zm"]))
    np.testing.assert_array_equal(np.asarray(out["invrs_tau_wp3_zt"]), np.asarray(out["invrs_tau_zt"]))
    # C1/C6 = invrs_tau_zm * stability_correction
    exp = np.asarray(out["invrs_tau_zm"]) * np.asarray(out["stability_correction"])
    np.testing.assert_allclose(np.asarray(out["invrs_tau_C1_zm"]), exp, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(out["invrs_tau_C6_zm"]), np.asarray(out["invrs_tau_C1_zm"]))


def test_jit_and_grad():
    gr, ng, nzm = _gr()
    p = _inputs(ng, nzm)
    cfg = CLUBBConfig()

    def loss(Lscale):
        out = T.compute_tau_family(Lscale, p["em"], p["sqrt_em_zt"], p["brunt"], gr, cfg)
        return sum(jnp.sum(out[k] ** 2) for k in ("invrs_tau_C1_zm", "invrs_tau_wp3_zt"))

    assert jnp.isfinite(jax.jit(loss)(p["Lscale"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(p["Lscale"])))


def test_stability_correction_grad_finite_at_low_em():
    """The em-floor must keep the C1/C6 gradient finite even at em -> 0 (the
    1/em stability-correction division) — self-audit AD hardening."""
    gr, ng, nzm = _gr()
    p = _inputs(ng, nzm)
    cfg = CLUBBConfig()
    # em with zeros and values below em_min, including positive N2 (active corr)
    em = jnp.zeros((ng, nzm)).at[:, : nzm // 2].set(1.0e-8)
    brunt = jnp.full((ng, nzm), 1.0e-4)   # all positive -> stability corr active

    def loss(em):
        out = T.compute_tau_family(p["Lscale"], em, p["sqrt_em_zt"], brunt, gr, cfg)
        return jnp.sum(out["invrs_tau_C1_zm"] ** 2)

    g = jax.grad(loss)(em)
    assert jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
