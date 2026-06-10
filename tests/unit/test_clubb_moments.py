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
)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
