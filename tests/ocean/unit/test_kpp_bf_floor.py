"""KPPConfig.actual_buoyancy_flux: the KPP velocity scales use the actual surface buoyancy flux.

The legacy path replaces any |B_f| below float32 eps (1.19e-7 m2/s3) by that floor, so weak forcing, the common
tropical case, is not seen as it is. With actual_buoyancy_flux = True, zeta = d_eff kappa B_f / u*^3 is formed
without a division by B_f, and w_s at sigma = epsilon of a candidate depth d (core-0022 "candidate" form) equals
CVMix's cvmix_kpp_compute_turbulent_scales_1d_OBL (l_LMD_ws = .false.) for weak fluxes too. CVMix's convective
constants come from zeta_s (a_s = -7 sqrt(17), c_s = 24 sqrt(17)); legoESM keeps LMD94's rounding, so that regime
is compared at 1e-4. Hand values from the review of core-0022: u* = 0.01 m/s, d = 100 m: B_f = -1e-8
(stable in legoESM's sign) gives w_s = 0.003333..., B_f = +1e-8 gives 0.0051225. The gradient with respect to B_f
must be finite in float32 down to B_f = 0 (reviewer finding: a division by B_f underflows below about 1e-19).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import kpp as kpp_mod
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig

KAPPA, EPS_SL, ZETA_S = 0.4, 0.1, -1.0
A_S, C_S = -7.0 * np.sqrt(17.0), 24.0 * np.sqrt(17.0)
LEGACY = float(np.finfo(np.float32).eps)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def cvmix_ws(ustar, B_cvmix, d):
    zeta = EPS_SL * d * B_cvmix * KAPPA / ustar ** 3
    if zeta >= 0.0:
        return KAPPA * ustar / (1.0 + 5.0 * zeta), False
    if zeta >= ZETA_S:
        return KAPPA * ustar / (1.0 - 16.0 * zeta) ** -0.5, False
    return KAPPA * ustar / (A_S - C_S * zeta) ** (-1.0 / 3.0), True


def lego_ws(cfg, ustar, Bf, d, dtype=None):
    a = (lambda x: jnp.asarray(x, dtype=dtype)) if dtype is not None else jnp.asarray
    dd = a([d])
    _, ws = kpp_mod._kpp_velocity_scales(a(ustar), a(Bf), dd, dd, cfg, kpp_mod._EPS)
    return float(np.ravel(ws)[0])


@pytest.mark.parametrize("ustar,Bf,d", [(0.01, -1e-8, 100.0), (0.01, 1e-8, 100.0), (0.005, 5e-8, 30.0),
                                        (0.005, -5e-8, 60.0), (0.02, 1e-7, 200.0), (0.003, 9e-8, 80.0),
                                        (0.01, -1e-7, 10.0), (0.003, 3e-7, 400.0)])
def test_actual_flux_equals_cvmix(ustar, Bf, d):
    cfg = KPPConfig(vt2_velocity_scale="candidate", actual_buoyancy_flux=True)
    ref, conv = cvmix_ws(ustar, -Bf, d)          # CVMix surf_buoy_force > 0 is stabilising
    np.testing.assert_allclose(lego_ws(cfg, ustar, Bf, d), ref, rtol=1e-4 if conv else 1e-12)


def test_independent_hand_values():
    cfg = KPPConfig(actual_buoyancy_flux=True)
    assert abs(lego_ws(cfg, 0.01, -1e-8, 100.0) - 0.0033333) < 1e-6
    assert abs(lego_ws(cfg, 0.01, 1e-8, 100.0) - 0.0051225) < 1e-6


def test_default_is_the_legacy_floor():
    assert KPPConfig().actual_buoyancy_flux is False
    cfg = KPPConfig()
    for s in (1.0, -1.0):
        np.testing.assert_allclose(lego_ws(cfg, 0.01, s * 1e-8, 100.0), lego_ws(cfg, 0.01, s * LEGACY, 100.0),
                                   rtol=0, atol=0)


def test_zero_flux_is_neutral():
    cfg = KPPConfig(actual_buoyancy_flux=True)
    np.testing.assert_allclose(lego_ws(cfg, 0.01, 0.0, 50.0), KAPPA * 0.01, rtol=1e-12)
    np.testing.assert_allclose(lego_ws(cfg, 0.01, -0.0, 50.0), KAPPA * 0.01, rtol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradients_finite_near_zero_flux(dtype):
    cfg = KPPConfig(actual_buoyancy_flux=True)
    for us in (1e-4, 5.8e-4, 1e-2):
        g = jax.grad(lambda b, us=us: kpp_mod._kpp_velocity_scales(
            jnp.asarray(us, dtype=dtype), b, jnp.asarray([50.0], dtype=dtype), jnp.asarray([50.0], dtype=dtype),
            cfg, kpp_mod._EPS)[1][0])
        for b in (0.0, -0.0, 1e-28, -1e-25, 1e-22, -1e-20, 1e-19, 1e-9, -1e-9, 2e-7):
            val = g(jnp.asarray(b, dtype=dtype))
            assert np.isfinite(float(val)), (us, b, dtype)
