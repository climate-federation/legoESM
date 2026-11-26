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
from jax.test_util import check_grads

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


def test_default_is_the_actual_flux():
    assert KPPConfig().actual_buoyancy_flux is True
    np.testing.assert_allclose(lego_ws(KPPConfig(), 0.01, 1e-8, 100.0), 0.0051225, atol=1e-6)


def test_legacy_floor_when_switched_off():
    cfg = KPPConfig(actual_buoyancy_flux=False)
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


# CVMix (MOM6 0e9a561 submodule, CVMix fce422195a) cvmix_kpp_compute_turbulent_scales, banked from the compiled
# library (ported from PR #2560): (u*, B_f [this module's sign, >0 destabilising], h, sigma, w_m, w_s).  Stable
# and weakly-unstable branches only: they use no rounded LMD94 a_*/c_* constants, so they pin to round-off.
_CVMIX_WEAK_FLUX = [
    (0.001, -1e-07, 50.0, 0.02, 1.990049751243781e-06, 1.990049751243781e-06),
    (0.001, -1e-07, 50.0, 0.5, 3.9960039960039974e-07, 3.9960039960039974e-07),
    (0.001, -1e-08, 50.0, 0.02, 1.904761904761905e-05, 1.904761904761905e-05),
    (0.001, -1e-08, 50.0, 0.5, 3.960396039603961e-06, 3.960396039603961e-06),
    (0.001, -1e-09, 50.0, 0.02, 0.00013333333333333334, 0.00013333333333333334),
    (0.001, -1e-09, 50.0, 0.5, 3.636363636363637e-05, 3.636363636363637e-05),
    (0.001, -1e-10, 50.0, 0.02, 0.0003333333333333334, 0.0003333333333333334),
    (0.001, -1e-10, 50.0, 0.5, 0.0002, 0.0002),
    (0.001, 1e-10, 50.0, 0.02, 0.0004526587849560099, 0.0005122499389946279),
    (0.01, -1e-07, 50.0, 0.02, 0.003333333333333334, 0.003333333333333334),
    (0.01, -1e-08, 50.0, 0.5, 0.0036363636363636364, 0.0036363636363636364),
    (0.01, -1e-10, 50.0, 0.5, 0.003996003996003997, 0.003996003996003997),
    (0.01, 1e-10, 50.0, 0.5, 0.004003196167152268, 0.004006394888175653),
    (0.01, 1e-09, 50.0, 0.02, 0.004006384697092892, 0.004012779585275026),
    (0.01, 1e-08, 50.0, 0.5, 0.004287493494913047, 0.004595650117230423),
]


@pytest.mark.parametrize("ustar,Bf,h,sigma,wm_cv,ws_cv", _CVMIX_WEAK_FLUX)
def test_default_weak_flux_matches_compiled_cvmix(ustar, Bf, h, sigma, wm_cv, ws_cv):
    """The production default reproduces compiled CVMix for |B_f| below float32 eps."""
    a = jnp.asarray
    wm, ws = kpp_mod._kpp_velocity_scales(a([ustar]), a([Bf]), a([[sigma * h]]), a([[h]]), KPPConfig(), kpp_mod._EPS)
    assert float(wm[0, 0]) == pytest.approx(wm_cv, rel=1e-12, abs=0.0)
    assert float(ws[0, 0]) == pytest.approx(ws_cv, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("Bf", [-1e-8, -5e-9, -1e-9])
def test_weak_stable_flux_gradient_is_nonzero_by_default(Bf):
    """Stable weak flux (0 > B_f > -eps): the legacy floor is constant in B_f, so dw_s/dB_f = 0 there; the
    default restores a nonzero derivative.  u* = 5.8e-4 m/s is the stress-path floor sqrt(sqrt(eps)/rho_0); h = 50 m, d = 2 m."""
    def ws(b, cfg):
        return kpp_mod._kpp_velocity_scales(jnp.asarray([5.8e-4]), b[None], jnp.asarray([[2.0]]),
                                            jnp.asarray([[50.0]]), cfg, kpp_mod._EPS)[1][0, 0]
    b = jnp.asarray(Bf)
    g_default = float(jax.grad(ws)(b, KPPConfig()))
    g_legacy = float(jax.grad(ws)(b, KPPConfig(actual_buoyancy_flux=False)))
    assert np.isfinite(g_default) and abs(g_default) > 0.0
    assert g_legacy == 0.0
    check_grads(lambda x: ws(x, KPPConfig()), (b,), order=1, modes=["rev"], eps=1e-12)


@pytest.mark.parametrize("bad", [None, "false", 0])
def test_non_bool_switch_raises(bad):
    with pytest.raises(ValueError, match="must be a bool"):
        lego_ws(KPPConfig(actual_buoyancy_flux=bad), 0.01, 1e-8, 100.0)
