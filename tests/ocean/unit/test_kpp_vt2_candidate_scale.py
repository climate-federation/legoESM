"""KPPConfig.vt2_velocity_scale = "candidate": the w_s inside the bulk-Richardson V_t^2 is the CVMix / MOM6 one.

Reference (read 5 November 2026, NOAA-GFDL MOM6 main and CVMix-src master):
* MOM6 MOM_CVMix_KPP.F90, KPP_compute_BLD: ``Vt_layer = 1.0`` and
  ``call CVMix_kpp_compute_turbulent_scales(Vt_layer, OBL_depth, surfBuoyFlux2, surfFricVel, ..., w_s=Ws_1d)``
  with ``OBL_depth(k) = -cellHeight(k)``, every cell-centre depth taken as a candidate boundary-layer depth.
* CVMix cvmix_kpp.F90, cvmix_kpp_compute_turbulent_scales_1d_OBL (not lStokesMOST):
  ``sigma_loc = min(surf_layer_ext, sigma_coord)``, kept unless ``l_LMD_ws`` (default .false.) and the forcing
  is stabilising; ``zeta = sigma_loc * OBL_depth * surf_buoy_force * vonkar / surf_fric_vel**3``;
  ``w_s = vonkar * surf_fric_vel / phi_s(zeta)``, phi_s = 1 + 5 zeta (zeta >= 0), (1 - 16 zeta)^(-1/2)
  (zeta_s < zeta < 0), (a_s - c_s zeta)^(-1/3) (zeta <= zeta_s).
So w_s(d) = w_s at zeta = epsilon * d * B * kappa / u*^3 with CVMix's sign (B > 0 stabilising).  The reference below
is written from those lines alone, not from legoESM's code.

Tests:
1. the w_s that _boundary_layer_depth passes to V_t^2 under "candidate" equals the CVMix reference to 1e-12,
   stable and unstable, weak and convective regimes;
2. the depth argument it passes is each candidate depth itself;
3. the default is "candidate"; "column" stays selectable and its V_t^2 velocity scale is unchanged;
4. an unknown option raises;
5. the boundary-layer depth under "candidate" is finite with finite gradients.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing import kpp as kpp_mod  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig  # noqa: E402

KAPPA, EPS_SL = 0.4, 0.1
ZETA_S = -1.0
# CVMix computes the convective constants from zeta_s (cvmix_kpp.F90 lines 374 to 376):
# a_s = sqrt(1 - 16 zeta_s) (1 + 8 zeta_s) = -7 sqrt(17), c_s = 24 sqrt(1 - 16 zeta_s) = 24 sqrt(17).
# legoESM keeps LMD94's published rounding (-28.86, 98.96; test_kpp_lmd94_faithful pins them), a relative
# difference of about 5e-5, so the convective regime is compared at 1e-4 and the others at 1e-12.
A_S, C_S = -7.0 * np.sqrt(17.0), 24.0 * np.sqrt(17.0)


def cvmix_ws(ustar, B_cvmix, d):
    """CVMix 1d_OBL w_s with sigma_coord = 1 (MOM6 Vt_layer), OBL_depth = d, l_LMD_ws = .false.
    Returns w_s and whether the convective branch was used."""
    zeta = EPS_SL * d * B_cvmix * KAPPA / ustar ** 3
    if zeta >= 0.0:
        phi, conv = 1.0 + 5.0 * zeta, False
    elif zeta >= ZETA_S:                      # CVMix: the weakly unstable form down to zeta_s inclusive
        phi, conv = (1.0 - 16.0 * zeta) ** -0.5, False
    else:
        phi, conv = (A_S - C_S * zeta) ** (-1.0 / 3.0), True
    return KAPPA * ustar / phi, conv


def _column(nlev=40):
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    dz = np.concatenate([np.full(20, 5.0), np.full(nlev - 20, 20.0)])
    zc = create_z_star_from_thicknesses(jnp.asarray(dz))
    z = np.cumsum(dz) - 0.5 * dz
    T = 28.0 - 14.0 * (1.0 + np.tanh((z - 60.0) / 15.0)) / 2.0
    S = np.full_like(T, 35.0)
    u = 0.6 * np.exp(-((z - 100.0) / 40.0) ** 2) - 0.3 * np.exp(-z / 20.0)
    v = np.zeros_like(T)
    return zc, z, T, S, u, v


def _capture_ws(cfg, ustar, Bf, h_bl_prev=None):
    """Run _boundary_layer_depth and capture the arguments of its _kpp_velocity_scales call."""
    from legoesm.ocean.eos import wright_eos
    zc, z, T, S, u, v = _column()
    rho = wright_eos(jnp.asarray(T), jnp.asarray(S), jnp.zeros_like(jnp.asarray(T)))
    seen = {}
    orig = kpp_mod._kpp_velocity_scales

    def spy(u_star, B_f, d, h_bl_col, cfg_, eps, ice_frac=None):
        out = orig(u_star, B_f, d, h_bl_col, cfg_, eps, ice_frac=ice_frac)
        seen.update(d=np.asarray(d), h=np.asarray(h_bl_col), ws=np.asarray(out[1]))
        return out
    kpp_mod._kpp_velocity_scales = spy
    try:
        h = kpp_mod._boundary_layer_depth(
            rho, jnp.asarray(T), jnp.asarray(S), jnp.asarray(u), jnp.asarray(v), zc, jnp.ones(()),
            jnp.asarray(ustar), jnp.asarray(Bf), cfg, h_bl_prev=h_bl_prev)
    finally:
        kpp_mod._kpp_velocity_scales = orig
    return z, seen, float(h)


# |B_f| = 0 or above float32 eps (1.2e-7 m2/s3), with the default actual-flux form; below eps (and at 0) the
# legacy floor (actual_buoyancy_flux=False) substitutes signed eps.  Weak fluxes: test_kpp_bf_floor.py vs CVMix.
@pytest.mark.parametrize("ustar,Bf", [(0.01, 0.0), (0.01, -2e-7), (0.01, 2e-7), (0.003, 3e-7), (0.02, 5e-7),
                                      (0.005, -1e-6)])
def test_candidate_ws_equals_cvmix(ustar, Bf):
    cfg = KPPConfig(vt2_velocity_scale="candidate")
    z, seen, _ = _capture_ws(cfg, ustar, Bf)
    # legoESM B_f > 0 is destabilising; CVMix surf_buoy_force > 0 is stabilising
    out = [cvmix_ws(ustar, -Bf, d) for d in z]
    ref = np.array([w for w, _ in out])
    conv = np.array([c for _, c in out])
    got = np.ravel(seen["ws"])
    np.testing.assert_allclose(got[~conv], ref[~conv], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got[conv], ref[conv], rtol=1e-4, atol=1e-15)


def test_candidate_passes_each_depth_as_its_own_layer():
    cfg = KPPConfig(vt2_velocity_scale="candidate")
    z, seen, _ = _capture_ws(cfg, 0.01, 2e-7)
    np.testing.assert_allclose(np.broadcast_to(seen["h"], z.shape), z, rtol=0, atol=1e-12)


def test_default_is_candidate():
    assert KPPConfig().vt2_velocity_scale == "candidate"


def test_column_still_selectable_and_unchanged():
    col = KPPConfig(vt2_velocity_scale="column")
    z, seen, _ = _capture_ws(col, 0.01, 2e-7)
    np.testing.assert_allclose(np.ravel(seen["h"]), [z[-1]], rtol=0, atol=1e-12)   # the full column depth
    z, seen, _ = _capture_ws(col, 0.01, 2e-7, h_bl_prev=jnp.asarray(35.0))
    np.testing.assert_allclose(np.ravel(seen["h"]), [35.0], rtol=0, atol=1e-12)


def test_candidate_shallows_a_convective_layer():
    """Under night-time cooling the capped scale gives a smaller V_t^2, so the layer is not deeper."""
    _, _, h_col = _capture_ws(KPPConfig(vt2_velocity_scale="column"), 0.008, 3e-7)
    _, _, h_cand = _capture_ws(KPPConfig(vt2_velocity_scale="candidate"), 0.008, 3e-7)
    assert h_cand <= h_col + 1e-9


def test_unknown_option_raises():
    with pytest.raises(ValueError):
        _capture_ws(KPPConfig(vt2_velocity_scale="bogus"), 0.01, 0.0)


def test_candidate_finite_with_gradients():
    from legoesm.ocean.eos import wright_eos
    zc, z, T, S, u, v = _column()
    cfg = KPPConfig(vt2_velocity_scale="candidate")

    def h_of(Tc):
        rho = wright_eos(Tc, jnp.asarray(S), jnp.zeros_like(Tc))
        return kpp_mod._boundary_layer_depth(rho, Tc, jnp.asarray(S), jnp.asarray(u), jnp.asarray(v), zc,
                                             jnp.ones(()), jnp.asarray(0.01), jnp.asarray(2e-7), cfg)
    h = h_of(jnp.asarray(T))
    g = jax.grad(h_of)(jnp.asarray(T))
    assert np.isfinite(float(h)) and np.all(np.isfinite(np.asarray(g)))
