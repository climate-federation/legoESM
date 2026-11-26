"""KPPConfig.monin_obukhov_limit: under stabilising surface buoyancy forcing the
boundary layer is no deeper than the Monin-Obukhov length L = u*^3 / (kappa B).

Reference (CVMix-src master cvmix_kpp.F90, cvmix_kpp_compute_OBL_depth, not
Stokes-MOST, lMonOb = .true.; read 26 November 2026):
``lstable = surf_buoy > 0``; stable: ``MoninObukhov = surf_fric**3 /
(surf_buoy * vonkarman)``, else the column bottom; ``OBL_limit = min(OBL_limit,
MoninObukhov)``; ``OBL_depth = min(OBL_depth, OBL_limit)``, and no shallower
than the top cell centre.  CVMix's surf_buoy > 0 is stabilising; legoESM's
B_f > 0 is destabilising, so CVMix surf_buoy = -B_f.  The reference below is
written from those lines, with legoESM's own top-layer floor (one top-layer
thickness).

RULE 5 tests: (1) both widths 0 reproduce the reference to round-off; (2) the
smooth default's gap from the hard limit, measured per case and stored; (3)
finite, correct derivatives through the limit at the default.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import wright_eos  # noqa: E402
from legoesm.ocean.physics.vertical_mixing import kpp as kpp_mod  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig  # noqa: E402

KAPPA = KPPConfig().kappa_vk
HARD = dict(mo_limit_width_m=0.0, mo_stable_width_bf=0.0)

# (u*, B_f): stable heating where the limit binds (L 36, 13, 50, 0.3 m; the last
# falls to the top-layer floor); stable inside the B_f smoothing width; neutral; unstable
BINDING = [(0.007, -2.4e-8), (0.004, -1.2e-8), (0.01, -5e-8), (0.003, -2e-7)]
# IN_BAND: L within the min's 1 m width of the free depth (gap from the min);
# L below the free depth with -B_f inside its 1e-10 smoothing width (gap from
# the positive part: L 17.15 m hard, 22.87 m smooth)
IN_BAND = [(0.015189, -2.4e-8), (0.0007, -5e-11)]
CASES = BINDING + IN_BAND + [(0.007, -5e-11), (0.01, 0.0), (0.007, 2e-7)]


def _column(nlev=40):
    """Weakly stratified cold column with an inertial current: without the limit
    the layer goes ~360 m deep under stable forcing."""
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    dz = np.concatenate([np.full(20, 5.0), np.full(nlev - 20, 20.0)])
    zc = create_z_star_from_thicknesses(jnp.asarray(dz))
    z = np.cumsum(dz) - 0.5 * dz
    T = 2.6 - 0.05 * z / z[-1]
    S = np.full_like(T, 35.0)
    u = 0.2 * np.exp(-z / 100.0)
    return zc, dz, jnp.asarray(T), jnp.asarray(S), jnp.asarray(u), jnp.zeros(nlev)


def _h(cfg, ustar, Bf):
    zc, _, T, S, u, v = _column()
    rho = wright_eos(T, S, jnp.zeros_like(T))
    return kpp_mod._boundary_layer_depth(rho, T, S, u, v, zc, jnp.ones(()),
                                         jnp.asarray(ustar), jnp.asarray(Bf), cfg)


def test_default_is_on():
    assert KPPConfig().monin_obukhov_limit is True


@pytest.mark.parametrize("ustar,Bf", CASES)
def test_hard_limit_matches_cvmix(ustar, Bf):
    _, dz, *_ = _column()
    h_free = float(_h(KPPConfig(monin_obukhov_limit=False), ustar, Bf))
    got = float(_h(KPPConfig(**HARD), ustar, Bf))
    L = ustar ** 3 / (KAPPA * -Bf) if Bf < 0 else np.inf
    want = max(min(h_free, L), dz[0])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    if (ustar, Bf) in BINDING:
        assert L < h_free - 5.0, "case must bind, or it cannot tell the limit from no limit"


# |h(default widths) - h(hard)| [m], measured 2026-11-26 at the defaults
# (mo_limit_width_m 1.0, mo_stable_width_bf 1e-10); re-measure if either changes.
SMOOTH_GAP = {
    (0.007, -2.4e-8): 0.0, (0.004, -1.2e-8): 0.0, (0.01, -5e-8): 0.0, (0.003, -2e-7): 0.0,
    (0.015189, -2.4e-8): -0.054461151542056996, (0.0007, -5e-11): 5.716666666666669,
    (0.007, -5e-11): 0.0, (0.01, 0.0): 0.0, (0.007, 2e-7): 0.0,
}


@pytest.mark.parametrize("ustar,Bf", CASES)
def test_smooth_default_gap_recorded(ustar, Bf):
    gap = float(_h(KPPConfig(), ustar, Bf)) - float(_h(KPPConfig(**HARD), ustar, Bf))
    want = SMOOTH_GAP[(ustar, Bf)]
    np.testing.assert_allclose(gap, want, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("ustar,Bf", BINDING[:3] + IN_BAND)
def test_gradients_through_limit(ustar, Bf):
    """Each argument on its own step: u* ~ 1e-2 m/s and B_f ~ 1e-8 m^2/s^3."""
    cfg = KPPConfig()
    check_grads(lambda us: _h(cfg, us, Bf), (jnp.asarray(ustar),), order=2, modes=["rev"],
                atol=1e-6, rtol=1e-4, eps=1e-7 * ustar)
    check_grads(lambda b: _h(cfg, ustar, b), (jnp.asarray(Bf),), order=2, modes=["rev"],
                atol=1e-6, rtol=1e-4, eps=1e-4 * abs(Bf))
    g = jax.grad(lambda us: _h(cfg, us, Bf))(jnp.asarray(ustar))
    assert np.isfinite(float(g)) and float(g) > 0.0


def test_stable_heating_layer_is_shallow():
    h = float(_h(KPPConfig(), 0.007, -2.4e-8))
    assert h < 100.0, h
    assert float(_h(KPPConfig(monin_obukhov_limit=False), 0.007, -2.4e-8)) > 2 * h


def _h_bottom(cfg, ustar, Bf):
    """Statically unstable column at rest (the layer goes ~429 m deep)."""
    zc, z, T, S, u, v = _column()
    T = jnp.asarray(2.6 + 0.5 * z / z[-1])
    rho = wright_eos(T, S, jnp.zeros_like(T))
    return kpp_mod._boundary_layer_depth(rho, T, S, jnp.zeros_like(u), v, zc, jnp.ones(()),
                                         jnp.asarray(ustar), jnp.asarray(Bf), cfg)


@pytest.mark.parametrize("ustar,Bf", [(0.01, 0.0), (0.007, 2e-7), (0.003, 5e-7)])
def test_neutral_and_unstable_untouched(ustar, Bf):
    """No stable forcing: the limit leaves h exactly as it was, also at the bottom."""
    for h_of in (_h, _h_bottom):
        on = float(h_of(KPPConfig(), ustar, Bf))
        off = float(h_of(KPPConfig(monin_obukhov_limit=False), ustar, Bf))
        assert on == off, (h_of.__name__, on, off)



@pytest.mark.parametrize("Bf", [0.0, 2e-7, -1e-12, -2.4e-8])
def test_layer_at_bottom_untouched_unless_binding(Bf):
    """h within the smoothing band of the column depth H: returned exactly unless
    L < H binds (u* = 0.05 makes L = 1.3e4 m for the stable case)."""
    H = jnp.asarray(490.0)
    for h in (H - 0.2, H):
        out = kpp_mod._monin_obukhov_limit(h, jnp.asarray(0.05), jnp.asarray(Bf), H, KPPConfig())
        assert float(out) == float(h)


def test_dry_column_gradients_finite():
    """Zero column depth (land, z-star Jacobian 0): value and gradients finite."""
    cfg = KPPConfig()
    f = lambda us, b: kpp_mod._monin_obukhov_limit(jnp.asarray(0.0), us, b, jnp.asarray(0.0), cfg)
    for b in (-2.4e-8, -5e-11, 0.0, 2e-7):
        val = f(jnp.asarray(0.007), jnp.asarray(b))
        gu, gb = jax.grad(f, argnums=(0, 1))(jnp.asarray(0.007), jnp.asarray(b))
        assert np.isfinite([float(val), float(gu), float(gb)]).all(), (b, val, gu, gb)
