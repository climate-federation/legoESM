"""static_land_roughness: the roughness the land-stress seed uses before any
land solve must be the one the column's own surface scheme would use."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.stability import compute_aerodynamics
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    solved_stress_magnitude, static_land_roughness,
)
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig


def test_two_leaf_uses_the_per_column_canopy_geometry():
    lp = SimpleNamespace(hc=jnp.array([20.0, 0.5]), LAI=jnp.array([5.0, 0.3]),
                         rz0m=jnp.array([0.055, 0.12]),
                         rd=jnp.array([0.67, 0.67]))
    z0, d = static_land_roughness(
        lp, MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig()), 2)
    z0e, de = compute_aerodynamics(lp.hc, lp.LAI, lp.rz0m, lp.rd)
    np.testing.assert_allclose(z0, z0e, rtol=1e-12)
    np.testing.assert_allclose(d, de, rtol=1e-12)
    assert z0[0] > 5.0 * z0[1] and d[0] > d[1]   # forest rougher than grass


def test_simple_seb_uses_its_z0_and_no_displacement():
    cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig())
    z0, d = static_land_roughness(SimpleNamespace(z0=jnp.array([0.3, 0.01])),
                                  cfg, 2)
    np.testing.assert_array_equal(z0, [0.3, 0.01])
    np.testing.assert_array_equal(d, [0.0, 0.0])
    z0, _ = static_land_roughness(None, cfg, 3)
    np.testing.assert_array_equal(z0, [cfg.z0_land] * 3)


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="no static roughness"):
        static_land_roughness(
            None, MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig()), 2)


@pytest.mark.parametrize("scheme", ["simple_seb", "two_leaf"])
def test_land_reports_its_full_stress_magnitude(scheme):
    """tau_mag is the solved rho*u*^2; the (tau_x, tau_y) vector can be
    shortened by a wind-speed floor in light wind, which is why the atmosphere
    is handed tau_mag, not the vector length."""
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land_with_diagnostics,
    )
    f = lambda x: jnp.full(2, x)  # noqa: E731
    forcing = AtmToSurface(
        sw_down=f(0.), lw_down=f(300.), precip_total=f(0.), precip_snow=f(0.),
        T_lowest=f(280.), q_lowest=f(.004), u_lowest=jnp.array([1.0, 8.0]),
        v_lowest=f(0.), p_lowest=f(98000.), p_surface=f(100000.),
        rho_lowest=f(1.2), cos_zenith=f(0.), co2_ppmv=f(412.),
        has_radiation=f(1.), has_precipitation=f(1.))
    sch = SimpleSEBConfig() if scheme == "simple_seb" else TwoLeafCanopyConfig()
    cfg = MultiLayerLandConfig(bulk_scheme="most", surface_scheme=sch)
    st = init_multilayer_land_state(2, cfg, T_init=280.)
    u_min = 1.0
    _, resp, _, sfc = step_multilayer_land_with_diagnostics(
        st, forcing, cfg, u_min, 600.0)
    mag = np.asarray(sfc.tau_mag)
    np.testing.assert_array_equal(solved_stress_magnitude(sfc, resp), mag)
    vec = np.hypot(np.asarray(resp.tau_x), np.asarray(resp.tau_y))
    assert np.all(np.isfinite(mag)) and np.all(mag > 0.0)
    # two-leaf lays rho*u*^2 along u/sqrt(|V|^2+U_min^2): 1/sqrt(2) of it at
    # 1 m/s; simple_seb's bulk law floors the speed at 0.01 m/s only.
    floor2 = u_min ** 2 if scheme == "two_leaf" else 1e-4
    V = np.array([1.0, 8.0])
    np.testing.assert_allclose(vec / mag, V / np.sqrt(V ** 2 + floor2),
                               rtol=1e-6)


def test_solved_stress_magnitude_fallback_keeps_nan():
    """No tau_mag: the vector length; a NaN vector stays NaN (a failed solve
    must not read as calm), a zero vector is 0 with a finite gradient."""
    import jax
    resp = SimpleNamespace(tau_x=jnp.array([0.3, jnp.nan, 0.0]),
                           tau_y=jnp.array([0.4, 0.0, 0.0]))
    m = np.asarray(solved_stress_magnitude(SimpleNamespace(tau_mag=None), resp))
    assert m[0] == pytest.approx(0.5) and np.isnan(m[1]) and m[2] == 0.0
    g = jax.grad(lambda x: jnp.sum(solved_stress_magnitude(
        None, SimpleNamespace(tau_x=x, tau_y=jnp.zeros(1)))))(jnp.zeros(1))
    assert np.isfinite(np.asarray(g)).all()
