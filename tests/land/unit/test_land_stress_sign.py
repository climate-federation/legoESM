"""Land surface stress is the stress ON THE AIR: it opposes the near-surface
wind, with the same sign as the ocean bulk stress it is tile-blended with
(#2031)."""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import compute_most_fluxes
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land_with_diagnostics,
)
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

U = jnp.array([5.0, -5.0, 3.0])
V = jnp.array([0.0, 0.0, -4.0])


def _land_stress(scheme):
    f = lambda x: jnp.full(3, x)  # noqa: E731
    forcing = AtmToSurface(
        sw_down=f(0.), lw_down=f(300.), precip_total=f(0.), precip_snow=f(0.),
        T_lowest=f(280.), q_lowest=f(.004), u_lowest=U, v_lowest=V,
        p_lowest=f(98000.), p_surface=f(100000.), rho_lowest=f(1.2),
        cos_zenith=f(0.), co2_ppmv=f(412.), has_radiation=f(1.),
        has_precipitation=f(1.))
    sch = SimpleSEBConfig() if scheme == "simple_seb" else TwoLeafCanopyConfig()
    cfg = MultiLayerLandConfig(bulk_scheme="most", surface_scheme=sch)
    st = init_multilayer_land_state(3, cfg, T_init=280.)
    _, resp, _, _ = step_multilayer_land_with_diagnostics(st, forcing, cfg, 1.0, 600.0)
    return np.asarray(resp.tau_x), np.asarray(resp.tau_y)


@pytest.mark.parametrize("scheme", ["simple_seb", "two_leaf"])
def test_land_stress_opposes_wind_like_ocean(scheme):
    tx, ty = _land_stress(scheme)
    assert np.all(tx * np.asarray(U) + ty * np.asarray(V) < 0.0)
    ox, oy, *_ = compute_most_fluxes(U, V, jnp.full(3, 280.), jnp.full(3, .004),
                                     jnp.full(3, 280.), jnp.full(3, .005),
                                     jnp.full(3, 1.2))
    ox, oy = np.asarray(ox), np.asarray(oy)
    assert np.all(ox * np.asarray(U) + oy * np.asarray(V) < 0.0)
    nz_x, nz_y = np.asarray(U) != 0.0, np.asarray(V) != 0.0
    np.testing.assert_array_equal(np.sign(tx[nz_x]), np.sign(ox[nz_x]))
    np.testing.assert_array_equal(np.sign(ty[nz_y]), np.sign(oy[nz_y]))


@pytest.mark.timeout(600)
def test_clm_ml_stress_opposes_wind(monkeypatch):
    """CLM-ML canopy stress opposes the wind: the real CLM-ML step of
    test_clm_ml_snow_ground_sublimation, run unchanged except for the winds,
    which are this file's."""
    import jax
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("CLM-ML step fixture is float64; JAX_ENABLE_X64=1")
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    from tests.land.unit import test_clm_ml_snow_ground_sublimation as fx
    base = fx._forcing
    for u, v in zip(np.asarray(U), np.asarray(V)):
        monkeypatch.setattr(fx, "_forcing", lambda u=u, v=v: base()._replace(
            u_lowest=jnp.full(1, u), v_lowest=jnp.full(1, v)))
        out = fx._call(monkeypatch)
        assert float(out.tau_x[0]) * u + float(out.tau_y[0]) * v < 0.0
