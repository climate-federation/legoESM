"""Direct tests of legoesm.training.era5_ocean_forcing: sign conventions of both modes, the
open-water scaling (stress/evaporation/heat scale with 1-A; precipitation/runoff full-cell)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.training import era5_ocean_forcing as F

jax.config.update("jax_enable_x64", True)
SHAPE = (4, 6)


def _day(**over):
    base = dict(u10=8.0, v10=0.0, t2m=290.0, d2m=285.0, msl=101300.0, sp=101300.0, ssrd=200.0,
                strd=350.0, tp=3e-8, sf=0.0, sshf=-20.0, slhf=-100.0, ssr=180.0, str=-50.0,
                ewss=0.1, nsss=-0.05, e=-4e-8)
    base.update(over)
    return np.stack([np.full(SHAPE, base[n]) for n in F.CHANNELS])


def _cfg():
    from legoesm.coupler.config import CouplerConfig
    return CouplerConfig(bulk_scheme="large_yeager", z_ref=10.0, z_t_atm=2.0, z_q_atm=2.0)


def _run(mode, ice=0.0, sst=20.0, **over):
    lat = jnp.deg2rad(jnp.linspace(-30, 30, SHAPE[0]))[:, None] * jnp.ones(SHAPE)
    lon = jnp.deg2rad(jnp.linspace(0, 300, SHAPE[1]))[None, :] * jnp.ones(SHAPE)
    return F.surface_forcing(mode, _day(**over), jnp.full(SHAPE, sst), jnp.full(SHAPE, 1e-5),
                             jnp.full(SHAPE, ice), jnp.ones(SHAPE), lat_rad=lat, lon_rad=lon,
                             day=10.5, ref_year=1995, coupler_cfg=_cfg())


def test_prescribed_signs_and_values():
    fw, sf = _run("prescribed")
    np.testing.assert_allclose(sf.q_net, 180.0 - 50.0 - 20.0 - 100.0)
    np.testing.assert_allclose(sf.tau_x, -0.1)          # coupler convention: stress on the air
    np.testing.assert_allclose(fw.evap, 4e-8 * constants.rho_water)   # positive up
    np.testing.assert_allclose(fw.precip, 3e-8 * constants.rho_water)


def test_bulk_stress_matches_coupler_sign_and_heat_loss():
    fw, sf = _run("bulk", sst=25.0)
    assert (np.asarray(sf.tau_x) < 0).all()             # eastward wind -> negative, as prescribed
    assert (np.asarray(fw.evap) > 0).all()
    fw2, sf2 = _run("bulk", sst=27.0)                    # warmer sea loses more heat
    assert (np.asarray(sf2.q_net) < np.asarray(sf.q_net)).all()


@pytest.mark.parametrize("mode", ["bulk", "prescribed"])
def test_full_ice_cover_scales_air_sea_fluxes_only(mode):
    fw, sf = _run(mode, ice=1.0)
    for x in (sf.q_net, sf.tau_x, sf.tau_y, sf.sw_down, fw.evap):
        np.testing.assert_allclose(x, 0.0, atol=1e-12)
    np.testing.assert_allclose(fw.runoff, 1e-5)
    assert (np.asarray(fw.precip) > 0).all()


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown forcing mode"):
        _run("jra55")
