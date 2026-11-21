"""CLM-ML canopy follows the run's orbit and solar constant."""

import jax
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.solar import OrbitalParameters, solar_declination
from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import (
    shr_orb_decl, shr_orb_derived, shr_orb_params,
)
from legoesm.land.canopy.config import CLMMLCanopyConfig

jax.config.update("jax_enable_x64", True)


# shr_orb_params outputs from commit 690fc7c3e5,
# before shr_orb_derived was split out (float.hex, frozen by job 11305475).
_FROZEN = {
    2001: ['0x1.11aa86c023e6dp-6', '0x1.7708c17b64277p+4', '0x1.9ba686f71a8d0p+6',
           '0x1.a2eacf3fc4b13p-2', '-0x1.09fd3231a3c6ap-5', '0x1.3c04350cef34ap+2'],
    -4050: ['0x1.32154021d24cap-6', '0x1.81afa3abcce6ap+4', '0x1.bd3de3e58707ep-1',
            '0x1.aed0c489eba26p-2', '-0x1.2d86283ae44f8p-11', '0x1.94110c43756d0p+1'],
    -125050: ['0x1.429580edaa566p-5', '0x1.80a47769d834ap+4', '0x1.1368816b9fe70p+8',
              '0x1.ada6552f87974p-2', '0x1.41f61ca218227p-4', '0x1.fcb22484aaf99p+2'],
}


@pytest.mark.parametrize("year", sorted(_FROZEN))
def test_derived_split_keeps_shr_orb_params_bitwise(year):
    assert [float(v).hex() for v in shr_orb_params(year)] == _FROZEN[year]
    eccen, obliq, mvelp, obliqr, lambm0, mvelpp = shr_orb_params(year)
    np.testing.assert_allclose(shr_orb_derived(eccen, obliq, mvelp),
                               (obliqr, lambm0, mvelpp), rtol=1e-14)


@pytest.mark.parametrize("orbit", [OrbitalParameters(0.018682, 24.105, 180.87),
                                   OrbitalParameters(0.09, 60.0, 95.0)])
def test_installed_clm_orbit_gives_the_radiation_declination(orbit):
    from legoesm.land.canopy.clm_ml_interface import _install_clm_orbit
    import legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb as vo
    _install_clm_orbit(CLMMLCanopyConfig(
        orbit_eccentricity=orbit.eccentricity, orbit_obliquity_deg=orbit.obliquity_deg,
        orbit_long_perihelion_deg=orbit.long_perihelion_deg))
    try:
        for day in (1.0, 80.5, 172.3, 300.0):
            clm, _ = shr_orb_decl(day, vo.eccen, vo.mvelpp, vo.lambm0, vo.obliqr)
            # CLM's own pi is 3.141592654 (CLM rpi): up to 1.9e-10 rad apart.
            assert abs(clm - float(solar_declination(day, orbit=orbit))) < 1e-9
    finally:
        _install_clm_orbit(None)


@pytest.mark.parametrize("kw", [dict(orbit_eccentricity=0.2, orbit_obliquity_deg=23.0,
                                     orbit_long_perihelion_deg=10.0),
                                dict(orbit_eccentricity=0.01)])
def test_clm_ml_refuses_high_or_partial_orbit(kw):
    with pytest.raises(ValueError, match="orbit"):
        CLMMLCanopyConfig(**kw).validate()


def test_clearness_index_uses_the_run_solar_constant():
    from legoesm.land.canopy.clm_ml_interface import _estimate_beam_fraction
    sw, cz = np.array([600.0]), np.array([0.8])
    assert float(_estimate_beam_fraction(sw, cz, 1361.0)[0]) != float(
        _estimate_beam_fraction(sw, cz, 1000.0)[0])
