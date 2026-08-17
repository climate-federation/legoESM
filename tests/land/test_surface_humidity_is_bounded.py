"""Land surface humidity must sit between the air and saturation.

`q_sfc = beta * q_sat` does not. beta is a flux efficiency, `r_a/(r_a+r_s)`;
applied to a humidity it leaves

    E = beta*E_pot - rho*(1-beta)*q_air/r_a

whose second term is a condensation source that GROWS as the soil dries. Offline
that pinned the latent flux at its -150 W/m2 condensation clamp over 91 % of
land (mean -103, i.e. 3.6 mm/day of dew, more than global mean rainfall), held
the surface at 310 K, and irrigated the Sahara to 0.276 volumetric moisture.

These assertions are the ones that fail on the product form and pass on the
gradient form, so they pin the distinction rather than the implementation.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # the 1e-15 identity below needs f64
jnp = jax.numpy

from legoesm.atmosphere.physics.turbulence.surface_layer import (  # noqa: E402
    beta_limited_surface_humidity,
)

_LAND = jnp.ones(())


def _q_sfc(q_sat, q_air, beta):
    return float(beta_limited_surface_humidity(
        jnp.asarray(q_sat), jnp.asarray(q_air), _LAND, jnp.asarray(beta)))


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.4, 0.9, 1.0])
def test_surface_humidity_stays_between_air_and_saturation(beta):
    q_air, q_sat = 0.010, 0.020
    q = _q_sfc(q_sat, q_air, beta)
    assert q_air - 1e-12 <= q <= q_sat + 1e-12, (
        f"beta={beta} put the surface at {q:.4f}, outside [{q_air}, {q_sat}]")


def test_a_bone_dry_surface_evaporates_nothing_rather_than_condensing():
    """The single guard that separates the two forms.

    At beta = 0 the flux must vanish. The product form gives -q_air, i.e. the
    full potential flux RUNNING BACKWARDS onto a surface that has no water.
    """
    q_air, q_sat = 0.010, 0.020
    assert _q_sfc(q_sat, q_air, 0.0) - q_air == pytest.approx(0.0, abs=1e-15)
    assert (0.0 * q_sat) - q_air < 0.0, (
        "the product form is supposed to fail this; if it does not, the "
        "example no longer discriminates and this test is vacuous")


def test_a_dry_soil_under_moist_air_still_evaporates():
    """Dry soil plus humid air plus low wind is where the product form breaks
    hardest: it manufactures condensation exactly where evaporation is weakest."""
    q_air, q_sat, beta = 0.010, 0.020, 0.4
    assert _q_sfc(q_sat, q_air, beta) - q_air > 0.0
    assert (beta * q_sat) - q_air < 0.0          # what the old form did


def test_real_dew_survives_when_the_surface_is_actually_cold():
    """A surface colder than the air is moist must still take water up --
    otherwise the fix would have removed frost and dew along with the fake ones.
    In the scheme this is reached by letting beta -> 1 when q_sat < q_air."""
    q_air, q_sat = 0.015, 0.008
    assert _q_sfc(q_sat, q_air, 1.0) - q_air < 0.0


def _seb_lhflx(q_air, beta_soil):
    """Latent flux from the LAND scheme itself (compute_simple_seb_fluxes), not
    the helper — pins the production path, which a helper-only test cannot
    (codex: a helper test passes even if the scheme still used the product form)."""
    import numpy as np
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import LandConfig
    from legoesm.land.surface_scheme import compute_simple_seb_fluxes
    n = 4
    c = lambda v: jnp.full((n,), float(v))
    forcing = AtmToSurface(
        sw_down=c(0.0), lw_down=c(320.0), precip_total=c(0.0), precip_snow=c(0.0),
        T_lowest=c(295.0), q_lowest=c(q_air), u_lowest=c(1.0), v_lowest=c(0.5),
        p_lowest=c(9.9e4), p_surface=c(1.0e5), rho_lowest=c(1.15),
        cos_zenith=c(0.0), co2_ppmv=c(412.0), has_radiation=c(1.0),
        has_precipitation=c(1.0))
    out = compute_simple_seb_fluxes(
        T_surface=c(300.0), snow=c(0.0), snow_age=c(0.0), beta_soil=c(beta_soil),
        forcing=forcing, land_config=LandConfig(), U_min=1.0, lat=jnp.zeros(n),
        carbon_state=None, dt=1800.0, land_params=None, albedo_land=c(0.2),
        emissivity=c(0.97), z0=c(0.05))
    return float(np.mean(np.asarray(out.lhflx)))


def test_seb_dry_soil_moist_air_does_not_condense():
    """The production land scheme, warm dry soil under humid air: the product
    form gave a large NEGATIVE (condensation) flux here; the gradient form must
    give a small non-negative one.  q_air=0.010 < q_sat(300 K)~0.022, beta=0.05."""
    assert _seb_lhflx(0.010, 0.05) >= 0.0


def test_seb_beta_zero_shuts_off_evaporation():
    """beta_soil -> beta_min floor cannot be bypassed here, so compare a dry and
    a wet soil: the wet one must evaporate strictly more, and the dry one must
    not run backwards (the product form's signature failure)."""
    dry, wet = _seb_lhflx(0.010, 0.02), _seb_lhflx(0.010, 1.0)
    assert dry >= 0.0
    assert wet > dry + 1.0
