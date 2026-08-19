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

jnp = pytest.importorskip("jax.numpy")

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
    # Tolerance is float32: the model's default dtype resolves 0.01 kg/kg to
    # about 2e-10, so a bound asserted tighter than that measures the dtype
    # rather than the physics.  Exact under x64.
    tol = 1e-6 * q_sat
    assert q_air - tol <= q <= q_sat + tol, (
        f"beta={beta} put the surface at {q:.4f}, outside [{q_air}, {q_sat}]")


def test_a_bone_dry_surface_evaporates_nothing_rather_than_condensing():
    """The single guard that separates the two forms.

    At beta = 0 the flux must vanish. The product form gives -q_air, i.e. the
    full potential flux RUNNING BACKWARDS onto a surface that has no water.
    """
    q_air, q_sat = 0.010, 0.020
    # float32 again: beta=0 returns q_air to within ~2e-10 of it.
    assert _q_sfc(q_sat, q_air, 0.0) - q_air == pytest.approx(0.0, abs=1e-6 * q_sat)
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
