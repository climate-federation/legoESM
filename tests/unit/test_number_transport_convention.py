"""Droplet number transports like its own mass, so particle size survives it.

The dycores advect every tracer with a mass-mixing-ratio operator: the quantity
is invariant following a parcel. A per-VOLUME number density does not obey that
equation, so storing cloud and rain number per volume let them drift against
their own mass and the diagnosed particle size went wrong by the density change
along the trajectory. They are now stored and transported per MASS, like ice
number always was, and converted at the physics bridge where density is known.

The invariant below needs no weights, no grid and no physics: mean droplet mass
is q_c / n_c with both per mass, so if the two are transported by the same
operator it cannot change, whatever the flow does.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.integration import (
    number_per_mass_to_per_volume, number_per_volume_to_per_mass,
)
from legoesm.core.tracers import make_full_moisture_registry

# 900 hPa to 300 hPa: the density a rising parcel loses. Under the old
# convention this factor landed directly on the diagnosed size.
_RHO_LOW, _RHO_HIGH = 1.10, 0.40


def test_the_registry_says_all_three_numbers_are_per_mass():
    units = {t.name: t.units for t in make_full_moisture_registry().tracers}
    assert units["N_c"] == units["N_r"] == units["N_i"] == "1/kg"


def test_the_conversion_round_trips():
    rho = jnp.array([_RHO_LOW, 0.8, _RHO_HIGH])
    n = jnp.array([1.0e8, 2.0e8, 3.0e8])
    back = number_per_volume_to_per_mass(
        number_per_mass_to_per_volume(n, rho), rho)
    assert jnp.allclose(back, n, rtol=1e-12)


def test_mean_droplet_mass_survives_transport_through_a_density_drop():
    """Lift a cloudy parcel and check the size it arrives with.

    Transport is the identity on a per-mass scalar following the parcel, so the
    test moves q_c and the number by the SAME operator and asks what the
    physics bridge then sees.
    """
    q_c = 5.0e-4                    # kg/kg, invariant following the parcel
    n_per_mass = 1.0e8 / _RHO_LOW   # 1e8 m^-3 at the starting density

    def mean_droplet_mass(rho):
        """What the microphysics computes: x_c = q_c * rho / N_volume."""
        n_vol = number_per_mass_to_per_volume(n_per_mass, rho)
        return q_c * rho / n_vol

    x_start = mean_droplet_mass(_RHO_LOW)
    x_end = mean_droplet_mass(_RHO_HIGH)
    assert x_end == pytest.approx(x_start, rel=1e-12), (
        f"droplet mass changed on transport: {x_start:.6e} -> {x_end:.6e}")

    # ... and the control: had the number been stored per VOLUME and advected
    # by the same mixing-ratio operator, it would have arrived unchanged at
    # 1e8 m^-3 and the size would be wrong by the full density ratio.
    n_vol_frozen = 1.0e8
    x_end_old = q_c * _RHO_HIGH / n_vol_frozen
    x_start_old = q_c * _RHO_LOW / n_vol_frozen
    assert x_end_old / x_start_old == pytest.approx(_RHO_HIGH / _RHO_LOW)
    assert abs(x_end_old / x_start_old - 1.0) > 0.5, (
        "the control must show the error the new convention removes")


def test_an_old_restart_is_refused_rather_than_misread():
    """A checkpoint written when droplet number was per VOLUME has no way to
    announce itself. Reloading it as per mass would scale every droplet count
    by the air density, silently, so it must fail instead."""
    import numpy as np

    from legoesm.driver.model_driver import _validate_number_convention

    # No droplet number in the file: nothing to get wrong.
    _validate_number_convention({}, ["q_v", "q_c"])
    # Stamped file: accepted.
    _validate_number_convention(
        {"number_convention": np.asarray("per_mass")}, ["q_v", "N_c"])
    # Unstamped file WITH droplet number: refused.
    with pytest.raises(ValueError, match="PER-VOLUME"):
        _validate_number_convention({}, ["q_v", "N_c"])


def test_an_old_spectral_checkpoint_is_refused_too():
    import numpy as np

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        reconstruct_spectral_state_from_npz,
    )

    d = {"spectral_layout": np.asarray(1),
         "tracer_names": np.asarray(["N_c", "q_v"])}
    with pytest.raises(ValueError, match="PER-VOLUME"):
        reconstruct_spectral_state_from_npz(d)
