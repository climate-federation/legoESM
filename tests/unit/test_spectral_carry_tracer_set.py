"""The water species that ride the spectral time stepping follow the scheme.

A nine-species microphysics (morrison, thompson, p3, seifert_beheng, fast_sbm)
predicts ice, snow, graupel and three number concentrations on top of vapour,
cloud and rain. The spectral converters used to build a three-species state
unconditionally; the microphysics bridge emits a tendency only for a key
already present on ``state.tracers``, so on a nine-species scheme the extra six
were created and discarded at every evaluation while the vapour and cloud sinks
that produced them, and their latent heating, were kept. Neither water nor
energy closed and nothing raised (WeatherBench classical arm, job 26929456).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.tracers import make_full_moisture_registry
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training.neural_gcm_spectral import (
    carry_to_spectral_state, spectral_state_to_carry,
)

_EXTRA = make_full_moisture_registry().names[3:]   # q_i q_s q_g N_c N_r N_i


def _carry(n_lat=8, n_lon=16, nlev=4, extras=False):
    """A minimal SegmentCarry, with or without the six optional species."""
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field

    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    s3, s2 = (n_lat, n_lon, nlev), (n_lat, n_lon)
    z3, z2 = jnp.zeros(s3), jnp.zeros(s2)
    hstate = HydrostaticState(
        u=Field(z3, name="u", dims=d3, units="m/s"),
        v=Field(z3, name="v", dims=d3, units="m/s"),
        T=Field(jnp.full(s3, 280.0), name="T", dims=d3, units="K"),
        p_s=Field(jnp.full(s2, 1.0e5), name="p_s", dims=d2, units="Pa"),
        phis=Field(z2, name="phis", dims=d2, units="m2/s2"),
    )
    # Distinct NON-ZERO values per species: a converter that creates the six
    # keys but drops or zeroes their contents would pass with zero seeds.
    seeds = ({name: jnp.full(s3, 1.0 + i) for i, name in enumerate(_EXTRA)}
             if extras else {})
    return pack_carry(
        hstate, q_v=jnp.full(s3, 1e-3), q_c=z3, q_r=z3,
        held_dT_rad=z3, held_sw_net_sfc=z2, held_lw_net_sfc=z2,
        held_sw_up_toa=z2, held_lw_up_toa=z2, held_sw_down_toa=z2,
        step_index=0, **seeds,
    )


@pytest.mark.parametrize("extras,expected", [(False, 3), (True, 9)])
def test_the_state_carries_what_the_carry_carries(extras, expected):
    grid = create_gaussian_grid(5)
    carry = _carry(grid.n_lat, grid.n_lon, extras=extras)
    state = carry_to_spectral_state(carry, grid)
    assert set(state.tracers) == set(("q_v", "q_c", "q_r") + (_EXTRA if extras else ()))
    assert len(state.tracers) == expected


@pytest.mark.parametrize("extras", [False, True])
def test_the_round_trip_preserves_the_species_set(extras):
    """Forecast and initial condition must stay structurally identical, or the
    loss compares two different pytrees."""
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    carry = _carry(grid.n_lat, grid.n_lon, extras=extras)
    back = spectral_state_to_carry(carry_to_spectral_state(carry, grid), grid, sigma)
    for name in _EXTRA:
        before, after = getattr(carry, name), getattr(back, name)
        assert (after is None) == (before is None), name
        if before is not None:      # values must survive, not just the key
            assert jnp.allclose(after, before), name
    assert jax.tree.structure(back) == jax.tree.structure(carry)


def test_a_nine_species_scheme_gets_nine_slots_from_the_loader():
    """The seeding the production driver already does, reached from the WB
    training loader's own scheme argument."""
    from legoesm.training.era5_to_state import prognostic_carry_seeds

    assert set(prognostic_carry_seeds("morrison", "none", (4, 8, 3))) == set(_EXTRA)
    assert prognostic_carry_seeds("sundqvist", "none", (4, 8, 3)) == {}


def test_a_state_that_cannot_hold_the_scheme_is_refused_at_build_time():
    """The WeatherBench arm trained to a NaN because nobody checked that the
    state it built could hold the species its microphysics writes. The check
    counts the CARRY, not the registry the scheme itself selected — counting
    the latter can only ever agree with itself and could never fire."""
    from legoesm.training.scale_build import validate_carry_holds_scheme

    three = _carry(8, 16, extras=False)
    nine = _carry(8, 16, extras=True)

    assert validate_carry_holds_scheme(three, "sundqvist", context="t") == 3
    assert validate_carry_holds_scheme(nine, "morrison", context="t") == 9
    with pytest.raises(ValueError, match="morrison"):
        validate_carry_holds_scheme(three, "morrison", context="t")
