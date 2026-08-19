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


# ---------------------------------------------------------------------------
# positive_tracers: every species non-negative after the spectral step
# ---------------------------------------------------------------------------

def _neg_tracers(nlev=4):
    """Tiny tracer dict with deliberate MIXED-SIGN columns (net positive) in
    a mass and a number slot — so the conserving borrow is distinguishable
    from a plain clip (an all-negative column is zeroed by both)."""
    q = jnp.array([[[2.0e-3, -1.0e-4, 5.0e-4, -2.0e-5]]])   # (1, 1, nlev)
    n = jnp.array([[[1.0e6, -3.0e4, 2.0e5, 0.0]]])
    return {"q_v": q, "N_i": n}


def test_positive_tracers_borrows_every_per_mass_species():
    """Numbers included: N_* are stored per mass, and the plain clip that a
    units split would give them INVENTED number at every transport
    undershoot (x2.2/day compound on century3; N_i hit 1e193 then NaN —
    ``TestAllTracersBorrowed``). Column integrals must be conserved for BOTH
    the mixing ratio and the number concentration."""
    from legoesm.training.neural_gcm_spectral import positive_tracers

    sigma = create_sigma_coordinate(4)
    dsigma = jnp.asarray(sigma.dsigma)
    tr = _neg_tracers()
    out = positive_tracers(tr, sigma)

    for name in ("q_v", "N_i"):
        assert float(jnp.min(out[name])) >= 0.0, name
        col_before = float(jnp.sum(tr[name] * dsigma))
        col_after = float(jnp.sum(out[name] * dsigma))
        # Borrow, not creation: the plain-clip integral would be larger.
        assert col_after == pytest.approx(col_before, rel=1e-12), name


def test_positive_tracers_gradient_finite_at_zero_and_negative():
    """Cold-start states sit exactly on the clip kinks; the fixer must hand
    back finite gradients there (it runs inside the training adjoint)."""
    from legoesm.training.neural_gcm_spectral import positive_tracers

    sigma = create_sigma_coordinate(4)

    def f(q):
        out = positive_tracers({"q_v": q, "N_c": jnp.zeros_like(q)}, sigma)
        return jnp.sum(out["q_v"] ** 2) + jnp.sum(out["N_c"])

    g = jax.grad(f)(_neg_tracers()["q_v"])
    assert bool(jnp.all(jnp.isfinite(g)))


def test_positive_tracers_unknown_species_raises():
    from legoesm.training.neural_gcm_spectral import positive_tracers

    with pytest.raises(ValueError, match="not a known per-mass"):
        positive_tracers({"so2": jnp.zeros((1, 1, 4))},
                         create_sigma_coordinate(4))


def test_positive_tracers_refuses_hybrid_coordinate():
    """dsigma is the layer-mass weight for PURE sigma only; a hybrid
    coordinate's layer mass is dA·p_ref + dB·p_s (per column), so borrowing
    with flat dsigma would conserve the wrong physical integral (codex)."""
    from legoesm.training.neural_gcm_spectral import positive_tracers
    from legoesm.grids.vertical import create_hybrid_coordinate

    k = jnp.linspace(0.0, 1.0, 9)
    hybrid = create_hybrid_coordinate(
        8, A_half=1.0e4 * (1.0 - k) * k, B_half=k**2)
    with pytest.raises(ValueError, match="pure-sigma"):
        positive_tracers({"q_v": jnp.zeros((1, 1, 8))}, hybrid)


def test_rollout_step_leaves_every_species_non_negative_and_conserves():
    """Wiring test on the classical/unforced scan body: an IC with
    mixed-sign columns (net positive, one negative level, horizontally
    uniform) must come back (a) non-negative in every species and (b) with
    the dsigma-weighted column integral of the poisoned species unchanged —
    zero winds + horizontally uniform fields make transport and the SH
    filter identities, so any integral change would be the fixer creating
    mass/number the way a plain clip does."""
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_rollout, spectral_state_to_carry,
    )
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig

    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    dsigma = jnp.asarray(sigma.dsigma)
    carry = _carry(grid.n_lat, grid.n_lon, extras=True)
    # Horizontally uniform, level-varying poison: net-positive columns with
    # one negative level each.
    q_c_prof = jnp.asarray([3.0e-4, -1.0e-4, 2.0e-4, 1.0e-4])
    n_r_prof = jnp.asarray([2.0e3, 1.0e3, -4.0e2, 5.0e2])
    shape = (grid.n_lat, grid.n_lon, 4)
    carry = carry._replace(
        q_c=jnp.broadcast_to(q_c_prof, shape),
        N_r=jnp.broadcast_to(n_r_prof, shape),
    )
    state0 = carry_to_spectral_state(carry, grid)

    def zero_physics(s, g, sc):
        return None

    final = spectral_rollout(
        state0, zero_physics, grid, sigma,
        SpectralPEConfig(semi_implicit=True), 600.0, 2,
        None, None,
    )
    back = spectral_state_to_carry(final, grid, sigma)
    for name in ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        v = getattr(back, name)
        assert v is not None and float(jnp.min(jnp.asarray(v))) >= 0.0, name
    for name, prof in (("q_c", q_c_prof), ("N_r", n_r_prof)):
        col_ref = float(jnp.sum(prof * dsigma))
        col = jnp.sum(jnp.asarray(getattr(back, name)) * dsigma, axis=-1)
        assert jnp.allclose(col, col_ref, rtol=1e-10), name
