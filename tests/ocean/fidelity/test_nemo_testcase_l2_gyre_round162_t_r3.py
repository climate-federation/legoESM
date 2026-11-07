"""Guards for round 162's T-point ratio substitution and its combined row.

Round 161 exonerated the free-surface ratio at the u and v points and left the
stage-2 vertical-velocity residual unowned.  Round 162 substitutes the last
operand of that continuity solve nothing had installed, the ratio at the
temperature point, and measures the sea surface height round 161 predicted
from arithmetic instead of measuring.  It also runs the measurement that
decides whether the one-step tracer advection error and the vertical-diffusion
error cancel.

These guards are structural and run without a GYRE step.  Two of them carry
the mechanisms the round's claims rest on: that the divergence block really
reads the thickness operand the walk substitutes, and that the operand's
influence is a rounding because the block divides by it and multiplies it back
-- which is why a ratio difference of order 1e-13 could never have owned a
residual that is 1e-07 of its own field.  The third shows that the combined
row's decomposition control is not a tautology: for generic inputs the
identity it checks is NOT exact, so the hard zero the receipt reports is a
result.  The measured numbers are in the round-162 receipt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_transport_wzv_divergence_level,
)

REPO = Path(__file__).resolve().parents[3]
WALK_PATH = (
    REPO / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_year_owners.py")


@pytest.fixture(scope="module")
def walk():
    spec = importlib.util.spec_from_file_location(
        "_gyre_year_owners_r162", WALK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _divergence_operands(ny=4, nx=5):
    rng = np.random.default_rng(1162)
    flux_u = jnp.asarray(rng.normal(0.0, 1.0e6, (ny, nx)))
    flux_u_west = jnp.asarray(rng.normal(0.0, 1.0e6, (ny, nx)))
    flux_v = jnp.asarray(rng.normal(0.0, 1.0e6, (ny, nx)))
    flux_v_south = jnp.asarray(rng.normal(0.0, 1.0e6, (ny, nx)))
    r1_area_t = jnp.asarray(1.0 / rng.uniform(1.0e9, 2.0e9, (ny, nx)))
    live_e3t = jnp.asarray(rng.uniform(40.0, 120.0, (ny, nx)))
    tmask = jnp.asarray(np.ones((ny, nx)))
    return flux_u, flux_u_west, flux_v, flux_v_south, r1_area_t, live_e3t, tmask


def test_the_walk_and_its_plants_exist(walk):
    """The round's plants are named on their walks, and an unknown plant is
    refused rather than silently ignored."""
    assert walk.DEVELOPED_T_R3_PLANTS == ("t-r3-inert", "t-r3-tracer")
    assert walk.DEVELOPED_RANK_SPLIT_PLANTS == ("rank-combined-mispair",)
    assert walk.COMBINED_CANCELLATION_PAIR == (
        "advection", "vertical_diffusion")
    assert callable(walk.developed_stage2_t_r3_walk)
    with pytest.raises(walk.GateError, match="unknown developed t-r3"):
        walk.developed_stage2_t_r3_walk(
            Path("/nonexistent"), Path("/nonexistent"), "0" * 40,
            Path("/nonexistent"), Path("/nonexistent"), plant="not-a-plant")
    with pytest.raises(walk.GateError, match="unknown developed rank-split"):
        walk.developed_process_rank_split(
            Path("/nonexistent"), Path("/nonexistent"), Path("/nonexistent"),
            "0" * 40, Path("/nonexistent"), plant="not-a-plant")


def test_the_divergence_block_reads_the_thickness_the_walk_substitutes():
    """Non-vacuity of the substitution point.  The walk replaces exactly one
    operand of ``divhor``'s velocity arm -- the live temperature-point
    thickness that carries the ratio.  If the block did not read it, the
    substitution would be a control that perturbs a zero."""
    operands = list(_divergence_operands())
    base = np.asarray(nemo_transport_wzv_divergence_level(*operands))
    moved = list(operands)
    moved[5] = jnp.asarray(
        np.asarray(operands[5]) * (1.0 + 1.0e-6))
    after = np.asarray(nemo_transport_wzv_divergence_level(*moved))
    assert not np.array_equal(base, after)


def test_the_thickness_divides_and_multiplies_back_so_it_is_a_rounding():
    """The mechanism the round's first prediction rested on, made mechanical.

    NEMO divides the horizontal divergence by the live thickness at
    ``divhor.f90:130`` and multiplies the same live thickness back at
    ``divhor.f90:153``, so a RELATIVE change in that thickness leaves the
    result unchanged except for rounding.  Perturbing it by a part in a
    million must move the answer by far less than a part in a million -- which
    is why a ratio difference of order 1e-13 cannot own a residual that is
    1e-07 of its own field.
    """
    operands = list(_divergence_operands())
    base = np.asarray(nemo_transport_wzv_divergence_level(*operands))
    moved = list(operands)
    moved[5] = jnp.asarray(np.asarray(operands[5]) * (1.0 + 1.0e-6))
    after = np.asarray(nemo_transport_wzv_divergence_level(*moved))
    scale = float(np.max(np.abs(base)))
    assert scale > 0.0
    relative = float(np.max(np.abs(after - base))) / scale
    assert relative < 1.0e-12, relative
    # And the cancellation is NOT exact, which is why the walk has to measure
    # the substitution rather than argue it away.
    assert relative > 0.0


def test_the_runoff_term_is_the_one_place_the_thickness_does_not_cancel():
    """Registered rather than left implied.  With a river mass flux present
    the block subtracts a term that is divided by the thickness but not
    multiplied back by it, so there the operand keeps first-order influence.
    The cancellation above is therefore a statement about the transport term,
    and this is the one place it does not hold; the receipt says what the
    GYRE card's own runoff is."""
    operands = list(_divergence_operands())
    runoff = jnp.asarray(np.full(np.asarray(operands[5]).shape, 1.0e-4))
    base = np.asarray(nemo_transport_wzv_divergence_level(
        *operands, runoff_mass_flux=runoff))
    moved = list(operands)
    moved[5] = jnp.asarray(np.asarray(operands[5]) * (1.0 + 1.0e-6))
    after = np.asarray(nemo_transport_wzv_divergence_level(
        *moved, runoff_mass_flux=runoff))
    scale = float(np.max(np.abs(base)))
    relative = float(np.max(np.abs(after - base))) / scale
    assert relative > 1.0e-18, relative


def test_the_combined_rows_decomposition_control_is_not_a_tautology():
    """The combined row must decompose into the two rows the ranking already
    scored.  For generic doubles of the same magnitude that identity is NOT
    exact, so the control can fail and the check is worth making."""
    rng = np.random.default_rng(162)
    n = 18000
    first = rng.normal(0.0, 1.0e-2, n)
    first_reference = first + rng.normal(0.0, 1.1e-8, n)
    second = rng.normal(0.0, 1.0e-2, n)
    second_reference = second + rng.normal(0.0, 2.2e-5, n)
    combined = (first + second) - (first_reference + second_reference)
    parts = (first - first_reference) + (second - second_reference)
    assert int(np.count_nonzero(combined - parts)) > n // 4


def test_rows_built_as_temperature_differences_make_that_identity_exact():
    """Why the measured control is a HARD ZERO rather than a few last places.

    Every process row is one temperature boundary minus another, both of them
    a few tens of kelvin, so each row is an exact integer multiple of the last
    place of that temperature -- and so is every sum and difference of them,
    because they never leave that grid.  The identity is then exact by
    construction, which is what the walk measures.  Built the same way here,
    the residual is zero on essentially every cell; built from independent
    doubles of the same magnitude, above, it is not.
    """
    rng = np.random.default_rng(1622)
    n = 18000
    base = rng.uniform(-2.0, 30.0, n)

    def boundary():
        return base + rng.normal(0.0, 1.0e-3, n)

    before, after_first = boundary(), boundary()
    before_second, after_second = boundary(), boundary()
    first = after_first - before
    second = after_second - before_second
    first_reference = ((after_first + rng.normal(0.0, 1.0e-8, n))
                       - (before + rng.normal(0.0, 1.0e-8, n)))
    second_reference = ((after_second + rng.normal(0.0, 2.0e-5, n))
                        - (before_second + rng.normal(0.0, 2.0e-5, n)))
    combined = (first + second) - (first_reference + second_reference)
    parts = (first - first_reference) + (second - second_reference)
    # A handful of cells straddle a binade boundary, where the two
    # temperatures no longer share one last place; the rest are exact.
    assert int(np.count_nonzero(combined - parts)) < n // 100


def test_an_out_of_range_level_index_is_clamped_and_not_raised():
    """Why the walk guards its own per-call level counter.

    JAX does NOT raise on an out-of-range integer index; it CLAMPS to the last
    element.  The first version of this round's substitution carried its level
    counter across calls, so from the second substituted arm onward every
    level was handed the DEEPEST thickness -- and the answer that came back
    was a plausible small number rather than an error.  The walk's own inert
    plant is what caught it.  This pins the behaviour that made it silent, so
    the guard is never removed as unnecessary.
    """
    column = jnp.arange(30.0).reshape(1, 1, 30)
    assert float(np.asarray(column[0, 0, 35])) == 29.0
    assert float(np.asarray(column[0, 0, 29])) == 29.0
    # The same silence under the trailing-axis form the walk actually writes.
    assert np.array_equal(
        np.asarray(column[..., 35]), np.asarray(column[..., 29]))
