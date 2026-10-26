"""#1029: a hybrid coordinate that inverts over the run's own orography.

``B(eta) = eta**transition_exponent`` makes ``dB/deta -> exponent`` at the
surface, so a near-surface layer carries positive mass only above a threshold
surface pressure the coordinate alone fixes.  Below it ``dp_from_hybrid``
returns NEGATIVE thicknesses, and that feeds the dycore.

The condition was known and warned about; nothing passed the argument that
would have made it fatal, so production ran inside it.  Measured with
``scripts/validate/hybrid_negative_layer_mass_exposure.py``: the default L40
coordinate forbids orography above ~3450 m, which is 0.92% of the planet by
area, and on the ``held_suarez_topo`` reproducer two such cells killed a
200-day run inside 200 steps.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.vertical import (assert_hybrid_valid_for_surface_pressure,
                                    hybrid_min_valid_surface_pressure,
                                    make_hybrid_levels,
                                    standard_hybrid_levels)


def _threshold(coord):
    return float(hybrid_min_valid_surface_pressure(
        np.asarray(coord.A_half), np.asarray(coord.B_half), constants.p_ref))


# ----------------------------------------------------------------------
# the thresholds themselves
# ----------------------------------------------------------------------

@pytest.mark.parametrize("nlev", [30, 32, 40])
def test_the_production_coordinate_forbids_real_orography(nlev):
    """The defect, stated as a number rather than as a warning string.

    ~664 hPa is about 3450 m of terrain. The Tibetan Plateau averages above
    4000 m and the Andean altiplano above 3600, so the production coordinate
    is invalid over ground that exists.
    """
    thr = _threshold(standard_hybrid_levels(nlev))
    assert 650e2 < thr < 670e2, f"nlev={nlev}: threshold moved to {thr/100:.1f} hPa"
    z = constants.R_d * 288.0 / constants.g * np.log(constants.p_ref / thr)
    assert 3400.0 < z < 3520.0, f"nlev={nlev}: max orography {z:.0f} m"


@pytest.mark.parametrize("nlev", [30, 32, 40])
def test_the_gentler_exponent_covers_every_elevation_on_earth(nlev):
    """Why exponent=2 is the remedy the error message names.

    ETOPO's highest 1-degree cell is 5855 m; the gentler coordinate admits
    ~5870 m and more. The bound has to hold at every level count, because a
    remedy that only worked at L40 would be a trap at L30.
    """
    thr = _threshold(make_hybrid_levels(nlev, transition_exponent=2,
                                        stretching=2.5))
    z = constants.R_d * 288.0 / constants.g * np.log(constants.p_ref / thr)
    assert z > 5855.0, (
        f"nlev={nlev}: exponent=2 admits only {z:.0f} m, below ETOPO's peak")


def test_a_higher_exponent_is_worse_not_better():
    """Non-vacuity on the direction: the fix has a sign.

    If the threshold did not fall monotonically with the exponent, "use 2"
    would be arbitrary rather than derived.
    """
    thrs = [_threshold(make_hybrid_levels(40, transition_exponent=e,
                                          stretching=2.5))
            for e in (2, 3, 4)]
    assert thrs[0] < thrs[1] < thrs[2], thrs


# ----------------------------------------------------------------------
# the guard
# ----------------------------------------------------------------------

def test_the_guard_refuses_a_run_that_reaches_below_the_threshold():
    coord = standard_hybrid_levels(40)
    thr = _threshold(coord)
    with pytest.raises(ValueError, match="NEGATIVE layer mass"):
        assert_hybrid_valid_for_surface_pressure(coord, thr - 1.0)


def test_the_guard_passes_a_run_that_stays_above_it():
    """Non-vacuity: a guard that refused everything would also pass the test
    above, and would stop every run on the planet."""
    coord = standard_hybrid_levels(40)
    thr = _threshold(coord)
    assert assert_hybrid_valid_for_surface_pressure(coord, thr + 1.0) is None


def test_the_gentler_coordinate_passes_where_production_fails():
    """The remedy has to actually admit the case that refused.

    Same surface pressure, same level count, only the exponent moves -- so
    this is the one-variable statement that exponent=2 is a fix and not just
    a different failure.
    """
    p_s = 550e2          # a Tibetan column
    with pytest.raises(ValueError):
        assert_hybrid_valid_for_surface_pressure(standard_hybrid_levels(40), p_s)
    gentle = make_hybrid_levels(40, transition_exponent=2, stretching=2.5)
    assert assert_hybrid_valid_for_surface_pressure(gentle, p_s) is None


def test_the_error_names_the_elevation_not_only_the_pressure():
    """A pressure is not actionable to a modeller; a height is.

    "663.9 hPa" does not read as "forbids the Tibetan Plateau", and the whole
    reason this sat as a warning for so long is that nobody converted it.
    """
    coord = standard_hybrid_levels(40)
    with pytest.raises(ValueError) as e:
        assert_hybrid_valid_for_surface_pressure(coord, 550e2, context="L40 run")
    msg = str(e.value)
    assert "L40 run" in msg
    assert " m)" in msg, "the message never converts the threshold to a height"
    assert "transition_exponent=2" in msg
    assert "sigma" in msg


def test_a_non_hybrid_coordinate_is_not_second_guessed():
    """sigma has no such threshold; the guard must not invent one."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(40)
    assert assert_hybrid_valid_for_surface_pressure(sigma, 1.0) is None


def test_a_nonfinite_minimum_is_refused_rather_than_silently_passing():
    """A NaN minimum means the geopotential was not built yet.

    ``NaN > threshold`` is False, so an unguarded comparison would take the
    'valid' branch and report success on a run whose orography is unknown --
    the plausible-sentinel failure this repo keeps meeting.
    """
    coord = standard_hybrid_levels(40)
    with pytest.raises(ValueError, match="positive, finite"):
        assert_hybrid_valid_for_surface_pressure(coord, float("nan"))
