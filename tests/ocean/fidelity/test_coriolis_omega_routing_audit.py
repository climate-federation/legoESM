"""Tests for the Coriolis rotation-rate routing audit (#1455 next-action 1).

The audit's whole job is to say WHICH Earth each Coriolis consumer received,
so the two ways it could lie are (a) recovering a rotation rate that is not
the one the array was built with, and (b) calling two different rates the
same Earth.  Every test below makes one of those wrong answers impossible,
against a case whose answer is known without running the code under test.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG

_PROBE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "validate",
    "ocean_fidelity", "dino_1226", "coriolis_omega_routing_audit.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "coriolis_omega_routing_audit", os.path.abspath(_PROBE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()

# The two rates are IMPORTED, never re-typed.  Typing them as literals put this
# file in the hardcoded-constant ratchet, and worse, it would let the test keep
# passing against a stale copy of a constant the production code had moved --
# which is the exact class of drift this whole action was about.
_OM_LEGO = float(constants.Omega)
_OM_NEMO = float(NEMO_CONSTANTS_CONFIG.Omega)


# --------------------------------------------------------------------------
# the inversion
# --------------------------------------------------------------------------
def test_recovers_a_known_rotation_rate_exactly():
    """f built from a known omega must invert back to that omega."""
    lat = np.deg2rad(np.linspace(-70.0, 70.0, 141))
    f = 2.0 * _OM_NEMO * np.sin(lat)[:, None] * np.ones((1, 4))
    got = M._omega_eff(f, lat, name="synthetic", lat_name="synthetic lat")
    assert abs(got["omega_median"] - _OM_NEMO) < 1e-15 * _OM_NEMO
    # A pure constant must show machine-level spread across rows; a nonzero
    # spread is exactly how a placement error announces itself.
    assert got["omega_spread_rel"] < 1e-12


def test_row_average_on_a_UNIFORM_grid_is_indistinguishable_from_a_slow_earth():
    """The trap this audit has to survive, with a hand-computable answer.

    On a grid with constant latitude spacing, averaging f from the two
    neighbouring cell centres onto the face gives exactly
    ``cos(dphi/2)`` times the face value -- a CONSTANT factor.  So on a
    uniform grid a placement error and a rotation-rate error are the same
    number and no inversion can separate them.  DINO's Mercator grid is not
    uniform (next test), which is the only reason the split is possible at
    all; this test pins the boundary of what the audit can claim.
    """
    dphi = np.deg2rad(1.0)
    lat_c = np.deg2rad(np.arange(-70.0, 70.01, 1.0))
    lat_f = 0.5 * (lat_c[:-1] + lat_c[1:])
    f_c = 2.0 * _OM_NEMO * np.sin(lat_c)
    f_face_avg = 0.5 * (f_c[:-1] + f_c[1:])
    got = M._omega_eff(f_face_avg, lat_f, name="row average (uniform)",
                       lat_name="true face lat")
    assert got["omega_spread_rel"] < 1e-12          # a clean constant
    assert got["omega_median"] == pytest.approx(_OM_NEMO * np.cos(dphi / 2.0),
                                                rel=1e-13)


def test_row_average_on_a_STRETCHED_grid_recovers_a_LATITUDE_VARYING_rate():
    """On DINO's stretched (Mercator) spacing the same construction gives a
    recovered rate that VARIES with latitude -- which is what makes the
    constant half and the placement half separable in the real audit.  If
    this read as a clean constant the audit's split would be vacuous."""
    # Mercator: uniform in the Mercator coordinate, stretched in latitude.
    y = np.linspace(-2.0, 2.0, 141)
    lat_c = 2.0 * np.arctan(np.exp(y)) - np.pi / 2.0
    lat_f = 0.5 * (lat_c[:-1] + lat_c[1:])
    f_c = 2.0 * _OM_NEMO * np.sin(lat_c)
    f_face_avg = 0.5 * (f_c[:-1] + f_c[1:])
    got = M._omega_eff(f_face_avg, lat_f, name="row average (mercator)",
                       lat_name="true face lat")
    assert got["omega_median"] < _OM_NEMO           # sin is concave off-equator
    assert got["omega_spread_rel"] > 1e-6


def test_equatorial_rows_are_dropped_not_averaged_in():
    """The inversion is ill-conditioned at the equator; those rows must go."""
    lat = np.deg2rad(np.array([-30.0, -1.0, 0.0, 1.0, 30.0]))
    f = 2.0 * _OM_NEMO * np.sin(lat)
    got = M._omega_eff(f, lat, name="synthetic", lat_name="synthetic lat")
    assert got["rows_scored"] == 2
    assert got["rows_dropped_equatorial"] == 3


def test_a_row_offset_is_a_hard_error_not_a_silent_reduction():
    lat = np.deg2rad(np.linspace(-70.0, 70.0, 141))
    f = 2.0 * _OM_NEMO * np.sin(lat)
    with pytest.raises(SystemExit):
        M._omega_eff(f[:-1], lat, name="short", lat_name="lat")


def test_zonal_structure_is_measured_not_assumed():
    lat = np.deg2rad(np.linspace(-70.0, 70.0, 141))
    f = 2.0 * _OM_NEMO * np.sin(lat)[:, None] * np.ones((1, 4))
    f[:, 2] *= 1.5
    got = M._omega_eff(f, lat, name="i-varying", lat_name="lat")
    assert got["zonal_spread_abs"] > 0.0


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------
def test_classifier_separates_the_two_earths():
    assert M._classify(_OM_LEGO, _OM_LEGO, _OM_NEMO) == "legoESM constants.Omega"
    # The NEMO label must name PHYCST, not "the card's pin": once the pin was
    # corrected to the sidereal-day value the two references coincide, and the
    # classifier resolves most-specific-first so the true provenance wins.
    assert "phycst" in M._classify(_OM_NEMO, _OM_LEGO, _OM_NEMO)


def test_classifier_names_a_STALE_card_pin_as_a_pin_not_as_NEMOs_omega():
    """The defect this audit exists to surface, in the classifier itself.

    NEMO computes omega = 2*pi/rsiday except under key_cice, where it is the
    literal 7.292116e-05.  DINO is not key_cice.  While the preset carried the
    key_cice literal, calling it "NEMO's omega" would have erased the
    discrepancy -- so a value matching the pin but NOT phycst must be named as
    a pin.  Exercised with the historical literal so the guard keeps working
    if a future preset regresses to it.
    """
    stale_pin = 7.292116e-05          # const-ok: the historical key_cice literal, the thing under test
    phycst = 7.2921150830e-05         # const-ok: NEMO phycst.F90 :91, the reference it must be told apart from
    got = M._classify(stale_pin, _OM_LEGO, stale_pin, phycst)
    assert "pin" in got and "phycst" not in got
    assert M._classify(phycst, _OM_LEGO, stale_pin, phycst) != got


def test_classifier_refuses_to_name_a_third_rate():
    """The row-averaged f_v inverts to neither Earth; calling it one would
    erase the finding this audit exists to make.  The stand-in is DERIVED as a
    rate a few parts per million below legoESM's, which is the size of the
    real placement contamination -- not a pasted measurement."""
    third = _OM_LEGO * (1.0 - 3.7e-05)
    assert M._classify(third, _OM_LEGO, _OM_NEMO) == "NEITHER (unrecognised)"


def test_classifier_tolerance_is_tight_enough_to_see_the_gap_under_audit():
    """The two Earths differ by 1.6e-05 relative.  A classifier whose
    tolerance swallowed that would report one Earth where there are two."""
    assert abs(_OM_LEGO - _OM_NEMO) / _OM_NEMO > 1e-5
    assert M._classify(_OM_LEGO, _OM_LEGO, _OM_NEMO) != \
        M._classify(_OM_NEMO, _OM_LEGO, _OM_NEMO)
