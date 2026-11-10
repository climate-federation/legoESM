"""A viscosity schedule must not rescale the A_h profile it rides on.

``--A-h-profile-file`` reads NEMO's ``eddy_viscosity_3D.nc`` and stores it as
RATIOS to the A_h in force, because the config field has to be a hashable
static tuple.  The Laplacian then applies the PRODUCT ``A_h * profile``.  So
``--visc-schedule`` replacing A_h alone multiplies the whole profile by
``A_h_new / A_h_old``: a schedule of 1e4 against a profile normalised at 2e4
applied HALF of the viscosity the file named, for the whole segment, and the
log line printed the knob rather than the applied value so nothing said so.

The invariant under test is therefore not "A_h changed" but "the product the
Laplacian actually applies is unchanged by the renormalisation" -- and the
vacuity test below asserts the OLD expression violates it, so this file goes
red if the fix is reverted.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.state import LatLonCGridOceanConfig
from scripts.run.run_omip_core2 import renormalise_ah_profile

# NEMO ORCA1's eddy_viscosity_3D spans 1000 m2/s at the equator to 20000 in
# midlatitudes, i.e. 0.05-1.00 once normalised at the 2e4 the campaign passes
# as --A-h.  Three rows is enough to catch a scalar-vs-elementwise slip.
_PROFILE = (0.05, 0.5, 1.0)
_BASE = 20000.0
_SCHEDULED = 10000.0


def _cfg(profile=_PROFILE, profile_v=None, A_h=_BASE):
    return LatLonCGridOceanConfig().replace_flat(
        A_h=A_h, A_h_lat_profile=profile,
        A_h_lat_profile_v=profile_v).lateral_viscosity


def _applied(lv):
    """What the Laplacian multiplies the velocity curvature by, per row."""
    return [float(lv.A_h) * float(p) for p in lv.A_h_lat_profile]


def test_scheduled_A_h_change_leaves_the_applied_viscosity_alone():
    lv = _cfg()
    before = _applied(lv)
    assert before == pytest.approx([1000.0, 10000.0, 20000.0])

    out = renormalise_ah_profile(lv, _SCHEDULED)

    assert float(out.A_h) == _SCHEDULED
    assert _applied(out) == pytest.approx(before, rel=1e-12)


def test_reverting_the_fix_halves_it():
    """Non-vacuity: the pre-fix expression must fail the same assertion."""
    lv = _cfg()
    naive = lv._replace(A_h=_SCHEDULED)          # what the schedule used to do
    assert _applied(naive) == pytest.approx(
        [0.5 * v for v in _applied(lv)], rel=1e-12)
    with pytest.raises(AssertionError):
        assert _applied(naive) == pytest.approx(_applied(lv), rel=1e-12)


def test_v_face_profile_is_rescaled_too_when_present():
    # The SPMD wrapper injects an explicit v-face profile; rescaling only the
    # u-face one would leave the two faces on different viscosities.
    prof_v = (0.05, 0.275, 0.75, 1.0)
    out = renormalise_ah_profile(_cfg(profile_v=prof_v), _SCHEDULED)
    assert out.A_h_lat_profile_v == pytest.approx(
        tuple(2.0 * p for p in prof_v), rel=1e-12)


def test_zero_A_h_with_a_profile_is_refused_not_a_zero_division():
    # codex: --visc-schedule accepts A_h=0, which has no finite
    # renormalisation.  It must fail at the segment boundary with a sentence,
    # not a ZeroDivisionError eight days into an eleven-hour job.
    with pytest.raises(SystemExit, match="contradictory"):
        renormalise_ah_profile(_cfg(), 0.0)
    # Without a profile, A_h=0 is a legitimate "no lateral viscosity" run.
    assert renormalise_ah_profile(_cfg(profile=None), 0.0).A_h == 0.0


def test_absent_profile_and_no_op_change_are_untouched():
    lv_none = _cfg(profile=None)
    assert renormalise_ah_profile(lv_none, _SCHEDULED).A_h_lat_profile is None
    assert renormalise_ah_profile(lv_none, _SCHEDULED).A_h == _SCHEDULED
    # Same A_h: the profile must not drift through a repeated segment boundary.
    same = renormalise_ah_profile(_cfg(), _BASE)
    assert same.A_h_lat_profile == pytest.approx(_PROFILE, rel=1e-12)
    assert same.A_h_lat_profile_v is None
