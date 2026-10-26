"""The heat-transport line invited a comparison it cannot support.

_mht_diag printed "MHT NH peak = X PW (NH obs ~1.8 PW)". The peak is the
maximum over all northern rows of an INSTANTANEOUS section, and on this grid
it lands at the equator: 18.74 PW at 1N on the day-30 ORCA1 state, where
+1010 Sv northward and -1011 Sv southward nearly cancel and the residual meets
a ~4.5 K contrast between the limbs. The observed ~1.8 PW is a multi-year mean
at a fixed latitude, so putting the two on one line is a category error.

What was checked and is NOT the explanation, each measured rather than argued:
  * the arithmetic and the v-face geometry -- a probe reproduced 18.735 PW
    against the reported 18.7386 from the model's own stored mass_flux_v;
  * the degC reference -- the premise of reference-independence IS violated
    (max net transport 35 Sv per row) but sensitivity is only 1.44 PW;
  * grid-scale noise -- adjacent-longitude sign changes are 24.5% at the peak
    row against 15-22% on three control rows, i.e. not distinguishable from
    the normal field;
  * the tripole fold -- the peak row's latitude spread along the row is 0.00
    degrees, so it is a genuine latitude circle.

So the fix is to REPORT HONESTLY, not to change the number: add 26N, where an
observational value exists, and label the peak as instantaneous.
"""
from __future__ import annotations

import pathlib

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _diag_src():
    src = RUNNER.read_text()
    i = src.index("def _mht_diag(")
    return src[i:src.index("\ndef ", i + 10)]


def test_26n_is_reported():
    """A latitude where an observational number actually exists."""
    body = _diag_src()
    assert "mht_26n_PW" in body
    assert "26.0" in body, "the 26N row must be selected, not hard-coded blind"


def test_the_peak_no_longer_claims_an_observational_comparison():
    """The old line read 'NH peak = X PW (NH obs ~1.8 PW)'. That pairing is
    what made an instantaneous equatorial spike look like a skill score."""
    body = _diag_src()
    assert "NH obs ~1.8 PW" not in body
    assert "INSTANTANEOUS" in body


def test_the_peak_latitude_is_recorded():
    """Without it a reader cannot tell that the maximum sits at the equator,
    which is the whole reason the number is not comparable."""
    assert "mht_nh_peak_lat" in _diag_src()


def test_the_docstring_states_what_was_ruled_out():
    """Otherwise the next reader re-runs the same four eliminations."""
    doc = _diag_src()
    for phrase in ("reproduced", "grid-scale noise", "1.44 PW", "1010 Sv"):
        assert phrase in doc, f"missing the {phrase!r} finding"


def test_the_curve_and_latitudes_are_still_available():
    """The 26N pick reads mht_PW/lat_deg out of the reducer's dict; if those
    keys ever go away the new line silently becomes NaN."""
    import inspect

    from legoesm.ocean.spinup import _mht_peaks
    src = inspect.getsource(_mht_peaks)
    assert '"mht_PW"' in src and '"lat_deg"' in src
