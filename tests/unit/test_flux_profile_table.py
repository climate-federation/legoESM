"""Round trip and unit guards of the common flux-profile table."""
import importlib.util
import pathlib

import numpy as np
import pytest

from legoesm import constants

_PATH = pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit/flux_profile_table.py"
_SPEC = importlib.util.spec_from_file_location("flux_profile_table", _PATH)
tb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tb)

KAPPA = constants.kappa_von_karman
ATTRS = dict(site="x", latitude=0.0, longitude=0.0, surface="grass", canopy_height_m=0.0, displacement_m=0.0,
             source="test", licence="none", averaging_minutes=30, flux_kind="native", flux_method="eddy covariance")


def table(tmp_path, temp_offset=0.0, attrs=ATTRS, levels=(2.0, 8.0)):
    n = 96
    t = 1800 * np.arange(1, n + 1)
    z = np.tile(levels, (n, 1))
    ustar = np.full((n, 1), 0.4)
    wind = ustar / KAPPA * np.log(z / 0.01)
    h = 200 * np.clip(np.sin(2 * np.pi * (t / 86400 - 0.25)), 0, None)[:, None]
    temp = 290.0 + temp_offset + np.zeros((n, 2))
    p = tmp_path / "t.nc"
    tb.write_table(p, t, z, wind, z, temp, np.full((n, 1), 4.0), ustar, h, h / 2, np.full(n, 1e5),
                   np.full((n, 1), 290.0), attrs)
    return p


def test_report_recovers_neutral_phi_and_noon_peak(tmp_path):
    row = tb.report(table(tmp_path))["flux_0"]
    assert row["matched"] == 96
    assert row["h_peak_local_hour"] in (11, 12)
    n, phi = row["near_neutral_phi_m"]["2-8"]
    assert n > 0 and phi == pytest.approx(1.0, rel=1e-3)


def test_celsius_is_refused(tmp_path):
    with pytest.raises(ValueError, match="temp median"):
        table(tmp_path, temp_offset=-280.0)


def test_missing_attribute_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing attributes"):
        table(tmp_path, attrs={k: v for k, v in ATTRS.items() if k != "licence"})


def test_vector_speed_is_below_scalar_mean_when_direction_turns():
    assert tb.vector_speed(0.0, 0.0) == 0.0
    assert tb.vector_speed(3.0, 4.0) == 5.0


def test_top_down_levels_are_refused(tmp_path):
    with pytest.raises(ValueError, match="levels must ascend"):
        table(tmp_path, levels=(8.0, 2.0))
