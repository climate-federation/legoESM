"""Direct test of scripts/validate/land_snow_compaction_check.py's CLM5 rates."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from legoesm import constants

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "land_snow_compaction_check.py"


def _load():
    sys.path.insert(0, str(_PATH.parent))
    spec = importlib.util.spec_from_file_location("land_snow_compaction_check", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clm5_rates_match_hand_values():
    m = _load()
    TF = constants.T_freeze
    one = np.ones(1)
    # dry snow at T_freeze below the 175 kg/m3 limit: metamorphism = -c3 exactly
    cr1, cr2 = m.clm5_rates(150.0 * one, TF * one, 0.0 * one, 10.0 * one, 0.0 * one, one)
    np.testing.assert_allclose(cr1, -2.777e-6)
    # 10 K colder: x exp(-0.4); 100 kg/m3 above the limit: x exp(-4.6)
    cr1c, _ = m.clm5_rates(275.0 * one, (TF - 10.0) * one, 0.0 * one, 10.0 * one,
                           0.0 * one, one)
    np.testing.assert_allclose(cr1c, -2.777e-6 * np.exp(-0.4) * np.exp(-4.6))
    # overburden: compaction (negative), proportional to burden + w/2
    _, a = m.clm5_rates(200.0 * one, TF * one, 10.0 * one, 10.0 * one, 0.0 * one, one)
    _, b = m.clm5_rates(200.0 * one, TF * one, 25.0 * one, 10.0 * one, 0.0 * one, one)
    assert a[0] < 0.0 and np.isclose(b[0] / a[0], 30.0 / 15.0)
