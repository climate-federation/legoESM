"""Pure helpers of the snow-insulation A/B scorer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias"
           / "snowheat_ab_score.py")


def _mod():
    spec = importlib.util.spec_from_file_location("snowheat_ab_score", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_checkpoint_days_are_jan1_offsets():
    m = _mod()
    days = m.checkpoint_days(2001, "2001-12-11", "2001-12-31")
    assert days[0] == 344 and days[-1] == 364 and len(days) == 21
    assert m.checkpoint_days(2001, "2001-11-01", "2001-11-30") == list(range(304, 334))


def test_at_depth_linear_between_bracketing_nodes():
    m = _mod()
    z = np.array([0.01, 0.136, 0.276, 0.5])
    T = np.array([[0.0, 10.0, 24.0, 0.0]])
    assert m.at_depth(T, z, 0.175)[0] == pytest.approx(10.0 + 14.0 * (0.175 - 0.136) / 0.14)
    with pytest.raises(SystemExit):
        m.at_depth(T, z, 0.6)


def test_overlap_clips_layers_at_depth():
    m = _mod()
    np.testing.assert_allclose(m.overlap(np.array([0.3, 0.5, 0.4, 1.0]), 1.0), [0.3, 0.5, 0.2, 0.0])


def test_wmean_weights_and_fatal_nonfinite():
    m = _mod()
    assert m.wmean(np.array([1.0, 3.0, np.nan]), np.array([1.0, 3.0, 0.0])) == 2.5
    with pytest.raises(SystemExit):
        m.wmean(np.array([1.0, np.nan]), np.array([1.0, 1.0]))


def test_heat_from_freezing_integrates_a_varying_capacity():
    m = _mod()
    from legoesm import constants
    Tf = constants.T_freeze
    cap = lambda T: 2.0 + 0.1 * (T - Tf) ** 2           # analytic: 2x + x^3/30
    T = np.array([[Tf - 10.0, Tf + 10.0]])
    np.testing.assert_allclose(m.heat_from_freezing(cap, T, Tf),
                               [[2 * -10 + (-10) ** 3 / 30, 2 * 10 + 10 ** 3 / 30]], rtol=1e-12)


def test_flat_config_dotted_keys(tmp_path):
    m = _mod()
    (tmp_path / "c.json").write_text('{"a": 1, "output": {"output_dir": "x", "n": {"k": true}}}')
    assert m.flat_config(tmp_path / "c.json") == {"a": 1, "output.output_dir": "x", "output.n.k": True}


def test_flat_config_missing_differs_from_null(tmp_path):
    m = _mod()
    (tmp_path / "a.json").write_text('{"x": null}')
    (tmp_path / "b.json").write_text('{}')
    a, b = m.flat_config(tmp_path / "a.json"), m.flat_config(tmp_path / "b.json")
    assert m.config_diff(a, b) == {"x"}
    assert m.config_diff({"y": 1, "x": None}, {"y": 1, "x": None}) == set()
