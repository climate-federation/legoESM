"""scripts/validate/les_suite/cloudbench_rad_{resolution,validate}.py and
cloudbench_dt_check.py: the probes' own arithmetic (their LES-store and scoring
paths need the data and run on compute nodes)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm import constants

_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "les_suite"


def _load(name):
    sys.path.insert(0, str(_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_DIR))
    return mod


def test_heating_from_net_flux_closes_on_the_boundary_fluxes():
    rr = _load("cloudbench_rad_resolution")
    p_half = np.array([5.0e4, 6.0e4, 8.0e4, 1.0e5])
    net = np.array([-60.0, -20.0, 5.0, 40.0])            # top-first net downward
    h = rr.heating_from_netdown(net, p_half)
    np.testing.assert_allclose(h, constants.g / constants.c_pd * np.array(
        [-40.0 / 1.0e4, -25.0 / 2.0e4, -35.0 / 2.0e4]), rtol=1e-14)
    col = constants.c_pd / constants.g * np.sum(np.diff(p_half) * h)
    assert col == pytest.approx(net[0] - net[-1], rel=1e-14)


def test_swirl_heating_is_its_forward_flux_over_central_pressure_difference():
    rr = _load("cloudbench_rad_resolution")
    p = np.array([1.0e5, 9.9e4, 9.6e4, 9.5e4])            # unequal spacing
    f = np.array([10.0, 12.0, 15.0, 19.0])                 # net up - down, rising
    h = rr.swirl_heating(f, p)
    want = np.array([9.81 * 3.0 / (0.5 * (9.6e4 - 1.0e5)),
                     9.81 * 4.0 / (0.5 * (9.5e4 - 9.9e4))]) / (1.4 * 286.69 / 0.4)
    np.testing.assert_allclose(h, want, rtol=1e-14)
    assert np.all(h < 0.0)                                 # more escaping upward: cooling


def test_onto_refuses_heights_outside_the_profile():
    rr = _load("cloudbench_rad_resolution")
    z = np.array([300.0, 200.0, 100.0, 0.0])               # top-first
    f = np.array([4.0, 3.0, 2.0, 1.0])
    assert rr.onto(z, f, [150.0])[0] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="outside"):
        rr.onto(z, f, [301.0])


def test_layer_error_is_pressure_weighted():
    rr = _load("cloudbench_rad_resolution")
    p_half = np.array([5.0e4, 6.0e4, 1.0e5])               # thicknesses 1e4, 4e4
    e = rr.layer_err(np.array([3.0, 1.0]) / 86400, np.zeros(2), p_half)
    assert e["rms_K_day"] == pytest.approx(np.sqrt((1e4 * 9 + 4e4 * 1) / 5e4))
    assert e["max_abs_K_day"] == pytest.approx(3.0) and e["k_max_topfirst"] == 0


def test_validate_layer_mean_and_vapour():
    rv = _load("cloudbench_rad_validate")
    ph = np.array([1.0e5, 9.0e4, 8.0e4])                   # surface-first interfaces
    pf = np.array([9.5e4, 8.5e4])
    x = np.array([10.0, 20.0])                             # surface-first, linear in p
    got = rv.layer_mean(x, ph, pf, np.array([8.0e4, 9.0e4, 1.0e5]))   # top-first layers
    # the layer means of the piecewise-linear profile (constant beyond the end levels)
    np.testing.assert_allclose(got, [18.75, 11.25], rtol=0.02)
    assert got[0] > got[1]                                 # top-first
    _, qv, _ = rv.thermo(np.full(2, 300.0), np.full(2, 0.01), np.full(2, 1e-3),
                         np.full(2, 5e-4), pf)
    np.testing.assert_allclose(qv, 0.009)                  # q_t - q_c, the LES's vapour


def _scores(ysu, cand):
    return {"dt": 0.0, "ysu_tuned": {f"c{i}": {"combined": v} for i, v in enumerate(ysu)},
            "candidate_scores": {f"c{i}": {"combined": v} for i, v in enumerate(cand)}}


def test_dt_agreement_rule():
    dc = _load("cloudbench_dt_check")
    ref = _scores([2.0, 2.0], [1.9, 2.1])
    assert dc.agreement(ref, _scores([2.03, 1.98], [1.92, 2.12]))["agree"]
    out = dc.agreement(ref, _scores([2.05, 2.0], [1.9, 2.1]))      # 2.5 % change
    assert not out["agree"] and not out["cases"]["c0"]["agree"] and out["cases"]["c1"]["agree"]
    assert not dc.agreement(ref, _scores([2.0, 2.0], [2.01, 2.1]))["agree"]   # sign flip
    bad = _scores([2.0, 2.0], [1.9, 2.1])
    bad["candidate_scores"]["c1"] = {"failed": "gate"}
    assert not dc.agreement(ref, bad)["agree"]
    bad["candidate_scores"]["c1"] = {"combined": float("nan")}
    assert not dc.agreement(ref, bad)["agree"]


def test_dt_check_refuses_without_x64(monkeypatch):
    dc = _load("cloudbench_dt_check")
    monkeypatch.setattr(jax.config, "read", lambda k: False)
    monkeypatch.setattr(sys, "argv", ["x", "out.json", "300", "c.py"])
    with pytest.raises(SystemExit, match="X64"):
        dc.main()
