"""Direct tests of the snow soot/dust probe's own logic (no data, no compiler)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_PY = (pathlib.Path(__file__).resolve().parents[2]
       / "scripts" / "validate" / "probe_snow_albedo_snicar.py")


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("probe_snow_albedo_snicar", _PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _res(n=5):
    clean = {b: np.full(n, 0.9) for b in ("dir_vis", "dir_nir", "dif_vis", "dif_nir", "bb")}
    dirty = {b: v - 0.02 for b, v in clean.items()}
    return {"clean": clean, "dirty": dirty}


def test_check_results_accepts_valid(probe):
    probe.check_results(_res(), np.linspace(60.0, 300.0, 5), np.full(5, 0.8))


@pytest.mark.parametrize("plant", ["nan", "brighter", "radius", "range", "ours_nan"])
def test_check_results_rejects_planted(probe, plant):
    res, r, ours = _res(), np.linspace(60.0, 300.0, 5), np.full(5, 0.8)
    if plant == "nan":
        res["dirty"]["dir_vis"][2] = np.nan
    elif plant == "brighter":
        res["dirty"]["dif_nir"][1] = 0.95
    elif plant == "radius":
        r[3] = 50.0
    elif plant == "range":
        res["clean"]["bb"][0] = 1.2
    else:
        ours[4] = np.nan
    with pytest.raises(SystemExit):
        probe.check_results(res, r, ours)


def test_layer_temps_linear_profile(probe):
    swe_top, swe_low, g, Ts = 5.0, 120.0, 20.0, 255.0
    t1, t2, tsoil = probe.layer_temps(Ts, g, swe_top, swe_low)
    dz1, dz2 = swe_top / probe.RHO_SNOW, swe_low / probe.RHO_SNOW
    assert abs(t1 - (Ts + g * dz1 / 2)) < 1e-12 and tsoil == t2
    # CLM's interpolated top/lower interface equals the linear-profile interface
    t_if = (t2 * dz1 + t1 * dz2) / (dz1 + dz2)
    assert abs(t_if - (Ts + g * dz1)) < 1e-9
    # and CLM therefore diagnoses half the gradient in the top layer
    assert abs(probe.clm_top_gradient(t1, t2, swe_top, swe_low) - g / 2) < 1e-9


def test_mam_selection(probe):
    # noleap month starts in 1979 (days since 1850-01-01)
    starts = (1979 - 1850) * 365.0 + np.r_[0.0, probe.MONTH_END[:-1]]
    sel = probe._mam(starts, 1850)
    assert np.flatnonzero(sel).tolist() == [2, 3, 4]
