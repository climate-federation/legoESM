"""Spectral (T21) lane of ``run_baroclinic``: the DCMIP-2008 dry rows run the
reference EUL dissipation of their own document (DCMIP 2008 Table 4 del4 K4,
no exponential filter); every other row keeps the lane default.

The lane's per-step exp filter (order 8, 0.01 retention, per 600 s step)
e-folds n = 14 in ~55 min at T21 and took ~25 % off the 5-0-0 / 6-3-0
response vs the cube / lat-lon / ico rows.  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def mm():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_specdiss", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


class _Stop(Exception):
    pass


def _lane(mm, monkeypatch, tmp_path, case, kwargs=None):
    """Run the spectral branch of run_baroclinic up to the time loop; return
    the model config, the IC, step_fn and dt it would integrate with."""
    import legoesm.atmosphere.dynamics.gcm.spectral_pe as spe
    seen = {}
    real = spe.SpectralPrimitiveEquationModel

    class Spy(real):
        def __init__(self, grid, sigma, config, *a, **k):
            seen["config"] = config
            super().__init__(grid, sigma, config, *a, **k)

    def fake(step_fn, state, dt, n_steps, *a, **k):
        seen.update(step_fn=step_fn, state=state, dt=dt)
        raise _Stop

    monkeypatch.setattr(spe, "SpectralPrimitiveEquationModel", Spy)
    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", case, "spectral", "T21", "hybrid",
                     2.0, 0.5, dict(kwargs or {}))
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    return seen


@pytest.mark.parametrize("case", sorted(
    ["inertio_gravity_3_2", "mountain_rossby_5_0", "rossby_haurwitz_6_0"]))
def test_dcmip2008_rows_run_table4_eul_dissipation(mm, monkeypatch, tmp_path,
                                                  case):
    cfg = _lane(mm, monkeypatch, tmp_path, case)["config"]
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.hyperdiff_coeff == 2.0e16          # Table 4, T21
    assert cfg.hyperdiff_order == 2               # del4
    assert cfg.dealiasing_fraction == 0.667       # 2/3 band untouched


@pytest.mark.parametrize("case,kwargs,hd", [
    ("baroclinic", {}, 2.338e15),
    ("rotated_steady", {"alpha": 0.7853981633974483, "perturbed": False},
     2.338e15),
    ("rest_state_topo", {"h_0": 2000.0}, 2.338e15),
    ("gravity_wave_3_1", {}, 2.338e15 / 125.0 ** 3),
])
def test_other_rows_keep_the_lane_default(mm, monkeypatch, tmp_path,
                                          case, kwargs, hd):
    cfg = _lane(mm, monkeypatch, tmp_path, case, kwargs)["config"]
    assert cfg.spectral_filter_strength == 0.01
    assert cfg.spectral_filter_order == 8
    assert cfg.hyperdiff_coeff == pytest.approx(hd, rel=1e-15)


@pytest.mark.parametrize("n_max,k4", [
    (21, 2.0e16), (42, 1.0e16), (85, 1.0e15), (106, 0.5e15),
    (170, 1.5e14), (340, 1.5e13)])
def test_table4_lookup_every_truncation(mm, n_max, k4):
    """DCMIP 2008 Sect. 2.1 Table 4, all six EUL rows."""
    hd, filt, note = mm._spectral_pe_dissipation("mountain_rossby_5_0", n_max)
    assert (hd, filt) == (k4, 0.0)
    assert "Table 4" in note


def test_table4_refuses_an_unlisted_truncation(mm):
    with pytest.raises(ValueError, match="Table 4"):
        mm._spectral_pe_dissipation("mountain_rossby_5_0", 63)
    assert mm._spectral_pe_dissipation("baroclinic", 63)[1] == 0.01


def test_ig_6_3_0_response_reaches_the_other_grids(mm, monkeypatch, tmp_path):
    """6-3-0 at T21 through the matrix lane: interior (below the top level)
    max |V| at 0.25 d.  Lane default (exp filter) gives 4.4 m/s; the
    reference dissipation 6.0; cube / lat-lon / ico peak 6.4 / 6.4 / 6.5."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    s = _lane(mm, monkeypatch, tmp_path, "inertio_gravity_3_2")
    assert s["dt"] == 600.0
    model_state = s["state"]
    for _ in range(36):                           # 0.25 d
        model_state = s["step_fn"](model_state, s["dt"])
    from legoesm.grids.gaussian import create_gaussian_grid
    grid = create_gaussian_grid(21)
    sigma = mm._create_vertical(mm.DEFAULT_NLEV, "hybrid")
    f = spectral_pe_to_grid(model_state, grid, sigma)
    speed = np.sqrt(np.asarray(f["u"]) ** 2 + np.asarray(f["v"]) ** 2)
    interior = float(speed[..., 1:].max())
    assert 5.5 < interior < 6.8, interior
