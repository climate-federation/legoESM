"""DCMIP 2012 test 3-1 (``gravity_wave_3_1``) matrix lanes: the small-planet
radius, the X-scaled step / diffusion and the App. F.3 L10 table reach each
lane's model, and the row's diagnostics land on the document's times.
The cube (FV3 duo) lane is covered in ``test_cube_duo_shared_ic.py``.
No stepping beyond construction.  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def mm():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_gw31", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("dt, k_expect", [(15.0, 5), (1.6, 25), (4.8, 5)])
def test_diag_cadence_hits_the_document_times(mm, dt, k_expect):
    """cube 15 s, latlon/ico 1.6 s, spectral 4.8 s"""
    k = mm._gw31_diag_every(dt)
    assert k == k_expect
    assert k * dt <= 100.0 + 1e-9
    for t in (3000.0, 3600.0):
        n = t / (k * dt)
        assert abs(n - round(n)) < 1e-9, (dt, k, t)
    for bad in (7.0, 4.800000000001, 6.8965517241379315):
        with pytest.raises(ValueError, match="does not divide 600"):
            mm._gw31_diag_every(bad)
    with pytest.raises(ValueError, match="not in"):
        mm._gw31_diag_every(120.0)


def test_step_count_lands_on_the_run_end(mm):
    days = 3600.0 / 86400.0
    assert mm._gw31_n_steps(days, 15.0) == 240
    assert mm._gw31_n_steps(days, 1.6) == 2250
    assert mm._gw31_n_steps(days, 4.8) == 750
    dt = 600.0 / 673       # int(3600 / dt) == 4037: one step short
    assert int(days * 86400 / dt) == 4037
    assert mm._gw31_n_steps(days, dt) == 4038
    with pytest.raises(ValueError, match="does not divide"):
        mm._gw31_n_steps(days, 7.0)


class _Stop(Exception):
    pass


def _loop_args(mm, monkeypatch, tmp_path, grid, res):
    """run_baroclinic on the 3-1 row with the time loop stubbed: the (dt,
    n_steps, diag_every) it would integrate with."""
    seen = {}

    def fake(step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
             diag_every, key_array_fn, **kw):
        seen.update(dt=dt, n_steps=n_steps, diag_every=diag_every)
        raise _Stop

    monkeypatch.setattr(mm, "_run_timeloop", fake)
    tc = mm.TestCase("hydrostatic", "gravity_wave_3_1", grid, res, "hybrid",
                     3600.0 / 86400.0, 3600.0 / 86400.0, {})
    with pytest.raises(_Stop):
        mm.run_baroclinic(tc, tmp_path, tc.quick_days)
    return seen


def test_cube_lane_integrates_to_3600_s(mm, monkeypatch, tmp_path):
    s = _loop_args(mm, monkeypatch, tmp_path, "cubed_sphere", "C36")
    assert s == {"dt": 15.0, "n_steps": 240, "diag_every": 5}
