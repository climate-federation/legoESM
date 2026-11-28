"""Spectral colliding-modons matrix row runs T63 with the 2/3 mask
(n_cut = 42).  Vote 2026-11-28 (codex T63 + mask, author T63 + mask; GLM
unavailable -> single review): at the paper's r0 = 500 km the T42 mask
(n_cut = 28) cut the filtered IC max |V| from 34.8 to 27.6 m/s, below the
other grids' 34.3-36.1 m/s band; maskless runs are unstable (T85 NaN)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import jax
import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def matrix_mod():
    spec = importlib.util.spec_from_file_location("_atmmatrix_modon_t63",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _spectral_row(matrix_mod):
    rows = [t for t in matrix_mod._build_test_matrix()
            if t.case == "colliding_modons" and t.grid_type == "spectral"]
    assert len(rows) == 1
    return rows[0]


class _Captured(Exception):
    pass


def _runner_ic(matrix_mod, tmp_path, monkeypatch):
    """The spectral row's initial state and grid exactly as the runner
    builds them (time loop stubbed out)."""
    jax.config.update("jax_enable_x64", True)
    got = {}

    def _stop(step_fn, state, dt, n_steps, check_fn, scalar_fn, *a, **k):
        got["state"], got["scalar_fn"], got["dt"] = state, scalar_fn, dt
        raise _Captured
    monkeypatch.setattr(matrix_mod, "_run_timeloop", _stop)
    with pytest.raises(_Captured):
        matrix_mod.run_shallow_water(_spectral_row(matrix_mod), tmp_path,
                                     days=1.0)
    return got


def test_row_is_t63_with_mask(matrix_mod, tmp_path, monkeypatch):
    """Row resolution T63, and the runner's IC carries the 2/3 mask:
    no power above n_cut = floor(0.667 * 63) = 42, power at n = 42."""
    row = _spectral_row(matrix_mod)
    assert row.resolution == "T63"
    from legoesm.grids.gaussian import create_gaussian_grid
    grid = create_gaussian_grid(63, omega=0.0)
    st = _runner_ic(matrix_mod, tmp_path, monkeypatch)["state"]
    ls = np.asarray(grid.ls)
    vor = np.abs(np.asarray(st.vor_hat.data))
    assert vor[ls > 42].max() == 0.0
    assert vor[ls == 42].max() > 0.0


def test_row_ic_max_wind_in_cross_grid_band(matrix_mod, tmp_path,
                                            monkeypatch):
    """The runner's spectral IC keeps max |V| (its own max_wind scalar)
    inside the other grids' 34.3-36.1 m/s band (T42 + mask: 27.5)."""
    got = _runner_ic(matrix_mod, tmp_path, monkeypatch)
    vmax = got["scalar_fn"](got["state"])["max_wind"]
    assert 34.0 <= vmax <= 37.0, vmax
