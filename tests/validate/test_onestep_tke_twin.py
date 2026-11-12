"""Direct tests for scripts/validate/ocean_fidelity/onestep_tke_twin.py.

The tracer-operator check must reproduce a field it built itself to roundoff
and must SEE a planted 1% K error; the strip reader must refuse a record whose
time stamp is not step*rn_Dt instead of silently reading the wrong step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")
_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"
sys.path.insert(0, str(_DIR))
import onestep_tke_twin as tw  # noqa: E402


def _strip(path: Path, nt=4, nz=12, ny=3, nx=4, dt=150.0, t0=150.0, kind="T"):
    rng = np.random.default_rng(0)
    d = nc.Dataset(path, "w")
    d.createDimension("time_counter", None); d.createDimension("z", nz)
    d.createDimension("y", ny); d.createDimension("x", nx)
    t = d.createVariable("time_instant", "f8", ("time_counter",))
    t[:] = t0 + dt * np.arange(nt)
    if kind in ("T", "W"):
        suf = "T" if kind == "T" else "W"
        la = d.createVariable(f"nav_lat_eqs{suf}", "f8", ("y", "x")); la[:] = np.zeros((ny, nx))
        lo = d.createVariable(f"nav_lon_eqs{suf}", "f8", ("y", "x")); lo[:] = 240.0 + np.arange(nx)[None, :]
    if kind == "T":
        e3 = d.createVariable("e3t", "f8", ("time_counter", "z", "y", "x"))
        T = d.createVariable("votemper", "f8", ("time_counter", "z", "y", "x"))
        base = np.linspace(1.0, 4.0, nz)[None, :, None, None]
        e3[:] = base * (1.0 + 1e-4 * rng.standard_normal((nt, nz, ny, nx)))
        T[:] = 28.0 - np.cumsum(rng.uniform(0.0, 0.3, (nt, nz, ny, nx)), axis=1)
    elif kind == "W":
        K = d.createVariable("avt", "f8", ("time_counter", "z", "y", "x"))
        K[:] = 10 ** rng.uniform(-5, -2, (nt, nz, ny, nx))
    else:
        tau = d.createVariable("taum", "f8", ("time_counter", "y", "x"))
        tau[:] = 0.05
        la = d.createVariable("nav_lat_eqsT", "f8", ("y", "x")); la[:] = 0.0
        lo = d.createVariable("nav_lon_eqsT", "f8", ("y", "x")); lo[:] = 240.0
    d.close()


def test_tracer_operator_roundtrip_and_plant(tmp_path):
    _strip(tmp_path / "ORCA1_1ts_x_eqs1ts_T.nc", kind="T")
    _strip(tmp_path / "ORCA1_1ts_x_eqs1ts_W.nc", kind="W")
    import jax
    jax.config.update("jax_enable_x64", True)
    sj = np.array([0, 1, 2, 0]); si = np.array([0, 1, 2, 3])
    e3w_1d = np.linspace(1.0, 4.0, 12)
    emax, eplant = tw.tracer_operator_check(tmp_path, 3, sj, si, np.zeros(4), e3w_1d, 10)
    assert emax < tw.TRACER_OP_TOL_K * 1e-4
    assert eplant > 100 * max(emax, 1e-15)


def test_tracer_operator_refuses_wrong_record(tmp_path):
    _strip(tmp_path / "ORCA1_1ts_x_eqs1ts_T.nc", kind="T", dt=300.0)
    _strip(tmp_path / "ORCA1_1ts_x_eqs1ts_W.nc", kind="W", dt=300.0)
    with pytest.raises(SystemExit, match="is not step"):
        tw.tracer_operator_check(tmp_path, 3, np.array([0]), np.array([0]), np.zeros(1),
                                 np.linspace(1.0, 4.0, 12), 10)


def test_strip_taum_refuses_misaligned_record(tmp_path):
    _strip(tmp_path / "ORCA1_1ts_x_eqs1ts_2D.nc", kind="2D", t0=150.0)
    a, lat, lon = tw.load_strip_taum(tmp_path, 2)
    assert np.allclose(a, 0.05) and lat.shape == a.shape
    bad = tmp_path / "bad"; bad.mkdir()
    _strip(bad / "ORCA1_1ts_x_eqs1ts_2D.nc", kind="2D", dt=300.0)
    with pytest.raises(SystemExit, match="!= step"):
        tw.load_strip_taum(bad, 2)
