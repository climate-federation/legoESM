"""P1.2 hfls-sanity tripwire: per-cell time-mean hfls > 500 W/m² fails
``scripts/validate/validate_amip_run.py``.

Covers the synthetic-violation self-test (non-vacuous: an injected
2000 W/m² cell — the 2-yr AMIP pilot's Caspian signature — must FAIL the
validator), the clean-field pass, the absent-CMOR skip, and a lock
assertion so the tripwire cannot be silently deleted (imports the
threshold and helper by name).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "validate_amip_run", REPO / "scripts" / "validate" / "validate_amip_run.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _make_run_dir(tmp_path, hfls_cell_value):
    """Minimal passing run dir + a CMOR Amon hfls with one hot cell."""
    run = tmp_path / "run"
    (run / "cmor" / "Amon").mkdir(parents=True)
    np.savez(run / "timeseries.npz",
             days=np.array([1.0, 2.0]),
             T_atm=np.array([280.0, 281.0]),
             precip=np.array([2.0, 2.1]),
             CWV=np.array([25.0, 25.5]),
             sic=np.array([0.1, 0.1]),
             max_wind=np.array([30.0, 31.0]),
             energy_toa_net=np.array([1.0, 1.0]))
    (run / "results.txt").write_text("Status: COMPLETED\n")

    with netCDF4.Dataset(run / "cmor" / "Amon" / "hfls_Amon_test_gn.nc",
                         "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("lat", 4)
        ds.createDimension("lon", 8)
        v = ds.createVariable("hfls", "f8", ("time", "lat", "lon"))
        field = np.full((2, 4, 8), 80.0)
        field[:, 2, 3] = hfls_cell_value
        v[:] = field
    return run


class TestHflsTripwire:

    def test_lock_symbols_exist(self):
        """Deleting the tripwire (threshold or helper) goes red here."""
        assert _mod.HFLS_CELL_MAX_W_M2 == 500.0
        assert callable(_mod.hfls_cell_max)

    def test_synthetic_violation_fails(self, tmp_path, capsys):
        run = _make_run_dir(tmp_path, 2000.0)   # the Caspian signature
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 1
        assert "hfls_cell_max" in out
        assert "[FAIL] time-mean hfls cell max" in out

    def test_clean_field_passes(self, tmp_path, capsys):
        run = _make_run_dir(tmp_path, 240.0)    # warm-pool-class maximum
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[PASS] time-mean hfls cell max" in out

    def test_absent_cmor_skips(self, tmp_path, capsys):
        run = _make_run_dir(tmp_path, 80.0)
        import shutil
        shutil.rmtree(run / "cmor")
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[SKIP] time-mean hfls cell max" in out

    def test_helper_reports_cell_max_not_mean(self, tmp_path):
        """Cell-resolved on purpose: the mean would dilute the hotspot."""
        run = _make_run_dir(tmp_path, 2000.0)
        assert _mod.hfls_cell_max(run) == pytest.approx(2000.0)
