"""P1.5 production-stack closure tripwire: the run validator FAILS a run
whose ledger convection energy row is decoupled from its water row.

The gate is ``scripts/validate/validate_amip_run.py::conv_pairing_rel``
reading ``budget_ledger.npz`` (written by ``--budget-ledger`` runs).
RED/GREEN provenance (2026-07-22, recorded for the fix PR): the PRE-fix
instrumented production runs measured rel = 0.826 (cold start) / 0.885
(equilibrium restart) — both FAIL; the POST-fix Checkpoint-B run measured
rel = 0.069 — PASS.  Threshold 0.35 sits 2.4x above the post-fix value
and 2.4x below the pre-fix one.

Covers: synthetic-violation self-test (a pre-fix-magnitude ledger must
FAIL validate()), paired-clean pass, absent-ledger skip, weak-convection
skip (no verdict on noise), and a lock so the tripwire cannot be
silently deleted.
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

from legoesm import constants  # noqa: E402
from legoesm.diagnostics.process_ledger import LEDGER_PROCESSES  # noqa: E402

MM_DAY = 1.0 / 86400.0  # kg/m2/s per mm/day


def _make_run_dir(tmp_path, conv_water_mm_day, conv_energy_w_m2):
    """Minimal passing run dir + a budget_ledger.npz with a convection row."""
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "timeseries.npz",
             days=np.array([1.0, 2.0]),
             T_atm=np.array([280.0, 281.0]),
             precip=np.array([2.0, 2.1]),
             CWV=np.array([25.0, 25.5]),
             sic=np.array([0.1, 0.1]),
             max_wind=np.array([30.0, 31.0]),
             energy_toa_net=np.array([1.0, 1.0]))
    (run / "results.txt").write_text("Status: COMPLETED\n")

    rates = np.zeros((2, len(LEDGER_PROCESSES), 2))
    i = list(LEDGER_PROCESSES).index("convection")
    rates[:, i, 0] = conv_water_mm_day * MM_DAY
    rates[:, i, 1] = conv_energy_w_m2
    np.savez(run / "budget_ledger.npz",
             rates=rates, processes=np.array(list(LEDGER_PROCESSES)))
    return run


class TestConvPairingTripwire:

    def test_lock_symbols_exist(self):
        """Deleting the tripwire (threshold or helper) goes red here."""
        assert _mod.CONV_PAIRING_REL_MAX == 0.35
        assert _mod.CONV_PAIRING_MIN_MM_DAY == 0.2
        assert callable(_mod.conv_pairing_rel)

    def test_synthetic_violation_fails(self, tmp_path, capsys):
        """Pre-fix magnitudes (heating ~5x its L_v-consistent value, the
        2-yr pilot's cold-start signature) must FAIL the validator."""
        run = _make_run_dir(tmp_path, conv_water_mm_day=-1.93,
                            conv_energy_w_m2=+322.0)
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 1
        assert "conv_pairing" in out
        assert "[FAIL] conv energy/water pairing rel" in out

    def test_paired_row_passes(self, tmp_path, capsys):
        """Post-fix regime: heating == -L_v * water removal exactly."""
        w = -2.0
        e = -w * MM_DAY * constants.L_v
        run = _make_run_dir(tmp_path, conv_water_mm_day=w, conv_energy_w_m2=e)
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[PASS] conv energy/water pairing rel" in out

    def test_absent_ledger_skips(self, tmp_path, capsys):
        run = _make_run_dir(tmp_path, -2.0, 58.0)
        (run / "budget_ledger.npz").unlink()
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[SKIP] conv energy/water pairing rel" in out

    def test_weak_convection_skips_not_certifies(self, tmp_path, capsys):
        """|W| below CONV_PAIRING_MIN_MM_DAY: no verdict on noise — even a
        grossly unpaired (but tiny) row must SKIP, not PASS or FAIL."""
        run = _make_run_dir(tmp_path, conv_water_mm_day=-0.01,
                            conv_energy_w_m2=+50.0)
        rc = _mod.validate(run)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[SKIP] conv energy/water pairing rel" in out

    def test_helper_measures_relative_mismatch(self, tmp_path):
        """rel = |E - (-L_v W)| / max(|E|, |L_v W|), on the run mean."""
        w = -2.0
        e_paired = -w * MM_DAY * constants.L_v
        run = _make_run_dir(tmp_path, conv_water_mm_day=w,
                            conv_energy_w_m2=2.0 * e_paired)
        assert _mod.conv_pairing_rel(run) == pytest.approx(0.5)
