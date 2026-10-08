import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias/rain_number_audit.py"
_spec = importlib.util.spec_from_file_location("rain_number_audit", _P)
rna = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rna)

LAMR = 5.0e4


def test_healthy_state_counts_nothing():
    q = np.full((3, 4), 1e-4)
    n = np.full((3, 4), 1e3)
    out = rna.audit(n, q, LAMR)
    assert out["orphan_levels"] == 0
    assert out["levels_N_r_without_mass"] == 0
    assert out["levels_above_psd_ceiling_x10"] == 0


def test_poisoned_levels_are_counted():
    q = np.full((3, 4), 1e-4)
    n = np.full((3, 4), 1e3)
    q[0, 0], n[0, 0] = 0.0, 4.8e22          # orphan, the #1515 signature
    n[1, 1] = 1e16                           # with mass, far above the ceiling
    out = rna.audit(n, q, LAMR)
    assert out["orphan_levels"] == 1
    assert out["levels_N_r_gt_1e15"] == 2
    assert out["levels_N_r_without_mass"] == 1
    assert out["levels_above_psd_ceiling_x10"] == 2
    assert out["max_N_r_per_kg"] == pytest.approx(4.8e22)


def test_nonfinite_state_is_refused():
    with pytest.raises(ValueError):
        rna.audit(np.array([np.nan]), np.array([0.0]), LAMR)


def test_checkpoint_needs_per_mass_stamp(tmp_path):
    p = tmp_path / "c.npz"
    np.savez(p, trc_N_r=np.zeros((2, 2)), trc_q_r=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="number_convention"):
        rna.audit_checkpoint(str(p))
    np.savez(p, trc_N_r=np.zeros((2, 2)), trc_q_r=np.zeros((2, 2)),
             number_convention=np.array("per_mass"))
    assert rna.audit_checkpoint(str(p))["orphan_levels"] == 0


def test_fires_on_the_real_poisoned_1515_column():
    """Instrument check on the real pre-fix state: MPAS cell 1432, day 967 of
    the fine_r1 arm (tree ff3bbdc1d), kept as literals in the replay test."""
    from tests.unit import test_replay_column_microphysics as rep
    n = np.asarray(rep.N_R_1432, float)[None, :]
    q = np.asarray(rep.Q_R_1432, float)[None, :]
    out = rna.audit(n, q, LAMR)
    assert out["orphan_levels"] >= 1
    assert out["levels_N_r_gt_1e15"] >= 1
    # the issue's documented maximum for this column: 1.41e19 (per volume)
    assert out["max_N_r_per_kg"] == pytest.approx(1.41e19, rel=0.01)
