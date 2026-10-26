"""The lens-trigger probe: a warm top cell is stable; cooled below the cell
under it, NEMO's bn2 trigger fires at the first interface and nowhere else."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/evd_lens_trigger_probe.py"
_SPEC = importlib.util.spec_from_file_location("evd_lens_probe", _P)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["evd_lens_probe"] = _MOD
_SPEC.loader.exec_module(_MOD)


def _column(t_top):
    gdept = np.array([0.5, 1.6, 2.7, 3.9, 5.1])
    gdepw = np.array([1.0, 2.1, 3.3, 4.5])
    T = np.array([[t_top, 25.7, 25.5, 25.3, 25.0]])
    S = np.full_like(T, 35.0)
    return T, S, gdept, gdepw


def test_lens_is_stable_and_cooled_top_cell_fires_only_at_interface_1():
    T, S, gdept, gdepw = _column(25.9)
    n2 = _MOD._n2_top_interfaces(T, S, gdept, gdepw)
    assert (n2 > 0).all()
    Tn = T.copy(); Tn[0, 0] = 25.4          # colder than the 1.6 m cell
    n2 = _MOD._n2_top_interfaces(Tn, S, gdept, gdepw)
    assert n2[0, 0] <= -1e-12 and (n2[0, 1:] > 0).all()
