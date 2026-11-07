"""Direct test for scripts/validate/diag_layer_thickness_condensation.py.

The probe decides whether the CAM table's thick lower-troposphere layers are the
cause of its cloud-water deficit, so its coarse-graining must conserve water and
its adjustment must actually condense.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_layer_thickness_condensation.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_thickness", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flags():
    m = _load()
    a = m.build_arg_parser().parse_args(
        ["--config", "fine.yaml", "--restart", "c.npz", "--coarse-config", "coarse.yaml"])
    assert a.iters == 3 and a.extra == []
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--config", "fine.yaml", "--restart", "c.npz"])


def test_coarse_grain_conserves_the_column_integral():
    m = _load()
    ph_f = np.array([[0.0, 100.0, 200.0, 300.0, 400.0]])
    ph_c = np.array([[0.0, 200.0, 400.0]])
    x = np.array([[1.0, 3.0, 5.0, 7.0]])
    out = m.coarse_grain(x, ph_f, ph_c)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[2.0, 6.0]])
    fine = float(np.sum(x * np.diff(ph_f, axis=1)))
    coarse = float(np.sum(out * np.diff(ph_c, axis=1)))
    assert abs(coarse - fine) / fine < 1e-12


def test_coarse_grain_is_exact_when_the_grids_match():
    m = _load()
    ph = np.array([[0.0, 100.0, 250.0, 400.0]])
    x = np.array([[2.0, -1.0, 4.0]])
    np.testing.assert_allclose(m.coarse_grain(x, ph, ph), x)


def test_moist_adjust_condenses_only_the_supersaturated_part():
    m = _load()
    T = np.array([[290.0, 250.0]])
    p = np.array([[90000.0, 50000.0]])
    from legoesm.thermo import saturation_mixing_ratio
    qsat = np.asarray(saturation_mixing_ratio(T, p))
    q = np.stack([qsat[0] * np.array([1.5, 0.5])])
    qc, T_adj = m.moist_adjust(T, q, p, iters=3)
    assert qc[0, 0] > 0.0, "a supersaturated layer must condense"
    assert qc[0, 1] == 0.0, "a subsaturated layer must not"
    assert T_adj[0, 0] > T[0, 0], "condensation must warm the layer"
    assert qc[0, 0] < q[0, 0], "it cannot condense more water than is present"
