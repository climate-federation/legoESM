"""Both lat-lon Held-Suarez matrix cases set the del-4 coefficient explicitly."""
import importlib
import sys
from pathlib import Path

import pytest

_MATRIX_DIR = Path(__file__).resolve().parents[2] / "scripts" / "matrix"


class _Built(Exception):
    pass


@pytest.mark.parametrize("case,vcoord", [
    ("held_suarez", "sigma"), ("held_suarez", "hybrid"),
    ("held_suarez_topo", "hybrid")])
def test_latlon_hs_cases_set_nu_del4(case, vcoord, monkeypatch, tmp_path):
    sys.path.insert(0, str(_MATRIX_DIR))
    mat = importlib.import_module("run_atmosphere_test_matrix")
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pe

    seen = {}

    class _Capture(pe.CGridLatLonPrimitiveEquationModel):
        def __init__(self, grid, sigma, config, dt):
            seen["config"] = config
            raise _Built

    monkeypatch.setattr(pe, "CGridLatLonPrimitiveEquationModel", _Capture)
    tc = mat.TestCase("hydrostatic", case, "latlon", "72x144", vcoord, 200, 2)
    with pytest.raises(_Built):
        mat.run_held_suarez(tc, tmp_path, 0.01)
    assert seen["config"].nu_del4 == pytest.approx(mat._biharmonic_visc_latlon(72))
    assert seen["config"].nu_del4 > 0.0
