"""Direct tests for the two F-pivot fold probes in scripts/validate/ocean_fidelity."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MESH = ROOT / "data" / "grids" / "eORCA1.2_mesh_mask.nc"


def _load(name):
    p = ROOT / "scripts" / "validate" / "ocean_fidelity" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fold_line_transport_ours(tmp_path):
    ft = _load("fold_line_transport")
    n_lat, n, nk = 4, 8, 2
    F = np.zeros((n_lat + 1, n, nk))
    F[-1, 2, 0], F[-1, 5, 0] = 3.0, -3.0          # antisymmetric pair, m^2/s
    F[-1, 0, 0] = 99.0                            # halo column: excluded
    dx = np.full((n_lat + 1, n), 2.0e5)
    v = np.zeros_like(F)
    v[-1, 2, 1] = -0.4
    np.savez(tmp_path / "s.npz", mass_flux_v=F, dx_v=dx, v=v,
             time_days=np.asarray(2.0))
    r = ft.ours(str(tmp_path / "s.npz"))
    assert r["gross_north_Sv"] == pytest.approx(0.6)
    assert r["field"] == "mass_flux_v"
    assert r["net_Sv"] == pytest.approx(0.0)
    assert r["max_abs_v_fold"] == pytest.approx(0.4)
    F[-1, 3, 1] = np.nan
    np.savez(tmp_path / "b.npz", mass_flux_v=F, dx_v=dx, v=v)
    with pytest.raises(SystemExit):
        ft.ours(str(tmp_path / "b.npz"))


@pytest.mark.skipif(not MESH.exists(), reason="eORCA1.2 mesh not present")
def test_measure_perms_on_real_mesh():
    mp = _load("measure_fpivot_fold_perms")
    res = mp.measure(str(MESH))
    assert res == {"T ghost": ["n-1-i"], "U ghost": ["(n-i)%n"],
                   "V fold self": ["n-1-i"], "V ghost": ["n-1-i"],
                   "F fold self": ["(n-i)%n"], "F ghost": ["(n-i)%n"]}
