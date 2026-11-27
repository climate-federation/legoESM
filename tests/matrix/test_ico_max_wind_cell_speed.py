"""Icosahedral matrix ``max_wind`` is the max cell-centre |V| (Perot
reconstruction), the same quantity the cube and lat-lon rows report, not the
max |edge-normal u|.  Vote 2026-11-27 (codex B, author B; GLM unavailable)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def matrix_mod():
    spec = importlib.util.spec_from_file_location("_atmmatrix_icowind", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        pass
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(subdivision_level=2)


@pytest.mark.parametrize("nlev", [None, 3])
def test_cell_speed_not_edge_max(matrix_mod, mesh, nlev):
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity
    n_e = int(np.asarray(mesh.dvEdge).shape[0])
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n_e,) if nlev is None else (n_e, nlev))
    got = matrix_mod._ico_max_cell_speed(u, mesh)
    ue, vn = (np.asarray(a) for a in reconstruct_cell_velocity(u, mesh))
    assert got == pytest.approx(float(np.max(np.hypot(ue, vn))), rel=1e-12)
    assert abs(got - float(np.max(np.abs(u)))) > 1e-3


def test_every_ico_lane_reports_cell_speed():
    src = _SCRIPT.read_text()
    assert '"max_wind": float(jnp.max(jnp.abs(s.u.data)))' not in src
    assert src.count('"max_wind": _ico_max_cell_speed(s.u.data, mesh)') == 4
