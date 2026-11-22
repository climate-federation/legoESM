"""Saved variable-resolution meshes: any cell count, marked and pinned.

``load_voronoi_mesh_for_run(variable=True)`` loads a mesh whose cell count is
not 10 x 4**r + 2 when the file is marked ``variable_mesh``; a marked file is
refused as a uniform mesh and an icosahedral file is refused as a variable one.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

from legoesm import constants  # noqa: E402
from legoesm.grids.voronoi import (  # noqa: E402
    _build_mesh_from_generators, _lloyd_relaxation, create_voronoi_mesh)
from legoesm.grids.voronoi_mesh_file import (  # noqa: E402
    MeshFileError, load_voronoi_mesh_for_run, save_voronoi_mesh)


def _variable_mesh(n=137):
    rng = np.random.default_rng(1)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    p = _lloyd_relaxation(p, n_iter=5)
    return _build_mesh_from_generators(p, constants.R_earth)


def test_variable_mesh_round_trip(tmp_path):
    mesh = _variable_mesh()
    assert mesh.nCells == 137 and (mesh.nCells - 2) % 10 != 0
    f = tmp_path / "v.npz"
    sha = save_voronoi_mesh(mesh, f, extra_metadata={"variable_mesh": True})
    m = load_voronoi_mesh_for_run(f, resolution=5, expected_sha256=sha,
                                  lloyd_iterations=50, radius=constants.R_earth,
                                  variable=True)
    assert m.nCells == 137
    np.testing.assert_array_equal(np.asarray(m.dcEdge), np.asarray(mesh.dcEdge))
    with pytest.raises(MeshFileError, match="variable"):
        load_voronoi_mesh_for_run(f, resolution=5, expected_sha256=sha)
    with pytest.raises(MeshFileError, match="does not match the stored|differs"):
        load_voronoi_mesh_for_run(f, resolution=5, expected_sha256="0" * 64,
                                  variable=True)


def test_unmarked_variable_mesh_refused(tmp_path):
    """Without the mark, a non-icosahedral count still fails the uniform check."""
    mesh = _variable_mesh()
    f = tmp_path / "u.npz"
    sha = save_voronoi_mesh(mesh, f, resolution=2)
    with pytest.raises(MeshFileError, match="cells"):
        load_voronoi_mesh_for_run(f, resolution=2, expected_sha256=sha)
    with pytest.raises(MeshFileError, match="icosahedral"):
        load_voronoi_mesh_for_run(f, resolution=2, expected_sha256=sha, variable=True)


def test_icosahedral_path_unchanged(tmp_path):
    mesh = create_voronoi_mesh(1, lloyd_iterations=0)
    f = tmp_path / "i.npz"
    sha = save_voronoi_mesh(mesh, f, resolution=1, lloyd_iterations=0)
    assert load_voronoi_mesh_for_run(f, resolution=1, expected_sha256=sha,
                                     lloyd_iterations=0).nCells == 42
