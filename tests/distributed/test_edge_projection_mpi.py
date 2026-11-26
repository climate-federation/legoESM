"""Physics cell -> edge projection under the Voronoi MPI lane (docs/85 13.34).

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_edge_projection_mpi.py

Every physics bridge (turbulence, ZM momentum transport, gravity-wave drag)
averages its cell tendency onto the edges.  On a partition cut one of the two
cells is a halo copy whose physics may differ from its owner's; the projection
must then use the OWNERS' values, so the owned edges equal the serial
projection bit for bit, for the plain mean and for the layer-mass-weighted
form ZM's momentum transport uses.  The halo rows are poisoned first, and the
same call with the owner refresh disabled must miss, so the test can fail.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.grids.voronoi import cell_vector_to_edge_normal, create_voronoi_mesh  # noqa: E402

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2,
                                reason="needs at least 2 MPI ranks")
NLEV = 6


@pytest.fixture
def setup():
    # per test: the distributed conftest forgets the active layout after each test
    jax.config.update("jax_enable_x64", True)
    from legoesm.parallel.voronoi_mpi import initialize_voronoi_mpi, reset_voronoi_layout
    mesh = create_voronoi_mesh(subdivision_level=3, lloyd_iterations=5)
    rng = np.random.default_rng(11)
    du = rng.normal(0.0, 1e-4, (mesh.nCells, NLEV))
    dv = rng.normal(0.0, 1e-4, (mesh.nCells, NLEV))
    dp = rng.uniform(500.0, 3000.0, (mesh.nCells, NLEV))
    # serial references BEFORE the layout is armed (identity hook on the global mesh anyway)
    ref = {w: np.asarray(cell_vector_to_edge_normal(
        jnp.asarray(du), jnp.asarray(dv), mesh, None if w == "plain" else jnp.asarray(dp)))
        for w in ("plain", "dp")}
    _, _, layout = initialize_voronoi_mpi(mesh)
    yield mesh, layout, du, dv, dp, ref
    reset_voronoi_layout()


def _local_poisoned(a, part):
    from legoesm.parallel.voronoi_partition import scatter_to_local
    loc = np.asarray(scatter_to_local(jnp.asarray(a), part, "cell")).copy()
    loc[part.n_owned_cells:] *= 1.37            # halo copy differs from its owner
    return jnp.asarray(loc)


@pytest.mark.parametrize("weights", ["plain", "dp"])
def test_owned_edges_equal_serial_bitwise(setup, weights, monkeypatch):
    mesh, layout, du, dv, dp, ref = setup
    part = layout.partition
    lm = layout.local_mesh
    args = [_local_poisoned(du, part), _local_poisoned(dv, part), lm]
    kw = {} if weights == "plain" else {"dp_cell": _local_poisoned(dp, part)}
    own = part.n_owned_edges
    want = ref[weights][np.asarray(part.local_edges[:own])]
    got = np.asarray(cell_vector_to_edge_normal(*args, **kw))[:own]
    np.testing.assert_array_equal(got, want, err_msg=f"{weights}: owned edges differ from serial")

    # control: without the owner refresh the poisoned halo reaches the cut edges
    import legoesm.parallel.voronoi_mpi as vm
    monkeypatch.setattr(vm, "get_matching_voronoi_layout", lambda m: None)
    stale = np.asarray(cell_vector_to_edge_normal(*args, **kw))[:own]
    assert not np.array_equal(stale, want), "a poisoned halo changed nothing: the test cannot fail"
