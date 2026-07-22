"""#1100 MPAS twin: partition-local baroclinic-wave state build parity.

``build_sharded_baroclinic_wave_state_mpas`` must match the
global-build-then-shard path
(``shard_pytree(baroclinic_wave_init_mpas(...), dev_config)``) shard by
shard — the voronoi analogue of the virtual-device parity contract
``tests/parallel/test_atm_latlon_bandlocal_build.py`` pins for the lat-lon
lane (#1216).  Every BCW field is elementwise per cell/edge
(fixed-iteration bisection, analytic wind/temperature, saturation taper),
so slices of the global computation equal the computation on the slices up
to shape-specialized codegen.

Two-tier contract (codex #1100 review (a)):

* REQUIRED: a few-ULP per-shard match (``assert_array_max_ulp``, 4 ULP) —
  XLA does not guarantee exact bits across differently shaped
  compilations, so this is the portable invariant a builder-expression
  divergence would breach by orders of magnitude.
* DIAGNOSTIC: exact value equality, measured to hold on the pinned CPU
  stack — ``xfail(strict=False)`` so a jaxlib/XLA upgrade that starts
  vectorizing tails differently degrades it to XFAIL instead of breaking
  CI, while a real regression still surfaces in the REQUIRED tier.

Runs single-process on virtual host devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``, set by the runner —
same contract as ``test_atm_latlon_bandlocal_build.py``); SKIPS when fewer
devices are available so the plain serial suite stays green.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

N_DEV = 4
SUBDIVISION = 3   # 642 cells before ghost padding — smallest real icosahedron


@pytest.fixture(scope="module")
def _mesh_and_config():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs {N_DEV} virtual devices "
                    f"(XLA_FLAGS did not take effect)")
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
    mesh = reorder_voronoi_for_sharding(mesh, N_DEV)
    assert mesh.nCells % N_DEV == 0 and mesh.nEdges % N_DEV == 0
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=N_DEV,
    )
    assert dev_config.n_devices == N_DEV
    return mesh, dev_config


def _build_both(mesh, dev_config, moist):
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.mesh import shard_pytree
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_mpas,
        build_sharded_baroclinic_wave_state_mpas,
    )

    sigma = create_sigma_coordinate(6, dtype=None)
    local = build_sharded_baroclinic_wave_state_mpas(
        mesh, sigma, dev_config, perturbed=True, moist=moist)
    ref = shard_pytree(
        baroclinic_wave_init_mpas(mesh, sigma, perturbed=True, moist=moist),
        dev_config)
    return local, ref


def _for_each_shard_pair(sharded, reference, check):
    """Walk both pytrees; run ``check(a_shard, b_shard)`` per shard pair."""
    import jax.tree_util as jtu

    leaves_a, treedef_a = jtu.tree_flatten(sharded)
    leaves_b, treedef_b = jtu.tree_flatten(reference)
    assert treedef_a == treedef_b
    for a, b in zip(leaves_a, leaves_b):
        if not hasattr(a, "addressable_shards"):
            assert a == b
            continue
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.sharding == b.sharding, (a.sharding, b.sharding)

        def _key(shard):
            # Shard.index is a tuple of slice objects (unhashable) —
            # normalize to a hashable (start, stop, step) tuple-of-tuples.
            return tuple((sl.start, sl.stop, sl.step) for sl in shard.index)

        shards_b = {_key(s): s.data for s in b.addressable_shards}
        for s in a.addressable_shards:
            check(np.asarray(s.data), np.asarray(shards_b[_key(s)]))


def _assert_max_ulp(x, y, maxulp=4):
    # assert_array_max_ulp rejects exact-zero-vs-zero at ulp 0 gracefully,
    # but guard the all-zero leaves (u/phis/q_c/q_r) with array_equal first
    # (ULP is undefined around signed zeros).
    if np.array_equal(x, y):
        return
    np.testing.assert_array_max_ulp(x, y, maxulp=maxulp)


@pytest.mark.parametrize("moist", [False, True])
def test_partitionlocal_build_matches_global_shard_ulp(
        _mesh_and_config, moist):
    """REQUIRED few-ULP parity: a builder-expression divergence from
    baroclinic_wave_init_mpas would breach this by orders of magnitude
    (the bench would silently time a different IC)."""
    mesh, dev_config = _mesh_and_config
    local, ref = _build_both(mesh, dev_config, moist)
    _for_each_shard_pair(local, ref, _assert_max_ulp)


@pytest.mark.xfail(
    strict=False,
    reason="exact value equality measured on the pinned CPU stack; not a "
           "JAX/XLA cross-shape guarantee — the REQUIRED contract is the "
           "ULP test above")
@pytest.mark.parametrize("moist", [False, True])
def test_partitionlocal_build_exact_on_pinned_stack(
        _mesh_and_config, moist):
    mesh, dev_config = _mesh_and_config
    local, ref = _build_both(mesh, dev_config, moist)
    _for_each_shard_pair(local, ref, np.testing.assert_array_equal)


def test_single_device_falls_back_to_global_builder(_mesh_and_config):
    """face_sharding=None (n_devices=1) must return the plain global state."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_mpas,
        build_sharded_baroclinic_wave_state_mpas,
    )

    mesh, _ = _mesh_and_config
    dev1 = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=1,
    )
    sigma = create_sigma_coordinate(6, dtype=None)
    a = build_sharded_baroclinic_wave_state_mpas(mesh, sigma, dev1)
    b = baroclinic_wave_init_mpas(mesh, sigma)
    np.testing.assert_array_equal(np.asarray(a.T.data), np.asarray(b.T.data))
    np.testing.assert_array_equal(np.asarray(a.u.data), np.asarray(b.u.data))
