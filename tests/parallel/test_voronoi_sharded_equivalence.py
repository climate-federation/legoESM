"""Voronoi (icosahedral / MPAS) SPMD-vs-single-device equivalence.

The Voronoi sharded path partitions the unstructured cell/edge/vertex
arrays across devices; ``make_voronoi_sharded_step`` builds a
``shard_map`` kernel that does ppermute (or all_gather) halo exchange
between neighbour partitions.  This test asserts that the result of
running a few SSP-RK3 steps under the SPMD path matches the
single-device path within floating-point tolerance — guarding the
iter-2 ``_total_area`` precompute, the iter-2 sharded mass-fix
allreduce merge, and the underlying ppermute/allgather kernels.

Skips if fewer than 4 CPU devices are exposed; run with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _need_multi_device(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


class TestVoronoiShardedEquivalence:
    """Single-device vs multi-device Voronoi MPAS step produce
    physically-equivalent state after a few SSP-RK3 steps.

    Tolerances are much looser than the spectral test (the spectral
    transform is purely linear so level-sharding is bit-equivalent up
    to FMA reordering at 1e-12).  Voronoi unstructured-mesh sharding
    introduces a more subtle drift: halo cells of one partition lack
    a few of their edges (the AND filter applied in
    ``_build_voronoi_partition_infra`` to keep ``cellsOnEdge`` valid
    excludes edges whose far cell is outside the cell halo), so
    operator quantities at halo cells are slightly different from
    single-device.  Owned-cell tendencies that read from neighbour
    halo cells inherit a small fraction of that drift.

    The post-iter-23 sharded path produces dynamical fields that
    track single-device to ~1% over a few RK3 steps — well below the
    pre-iter-23 garbage state (u → 1e76 after one step).  The
    tolerance below codifies "physically reasonable; no NaN; no
    runaway" rather than "bit-equivalent".

    Real-hardware MPI Voronoi production runs use the
    ``make_voronoi_mpi_step`` path in ``parallel/voronoi_mpi.py``,
    which has a different (mpi4jax sendrecv-based) halo exchange and
    is already tested at scale by the JW BCW conservation tests.
    """

    def _run(self, *, devices: int, n_steps: int = 5):
        """Run the SSP-RK3 evolution on a Voronoi mesh that has been
        pre-reordered for ``devices``-way sharding.  Both the
        single-device and multi-device paths use the *same* reordered
        mesh so cell/edge indices match — direct array comparison is
        meaningful.
        """
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
        )
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

        n_lev, dt = 8, 600.0
        mesh = create_voronoi_mesh(subdivision_level=4)
        # Reorder for the *target* device count even on the
        # single-device baseline so both runs use identical cell/edge
        # indices.  Reorder is a no-op for n_devices=1 (or close to
        # one).
        mesh = reorder_voronoi_for_sharding(
            mesh, max(devices, 2),  # always reorder for 2+
        )
        sigma = create_sigma_coordinate(n_lev)
        cfg = MPASPrimitiveEquationConfig(
            nu_del4=1e16, nu_del4_ps=1e16,
            fix_mass=True, pv_scheme='energy',
            time_integrator='ssp_rk3',
        )

        if devices == 1:
            model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
            state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
            for _ in range(n_steps):
                state = model.step(state, dt)
            return state

        # Multi-device: same reordered mesh, but build a sharded step.
        from legoesm.parallel.mesh import (
            create_voronoi_device_mesh, replicate_pytree,
        )
        from legoesm.parallel.sharded_dynamics import (
            make_voronoi_sharded_step,
        )

        dev_config = create_voronoi_device_mesh(
            nCells=mesh.nCells, nEdges=mesh.nEdges,
            nVertices=mesh.nVertices, n_devices=devices,
        )
        mesh_replicated = replicate_pytree(mesh, dev_config)

        model = MPASPrimitiveEquationModel(mesh_replicated, sigma, cfg)
        state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
        step_fn = make_voronoi_sharded_step(model, dev_config)
        for _ in range(n_steps):
            state = step_fn(state, dt)
        return state

    def test_2device_matches_1device(self):
        """Verify the 2-device Voronoi sharded step matches single-device
        to floating-point precision after one SSP-RK3 step.

        Iter-23 fix removed the pre-existing 1e76 explosion (root cause:
        ``-1`` cellsOnEdge from the OR-edge-halo filter).  Iter-25
        added an iterative augmentation that pulls in the OTHER cell of
        every halo edge until the halo is closed under the cellsOnEdge
        relation; this eliminates the residual ~1% drift from halo
        cells lacking some of their edges.  Post-iter-25 the path is
        bit-equivalent up to floating-point accumulation order:
        max abs diff ≈ 1e-8 on u, 1e-9 on T, 1e-2 on p_s (1e-7 relative).
        """
        _need_multi_device(2)
        ref = self._run(devices=1, n_steps=1)
        out = self._run(devices=2, n_steps=1)
        for name, atol, rtol in (
            ("u", 1e-6, 1e-6),
            ("T", 1e-6, 1e-7),
            ("p_s", 1e-1, 1e-6),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 2-device drift exceeds float-pt envelope",
            )
