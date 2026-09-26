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

    def _run(self, *, devices: int, reorder_for: int | None = None,
             n_steps: int = 5, subdivision_level: int = 4,
             physics_fn=None, outer_scan: bool = False):
        """Run the SSP-RK3 evolution on a Voronoi mesh that has been
        pre-reordered for ``reorder_for``-way sharding (default:
        same as ``devices``).  When comparing single-device against
        N-device, set ``reorder_for=N`` on both so cell/edge indices
        match for direct array comparison.

        ``physics_fn`` (MPAS operator-split convention, e.g.
        ``held_suarez_forcing_mpas``) is threaded to ``model.step`` on
        the single-device path and passed at CALL time as a kwarg on
        the sharded path — the exact calling pattern of the levante
        GPU-scaling benchmark.  ``outer_scan=True`` (sharded path only)
        additionally traces the step inside an outer ``jax.jit`` +
        ``lax.scan``, mirroring the bench's timed ``_build_timed_scan_runner``
        execution shape.
        """
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
        )
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

        # Lower dt at higher subdivision to stay CFL-stable (dx scales
        # ~ 2^level so dt should scale 4× per level).
        dt = 600.0 if subdivision_level == 4 else 200.0
        n_lev = 8
        mesh = create_voronoi_mesh(subdivision_level=subdivision_level)
        # Reorder for the *target* device count on both single-device
        # and multi-device paths so cell/edge indices match — direct
        # array comparison is meaningful.
        target = reorder_for if reorder_for is not None else max(devices, 2)
        mesh = reorder_voronoi_for_sharding(mesh, target)
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
                state = model.step(state, dt, physics_fn=physics_fn)
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

        if outer_scan:
            # Mirror the bench's TIMED execution shape
            # (run_levante_gpu_scaling._build_timed_scan_runner): the
            # call-time-physics step wrapped in a once-created lambda,
            # traced inside an outer jit + lax.scan with dt captured as
            # a Python float in the closure.
            _phys = physics_fn
            if _phys is not None:
                _wrapped = lambda s, d: step_fn(s, d, physics_fn=_phys)
            else:
                _wrapped = step_fn
            dt_const = float(dt)

            @jax.jit
            def _runner(st):
                def _body(carry, _):
                    return _wrapped(carry, dt_const), None
                return jax.lax.scan(_body, st, None, length=n_steps)[0]

            return _runner(state)

        for _ in range(n_steps):
            if physics_fn is not None:
                # CALL-time kwarg — the levante bench pattern
                # (run_levante_gpu_scaling.py wraps the sharded step in
                # ``lambda s, dt: step(s, dt, physics_fn=_phys)``).
                state = step_fn(state, dt, physics_fn=physics_fn)
            else:
                state = step_fn(state, dt)
        return state

    @pytest.mark.parametrize("devices", [2, 3, 4])
    def test_Ndevice_matches_1device(self, devices):
        self._run_equivalence(devices=devices, subdivision_level=4)

    def test_subdiv5_2device_matches(self):
        """Verify the iter-37 cap=2 generalises to higher resolution
        (subdivision_level=5, 10242 cells).  Same envelope as subdiv=4.
        """
        self._run_equivalence(devices=2, subdivision_level=5)

    def test_2device_calltime_physics_fn(self):
        """Regression guard (levante job 8457521): ``physics_fn`` passed
        at CALL time must be closure-captured, never traced.

        The GPU-scaling bench wraps the sharded step as
        ``lambda s, dt: step(s, dt, physics_fn=_phys)``; the previous
        ``make_voronoi_sharded_step`` returned a bare ``jax.jit`` of
        ``(state, dt)``, so JAX tried to abstractify the callable kwarg:
        "Error interpreting argument ... <class 'function'> ... passed
        ... at path kwargs['physics_fn']".  This builds the 2-device
        sharded step WITH Held-Suarez physics, runs 2 steps, and checks
        the result against the single-device operator-split reference
        (``model.step(state, dt, physics_fn=...)``) within the existing
        float-pt envelope.  A no-physics single-device run additionally
        proves the physics signal exceeds that envelope — so silently
        DROPPING physics_fn (not just crashing on it) also fails.
        A fourth run re-traces the same step inside an outer
        ``jax.jit`` + ``lax.scan`` — the bench's timed
        ``_build_timed_scan_runner`` shape — and must match too.
        """
        _need_multi_device(2)
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas

        common = dict(reorder_for=2, n_steps=2, subdivision_level=4)
        ref_nophys = self._run(devices=1, **common)
        ref = self._run(
            devices=1, physics_fn=held_suarez_forcing_mpas, **common)
        out = self._run(
            devices=2, physics_fn=held_suarez_forcing_mpas, **common)
        out_scan = self._run(
            devices=2, physics_fn=held_suarez_forcing_mpas,
            outer_scan=True, **common)

        # Non-vacuity: the Held-Suarez increment on the reference must
        # exceed the equivalence tolerance below, otherwise "sharded
        # matches single-device" could not distinguish applied physics
        # from silently-dropped physics.  (Newtonian relaxation over
        # 2 x 600 s against an O(10 K) T - T_eq offset gives
        # max|dT| ~ 1e-2 K; the T envelope is atol=1e-6.)
        dT_phys = np.max(np.abs(
            np.asarray(ref.T.data) - np.asarray(ref_nophys.T.data)))
        assert dT_phys > 1e-4, (
            f"Held-Suarez T increment too small to make the equivalence "
            f"check a physics-applied guard (max |dT| = {dT_phys:.3e} K)"
        )

        for name, atol, rtol in (
            ("u", 1e-6, 1e-6),
            ("T", 1e-6, 1e-7),
            ("p_s", 1e-1, 1e-6),
        ):
            r = np.asarray(getattr(ref, name).data)
            for label, result in (("host-loop", out), ("outer-scan", out_scan)):
                o = np.asarray(getattr(result, name).data)
                assert np.all(np.isfinite(o)), (
                    f"{name} ({label}): non-finite values in 2-device "
                    f"with-physics output"
                )
                np.testing.assert_allclose(
                    o, r, atol=atol, rtol=rtol,
                    err_msg=(
                        f"{name} ({label}): 2-device step with call-time "
                        f"physics_fn diverges from single-device "
                        f"operator-split reference"
                    ),
                )

    def _run_equivalence(self, *, devices: int, subdivision_level: int):
        """Verify the N-device Voronoi sharded step matches single-device
        to floating-point precision after one SSP-RK3 step.

        Iter-23 fix removed the pre-existing 1e76 explosion (root cause:
        ``-1`` cellsOnEdge from the OR-edge-halo filter).  Iter-25
        added an iterative augmentation that pulls in the OTHER cell of
        every halo edge until the halo is closed under the cellsOnEdge
        relation; this eliminates the residual ~1% drift from halo
        cells lacking some of their edges.  Iter-27 vectorised the
        augmentation loop.  Iter-28 extends test coverage to 2, 3,
        and 4 devices.

        Post-iter-25 the path is bit-equivalent up to floating-point
        accumulation order: max abs diff ≈ 1e-8 on u, 1e-9 on T,
        ~1e-3 on p_s (1e-7 relative).
        """
        _need_multi_device(devices)
        # Reorder both single-device and multi-device runs for the same
        # target device count so cell/edge indices match.
        ref = self._run(
            devices=1, reorder_for=devices, n_steps=1,
            subdivision_level=subdivision_level,
        )
        out = self._run(
            devices=devices, reorder_for=devices, n_steps=1,
            subdivision_level=subdivision_level,
        )
        for name, atol, rtol in (
            ("u", 1e-6, 1e-6),
            ("T", 1e-6, 1e-7),
            ("p_s", 1e-1, 1e-6),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: {devices}-device drift exceeds float-pt envelope",
            )


def test_sharded_borrow_without_mass_fix_gets_real_cell_area():
    """The sharded floors pass areaCell to the borrow's global residual; with
    fix_mass=False that array must still be the real mesh area, not the
    empty mass-fixer placeholder."""
    _need_multi_device(2)
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.mesh import create_voronoi_device_mesh, replicate_pytree
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step

    mesh = reorder_voronoi_for_sharding(create_voronoi_mesh(subdivision_level=3), 2)
    sigma = create_sigma_coordinate(6)
    cfg = MPASPrimitiveEquationConfig(
        fix_mass=False, conservative_tracer_clamp=True,
        time_integrator="ssp_rk3")
    dev = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=2)
    model = MPASPrimitiveEquationModel(replicate_pytree(mesh, dev), sigma, cfg)
    state = held_suarez_init_mpas(mesh, sigma)
    ncell, nlev = state.T.data.shape
    q = jnp.asarray(np.random.default_rng(0).uniform(0.0, 1e-3, (ncell, nlev)))
    q = q.at[:, 2].add(-2e-4)
    state = state._replace(tracers={"q_v": state.p_s.replace(data=q)})
    out = make_voronoi_sharded_step(model, dev)(state, 75.0)
    arr = np.asarray(out.tracers["q_v"].data)
    assert np.isfinite(arr).all() and arr.min() >= 0.0
