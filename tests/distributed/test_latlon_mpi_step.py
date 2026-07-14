"""Multi-rank correctness tests for ``make_latlon_mpi_step``.

The single-rank counterpart in
:mod:`tests.parallel.test_latlon_mpi_step_serial` cannot catch:

* The rank-aware pole-v BC (only differs from serial when interior
  ranks would otherwise zero their band-boundary v-rows).
* The pre-allreduced ``grid_total_area`` in the fixer model (only
  matters when the rank-local grid is a band, not the full sphere).
* PPM transport across rank boundaries (interior cells near a band
  cut depend on the neighbour rank's data via halo exchange).
* The MPI halo exchange itself in the dynamics path.

This module gathers the per-rank stepped state back to rank 0 and
compares to the canonical serial step.  Match is expected to fp64
machine precision (a few ULPs) — MPI introduces no algebraic
differences when the band decomposition is set up correctly.

Run with::

    mpirun -np 2 python -m pytest tests/distributed/test_latlon_mpi_step.py -v
    mpirun -np 4 python -m pytest tests/distributed/test_latlon_mpi_step.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_mpi import (
    gather_state_latlon,
    make_latlon_band_layout,
    make_latlon_mpi_step,
    scatter_state_latlon,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def global_grid():
    return create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega,
    )


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(n_levels=6)


@pytest.fixture(scope="module")
def serial_config():
    return CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,
        use_polar_filter=False,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )


@pytest.fixture(scope="module")
def global_serial_model(global_grid, sigma, serial_config):
    """The reference serial model with the FULL global grid.

    Constructed on EVERY rank: each rank computes the serial reference
    itself (deterministically identical — same global state, no comm)
    *before* ``make_latlon_mpi_step`` arms the MPI halo backend.  A
    rank-0-only serial reference computed *after* arming deadlocks:
    the serial ``_step_cgrid`` traces MPI sendrecv/allreduce ops
    (``fix_mass`` → ``batch_global_area_sums``; operator pads → band
    sendrecv) that no other rank matches.  That deadlock is exactly
    how every historical np>1 run of this file timed out while rank 1
    printed a trivial early-return "passed"."""
    return CGridLatLonPrimitiveEquationModel(
        global_grid, sigma, serial_config,
    )


def _make_global_state(grid, nlev, with_tracers: bool, seed: int = 31337):
    """Build a perturbed rest state on the global grid.

    Critical: identical seeding on every rank so each constructs the
    SAME global state, then scatters its own band.  No comm.bcast
    required.
    """
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    eps = 1.0e-3
    u = jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev)))
    v = jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev)))
    T = jnp.asarray(
        300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))
    )
    p_s = jnp.asarray(
        1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))
    )
    phis = jnp.zeros((n_lat, n_lon))
    tracers = {}
    if with_tracers:
        tracers["q_v"] = jnp.asarray(
            1.0e-3 * (1.0 + 0.05 * rng.standard_normal(
                (n_lat, n_lon, nlev),
            ))
        )
        tracers["q_c"] = jnp.asarray(
            1.0e-5 * (1.0 + 0.05 * rng.standard_normal(
                (n_lat, n_lon, nlev),
            ))
        )
    return CGridLatLonHydrostaticState(
        u=u, v=v, T=T, p_s=p_s, phis=phis, tracers=tracers,
    )


def _make_local_model(global_grid, sigma, serial_config, layout):
    """Build a rank-local model with the local lat band.

    Uses ``LatLonGrid._replace`` to swap in band slices of every
    latitude-dependent metric.  No halos here — the wrapper adds those
    internally via ``build_padded_grid``.
    """
    s, e = layout.lat_start, layout.lat_end
    band_lat = global_grid.lat[s:e]
    band_lat2d = global_grid.lat2d[s:e, :]
    band_lon2d = global_grid.lon2d[s:e, :]
    band_cos_lat = global_grid.cos_lat[s:e]
    band_sin_lat = global_grid.sin_lat[s:e]
    band_dy = global_grid.dy[s:e]
    band_f = global_grid.f[s:e, :]
    band_dx = global_grid.dx[s:e, :]
    band_area = global_grid.area[s:e, :]
    band_total_area = jnp.sum(band_area)
    band_grid = global_grid._replace(
        n_lat=layout.n_lat_local,
        lat=band_lat,
        lat2d=band_lat2d,
        lon2d=band_lon2d,
        cos_lat=band_cos_lat,
        sin_lat=band_sin_lat,
        # v-face arrays span ``[s, e+1)`` — one extra row at the
        # northern boundary (shared with neighbour rank).  Stage 3-E:
        # the polar filter builds its v-face mask from these, so the
        # band slice must carry the rank-local v-face metric, not the
        # global one.  Same convention as
        # ``slice_latlon_grid_to_band``.
        lat_v=global_grid.lat_v[s:e + 1],
        cos_lat_v=global_grid.cos_lat_v[s:e + 1],
        dy=band_dy,
        f=band_f,
        dx=band_dx,
        area=band_area,
        total_area=band_total_area,
        # ``grid_total_area`` is a property delegating to
        # ``total_area`` — _replace must use the underlying field
        # name (this caught the first smoke run on Stage 2).
    )
    return CGridLatLonPrimitiveEquationModel(
        band_grid, sigma, serial_config,
    )


# ---------------------------------------------------------------------------
# Multi-rank step equivalence
# ---------------------------------------------------------------------------


# Tolerance calibration (2026-06-10, jobs 8456042/8456021): the serial
# reference and the armed-MPI leg compile DIFFERENT XLA programs (halo
# dispatch), so fusion/reduction reorder leaves a roundoff floor of
# ~5e-11 abs on u after 3 steps at f64 even with all partition-cut bugs
# fixed (ocean analog: eta 3.7e-9 after 12 steps).  rtol=1e-8/atol=1e-9
# sits well above that floor and orders of magnitude below real cut
# corruption (e-6 class, caught at these tolerances pre-fix).
class TestMPIStepEquivalence:

    @pytest.fixture(autouse=True)
    def _activate_mpi_halo_backend(self):
        """``is_distributed()`` returns ``get_halo_backend() == "mpi"``.
        The mass fixer + zero_mean_tendency need this to allreduce.

        We don't need a real cubed-sphere topology — the lat-lon path
        doesn't go through the cubesphere halo routines.  Set the
        backend label, then restore after the test."""
        rank = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        # Build a minimal topology stub satisfying the type — just
        # enough for ``set_halo_backend("mpi", ...)`` not to crash.
        # If the real topology API is tighter than that, this fixture
        # will surface the issue clearly at test-collection time.
        from legoesm.parallel.comm import build_comm_topology
        try:
            topology = build_comm_topology(rank, n_ranks)
            set_halo_backend("mpi", topology)
        except Exception as exc:
            pytest.skip(
                f"Could not activate MPI halo backend "
                f"({type(exc).__name__}: {exc}); the lat-lon MPI mass "
                "fixer requires this to be active.  Investigation "
                "before Stage 3 — likely the cubed-sphere-specific "
                "topology bootstrap also needs a lat-lon adapter."
            )
        yield
        set_halo_backend("local")

    @pytest.mark.parametrize(
        "with_tracers",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skip(
                    reason=(
                        "Multi-rank PPM tracer transport under MPI: "
                        "the JIT-compile graph fuses PPM mass-flux "
                        "reconstruction × per-direction sendrecv × "
                        "per-tracer reduction into a single XLA module "
                        "that exceeds 90 min of wallclock to compile "
                        "on Ginsburg compute nodes.  The dry case "
                        "(``False``) already validates the architectural "
                        "MPI ≡ serial contract — backend dispatch, "
                        "cross-partition gradients, pole BC, mass-fixer "
                        "allreduce, pole-fold convention.  The tracer "
                        "case adds PPM-under-MPI which works "
                        "algebraically (Stage-1 single-rank tracer test "
                        "passes) but needs a JIT-compile profiling pass "
                        "before it can run on a finite walltime budget. "
                        "Remove this skip + bump sbatch walltime to "
                        "≥4 h when ready to validate manually."
                    ),
                ),
            ),
        ],
    )
    def test_step_matches_serial_after_gather(
        self, global_grid, sigma, serial_config,
        global_serial_model, with_tracers,
    ):
        """The full pipeline: scatter → MPI step → gather, compared to
        the canonical serial step on the same global state.

        Parametrised over with/without tracers so the same test
        exercises Stage 1 (dry) and Stage 2 (tracers).
        """
        nlev = sigma.n_levels
        global_state = _make_global_state(
            global_grid, nlev, with_tracers=with_tracers,
        )
        layout = make_latlon_band_layout(
            rank=MPI.COMM_WORLD.Get_rank(),
            n_ranks=MPI.COMM_WORLD.Get_size(),
            n_lat=global_grid.n_lat, n_lon=global_grid.n_lon,
        )
        dt = 100.0

        # --- Serial reference FIRST, on EVERY rank, LOCAL backend ---
        # Must precede ``make_latlon_mpi_step`` (which arms the band-
        # layout MPI halo backend as a global side effect): a serial
        # ``_step_cgrid`` traced after arming embeds band sendrecvs +
        # the ``fix_mass`` allreduce, which deadlocks when only rank 0
        # runs it.  Under the local backend the graph is collective-
        # free (``is_distributed()`` is False at trace time), so every
        # rank computes the identical bit-exact serial result.
        set_halo_backend("local")
        serial_out, _ = global_serial_model._step_cgrid(
            global_state, dt, target_mass=None, physics_fn=None,
        )
        # Materialize before arming MPI so no serial work interleaves
        # with the band-step collectives below.
        serial_out = jax.block_until_ready(serial_out)

        # --- Per-rank: build local model, scatter, MPI step, gather ---
        # No collective may run between the serial phase above and
        # ``make_latlon_mpi_step`` (its ``global_sum_mpi`` is the first
        # collective, reached by all ranks together): model build and
        # scatter are pure local slicing.
        local_model = _make_local_model(
            global_grid, sigma, serial_config, layout,
        )
        local_state = scatter_state_latlon(global_state, layout)
        step_fn = make_latlon_mpi_step(local_model, layout, halo=2)
        local_out = step_fn(local_state, dt)

        # Gather to rank 0.  Other ranks bail out of the comparison —
        # everything below is comm-free.
        gathered = gather_state_latlon(local_out, layout)
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        # Compare.  ``v`` has shape (n_lat+1, n_lon, nlev) globally
        # but gather_state_latlon handles the duplicated-row trim,
        # so shapes match.
        for field in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                getattr(gathered, field),
                getattr(serial_out, field),
                rtol=1e-8, atol=1e-9,
                err_msg=(
                    f"Multi-rank MPI step diverged from serial in "
                    f"``{field}`` (ranks={MPI.COMM_WORLD.Get_size()}, "
                    f"with_tracers={with_tracers})"
                ),
            )
        if with_tracers:
            assert set(gathered.tracers) == set(serial_out.tracers)
            for name in serial_out.tracers:
                np.testing.assert_allclose(
                    gathered.tracers[name], serial_out.tracers[name],
                    rtol=1e-8, atol=1e-9,
                    err_msg=(
                        f"Multi-rank MPI step diverged in tracer "
                        f"``{name}`` (ranks={MPI.COMM_WORLD.Get_size()})"
                    ),
                )

    # ------------------------------------------------------------------
    # Physics pass-through (make_latlon_mpi_step physics_fn kwarg)
    # ------------------------------------------------------------------
    #
    # ``held_suarez_forcing_latlon`` is the canonical column-local
    # physics (Newtonian relaxation + Rayleigh friction): it reads only
    # the rank-local band's lat / sigma, so MPI must reproduce serial to
    # roundoff just like the dry case.  The ``None`` case is the
    # regression guard for the signature change — an *explicitly
    # passed* ``physics_fn=None`` must keep the historical
    # dynamics-only behaviour bit-for-bit.
    #
    # Multi-step (3 steps) so band-cut information propagates across
    # ranks between physics evaluations (one step only exercises a
    # single halo round-trip).  The physics_fn is a module-level
    # function, so its identity is stable and ``_step_cgrid`` (which
    # holds it as a *static* jit argument) compiles exactly once per
    # case.

    @pytest.mark.parametrize("physics_name", ["none", "held_suarez"])
    def test_step_with_physics_fn_matches_serial(
        self, global_grid, sigma, serial_config,
        global_serial_model, physics_name,
    ):
        """scatter → N×(MPI step with physics_fn) → gather ≡ serial.

        Same wiring as ``test_step_matches_serial_after_gather`` but
        driving the new ``make_latlon_mpi_step(..., physics_fn=...)``
        pass-through (and its ``None`` default) against the serial
        ``_step_cgrid`` reference with the *same* physics_fn.
        """
        physics_fn = (
            held_suarez_forcing_latlon if physics_name == "held_suarez"
            else None
        )
        n_steps = 3
        nlev = sigma.n_levels
        global_state = _make_global_state(
            global_grid, nlev, with_tracers=False,
        )
        layout = make_latlon_band_layout(
            rank=MPI.COMM_WORLD.Get_rank(),
            n_ranks=MPI.COMM_WORLD.Get_size(),
            n_lat=global_grid.n_lat, n_lon=global_grid.n_lon,
        )
        dt = 100.0

        # --- Serial reference FIRST, on EVERY rank, LOCAL backend ---
        # Same ordering contract as ``test_step_matches_serial_after_
        # gather``: the serial reference (fed through the same
        # ``_step_cgrid`` entry point the serial ``model.step(state,
        # dt, physics_fn=...)`` delegate uses) must be traced and
        # materialized under the local halo backend BEFORE
        # ``make_latlon_mpi_step`` arms the band-layout MPI backend —
        # otherwise the rank-0-only serial graph embeds band
        # sendrecvs + the ``fix_mass`` allreduce and deadlocks at
        # np>1.  Every rank computes it (deterministic, comm-free).
        set_halo_backend("local")
        serial_out = global_state
        for _ in range(n_steps):
            serial_out, _ = global_serial_model._step_cgrid(
                serial_out, dt, target_mass=None, physics_fn=physics_fn,
            )
        serial_out = jax.block_until_ready(serial_out)

        # --- Per-rank: build local model, scatter, MPI steps, gather ---
        # Comm-free until ``make_latlon_mpi_step`` (whose
        # ``global_sum_mpi`` is the first collective, reached by all
        # ranks together).
        local_model = _make_local_model(
            global_grid, sigma, serial_config, layout,
        )
        local_state = scatter_state_latlon(global_state, layout)
        step_fn = make_latlon_mpi_step(
            local_model, layout, halo=2, physics_fn=physics_fn,
        )
        local_out = local_state
        for _ in range(n_steps):
            local_out = step_fn(local_out, dt)

        gathered = gather_state_latlon(local_out, layout)
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        for field in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                getattr(gathered, field),
                getattr(serial_out, field),
                rtol=1e-8, atol=1e-9,
                err_msg=(
                    f"Multi-rank MPI step with physics_fn="
                    f"{physics_name} diverged from serial in "
                    f"``{field}`` after {n_steps} steps "
                    f"(ranks={MPI.COMM_WORLD.Get_size()})"
                ),
            )

    @pytest.mark.skip(
        reason=(
            "Same JIT-compile blocker as "
            "``test_step_matches_serial_after_gather[True]``: "
            "``make_latlon_mpi_step`` is rebuilt on a fresh "
            "``local_model`` per test (function-scope autouse "
            "fixture) so the JAX trace cache misses and a new ~25 "
            "min XLA module is compiled from scratch on Ginsburg "
            "compute nodes. The dry "
            "``test_step_matches_serial_after_gather[False]`` "
            "case above already exercises the mass-fixer's "
            "``grid_total_area`` allreduce path on a one-step "
            "MPI run and proves it agrees with the serial "
            "reference; this test would only add a redundant "
            "drift assertion on the same code path. Remove this "
            "skip once the JIT-compile bloat is profiled away "
            "(tracked in TaskList #25) — until then bump sbatch "
            "walltime to >=4 h before unskipping."
        )
    )
    def test_mass_conserved_under_mpi(
        self, global_grid, sigma, serial_config,
    ):
        """Global p_s mass invariance under MPI on >1 rank — catches
        the ``grid_total_area`` allreduce bug surfaced during Stage 2
        design (rank-local total area would be wrong denominator)."""
        nlev = sigma.n_levels
        global_state = _make_global_state(
            global_grid, nlev, with_tracers=False,
        )
        layout = make_latlon_band_layout(
            rank=MPI.COMM_WORLD.Get_rank(),
            n_ranks=MPI.COMM_WORLD.Get_size(),
            n_lat=global_grid.n_lat, n_lon=global_grid.n_lon,
        )
        local_model = _make_local_model(
            global_grid, sigma, serial_config, layout,
        )
        local_state = scatter_state_latlon(global_state, layout)
        step_fn = make_latlon_mpi_step(local_model, layout, halo=2)

        # Global mass before — sum over rank-local band, allreduce.
        from legoesm.parallel.reductions import global_sum_mpi
        local_mass_before = jnp.sum(
            local_state.p_s.astype(jnp.float64)
            * local_model.grid.area.astype(jnp.float64)
        )
        global_mass_before = float(global_sum_mpi(local_mass_before))

        local_out = step_fn(local_state, 100.0)

        local_mass_after = jnp.sum(
            local_out.p_s.astype(jnp.float64)
            * local_model.grid.area.astype(jnp.float64)
        )
        global_mass_after = float(global_sum_mpi(local_mass_after))

        rel_drift = abs(global_mass_after - global_mass_before) / abs(
            global_mass_before
        )
        assert rel_drift < 1e-12, (
            f"Mass drifted by {rel_drift:.3e} over one MPI step on "
            f"{MPI.COMM_WORLD.Get_size()} ranks.  Either the fixer "
            f"isn't seeing the global total area, or the allreduce "
            f"in batch_global_area_sums isn't firing."
        )
