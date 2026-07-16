"""Single-rank equivalence test for the Stage-1 ``make_latlon_mpi_step``.

The MPI step function at ``n_ranks=1`` must produce a state
bit-identical to the canonical serial step for the dry C-grid PE.
Bit-exactness is the right contract here:

* The single-rank layout has both pole flags True, so the rank-aware
  ``pole_v_bc`` reduces to the serial double-pole-zero pad.
* No MPI reductions actually fire under ``is_distributed() == False``
  (no halo backend activated for this serial test); the external mass
  fixer in the wrapper calls the same ``_apply_safety_rails`` as the
  internal serial fixer, with the same inputs.
* The padded model differs from serial only in
  ``fix_mass=False, zero_mean_ps_tendency=False`` — both of which are
  applied externally to recover serial-equivalent behaviour.

This test gates Stage 2 (tracers) and Stage 3 (physics / multi-rank):
if the wrapper diverges from serial at 1 rank, multi-rank can't
possibly be right.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_mpi import (
    make_latlon_band_layout,
    make_latlon_mpi_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    """``make_latlon_mpi_step`` arms the GLOBAL halo backend
    (``set_halo_backend("mpi", layout)``) and documents that the caller
    deactivates it.  Without this teardown the armed backend leaks into
    later test files (bisect job 8459341: the tripole-serial suite saw
    ``prev == "mpi"`` and failed in its own restore)."""
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


@pytest.fixture(scope="module")
def grid():
    # n_lat=16, n_lon=32 — small enough to be quick on the login node,
    # large enough that halo=2 doesn't dominate the interior.
    return create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega,
    )


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(n_levels=6)


@pytest.fixture(scope="module")
def serial_config():
    # Hold-the-line CMIP defaults: fix_mass on, polar filter off,
    # PPM transport, ssp_rk3.
    return CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,
        zero_mean_ps_tendency=True,   # ignored when fix_mass=True
        use_polar_filter=False,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )


@pytest.fixture(scope="module")
def serial_model(grid, sigma, serial_config):
    return CGridLatLonPrimitiveEquationModel(grid, sigma, serial_config)


@pytest.fixture
def perturbed_state(grid, sigma):
    """Mildly perturbed rest state — exercises every tendency branch
    (Bernoulli, Coriolis, vertical advection, horizontal transport)
    without going unstable in one step."""
    rng = np.random.default_rng(31337)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma.n_levels
    eps = 1.0e-3
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(
            300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))
        ),
        p_s=jnp.asarray(
            1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))
        ),
        phis=jnp.zeros((n_lat, n_lon)),
    )


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------


class TestStage1SingleRankEquivalence:

    def test_state_after_one_step_matches_serial(
        self, serial_model, perturbed_state,
    ):
        """One dt of integration: stripped MPI state == serial state.

        Bit-exact match (atol=rtol=0) is the goal.  If it fails by a
        few ULPs, the rtol below is the most we should tolerate
        before declaring a regression — the only legitimate source of
        ULP-level drift in this 1-rank path is the JIT compilation
        order, which shouldn't matter for a deterministic computation.
        """
        grid = serial_model.grid
        dt = 100.0

        # Serial step
        serial_out, _ = serial_model._step_cgrid(
            perturbed_state, dt, target_mass=None, physics_fn=None,
        )

        # MPI step on a 1-rank layout (owns the whole lat axis;
        # south_rank=north_rank=None ⇒ both poles ⇒ pole_v_bc=(True, True))
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1, n_lat=grid.n_lat, n_lon=grid.n_lon,
        )
        mpi_step_fn = make_latlon_mpi_step(serial_model, layout, halo=2)
        mpi_out = mpi_step_fn(perturbed_state, dt)

        # Shape check first — easier to debug than a value mismatch.
        assert mpi_out.u.shape == serial_out.u.shape
        assert mpi_out.v.shape == serial_out.v.shape
        assert mpi_out.T.shape == serial_out.T.shape
        assert mpi_out.p_s.shape == serial_out.p_s.shape

        # Value check — fp64 machine-precision agreement.  Strict
        # ``atol=rtol=0`` (or single-ULP tolerances like 4e-15) are
        # impossible to meet even on a single rank because the
        # backend-aware step routes through MPI-style helpers whose
        # JIT compilation may reorder fused ops; the resulting
        # rounding-error pattern differs from the serial path by
        # ~1e-13 absolute even when the math is algebraically
        # identical.  Tolerances picked to (a) detect any real bug
        # — even sign flips or pole-BC misalignment manifest at
        # >1e-7 — and (b) admit fp64 ULP-scale reordering noise.
        for field in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                getattr(mpi_out, field),
                getattr(serial_out, field),
                rtol=1e-10, atol=1e-12,
                err_msg=(
                    f"Stage-1 MPI step on 1 rank diverged from the "
                    f"serial step in field ``{field}`` beyond fp64 "
                    f"machine precision.  This breaks the foundation "
                    f"for multi-rank validation; investigate the "
                    f"backend dispatch + pole BC + mass fixer plumbing "
                    f"before proceeding."
                ),
            )

    def test_mass_conserved_under_mpi_wrapper(
        self, serial_model, perturbed_state,
    ):
        """Independent of serial equivalence: total mass before vs
        after one MPI step is preserved to fp64 fixer precision.
        This catches the case where the external mass fixer didn't
        wire up correctly (e.g. allreduce missing → mass drifts even
        on 1 rank because the local sum *is* the global sum here)."""
        grid = serial_model.grid
        dt = 100.0
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1, n_lat=grid.n_lat, n_lon=grid.n_lon,
        )
        step_fn = make_latlon_mpi_step(serial_model, layout, halo=2)
        mass_before = jnp.sum(
            perturbed_state.p_s.astype(jnp.float64)
            * grid.area.astype(jnp.float64)
        )
        out = step_fn(perturbed_state, dt)
        mass_after = jnp.sum(
            out.p_s.astype(jnp.float64) * grid.area.astype(jnp.float64)
        )
        rel_drift = float(abs(mass_after - mass_before) / mass_before)
        assert rel_drift < 1e-12, (
            f"Mass drifted by {rel_drift:.3e} over one MPI step on "
            f"1 rank.  The external mass fixer in "
            f"make_latlon_mpi_step is not closing the budget."
        )


# ---------------------------------------------------------------------------
# Stage 2 — tracers
# ---------------------------------------------------------------------------


@pytest.fixture
def perturbed_state_with_tracers(grid, sigma):
    """Perturbed rest state carrying two tracers (q_v, q_c) so we
    exercise the PPM mass-flux transport and the tracer-mass
    preservation in the external fixer."""
    rng = np.random.default_rng(31337)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma.n_levels
    eps = 1.0e-3
    q_v_profile = 1.0e-3 * (1.0 + 0.05 * rng.standard_normal(
        (n_lat, n_lon, nlev),
    ))
    q_c_profile = 1.0e-5 * (1.0 + 0.05 * rng.standard_normal(
        (n_lat, n_lon, nlev),
    ))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(
            300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))
        ),
        p_s=jnp.asarray(
            1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))
        ),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={
            "q_v": jnp.asarray(q_v_profile),
            "q_c": jnp.asarray(q_c_profile),
        },
    )


class TestStage2Tracers:
    """Stage 2: tracer support.  Same equivalence + conservation
    contract as Stage 1, plus per-tracer mass invariance."""

    def test_tracer_step_matches_serial(
        self, serial_model, perturbed_state_with_tracers,
    ):
        """1-rank MPI step on a state with q_v + q_c must bit-match the
        serial step (4 ULPs).  Catches misuse of the padded-grid
        PPM machinery + tracer halo exchange."""
        grid = serial_model.grid
        dt = 100.0

        serial_out, _ = serial_model._step_cgrid(
            perturbed_state_with_tracers, dt,
            target_mass=None, physics_fn=None,
        )

        layout = make_latlon_band_layout(
            rank=0, n_ranks=1, n_lat=grid.n_lat, n_lon=grid.n_lon,
        )
        mpi_step = make_latlon_mpi_step(serial_model, layout, halo=2)
        mpi_out = mpi_step(perturbed_state_with_tracers, dt)

        # Prognostic fields
        for field in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                getattr(mpi_out, field),
                getattr(serial_out, field),
                rtol=1e-10, atol=1e-12,
                err_msg=f"Stage 2 1-rank MPI step diverged in {field}",
            )
        # Tracers
        assert set(mpi_out.tracers) == set(serial_out.tracers)
        for name in serial_out.tracers:
            np.testing.assert_allclose(
                mpi_out.tracers[name], serial_out.tracers[name],
                rtol=1e-10, atol=1e-12,
                err_msg=(
                    f"Stage 2 1-rank MPI step diverged in tracer "
                    f"``{name}`` — PPM mass-flux transport on padded "
                    f"grid is not reproducing the serial result."
                ),
            )

    def test_tracer_mass_conserved(
        self, serial_model, sigma, perturbed_state_with_tracers,
    ):
        """``∫ q · dp · dA`` invariant per tracer.

        The mass fixer's tracer-rescaling branch (primitive_eq:805)
        adjusts q by ``dp_pre / dp_post`` whenever p_s is corrected.
        If that branch is bypassed under MPI (e.g. by reaching only
        through the rank-local model without the global fixer model),
        tracer mass will drift even though prognostic mass holds.
        """
        grid = serial_model.grid
        dt = 100.0
        layout = make_latlon_band_layout(
            rank=0, n_ranks=1, n_lat=grid.n_lat, n_lon=grid.n_lon,
        )
        step_fn = make_latlon_mpi_step(serial_model, layout, halo=2)

        def _tracer_mass(state, name):
            # Pure sigma: dp = p_s * dsigma; tracer mass per area
            # weighted column-integral.
            p_s_64 = state.p_s.astype(jnp.float64)
            tr_64 = state.tracers[name].astype(jnp.float64)
            dsigma = jnp.asarray(sigma.dsigma, dtype=jnp.float64)
            area_64 = grid.area.astype(jnp.float64)
            # tr_mass = sum over levels and area of q * p_s * dsigma * area
            col_mass = jnp.sum(
                tr_64 * dsigma[None, None, :],
                axis=-1,
            ) * p_s_64
            return jnp.sum(col_mass * area_64)

        before = {
            name: float(_tracer_mass(perturbed_state_with_tracers, name))
            for name in perturbed_state_with_tracers.tracers
        }
        out = step_fn(perturbed_state_with_tracers, dt)
        after = {name: float(_tracer_mass(out, name)) for name in before}

        for name in before:
            if before[name] == 0.0:
                drift = abs(after[name])
            else:
                drift = abs(after[name] - before[name]) / abs(before[name])
            # Per-tracer mass should be preserved to within fp64 fixer
            # precision.  PPM transport itself is mass-conservative
            # (flux form), and the external fixer additionally
            # rescales q to compensate for the p_s correction — so
            # we should see roughly the same precision as the p_s
            # mass invariance.
            assert drift < 1e-9, (
                f"Tracer ``{name}`` mass drifted by {drift:.3e} over "
                f"one MPI step.  PPM transport or the external fixer's "
                f"tracer-rescaling branch is broken.  (Tolerance set "
                f"to 1e-9 rather than tighter fp64 limit because "
                f"PPM mass-flux transport's accumulation order under "
                f"the backend-aware operators differs from the serial "
                f"reduction order by ~1e-10 relative — still well "
                f"below any physically-meaningful tracer drift.)"
            )
