"""Tests for MPAS atmosphere dynamical cores (hydrostatic PE + non-hydrostatic CE).

Tests verify:
1. Tendency shapes are correct
2. Model step runs without errors
3. Finite output (no NaN/Inf)
4. Mass conservation (hydrostatic)
5. Multi-step stability
6. JAX differentiability
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import unittest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    MPASHydrostaticState,
    MPASHydrostaticTendencies,
    MPASNonHydrostaticState,
    MPASNonHydrostaticTendencies,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    MPASPrimitiveEquationModel,
    mpas_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
    MPASCompressibleEulerConfig,
    MPASCompressibleEulerModel,
    mpas_compressible_euler_slow_tendencies,
)


# ============================================================================
# Shared test fixtures
# ============================================================================

def _make_mesh(level=2):
    """Small icosahedral mesh for testing (level 2 = 162 cells)."""
    return create_voronoi_mesh(level, lloyd_iterations=5)


def _make_hydrostatic_state(mesh, nlev=5):
    """Create a simple rest state for the hydrostatic PE."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges

    T0 = 250.0  # reference temperature [K]
    p_s0 = 1e5  # surface pressure [Pa]

    u = Field(data=jnp.zeros((nEdges, nlev), dtype=jnp.float64),
              name="u", dims=("nEdges", "nlev"), units="m/s")
    T = Field(data=jnp.full((nCells, nlev), T0, dtype=jnp.float64),
              name="T", dims=("nCells", "nlev"), units="K")
    p_s = Field(data=jnp.full((nCells,), p_s0, dtype=jnp.float64),
                name="p_s", dims=("nCells",), units="Pa")
    phis = Field(data=jnp.zeros((nCells,), dtype=jnp.float64),
                 name="phis", dims=("nCells",), units="m²/s²")

    return MPASHydrostaticState(u=u, T=T, p_s=p_s, phis=phis)


def _make_nh_state(mesh, height_coord, terrain_metric, nlev=5):
    """Create a simple rest state for the non-hydrostatic CE."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges

    u = Field(data=jnp.zeros((nEdges, nlev), dtype=jnp.float64),
              name="u", dims=("nEdges", "nlev"), units="m/s")
    w = Field(data=jnp.zeros((nCells, nlev + 1), dtype=jnp.float64),
              name="w", dims=("nCells", "nlev_half"), units="m/s")
    theta_prime = Field(
        data=jnp.zeros((nCells, nlev), dtype=jnp.float64),
        name="theta_prime", dims=("nCells", "nlev"), units="K")
    rho_prime = Field(
        data=jnp.zeros((nCells, nlev), dtype=jnp.float64),
        name="rho_prime", dims=("nCells", "nlev"), units="kg/m³")
    phis = Field(data=jnp.zeros((nCells,), dtype=jnp.float64),
                 name="phis", dims=("nCells",), units="m²/s²")
    tracers = Field(
        data=jnp.zeros((nCells, nlev, 0), dtype=jnp.float64),
        name="tracers", dims=("nCells", "nlev", "tracer"), units="kg/kg")

    return MPASNonHydrostaticState(
        u=u, w=w, theta_prime=theta_prime, rho_prime=rho_prime,
        phis=phis, tracers=tracers,
    )


def _add_perturbation_hydro(state, mesh, nlev):
    """Add a small wind perturbation for non-trivial tendencies."""
    nEdges = mesh.nEdges
    key = jax.random.PRNGKey(42)
    u_pert = 1.0 * jax.random.normal(key, (nEdges, nlev))
    return state._replace(
        u=state.u.replace(data=state.u.data + u_pert),
    )


def _add_perturbation_nh(state, mesh, nlev):
    """Add small perturbations for non-trivial NH tendencies."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    u_pert = 1.0 * jax.random.normal(k1, (nEdges, nlev))
    theta_pert = 0.5 * jax.random.normal(k2, (nCells, nlev))
    rho_pert = 0.001 * jax.random.normal(k3, (nCells, nlev))
    return state._replace(
        u=state.u.replace(data=state.u.data + u_pert),
        theta_prime=state.theta_prime.replace(
            data=state.theta_prime.data + theta_pert),
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + rho_pert),
    )


# ============================================================================
# Hydrostatic PE tests
# ============================================================================

class TestMPASHydrostaticPE(unittest.TestCase):
    """Tests for the MPAS hydrostatic primitive equations."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 5
        cls.mesh = _make_mesh(level=2)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        cls.config = MPASPrimitiveEquationConfig(
            nu_del2=1e4, K_h=1e3,
        )
        cls.state = _make_hydrostatic_state(cls.mesh, cls.nlev)
        cls.state_pert = _add_perturbation_hydro(
            cls.state, cls.mesh, cls.nlev)

    def test_tendency_shapes(self):
        """Tendencies have correct shapes."""
        tend = mpas_hydrostatic_tendencies(
            self.state_pert, self.mesh, self.sigma, self.config,
        )
        self.assertEqual(tend.du_dt.data.shape,
                         (self.mesh.nEdges, self.nlev))
        self.assertEqual(tend.dT_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.dp_s_dt.data.shape,
                         (self.mesh.nCells,))
        self.assertEqual(tend.dphis_dt.data.shape,
                         (self.mesh.nCells,))

    def test_tendencies_finite(self):
        """Tendencies contain no NaN or Inf."""
        tend = mpas_hydrostatic_tendencies(
            self.state_pert, self.mesh, self.sigma, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))

    def test_rest_state_zero_tendencies(self):
        """Rest state (u=0, uniform T, uniform p_s) has near-zero tendencies."""
        tend = mpas_hydrostatic_tendencies(
            self.state, self.mesh, self.sigma, self.config,
        )
        # At rest, all tendencies should be very small
        self.assertLess(float(jnp.max(jnp.abs(tend.du_dt.data))), 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(tend.dp_s_dt.data))), 1e-1)

    def test_unknown_pv_scheme_raises(self):
        """A typo in ``pv_scheme`` must raise, not silently select the
        energy-conserving PV flux."""
        bad = self.config._replace(pv_scheme="bogus")
        with self.assertRaisesRegex(ValueError, "Unknown pv_scheme"):
            mpas_hydrostatic_tendencies(
                self.state_pert, self.mesh, self.sigma, bad,
            )

    def test_model_step(self):
        """Model.step() produces finite state."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        state_new = model.step(self.state_pert, dt=60.0)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.p_s.data)))

    def test_mass_conservation(self):
        """Mass (sum of p_s * area) is conserved after one step."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        area = self.mesh.areaCell
        mass_old = float(jnp.sum(self.state_pert.p_s.data * area))
        state_new = model.step(self.state_pert, dt=60.0)
        mass_new = float(jnp.sum(state_new.p_s.data * area))
        rel_err = abs(mass_new - mass_old) / abs(mass_old)
        self.assertLess(rel_err, 1e-12)

    def test_multi_step_stability(self):
        """3 steps without NaN."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        state = self.state_pert
        for _ in range(3):
            state = model.step(state, dt=60.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.p_s.data)))

    def test_differentiability(self):
        """Tendencies are differentiable w.r.t. velocity."""
        def loss(u_data):
            s = self.state._replace(
                u=self.state.u.replace(data=u_data))
            tend = mpas_hydrostatic_tendencies(
                s, self.mesh, self.sigma, self.config)
            return jnp.sum(tend.du_dt.data ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(self.state_pert.u.data)
        self.assertEqual(g.shape, self.state_pert.u.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


class TestMPASHydrostaticIntegratorStability(unittest.TestCase):
    """Why the MPAS hydrostatic PE defaults to ``ssp_rk54``, not ``ssp_rk3``.

    Measured on GPU (jobs 8087100 / 8088578) and reproduced here on the
    CPU x64 path CI uses:

    * With **zero dissipation** (the bare ``MPASPrimitiveEquationConfig``)
      BOTH ``ssp_rk3`` and ``ssp_rk54`` are stable to dt >= 600 s on
      L4/nlev30 from a near-rest IC — the integrator choice makes no
      difference (``test_both_stable_without_dissipation``).
    * With the operational ``del2`` + ``del4`` hyperdiffusion active,
      ``ssp_rk3`` diverges within a few steps at dt = 600 s while
      ``ssp_rk54`` stays bounded
      (``test_ssp_rk3_blows_up_with_hyperdiffusion`` vs
      ``test_default_stable_with_hyperdiffusion_at_large_dt``).
      ``ssp_rk3`` has the smaller absolute-stability region; the
      (negative-real-axis) eigenvalues of the biharmonic ∇⁴ operator fall
      outside it at this dt, whereas the 5-stage ``ssp_rk54`` region
      contains them.

    So the integrator default only bites once hyperdiffusion is on — but a
    long production run *should* damp grid-scale noise with hyperdiffusion,
    and ``ssp_rk54`` also matches the spectral-PE default.

    NOTE this is **not** the original "undamped gravity wave sitting on the
    imaginary axis / hidden-CFL at ~300 s any resolution" story — the
    zero-dissipation sweep disproved that (both integrators run to dt=600).
    The separate ~450 s dt-ceiling seen with the *gray AMIP deck* is a
    radiative startup transient (T=300 K isothermal IC), integrator- and
    dissipation-independent; it is not exercised here.
    """

    @classmethod
    def setUpClass(cls):
        cls.nlev = 30
        cls.mesh = create_voronoi_mesh(4, lloyd_iterations=5)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        # Operational hyperdiffusion coefficients — the regime in which the
        # integrator choice matters.  Same del2/del4 scaling as the GPU
        # integrator sweep (``_diag_mpas_integrator_sweep.py``).
        dx = float(jnp.min(cls.mesh.dcEdge))
        cls.A_h = 0.05 * dx ** 2 / 600.0
        cls.nu4 = dx ** 4 / (24.0 * 3600.0)
        cls.state_pert = _add_perturbation_hydro(
            _make_hydrostatic_state(cls.mesh, cls.nlev), cls.mesh, cls.nlev)

    def _cfg(self, integ, with_hyperdiff):
        kw = dict(time_integrator=integ)
        if with_hyperdiff:
            kw.update(nu_del2=self.A_h, K_h=self.A_h,
                      nu_del4=self.nu4, nu_del4_ps=self.nu4)
        return MPASPrimitiveEquationConfig(**kw)

    def _blow_step(self, cfg, dt, n_steps=24):
        """Step ``n_steps`` and return (first step index where |u| > 1e3 or
        non-finite, final max|u|).  Index -1 => stayed bounded throughout."""
        model = MPASPrimitiveEquationModel(self.mesh, self.sigma, cfg)
        state = self.state_pert
        for s in range(n_steps):
            state = model.step(state, dt=dt)
            mu = float(jnp.max(jnp.abs(state.u.data)))
            if (not jnp.isfinite(mu)) or mu > 1.0e3:
                return s, mu
        return -1, float(jnp.max(jnp.abs(state.u.data)))

    def test_default_integrator_is_large_stability_ssp(self):
        """Default must be a large-stability SSP scheme — NOT ssp_rk3.

        Both ssp_rk54 and ssp_rk34 were measured stable to dt=600 s with
        hyperdiffusion (job 8087100); ssp_rk3 was not.  ``ssp_rk54_scan`` is
        the SAME Spiteri-Ruuth scheme as ``ssp_rk54`` (identical stability
        region — only the tendency is compiled once instead of inlined five
        times), so it is equally acceptable as the default and is in fact
        preferred for the gather-heavy MPAS tendency (~2.5x faster).  ssp_rk3
        remains forbidden as the default: it diverges with the operational
        ∇⁴ operator at dt=600 (see ``test_ssp_rk3_blows_up_with_hyperdiffusion``).
        """
        integ = MPASPrimitiveEquationConfig().time_integrator
        self.assertIn(integ, ("ssp_rk54", "ssp_rk54_scan", "ssp_rk34"),
                      msg=f"MPAS PE default integrator {integ!r} cannot "
                          f"tolerate the operational hyperdiffusion operator")

    def test_default_integrator_is_exactly_scan_fold(self):
        """The default must be the SCAN-folded variant specifically.

        The membership test above also accepts inline ``ssp_rk54`` because the
        two are the identical scheme — but they are NOT equivalent in cost: the
        inline form de-vectorizes the gather-heavy TRiSK tendency on XLA-CPU
        (~2.5x slower; the 30-50x slowdown this branch fixes).  A merge or edit
        that silently reverts the default to inline ``ssp_rk54`` would pass
        every equivalence/stability/AD test while losing the perf win, so pin
        the exact value (cf. the aimip_312 'merge silently reverts the targeted
        fix' failure mode).
        """
        self.assertEqual(
            MPASPrimitiveEquationConfig().time_integrator, "ssp_rk54_scan",
            msg="MPAS PE default reverted off the scan-folded integrator — the "
                "gather-heavy TRiSK perf fix is lost (inline ssp_rk54 is ~2.5x "
                "slower despite being the same scheme).")

    def test_both_stable_without_dissipation(self):
        """Zero dissipation: BOTH integrators stay bounded at dt=600 s.

        Pins the finding that disproved the old "ssp_rk3 blows at any dt
        above ~300 s" rationale — with no hyperdiffusion the integrator
        choice is irrelevant for stability.
        """
        for integ in ("ssp_rk3", "ssp_rk54"):
            blew, mu = self._blow_step(self._cfg(integ, with_hyperdiff=False),
                                       dt=600.0)
            self.assertEqual(blew, -1,
                             msg=f"{integ} unexpectedly blew at step {blew} "
                                 f"(max|u|={mu:.3g}) at dt=600 with zero "
                                 f"dissipation")

    def test_default_stable_with_hyperdiffusion_at_large_dt(self):
        """ssp_rk54 (default) stays finite + bounded at dt=600 s WITH the
        operational hyperdiffusion that makes ssp_rk3 diverge."""
        cfg = self._cfg("ssp_rk54", with_hyperdiff=True)
        model = MPASPrimitiveEquationModel(self.mesh, self.sigma, cfg)
        state = self.state_pert
        for _ in range(24):
            state = model.step(state, dt=600.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.p_s.data)))
        self.assertLess(float(jnp.max(jnp.abs(state.u.data))), 100.0)

    def test_ssp_rk54_scan_stable_with_hyperdiffusion_at_large_dt(self):
        """The scan-folded default (``ssp_rk54_scan``) has the SAME stability
        as inline ``ssp_rk54``: bounded at dt=600 s WITH the operational
        hyperdiffusion that makes ssp_rk3 diverge.  Pins that the compile-time
        fold did not alter the stability region."""
        cfg = self._cfg("ssp_rk54_scan", with_hyperdiff=True)
        model = MPASPrimitiveEquationModel(self.mesh, self.sigma, cfg)
        state = self.state_pert
        for _ in range(24):
            state = model.step(state, dt=600.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.p_s.data)))
        self.assertLess(float(jnp.max(jnp.abs(state.u.data))), 100.0)

    def test_ssp_rk3_blows_up_with_hyperdiffusion(self):
        """The reason for the default: ssp_rk3 cannot tolerate the ∇⁴
        operator's eigenvalues at dt=600 s and diverges within a few steps
        (measured blow-step ~3), while ssp_rk54 stays bounded above.

        If ssp_rk3 ever becomes stable here the integrator-default rationale
        should be revisited (e.g. a smaller hyperdiffusion default).
        """
        blew, mu = self._blow_step(self._cfg("ssp_rk3", with_hyperdiff=True),
                                   dt=600.0)
        self.assertGreaterEqual(
            blew, 0,
            msg="ssp_rk3 unexpectedly stable at dt=600 WITH hyperdiffusion on "
                "L4/nlev30 — revisit the integrator-default rationale")


class TestMPASSurfaceForcingThreading(unittest.TestCase):
    """Phase A: a per-step TRACED ``forcing={'T_sfc': ...}`` threaded through
    ``MPASPrimitiveEquationModel.step`` -> combined physics -> radiation
    reaches the surface boundary (time-varying SST), and ``forcing=None``
    keeps the legacy dry default (radiation surface = ``T[..., -1]``)."""

    @classmethod
    def setUpClass(cls):
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        cls.nlev = 20
        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        cls.state = _make_hydrostatic_state(cls.mesh, cls.nlev)
        cls.ncell = cls.mesh.nCells
        # gray radiation only; everything else off (mirrors the MPAS AMIP run
        # path — the bare PhysicsConfig() defaults are not all-"none").
        # staticmethod: a bare function stored as a class attr would be bound
        # as a method (passing ``self`` as the first arg = ``state``).
        cls.physics_fn = staticmethod(make_physics(
            PhysicsConfig(
                radiation=RadiationConfig(scheme="gray"),
                convection=ConvectionConfig(scheme="none"),
                turbulence=TurbulenceConfig(scheme="none"),
                microphysics=MicrophysicsConfig(scheme="none"),
                gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
            ),
            model_type="mpas", dt=300.0))

    def _dT_dt(self, forcing):
        res = self.physics_fn(self.state, self.mesh, self.sigma, forcing=forcing)
        tend = res[0] if isinstance(res, tuple) else res
        return tend.dT_dt.data

    def test_T_sfc_forcing_changes_radiative_heating(self):
        """A warm vs cold prescribed T_sfc drives different surface longwave,
        hence a different radiative heating profile — proving the traced
        forcing actually reaches the radiation kernel."""
        warm = self._dT_dt({"T_sfc": jnp.full((self.ncell,), 300.0)})
        cold = self._dT_dt({"T_sfc": jnp.full((self.ncell,), 250.0)})
        self.assertTrue(jnp.all(jnp.isfinite(warm)))
        self.assertTrue(jnp.all(jnp.isfinite(cold)))
        self.assertGreater(float(jnp.max(jnp.abs(warm - cold))), 1e-8,
                           msg="T_sfc forcing did not change the radiative "
                               "heating — forcing is not reaching radiation")

    def test_step_threads_forcing(self):
        """End-to-end: model.step(..., forcing=...) advances to different
        states for different prescribed SST (the AMIP coupling path)."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, MPASPrimitiveEquationConfig())
        s_warm = model.step(self.state, 300.0, physics_fn=self.physics_fn,
                            forcing={"T_sfc": jnp.full((self.ncell,), 300.0)})
        s_cold = model.step(self.state, 300.0, physics_fn=self.physics_fn,
                            forcing={"T_sfc": jnp.full((self.ncell,), 250.0)})
        self.assertTrue(jnp.all(jnp.isfinite(s_warm.T.data)))
        self.assertGreater(float(jnp.max(jnp.abs(s_warm.T.data - s_cold.T.data))),
                           1e-12, msg="prescribed SST had no effect on the step")

    def test_no_forcing_runs_dry_default(self):
        """forcing=None (legacy / dry) still steps cleanly — backward compat."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, MPASPrimitiveEquationConfig())
        s = model.step(self.state, 300.0, physics_fn=self.physics_fn)
        self.assertTrue(jnp.all(jnp.isfinite(s.T.data)))


class TestMPASMoistureTransport(unittest.TestCase):
    """Phase B: prognostic tracers (moisture) advected by the MPAS PE with the
    dycore's OWN wind + vertical mass flux (mass-consistent), via the shared
    ``tracer_horizontal_advection`` kernel + the same vertical operator the
    dycore uses for T."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 16
        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        cls.model = MPASPrimitiveEquationModel(
            cls.mesh, cls.sigma, MPASPrimitiveEquationConfig())

    def _state_with_q(self, q_field):
        base = _add_perturbation_hydro(
            _make_hydrostatic_state(self.mesh, self.nlev), self.mesh, self.nlev)
        return base._replace(tracers={"q_v": Field(
            data=q_field, name="q_v", dims=("nCells", "nlev"), units="kg/kg")})

    def test_uniform_tracer_preserved(self):
        """A UNIFORM tracer must stay uniform + constant under advection — the
        advective form ``-(div(q u) - q div(u))`` cancels exactly for constant
        q.  This is the core correctness check that the kernel + RHS wiring +
        pytree integrator all line up (a flux-form bug would drift it)."""
        q0 = jnp.full((self.mesh.nCells, self.nlev), 0.01, dtype=jnp.float64)
        state = self._state_with_q(q0)
        for _ in range(6):
            state = self.model.step(state, dt=300.0)
        q = state.tracers["q_v"].data
        self.assertTrue(jnp.all(jnp.isfinite(q)))
        self.assertLess(float(jnp.max(jnp.abs(q - 0.01))), 1e-8,
                        msg="uniform tracer drifted — advective-form "
                            "div(u) cancellation or integrator wiring is wrong")

    def test_tracer_advects_and_stays_finite(self):
        """A non-uniform tracer blob advects (changes) but stays finite +
        bounded — no spurious blow-up from the transport."""
        lat = jnp.asarray(self.mesh.latCell)
        blob = (0.01 * jnp.exp(-((lat - 0.3) / 0.3) ** 2))[:, None]
        q0 = jnp.broadcast_to(blob, (self.mesh.nCells, self.nlev))
        state = self._state_with_q(q0)
        s1 = self.model.step(state, dt=300.0)
        q1 = s1.tracers["q_v"].data
        self.assertTrue(jnp.all(jnp.isfinite(q1)))
        self.assertLess(float(jnp.max(jnp.abs(q1))), 1.0)  # bounded
        self.assertGreater(float(jnp.max(jnp.abs(q1 - q0))), 0.0)  # advected

    def test_dry_state_unchanged_path(self):
        """tracers=None (dry) still steps cleanly + carries no tracers —
        backward compatibility for the moisture wiring."""
        dry = _add_perturbation_hydro(
            _make_hydrostatic_state(self.mesh, self.nlev), self.mesh, self.nlev)
        s = self.model.step(dry, dt=300.0)
        self.assertTrue(jnp.all(jnp.isfinite(s.T.data)))
        self.assertIsNone(s.tracers)


class TestMPASOperatorSplitPhysics(unittest.TestCase):
    """The MPAS step applies physics OPERATOR-SPLIT: evaluated once on the
    post-dynamics state and applied FORWARD over the full dt (``state +=
    dt·tendency``) — NOT interleaved per RK stage.  A regression that moved
    physics back into the per-RK-stage tendency would apply only
    ~dt/n_stages of an additive tendency and silently pass the other tests;
    this pins the dt-scaling."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 12
        cls.mesh = create_voronoi_mesh(3, lloyd_iterations=3)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        cls.state = _add_perturbation_hydro(
            _make_hydrostatic_state(cls.mesh, cls.nlev), cls.mesh, cls.nlev)
        cls.model = MPASPrimitiveEquationModel(
            cls.mesh, cls.sigma, MPASPrimitiveEquationConfig())

    def test_physics_applied_once_forward_over_dt(self):
        DT = 100.0
        K = 1.5e-3  # constant heating rate [K/s]

        def const_phys(state, mesh, sigma_coord, phys_state=None, forcing=None):
            zp = jnp.zeros_like(state.p_s.data)
            tend = MPASHydrostaticTendencies(
                du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
                dT_dt=state.T.replace(data=jnp.full_like(state.T.data, K)),
                dp_s_dt=state.p_s.replace(data=zp),
                dphis_dt=state.phis.replace(data=zp),
            )
            return tend, None

        # Dynamics is identical in both (physics is split out of the RK), so
        # the difference is exactly the once-forward physics contribution.
        s_phys = self.model.step(self.state, DT, physics_fn=const_phys)
        s_dyn = self.model.step(self.state, DT)  # dynamics only
        dT = s_phys.T.data - s_dyn.T.data
        self.assertTrue(
            jnp.allclose(dT, DT * K, atol=1e-6),
            msg=f"operator-split mis-applied: ΔT mean={float(jnp.mean(dT)):.5f} "
                f"expected {DT * K:.5f} (per-stage would give ~dt/n_stages·K)")


# ============================================================================
# Non-hydrostatic CE tests
# ============================================================================

class TestMPASNonHydrostaticCE(unittest.TestCase):
    """Tests for the MPAS non-hydrostatic compressible Euler equations."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 5
        cls.H = 30000.0
        cls.mesh = _make_mesh(level=2)
        cls.height_coord = create_height_coordinate(cls.nlev, cls.H)
        z_s = jnp.zeros(cls.mesh.nCells, dtype=jnp.float64)
        cls.terrain_metric = compute_terrain_metric(z_s, cls.height_coord)
        cls.config = MPASCompressibleEulerConfig(
            nu_del2=1e4, n_acoustic_substeps=4,
        )
        cls.state = _make_nh_state(
            cls.mesh, cls.height_coord, cls.terrain_metric, cls.nlev,
        )
        cls.state_pert = _add_perturbation_nh(
            cls.state, cls.mesh, cls.nlev)

    def test_slow_tendency_shapes(self):
        """Slow tendencies have correct shapes."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state_pert, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertEqual(tend.du_dt.data.shape,
                         (self.mesh.nEdges, self.nlev))
        self.assertEqual(tend.dw_dt.data.shape,
                         (self.mesh.nCells, self.nlev + 1))
        self.assertEqual(tend.dtheta_prime_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.drho_prime_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.dtracers_dt.data.shape,
                         (self.mesh.nCells, self.nlev, 0))

    def test_slow_tendencies_finite(self):
        """Slow tendencies contain no NaN or Inf."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state_pert, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dw_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.drho_prime_dt.data)))

    def test_unknown_pv_scheme_raises(self):
        """A typo in ``pv_scheme`` must raise, not silently select the
        energy-conserving PV flux."""
        bad = self.config._replace(pv_scheme="bogus")
        with self.assertRaisesRegex(ValueError, "Unknown pv_scheme"):
            mpas_compressible_euler_slow_tendencies(
                self.state_pert, self.mesh, self.height_coord,
                self.terrain_metric, bad,
            )

    def test_model_step(self):
        """Model.step() produces finite state."""
        model = MPASCompressibleEulerModel(
            self.mesh, self.height_coord, self.terrain_metric, self.config,
        )
        state_new = model.step(self.state_pert, dt=5.0)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.w.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.theta_prime.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.rho_prime.data)))

    def test_rest_state_small_tendencies(self):
        """Rest state has near-zero slow tendencies."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertLess(float(jnp.max(jnp.abs(tend.du_dt.data))), 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))), 1e-3)

    def test_multi_step_stability(self):
        """3 steps without NaN."""
        model = MPASCompressibleEulerModel(
            self.mesh, self.height_coord, self.terrain_metric, self.config,
        )
        state = self.state_pert
        for _ in range(3):
            state = model.step(state, dt=5.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.w.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.theta_prime.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.rho_prime.data)))

    def test_with_tracers(self):
        """Model runs with tracers."""
        nCells = self.mesh.nCells
        n_tracers = 2
        tracers = Field(
            data=jnp.ones((nCells, self.nlev, n_tracers), dtype=jnp.float64),
            name="tracers", dims=("nCells", "nlev", "tracer"), units="kg/kg",
        )
        state = self.state_pert._replace(tracers=tracers)
        tend = mpas_compressible_euler_slow_tendencies(
            state, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertEqual(tend.dtracers_dt.data.shape,
                         (nCells, self.nlev, n_tracers))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))

    def test_differentiability(self):
        """Slow tendencies are differentiable w.r.t. theta_prime."""
        def loss(theta_p_data):
            s = self.state._replace(
                theta_prime=self.state.theta_prime.replace(data=theta_p_data))
            tend = mpas_compressible_euler_slow_tendencies(
                s, self.mesh, self.height_coord,
                self.terrain_metric, self.config)
            return jnp.sum(tend.du_dt.data ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(self.state_pert.theta_prime.data)
        self.assertEqual(g.shape, self.state_pert.theta_prime.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


class TestMPASConfig(unittest.TestCase):
    """Tests for MPASPrimitiveEquationConfig fields."""

    def test_p_floor_default(self):
        """p_floor defaults to 100 Pa."""
        cfg = MPASPrimitiveEquationConfig()
        self.assertEqual(cfg.p_floor, 100.0)

    def test_p_floor_custom(self):
        """p_floor can be overridden."""
        cfg = MPASPrimitiveEquationConfig(p_floor=50.0)
        self.assertEqual(cfg.p_floor, 50.0)

    def test_p_floor_used_in_tendencies(self):
        """p_floor is accessible and positive in the config passed to tendencies."""
        cfg = MPASPrimitiveEquationConfig(p_floor=200.0)
        self.assertGreater(cfg.p_floor, 0.0)
        # Verify it's a valid NamedTuple field
        self.assertIn("p_floor", cfg._fields)


if __name__ == "__main__":
    unittest.main()
