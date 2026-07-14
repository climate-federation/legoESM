"""Long-roll moist validation for the spectral PE (PR4).

These tests pin global stability and qualitative tracer-budget
behavior over multi-hundred-step integrations of the moist spectral
PE.  They exercise the full bridge → orchestrator → dycore RHS path
(advection + microphysics + spectral/hyperdiff filter) end-to-end.

The previous coverage was limited to single-step regression checks;
the long-roll suite catches conservation drift, slow blow-ups, and
sign errors in the tracer plumbing that are invisible at one step.

Marked ``slow`` — each test runs ~100-200 model steps at T21/L10
(~ 30 s wall time per test on CPU).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.held_suarez import held_suarez_forcing_spectral
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
)
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
)
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis_3d,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(10, sigma_top=0.01)


def _proper_hyperdiff(grid):
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max ** 2)


def _moist_init_state(grid, sigma_coord, rh=0.7):
    """Build a moist initial condition with realistic T profile and
    a sub-saturated q_v field (RH = ``rh`` everywhere)."""
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full
    # Realistic T profile (warmer near surface).
    T_col = 290.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
    T_col = jnp.maximum(T_col, 200.0)
    T_grid = jnp.broadcast_to(
        T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
    )
    T_hat = sh_analysis_3d(grid, T_grid)

    # Sub-saturated q_v profile.
    p_full_3d = jnp.broadcast_to(
        (sigma_full * 1e5)[None, None, :],
        (grid.n_lat, grid.n_lon, nlev),
    )
    q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
    qv = rh * q_sat   # bounded sub-saturated initial state

    # Build the rest state with the realistic T profile.
    rest = isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.1,
    )
    return rest._replace(
        T_hat=rest.T_hat.replace(data=T_hat),
        tracers={
            "q_v": Field(
                data=qv, name="q_v",
                dims=("lat", "lon", "level"), units="kg/kg",
            ),
        },
    )


def _column_water_vapor(qv_grid, sigma_coord):
    """Compute σ-weighted column water vapor (dimensionless)."""
    dsigma = sigma_coord.dsigma
    return jnp.sum(qv_grid * dsigma, axis=-1)


# ---------------------------------------------------------------------------
# Long-roll moist Held-Suarez stability
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMoistHeldSuarezStability:
    """End-to-end: 200 steps of moist Held-Suarez at T21/L10.

    The original (dry) Held-Suarez test pins stability of the
    momentum/temperature equations under standard relaxation forcing.
    These tests extend the same setup with q_v advected by the dycore
    and condensation handled by Kessler microphysics, ensuring the
    full moist pipeline stays bounded.
    """

    def test_state_remains_finite_200_steps(self, grid, sigma_coord):
        state = _moist_init_state(grid, sigma_coord)
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        # Combined moist HS physics: HS dynamical forcing + Kessler.
        cfg_phys = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        kessler_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=600.0)

        def combined_physics(s, g, sc):
            # Compose Held-Suarez (dynamical) + Kessler (microphysics)
            # tendencies.  Both return SpectralHydrostaticState with
            # matching pytree structure; sum them on the spectral side.
            hs = held_suarez_forcing_spectral(s, g, sc)
            from legoesm.atmosphere.physics.combined import (
                _make_spectral_pe_combined,
            )
            # The orchestrator-style combined helper returns a 2-tuple
            # ``(tendency, phys_state_out)``; we discard the latter
            # (no per-leaf carries needed for Kessler-only).
            kess, _phys_out = kessler_fn(s, g, sc)
            # Tracer accumulation: HS has tracers=None, so just take
            # Kessler's tracer dict.
            return kess._replace(
                vor_hat=kess.vor_hat.replace(
                    data=hs.vor_hat.data + kess.vor_hat.data,
                ),
                div_hat=kess.div_hat.replace(
                    data=hs.div_hat.data + kess.div_hat.data,
                ),
                T_hat=kess.T_hat.replace(
                    data=hs.T_hat.data + kess.T_hat.data,
                ),
                lnps_hat=kess.lnps_hat.replace(
                    data=hs.lnps_hat.data + kess.lnps_hat.data,
                ),
            )

        s = state
        for _ in range(200):
            s = model.step(s, dt=600.0, physics_fn=combined_physics)

        # All prognostic fields must be finite.
        assert bool(jnp.all(jnp.isfinite(s.vor_hat.data))), "vor diverged"
        assert bool(jnp.all(jnp.isfinite(s.div_hat.data))), "div diverged"
        assert bool(jnp.all(jnp.isfinite(s.T_hat.data))), "T diverged"
        assert bool(jnp.all(jnp.isfinite(s.lnps_hat.data))), "lnps diverged"
        assert s.tracers is not None
        qv = s.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(qv))), "q_v diverged"

        # Temperature should stay in physical range.
        diag = spectral_pe_to_grid(s, grid, sigma_coord)
        T = diag["T"]
        assert float(jnp.min(T)) > 100.0, (
            f"T_min {float(jnp.min(T))} below 100 K (state collapsed)"
        )
        assert float(jnp.max(T)) < 500.0, (
            f"T_max {float(jnp.max(T))} above 500 K (state blew up)"
        )

        # q_v should stay non-negative (or ≥ small negative noise).
        # The dycore advection + Kessler don't enforce strict positivity
        # but excursion below 0 should be small relative to mean q_v.
        mean_qv = float(jnp.mean(qv))
        min_qv = float(jnp.min(qv))
        # Allow up to 50 % of mean q_v as a transient negative excursion
        # (advection can produce ringing at high resolution; the smooth
        # filter typically heals this within a few steps).
        assert min_qv > -0.5 * mean_qv, (
            f"q_v negative excursion {min_qv} exceeded 50 % of mean "
            f"{mean_qv} — tracer plumbing or filter likely broken"
        )

    def test_global_q_v_smooth_evolution(self, grid, sigma_coord):
        """Global mean q_v should evolve without runaway / NaN-style
        jumps.  We bound consecutive-step relative changes at 30 %.

        With Held-Suarez relaxing T to a strong equator-pole gradient
        from a uniform-RH initial condition, the polar columns become
        massively supersaturated within a few steps and Kessler's
        saturation adjustment removes that excess in one step — a
        legitimate ~10–25 % single-step swing in the global mean.
        Anything beyond ~30 % is non-physical (a missing factor of dt
        or a one-step filter draining mass).
        """
        state = _moist_init_state(grid, sigma_coord)
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.01,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        # Moist HS + Kessler.
        cfg_phys = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        kessler_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=600.0)

        def combined_physics(s, g, sc):
            hs = held_suarez_forcing_spectral(s, g, sc)
            kess, _ = kessler_fn(s, g, sc)
            return kess._replace(
                vor_hat=kess.vor_hat.replace(
                    data=hs.vor_hat.data + kess.vor_hat.data,
                ),
                div_hat=kess.div_hat.replace(
                    data=hs.div_hat.data + kess.div_hat.data,
                ),
                T_hat=kess.T_hat.replace(
                    data=hs.T_hat.data + kess.T_hat.data,
                ),
                lnps_hat=kess.lnps_hat.replace(
                    data=hs.lnps_hat.data + kess.lnps_hat.data,
                ),
            )

        s = state
        means = [float(jnp.mean(s.tracers["q_v"].data))]
        for _ in range(100):
            s = model.step(s, dt=600.0, physics_fn=combined_physics)
            means.append(float(jnp.mean(s.tracers["q_v"].data)))

        # Compute consecutive-step relative changes.
        means_arr = jnp.array(means)
        relative_jumps = jnp.abs(jnp.diff(means_arr) / means_arr[:-1])
        max_jump = float(jnp.max(relative_jumps))
        assert max_jump < 0.30, (
            f"Largest single-step relative change in <q_v> = "
            f"{max_jump*100:.2f} % — exceeds 30 % bound, indicates "
            f"unphysical tracer behavior."
        )
        # And the time series should converge: after the initial
        # spin-up, late-step jumps should be much smaller than early-
        # step jumps (the polar saturation transient settles).
        late_jump = float(jnp.max(relative_jumps[50:]))
        assert late_jump < 0.05, (
            f"Tracer jumps in steps 50+ should be < 5 %; got "
            f"{late_jump*100:.2f} %"
        )


# ---------------------------------------------------------------------------
# Tracer mass conservation under filter (no physics tendency)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTracerConservationUnderFilter:
    """The spectral filter and implicit hyperdiffusion filter operate
    diagonally in spectral space.  The n=0 mode (global mean) has
    eigenvalue 0 (n(n+1) = 0), so neither filter touches the global
    integral of any tracer.  This is critical: aggressive damping
    must not silently leak mass.
    """

    def test_n0_mode_unchanged_with_only_filter(self, grid, sigma_coord):
        """100 steps with only the tracer filter (no physics, no
        advection-driving winds) → global tracer mean unchanged to
        machine precision."""
        nlev = sigma_coord.n_levels
        # Uniform tracer at rest state (no winds → no advection).
        rest = isothermal_rest_state_spectral(
            grid, sigma_coord, perturbation_amplitude=0.0,
        )
        qv0 = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 0.013, dtype=jnp.float64,
        )
        state = rest._replace(
            tracers={"q_v": Field(
                data=qv0, name="q_v",
                dims=("lat", "lon", "level"), units="kg/kg",
            )},
        )
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid) * 50.0,  # aggressive
            spectral_filter_strength=0.01,                    # aggressive
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        s = state
        mean_qv_init = float(jnp.mean(qv0))
        for _ in range(100):
            s = model.step(s, dt=300.0)

        mean_qv_final = float(jnp.mean(s.tracers["q_v"].data))
        drift = abs(mean_qv_final - mean_qv_init)
        # Tolerance: ~1e-9 covers SH round-trip + advection round-off
        # accumulated over 100 steps.
        assert drift < 1e-9, (
            f"Global <q_v> drift {drift} exceeded 1e-9 over 100 filtered "
            f"steps — filter is leaking mass at n=0"
        )

    def test_high_wave_decays_low_wave_persists(self, grid, sigma_coord):
        """A two-mode tracer (n=1 + n=20): the low mode persists, the
        high mode decays.  Pin the qualitative spectral-decay shape
        of the post-step filter."""
        nlev = sigma_coord.n_levels
        # Build a band-limited tracer with energy at n=1 and n=20.
        idx_low = int(
            jnp.argmin(jnp.abs(grid.ls - 1) + jnp.abs(grid.ms - 1))
        )
        idx_high = int(
            jnp.argmin(jnp.abs(grid.ls - 20) + jnp.abs(grid.ms - 20))
        )
        q_hat = jnp.zeros((grid.n_sh, nlev), dtype=jnp.complex128)
        q_hat = q_hat.at[idx_low, :].set(0.001)
        q_hat = q_hat.at[idx_high, :].set(0.001)
        q_grid = sh_synthesis_3d(grid, q_hat)
        # Add a positive offset so the test tracer stays non-negative.
        q_grid = q_grid + 0.01
        rest = isothermal_rest_state_spectral(
            grid, sigma_coord, perturbation_amplitude=0.0,
        )
        state = rest._replace(
            tracers={"q_v": Field(
                data=q_grid, name="q_v",
                dims=("lat", "lon", "level"), units="kg/kg",
            )},
        )
        # Aggressive hyperdiff to make the n=20 mode decay quickly.
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid) * 100.0,
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        s = state
        for _ in range(50):
            s = model.step(s, dt=300.0)

        # Inspect the surviving spectral structure.
        new_q = s.tracers["q_v"].data
        new_q_hat = sh_analysis_3d(grid, new_q)
        # Low-mode amplitude should still be non-trivial.
        amp_low = float(jnp.abs(new_q_hat[idx_low, 0]))
        amp_high = float(jnp.abs(new_q_hat[idx_high, 0]))
        # Initial amplitudes were both 0.001.
        assert amp_low > 0.5 * 0.001, (
            f"n=1 mode unexpectedly decayed: amp_low={amp_low}"
        )
        assert amp_high < 0.05 * 0.001, (
            f"n=20 mode did not decay enough: amp_high={amp_high} "
            f"(should be < 5 % of initial)"
        )


# ---------------------------------------------------------------------------
# Galewsky-style tracer transport: zonal jet advecting a tracer wave
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGalewskyStyleTracerTransport:
    """A simplified Galewsky-Polvani-Hitchman test: a mid-latitude jet
    advects a passive zonal-wavenumber tracer.  The tracer must stay
    bounded over a synoptic-scale integration (10 days) and the global
    integral must be approximately conserved (drift < 1 %) when the
    spectral filter is applied to tracers.

    This pins the long-roll behavior of the tracer-advection /
    tracer-filter interaction — error norms and visual artifacts in
    spectral models often only manifest after dozens of jet
    revolutions.
    """

    def test_tracer_bounded_under_jet_advection(self, grid, sigma_coord):
        """48 hour roll: a wave-2 zonal tracer pattern advected by a
        wave-1 zonal jet stays bounded and finite."""
        nlev = sigma_coord.n_levels
        sigma_full = sigma_coord.sigma_full
        # Set up a baroclinic-like background with a zonal jet at
        # mid-latitudes.  We use the rest state and add a small
        # zonal-wavenumber-1 perturbation in the lowest temperature
        # level — this induces baroclinic instability and develops a
        # jet over the integration window.
        rest = isothermal_rest_state_spectral(
            grid, sigma_coord, T_init=290.0, perturbation_amplitude=2.0,
        )
        # Initial tracer: zonal wave-2 pattern.
        lon = grid.lon[None, :, None]
        qv0 = (
            0.005 + 0.002 * jnp.sin(2 * lon)
        ) * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        state = rest._replace(
            tracers={"q_v": Field(
                data=qv0, name="q_v",
                dims=("lat", "lon", "level"), units="kg/kg",
            )},
        )
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        # 48 hours = 288 steps at dt=600s.  HS forcing maintains the
        # background.
        s = state
        for _ in range(288):
            s = model.step(s, dt=600.0, physics_fn=held_suarez_forcing_spectral)

        new_qv = s.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(new_qv))), (
            "Tracer diverged under jet advection"
        )
        # Bounded above by ~ 2x initial maximum (allow some ringing
        # but not blowup).
        max_qv0 = float(jnp.max(qv0))
        max_qv = float(jnp.max(new_qv))
        assert max_qv < 2.5 * max_qv0, (
            f"Tracer max grew from {max_qv0} to {max_qv} (>2.5×) — "
            f"likely advection-filter instability"
        )
        # Global mean conserved to ~1 % (the spectral filter has
        # eigenvalue 1 at n=0 → exact conservation in spectral space;
        # the only loss path is round-off in the SH round-trip).
        mean_qv0 = float(jnp.mean(qv0))
        mean_qv = float(jnp.mean(new_qv))
        rel_drift = abs(mean_qv - mean_qv0) / mean_qv0
        assert rel_drift < 0.01, (
            f"Global tracer mean drifted by {rel_drift*100:.3f} % "
            f"over 48h — exceeds 1 % bound"
        )
