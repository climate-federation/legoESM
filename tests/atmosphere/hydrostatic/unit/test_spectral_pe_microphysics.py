"""Tests for spectral PE microphysics integration (PR2).

Pins:

* The microphysics bridge for spectral PE pulls ``q_v`` and the full
  hydrometeor set out of ``state.tracers`` instead of feeding the
  backend zeros.
* The bridge returns a ``SpectralHydrostaticState`` whose ``tracers``
  dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
  etc. for every key present in the input state's tracer dict.
* ``dT_hat`` carries the spectral latent-heating tendency.
* No tracers in the input state → ``tracers=None`` in the output (no
  crash, pytree-clean).
* End-to-end: a near-saturated column run through a full dycore step
  with microphysics produces a finite ``q_v`` decrease relative to the
  dycore-only baseline (condensation sink).
* Orchestrator accumulation: convection + microphysics run together
  combine their tracer tendencies (``q_v`` and ``q_c`` evolve
  differently than with either scheme alone).
* ``jax.grad`` through one bridge call w.r.t. an initial-q amplitude
  remains finite — full AD compatibility.
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
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(10)


@pytest.fixture(scope="module")
def rest_state(grid, sigma_coord):
    return isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )


def _proper_hyperdiff(grid):
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max ** 2)


def _moist_state(grid, sigma_coord, rest_state, rh_low=1.10, rh_high=0.5):
    """Build a near-saturated tropospheric state for warm-rain tests.

    rh_low default raised 0.95 → 1.10 in iter-196: at RH=0.95 the
    state is sub-saturated, so kessler condensation produces zero
    tendency (q_v < q_sat throughout) and several differential
    microphysics tests collapse onto the noise floor.  RH=1.10 is
    mildly supersaturated, firing condensation cleanly without
    breaking the four call sites that pass an explicit ``rh_low``.
    """
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full
    # Realistic vertical T profile (warmer near surface).
    T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
    T_col = jnp.maximum(T_col, 200.0)
    T_grid = jnp.broadcast_to(
        T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
    )
    T_hat = sh_analysis_3d(grid, T_grid)

    p_full_3d = jnp.broadcast_to(
        (sigma_full * 1e5)[None, None, :],
        (grid.n_lat, grid.n_lon, nlev),
    )
    q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
    rh_profile = jnp.where(sigma_full > 0.7, rh_low, rh_high)
    qv = rh_profile[None, None, :] * q_sat
    return rest_state._replace(
        T_hat=rest_state.T_hat.replace(data=T_hat),
        tracers={
            "q_v": Field(
                data=qv, name="q_v",
                dims=("lat", "lon", "level"), units="kg/kg",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Bridge unit tests
# ---------------------------------------------------------------------------

class TestSpectralPEMicrophysicsBridge:
    def test_no_tracers_no_crash(self, grid, sigma_coord, rest_state):
        """No tracers in state → output.tracers is None, no exception."""
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        out = physics_fn(rest_state, grid, sigma_coord)
        assert out.tracers is None
        # Spectral fields must be valid arrays (not NaN).
        assert bool(jnp.all(jnp.isfinite(out.T_hat.data)))

    def test_q_v_extracted_from_state(self, grid, sigma_coord, rest_state):
        """When state.tracers carries q_v, the bridge consumes it.

        Pin the plumbing: the dT_hat output for a SATURATED state
        differs from the dT_hat output when state.tracers has q_v set
        to ZEROS (with the same T profile in both cases).  If the
        bridge silently ignored ``state.tracers["q_v"]`` and always fed
        the backend zeros, the two outputs would be IDENTICAL.
        """
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )

        # Saturated state with realistic T profile.
        state_wet = _moist_state(grid, sigma_coord, rest_state)
        out_wet = physics_fn(state_wet, grid, sigma_coord)
        assert out_wet.tracers is not None
        assert "q_v" in out_wet.tracers
        dq_v_wet = out_wet.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(dq_v_wet)))
        # In a saturated column condensation must produce a NON-ZERO
        # q_v sink somewhere on the sphere.
        assert float(jnp.max(jnp.abs(dq_v_wet))) > 0.0, (
            "Saturated column should produce non-zero dq_v_dt"
        )

        # Same T profile but q_v == 0 — bridge must feed zeros to
        # the backend, so the saturation deficit is huge but
        # condensation = sigmoid(-K_huge) * (-q_sat) → 0 exactly.
        nlev = sigma_coord.n_levels
        qv_zero = jnp.zeros(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        state_dry = state_wet._replace(
            tracers={
                "q_v": Field(
                    data=qv_zero, name="q_v",
                    dims=("lat", "lon", "level"), units="kg/kg",
                ),
            },
        )
        out_dry = physics_fn(state_dry, grid, sigma_coord)
        dq_v_dry = out_dry.tracers["q_v"].data
        # The TWO outputs must DIFFER — the bridge plumbed q_v through
        # to the backend.  If the bridge ignored q_v, both runs would
        # use the same backend input and produce identical tendencies.
        max_diff = float(jnp.max(jnp.abs(dq_v_wet - dq_v_dry)))
        assert max_diff > 1e-12, (
            f"Bridge must propagate state.tracers['q_v'] to the "
            f"microphysics backend; got identical dq_v_dt for wet vs "
            f"q_v=0 inputs (max diff {max_diff})."
        )
        # Same plumbing test on the latent-heating signal — different
        # q_v should drive different condensation, hence different
        # ``dT_hat``.
        dT_diff = float(
            jnp.max(jnp.abs(out_wet.T_hat.data - out_dry.T_hat.data))
        )
        assert dT_diff > 1e-12, (
            f"Bridge must propagate q_v to dT/dt latent heating; "
            f"got identical dT_hat for wet vs q_v=0 (diff {dT_diff})."
        )

    def test_full_hydrometeor_set_propagates(self, grid, sigma_coord, rest_state):
        """All hydrometeor tendencies that have a corresponding key in
        state.tracers are returned in the output dict."""
        nlev = sigma_coord.n_levels
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )

        # State with q_v, q_c, q_r in tracers.  Use q_c > Kessler's
        # default autoconversion threshold (1e-3 kg/kg) so the
        # autoconversion → rain pathway fires, otherwise dq_r_dt is
        # exactly zero everywhere.
        state_wet = _moist_state(grid, sigma_coord, rest_state)
        qv = state_wet.tracers["q_v"].data
        qc_init = jnp.full(
            (grid.n_lat, grid.n_lon, nlev), 5.0e-3, dtype=jnp.float64,
        )
        qr_init = jnp.zeros_like(qc_init)
        state_wet = state_wet._replace(
            tracers={
                "q_v": state_wet.tracers["q_v"],
                "q_c": Field(
                    data=qc_init, name="q_c",
                    dims=("lat", "lon", "level"), units="kg/kg",
                ),
                "q_r": Field(
                    data=qr_init, name="q_r",
                    dims=("lat", "lon", "level"), units="kg/kg",
                ),
            },
        )
        out = physics_fn(state_wet, grid, sigma_coord)
        assert out.tracers is not None
        assert set(out.tracers.keys()) == {"q_v", "q_c", "q_r"}
        # Kessler converts q_c → q_r above the autoconversion threshold:
        # so dq_c_dt should be NEGATIVE and dq_r_dt POSITIVE somewhere.
        dq_c = out.tracers["q_c"].data
        dq_r = out.tracers["q_r"].data
        assert float(jnp.min(dq_c)) < 0.0, (
            "Kessler with q_c >> autoconversion threshold should produce "
            "q_c → q_r conversion (negative dq_c_dt)"
        )
        assert float(jnp.max(dq_r)) > 0.0, (
            "Kessler with cloud water should produce rain (positive dq_r_dt)"
        )

    def test_untouched_tracer_is_zero(self, grid, sigma_coord, rest_state):
        """A tracer that the backend doesn't write to (e.g., ``q_s`` on
        Kessler which has no snow physics) is mirrored as zeros so the
        dycore RHS sees a complete pytree."""
        nlev = sigma_coord.n_levels
        # Attach a passive tracer key that Kessler doesn't have.  Only
        # mapped tracer-output names are propagated.  We use ``q_g`` (
        # Kessler doesn't predict graupel) which IS in the output map
        # but should be zero.
        state_wet = _moist_state(grid, sigma_coord, rest_state)
        qg = jnp.zeros(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        state_wet = state_wet._replace(
            tracers={
                "q_v": state_wet.tracers["q_v"],
                "q_g": Field(
                    data=qg, name="q_g",
                    dims=("lat", "lon", "level"), units="kg/kg",
                ),
            },
        )

        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        out = physics_fn(state_wet, grid, sigma_coord)
        assert out.tracers is not None
        assert set(out.tracers.keys()) == {"q_v", "q_g"}
        # Kessler doesn't touch q_g → tendency is exactly zero.
        dq_g = out.tracers["q_g"].data
        assert float(jnp.max(jnp.abs(dq_g))) == 0.0

    def test_T_hat_carries_latent_heating(self, grid, sigma_coord, rest_state):
        """In a saturated column, condensation must release latent heat
        → ``dT_hat`` must be non-trivial."""
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        state_wet = _moist_state(grid, sigma_coord, rest_state)
        out = physics_fn(state_wet, grid, sigma_coord)
        # Spectral T tendency must be non-zero somewhere (latent heating).
        assert float(jnp.max(jnp.abs(out.T_hat.data))) > 0.0
        assert bool(jnp.all(jnp.isfinite(out.T_hat.data)))

    def test_dt_dt_sign_condensation(self, grid, sigma_coord, rest_state):
        """In a SUPERSATURATED column (q_v > q_sat), condensation must
        produce POSITIVE latent heating in grid space."""
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        # Build a truly supersaturating state — q_v = 1.10 * q_sat in
        # the lower troposphere — so Kessler's smooth saturation
        # adjustment ``cond = sigmoid(K · excess) · excess / dt``
        # comes out POSITIVE (excess > 0).  At rh=0.95 (the
        # ``_moist_state`` default) the smooth adjustment evaporates the
        # tiny saturation deficit, which gives dT/dt < 0.
        nlev = sigma_coord.n_levels
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        rh_profile = jnp.where(sigma_full > 0.7, 1.10, 0.5)
        qv = rh_profile[None, None, :] * q_sat
        state = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            tracers={
                "q_v": Field(
                    data=qv, name="q_v",
                    dims=("lat", "lon", "level"), units="kg/kg",
                ),
            },
        )
        out = physics_fn(state, grid, sigma_coord)
        # Inverse-transform dT_hat to grid space for sign inspection.
        from legoesm.grids.gaussian import sh_synthesis_3d
        dT_grid = sh_synthesis_3d(grid, out.T_hat.data)
        assert float(jnp.max(dT_grid)) > 0.0, (
            "Condensation in a supersaturated column should produce "
            "positive grid-space dT/dt somewhere"
        )

    def test_raw_array_tracers_supported(self, grid, sigma_coord, rest_state):
        """Tracers may be raw JAX arrays (not Field-wrapped); the bridge
        should round-trip them without crashing."""
        nlev = sigma_coord.n_levels
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.5)
        qv_raw = rh_profile[None, None, :] * q_sat   # raw jax array
        state = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            tracers={"q_v": qv_raw},   # raw, not Field
        )
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        out = physics_fn(state, grid, sigma_coord)
        # Output container must mirror the input: raw → raw (no Field
        # promotion).
        assert out.tracers is not None
        dq_v = out.tracers["q_v"]
        assert not hasattr(dq_v, "data"), (
            "Raw-array input should produce raw-array output"
        )
        assert bool(jnp.all(jnp.isfinite(dq_v)))

    def test_disabled_microphysics_emits_zero_tracer_pytree(
        self, grid, sigma_coord, rest_state,
    ):
        """``scheme="none"`` must still mirror the input tracer pytree
        as zero tendencies (otherwise the orchestrator's accumulator
        and the dycore RHS see a structural mismatch)."""
        cfg = MicrophysicsConfig(scheme="none")
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        state_wet = _moist_state(grid, sigma_coord, rest_state)
        out = physics_fn(state_wet, grid, sigma_coord)
        assert out.tracers is not None
        assert "q_v" in out.tracers
        # All tendencies are zero.
        dq_v = out.tracers["q_v"].data
        assert float(jnp.max(jnp.abs(dq_v))) == 0.0


# ---------------------------------------------------------------------------
# End-to-end: full dycore step
# ---------------------------------------------------------------------------

class TestSpectralPEDycoreWithMicrophysics:
    def test_full_step_preserves_finite_state(
        self, grid, sigma_coord, rest_state,
    ):
        """Single dycore step with microphysics + tracer state must
        leave all prognostic fields finite."""
        cfg_phys = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=300.0)
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        state = _moist_state(grid, sigma_coord, rest_state)
        new_state = model.step(state, dt=300.0, physics_fn=physics_fn)
        assert bool(jnp.all(jnp.isfinite(new_state.vor_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.div_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.T_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.lnps_hat.data)))
        assert new_state.tracers is not None
        assert bool(jnp.all(jnp.isfinite(new_state.tracers["q_v"].data)))

    def test_q_v_changes_relative_to_dycore_only(
        self, grid, sigma_coord, rest_state,
    ):
        """q_v with microphysics differs from q_v without microphysics
        after a few steps.  This pins the full bridge → orchestrator →
        dycore RHS path: the tracer-tendency dict from the bridge
        propagates into the post-step state.  Without the plumbing,
        the two trajectories would be identical (the dycore-only
        baseline advects zero winds → unchanged tracer)."""
        cfg_phys = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        physics_fn = make_physics(cfg_phys, model_type="spectral_pe", dt=300.0)
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        state = _moist_state(grid, sigma_coord, rest_state, rh_low=1.10)
        # Run for a few steps to amplify any condensation effect.
        s_w_micro = state
        s_no_micro = state
        for _ in range(3):
            s_w_micro = model.step(s_w_micro, dt=300.0, physics_fn=physics_fn)
            s_no_micro = model.step(s_no_micro, dt=300.0)

        qv_w_micro = s_w_micro.tracers["q_v"].data
        qv_no_micro = s_no_micro.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(qv_w_micro)))
        assert bool(jnp.all(jnp.isfinite(qv_no_micro)))

        # Difference must be NON-ZERO somewhere (the bridge plumbed
        # tracer tendencies through to the dycore RHS).
        diff = float(jnp.max(jnp.abs(qv_w_micro - qv_no_micro)))
        assert diff > 0.0, (
            "Kessler microphysics should change q_v relative to the "
            "dycore-only baseline; got zero diff — bridge probably "
            "drops tracer tendencies"
        )

        # In supersaturated layers (rh_low=1.10, sigma > 0.7) the
        # microphysics run must DRY relative to baseline — those
        # layers have a positive saturation excess that Kessler
        # condenses away.  Sample on the high-sigma slice only.
        sigma_full = sigma_coord.sigma_full
        wet_mask = sigma_full > 0.7
        qv_lower_w = qv_w_micro[..., wet_mask]
        qv_lower_no = qv_no_micro[..., wet_mask]
        # Compare layer-mean q_v in the supersaturated layers.  At
        # least one of these layers must dry relative to the baseline.
        layer_means_w = jnp.mean(qv_lower_w, axis=(0, 1))
        layer_means_no = jnp.mean(qv_lower_no, axis=(0, 1))
        n_dried = int(jnp.sum(layer_means_w < layer_means_no))
        assert n_dried >= 1, (
            f"Expected at least one supersaturated layer to dry under "
            f"Kessler; got n_dried={n_dried}/{int(wet_mask.sum())}.  "
            f"Bridge may have wrong sign of dq_v_dt."
        )

    def test_orchestrator_combines_convection_plus_microphysics(
        self, grid, sigma_coord, rest_state,
    ):
        """Convection + microphysics together produce a different
        post-step q_v than either scheme alone (confirms the
        per-tracer accumulation in the orchestrator).

        Uses the STATELESS ``sbm`` convection scheme: spectral PE has no
        per-column PhysicsState carry slot, so its ``step()`` refuses a
        profile-prognostic scheme (tiedtke/ZM/KF/emanuel/bechtold) that
        would silently reseed its memory every step (issue #405/#413).
        ``sbm`` is the deep-convection scheme that exercises the same
        orchestrator tracer-accumulation path without a carry; it produces
        a real convective q_v tendency here (verified ~2e-4 kg/kg/step).
        """
        config = SpectralPEConfig(
            hyperdiff_coeff=_proper_hyperdiff(grid),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)

        # Convection-only.
        cfg_conv = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        # Microphysics-only.
        cfg_micro = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        # Both.
        cfg_both = PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(
                scheme="kessler", kessler=KesslerConfig(),
            ),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        fn_conv = make_physics(cfg_conv, model_type="spectral_pe", dt=300.0)
        fn_micro = make_physics(cfg_micro, model_type="spectral_pe", dt=300.0)
        fn_both = make_physics(cfg_both, model_type="spectral_pe", dt=300.0)

        state = _moist_state(grid, sigma_coord, rest_state)
        s_conv = model.step(state, dt=300.0, physics_fn=fn_conv)
        s_micro = model.step(state, dt=300.0, physics_fn=fn_micro)
        s_both = model.step(state, dt=300.0, physics_fn=fn_both)

        qv_conv = s_conv.tracers["q_v"].data
        qv_micro = s_micro.tracers["q_v"].data
        qv_both = s_both.tracers["q_v"].data
        # All finite.
        assert bool(jnp.all(jnp.isfinite(qv_conv)))
        assert bool(jnp.all(jnp.isfinite(qv_micro)))
        assert bool(jnp.all(jnp.isfinite(qv_both)))
        # The combined run differs from BOTH single-scheme runs (i.e.,
        # the orchestrator is summing both contributions, not silently
        # dropping one).
        diff_to_conv = float(jnp.max(jnp.abs(qv_both - qv_conv)))
        diff_to_micro = float(jnp.max(jnp.abs(qv_both - qv_micro)))
        assert diff_to_conv > 0.0, (
            "Combined run should differ from convection-only — "
            "microphysics's tracer tendency is missing from the "
            "orchestrator accumulator"
        )
        assert diff_to_micro > 0.0, (
            "Combined run should differ from microphysics-only — "
            "convection's tracer tendency is missing from the "
            "orchestrator accumulator"
        )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

class TestSpectralPEMicrophysicsAD:
    def test_grad_through_bridge_call(self, grid, sigma_coord, rest_state):
        """``jax.grad`` flows through one microphysics-bridge call w.r.t.
        the q_v amplitude (no NotImplementedError, no NaN gradient)."""
        cfg = MicrophysicsConfig(
            scheme="sundqvist", sundqvist=SundqvistConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        nlev = sigma_coord.n_levels
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)

        def loss(qv_scale):
            qv = qv_scale * q_sat
            state = rest_state._replace(
                T_hat=rest_state.T_hat.replace(data=T_hat),
                tracers={
                    "q_v": Field(
                        data=qv, name="q_v",
                        dims=("lat", "lon", "level"), units="kg/kg",
                    ),
                },
            )
            out = physics_fn(state, grid, sigma_coord)
            return jnp.sum(out.T_hat.data.real ** 2 + out.T_hat.data.imag ** 2)

        g = jax.grad(loss)(jnp.array(0.95))
        assert bool(jnp.isfinite(g))
