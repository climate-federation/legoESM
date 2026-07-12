"""Tests for swappable radiation + microphysics on the plane NH dycore.

Covers the PR5-follow-up that lifts the inlined Newtonian /
no-microphysics placeholder in ``scripts/run/run_rcemip_plane.py`` to
the canonical factory dispatch:

* ``make_radiation_physics(radiation_config, model_type="plane")`` —
  supports the same ``scheme`` literals as the cubed-sphere /
  MPAS NH factories (``"gray"``, ``"rrtmgp"``).
* ``make_microphysics_physics(microphysics_config, model_type="plane", dt=dt)``
  — supports ``"kessler"``, ``"morrison"``, ``"sundqvist"``,
  ``"seifert_beheng"``, ``"thompson"``, ``"ml_emulator"``,
  ``"none"``.

The plane factories share ``_call_radiation_backend``,
``_get_microphysics_fn``, ``HydrometeorState``, the EOS / Exner
machinery, and the column-reshape pattern with the cubed-sphere
factories — these tests pin the plane-side adapter shape and
dispatch behaviour, NOT the per-scheme numerics (those are covered
by the existing scheme-level + nonhydrostatic tests).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def plane_setup():
    """Small plane (4x4x10) with three moist tracers (q_v, q_c, q_r)."""
    nx = ny = 4
    nlev = 10
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=4_000.0, dy=4_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(nlev, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # Allocate 3 tracers (q_v, q_c, q_r) at q_v = 1e-2 (10 g/kg) at
    # the lowest level, zero elsewhere — gives microphysics + radiation
    # a real moisture field to work with.
    tracers = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64)
    tracers = tracers.at[..., -1, 0].set(0.01)
    state = rest._replace(
        tracers=rest.tracers.replace(data=tracers),
    )
    return {
        "grid": grid, "hc": hc, "tm": tm, "state": state,
        "shape_3d": (ny, nx, nlev),
        "shape_w": (ny, nx, nlev + 1),
        "shape_2d": (ny, nx),
    }


# --------------------------------------------------------------------- #
# Radiation                                                             #
# --------------------------------------------------------------------- #


class TestPlaneRadiation:
    """Plane radiation dispatch via ``model_type='plane'``."""

    def test_gray_returns_plane_tendencies_with_correct_shapes(
        self, plane_setup,
    ):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="plane")
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert isinstance(tend, PlaneNonHydrostaticTendencies)
        # All tendency shapes match the plane state shapes — not the
        # cubed-sphere 4D shapes the NH factory uses.
        assert tend.dtheta_prime_dt.data.shape == plane_setup["shape_3d"]
        assert tend.drho_prime_dt.data.shape == plane_setup["shape_3d"]
        assert tend.du_dt.data.shape == plane_setup["shape_3d"]
        assert tend.dv_dt.data.shape == plane_setup["shape_3d"]
        assert tend.dw_dt.data.shape == plane_setup["shape_w"]
        assert tend.dphis_dt.data.shape == plane_setup["shape_2d"]
        # Tracer tendency from radiation: zero (radiation doesn't move
        # mass; microphysics + advection do).
        assert tend.dtracers_dt.data.shape == (
            *plane_setup["shape_3d"], 3,
        )
        assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0

    def test_gray_dim_labels_use_plane_axes(self, plane_setup):
        """Plane tendency dims must be ``('y','x','level')`` etc.,
        not the cubed-sphere ``('face','x','y','level')`` labels."""
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="plane")
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        # Plane factory dims must match the PlaneNonHydrostaticState
        # convention (z / z_half) so summed tendencies across
        # surface_flux + radiation + microphysics agree on metadata
        # — NOT the cubed-sphere ("face","x","y","level") labels.
        assert tend.dtheta_prime_dt.dims == ("y", "x", "z")
        assert tend.dw_dt.dims == ("y", "x", "z_half")
        assert tend.dphis_dt.dims == ("y", "x")
        assert tend.dtracers_dt.dims == ("y", "x", "z", "tracer")

    def test_gray_produces_nonzero_heating(self, plane_setup):
        """Gray radiation must produce a non-trivial dtheta'/dt — a
        column at T_ref ≈ 300 K under TOA insolation cools in the
        stratosphere and warms in the boundary layer; absolute max
        should be > 1e-9 K/s."""
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="plane")
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        max_heating = float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data)))
        assert max_heating > 1.0e-9, (
            f"gray radiation heating tendency too small: {max_heating:.3e}"
        )
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))

    def test_unknown_model_type_raises(self):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_radiation_physics(cfg, model_type="bogus")


# --------------------------------------------------------------------- #
# Microphysics                                                          #
# --------------------------------------------------------------------- #


class TestPlaneMicrophysics:
    """Plane microphysics dispatch via ``model_type='plane'``."""

    def test_kessler_returns_plane_tendencies_with_correct_shapes(
        self, plane_setup,
    ):
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert isinstance(tend, PlaneNonHydrostaticTendencies)
        assert tend.dtheta_prime_dt.data.shape == plane_setup["shape_3d"]
        assert tend.dtracers_dt.data.shape == (
            *plane_setup["shape_3d"], 3,
        )

    def test_kessler_finite_tendencies_with_moisture(self, plane_setup):
        """Kessler with q_v = 10 g/kg at the surface must yield finite
        tendencies on every leaf."""
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
        assert bool(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))

    def test_none_returns_zero_tendencies(self, plane_setup):
        cfg = MicrophysicsConfig(scheme="none")
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))) == 0.0
        assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0

    def test_unknown_microphysics_scheme_raises(self, plane_setup):
        cfg = MicrophysicsConfig(scheme="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown microphysics scheme"):
            make_microphysics_physics(cfg, model_type="plane", dt=1.0)

    def test_unknown_model_type_raises(self):
        cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig())
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_microphysics_physics(cfg, model_type="bogus", dt=1.0)

    def test_morrison_with_too_few_tracer_slots_raises(self, plane_setup):
        """Codex review 2026-05-24: schemes that write tendencies into
        high tracer slots (Morrison writes through slot 8 = N_i) MUST
        raise when the state cannot carry every tendency, rather than
        silently dropping ice / snow / number outputs. The fixture
        provides 3 slots; Morrison needs 9."""
        cfg = MicrophysicsConfig(
            scheme="morrison", morrison=MorrisonConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        with pytest.raises(
            ValueError, match="writes up to 9 tracer-slot tendencies",
        ):
            physics_fn(
                plane_setup["state"], plane_setup["grid"],
                plane_setup["hc"], plane_setup["tm"],
            )

    def test_morrison_runs_with_full_9_slot_state(self, plane_setup):
        """Smoke: Morrison runs cleanly when the state carries all 9
        tracer slots. Builds a wider tracer tensor on top of the
        module-level fixture state."""
        ny, nx, nlev = plane_setup["shape_3d"]
        tracers = jnp.zeros((ny, nx, nlev, 9), dtype=jnp.float64)
        tracers = tracers.at[..., -1, 0].set(0.01)
        state9 = plane_setup["state"]._replace(
            tracers=plane_setup["state"].tracers.replace(data=tracers),
        )
        cfg = MicrophysicsConfig(
            scheme="morrison", morrison=MorrisonConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        tend = physics_fn(
            state9, plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert isinstance(tend, PlaneNonHydrostaticTendencies)
        assert tend.dtracers_dt.data.shape == (ny, nx, nlev, 9)
        assert bool(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))


# --------------------------------------------------------------------- #
# Differentiability                                                     #
# --------------------------------------------------------------------- #


class TestPlanePhysicsDifferentiable:
    """``jax.grad`` smoke through the plane adapters.

    Codex review 2026-05-24: the plane radiation + microphysics
    adapters introduce reshape, clipping, Field replacement, and
    backend dispatch — none of which are AD-hostile *by themselves*,
    but the composition was untested. These tests verify
    ``jax.grad(scalar_loss)(theta_prime / q_v)`` returns finite
    arrays of the right shape — they do NOT validate gradient
    correctness against finite differences, which is left to the
    per-scheme tests for gray_radiation, kessler_microphysics, etc.
    """

    def test_gray_dtheta_prime_grad_finite(self, plane_setup):
        cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
        physics_fn = make_radiation_physics(cfg, model_type="plane")
        base_state = plane_setup["state"]

        def loss_fn(theta_prime_data):
            state = base_state._replace(
                theta_prime=base_state.theta_prime.replace(
                    data=theta_prime_data,
                ),
            )
            tend = physics_fn(
                state, plane_setup["grid"],
                plane_setup["hc"], plane_setup["tm"],
            )
            return jnp.sum(tend.dtheta_prime_dt.data ** 2)

        grad = jax.grad(loss_fn)(base_state.theta_prime.data)
        assert grad.shape == plane_setup["shape_3d"]
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_kessler_q_v_grad_finite(self, plane_setup):
        cfg = MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        )
        physics_fn = make_microphysics_physics(
            cfg, model_type="plane", dt=1.0,
        )
        base_state = plane_setup["state"]

        def loss_fn(tracers_data):
            state = base_state._replace(
                tracers=base_state.tracers.replace(data=tracers_data),
            )
            tend = physics_fn(
                state, plane_setup["grid"],
                plane_setup["hc"], plane_setup["tm"],
            )
            return jnp.sum(tend.dtheta_prime_dt.data ** 2)

        grad = jax.grad(loss_fn)(base_state.tracers.data)
        assert grad.shape == (*plane_setup["shape_3d"], 3)
        assert bool(jnp.all(jnp.isfinite(grad)))


# --------------------------------------------------------------------- #
# Composition via the RCEMIP harness wrapper                            #
# --------------------------------------------------------------------- #


class TestRCEMIPCompose:
    """``make_rcemip_physics`` from ``scripts/run/run_rcemip_plane.py``
    composes surface_fluxes + radiation + microphysics via field-wise
    tendency summation; verify it stays finite through a few steps."""

    def test_compose_gray_kessler_runs_5_steps(self, plane_setup):
        from scripts.run.run_rcemip_plane import make_rcemip_physics
        from legoesm.atmosphere.dynamics.compressible_euler import (
            CompressibleEulerConfig,
        )
        from legoesm.atmosphere.dynamics.compressible_euler_plane import (
            PlaneCompressibleEulerModel,
        )

        radiation_config = RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        )
        microphysics_config = MicrophysicsConfig(
            scheme="kessler", kessler=KesslerConfig(),
        )
        physics_fn = make_rcemip_physics(
            plane_setup["grid"], plane_setup["hc"], plane_setup["tm"],
            radiation_config=radiation_config,
            microphysics_config=microphysics_config,
            dt=1.0,
        )
        cfg = CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=1.0e6,
            hyperdiff_rho_coeff=1.0e6, hyperdiff_w_coeff=1.0e6,
            semi_implicit_acoustic=False, use_coriolis=False,
            fix_mass=False, smagorinsky_cs=0.0,
        )
        model = PlaneCompressibleEulerModel(
            plane_setup["grid"], plane_setup["hc"],
            plane_setup["tm"], cfg,
        )
        state = plane_setup["state"]
        for _ in range(5):
            state = model.step(state, dt=1.0, physics_fn=physics_fn)
        assert bool(jnp.all(jnp.isfinite(state.w.data)))
        assert bool(jnp.all(jnp.isfinite(state.theta_prime.data)))
        assert bool(jnp.all(jnp.isfinite(state.tracers.data)))

    def test_sum_plane_tendencies_empty_raises(self):
        """Codex review 2026-05-24: ``_sum_plane_tendencies()`` with no
        inputs has no canonical empty tendency to return — must raise
        rather than IndexError on ``tendencies[0]`` inside the helper."""
        from legoesm.atmosphere.idealized.land_rce import sum_plane_tendencies
        with pytest.raises(ValueError, match="at least one tendency"):
            sum_plane_tendencies()

    def test_sum_plane_tendencies_field_wise_correctness(self, plane_setup):
        """Summing surface_flux + radiation tendencies must equal the
        element-wise sum on every field's ``.data`` array."""
        from scripts.run.run_rcemip_plane import _make_surface_flux_physics
        from legoesm.atmosphere.idealized.land_rce import sum_plane_tendencies
        rad_cfg = RadiationConfig(
            scheme="gray", gray=GrayRadiationConfig(),
        )
        rad_fn = make_radiation_physics(rad_cfg, model_type="plane")
        sfc_fn = _make_surface_flux_physics(
            plane_setup["grid"], plane_setup["hc"], plane_setup["tm"],
            T_sfc=300.0, p_sfc=1.0e5,
        )
        t_rad = rad_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        t_sfc = sfc_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        t_sum = sum_plane_tendencies(t_sfc, t_rad)
        # dtheta_prime_dt: radiation cools/warms aloft; surface heats
        # the lowest level. Sum must equal element-wise.
        assert jnp.allclose(
            t_sum.dtheta_prime_dt.data,
            t_sfc.dtheta_prime_dt.data + t_rad.dtheta_prime_dt.data,
        )
        # du_dt: radiation contributes zero; surface contributes drag
        # at the lowest level.
        assert jnp.allclose(
            t_sum.du_dt.data,
            t_sfc.du_dt.data + t_rad.du_dt.data,
        )

    def test_compose_none_radiation_none_microphysics(self, plane_setup):
        """With both branches set to ``None``, composition collapses to
        surface fluxes only — still a valid physics_fn. Rest state has
        u=v=0 (no drag) and T_lo≈T_sfc (no SHF), so use the latent-heat
        branch: q_lo=0.01 < q_sfc=0.018 deposits q_v at lowest level."""
        from scripts.run.run_rcemip_plane import make_rcemip_physics
        physics_fn = make_rcemip_physics(
            plane_setup["grid"], plane_setup["hc"], plane_setup["tm"],
            radiation_config=None, microphysics_config=None, dt=1.0,
        )
        tend = physics_fn(
            plane_setup["state"], plane_setup["grid"],
            plane_setup["hc"], plane_setup["tm"],
        )
        assert isinstance(tend, PlaneNonHydrostaticTendencies)
        # Surface latent-heat flux deposits q_v at the lowest level.
        dq_v_max = float(jnp.max(jnp.abs(tend.dtracers_dt.data[..., 0])))
        assert dq_v_max > 0.0, (
            f"surface latent flux should deposit q_v at bottom level; "
            f"max|dq_v/dt|={dq_v_max}"
        )
